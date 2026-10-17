# coding=utf-8
"""
Data: flat token streams on disk, a stateless multi-source sampler, and prefetch.

On disk, one directory per source::

    {out}/{name}_{hash16}/
        tokens.u32      flat little-endian uint32, documents separated by ONE eos
        meta.json       files+sizes, tokenizer fingerprint, min_doc_tokens, counts

Only ``input_ids`` is stored.  The old pipeline stored four int32 columns of which
three were derivable (``attention_mask`` all ones, ``position_ids`` arange,
``loss_mask`` ~all ones) - a 4x IO tax on a zero-disk-cache GCS FUSE mount.
15B tokens is 60 GB here, not 240.

The cache key does NOT include ``seq_len``: sequences are ``stream[i*T:(i+1)*T]``
at load time, so the 8K -> 32K extension reuses the same file.

Sampling is a **pure function of ``(seed, step)``**: which source each global
sample draws from, and which sequence within it, are both derived from counter-
based RNG streams.  Consequences: the only resume state is ``step``; extending
``--steps`` never changes earlier batches; every sequence of a source is seen
exactly once per source-epoch; all hosts agree on the global batch by construction.
"""
from __future__ import annotations

import hashlib
import html
import json
import os
import queue
import re
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_BLK = 1 << 16          # samples per source-choice block; rank prefix counts are per block


# ----------------------------------------------------------------------------
# tokenizer fingerprint (what actually changes the ids)
# ----------------------------------------------------------------------------
def tokenizer_fingerprint(tok) -> Dict[str, object]:
    """The class name alone is NOT enough: Qwen2's and Llama-3's tokenizers are both
    ``PreTrainedTokenizerFast``, so a vocab swap would silently reuse a stale cache."""
    return {
        "class": tok.__class__.__name__,
        "len": int(len(tok)),
        "vocab_size": int(getattr(tok, "vocab_size", -1)),
        "eos_token_id": int(tok.eos_token_id) if tok.eos_token_id is not None else None,
        "bos_token_id": int(tok.bos_token_id) if tok.bos_token_id is not None else None,
    }


def source_hash(files: Sequence[str], fp: Dict[str, object], min_doc_tokens: int, bos: bool,
                clean: str = "") -> str:
    h = hashlib.sha256()
    for f in sorted(os.path.abspath(p) for p in files):
        h.update(f.encode()); h.update(str(os.path.getsize(f)).encode())
    h.update(json.dumps(fp, sort_keys=True).encode())
    h.update(f"min_doc_tokens={min_doc_tokens};bos={bos};sep=eos".encode())
    if clean:                                    # streams without cleaning keep their old hash
        h.update(f";clean={clean}".encode())
    return h.hexdigest()[:16]


# ----------------------------------------------------------------------------
# text cleaning (per source, applied before tokenisation)
# ----------------------------------------------------------------------------
_TAG_RE = re.compile(r"<[^>]{1,200}>")
_BLANK_RE = re.compile(r"\n{3,}")


def clean_text(text: str, mode: str) -> str:
    """``""``: unchanged.  ``"html"``: unescape entities and drop tags (Stack Exchange
    dumps store the rendered HTML: ``&gt;``, ``<pre><code>``, ``<p>``); the text
    inside tags is kept, so code blocks survive as plain text."""
    if not mode:
        return text
    if mode == "html":
        text = _TAG_RE.sub("", html.unescape(text))
        return _BLANK_RE.sub("\n\n", text)
    raise ValueError(f"unknown clean mode {mode!r}")


# ----------------------------------------------------------------------------
# output on a GCS FUSE mount: upload with the GCS client, never through the mount
# ----------------------------------------------------------------------------
# gcsfuse (0.35 here) stages every written file in full on the LOCAL disk before it
# uploads at close.  The TPU VMs' root disk writes at ~12 MB/s, so a 40 GB stream
# written through the mount is an hour of local IO and a full root disk.  Objects
# are therefore streamed straight to GCS with a resumable upload (no local bytes)
# and become visible under the mount like any other object.  Reads stay on FUSE.
def gcs_location(path: str) -> Optional[Tuple[str, str]]:
    """``(bucket, object_name)`` if ``path`` lies under a gcsfuse mount, else ``None``."""
    ap = os.path.abspath(path)
    best = None
    try:
        mounts = [ln.split()[:3] for ln in open("/proc/mounts")]
    except OSError:
        return None
    for src, mnt, fstype in mounts:
        if fstype == "fuse" and "/" not in src and ":" not in src and (ap == mnt or ap.startswith(mnt + "/")):
            if best is None or len(mnt) > len(best[1]):
                best = (src, mnt)
    return None if best is None else (best[0], ap[len(best[1]) + 1:])


def _gcs_blob(loc):
    from google.cloud import storage
    return storage.Client().bucket(loc[0]).blob(loc[1])


def output_exists(path: str) -> bool:
    """``os.path.exists`` that bypasses gcsfuse's (negative) stat cache on a bucket."""
    loc = gcs_location(path)
    return _gcs_blob(loc).exists() if loc else os.path.exists(path)


@contextmanager
def open_output(path: str, chunk_mb: int = 64):
    """Binary writer for ``path``: a local file, or a streaming GCS upload under a
    gcsfuse mount.  Written to ``<path>.tmp`` and renamed on success, so a crash
    never leaves a truncated final file."""
    loc = gcs_location(path)
    if loc is None:
        with open(path + ".tmp", "wb") as f:
            yield f
        os.replace(path + ".tmp", path)
        return
    bucket, name = loc
    tmp = _gcs_blob((bucket, name + ".tmp"))
    with tmp.open("wb", chunk_size=chunk_mb << 20, ignore_flush=True) as f:
        yield f
    dst = tmp.bucket.blob(name)
    token, _, _ = dst.rewrite(tmp)                                  # server-side copy, any size
    while token is not None:
        token, _, _ = dst.rewrite(tmp, token=token)
    tmp.delete()


# ----------------------------------------------------------------------------
# prepare: parquet -> tokens.u32
# ----------------------------------------------------------------------------
_WORKER_TOK = None


def _worker_init(tok_path: str):
    """Pool initialiser: one single-threaded tokenizer per worker.

    ``TOKENIZERS_PARALLELISM=false`` matters twice: the Rust tokenizer would
    otherwise spin a Rayon pool per worker (N workers x all cores), and a Rayon
    pool that exists across ``fork`` can deadlock the child.
    """
    global _WORKER_TOK
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoTokenizer
    _WORKER_TOK = AutoTokenizer.from_pretrained(tok_path)


def _tokenize_docs(args):
    """Worker: tokenise a batch of documents -> (flat u32 with one EOS per doc, kept, dropped)."""
    texts, eos, bos, min_doc = args
    enc = _WORKER_TOK(texts, add_special_tokens=False, return_attention_mask=False)["input_ids"]
    parts, kept, dropped = [], 0, 0
    for ids in enc:
        if len(ids) < min_doc:
            dropped += 1
            continue
        if bos is not None:
            parts.append(np.array([bos], dtype=np.uint32))
        parts.append(np.asarray(ids, dtype=np.uint32))
        parts.append(np.array([eos], dtype=np.uint32))
        kept += 1
    flat = np.concatenate(parts) if parts else np.zeros((0,), dtype=np.uint32)
    return flat, kept, dropped


def _doc_batches(texts: List[str], target_bytes: int = 4 << 20):
    """Split a row group's documents into batches of roughly ``target_bytes`` of text."""
    batch, size = [], 0
    for t in texts:
        batch.append(t); size += len(t)
        if size >= target_bytes:
            yield batch
            batch, size = [], 0
    if batch:
        yield batch


def prepare(name: str, files: Sequence[str], text_key: str, tokenizer_path: str, out_dir: str,
            min_doc_tokens: int = 0, bos: bool = False, workers: int = 32, clean: str = "") -> str:
    """Tokenise ``files`` into ``{out_dir}/{name}_{hash}/tokens.u32``.  Returns the directory.

    Order on disk is file order, row-group order, document order - so the stream is
    reproducible regardless of ``workers``.  One row group is tokenised at a time
    (its documents spread over the pool) while the next one is being read, which
    bounds memory to two row groups of text plus their tokens; a whole-row-group-
    per-task design held ~15 GB of Python ints per worker and took hours on the
    18 GB books3 file.

    The parquet files and the output live on a GCS FUSE mount.  Reads: ``pre_buffer``
    + a large ``buffer_size`` make pyarrow issue a few large reads per row group
    instead of thousands of page-sized ones (measured 9.6 s vs 40+ min per 1.1 GB
    row group).  Writes go through ``open_output``: streamed to GCS as ``.tmp`` and
    renamed when complete - nothing large ever touches the local root disk.
    """
    import pyarrow.parquet as pq
    from concurrent.futures import ThreadPoolExecutor
    from multiprocessing import Pool
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    fp = tokenizer_fingerprint(tok)
    eos = tok.eos_token_id
    bos_id = tok.bos_token_id if bos else None
    if eos is None:
        raise ValueError("tokenizer has no eos_token_id; documents are separated by EOS")

    d = os.path.join(out_dir, f"{name}_{source_hash(files, fp, min_doc_tokens, bos, clean)}")
    if output_exists(os.path.join(d, "meta.json")):
        print(f"[data] {d} already prepared")
        return d
    os.makedirs(d, exist_ok=True)

    pfs = [pq.ParquetFile(p, pre_buffer=True, buffer_size=64 << 20) for p in files]
    units = [(pf, rg) for pf in pfs for rg in range(pf.num_row_groups)]
    print(f"[data] {name}: {len(files)} files, {len(units)} row groups, {workers} workers", flush=True)
    kept = dropped = total = 0
    t0 = time.time()
    read = lambda u: [clean_text(t, clean) for t in u[0].read_row_group(u[1], columns=[text_key])[text_key].to_pylist()]
    with Pool(workers, initializer=_worker_init, initargs=(tokenizer_path,)) as pool, \
            ThreadPoolExecutor(1) as reader, open_output(os.path.join(d, "tokens.u32")) as f:
        pending = reader.submit(read, units[0])
        for i, _ in enumerate(units):
            texts = pending.result()
            if i + 1 < len(units):
                pending = reader.submit(read, units[i + 1])        # overlap the next read
            tasks = ((b, eos, bos_id, min_doc_tokens) for b in _doc_batches(texts))
            for arr, k, dr in pool.imap(_tokenize_docs, tasks, chunksize=1):
                f.write(arr.astype("<u4").tobytes())
                kept += k; dropped += dr; total += len(arr)
            el = time.time() - t0
            print(f"[data] {name}: row group {i + 1}/{len(units)}  {total/1e9:.2f}B tokens  "
                  f"{total/1e6/max(el,1e-9):.1f}M tok/s  {el/60:.1f} min", flush=True)
    meta = {"name": name, "files": [{"path": os.path.abspath(p), "size": os.path.getsize(p)} for p in files],
            "tokenizer": fp, "min_doc_tokens": min_doc_tokens, "bos": bos, "sep": "eos", "clean": clean,
            "num_docs": kept, "dropped_docs": dropped, "num_tokens": total}
    with open_output(os.path.join(d, "meta.json")) as f:
        f.write(json.dumps(meta, indent=2).encode())
    print(f"[data] {name}: {total/1e9:.2f}B tokens, {kept} docs kept, {dropped} dropped -> {d}", flush=True)
    return d


def _tokenize_groups(args):
    """Worker: tokenise groups of documents (one group = one repository's files) ->
    (flat u32, groups kept, groups dropped, docs kept); a group is dropped when its
    files total fewer than ``min_group`` tokens, else all its files are written
    consecutively, each followed by one EOS."""
    groups, eos, bos, min_group = args
    parts, kept, dropped, docs = [], 0, 0, 0
    for texts in groups:
        enc = _WORKER_TOK(texts, add_special_tokens=False, return_attention_mask=False)["input_ids"]
        n = sum(len(ids) + 1 + (bos is not None) for ids in enc)
        if n < min_group:
            dropped += 1
            continue
        for ids in enc:
            if bos is not None:
                parts.append(np.array([bos], dtype=np.uint32))
            parts.append(np.asarray(ids, dtype=np.uint32))
            parts.append(np.array([eos], dtype=np.uint32))
        kept += 1; docs += len(enc)
    flat = np.concatenate(parts) if parts else np.zeros((0,), dtype=np.uint32)
    return flat, kept, dropped, docs


def prepare_grouped(name: str, files: Sequence[str], text_key: str, group_key: str, order_key: str,
                    tokenizer_path: str, out_dir: str, min_group_tokens: int, workers: int = 32,
                    bos: bool = False) -> str:
    """Like ``prepare`` but the documents are grouped by ``group_key`` (e.g. a code
    repository), ordered by ``order_key`` inside a group, written consecutively, and
    a whole group is kept or dropped by its total token count.  Everything is read
    into memory and sorted first (StarCoder: ~30 GB of text; the hosts have 400 GB).

    Why: a repository's files are short on their own but together form one
    "document" whose earlier files lie beyond the attention window of the later ones.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    from multiprocessing import Pool
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    fp = tokenizer_fingerprint(tok)
    eos, bos_id = int(tok.eos_token_id), (int(tok.bos_token_id) if bos else None)
    d = os.path.join(out_dir, f"{name}_{source_hash(files, fp, min_group_tokens, bos, f'group={group_key}')}")
    if output_exists(os.path.join(d, "meta.json")):
        print(f"[data] {d} already prepared")
        return d
    os.makedirs(d, exist_ok=True)
    t0 = time.time()
    tables = []
    schema = pa.schema([(group_key, pa.string()), (order_key, pa.string()), (text_key, pa.large_string())])
    for i, p in enumerate(files):
        # large_string: a plain string column overflows its 2 GB offsets when the tables are concatenated
        tables.append(pq.read_table(p, columns=[group_key, order_key, text_key], pre_buffer=True).cast(schema))
        print(f"[data] {name}: read {i + 1}/{len(files)} ({sum(t.num_rows for t in tables)} rows, "
              f"{(time.time() - t0) / 60:.1f} min)", flush=True)
    table = pa.concat_tables(tables); del tables
    table = table.sort_by([(group_key, "ascending"), (order_key, "ascending")])
    keys = table.column(group_key).to_pylist()
    print(f"[data] {name}: {table.num_rows} documents sorted by {group_key}, {(time.time() - t0) / 60:.1f} min", flush=True)
    # group boundaries
    bounds = [0] + [i for i in range(1, len(keys)) if keys[i] != keys[i - 1]] + [len(keys)]
    n_groups = len(bounds) - 1
    print(f"[data] {name}: {n_groups} groups", flush=True)
    texts = table.column(text_key)

    def batches(target_bytes=8 << 20):
        batch, size = [], 0
        for g in range(n_groups):
            lo, hi = bounds[g], bounds[g + 1]
            docs = texts.slice(lo, hi - lo).to_pylist()
            batch.append(docs); size += sum(len(t) for t in docs)
            if size >= target_bytes:
                yield batch
                batch, size = [], 0
        if batch:
            yield batch

    kept = dropped = ndocs = total = 0
    with Pool(workers, initializer=_worker_init, initargs=(tokenizer_path,)) as pool, \
            open_output(os.path.join(d, "tokens.u32")) as f:
        for i, (arr, k, dr, nd) in enumerate(pool.imap(_tokenize_groups,
                                                       ((b, eos, bos_id, min_group_tokens) for b in batches()),
                                                       chunksize=1)):
            f.write(arr.astype("<u4").tobytes())
            kept += k; dropped += dr; ndocs += nd; total += len(arr)
            if i % 200 == 0:
                el = time.time() - t0
                print(f"[data] {name}: {kept + dropped}/{n_groups} groups  {total/1e9:.2f}B tokens  "
                      f"{total/1e6/max(el,1e-9):.1f}M tok/s  {el/60:.1f} min", flush=True)
    meta = {"name": name, "files": [{"path": os.path.abspath(p), "size": os.path.getsize(p)} for p in files],
            "tokenizer": fp, "min_doc_tokens": 0, "bos": bos, "sep": "eos", "clean": "",
            "group_key": group_key, "order_key": order_key, "min_group_tokens": min_group_tokens,
            "num_groups": kept, "dropped_groups": dropped, "num_docs": ndocs, "num_tokens": total}
    with open_output(os.path.join(d, "meta.json")) as f:
        f.write(json.dumps(meta, indent=2).encode())
    print(f"[data] {name}: {total/1e9:.2f}B tokens, {kept} groups ({ndocs} docs) kept, {dropped} groups dropped -> {d}",
          flush=True)
    return d


def filter_long(src_dir: str, name: str, out_dir: str, min_doc_tokens: int = 0,
                max_doc_tokens: int = 0, block_tokens: int = 1 << 24, exclude: Sequence[int] = ()) -> str:
    """Copy of the stream at ``src_dir`` keeping only documents of length within
    [min_doc_tokens, max_doc_tokens) -> ``{out_dir}/{name}_{hash}/tokens.u32``.
    Same format (EOS-separated), so it is a normal ``Source`` with its own hold-out.
    ``exclude``: indices of the source's documents (0-based, stream order) dropped as well - the documents of an
    evaluation set found in the training data (``log/data_v2/analysis``: the PG-19 test books' 13-grams).

    Why: a sequence that starts inside a long document has same-document content
    beyond the attention window - the only thing the fast weight can usefully store;
    conversely short flows (<4096) eliminate duplication with *_long streams.
    """
    with open(os.path.join(src_dir, "meta.json")) as f:
        meta = json.load(f)
    if meta.get("sep", "eos") != "eos":
        raise ValueError("filter_long needs an EOS-separated stream")
    eos = int(meta["tokenizer"]["eos_token_id"])
    src = os.path.join(src_dir, "tokens.u32")
    tag = f"min={min_doc_tokens};max={max_doc_tokens}" if max_doc_tokens > 0 else (f"min={min_doc_tokens}" if min_doc_tokens > 0 else "long")
    excl = np.unique(np.asarray(exclude, dtype=np.int64))
    if len(excl):
        tag += ";exclude=" + hashlib.sha1(excl.tobytes()).hexdigest()[:16]
    d = os.path.join(out_dir, f"{name}_{source_hash([src], meta['tokenizer'], min_doc_tokens, bool(meta.get('bos')), tag)}")
    if output_exists(os.path.join(d, "meta.json")):
        print(f"[data] {d} already prepared")
        return d
    os.makedirs(d, exist_ok=True)
    stream = np.memmap(src, dtype="<u4", mode="r")
    n = len(stream)
    kept = dropped = excluded = total = doc0 = 0     # doc0: the source index of the block's first document
    carry = np.zeros((0,), "<u4")                    # the partial document at the end of the last block
    t0 = time.time()
    with open_output(os.path.join(d, "tokens.u32")) as f:
        for start in range(0, n, block_tokens):
            blk = np.asarray(stream[start:start + block_tokens])
            ends = np.flatnonzero(blk == eos)
            if len(ends) == 0:
                carry = np.concatenate([carry, blk]); continue
            # documents = carry + blk[:ends[0]], blk[ends[i-1]+1 : ends[i]] ...; each ends with its EOS
            starts = np.concatenate([[0], ends[:-1] + 1])
            lengths = ends - starts + 1                                  # incl. EOS
            lengths[0] += len(carry)
            keep = np.ones(len(lengths), dtype=bool)
            if min_doc_tokens > 0:
                keep &= (lengths >= min_doc_tokens)
            if max_doc_tokens > 0:
                keep &= (lengths < max_doc_tokens)
            if len(excl):
                hit = np.isin(doc0 + np.arange(len(lengths)), excl)
                excluded += int((keep & hit).sum())
                keep &= ~hit
            doc0 += len(lengths)
            out = []
            for i in np.flatnonzero(keep):
                seg = blk[starts[i]:ends[i] + 1]
                out.append(np.concatenate([carry, seg]) if i == 0 else seg)
            if out:
                arr = np.concatenate(out); f.write(arr.astype("<u4").tobytes()); total += len(arr)
            kept += int(keep.sum()); dropped += int((~keep).sum())
            carry = blk[ends[-1] + 1:]
            if (start // block_tokens) % 64 == 0:
                el = time.time() - t0
                print(f"[data] {name}: {start/1e9:.2f}B/{n/1e9:.2f}B read, {total/1e9:.2f}B kept, "
                      f"{kept} docs, {el/60:.1f} min", flush=True)
        carry_len = len(carry)
        keep_carry = True
        if min_doc_tokens > 0 and carry_len < min_doc_tokens:
            keep_carry = False
        if max_doc_tokens > 0 and carry_len >= max_doc_tokens:
            keep_carry = False
        if keep_carry and carry_len and np.isin(doc0, excl):
            keep_carry = False; excluded += 1
        if keep_carry and carry_len:                                 # a last document without EOS
            f.write(carry.astype("<u4").tobytes()); total += carry_len; kept += 1
        elif carry_len:
            dropped += 1
    out_meta = {"name": name, "files": [{"path": os.path.abspath(src), "size": os.path.getsize(src)}],
                "derived_from": os.path.abspath(src_dir), "tokenizer": meta["tokenizer"],
                "min_doc_tokens": min_doc_tokens, "max_doc_tokens": max_doc_tokens, "bos": meta.get("bos", False), "sep": "eos",
                "clean": meta.get("clean", ""), "num_docs": kept, "dropped_docs": dropped, "num_tokens": total}
    if len(excl):
        out_meta.update(exclude=excl.tolist(), excluded_docs=excluded)
    with open_output(os.path.join(d, "meta.json")) as f:
        f.write(json.dumps(out_meta, indent=2).encode())
    print(f"[data] {name}: {total/1e9:.2f}B tokens in {kept} docs "
          f"({dropped} docs dropped, {excluded} of them excluded) -> {d}", flush=True)
    return d


def doc_lengths(src_dir: str, block_tokens: int = 1 << 24, group_key: Optional[str] = None) -> np.ndarray:
    """Lengths (incl. EOS) of every document of an EOS-separated stream, one pass."""
    with open(os.path.join(src_dir, "meta.json")) as f:
        meta = json.load(f)
    eos = int(meta["tokenizer"]["eos_token_id"])
    stream = np.memmap(os.path.join(src_dir, "tokens.u32"), dtype="<u4", mode="r")
    n = len(stream)
    lengths, carry, t0 = [], 0, time.time()
    for start in range(0, n, block_tokens):
        blk = np.asarray(stream[start:start + block_tokens])
        ends = np.flatnonzero(blk == eos)
        if len(ends) == 0:
            carry += len(blk); continue
        starts = np.concatenate([[0], ends[:-1] + 1])
        ln = ends - starts + 1
        ln[0] += carry
        lengths.append(ln)
        carry = len(blk) - ends[-1] - 1
        if (start // block_tokens) % 128 == 0:
            print(f"[data] {meta['name']}: {start/1e9:.2f}B/{n/1e9:.2f}B, {(time.time()-t0)/60:.1f} min", flush=True)
    if carry:
        lengths.append(np.array([carry]))
    return np.concatenate(lengths) if lengths else np.zeros((0,), np.int64)


def windows_path(src_dir: str, seq_len: int, holdout_tokens: int) -> str:
    return os.path.join(src_dir, f"windows_{seq_len}_{holdout_tokens}.npy")


def doc_windows(src_dir: str, seq_len: int, holdout_tokens: int) -> np.ndarray:
    """Start offsets of every ``seq_len``-token window lying inside ONE document of the stream:
    non-overlapping, aligned at the document's first token, the EOS excluded, and ending before the
    last ``holdout_tokens`` tokens (the held-out tail) - ``Sampler(windows=...)``'s training sequences."""
    lengths = doc_lengths(src_dir).astype(np.int64)                    # incl. the EOS
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    n_win = (lengths - 1) // seq_len
    keep = n_win > 0
    starts, n_win = starts[keep], n_win[keep]
    out = (np.repeat(starts, n_win) + seq_len * (np.arange(n_win.sum()) - np.repeat(np.cumsum(n_win) - n_win, n_win)))
    limit = int(lengths.sum()) - holdout_tokens
    return out[out + seq_len <= limit].astype(np.int64)


def unigram_counts(src_dirs, vocab: int, sample_tokens: int = 2_000_000_000, block_tokens: int = 1 << 24) -> np.ndarray:
    """``[vocab]`` int64 token counts over up to ``sample_tokens`` tokens per source,
    read in evenly spaced blocks (so the sample covers each stream)."""
    counts = np.zeros(vocab, np.int64)
    for d in src_dirs:
        mm = np.memmap(os.path.join(d, "tokens.u32"), dtype="<u4", mode="r")
        n = len(mm)
        n_blocks = max(1, min(n, sample_tokens) // block_tokens)
        starts = np.linspace(0, max(0, n - block_tokens), n_blocks).astype(np.int64)
        for st in starts:
            blk = np.asarray(mm[st:st + block_tokens]).astype(np.int64)
            counts += np.bincount(blk, minlength=vocab)[:vocab]
        print(f"[data] unigram: {os.path.basename(d)} {n_blocks} blocks", flush=True)
    return counts


def length_stats(src_dir: str, thresholds=(4096, 8192, 16384, 32768, 131072)) -> dict:
    """Token share and document count above each length threshold."""
    ln = doc_lengths(src_dir).astype(np.int64)
    total = int(ln.sum())
    out = {"docs": int(len(ln)), "tokens": total, "mean_len": float(ln.mean()) if len(ln) else 0.0,
           "median_len": float(np.median(ln)) if len(ln) else 0.0}
    for t in thresholds:
        m = ln >= t
        out[f">={t}"] = {"docs": int(m.sum()), "tokens": int(ln[m].sum()), "token_share": float(ln[m].sum() / max(total, 1))}
    return out


# ----------------------------------------------------------------------------
# sources + sampler
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Source:
    name: str
    path: str
    weight: float
    num_tokens: int
    vocab_len: int = 0          # tokenizer length recorded at preparation (0 = unknown)
    eos_token_id: int = 128001  # tokenizer EOS token id (default: llama3 / 1b_e2e)

    def open(self) -> np.memmap:
        return np.memmap(os.path.join(self.path, "tokens.u32"), dtype="<u4", mode="r")


def split_sources(spec: str, root: str = "") -> List[Tuple[str, str, float]]:
    """``"books3=dir:0.7,fineweb=dir:0.3"`` -> ``[(name, dir, weight)]`` (nothing opened); a relative ``dir``
    is taken under ``root`` (``--data_root``) when one is given."""
    out = []
    for item in filter(None, (s.strip() for s in spec.split(","))):
        name, rest = item.split("=", 1)
        path, w = rest.rsplit(":", 1) if ":" in rest else (rest, "1.0")
        out.append((name, os.path.join(root, path) if root else path, float(w)))
    return out


def parse_sources(spec: str, root: str = "") -> List[Source]:
    """``"books3=dir:0.7,fineweb=dir:0.3"`` -> normalised ``Source`` list (relative dirs under ``root``)."""
    out = []
    for name, path, w in split_sources(spec, root):
        with open(os.path.join(path, "meta.json")) as f:
            meta = json.load(f)
        tok_meta = meta.get("tokenizer", {})
        eos_id = int(tok_meta.get("eos_token_id", 128001)) if tok_meta.get("eos_token_id") is not None else 128001
        out.append(Source(name, path, w, int(meta["num_tokens"]), int(tok_meta.get("len", 0)), eos_id))
    if not out:
        raise ValueError("no sources")
    z = sum(s.weight for s in out)
    return [Source(s.name, s.path, s.weight / z, s.num_tokens, s.vocab_len, s.eos_token_id) for s in out]


def check_vocab(sources: Sequence[Source], vocab_size: int) -> None:
    """Every token id in a stream is below the tokenizer length recorded in its
    meta.json; that length must fit the model's vocabulary (the embedding lookup
    clips out-of-range ids silently)."""
    for s in sources:
        if s.vocab_len and s.vocab_len > vocab_size:
            raise ValueError(f"source {s.name}: tokenizer length {s.vocab_len} exceeds model vocab_size {vocab_size}")


def sources_hash(sources: Sequence[Source], seed: int, seq_len: int, global_batch: int,
                 holdout_tokens: int = 0, extra: Optional[list] = None,
                 paths: Optional[Sequence[str]] = None) -> str:
    """Identity of a training stream: every source's name, weight and token count, the seed,
    the sequence length, ``global_batch`` (sequences per optimiser step), the holdout and
    ``extra`` (planting, window indexes).  Path-free: the same streams in another directory
    or on another machine keep the hash.  ``paths``: the form of checkpoints written before
    2026-10 (``data_hash_version`` 1), which also hashed each stream's absolute path and
    passed the micro-batch size as ``global_batch``."""
    rows = ([(s.name, round(s.weight, 9), s.num_tokens) for s in sources] if paths is None else
            [(s.name, os.path.abspath(p), round(s.weight, 9), s.num_tokens) for s, p in zip(sources, paths)])
    h = hashlib.sha256(json.dumps(rows + [seed, seq_len, global_batch, holdout_tokens]
                                  + (list(extra) if extra else []), sort_keys=True).encode())
    return h.hexdigest()[:16]


@dataclass(frozen=True)
class Plant:
    """Planted recall in the training stream: a ``frac`` of the sequences get one span of
    ``len_min..len_max`` tokens copied from an early chunk into a chunk at least
    ``window + 1`` chunks later, i.e. beyond the attention window of the copy - a token
    stream in which recalling out-of-window content pays (the natural stream rarely asks
    for it).  ``chunk`` / ``window``: the model's chunk size and attention window in chunks."""
    frac: float
    len_min: int
    len_max: int
    chunk: int
    window: int


def plant_copy(row: np.ndarray, rng: np.random.Generator, len_min: int, len_max: int, chunk: int,
               window: int, eos: Optional[int] = None, tries: int = 8) -> Optional[Tuple[int, int, int]]:
    """Copy one span of ``row`` [T] (in place) from chunk ``cs`` to chunk ``ct >= cs + window + 1``;
    returns ``(src, dst, length)`` or None when no EOS-free source and target were found in
    ``tries`` draws (a copy must not splice documents together)."""
    nc = len(row) // chunk
    if nc < window + 2:
        return None
    for _ in range(tries):
        L = int(rng.integers(len_min, len_max + 1))
        cs = int(rng.integers(0, nc - window - 1))
        ct = int(rng.integers(cs + window + 1, nc))
        a = cs * chunk + int(rng.integers(0, chunk - L + 1))
        b = ct * chunk + int(rng.integers(0, chunk - L + 1))
        if eos is not None and ((row[a:a + L] == eos).any() or (row[b:b + L] == eos).any()):
            continue
        row[b:b + L] = row[a:a + L]
        return a, b, L
    return None


class Sampler:
    """``batch(step) -> int32[local_batch, seq_len]``, a pure function of ``(seed, step)``.

    Global sample ``g = step * G + j``; this host owns ``j`` in
    ``[rank * G/P, (rank+1) * G/P)``.  ``choice[g]`` is drawn per block of ``_BLK``
    samples from ``rng([seed, 1, g // _BLK])``; ``rank_in_source[g]`` is the count
    of earlier samples with the same choice (block prefix counts cached lazily);
    within a source, epoch ``e = rank // n_s`` uses permutation
    ``rng([seed, 2, s, e]).permutation(n_s)``.
    """

    def __init__(self, sources: Sequence[Source], seq_len: int, global_batch: int, seed: int,
                 rank: int = 0, world: int = 1, holdout_tokens: int = 0,
                 holdout: Optional[int] = None, eos: Optional[int] = None,
                 eval_only: bool = False, plant: Optional[Plant] = None,
                 windows: Optional[Sequence[np.ndarray]] = None, accum: int = 1):
        """``holdout_tokens``: the LAST ``holdout_tokens`` tokens of every source are never
        sampled for training; ``eval_batch`` reads them starting at the first EOS after
        ``num_tokens - holdout_tokens``. Part of the data hash.  ``plant``: planted recall in
        the training batches (``batch`` / ``global_batch``; held-out batches are never
        planted), a pure function of ``(seed, g)``; part of the hash when set.  ``windows``:
        per source, the start offsets of the training sequences (``doc_windows``: single-document
        windows) instead of the contiguous ``seq_len`` slices of the packed stream; part of the hash.
        ``accum``: micro-batches of ``global_batch`` per optimiser step (``--grad_accum``).  Step
        ``s`` reads batches ``s*accum ..``, so the order depends only on the step batch
        ``global_batch * accum``: that is what the hash counts and what ``eval_doc_info`` scans
        around (16 x 4 reads, hashes and evaluates exactly as 64 x 1)."""
        if global_batch % world:
            raise ValueError(f"global_batch {global_batch} not divisible by {world} hosts")
        self.sources = list(sources)
        self.T, self.G, self.seed = seq_len, global_batch, seed
        self.rank, self.world = rank, world
        self.local = global_batch // world
        self.accum = accum
        self.step_batch = global_batch * accum
        ht = holdout_tokens if holdout is None else holdout
        self.holdout_tokens = ht
        self.holdout = ht
        self.eval_only = eval_only
        self.w = np.array([s.weight for s in sources], np.float64)
        self.n_total = np.array([s.num_tokens // seq_len for s in sources], np.int64)
        if eval_only:
            self.n_seq = np.zeros(len(sources), np.int64)
        elif self.holdout_tokens > 0:
            self.n_seq = np.array([max(0, (s.num_tokens - self.holdout_tokens) // seq_len) for s in sources], np.int64)
        else:
            self.n_seq = np.array([s.num_tokens // seq_len for s in sources], np.int64)
        self.windows = None if windows is None else [np.asarray(w, np.int64) for w in windows]
        if self.windows is not None and not eval_only:
            if len(self.windows) != len(sources):
                raise ValueError("windows: one array per source")
            self.n_seq = np.array([len(w) for w in self.windows], np.int64)
        if not eval_only and (self.n_seq <= 0).any():
            raise ValueError("a source is shorter than one training sequence after the holdout")
        self._mm = [s.open() for s in sources]
        self._choice_blocks: Dict[int, np.ndarray] = {}
        self._prefix: Dict[int, np.ndarray] = {}         # block -> counts per source BEFORE the block
        self._perm: Dict[Tuple[int, int], np.ndarray] = {}
        self.plant = plant if plant is not None and plant.frac > 0 else None
        self.eos = eos
        extra = [] if self.plant is None else ["plant", self.plant.frac, self.plant.len_min, self.plant.len_max,
                                               self.plant.chunk, self.plant.window]
        if self.windows is not None:
            extra += ["windows"] + [hashlib.sha256(w.tobytes()).hexdigest()[:16] for w in self.windows]
        self._extra = extra or None
        self.hash = sources_hash(sources, seed, seq_len, self.step_batch, self.holdout_tokens, extra=self._extra)

        # Align eval_start for each source to first EOS at or after [num_tokens - holdout_tokens]
        self.eval_start = np.zeros(len(sources), np.int64)
        self.n_eval_seq = np.zeros(len(sources), np.int64)
        for s_idx, s in enumerate(sources):
            if eval_only:
                raw_start = 0
            elif self.holdout_tokens > 0:
                raw_start = max(0, s.num_tokens - self.holdout_tokens)
            else:
                raw_start = s.num_tokens
            mm = self._mm[s_idx]
            if raw_start < s.num_tokens:
                s_eos = s.eos_token_id if s.eos_token_id else (eos if eos is not None else 128001)
                search_len = min(len(mm) - raw_start, 1_048_576)
                sub = np.asarray(mm[raw_start : raw_start + search_len])
                eos_pos = np.flatnonzero(sub == s_eos)
                if len(eos_pos) > 0:
                    start = raw_start + int(eos_pos[0]) + 1
                else:
                    sub_full = np.asarray(mm[raw_start:])
                    eos_pos_full = np.flatnonzero(sub_full == s_eos)
                    start = raw_start + int(eos_pos_full[0]) + 1 if len(eos_pos_full) > 0 else raw_start
            else:
                start = raw_start
            self.eval_start[s_idx] = start
            self.n_eval_seq[s_idx] = max(0, (s.num_tokens - start) // seq_len)

    # -- rng streams --------------------------------------------------------
    def _choices(self, blk: int) -> np.ndarray:
        if blk not in self._choice_blocks:
            rng = np.random.default_rng([self.seed, 1, blk])
            self._choice_blocks[blk] = rng.choice(len(self.sources), size=_BLK, p=self.w).astype(np.int32)
        return self._choice_blocks[blk]

    def _prefix_counts(self, blk: int) -> np.ndarray:
        """Per-source counts of samples in blocks ``< blk``."""
        if blk not in self._prefix:
            if blk == 0:
                self._prefix[0] = np.zeros(len(self.sources), np.int64)
            else:
                prev = self._prefix_counts(blk - 1)
                self._prefix[blk] = prev + np.bincount(self._choices(blk - 1), minlength=len(self.sources))
        return self._prefix[blk]

    def _perm_for(self, s: int, epoch: int) -> np.ndarray:
        key = (s, epoch)
        if key not in self._perm:
            self._perm[key] = np.random.default_rng([self.seed, 2, s, epoch]).permutation(self.n_seq[s])
            if len(self._perm) > 8:           # bound the cache
                self._perm.pop(next(iter(self._perm)))
        return self._perm[key]

    # -- addressing ---------------------------------------------------------
    def locate(self, g: int) -> Tuple[int, int]:
        """Global sample ``g`` -> ``(source index, sequence index within source)``."""
        blk, off = divmod(g, _BLK)
        ch = self._choices(blk)
        s = int(ch[off])
        rank = int(self._prefix_counts(blk)[s] + np.count_nonzero(ch[:off] == s))
        epoch, r = divmod(rank, int(self.n_seq[s]))
        return s, int(self._perm_for(s, epoch)[r])

    def _maybe_plant(self, row: np.ndarray, g: int) -> np.ndarray:
        p = self.plant
        if p is not None:
            rng = np.random.default_rng([self.seed, 3, g])
            if rng.random() < p.frac:
                plant_copy(row, rng, p.len_min, p.len_max, p.chunk, p.window, self.eos)
        return row

    def batch(self, step: int) -> np.ndarray:
        out = np.empty((self.local, self.T), np.int32)
        base = step * self.G + self.rank * self.local
        for j in range(self.local):
            s, i = self.locate(base + j)
            off = i * self.T if self.windows is None else int(self.windows[s][i])
            out[j] = self._mm[s][off:off + self.T]
            self._maybe_plant(out[j], base + j)
        return out

    def global_batch(self, step: int) -> np.ndarray:
        """All hosts' batches concatenated - for tests."""
        rows = [self.locate(step * self.G + j) for j in range(self.G)]
        offs = [i * self.T if self.windows is None else int(self.windows[s][i]) for s, i in rows]
        out = np.stack([self._mm[s][o:o + self.T] for (s, _), o in zip(rows, offs)]).astype(np.int32)
        for j in range(self.G):
            self._maybe_plant(out[j], step * self.G + j)
        return out

    def hash_at(self, step_batch: int) -> str:
        """This stream's data hash at another number of sequences per optimiser step."""
        return sources_hash(self.sources, self.seed, self.T, step_batch, self.holdout_tokens, extra=self._extra)

    def legacy_hash(self, paths: Sequence[str], global_batch: int) -> str:
        """The data hash as checkpoints before 2026-10 wrote it (``sources_hash`` with ``paths``):
        the streams' absolute ``paths`` and that run's micro-batch size ``global_batch``."""
        return sources_hash(self.sources, self.seed, self.T, global_batch, self.holdout_tokens,
                            extra=self._extra, paths=paths)

    # -- held-out evaluation -------------------------------------------------
    def eval_batches(self, source: int) -> int:
        """Number of global eval batches (``G`` sequences each) available for ``source``."""
        return int(self.n_eval_seq[source] // self.G)

    def eval_step_batches(self, source: int) -> int:
        """Number of eval batches of ``step_batch`` sequences: evaluation counts these, so the
        evaluated sequences do not depend on the micro-batch split (batch ``k`` of them is
        micro-batches ``k*accum .. k*accum+accum-1``)."""
        return int(self.n_eval_seq[source] // self.step_batch)

    def eval_doc_info(self, source: int, k: int, eos: int, window_tokens: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """For the ``k``-th held-out batch (global order, all ``G`` sequences): the length
        of the document each sequence starts in, the number of its tokens that belong
        to that document beyond the attention window, and the document start position in stream.
        Scans the stream once around the step batch holding it (documents may start far before
        it), so a micro-batch gets the same answers as in one batch of ``step_batch``."""
        T = self.T
        base = int(self.eval_start[source]) + k * self.G * T
        mm = self._mm[source]
        step0 = int(self.eval_start[source]) + (k * self.G // self.step_batch) * self.step_batch * T
        lo = max(0, step0 - 64 * T)                             # look back up to 64 sequences for the doc start
        hi = min(len(mm), step0 + self.step_batch * T + 64 * T)
        blk = np.asarray(mm[lo:hi])
        eos_pos = np.flatnonzero(blk == eos) + lo
        doc_len = np.zeros(self.G, np.int64); beyond = np.zeros(self.G, np.int64)
        doc_id = np.zeros(self.G, np.int64)
        for j in range(self.G):
            o = base + j * T
            i = np.searchsorted(eos_pos, o)                          # first EOS at/after o
            start = eos_pos[i - 1] + 1 if i > 0 else lo
            end = eos_pos[i] if i < len(eos_pos) else hi
            doc_len[j] = end - start + 1
            beyond[j] = max(0, min(end, o + T) - (o + window_tokens))
            doc_id[j] = start
        return doc_len, beyond, doc_id

    def eval_context(self, source: int, k: int, eos: int, lookback: int):
        """For the ``k``-th held-out batch (all ``G`` sequences): per sequence the stream
        tokens from up to ``lookback`` before it to its end, the sequence's offset in that
        context, and the document starts inside the context (evaluation probes)."""
        T = self.T
        base = int(self.eval_start[source]) + k * self.G * T
        mm = self._mm[source]
        out = []
        for j in range(self.G):
            o = base + j * T
            lo = max(0, o - lookback)
            ctx = np.asarray(mm[lo:o + T])
            starts = np.concatenate([[0], np.flatnonzero(ctx == eos) + 1])
            out.append((ctx, o - lo, starts[starts < len(ctx)]))
        return out

    def eval_batch(self, source: int, k: int) -> np.ndarray:
        """This host's slice of the ``k``-th held-out batch of ``source``: sequences
        ``k*G + [rank*local, (rank+1)*local)`` after ``eval_start[source]``."""
        if not 0 <= k < self.eval_batches(source):
            raise IndexError(f"eval batch {k} of source {source}: only {self.eval_batches(source)}")
        base = int(self.eval_start[source]) + (k * self.G + self.rank * self.local) * self.T
        out = np.empty((self.local, self.T), np.int32)
        for j in range(self.local):
            off = base + j * self.T
            out[j] = self._mm[source][off:off + self.T]
        return out


class MaskMaker:
    """Host-side inner masks for a batch (``inner_masks.inner_mask_rule``), or None."""

    def __init__(self, mode: str, chunk: int, eos: int, pad: int, unigram_path: str = "",
                 target_frac: float = 0.5, generic_floor: float = 0.1, seed: int = 0, task: str = "mask"):
        self.mode, self.chunk, self.eos, self.pad, self.task = mode, chunk, eos, pad, task
        self.target_frac, self.generic_floor, self.seed = target_frac, generic_floor, seed
        self.unigram = None
        if (mode != "random" or task == "split") and unigram_path and os.path.exists(unigram_path):
            self.unigram = np.fromfile(unigram_path, dtype=np.int64)

    @property
    def active(self) -> bool:
        return self.task == "split" or self.mode != "random"

    def __call__(self, ids: np.ndarray) -> Optional[np.ndarray]:
        """``[B, T]`` bool masks (mask task) or int32 permutations (split task); None = device-side random."""
        if self.task == "split":
            from .inner_masks import split_perm
            return np.stack([split_perm(row, self.mode, self.chunk, self.eos, self.pad, 1.0 - self.target_frac,
                                        self.unigram, self.generic_floor, self.seed) for row in ids])
        if self.mode == "random":
            return None
        from .inner_masks import inner_mask_rule
        return np.stack([inner_mask_rule(row, self.mode, self.chunk, self.eos, self.pad, self.unigram,
                                         self.target_frac, self.generic_floor, self.seed) for row in ids])


class Prefetcher:
    """Compute ``sampler.batch(step)`` for the next ``depth`` steps on a thread.

    GCS FUSE random reads are latency-bound; a few outstanding requests hide it."""

    def __init__(self, sampler: Sampler, start_step: int, depth: int = 4, mask_maker=None):
        self.mask_maker = mask_maker
        self.s = sampler
        self.q: "queue.Queue[Tuple[int, np.ndarray]]" = queue.Queue(maxsize=depth)
        self._next = start_step
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        step = self._next
        while not self._stop.is_set():
            ids = self.s.batch(step)
            self.q.put((step, ids, self.mask_maker(ids) if self.mask_maker else None))
            step += 1

    def get(self, step: int):
        """``(ids [local, T], inner_mask [local, T] or None)`` for ``step``."""
        s, b, m = self.q.get()
        if s != step:
            raise RuntimeError(f"prefetch out of order: got {s}, wanted {step}")
        return b, m

    def close(self):
        self._stop.set()


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None):
    import argparse
    ap = argparse.ArgumentParser(prog="python -m ttt.data")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="tokenise parquet files into a flat token stream")
    p.add_argument("--name", required=True)
    p.add_argument("--files", nargs="+", required=True)
    p.add_argument("--text_key", default="text")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", default="bucket/data/tokens")
    p.add_argument("--min_doc_tokens", type=int, default=0)
    p.add_argument("--bos", action="store_true")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--clean", default="", choices=["", "html"], help="text cleaning applied before tokenisation")
    g = sub.add_parser("prepare_grouped", help="tokenise with documents grouped (e.g. code files by repository)")
    g.add_argument("--name", required=True)
    g.add_argument("--files", nargs="+", required=True)
    g.add_argument("--text_key", default="content")
    g.add_argument("--group_key", default="max_stars_repo_name")
    g.add_argument("--order_key", default="max_stars_repo_path")
    g.add_argument("--tokenizer", required=True)
    g.add_argument("--out", default="bucket/data/tokens")
    g.add_argument("--min_group_tokens", type=int, default=4096)
    g.add_argument("--workers", type=int, default=32)
    ug = sub.add_parser("unigram", help="token frequency table over a sample of prepared streams")
    ug.add_argument("--src", nargs="+", required=True)
    ug.add_argument("--vocab", type=int, default=128256)
    ug.add_argument("--sample_tokens", type=int, default=2_000_000_000, help="per source")
    ug.add_argument("--out", required=True, help=".npy of int64 counts")
    st = sub.add_parser("stats", help="document-length statistics of prepared streams (token share above thresholds)")
    st.add_argument("--src", nargs="+", required=True)
    st.add_argument("--out", default="", help="write the JSON here (default: print)")
    q = sub.add_parser("filter", help="keep only the documents of a prepared stream that match length bounds")
    q.add_argument("--src", required=True, help="prepared source directory")
    q.add_argument("--name", required=True, help="name of the filtered source, e.g. fineweb_edu_short")
    q.add_argument("--out", default="bucket/data/tokens")
    q.add_argument("--min_doc_tokens", type=int, default=0)
    q.add_argument("--max_doc_tokens", type=int, default=0, help="if > 0, keep only docs with length < max_doc_tokens")
    q.add_argument("--exclude", default="", help="JSON file: a list of the source's document indices to drop")
    wn = sub.add_parser("windows", help="index the single-document training windows of prepared streams")
    wn.add_argument("--src", nargs="+", required=True)
    wn.add_argument("--seq_len", type=int, default=8192)
    wn.add_argument("--holdout", type=int, default=4_194_304)
    a = ap.parse_args(argv)
    if a.cmd == "windows":
        for d in a.src:
            w = doc_windows(d, a.seq_len, a.holdout)
            np.save(windows_path(d, a.seq_len, a.holdout), w)
            print(f"[data] {d}: {len(w)} windows = {len(w) * a.seq_len / 1e9:.2f}B tokens -> "
                  f"{windows_path(d, a.seq_len, a.holdout)}", flush=True)
        return
    if a.cmd == "prepare":
        prepare(a.name, a.files, a.text_key, a.tokenizer, a.out, a.min_doc_tokens, a.bos, a.workers, a.clean)
    elif a.cmd == "prepare_grouped":
        prepare_grouped(a.name, a.files, a.text_key, a.group_key, a.order_key, a.tokenizer, a.out,
                        a.min_group_tokens, a.workers)
    elif a.cmd == "filter":
        filter_long(a.src, a.name, a.out, min_doc_tokens=a.min_doc_tokens, max_doc_tokens=a.max_doc_tokens,
                    exclude=json.load(open(a.exclude)) if a.exclude else ())
    elif a.cmd == "unigram":
        c = unigram_counts(a.src, a.vocab, a.sample_tokens)
        with open_output(a.out) as f:
            f.write(c.tobytes())
        nz = int((c > 0).sum()); print(f"[data] unigram: {c.sum()/1e9:.2f}B tokens counted, {nz} distinct ids -> {a.out}")
    elif a.cmd == "stats":
        res = {os.path.basename(os.path.normpath(d)): length_stats(d) for d in a.src}
        text = json.dumps(res, indent=2)
        if a.out:
            with open(a.out, "w") as f:
                f.write(text)
        print(text)


if __name__ == "__main__":
    main()
