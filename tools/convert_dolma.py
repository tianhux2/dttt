#!/usr/bin/env python3
# coding=utf-8
"""
Dolma JSON lines, gzipped (the Common Pile's form; ``tools/download_common_pile.py``), to parquet for
``python -m ttt.data prepare``: one parquet per input file with ``id``, ``title`` and ``text``, row groups of about
--row_group_mb of text (the tokeniser's work units).  Documents of our evaluation set are removed: every document of a
PG-19 test book (``bucket/data/raw/pg19_test/test.parquet``, the 100 books of ``pg19_test``) -

    by id     a Project Gutenberg record (``metadata.url`` .../ebooks/<n>...) of a test book's number;
    by text   in every source, the share of the document's word 13-grams (lower-cased alphanumeric words, every
              --stride-th position) found among the test books' 13-grams above --overlap.

The removed documents are listed in ``<out>/removed.jsonl``, the counts in ``<out>/summary.json``.  CPU work: run it
on a compute node (a node of a running job over ssh), never on a login node.

--group KEY:ORDER joins the records that share KEY into one document, ordered by ORDER (numerically when it is a
number), after the test books' removal page by page: the Biodiversity Heritage Library's records are single pages
(median ~490 tokens), with ``item_id`` and ``page_num``; a volume's pages are consecutive in one file (an item that
straddles two files becomes two documents).

    .venv/bin/python tools/convert_dolma.py --src bucket/data/raw/common_pile/project_gutenberg \\
        --out bucket/data/raw/common_pile_parquet/project_gutenberg --workers 16
"""
from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import os
import re
import sys
import time
from multiprocessing import Pool

WORD = re.compile(r"[a-z0-9]+")
N = 13
_TEST_GRAMS: set = set()
_TEST_IDS: set = set()
_ARGS = None


def grams(text: str, stride: int = 1):
    """64-bit hashes of the word 13-grams at every ``stride``-th position (deterministic across processes)."""
    w = WORD.findall(text.lower())
    for i in range(0, len(w) - N + 1, stride):
        yield int.from_bytes(hashlib.blake2b(" ".join(w[i:i + N]).encode(), digest_size=8).digest(), "little")


def gutenberg_id(rec: dict):
    m = re.search(r"gutenberg\.org/ebooks/(\d+)", str((rec.get("metadata") or {}).get("url", "")))
    return int(m.group(1)) if m else None


def overlap(text: str, stride: int) -> float:
    n = hit = 0
    for g in grams(text, stride):
        n += 1
        hit += g in _TEST_GRAMS
    return hit / n if n else 0.0


def _order(v):
    v = str(v)
    return (0, int(v), "") if v.isdigit() else (1, 0, v)


def records(fh, group):
    """The file's records, or with ``group`` (KEY, ORDER) one record per run of records sharing KEY: its pages'
    texts joined by blank lines in ORDER, the pages kept in ``_pages`` for the removal."""
    if not group:
        yield from (json.loads(line) for line in fh)
        return
    key, order = group
    run = []
    for line in fh:
        rec = json.loads(line)
        if run and rec.get(key) != run[0].get(key):
            yield {"id": str(run[0].get(key)), "_pages": sorted(run, key=lambda r: _order(r.get(order)))}
            run = []
        run.append(rec)
    if run:
        yield {"id": str(run[0].get(key)), "_pages": sorted(run, key=lambda r: _order(r.get(order)))}


def test_book(rec: dict, a):
    """Why ``rec`` is a PG-19 test book's text, or None."""
    gid = gutenberg_id(rec)
    if gid is not None and gid in _TEST_IDS:
        return f"pg19_test id {gid}"
    o = overlap(rec.get("text") or "", a.stride)
    return f"pg19_test 13-gram overlap {o:.3f}" if o > a.overlap else None


def convert(path: str):
    import pyarrow as pa
    import pyarrow.parquet as pq
    a = _ARGS
    group = tuple(a.group.split(":")) if a.group else None
    out = os.path.join(a.out, os.path.basename(path).replace(".json.gz", ".parquet"))
    schema = pa.schema([("id", pa.string()), ("title", pa.string()), ("text", pa.large_string())])
    rows, size, removed = {"id": [], "title": [], "text": []}, 0, []
    n_in = n_out = c_in = c_out = 0
    tmp = out + ".tmp"
    with pq.ParquetWriter(tmp, schema, compression="zstd") as w, gzip.open(path, "rt", encoding="utf-8") as fh:
        def flush():
            nonlocal rows, size
            if rows["id"]:
                w.write_table(pa.table(rows, schema=schema))
            rows, size = {"id": [], "title": [], "text": []}, 0
        for rec in records(fh, group):
            pages = rec.pop("_pages", None) or [rec]
            kept = []
            for pg in pages:
                text = pg.get("text") or ""
                n_in += 1
                c_in += len(text)
                if len(text) < a.min_chars:
                    continue
                why = test_book(pg, a)
                if why:
                    removed.append({"file": os.path.basename(path), "id": str(pg.get("id")),
                                    "title": (pg.get("metadata") or {}).get("title"), "chars": len(text), "why": why})
                    continue
                kept.append(text)
            if not kept:
                continue
            text = "\n\n".join(kept)
            rows["id"].append(str(rec.get("id")))
            rows["title"].append(str((rec.get("metadata") or {}).get("title") or ""))
            rows["text"].append(text)
            size += len(text)
            n_out += 1
            c_out += len(text)
            if size >= a.row_group_mb * 2 ** 20:
                flush()
        flush()
    os.replace(tmp, out)
    return {"file": os.path.basename(path), "docs_in": n_in, "docs_out": n_out, "chars_in": c_in, "chars_out": c_out,
            "removed": removed}


def main(argv=None):
    global _TEST_GRAMS, _TEST_IDS, _ARGS
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", required=True, help="directory of *.json.gz (searched recursively)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pg19_test", default="bucket/data/raw/pg19_test/test.parquet")
    ap.add_argument("--stride", type=int, default=32, help="every stride-th 13-gram of a candidate is checked")
    ap.add_argument("--overlap", type=float, default=0.2, help="remove above this share of the test books' 13-grams")
    ap.add_argument("--min_chars", type=int, default=0)
    ap.add_argument("--row_group_mb", type=float, default=64)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--group", default="", help="KEY:ORDER - join records sharing KEY into one document (pages -> volumes)")
    a = ap.parse_args(argv)
    _ARGS = a
    import pyarrow.parquet as pq
    t0 = time.time()
    test = pq.read_table(a.pg19_test, columns=["url", "text"])
    _TEST_IDS = {int(u.rstrip("/").split("/")[-1]) for u in test.column("url").to_pylist()}
    for text in test.column("text").to_pylist():
        _TEST_GRAMS.update(grams(text))
    print(f"[dolma] {len(_TEST_IDS)} PG-19 test books, {len(_TEST_GRAMS)} 13-grams ({time.time() - t0:.0f}s)", flush=True)
    files = sorted(glob.glob(os.path.join(a.src, "**", "*.json.gz"), recursive=True))
    if not files:
        sys.exit(f"no *.json.gz under {a.src}")
    os.makedirs(a.out, exist_ok=True)
    results = []
    with Pool(a.workers) as pool:                  # forked after the test set is built: it is shared
        for r in pool.imap_unordered(convert, files):
            results.append(r)
            print(f"[dolma] {r['file']}: {r['docs_out']}/{r['docs_in']} documents, {len(r['removed'])} removed "
                  f"({time.time() - t0:.0f}s)", flush=True)
    removed = [x for r in results for x in r["removed"]]
    with open(os.path.join(a.out, "removed.jsonl"), "w") as f:
        for x in removed:
            f.write(json.dumps(x) + "\n")
    summary = {"src": a.src, "files": len(files), "docs_in": sum(r["docs_in"] for r in results),
               "docs_out": sum(r["docs_out"] for r in results), "chars_in": sum(r["chars_in"] for r in results),
               "chars_out": sum(r["chars_out"] for r in results), "removed": len(removed),
               "removed_by_id": sum(x["why"].startswith("pg19_test id") for x in removed),
               "stride": a.stride, "overlap": a.overlap, "min_chars": a.min_chars, "group": a.group}
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print(f"[dolma] done: {json.dumps(summary)} ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
