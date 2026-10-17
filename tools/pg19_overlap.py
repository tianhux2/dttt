#!/usr/bin/env python3
# coding=utf-8
"""
PG-19 test books in prepared token streams (docs/data.md): every document of at least
--min_chars characters in a stream's parquet files is scored by the share of its word 13-grams (every --stride-th,
``tools/convert_dolma.py``'s measure) found among the 100 test books' 13-grams.  For the documents above --overlap
(0.2, convert_dolma's threshold) the test book they come from is named and their index in the stream is written to
``<out>/exclude_<name>.json`` for ``python -m ttt.data filter --exclude`` - each index checked against the stream (the
document's first 32 tokens and its length equal the tokenised parquet text).  ``<out>/scan.json`` keeps every
document above 0.05.

A stream's document index is its row's position in the parquet files (``ttt.data prepare`` writes file, row-group and
row order and, with --min_doc_tokens 0, keeps every document); streams with dropped documents are refused.  CPU work:
run it on a compute node, never on a login node.

    .venv/bin/python tools/pg19_overlap.py --out log/data_v2/pg19_overlap --workers 32 \\
        --streams books3=bucket/data/tokens/books3_676852e1528366e3 dclm=bucket/data/tokens/dclm_365affb8c03b13a6
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import convert_dolma as cd  # noqa: E402

_A = None


def scan(job):
    import pyarrow.parquet as pq
    name, path, rg = job
    texts = pq.ParquetFile(path).read_row_group(rg, columns=["text"]).column("text").to_pylist()
    n = n_long = 0
    hits = []
    for row, x in enumerate(texts):
        n += 1
        if not x or len(x) < _A.min_chars:
            continue
        n_long += 1
        o = cd.overlap(x, _A.stride)
        if o > 0.05:
            hits.append({"file": os.path.basename(path), "rg": rg, "row": row, "chars": len(x), "overlap": round(o, 4),
                         "head": x[:120].replace("\n", " ")})
    return name, n, n_long, hits


def main(argv=None):
    global _A
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--streams", nargs="+", required=True, help="NAME=STREAM_DIR (a ttt.data prepare output)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pg19_test", default="bucket/data/raw/pg19_test/test.parquet")
    ap.add_argument("--tokenizer", default="configs/1b_e2e")
    ap.add_argument("--min_chars", type=int, default=8000, help="shorter documents are not scored")
    ap.add_argument("--stride", type=int, default=32)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=16)
    a = _A = ap.parse_args(argv)
    import numpy as np
    import pyarrow.parquet as pq
    t0 = time.time()
    test = pq.read_table(a.pg19_test, columns=["short_book_title", "url", "text"])
    titles, urls = test.column("short_book_title").to_pylist(), test.column("url").to_pylist()
    owner = {}
    for b, text in enumerate(test.column("text").to_pylist()):
        for g in cd.grams(text):
            owner.setdefault(g, b)
    cd._TEST_GRAMS.update(owner)
    print(f"[pg19] {len(titles)} test books, {len(owner)} 13-grams ({time.time() - t0:.0f}s)", flush=True)

    streams, jobs = {}, []
    for spec in a.streams:
        name, d = spec.split("=", 1)
        meta = json.load(open(os.path.join(d, "meta.json")))
        if meta.get("dropped_docs") or meta.get("min_doc_tokens") or "derived_from" in meta or "group_key" in meta:
            sys.exit(f"{name}: documents were dropped, regrouped or derived - its indices are not parquet rows")
        streams[name] = (d, meta)
        for f in meta["files"]:
            jobs += [(name, f["path"], rg) for rg in range(pq.ParquetFile(f["path"]).num_row_groups)]
    res = {n: {"docs": 0, "scored": 0, "hits": []} for n in streams}
    with Pool(a.workers) as pool:                       # forked after the test 13-grams are built: shared
        for k, (name, n, n_long, hits) in enumerate(pool.imap_unordered(scan, jobs)):
            r = res[name]
            r["docs"] += n
            r["scored"] += n_long
            r["hits"] += hits
            if k % 500 == 0:
                print(f"[pg19] {k}/{len(jobs)} row groups ({time.time() - t0:.0f}s)", flush=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    os.makedirs(a.out, exist_ok=True)
    for name, (d, meta) in streams.items():
        r = res[name]
        r["hits"].sort(key=lambda h: (h["file"], h["rg"], h["row"]))
        base, n, pfs = {}, 0, {}
        for f in meta["files"]:
            pf = pfs[os.path.basename(f["path"])] = pq.ParquetFile(f["path"])
            for rg in range(pf.num_row_groups):
                base[(os.path.basename(f["path"]), rg)] = n
                n += pf.metadata.row_group(rg).num_rows
        if n != meta["num_docs"]:
            sys.exit(f"{name}: {n} parquet rows, {meta['num_docs']} documents")
        big = [h for h in r["hits"] if h["overlap"] > a.overlap]
        ends = None
        if big:
            stream = np.memmap(os.path.join(d, "tokens.u32"), dtype="<u4", mode="r")
            eos = int(meta["tokenizer"]["eos_token_id"])
            ends = np.concatenate([s + np.flatnonzero(np.asarray(stream[s:s + (1 << 28)]) == eos)
                                   for s in range(0, len(stream), 1 << 28)])
            if len(ends) != meta["num_docs"]:
                sys.exit(f"{name}: {len(ends)} EOS, {meta['num_docs']} documents")
        excl = []
        for h in big:
            i = base[(h["file"], h["rg"])] + h["row"]
            text = pfs[h["file"]].read_row_group(h["rg"], columns=["text"]).column("text")[h["row"]].as_py()
            start = 0 if i == 0 else int(ends[i - 1]) + 1
            ids = tok(text, add_special_tokens=False)["input_ids"]
            if np.asarray(stream[start:start + 32]).tolist() != ids[:32] or int(ends[i]) - start != len(ids):
                sys.exit(f"{name}: document {i} ({h['file']} rg {h['rg']} row {h['row']}) does not match its text")
            c = collections.Counter(owner[g] for g in cd.grams(text, a.stride) if g in owner)
            b, m = c.most_common(1)[0]
            h.update(doc=i, tokens=len(ids), book=titles[b], book_url=urls[b])
            excl.append(i)
            print(f"[pg19] {name}: doc {i} ({len(ids)} tokens, overlap {h['overlap']:.3f}) <- {titles[b][:60]} "
                  f"({urls[b]})", flush=True)
        json.dump(sorted(excl), open(os.path.join(a.out, f"exclude_{name}.json"), "w"))
        r.update(stream=d, above=len(big), books=sorted({h["book"] for h in big}))
        print(f"[pg19] {name}: {r['scored']} of {r['docs']} documents scored, {len(big)} above {a.overlap} from "
              f"{len(r['books'])} test books -> {a.out}/exclude_{name}.json", flush=True)
    json.dump({"min_chars": a.min_chars, "stride": a.stride, "overlap": a.overlap, "streams": res},
              open(os.path.join(a.out, "scan.json"), "w"), indent=1)
    books = collections.defaultdict(list)
    for name, r in res.items():
        for h in r["hits"]:
            if h["overlap"] > a.overlap:
                books[h["book"]].append(f"{name} {h['tokens']} tokens ({h['overlap']:.2f})")
    print(f"[pg19] {len(books)} test books found ({time.time() - t0:.0f}s):", flush=True)
    for b, v in sorted(books.items()):
        print(f"  {b[:60]}: {'; '.join(v)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
