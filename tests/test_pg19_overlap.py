# coding=utf-8
"""``tools/pg19_overlap.py`` on a synthetic stream: two of four documents carry a PG-19 test book (one a copy, one
quoting most of the other book), one is unrelated and one is too short to be scored.  The tool must name the two
books, write their stream indices - checked against the tokens - for ``ttt.data filter --exclude``, and refuse a
derived stream; the filter then drops exactly those documents.  CPU only (the tokenizer, no JAX).

    .venv/bin/python tests/test_pg19_overlap.py
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
failures = []


def report(name, ok, detail=""):
    failures.append(name) if not ok else None
    print(f"[{'PASS' if ok else 'FAIL'}] {name:<78} {detail}")


def words(rng, n):
    return " ".join("".join(rng.choice("abcdefghijklmnop") for _ in range(rng.randint(3, 8))) for _ in range(n))


def main():
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from src.ttt.data import filter_long, prepare
    rng = random.Random(0)
    book_a, book_b = words(rng, 3000), words(rng, 3000)
    with tempfile.TemporaryDirectory() as tmp:
        test = os.path.join(tmp, "pg19.parquet")
        pq.write_table(pa.table({"short_book_title": ["Book A", "Book B"], "publication_date": [1900, 1901],
                                 "url": ["http://www.gutenberg.org/ebooks/111", "http://www.gutenberg.org/ebooks/222"],
                                 "text": [book_a, book_b]}), test)
        docs = [words(rng, 2000), book_b[:15000] + " " + words(rng, 300), "a short note", book_a]
        src = os.path.join(tmp, "src.parquet")
        pq.write_table(pa.table({"text": docs}), src, row_group_size=2)
        d = prepare("s", [src], "text", os.path.join(ROOT, "configs", "1b_e2e"), os.path.join(tmp, "tok"), workers=1)
        out = os.path.join(tmp, "out")
        cmd = [sys.executable, "tools/pg19_overlap.py", "--out", out, "--pg19_test", test, "--min_chars", "1000",
               "--stride", "4", "--workers", "1", "--streams"]
        r = subprocess.run(cmd + [f"s={d}"], cwd=ROOT, capture_output=True, text=True, timeout=600)
        ok = r.returncode == 0
        detail = r.stderr[-400:] if not ok else ""
        if ok:
            excl = json.load(open(os.path.join(out, "exclude_s.json")))
            s = json.load(open(os.path.join(out, "scan.json")))["streams"]["s"]
            ok = excl == [1, 3] and s["books"] == ["Book A", "Book B"] and s["docs"] == 4 and s["scored"] == 3
            detail = f"exclude {excl}, books {s['books']}, scored {s['scored']}/{s['docs']}"
        report("pg19_overlap: the two test-book documents found, named and indexed", ok, detail)

        f = filter_long(d, "s_dc", tmp, exclude=json.load(open(os.path.join(out, "exclude_s.json"))) if ok else [1, 3])
        m = json.load(open(os.path.join(f, "meta.json")))
        full = np.fromfile(os.path.join(d, "tokens.u32"), dtype="<u4")
        got = np.fromfile(os.path.join(f, "tokens.u32"), dtype="<u4")
        eos = int(m["tokenizer"]["eos_token_id"])
        ends = np.flatnonzero(full == eos)
        want = np.concatenate([full[0:ends[0] + 1], full[ends[1] + 1:ends[2] + 1]])
        ok = np.array_equal(got, want) and m["num_docs"] == 2 and m["excluded_docs"] == 2
        report("ttt.data filter --exclude drops them and keeps the rest", ok, f"{m['num_docs']} docs kept")

        r = subprocess.run(cmd + [f"s={f}"], cwd=ROOT, capture_output=True, text=True, timeout=600)
        report("pg19_overlap refuses a derived stream (its indices are not parquet rows)", r.returncode != 0,
               r.stderr.strip().splitlines()[-1][:80] if r.stderr.strip() else "")
    print("ALL PASS" if not failures else "FAILURES: " + "; ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
