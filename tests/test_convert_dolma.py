# coding=utf-8
"""``tools/convert_dolma.py`` on synthetic files: a PG-19 test book's Project Gutenberg record is removed by its id, a
document that quotes most of a test book by its 13-grams, an unrelated one and a short one are kept or removed as
--overlap / --min_chars say; the parquet holds the kept documents' id, title and text in order, in row groups of
about --row_group_mb; the summary and removed.jsonl count them.  With --group item_id:page_num, pages (shuffled within
their volume, one quoting a test book) become one document per volume, in page order, without the quoting page.
Pure Python (no JAX).

    .venv/bin/python tests/test_convert_dolma.py
"""
from __future__ import annotations

import gzip
import json
import os
import random
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
failures = []


def report(name, ok, detail=""):
    failures.append(name) if not ok else None
    print(f"[{'PASS' if ok else 'FAIL'}] {name:<78} {detail}")


def words(rng, n):
    return " ".join("".join(rng.choice("abcdefghijklmnop") for _ in range(rng.randint(3, 8))) for _ in range(n))


def main():
    import pyarrow as pa
    import pyarrow.parquet as pq
    rng = random.Random(0)
    book_a, book_b = words(rng, 4000), words(rng, 4000)
    with tempfile.TemporaryDirectory() as tmp:
        pq.write_table(pa.table({"short_book_title": ["A", "B"], "publication_date": [1900, 1901],
                                 "url": ["http://www.gutenberg.org/ebooks/111", "http://www.gutenberg.org/ebooks/222"],
                                 "text": [book_a, book_b]}), os.path.join(tmp, "pg19.parquet"))
        src = os.path.join(tmp, "src")
        os.makedirs(src)
        recs = [
            {"id": "111", "metadata": {"title": "A copy", "url": "https://www.gutenberg.org/ebooks/111.txt.utf-8"},
             "text": words(rng, 3000)},                                                   # removed by id
            {"id": "x1", "metadata": {"title": "quotes B"}, "text": book_b[:30000] + " " + words(rng, 500)},  # by text
            {"id": "x2", "metadata": {"title": "other"}, "text": words(rng, 6000)},          # kept
            {"id": "x3", "metadata": {"title": "short"}, "text": "tiny"},                     # below min_chars
            {"id": "x4", "metadata": {"title": "other 2"}, "text": words(rng, 6000)},        # kept
        ]
        with gzip.open(os.path.join(src, "part-0000.json.gz"), "wt") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        out = os.path.join(tmp, "out")
        r = subprocess.run([sys.executable, "tools/convert_dolma.py", "--src", src, "--out", out, "--pg19_test",
                            os.path.join(tmp, "pg19.parquet"), "--stride", "4", "--overlap", "0.2", "--min_chars", "10",
                            "--row_group_mb", "0.02", "--workers", "1"], cwd=ROOT, capture_output=True, text=True,
                           timeout=600)
        ok = r.returncode == 0
        detail = r.stderr[-400:] if not ok else ""
        if ok:
            pf = pq.ParquetFile(os.path.join(out, "part-0000.parquet"))
            t = pf.read()
            ids = t.column("id").to_pylist()
            removed = [json.loads(x) for x in open(os.path.join(out, "removed.jsonl"))]
            s = json.load(open(os.path.join(out, "summary.json")))
            whys = sorted(x["why"].split()[1] for x in removed)
            ok = (ids == ["x2", "x4"] and t.column_names == ["id", "title", "text"] and pf.num_row_groups == 2
                  and whys == ["13-gram", "id"] and s["docs_in"] == 5 and s["docs_out"] == 2 and s["removed"] == 2
                  and s["removed_by_id"] == 1 and t.column("title").to_pylist() == ["other", "other 2"])
            detail = f"kept {ids}, removed {whys}, row groups {pf.num_row_groups}, summary {s['docs_out']}/{s['docs_in']}"
        report("convert_dolma: test books removed by id and by 13-grams, short dropped, the rest kept in order", ok,
               detail)

        pages = {("A", "0003"): words(rng, 300), ("A", "0001"): words(rng, 300), ("A", "0010"): words(rng, 300),
                 ("A", "0002"): book_b[:30000], ("B", "0002"): words(rng, 300), ("B", "0001"): words(rng, 300)}
        src2 = os.path.join(tmp, "src2")
        os.makedirs(src2)
        with gzip.open(os.path.join(src2, "pages-0000.json.gz"), "wt") as f:
            for (item, num), text in pages.items():
                f.write(json.dumps({"id": f"{item}-{num}", "item_id": item, "page_num": num, "metadata": {},
                                    "text": text}) + "\n")
        out2 = os.path.join(tmp, "out2")
        r = subprocess.run([sys.executable, "tools/convert_dolma.py", "--src", src2, "--out", out2, "--pg19_test",
                            os.path.join(tmp, "pg19.parquet"), "--stride", "4", "--group", "item_id:page_num",
                            "--workers", "1"], cwd=ROOT, capture_output=True, text=True, timeout=600)
        ok = r.returncode == 0
        detail = r.stderr[-400:] if not ok else ""
        if ok:
            t = pq.read_table(os.path.join(out2, "pages-0000.parquet"))
            got = dict(zip(t.column("id").to_pylist(), t.column("text").to_pylist()))
            want = {"A": "\n\n".join(pages[("A", n)] for n in ("0001", "0003", "0010")),
                    "B": "\n\n".join(pages[("B", n)] for n in ("0001", "0002"))}
            removed = [json.loads(x)["id"] for x in open(os.path.join(out2, "removed.jsonl"))]
            s = json.load(open(os.path.join(out2, "summary.json")))
            ok = (got == want and t.column("id").to_pylist() == ["A", "B"] and removed == ["A-0002"]
                  and s["docs_in"] == 6 and s["docs_out"] == 2 and s["group"] == "item_id:page_num")
            detail = f"volumes {list(got)}, removed {removed}, summary {s['docs_out']}/{s['docs_in']}"
        report("convert_dolma --group: pages joined per volume in page order, a quoting page removed", ok, detail)
    print("ALL PASS" if not failures else "FAILURES: " + "; ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
