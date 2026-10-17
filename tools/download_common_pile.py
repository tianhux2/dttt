#!/usr/bin/env python3
# coding=utf-8
"""
Download sources of the Common Pile v0.1 (EleutherAI, 2025; public domain and openly licensed text) in their
filtered form - dolma JSON lines, gzipped - into ``<out>/<source>/``, and record the dataset revisions in
``<out>/manifest.json`` (docs/data.md).  Network only: safe on a login node with
few workers.  The conversion to parquet (and the PG-19 test books' removal) is ``tools/convert_dolma.py``.

    .venv/bin/python tools/download_common_pile.py --sources project_gutenberg,pre_1929_books --workers 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

# long-document sources (books, papers) first
DEFAULT = ("project_gutenberg,pre_1929_books,library_of_congress,biodiversity_heritage_library,doab,arxiv_papers")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sources", default=DEFAULT, help="comma-separated names; the repository is common-pile/<name>_filtered")
    ap.add_argument("--out", default="bucket/data/raw/common_pile")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max_files", type=int, default=0, help="if > 0, only a source's first files by name (a part)")
    a = ap.parse_args(argv)
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi()
    os.makedirs(a.out, exist_ok=True)
    man_path = os.path.join(a.out, "manifest.json")
    manifest = json.load(open(man_path)) if os.path.exists(man_path) else {}
    for name in a.sources.split(","):
        repo = f"common-pile/{name}_filtered"
        info = api.dataset_info(repo, files_metadata=True)
        files = sorted((s for s in info.siblings if s.rfilename.endswith(".json.gz")), key=lambda s: s.rfilename)
        if a.max_files > 0:
            files = files[:a.max_files]
        size = sum(s.size or 0 for s in files)
        t0 = time.time()
        print(f"[cp] {repo} @ {info.sha}: {len(files)} files, {size / 2 ** 30:.1f} GiB", flush=True)
        snapshot_download(repo, repo_type="dataset", revision=info.sha, local_dir=os.path.join(a.out, name),
                          allow_patterns=[s.rfilename for s in files] + ["README.md"], max_workers=a.workers)
        manifest[name] = {"repo": repo, "revision": info.sha, "files": len(files), "bytes": size,
                          "file_names": [s.rfilename for s in files], "downloaded": time.strftime("%Y-%m-%d %H:%M:%S")}
        json.dump(manifest, open(man_path, "w"), indent=1)
        print(f"[cp] {name} done in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
