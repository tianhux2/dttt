# coding=utf-8
"""Data pipeline: prepare round-trip, stateless sampler, multi-host agreement, mixing, epochs."""
from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.ttt.data import Prefetcher, Sampler, Source, parse_sources, prepare, tokenizer_fingerprint

TOK = os.path.join(os.path.dirname(__file__), "..", "configs", "1b_e2e")


def _write_parquet(path, texts):
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.table({"text": texts}), path, row_group_size=7)


def main():
    failures = []
    tmp = tempfile.mkdtemp()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOK)
    eos = tok.eos_token_id

    # 1. prepare: documents separated by exactly one EOS; short docs dropped; meta correct.
    texts = [f"document number {i} " * (3 + i % 5) for i in range(40)]
    pq_a = os.path.join(tmp, "a.parquet"); _write_parquet(pq_a, texts)
    d_a = prepare("a", [pq_a], "text", TOK, os.path.join(tmp, "tok"), min_doc_tokens=8, workers=2)
    stream = np.memmap(os.path.join(d_a, "tokens.u32"), dtype="<u4", mode="r")
    meta = json.load(open(os.path.join(d_a, "meta.json")))
    n_eos = int(np.sum(stream == eos))
    ref_kept = sum(1 for t in texts if len(tok(t, add_special_tokens=False)["input_ids"]) >= 8)
    ok = (n_eos == meta["num_docs"] == ref_kept and meta["num_tokens"] == len(stream)
          and meta["dropped_docs"] == 40 - ref_kept and meta["tokenizer"] == tokenizer_fingerprint(tok))
    failures += [] if ok else [f"prepare: eos={n_eos} docs={meta['num_docs']} ref={ref_kept}"]
    print(f"[{'PASS' if ok else 'FAIL'}] prepare: one EOS per doc, min_doc filter    {meta['num_docs']} docs, {len(stream)} tokens")

    # idempotent + content-addressed
    ok = prepare("a", [pq_a], "text", TOK, os.path.join(tmp, "tok"), min_doc_tokens=8, workers=2) == d_a
    d_a2 = prepare("a", [pq_a], "text", TOK, os.path.join(tmp, "tok"), min_doc_tokens=0, workers=2)
    ok = ok and d_a2 != d_a
    failures += [] if ok else ["cache path not content-addressed"]
    print(f"[{'PASS' if ok else 'FAIL'}] cache is content-addressed and idempotent")

    # 2. sampler is a pure function of (seed, step): two instances agree; different seed differs.
    pq_b = os.path.join(tmp, "b.parquet"); _write_parquet(pq_b, [f"other corpus text {i} " * 6 for i in range(60)])
    d_b = prepare("b", [pq_b], "text", TOK, os.path.join(tmp, "tok"), workers=2)
    srcs = parse_sources(f"a={d_a2}:0.7,b={d_b}:0.3")
    T, G = 16, 8
    s1 = Sampler(srcs, T, G, seed=1); s2 = Sampler(srcs, T, G, seed=1); s3 = Sampler(srcs, T, G, seed=2)
    ok = all(np.array_equal(s1.batch(st), s2.batch(st)) for st in (0, 5, 100_000)) \
        and not np.array_equal(s1.batch(0), s3.batch(0))
    failures += [] if ok else ["sampler not deterministic in (seed, step)"]
    print(f"[{'PASS' if ok else 'FAIL'}] batch(step) pure in (seed, step)")

    # 3. multi-host: union of per-host local batches == single-host global batch, in order.
    hosts = [Sampler(srcs, T, G, seed=1, rank=r, world=4) for r in range(4)]
    for st in (0, 3):
        got = np.concatenate([h.batch(st) for h in hosts])
        ok = np.array_equal(got, s1.global_batch(st))
        failures += [] if ok else [f"multi-host batch != global at step {st}"]
    print(f"[{'PASS' if ok else 'FAIL'}] union(per-host) == global batch")

    # 4. mixing ratio over many draws; each source sequence seen exactly once per source-epoch.
    N = 20_000
    ch = np.array([s1.locate(g)[0] for g in range(N)])
    frac = np.bincount(ch, minlength=2) / N
    ok = abs(frac[0] - 0.7) < 0.01
    failures += [] if ok else [f"mixing ratio {frac}"]
    print(f"[{'PASS' if ok else 'FAIL'}] mixing ratio within 1%                      {frac.round(3)}")
    n0 = int(s1.n_seq[0])
    idx0 = [s1.locate(g)[1] for g in range(N) if ch[g] == 0]
    first_epoch = idx0[:n0]
    ok = sorted(first_epoch) == list(range(n0)) and sorted(idx0[n0:2 * n0]) == list(range(n0))
    failures += [] if ok else ["source epoch is not a permutation"]
    print(f"[{'PASS' if ok else 'FAIL'}] each sequence once per source-epoch         n_seq[0]={n0}")

    # 5. extending steps never changes earlier batches (block-local RNG); resume = step only.
    late = Sampler(srcs, T, G, seed=1)
    late.batch(3000)                                # touch far-ahead state first
    ok = np.array_equal(late.batch(2), s1.batch(2))
    failures += [] if ok else ["earlier batches depend on later state"]
    print(f"[{'PASS' if ok else 'FAIL'}] earlier batches independent of later access")

    # 6. Prefetcher delivers in order.
    pf = Prefetcher(s1, start_step=10, depth=3)
    ok = all(np.array_equal(pf.get(st)[0], s1.batch(st)) and pf_m is None for st in (10, 11, 12, 13) for pf_m in [None])
    ok = ok and pf.get(14)[1] is None
    pf.close()
    failures += [] if ok else ["prefetcher order/content"]
    print(f"[{'PASS' if ok else 'FAIL'}] prefetcher in-order delivery")

    # filter_long: only documents of >= min tokens survive, in order, across block boundaries.
    from src.ttt.data import filter_long
    src = os.path.join(tmp, "flt_src"); os.makedirs(src)
    rng = np.random.default_rng(3)
    docs = [rng.integers(1, 50, size=int(n), dtype=np.uint32) for n in rng.integers(1, 12, size=400)]
    stream = np.concatenate([np.concatenate([d, [0]]) for d in docs]).astype("<u4")
    stream.tofile(os.path.join(src, "tokens.u32"))
    json.dump({"name": "flt_src", "num_tokens": int(len(stream)), "num_docs": len(docs), "sep": "eos",
               "tokenizer": {"class": "synthetic", "len": 64, "vocab_size": 64, "eos_token_id": 0}},
              open(os.path.join(src, "meta.json"), "w"))
    d = filter_long(src, "flt_long", tmp, 6, block_tokens=97)              # tiny blocks: many straddling docs
    got = np.fromfile(os.path.join(d, "tokens.u32"), dtype="<u4")
    want = np.concatenate([np.concatenate([x, [0]]) for x in docs if len(x) + 1 >= 6]).astype("<u4")
    m = json.load(open(os.path.join(d, "meta.json")))
    ok = np.array_equal(got, want) and m["num_docs"] == sum(len(x) + 1 >= 6 for x in docs) and m["num_tokens"] == len(want)
    failures += [] if ok else [f"filter_long: {len(got)} vs {len(want)} tokens, {m['num_docs']} docs"]
    print(f"[{'PASS' if ok else 'FAIL'}] filter_long keeps long docs across blocks   {m['num_docs']}/{len(docs)} docs, {len(want)} tokens")
    srcs = parse_sources(f"a={d}:1.0")
    ok = srcs[0].num_tokens == len(want) and srcs[0].vocab_len == 64
    failures += [] if ok else ["filtered source does not load"]
    print(f"[{'PASS' if ok else 'FAIL'}] filtered source loads as a Source")

    # filter_long with exclude: the listed documents (source indices, stream order) dropped too, across block
    # boundaries and for a last document without its EOS; the rest unchanged; its own cache entry.
    docs2 = docs[:-1] + [np.arange(1, 11, dtype=np.uint32)]
    src2 = os.path.join(tmp, "flt_src2"); os.makedirs(src2)
    np.concatenate([np.concatenate([x, [0]]) for x in docs2])[:-1].astype("<u4").tofile(os.path.join(src2, "tokens.u32"))
    json.dump(dict(json.load(open(os.path.join(src, "meta.json"))), name="flt_src2"), open(os.path.join(src2, "meta.json"), "w"))
    long_ids = [i for i, x in enumerate(docs2) if len(x) + 1 >= 6]
    short_ids = [i for i in range(len(docs2)) if i not in long_ids]
    ex = sorted({long_ids[0], long_ids[10], long_ids[11], len(docs2) - 1, short_ids[0]})     # one would go anyway
    keep_ids = [i for i in long_ids if i not in ex]
    d_ex = filter_long(src2, "flt_long", tmp, 6, block_tokens=97, exclude=ex)
    got = np.fromfile(os.path.join(d_ex, "tokens.u32"), dtype="<u4")
    want = np.concatenate([np.concatenate([docs2[i], [0]]) for i in keep_ids]).astype("<u4")
    m = json.load(open(os.path.join(d_ex, "meta.json")))
    ok = (np.array_equal(got, want) and m["num_docs"] == len(keep_ids) and m["exclude"] == ex
          and m["excluded_docs"] == len(set(ex) & set(long_ids)) and m["dropped_docs"] == len(docs2) - len(keep_ids)
          and d_ex != filter_long(src2, "flt_long", tmp, 6, block_tokens=97))
    failures += [] if ok else [f"filter_long exclude: {len(got)} vs {len(want)} tokens, {m.get('excluded_docs')} excluded"]
    print(f"[{'PASS' if ok else 'FAIL'}] filter_long drops the excluded documents      {m.get('excluded_docs')} excluded, {m['num_docs']} kept")

    # prepare_grouped: files sorted by (repo, path), repos below the token floor dropped
    # whole, files of a repo consecutive with one EOS each.
    from src.ttt.data import prepare_grouped
    import pyarrow as pa, pyarrow.parquet as pq
    rows = [("r2", "b.py", "def two(): return 2 " * 6), ("r1", "z.py", "print('z') " * 3), ("r3", "only.py", "x = 1"),
            ("r2", "a.py", "def one(): return 1 " * 6), ("r1", "a.py", "import os " * 3)]
    pq_g = os.path.join(tmp, "g.parquet")
    pq.write_table(pa.table({"max_stars_repo_name": [r[0] for r in rows], "max_stars_repo_path": [r[1] for r in rows],
                             "content": [r[2] for r in rows]}), pq_g, row_group_size=2)
    d_g = prepare_grouped("g", [pq_g], "content", "max_stars_repo_name", "max_stars_repo_path", TOK,
                          os.path.join(tmp, "tok"), min_group_tokens=12, workers=2)
    got = np.fromfile(os.path.join(d_g, "tokens.u32"), dtype="<u4")
    enc = lambda t: tok(t, add_special_tokens=False)["input_ids"]
    order = sorted(rows)                                                   # (repo, path)
    want, kept_groups = [], 0
    for repo in ("r1", "r2", "r3"):
        files = [r for r in order if r[0] == repo]
        ids = [enc(r[2]) for r in files]
        if sum(len(x) + 1 for x in ids) >= 12:
            kept_groups += 1
            for x in ids:
                want += x + [eos]
    mg = json.load(open(os.path.join(d_g, "meta.json")))
    ok = np.array_equal(got, np.array(want, dtype="<u4")) and mg["num_groups"] == kept_groups and mg["dropped_groups"] == 3 - kept_groups
    failures += [] if ok else [f"prepare_grouped: {len(got)} vs {len(want)} tokens, groups {mg['num_groups']}/{mg['dropped_groups']}"]
    print(f"[{'PASS' if ok else 'FAIL'}] prepare_grouped: sorted repos, whole-repo floor  {mg['num_groups']} kept, {mg['dropped_groups']} dropped")

    # length_stats on the synthetic stream: counts and shares match the generating lengths.
    from src.ttt.data import length_stats
    st_ = length_stats(src, thresholds=(6, 10))
    want6 = sum(len(x) + 1 for x in docs if len(x) + 1 >= 6)
    ok = st_["docs"] == len(docs) and st_["tokens"] == len(stream) and st_[">=6"]["tokens"] == want6
    failures += [] if ok else [f"length_stats: {st_}"]
    print(f"[{'PASS' if ok else 'FAIL'}] length_stats counts and token shares          {st_['docs']} docs, >=6: {st_['>=6']['token_share']:.2f}")

    # filter_long with max_doc_tokens: short document extraction (< 8 tokens)
    d_short = filter_long(src, "flt_short", tmp, min_doc_tokens=0, max_doc_tokens=8)
    st_s = length_stats(d_short, thresholds=(8,))
    ok = st_s[">=8"]["docs"] == 0 and st_s["docs"] > 0
    failures += [] if ok else [f"filter_short: docs >= 8 survived: {st_s}"]
    print(f"[{'PASS' if ok else 'FAIL'}] filter with max_doc_tokens keeps only < 8    {st_s['docs']} docs, max < 8")

    # Section 4.1: holdout fixed token interval disjointness & EOS alignment
    # Create a synthetic source of 300,000 tokens with periodic EOS tokens
    syn_dir = os.path.join(tmp, "syn_large")
    os.makedirs(syn_dir, exist_ok=True)
    rng = np.random.default_rng(123)
    n_tokens = 300_000
    syn_tokens = rng.integers(10, 1000, size=n_tokens, dtype=np.uint32)
    # Insert EOS every 500-1500 tokens
    pos = 0
    while pos < n_tokens - 100:
        pos += int(rng.integers(500, 1500))
        if pos < n_tokens:
            syn_tokens[pos] = eos
    syn_tokens.tofile(os.path.join(syn_dir, "tokens.u32"))
    with open(os.path.join(syn_dir, "meta.json"), "w") as f:
        json.dump({"name": "syn", "num_tokens": n_tokens, "tokenizer": {"len": 128256, "eos_token_id": int(eos)}}, f)

    syn_src = Source("syn", syn_dir, 1.0, n_tokens, 128256, int(eos))
    holdout_tok = 70_000

    s_8k = Sampler([syn_src], seq_len=8192, global_batch=2, seed=42, holdout_tokens=holdout_tok, eos=eos)
    s_32k = Sampler([syn_src], seq_len=32768, global_batch=1, seed=42, holdout_tokens=holdout_tok, eos=eos)
    s_128k = Sampler([syn_src], seq_len=131072, global_batch=1, seed=42, holdout_tokens=holdout_tok, eos=eos)

    # 1. 8K training max token range strictly < 32K/128K eval range
    max_8k_train_token = int(s_8k.n_seq[0]) * 8192
    eval_32k_start = int(s_32k.eval_start[0])
    eval_128k_start = int(s_128k.eval_start[0])

    ok_disjoint = (max_8k_train_token <= n_tokens - holdout_tok <= eval_32k_start) and (eval_32k_start == eval_128k_start)
    # 2. Token immediately preceding eval_start is EOS
    mm_syn = syn_src.open()
    ok_eos = int(mm_syn[eval_32k_start - 1]) == eos
    ok_holdout = ok_disjoint and ok_eos and s_32k.eval_batches(0) > 0
    failures += [] if ok_holdout else [f"holdout check failed: max_8k={max_8k_train_token}, eval_start={eval_32k_start}, prev_tok={mm_syn[eval_32k_start-1]}, eos={eos}"]
    print(f"[{'PASS' if ok_holdout else 'FAIL'}] holdout disjointness: 8K train [0, {max_8k_train_token}) vs 32K eval [{eval_32k_start}, {n_tokens}), prev token is EOS")

    # single-document windows (doc_windows + Sampler(windows=...)): aligned at document starts,
    # non-overlapping, inside one document (no EOS), before the held-out tail; rows == the stream at
    # the window; deterministic, ranks == global; a different data hash than packed slices
    from src.ttt.data import doc_windows
    wd = tempfile.mkdtemp()
    rng = np.random.default_rng(3)
    EOSW = 7
    lens = rng.integers(1, 200, 400)
    toks = np.concatenate([np.concatenate([rng.integers(10, 99, L - 1), [EOSW]]) for L in lens]).astype("<u4")
    toks.tofile(os.path.join(wd, "tokens.u32"))
    json.dump({"name": "w", "num_tokens": int(len(toks)), "num_docs": int(len(lens)),
               "tokenizer": {"len": 128, "eos_token_id": EOSW}}, open(os.path.join(wd, "meta.json"), "w"))
    T, H = 32, 2000
    w = doc_windows(wd, T, H)
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    expect = [int(st + k * T) for st, L in zip(starts, lens) for k in range((L - 1) // T) if st + (k + 1) * T <= len(toks) - H]
    ok = w.tolist() == expect and all(not (toks[x:x + T] == EOSW).any() for x in w)
    srcw = [Source("w", wd, 1.0, int(len(toks)), 128, EOSW)]
    smp = Sampler(srcw, T, 8, 5, holdout_tokens=H, eos=EOSW, windows=[w])
    smp2 = Sampler(srcw, T, 8, 5, holdout_tokens=H, eos=EOSW, windows=[w])
    ranks_w = [Sampler(srcw, T, 8, 5, rank=r, world=2, holdout_tokens=H, eos=EOSW, windows=[w]) for r in range(2)]
    packed = Sampler(srcw, T, 8, 5, holdout_tokens=H, eos=EOSW)
    wset = set(int(x) for x in w)
    for st in range(6):
        b = smp.global_batch(st)
        ok &= np.array_equal(b, smp2.global_batch(st)) and not (b == EOSW).any()
        ok &= np.array_equal(b, np.concatenate([r_.batch(st) for r_ in ranks_w]))
        ok &= all(any(np.array_equal(row, toks[x:x + T]) for x in wset) for row in b[:2])
    ok &= smp.n_seq[0] == len(w) and smp.hash != packed.hash and np.array_equal(smp.eval_batch(0, 0), packed.eval_batch(0, 0))
    failures += [] if ok else ["doc windows"]
    print(f"[{'PASS' if ok else 'FAIL'}] doc windows: aligned single-document windows before the holdout, "
          f"sampled deterministically ({len(w)} windows)")

    print("ALL PASS" if not failures else "FAILURES: " + "; ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
