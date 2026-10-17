#!/usr/bin/env python3
# coding=utf-8
"""
A document switch in front of a TTT memory trained on single-document windows (the packing question of the main run):
sequences of document A (``--n_a`` chunks, its last token replaced by EOS) followed by document B (``--n_b`` chunks),
both whole documents from the held-out tails of their sources (never a window across a boundary), B starting at a
chunk boundary, global positions running on.  Every chunk is written with the checkpoint's own write and read first
(i.i.d. masks at each ``--t``, the same masks for every variant) against

    carry   the memory as written so far - A, then B: what a packed sequence gives the model
    reset   B's chunks alone: the memory reset to W_0 at the switch, the windows unchanged (A's last chunks stay in
            B's first windows), so carry - reset isolates A's content left in the memory
    off     W_0 with the carry state's windows: no memory

and per chunk offset after the switch: stale = carry - reset (A's leftover content; > 0 hurts), gain = off - reset
(B's own memory), net = off - carry.  A's chunks give the reference gain (off - carry).  Pairs: A from source i and
B from source i + 1 (``cross``: another domain), and A, B two documents of source i (``same``).

``--targets``: also the passkeys of ``tools/j_lens/context_learning.py`` (``--tasks``, l2r read) on the switch
sequences - taught in A (the teaching block ends with chunk 3), asked at these chunks of B: whether a fact crosses the
switch.  Compare the shares with the same tool's single-document results.

    JAX_PLATFORMS=cuda python -m tools.j_lens.doc_switch --ckpt <step dir> --sources <name=dir:1.0,...> \\
        --n_a 8 --n_b 8 --n_per_pair 3 --t 0.3,0.6,0.9 --targets 9,13 --out <npz>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np

from tools.j_lens import _cli

VARIANTS = ("carry", "reset", "off")


def held_out_docs(src_dir: str, holdout: int, eos: int, need: int, count: int):
    """The first ``count`` whole documents of at least ``need`` tokens in a stream's held-out tail (its last
    ``holdout`` tokens, from the first EOS on), each as its first ``need`` tokens."""
    mm = np.memmap(os.path.join(src_dir, "tokens.u32"), dtype="<u4", mode="r")
    tail = np.asarray(mm[max(0, len(mm) - holdout):])
    ends = np.flatnonzero(tail == eos)
    out = []
    for s, e in zip(ends[:-1] + 1, ends[1:]):
        if e - s >= need:
            out.append(tail[s:s + need].astype(np.int32))
            if len(out) == count:
                break
    if len(out) < count:
        raise ValueError(f"{src_dir}: {len(out)} held-out documents of >= {need} tokens, {count} needed")
    return out


def switch_sequences(sources, n_a: int, n_b: int, C: int, n_per_pair: int, holdout: int, eos: int):
    """``([(pair name, tokens [(n_a + n_b) C])], kinds)``: per source i, ``n_per_pair`` cross pairs (A from i, B from
    i + 1) and as many same-source pairs (B another document of i)."""
    need = max(n_a, n_b) * C
    docs = {name: held_out_docs(d, holdout, eos, need, 2 * n_per_pair) for name, d in sources}
    names = list(docs)
    seqs, kinds = [], []
    for i, a in enumerate(names):
        nxt = names[(i + 1) % len(names)]
        for j in range(n_per_pair):
            for kind, b_name, B in (("cross", nxt, docs[nxt][j]), ("same", a, docs[a][n_per_pair + j])):
                A = docs[a][j]
                seq = np.concatenate([A[:n_a * C - 1], np.array([eos], np.int32), B[:n_b * C]]).astype(np.int32)
                seqs.append((f"{a}>{b_name}", seq))
                kinds.append(kind)
    return seqs, kinds


def run(model, seqs, ws, n_a: int, ts, seed: int, compute_dtype, log=print):
    """``{"ce": [n, chunks, variants, t]`` CE sums over the scored slots (NaN where a variant does not apply: reset
    before the switch), ``"count": [n, chunks, t]``}``."""
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    from tools.j_lens import core
    cfg = model.cfg
    C = cfg.chunk_size
    if not ws.outer_shift or model.gdn is not None or cfg.read_block:
        raise ValueError("doc switch: the TTT memory with the outer shift (no gdn, no read_block)")
    f32 = jnp.float32
    key0 = jax.random.PRNGKey(0)                                      # the causal write draws nothing
    commit = eqx.filter_jit(lambda m, ids, pos, st: core.commit(m, ids, pos, st, key0, ws, compute_dtype))
    reset = eqx.filter_jit(lambda m, st: core.reset_memory(m, st))

    def _ce(m, x, pos, st, tgt, valid):
        lg = core.read_forward(m, x, pos, st, None, True, compute_dtype).logits.astype(f32)
        ce = -jnp.take_along_axis(jax.nn.log_softmax(lg, axis=-1), tgt[:, None], axis=-1)[:, 0]
        return jnp.sum(jnp.where(valid, ce, 0.0))
    ce = eqx.filter_jit(_ce)
    n, T = len(seqs), len(seqs[0][1])
    nc = T // C
    if not 0 < n_a < nc:
        raise ValueError(f"n_a {n_a}: the switch must fall inside the {nc} chunks")
    out = {"ce": np.full((n, nc, len(VARIANTS), len(ts)), np.nan), "count": np.zeros((n, nc, len(ts)))}
    t0 = time.time()
    for si, (_, seq) in enumerate(seqs):
        st, st_r = core.init_memory(model, compute_dtype), None
        for c in range(nc):
            pos, clean = jnp.arange(c * C, (c + 1) * C), jnp.asarray(seq[c * C:(c + 1) * C])
            if c == n_a:
                st_r = reset(model, st)                              # the memory reset at the switch
            tgt = jnp.concatenate([clean[1:], clean[-1:]])           # slot j predicts token j + 1
            off = reset(model, st)
            for ti, t in enumerate(ts):
                k = jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(seed), si), c * 64 + ti)
                x, _, valid = core.read_mask(clean, float(t), k, cfg.mask_token_id, True)
                out["count"][si, c, ti] = float(jnp.sum(valid))
                out["ce"][si, c, 0, ti] = float(ce(model, x, pos, st, tgt, valid))
                out["ce"][si, c, 2, ti] = float(ce(model, x, pos, off, tgt, valid))
                if st_r is not None:
                    out["ce"][si, c, 1, ti] = float(ce(model, x, pos, st_r, tgt, valid))
            st = commit(model, clean, pos, st)
            if st_r is not None:
                st_r = commit(model, clean, pos, st_r)
        log(f"[switch] sequence {si + 1}/{n}: {time.time() - t0:.0f}s", flush=True)
    return out


def summarise(o: dict, kinds, n_a: int, n_boot: int = 1000, seed: int = 0) -> dict:
    """Per chunk offset from the switch (negative: A's chunks) and pair kind: token-weighted CE of every variant and
    the paired differences with bootstrap SEs over sequences."""
    ce, cnt = o["ce"].sum(-1), o["count"].sum(-1)                    # [n, chunks, variants], [n, chunks]
    kinds = np.asarray(kinds)
    rng = np.random.default_rng(seed)
    rows = []
    for kind in ("all", "cross", "same"):
        sel = np.flatnonzero(np.ones(len(kinds), bool) if kind == "all" else kinds == kind)
        if not len(sel):
            continue
        boots = sel[rng.integers(0, len(sel), (n_boot, len(sel)))]
        for c in range(ce.shape[1]):
            m = lambda v, idx: ce[idx, c, v].sum(-1) / cnt[idx, c].sum(-1)
            row = {"kind": kind, "offset": c - n_a, "tokens": int(cnt[sel, c].sum())}
            for vi, v in enumerate(VARIANTS):
                row[v] = float(m(vi, sel))
            pairs = {"net": (2, 0)} if c < n_a else {"stale": (0, 1), "gain": (2, 1), "net": (2, 0)}
            for name, (p, q) in pairs.items():
                row[name] = [float(m(p, sel) - m(q, sel)), float(np.std(m(p, boots) - m(q, boots), ddof=1))]
            rows.append(row)
    return {"n_a": n_a, "rows": rows}


def report(s: dict, tag: str):
    print(f"[switch] {tag}: CE per scored token; stale = carry - reset (> 0: A's content hurts B), gain = off - reset, "
          f"net = off - carry", flush=True)
    for r in s["rows"]:
        if r["kind"] == "all" or r["offset"] >= 0:
            extra = "" if r["offset"] < 0 else (f"  stale {r['stale'][0]:+.4f} ± {r['stale'][1]:.4f}  "
                                                f"gain {r['gain'][0]:+.4f} ± {r['gain'][1]:.4f}")
            print(f"[switch] {r['kind']:>5s} {r['offset']:+3d}  carry {r['carry']:.4f}  off {r['off']:.4f}  "
                  f"net {r['net'][0]:+.4f} ± {r['net'][1]:.4f}{extra}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    _cli.add_common(ap)
    ap.add_argument("--n_a", type=int, default=8, help="chunks of document A (the switch at chunk n_a)")
    ap.add_argument("--n_b", type=int, default=8, help="chunks of document B")
    ap.add_argument("--n_per_pair", type=int, default=3, help="cross pairs and same-source pairs per source")
    ap.add_argument("--t", default="0.3,0.6,0.9", help="mask ratios of the reads")
    ap.add_argument("--targets", default="", help="chunks of B to ask the passkeys at (empty: no passkeys)")
    ap.add_argument("--tasks", default="key1,key4")
    ap.add_argument("--tokenizer", default="configs/1b_e2e/tokenizer.json")
    ap.add_argument("--task_seed", type=int, default=20261009)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    _cli.require_platform()
    import jax.numpy as jnp
    model, ws = _cli.load(a)
    cfg = model.cfg
    from src.ttt.data import split_sources
    srcs = [(name, path) for name, path, _ in split_sources(a.sources)]
    seqs, kinds = switch_sequences(srcs, a.n_a, a.n_b, cfg.chunk_size, a.n_per_pair, a.holdout, cfg.eos_token_id)
    ts = [float(x) for x in a.t.split(",") if x]
    o = run(model, seqs, ws, a.n_a, ts, a.seed, jnp.dtype(a.dtype))
    s = summarise(o, kinds, a.n_a)
    s.update(t=ts, sequences=len(seqs))
    tag = a.tag or (a.ckpt or "")
    report(s, tag)
    extra = {}
    if a.targets:
        from tools.j_lens import context_learning as CL
        tasks = [x for x in a.tasks.split(",") if x]
        tok = CL.Tok(a.tokenizer, cfg.vocab_size)
        if not (tok.real or a.random_init):
            sys.exit(f"--tokenizer {a.tokenizer}: missing, or larger than the model's vocabulary")
        o2, targets = CL.run(model, seqs, ws, [int(x) for x in a.targets.split(",") if x], a.task_seed,
                             jnp.dtype(a.dtype), tok, tasks, read_mode="l2r")
        s2 = CL.summarise(o2, tasks, targets, cfg.chunk_size)
        CL.report(s2, f"{tag} (switch at chunk {a.n_a})")
        s["passkeys"] = s2
        extra = {f"cl_{k}": v for k, v in o2.items()}
    meta = {"ckpt": a.ckpt, "inner_lr": ws.inner_lr, "n_a": a.n_a, "n_b": a.n_b, "n_per_pair": a.n_per_pair,
            "t": ts, "targets": a.targets, "tasks": a.tasks, "dtype": a.dtype}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, names=np.array([nm for nm, _ in seqs]), kinds=np.array(kinds), meta=np.array(json.dumps(meta)),
             summary=np.array(json.dumps(s)), **o, **extra)
    print(f"[switch] saved {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
