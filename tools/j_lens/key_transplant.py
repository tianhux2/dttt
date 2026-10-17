#!/usr/bin/env python3
# coding=utf-8
"""
Key transplant: is one-off recall beyond the attention window lost at the read's keys, or in what the
memory holds?

The planted-recall probe as the evaluation's ``--eval_planted_recall`` runs it (``probes.planted_probe_batch``:
a span of 64 tokens from chunk 1 copied into chunk window + 3, its first 16 tokens visible, "alt": every other
later token masked, each scored token with a visible predecessor; seed 12345).  Chunks 0 .. window + 2 are
written with the checkpoint's own write; the copy's chunk is read in the outer stream's condition (only the
scored tokens masked, windows kept).  Per scored slot the target's CE is read with

    off       the memory reset at the read
    on        the memory after all the writes
    on_tx     "on", with every fast-weight layer's memory contribution at the scored slots taken at the
              WRITE's input of the source slot holding the same token (``core.Transplant``): the memory
              queried with the write's own keys; the read's context, the template's part and the stored
              memory are unchanged
    src       only chunks 0 .. 1 written (the later writes removed), and src_tx
    one       W_0 + the source chunk's write alone, and one_tx
    off_within  the copy also inside the window (chunk window + 2), memory off: attention's copy
    off_none / on_none  nothing planted, the same slots: the memory's generic gain there

recall = (off - on) - (off_none - on_none) and ceiling = off - off_within, as the evaluation's probe; every
gain is also given as a share of the ceiling.  on_tx >> on: the read's keys do not match the write's (the
copy's context makes other fast-weight inputs than the source's); on_tx ~ on: what is stored is too weak or
drowned (src / one: by the other chunks' writes or not).

    JAX_PLATFORMS=cuda python -m tools.j_lens.key_transplant --ckpt <step dir> --sources <name=dir:1.0,...> \\
        --n_per_source 8 --out log/key_transplant/<tag>.npz
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

VARIANTS = ("off", "on", "on_tx", "src", "src_tx", "one", "one_tx", "off_within", "off_none", "on_none")
PLANT_SEED = 12345                      # TrainConfig.eval_seed: the evaluation's planted draws


def match(ids: np.ndarray, lm: np.ndarray, C: int, cs: int, ct: int, length: int, cue: int, mode: str):
    """Pair the read with the write for one planted sequence ``ids [T]`` (scored targets ``lm [T]``).
    Returns ``(sel [C] bool, src [C] int)`` in chunk-relative slots: the read slots of chunk ``ct`` whose
    target is scored (slot j holds token j and predicts token j + 1, the outer shift) and, for each, the
    slot of the source chunk ``cs`` holding the same token (whose write predicted the same next token)."""
    scored = np.flatnonzero(lm)
    b = scored[0] - cue - (1 if mode == "alt" else 0)               # the copy's first position
    span = ids[b:b + length]
    chunk = ids[cs * C:(cs + 1) * C]
    hits = [a for a in range(C - length + 1) if np.array_equal(chunk[a:a + length], span)]
    if not hits or not (ct * C <= b and b + length <= (ct + 1) * C):
        raise RuntimeError("planted span not found in the source chunk")
    a = cs * C + hits[0]
    sel, src = np.zeros(C, bool), np.zeros(C, np.int32)
    for p in scored:
        j, i = p - 1, a + (p - 1 - b)                               # read slot and source slot (absolute)
        assert ids[j] == ids[i] and ids[j + 1] == ids[i + 1]
        sel[j - ct * C], src[j - ct * C] = True, i - cs * C
    return sel, src


def run(model, seqs, ws, length: int, cue: int, mode: str, seed: int, compute_dtype, log=print):
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    from src.ttt import probes as PR
    from src.ttt.model import cast_floating
    from tools.j_lens import core
    cfg = model.cfg
    C, Wn = cfg.chunk_size, cfg.window_chunks
    if not ws.outer_shift or ws.inner_task != "causal":
        raise ValueError("key transplant: the causal write with the outer shift only")
    cs, ct, cw = PR.planted_chunks(len(seqs[0]) // C, Wn)
    if ct != cw + 1:
        raise ValueError(f"planted chunks {cs, ct, cw}: the in-window copy must be the chunk before the target")
    key0 = jax.random.PRNGKey(0)                                    # the causal write draws nothing
    commit = eqx.filter_jit(lambda m, ids, pos, st: core.commit(m, ids, pos, st, key0, ws, compute_dtype))
    keys = eqx.filter_jit(lambda m, ids, pos, st: core.write_keys(m, ids, pos, st, ws, compute_dtype))

    def _ce(m, ids, pos, st, tgt, tx):
        x, clean = ids                                              # the masked chunk and its clean tokens
        own = core.clean_own(m, clean, pos, st, compute_dtype) if m.cfg.read_block else None
        lg = core.read_forward(m, x, pos, st, None, True, compute_dtype, transplant=tx, own_kv=own).logits
        lp = jax.nn.log_softmax(lg.astype(jnp.float32), axis=-1)
        return -jnp.take_along_axis(lp, tgt[:, None], axis=-1)[:, 0]
    ce = eqx.filter_jit(lambda m, ids, pos, st, tgt: _ce(m, ids, pos, st, tgt, None))
    ce_tx = eqx.filter_jit(lambda m, ids, pos, st, tgt, xs, sel: _ce(m, ids, pos, st, tgt, core.Transplant(xs, sel)))
    W0 = cast_floating(model.fast_weight, jnp.float32)
    one_w = eqx.filter_jit(lambda w0, a, b: jax.tree_util.tree_map(
        lambda x, y, z: x + (y - z) if eqx.is_inexact_array(x) else x, w0, a, b))

    def read_ids(ids, lm):
        """``((masked target chunk, its clean tokens), targets)`` - the clean tokens: a read_block model's own K/V"""
        chunk, m = ids[ct * C:(ct + 1) * C], lm[ct * C:(ct + 1) * C]
        tgt = np.concatenate([chunk[1:], chunk[-1:]])                # slot j predicts token j + 1
        return (jnp.asarray(np.where(m, cfg.mask_token_id, chunk)), jnp.asarray(chunk)), jnp.asarray(tgt)

    sums = np.zeros((len(seqs), len(VARIANTS)))
    counts = np.zeros(len(seqs))
    t0 = time.time()
    for si, seq in enumerate(seqs):
        pl = {p: PR.planted_probe_batch(np.asarray(seq, np.int32)[None], [si], seed, C, Wn, length, cue, p, mode,
                                        eos=cfg.eos_token_id) for p in PR.PLANT_PLACEMENTS}
        (ib, lb), (iw, lw), (i0, l0) = [(x[0][0], x[1][0]) for x in (pl["beyond"], pl["within"], pl["none"])]
        assert np.array_equal(lb, lw) and np.array_equal(lb, l0)
        assert np.array_equal(ib[:ct * C], i0[:ct * C]) and np.array_equal(ib[:cw * C], iw[:cw * C])
        sel, src = match(ib, lb, C, cs, ct, length, cue, mode)
        st = core.init_memory(model, compute_dtype)
        for c in range(cw):                                          # chunks 0 .. cw-1: the same in every placement
            pos, clean = jnp.arange(c * C, (c + 1) * C), jnp.asarray(ib[c * C:(c + 1) * C])
            if c == cs:
                x_src, W_before = keys(model, clean, pos, st), st.W
            st = commit(model, clean, pos, st)
            if c == cs:
                W_after = st.W
        pos = jnp.arange(cw * C, (cw + 1) * C)
        st_n = commit(model, jnp.asarray(ib[cw * C:(cw + 1) * C]), pos, st)        # beyond / none
        st_w = commit(model, jnp.asarray(iw[cw * C:(cw + 1) * C]), pos, st)        # the in-window copy
        st_src = st_n._replace(W=W_after)
        st_one = st_n._replace(W=one_w(W0, W_after, W_before))
        reset = lambda s: core.reset_memory(model, s)
        xs, sl = x_src[:, jnp.asarray(src)], jnp.asarray(sel)        # the write's keys at the read's slots
        pos = jnp.arange(ct * C, (ct + 1) * C)
        rb, tb = read_ids(ib, lb)
        r = {"off": ce(model, rb, pos, reset(st_n), tb), "on": ce(model, rb, pos, st_n, tb),
             "on_tx": ce_tx(model, rb, pos, st_n, tb, xs, sl),
             "src": ce(model, rb, pos, st_src, tb), "src_tx": ce_tx(model, rb, pos, st_src, tb, xs, sl),
             "one": ce(model, rb, pos, st_one, tb), "one_tx": ce_tx(model, rb, pos, st_one, tb, xs, sl)}
        rw, tw = read_ids(iw, lw)
        r["off_within"] = ce(model, rw, pos, reset(st_w), tw)
        r0, t0_ = read_ids(i0, l0)
        r["off_none"], r["on_none"] = ce(model, r0, pos, reset(st_n), t0_), ce(model, r0, pos, st_n, t0_)
        for k, v in enumerate(VARIANTS):
            sums[si, k] = float(np.asarray(r[v], np.float64)[sel].sum())
        counts[si] = sel.sum()
        log(f"[kt] sequence {si + 1}/{len(seqs)}: {time.time() - t0:.0f}s")
    return sums, counts


def summarise(sums: np.ndarray, counts: np.ndarray) -> dict:
    """Token-weighted means over the scored slots; each difference with its SE over sequences (the
    per-sequence mean differences, weighted by the sequences' slot counts)."""
    S = len(counts)
    col = {v: sums[:, k] for k, v in enumerate(VARIANTS)}
    w = counts / counts.sum()

    def diff(a, b, c=None, d=None):
        x = (col[a] - col[b]) / counts
        if c is not None:
            x = x - (col[c] - col[d]) / counts
        m = float((w * x).sum())
        se = float(np.sqrt((w ** 2 * (x - m) ** 2).sum() * S / max(S - 1, 1)))
        return m, se

    out = {"sequences": S, "slots": int(counts.sum()),
           "ce": {v: float(col[v].sum() / counts.sum()) for v in VARIANTS},
           "ceiling": diff("off", "off_within"), "generic": diff("off_none", "on_none")}
    for g in ("on", "on_tx", "src", "src_tx", "one", "one_tx"):
        out[f"gain_{g}"] = diff("off", g)
        out[f"recall_{g}"] = diff("off", g, "off_none", "on_none")
    out["transplant_effect"] = diff("on", "on_tx")
    return out


def report(s: dict, tag: str = "") -> None:
    ceil = s["ceiling"][0]
    pct = lambda m: f"{100 * m / ceil:6.1f} %" if ceil > 0 else "     - "
    print(f"[kt] {tag} {s['sequences']} sequences, {s['slots']} scored slots; CE: "
          + ", ".join(f"{v} {x:.4f}" for v, x in s["ce"].items()))
    print(f"[kt] {'quantity':<44s} {'nats':>8s} {'se':>7s} {'of ceiling':>10s}")
    rows = [("ceiling (attention's in-window copy)", "ceiling"), ("generic gain (nothing planted)", "generic")]
    for g, name in (("on", "memory"), ("on_tx", "memory, write's keys"), ("src", "chunks 0-1 only"),
                    ("src_tx", "chunks 0-1 only, write's keys"), ("one", "source write alone"),
                    ("one_tx", "source write alone, write's keys")):
        rows += [(f"gain: {name}", f"gain_{g}"), (f"recall: {name}", f"recall_{g}")]
    rows.append(("transplant effect (on - on_tx)", "transplant_effect"))
    for name, k in rows:
        m, se = s[k]
        print(f"[kt] {name:<44s} {m:8.4f} {se:7.4f} {pct(m):>10s}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    _cli.add_common(ap)
    ap.add_argument("--length", type=int, default=64)
    ap.add_argument("--cue", type=int, default=16)
    ap.add_argument("--mode", default="alt", choices=("alt", "block"))
    ap.add_argument("--plant_seed", type=int, default=PLANT_SEED)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    _cli.require_platform()
    import jax.numpy as jnp
    model, ws = _cli.load(a)
    seqs = _cli.sequences(a, model.cfg)
    sums, counts = run(model, [s for _, s in seqs], ws, a.length, a.cue, a.mode, a.plant_seed, jnp.dtype(a.dtype))
    s = summarise(sums, counts)
    meta = {"ckpt": a.ckpt, "inner_lr": ws.inner_lr, "length": a.length, "cue": a.cue, "mode": a.mode,
            "plant_seed": a.plant_seed, "dtype": a.dtype, "variants": list(VARIANTS)}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, sums=sums, counts=counts, sources=np.array([n for n, _ in seqs]),
             meta=np.array(json.dumps(meta)), summary=np.array(json.dumps(s)))
    report(s, a.tag or (a.ckpt or ""))
    print(f"[kt] saved {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
