#!/usr/bin/env python3
# coding=utf-8
"""
Recall over distance: how much of a one-off span does the memory still return when its copy is read d chunks after
the span's chunk, with every chunk in between written too?

The planted-recall probe (``probes.planted_probe_batch``) with the copy in chunk 1 + d for every d of
``--distances``: the same draws at every distance - the span of chunk 1, its offsets within the chunks, the scored
slots (paired; a draw that meets an EOS is redrawn, which can differ between targets).  The chunks before a target are natural text whatever the target, so each sequence is written
once, chunk by chunk, with the checkpoint's own write (or the normalised one, ``--norm``, for a checkpoint trained
with the plain write); every target chunk is read before it is written, then its natural version is written and
the pass goes on.  Per distance, in % of the ceiling (attention's copy: the span also in the chunk before the
target, memory off), recall = the copy's gain - the same memory's gain on the unplanted text at the same slots:

    own       the read's keys: the memory as it is
    write     the write's keys: the memory's contribution at the copy's slots taken at the source slots' inputs
    ctx       the read chunk's own clean causal inputs (the copy unmasked, causal attention, the read's history)
    isolated  the read's keys against W_0 + the source chunk's own write (the windows as they are): what the
              source chunk wrote, without the other chunks' writes - own / isolated is how much of it survives them
    den_iso   the memory as it is, but every fast branch normalised by the RMS of the isolated memory's output
              (``core.read_forward(den_W=...)``): the later writes' cost through the shared RMS denominator removed
    num_iso   the isolated memory normalised by the full memory's RMS: only the denominator's cost kept
              (den_iso ~ isolated: the later writes dilute the source's share through the normalisation;
              den_iso ~ own: they crowd it out in the numerator - interference)

the memory's drift |W - W_0| / |W_0| at the read, and on an i.i.d.-masked read of the target chunk's natural text
(ratio 0.5, the same draws in every run: paired across checkpoints and writes) the memory's general gain, also with
the den_iso normalisation (whether that read-time change keeps the model working).

    JAX_PLATFORMS=cuda python -m tools.j_lens.memory_distance --ckpt <step dir> --sources <name=dir:1.0,...> \\
        --seq_len 32768 --distances 5,9,17,30 --n_per_source 8 [--norm 1:64] \\
        --out <npz>
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
from tools.j_lens.key_transplant import PLANT_SEED, match

KEYS = ("own", "write", "ctx", "isolated", "den_iso", "num_iso")


def run(model, seqs, ws, distances, length: int, cue: int, mode: str, seed: int, compute_dtype, log=print, norm=None):
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    from src.ttt import probes as PR
    from src.ttt.inner_loop import WriteConfig
    from src.ttt.model import cast_floating
    from tools.j_lens import core
    cfg = model.cfg
    C, Wn = cfg.chunk_size, cfg.window_chunks
    if not ws.outer_shift or ws.inner_task != "causal" or model.gdn is not None:
        raise ValueError("memory distance: the TTT memory's causal write with the outer shift")
    if norm is not None:
        if not ws.wc.plain:
            raise ValueError("--norm replaces the checkpoint's write: a checkpoint trained with the plain one")
        ws = ws._replace(wc=WriteConfig(norm_alpha=float(norm[0]), norm_cap=float(norm[1])))
    n_chunks = len(seqs[0]) // C
    targets = sorted({1 + int(d) for d in distances})
    for t in targets:
        PR.planted_chunks(n_chunks, Wn, t)                            # refuses targets too close or too far
    f32 = jnp.float32
    key0 = jax.random.PRNGKey(0)                                      # the causal write draws nothing
    commit = eqx.filter_jit(lambda m, ids, pos, st: core.commit(m, ids, pos, st, key0, ws, compute_dtype))
    keys = eqx.filter_jit(lambda m, ids, pos, st: core.write_keys(m, ids, pos, st, ws, compute_dtype))

    def _ce(m, ids, pos, st, tgt, tx, den=None):
        x, clean = ids                                            # the masked chunk and its clean tokens
        own = core.clean_own(m, clean, pos, st, compute_dtype) if m.cfg.read_block else None
        lg = core.read_forward(m, x, pos, st, None, True, compute_dtype, transplant=tx, own_kv=own,
                               den_W=den).logits.astype(f32)
        return -jnp.take_along_axis(jax.nn.log_softmax(lg, axis=-1), tgt[:, None], axis=-1)[:, 0]
    ce = eqx.filter_jit(lambda m, ids, pos, st, tgt: _ce(m, ids, pos, st, tgt, None))
    ce_tx = eqx.filter_jit(lambda m, ids, pos, st, tgt, xs, sel: _ce(m, ids, pos, st, tgt, core.Transplant(xs, sel)))
    ce_den = eqx.filter_jit(lambda m, ids, pos, st, tgt, den: _ce(m, ids, pos, st, tgt, None, den))
    W0 = cast_floating(model.fast_weight, f32)
    lv = lambda t: jax.tree_util.tree_leaves(eqx.filter(t, eqx.is_inexact_array))
    drift = eqx.filter_jit(lambda W: jnp.sqrt(sum(jnp.sum(jnp.square(a - b)) for a, b in zip(lv(W), lv(W0)))
                                              / sum(jnp.sum(jnp.square(b)) for b in lv(W0))))
    isolated = eqx.filter_jit(lambda w0, after, before: jax.tree_util.tree_map(
        lambda x, y, z: x + (y - z) if eqx.is_inexact_array(x) else x, w0, after, before))

    def masked(ids, lm, t):
        """``((masked chunk t, its clean tokens), targets)``: the clean tokens give a read_block model its
        sub-block read's clean K/V."""
        chunk, mk = ids[t * C:(t + 1) * C], lm[t * C:(t + 1) * C]
        return ((jnp.asarray(np.where(mk, cfg.mask_token_id, chunk)), jnp.asarray(chunk)),
                jnp.asarray(np.concatenate([chunk[1:], chunk[-1:]])))    # slot j predicts token j + 1

    n_seq, nd = len(seqs), len(targets)
    out = {"gain_copy": np.zeros((n_seq, nd, len(KEYS))), "gain_none": np.zeros((n_seq, nd, len(KEYS))),
           "ceiling": np.zeros((n_seq, nd)), "count": np.zeros((n_seq, nd)), "drift": np.zeros((n_seq, nd)),
           "gen_ce": np.zeros((n_seq, nd)), "gen_off": np.zeros((n_seq, nd)), "count_gen": np.zeros((n_seq, nd)),
           "gen_ce_den": np.zeros((n_seq, nd))}
    t0 = time.time()
    for si, seq in enumerate(seqs):
        seq = np.asarray(seq, np.int32)
        pl = {t: {p: PR.planted_probe_batch(seq[None], [si], seed, C, Wn, length, cue, p, mode, eos=cfg.eos_token_id,
                                            target=t) for p in PR.PLANT_PLACEMENTS} for t in targets}
        st = core.init_memory(model, compute_dtype)
        for c in range(targets[-1]):
            pos, clean = jnp.arange(c * C, (c + 1) * C), jnp.asarray(seq[c * C:(c + 1) * C])
            if c == 1:                                                # the source chunk: its keys and its own write
                x_src, W_before = keys(model, clean, pos, st), st.W
            t = c + 1
            if t in pl:                                               # chunk c is target t's in-window chunk
                iw = pl[t]["within"][0][0]
                st_w = commit(model, jnp.asarray(iw[c * C:(c + 1) * C]), pos, st)
            st = commit(model, clean, pos, st)
            if c == 1:
                W_iso = isolated(W0, st.W, W_before)
            if t not in pl:
                continue
            # read target t: the memory after chunks 0 .. t-1 (natural text, as in the planted sequence)
            di = targets.index(t)
            (ib, lb), (iw, lw), (i0, l0) = [(pl[t][p][0][0], pl[t][p][1][0]) for p in ("beyond", "within", "none")]
            assert np.array_equal(ib[:t * C], seq[:t * C]) and np.array_equal(i0, seq) and np.array_equal(lb, l0)
            sel, src = match(ib, lb, C, 1, t, length, cue, mode)
            pos_r = jnp.arange(t * C, (t + 1) * C)
            rb, tb = masked(ib, lb, t)
            r0, t0_ = masked(i0, l0, t)
            rw, tw = masked(iw, lw, t)
            reset = lambda s_: core.reset_memory(model, s_)
            f64 = lambda x: np.asarray(x, np.float64)
            off_b, off_0 = f64(ce(model, rb, pos_r, reset(st), tb)), f64(ce(model, r0, pos_r, reset(st), t0_))
            out["ceiling"][si, di] = float((off_b - f64(ce(model, rw, pos_r, reset(st_w), tw)))[sel].sum())
            out["count"][si, di] = sel.sum()
            xs, selj = x_src[:, jnp.asarray(src)], jnp.asarray(sel)
            st_iso = st._replace(W=W_iso)
            for field, ids_r, tgt, off, chunk_ids in (("gain_copy", rb, tb, off_b, ib), ("gain_none", r0, t0_, off_0, i0)):
                ctx = keys(model, jnp.asarray(chunk_ids[t * C:(t + 1) * C]), pos_r, st)
                reads = {"own": ce(model, ids_r, pos_r, st, tgt), "isolated": ce(model, ids_r, pos_r, st_iso, tgt),
                         "write": ce_tx(model, ids_r, pos_r, st, tgt, xs, selj),
                         "ctx": ce_tx(model, ids_r, pos_r, st, tgt, ctx, selj),
                         "den_iso": ce_den(model, ids_r, pos_r, st, tgt, W_iso),
                         "num_iso": ce_den(model, ids_r, pos_r, st_iso, tgt, st.W)}
                for ki, k in enumerate(KEYS):
                    out[field][si, di, ki] = float((off - f64(reads[k]))[sel].sum())
            out["drift"][si, di] = float(drift(st.W))
            nat = jnp.asarray(seq[t * C:(t + 1) * C])                 # the general read: i.i.d. mask, paired draws
            rg, _, vg = core.read_mask(nat, 0.5, jax.random.PRNGKey(1000 + 97 * si + t), cfg.mask_token_id, True)
            tg, vg = jnp.concatenate([nat[1:], nat[-1:]]), np.asarray(vg)
            out["gen_ce"][si, di] = float(f64(ce(model, (rg, nat), pos_r, st, tg))[vg].sum())
            out["gen_off"][si, di] = float(f64(ce(model, (rg, nat), pos_r, reset(st), tg))[vg].sum())
            out["gen_ce_den"][si, di] = float(f64(ce_den(model, (rg, nat), pos_r, st, tg, W_iso))[vg].sum())
            out["count_gen"][si, di] = vg.sum()
        log(f"[md] sequence {si + 1}/{n_seq}: {time.time() - t0:.0f}s")
    return out, targets


def summarise(o: dict, targets, chunk: int) -> dict:
    n, S = o["count"], o["count"].shape[0]

    def wm(x, cnt):
        w = cnt / cnt.sum()
        m = float((w * x).sum())
        return m, float(np.sqrt((w ** 2 * (x - m) ** 2).sum() * S / max(S - 1, 1)))
    s = {"sequences": S, "targets": list(targets), "distances": [t - 1 for t in targets], "chunk": chunk, "rows": []}
    for di, t in enumerate(targets):
        cnt = n[:, di]
        ceil = wm(o["ceiling"][:, di] / cnt, cnt)
        rec = {k: wm((o["gain_copy"][:, di, ki] - o["gain_none"][:, di, ki]) / cnt, cnt) for ki, k in enumerate(KEYS)}
        own, iso = rec["own"][0], rec["isolated"][0]
        cg = o["count_gen"][:, di]
        s["rows"].append({"distance": t - 1, "tokens": (t - 1) * chunk, "ceiling": ceil, "slots": int(cnt.sum()),
                          **{f"recall_{k}": v for k, v in rec.items()},
                          "survival": own / iso if abs(iso) > 1e-9 else float("nan"),
                          "drift": float(o["drift"][:, di].mean()),
                          "general_gain": wm((o["gen_off"][:, di] - o["gen_ce"][:, di]) / cg, cg),
                          "general_ce": float(o["gen_ce"][:, di].sum() / cg.sum()),
                          **({"general_den_cost": wm((o["gen_ce_den"][:, di] - o["gen_ce"][:, di]) / cg, cg)}
                             if "gen_ce_den" in o else {})})
    return s


def report(s: dict, tag: str = "") -> None:
    print(f"[md] {tag}: {s['sequences']} sequences; the copy read d chunks ({s['chunk']} tokens each) after the span's chunk")
    print(f"[md] {'d':>4s} {'~tokens':>8s} {'ceiling':>8s} {'recall own':>11s} {'write keys':>11s} {'ctx keys':>9s} "
          f"{'isolated':>9s} {'own/iso':>8s} {'drift':>7s} {'general gain':>13s}")
    for r in s["rows"]:
        ceil = r["ceiling"][0]
        pct = lambda m: f"{100 * m / ceil:5.1f}%" if ceil > 0.05 else "   - "
        se = r["recall_own"][1]
        print(f"[md] {r['distance']:4d} {r['tokens']:8d} {ceil:8.4f} {pct(r['recall_own'][0]):>7s}±{100 * se / max(ceil, 1e-9):4.1f} "
              f"{pct(r['recall_write'][0]):>11s} {pct(r['recall_ctx'][0]):>9s} {pct(r['recall_isolated'][0]):>9s} "
              f"{r['survival']:8.2f} {r['drift']:7.4f} {r['general_gain'][0]:8.4f}±{r['general_gain'][1]:.4f}")
        if "recall_den_iso" in r:
            print(f"[md]      survival split: den_iso {pct(r['recall_den_iso'][0])}±{100 * r['recall_den_iso'][1] / max(ceil, 1e-9):.1f}"
                  f"  num_iso {pct(r['recall_num_iso'][0])}±{100 * r['recall_num_iso'][1] / max(ceil, 1e-9):.1f}"
                  f"  general CE cost of den_iso {r['general_den_cost'][0]:+.4f}±{r['general_den_cost'][1]:.4f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    _cli.add_common(ap)
    ap.add_argument("--distances", default="5,9,17,30", help="chunks between the span's chunk (1) and the copy's")
    ap.add_argument("--norm", default="", help="alpha:cap - write with the normalised step instead (a plain checkpoint)")
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
    norm = tuple(float(v) for v in a.norm.split(":")) if a.norm else None
    distances = [int(x) for x in a.distances.split(",") if x]
    o, targets = run(model, [x for _, x in seqs], ws, distances, a.length, a.cue, a.mode, a.plant_seed,
                     jnp.dtype(a.dtype), norm=norm)
    s = summarise(o, targets, model.cfg.chunk_size)
    meta = {"ckpt": a.ckpt, "inner_lr": ws.inner_lr, "norm": norm, "seq_len": a.seq_len,
            "distances": distances, "batch_index": a.batch_index,
            "length": a.length, "cue": a.cue, "mode": a.mode, "plant_seed": a.plant_seed, "dtype": a.dtype}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, sources=np.array([n for n, _ in seqs]), meta=np.array(json.dumps(meta)),
             summary=np.array(json.dumps(s)), targets=np.array(targets), **o)
    report(s, a.tag or (a.ckpt or ""))
    print(f"[md] saved {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
