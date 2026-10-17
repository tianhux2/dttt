#!/usr/bin/env python3
# coding=utf-8
"""
The standard masked-diffusion likelihood bound of a checkpoint, and its CE at every noise level, on held-out
sequences: the unweighted counterpart of the CART-weighted headline.

The CE of the masked tokens at fixed noise levels t (``Batch.outer_t``, no time weighting) with the write on and off
(the no-write twin; gdn: the memory kept empty), the checkpoint's own training loss otherwise (write, outer shift).
Every checkpoint scores the same sequences with the same masks (one uniform draw per token and sequence, so the masks
of a sequence are nested in t): differences between checkpoints are paired.  With i.i.d. masking at rate t the
continuous-time bound per token is the integral of CE(t) over [0, 1] (the 1/t weight and the t masked tokens cancel):
the mean of CE at the midpoints of ``--n_t`` bins.

    JAX_PLATFORMS=cuda python tools/elbo_eval.py --ckpt <step dir> --n_per_source 6 --out log/<dir>/elbo_<tag>.npz
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np


def measure(model, lc, step: int, ids: np.ndarray, keys, ts, batch: int = 4, scales=(1.0, 0.0), log=print):
    """``(ce [n_t, n_scales, n_seq], count [n_t, n_scales, n_seq])``: per sequence the CE sum over its masked tokens
    and their number, at every noise level of ``ts`` and inner-lr scale of ``scales`` (``train.at_scale``).
    ``keys [n_seq, 2]``: the per-sequence draws (the same in every call: paired)."""
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    from src.ttt.chunk_scan import Batch, loss_fn
    from src.ttt.train import at_scale
    n = len(ids)
    if n % batch:
        raise ValueError(f"batch {batch} must divide the {n} sequences")
    ce = np.zeros((len(ts), len(scales), n))
    cnt = np.zeros((len(ts), len(scales), n))
    t0 = time.time()
    for si, sc in enumerate(scales):
        lc_s = at_scale(lc, sc)
        f = eqx.filter_jit(lambda m, x, t, k: loss_fn(m, Batch(input_ids=x, outer_t=t), k, jnp.int32(step), lc_s)[1])
        for ti, t in enumerate(ts):
            for i in range(0, n, batch):
                st = f(model, jnp.asarray(ids[i:i + batch]), jnp.full((batch,), t, jnp.float32), keys[i:i + batch])
                m = np.asarray(st["mask_by_seq"], np.float64)
                ce[ti, si, i:i + batch] = np.asarray(st["loss_by_seq"], np.float64) * m
                cnt[ti, si, i:i + batch] = m
            log(f"[elbo] scale {sc:g}, t {t:.3f}: CE {ce[ti, si].sum() / cnt[ti, si].sum():.4f} "
                f"({time.time() - t0:.0f}s)")
    return ce, cnt


def summarise(ce, cnt, ts, scales, sources) -> dict:
    s = {"t": list(map(float, ts)), "scales": list(map(float, scales)), "sequences": int(ce.shape[-1])}
    per_t = ce.sum(-1) / cnt.sum(-1)                                   # [n_t, n_scales]
    s["ce_by_t"] = per_t.tolist()
    s["nelbo"] = per_t.mean(0).tolist()                                # the midpoint rule over [0, 1]
    s["by_source"] = {}
    for src in sorted(set(sources)):
        sel = np.array([x == src for x in sources])
        s["by_source"][src] = (ce[..., sel].sum(-1) / cnt[..., sel].sum(-1)).mean(0).tolist()
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model_config", default=None)
    ap.add_argument("--sources", default=None, help="default: the checkpoint's training sources and pg19_test")
    ap.add_argument("--n_per_source", type=int, default=6)
    ap.add_argument("--n_t", type=int, default=10)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if not os.environ.get("JAX_PLATFORMS"):
        sys.exit("set JAX_PLATFORMS (cuda on a GPU node)")
    import jax
    import jax.numpy as jnp
    from src.ttt.config import TrainConfig
    from src.ttt.data import Sampler, parse_sources
    from src.ttt.generate import load_model
    from src.ttt.model import cast_floating
    from src.ttt.train import loss_config

    tc = TrainConfig.load(os.path.join(a.ckpt, "train_config.json"))
    with open(os.path.join(a.ckpt, "state.json")) as f:
        step = int(json.load(f)["step"])
    model = cast_floating(load_model(a.ckpt, a.model_config), jnp.bfloat16)   # the evaluation's compute copy
    lc = dataclasses.replace(loss_config(tc), per_sequence=True, per_token=False, mtp_weight=0.0, aux_causal_weight=0.0,
                             window_drop=0.0, self_prior=0.0, outer_views=1, time_reweighting="none", outer_t=-1.0)
    # default: the checkpoint's training sources (under its data_root) and PG-19 test
    pg19 = "pg19_test_a4dfccb3b3ac65f9" if tc.data_root else "bucket/data/tokens/pg19_test_a4dfccb3b3ac65f9"
    srcs = parse_sources(a.sources) if a.sources else parse_sources(f"{tc.sources},pg19_test={pg19}:1.0", tc.data_root)
    smp = Sampler(srcs, tc.seq_len, a.n_per_source, 42, holdout_tokens=tc.holdout or 4_194_304,
                  eos=model.cfg.eos_token_id)
    rows = [(s.name, np.asarray(r, np.int32)) for si, s in enumerate(srcs) for r in smp.eval_batch(si, 0)]
    ids = np.stack([r for _, r in rows])
    keys = jax.vmap(lambda j: jax.random.split(jax.random.fold_in(jax.random.PRNGKey(a.seed), j))[1])(
        jnp.arange(len(ids)))
    ts = (np.arange(a.n_t) + 0.5) / a.n_t
    scales = (1.0,) if tc.no_write else (1.0, 0.0)
    print(f"[elbo] {a.ckpt}: step {step}, {len(ids)} sequences of {tc.seq_len}, memory {model.cfg.memory}, "
          f"t = {', '.join(f'{t:.2f}' for t in ts)}", flush=True)
    ce, cnt = measure(model, lc, step, ids, keys, ts, a.batch, scales)
    s = summarise(ce, cnt, ts, scales, [n for n, _ in rows])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, ce=ce, count=cnt, sources=np.array([n for n, _ in rows]), summary=np.array(json.dumps(s)),
             meta=np.array(json.dumps({"ckpt": a.ckpt, "step": step, "seq_len": tc.seq_len, "seed": a.seed})))
    print(f"[elbo] {a.tag or a.ckpt}: NELBO per token (midpoint over t) " +
          ", ".join(f"scale {sc:g} {v:.4f}" for sc, v in zip(scales, s["nelbo"])) +
          " | CE(t) write on: " + " ".join(f"{v[0]:.3f}" for v in s["ce_by_t"]), flush=True)
    print(f"[elbo] saved {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
