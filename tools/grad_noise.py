#!/usr/bin/env python3
# coding=utf-8
"""
How much of the training gradient's noise comes from the mask draw, and how much from the data?  Decides whether
several noisy reads of each chunk per step (``--outer_views``) can pay for themselves.

At a checkpoint, with its own training loss (``train.loss_config`` of its ``train_config.json``: write, CART,
second order, self-prior ...), ``--n_seq`` training sequences (the run's own sampler, doc windows and mix, a step
far past the run) each get ``--n_masks`` gradients of their loss, one per key (the key draws the noise level, the
mask and the self-prior read).  The loss is the sequence's CE SUM over its scored tokens (with V views their mean):
a training batch divides by the whole batch's count, so this is one sequence's share of the batch gradient (the
per-sequence mean would overweight a draw with few masked tokens by 1 / its count).  Per parameter group (embedding, prefix, suffix, the fast-weight template, the rest)
and in total, with g_dm the gradient of sequence d under key m:

    mask    W = mean_d [ sum_m |g_dm - g_d.|^2 / (M - 1) ]          the noise of the draw, per sequence
    data    D = var_d(g_d.) - W / M                                   the noise of the sequence
    signal  |G|^2 = |g_..|^2 - var_d(g_d.) / n_seq                     (unbiased)
    share   W / (W + D): the part of one sequence's gradient noise that V views would cut to W / V
    B_simple (W + D) / |G|^2, in sequences (McCandlish et al. 2018)

V views cost (measured: ``--views V`` also times the gradient with V views) and cut the noise per sequence to
D + W / V: they beat a batch that is as much larger only when 1 / (1 - share (1 - 1/V)) exceeds the cost ratio.
``--views V`` also measures W with V views directly (the views share the write; independent views give W / V).
``--self_prior``: override the checkpoint's (0: the noise of the mask and noise level alone).

    JAX_PLATFORMS=cuda python tools/grad_noise.py --ckpt <step dir> --n_seq 16 --n_masks 4 --views 2 \\
        --out log/<dir>/grad_noise_<tag>.json
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

GROUPS = ("embed", "prefix", "suffix", "fast_weight", "other")


def group_sq(g) -> dict:
    """Squared norm of a gradient per parameter group, and in total (``train.make_train_step``'s groups)."""
    import equinox as eqx
    import optax
    sel = {"embed": lambda t: (t.embed,), "prefix": lambda t: (t.prefix,), "suffix": lambda t: (t.suffix,),
           "fast_weight": lambda t: (t.fast_weight,)}
    out = {k: optax.global_norm(eqx.filter(f(g), eqx.is_array)) ** 2 for k, f in sel.items()}
    rest = eqx.tree_at(lambda t: (t.embed, t.prefix, t.suffix, t.fast_weight), g, replace_fn=lambda _: None)
    out["other"] = optax.global_norm(eqx.filter(rest, eqx.is_array)) ** 2
    out["all"] = sum(out[k] for k in GROUPS)
    return out


def decompose(sq_each, sq_mean_d, sq_mean_all, M: int, D: int) -> dict:
    """The estimates above from squared norms (each a dict of group -> value): ``sq_each [D][M]`` |g_dm|^2,
    ``sq_mean_d [D]`` |g_d.|^2, ``sq_mean_all`` |g_..|^2."""
    out = {}
    for k in sq_mean_all:
        within = np.mean([(sum(sq_each[d][m][k] for m in range(M)) - M * sq_mean_d[d][k]) / (M - 1) for d in range(D)])
        var_mean = (sum(sq_mean_d[d][k] for d in range(D)) - D * sq_mean_all[k]) / (D - 1)
        data = var_mean - within / M
        signal = sq_mean_all[k] - var_mean / D
        tot = within + data
        out[k] = {"mask": float(within), "data": float(data), "signal": float(signal),
                  "share": float(within / tot) if tot > 0 else float("nan"),
                  "b_simple": float(tot / signal) if signal > 0 else float("nan")}
    return out


def measure(model, ids, lc, step: int, n_masks: int, views: int = 0, seed: int = 7, cast_dtype=None, log=print):
    """``(results {V: decompose(...)}, seconds per gradient {V: s})`` for the sequences ``ids [D, T]`` under the loss
    ``lc`` (``V`` = 1, and ``views`` if > 1): ``n_masks`` keys per sequence.  The model is cast to ``cast_dtype``
    (default bfloat16, the training's compute copy) inside the differentiated function, as the training step does."""
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    from src.ttt.chunk_scan import Batch, loss_fn
    from src.ttt.model import cast_floating
    cast_dtype = jnp.bfloat16 if cast_dtype is None else cast_dtype
    D, M = int(ids.shape[0]), n_masks
    params, static = eqx.partition(model, eqx.is_inexact_array)

    def grad_fn(lcv):
        def objective(p, x, key):
            # the sequence's UNNORMALISED loss (its CE sum over its scored tokens; V views: their mean): a batch
            # normalises by the whole batch's count, so this is what one sequence adds to the batch gradient.
            # The per-sequence mean would weigh a draw with few masked tokens (low noise level) up by 1 / count.
            m = cast_floating(eqx.combine(p, static), cast_dtype)
            loss, st = loss_fn(m, Batch(input_ids=x[None]), key, jnp.int32(step), lcv)
            return loss * jax.lax.stop_gradient(st["mask_count"]) / lcv.outer_views

        def both(p, x, key):
            g = jax.grad(objective)(p, x, key)
            return g, group_sq(g)
        return eqx.filter_jit(both)

    tree_add = eqx.filter_jit(lambda acc, g, w: jax.tree_util.tree_map(lambda u, v: u + w * v, acc, g))
    sq_tree = eqx.filter_jit(group_sq)
    to_np = lambda d: {k: float(v) for k, v in d.items()}
    base = jax.random.PRNGKey(seed)
    results, timing = {}, {}
    for V in [1] + ([views] if views > 1 else []):
        gf = grad_fn(dataclasses.replace(lc, outer_views=V))
        zeros = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x, jnp.float32), params)
        acc_all = zeros
        sq_each, sq_mean_d, times = [], [], []
        for d in range(D):
            acc_d, row = zeros, []
            x = jnp.asarray(ids[d])
            for m in range(M):
                t0 = time.time()
                g, sq = gf(params, x, jax.random.fold_in(base, 1000 * d + m))
                sq = to_np(sq)                                         # blocks until the gradient is done
                times.append(time.time() - t0)
                row.append(sq)
                acc_d = tree_add(acc_d, g, 1.0 / M)
            sq_each.append(row)
            sq_mean_d.append(to_np(sq_tree(acc_d)))
            acc_all = tree_add(acc_all, acc_d, 1.0 / D)
            log(f"[noise] V {V}: sequence {d + 1}/{D}, {np.mean(times[1:] or times):.2f} s per gradient")
        results[V] = decompose(sq_each, sq_mean_d, to_np(sq_tree(acc_all)), M, D)
        timing[V] = float(np.median(times[1:] or times))               # the first call compiles
    return results, timing


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n_seq", type=int, default=16)
    ap.add_argument("--n_masks", type=int, default=4)
    ap.add_argument("--data_step", type=int, default=100_000, help="the sampler step the sequences come from")
    ap.add_argument("--self_prior", type=float, default=None, help="default: the checkpoint's")
    ap.add_argument("--views", type=int, default=0, help="> 1: also measure and time V noisy reads per chunk")
    ap.add_argument("--sources", default=None, help="default: the checkpoint's training sources")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if not os.environ.get("JAX_PLATFORMS"):
        sys.exit("set JAX_PLATFORMS (cuda on a GPU node)")
    print(f"[noise] {a.ckpt}", flush=True)
    from src.ttt.config import TrainConfig
    from src.ttt.data import Sampler, parse_sources, windows_path
    from src.ttt.generate import load_model
    from src.ttt.train import loss_config

    tc = TrainConfig.load(os.path.join(a.ckpt, "train_config.json"))
    with open(os.path.join(a.ckpt, "state.json")) as f:
        step = int(json.load(f)["step"])
    model = load_model(a.ckpt)
    lc = loss_config(tc)
    if a.self_prior is not None:
        lc = dataclasses.replace(lc, self_prior=a.self_prior)
    sources = parse_sources(a.sources) if a.sources else parse_sources(tc.sources, tc.data_root)
    windows = None
    if tc.doc_windows:
        paths = [windows_path(s.path, tc.seq_len, tc.holdout) for s in sources]
        windows = [np.load(p) for p in paths] if all(os.path.exists(p) for p in paths) else None
    smp = Sampler(sources, tc.seq_len, a.n_seq, tc.seed, holdout_tokens=tc.holdout, eos=model.cfg.eos_token_id,
                  windows=windows)
    ids = smp.batch(a.data_step)                                       # [D, T]
    D, M = a.n_seq, a.n_masks
    results, timing = measure(model, ids, lc, step, M, a.views, a.seed)
    out = {"ckpt": a.ckpt, "step": step, "self_prior": lc.self_prior, "n_seq": D, "n_masks": M,
           "data_step": a.data_step, "seconds_per_gradient": timing, "by_views": {str(k): v for k, v in results.items()}}
    if a.views > 1:
        V = a.views
        r1, rv = results[1]["all"], results[V]["all"]
        cost = timing[V] / timing[1]
        out["views_check"] = {"mask_ratio_measured": rv["mask"] / r1["mask"], "mask_ratio_independent": 1.0 / V,
                              "cost_ratio": cost,
                              "effective_batch_gain": 1.0 / (1.0 - r1["share"] * (1.0 - 1.0 / V)),
                              "effective_batch_gain_measured": (r1["mask"] + r1["data"]) / (rv["mask"] + r1["data"])}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    for V, r in results.items():
        print(f"[noise] V {V} ({timing[V]:.2f} s/gradient): " + "; ".join(
            f"{k} share {100 * x['share']:.0f}% B_simple {x['b_simple']:.1f}" for k, x in r.items()), flush=True)
    if "views_check" in out:
        print("[noise] views: " + json.dumps(out["views_check"]), flush=True)
    print(f"[noise] saved {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
