# coding=utf-8
"""Multi-view inner writes: stratified mask ratios, V sequential inner SGD steps per chunk,
CART-weighted inner loss.

    JAX_PLATFORMS=cpu .venv/bin/python tests/test_inner_views.py
"""
from __future__ import annotations

import dataclasses
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")
os.environ.setdefault("TTT_ATTENTION", "xla")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import jax
import jax.numpy as jnp
import numpy as np

from src.ttt.chunk_scan import Batch, LossConfig, loss_fn
from src.ttt.config import ModelConfig
from src.ttt.model import Model
from src.ttt.objective import make_inner_mask

CFG = ModelConfig(
    vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
    num_attention_heads=4, num_key_value_heads=2, head_dim=8, suffix_len=1,
    chunk_size=8, window_chunks=1, mask_token_id=120, pad_token_id=0, eos_token_id=0,
)
B, T = 4, 64
BASE = LossConfig(cart_window=32, inner_lr=0.005, fused_ce_block=0, compute_dtype="float32", attn_chunk_batch=2)


def main():
    failures = []
    # 1. stratified views: V independent draws whose ratios fall in distinct strata of (lo, hi)
    V, C = 4, 4096
    f = make_inner_mask(-1.0, (0.2, 1.0), 120, 0, views=V, view_mode="strata")
    chunk = jnp.full((C,), 5, jnp.int32)
    x, m = f(chunk, jax.random.PRNGKey(0))
    rat = sorted(float(v) for v in m.mean(axis=1))
    centres = [0.2 + 0.8 * (k + 0.5) / V for k in range(V)]
    ok = x.shape == (V, C) and m.shape == (V, C) and all(abs(r - c) < 0.8 / V / 2 + 0.03 for r, c in zip(rat, centres)) \
        and bool(jnp.all((x == 120) == m))
    failures += [] if ok else [f"strata ratios {rat} vs strata centres {centres}"]
    print(f"[{'PASS' if ok else 'FAIL'}] stratified views: ratios {[round(r, 2) for r in rat]} in the 4 strata of (0.2, 1)")

    model = Model(CFG, jax.random.PRNGKey(0))
    batch = Batch(input_ids=jax.random.randint(jax.random.PRNGKey(1), (B, T), 1, 100))
    key, step = jax.random.PRNGKey(2), jnp.int32(5)
    run = lambda **kw: loss_fn(model, batch, key, step, dataclasses.replace(BASE, **kw))

    # 2. sequential V steps at lr/V equal one averaged step to first order (tiny lr), and
    #    differ at a large lr (the later views see the updated W).
    # (clipping off: with it binding, "normalised mean" != "mean of normalised" by design)
    par_s, _ = run(inner_views=V, inner_view_mode="strata", inner_lr=1e-4, inner_clip=0.0)
    seq_s, st_s = run(inner_views=V, inner_view_mode="strata", inner_views_sequential=True, inner_lr=1e-4, inner_clip=0.0)
    par_l, _ = run(inner_views=V, inner_view_mode="strata", inner_lr=1.0)
    seq_l, _ = run(inner_views=V, inner_view_mode="strata", inner_views_sequential=True, inner_lr=1.0)
    d_small, d_large = abs(float(par_s - seq_s)), abs(float(par_l - seq_l))
    ok = d_small < 1e-5 and d_large > 1e-5 and np.isfinite(float(seq_l))
    failures += [] if ok else [f"sequential vs parallel: small-lr diff {d_small:.2e}, large-lr diff {d_large:.2e}"]
    print(f"[{'PASS' if ok else 'FAIL'}] sequential views == averaged step to first order (diff {d_small:.1e} at lr 1e-4, {d_large:.1e} at lr 1)")

    # 3. the meta-gradient flows through all V sequential steps (finite, nonzero)
    g = jax.grad(lambda mdl: loss_fn(mdl, batch, key, step, dataclasses.replace(
        BASE, inner_views=V, inner_view_mode="strata", inner_views_sequential=True, inner_lr=1.0))[0])(model)
    leaves = [np.asarray(x) for x in jax.tree_util.tree_leaves(jax.tree_util.tree_map(
        lambda x: x if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating) else None, g))]
    ok = all(np.all(np.isfinite(l)) for l in leaves) and any(np.any(l != 0) for l in leaves)
    failures += [] if ok else ["meta-gradient through sequential views not finite / all zero"]
    print(f"[{'PASS' if ok else 'FAIL'}] meta-gradient through {V} sequential inner steps is finite")

    # 4. CART-weighted inner loss: runs, finite, changes the write (outer loss differs from unweighted)
    l0, _ = run(inner_views=V, inner_view_mode="strata", inner_views_sequential=True, inner_lr=1.0)
    l1, st1 = run(inner_views=V, inner_view_mode="strata", inner_views_sequential=True, inner_lr=1.0, inner_cart=True)
    ok = np.isfinite(float(l1)) and abs(float(l1 - l0)) > 1e-7 and np.isfinite(float(st1["inner_loss"]))
    failures += [] if ok else [f"inner_cart: {float(l0)} vs {float(l1)}"]
    print(f"[{'PASS' if ok else 'FAIL'}] inner_cart weights change the write (outer {float(l0):.5f} -> {float(l1):.5f})")

    # 5. one view, complement mode and the default path are untouched
    a, _ = run(); b, _ = run(inner_view_mode="complement")
    ok = float(a) == float(b)
    failures += [] if ok else ["default single-view path changed"]
    print(f"[{'PASS' if ok else 'FAIL'}] single-view default unchanged")

    print("ALL PASS" if not failures else "FAILURES: " + "; ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
