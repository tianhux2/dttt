# coding=utf-8
"""Several noisy reads of each chunk per step (``LossConfig.outer_views``) on a tiny random model:

1. the loss with V views equals the scored-token average of V single-view runs, each fixing one view's mask (noise
   level 1 with that mask as the loss mask), for V = 2 and 3 - every view reads the same window and memory, and the
   write is shared;
2. the gradient with 2 views equals the same weighted average of the single-view gradients (the meta-gradient
   through the shared write sums both reads);
3. CART weighting runs and is finite; the evaluation-only scores and the incompatible options are refused.

    JAX_PLATFORMS=cpu .venv/bin/python tests/test_outer_views.py
"""
from __future__ import annotations

import dataclasses
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")
os.environ.setdefault("TTT_ATTENTION", "xla")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import equinox as eqx
import jax
import jax.numpy as jnp

from src.ttt.chunk_scan import Batch, LossConfig, loss_fn
from src.ttt.config import ModelConfig, TrainConfig
from src.ttt.model import Model
from src.ttt.objective import q_sample

CFG = ModelConfig(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
                  num_key_value_heads=2, head_dim=8, suffix_len=2, chunk_size=8, window_chunks=2,
                  mask_token_id=120, pad_token_id=0, eos_token_id=0)
failures = []


def report(name, ok, detail=""):
    failures.append(name) if not ok else None
    print(f"[{'PASS' if ok else 'FAIL'}] {name:84s} {detail}")


def leaves(t):
    return jax.tree_util.tree_leaves(eqx.filter(t, eqx.is_inexact_array))


def main():
    model = Model(CFG, jax.random.PRNGKey(0))
    ids = jax.random.randint(jax.random.PRNGKey(5), (2, 4 * CFG.chunk_size), 1, 100)
    B = ids.shape[0]
    key, step = jax.random.PRNGKey(11), jnp.int32(5)
    lc = LossConfig(time_reweighting="none", inner_lr=0.5, inner_task="causal", outer_shift=True, fused_ce_block=0,
                    compute_dtype="float32", attn_chunk_batch=2, outer_t=0.6)
    _, key_mask, _ = jax.random.split(key, 3)

    def view_mask(v):                       # the masked positions of view v (view 0: the single-view draw)
        k = key_mask if v == 0 else jax.random.fold_in(key_mask, v)
        return q_sample(k, ids, None, CFG.mask_token_id, t=jnp.full((B,), 0.6))[2]

    def single(m):                          # one view, its mask fixed: noise level 1 inside the loss mask
        return loss_fn(model, Batch(input_ids=ids, loss_mask=m, outer_t=jnp.ones((B,), jnp.float32)), key, step,
                       dataclasses.replace(lc, outer_t=-1.0))

    # 1. V views == the scored-token average of V single-view runs
    for VO in (2, 3):
        loss_v, st_v = loss_fn(model, Batch(input_ids=ids), key, step, dataclasses.replace(lc, outer_views=VO))
        parts = [single(view_mask(v)) for v in range(VO)]
        n = [float(st["mask_count"]) for _, st in parts]
        want = sum(float(l) * k for (l, _), k in zip(parts, n)) / sum(n)
        masks_differ = all(bool(jnp.any(view_mask(0) != view_mask(v))) for v in range(1, VO))
        report(f"{VO} views == the scored-token average of {VO} single-view runs",
               abs(float(loss_v) - want) < 2e-5 and abs(float(st_v["mask_count"]) - sum(n)) < 1e-3 and masks_differ,
               f"{float(loss_v):.6f} vs {want:.6f}; scored {float(st_v['mask_count']):.0f} = {' + '.join(f'{k:.0f}' for k in n)}")

    # 2. the gradient: the same weighted average (the shared write's meta-gradient sums both reads)
    params, static = eqx.partition(model, eqx.is_array)
    m0, m1 = view_mask(0), view_mask(1)
    n0 = float(single(m0)[1]["mask_count"]); n1 = float(single(m1)[1]["mask_count"])

    def grad_of(batch, cfg):
        return jax.jit(jax.grad(lambda pr: loss_fn(eqx.combine(pr, static), batch, key, step, cfg)[0]))(params)
    g2 = grad_of(Batch(input_ids=ids), dataclasses.replace(lc, outer_views=2))
    one = dataclasses.replace(lc, outer_t=-1.0)
    t1 = jnp.ones((B,), jnp.float32)
    g0 = grad_of(Batch(input_ids=ids, loss_mask=m0, outer_t=t1), one)
    g1 = grad_of(Batch(input_ids=ids, loss_mask=m1, outer_t=t1), one)
    err = max(float(jnp.max(jnp.abs(a - (n0 * b + n1 * c) / (n0 + n1)))) for a, b, c in zip(leaves(g2), leaves(g0), leaves(g1)))
    scale = max(float(jnp.max(jnp.abs(a))) for a in leaves(g2))
    fin = all(bool(jnp.all(jnp.isfinite(a))) for a in leaves(g2))
    report("2 views: gradient == the weighted average of the single-view gradients", fin and err < 1e-5 * max(scale, 1.0),
           f"max |diff| {err:.1e} (largest gradient entry {scale:.2e})")

    # 3. CART weighting; refusals
    l_cart, st_cart = loss_fn(model, Batch(input_ids=ids), key, step,
                              dataclasses.replace(lc, time_reweighting="cart", outer_t=-1.0, outer_views=2))
    ok = bool(jnp.isfinite(l_cart)) and float(st_cart["masked_frac"]) < 1.0
    refused = 0
    for bad in (dataclasses.replace(lc, outer_views=0), dataclasses.replace(lc, outer_views=2, per_sequence=True),
                dataclasses.replace(lc, outer_views=2, per_token=True)):
        try:
            loss_fn(model, Batch(input_ids=ids), key, step, bad)
        except ValueError:
            refused += 1
    for kw in (dict(outer_views=0), dict(outer_views=2, distill=True), dict(outer_views=2, inner_from_outer=True)):
        try:
            TrainConfig(**kw)
        except ValueError:
            refused += 1
    report("CART weighting runs, finite; evaluation scores and incompatible options refused", ok and refused == 6,
           f"loss {float(l_cart):.4f}, masked share {float(st_cart['masked_frac']):.3f}, refused {refused}/6")

    print("ALL PASS" if not failures else "FAILURES: " + "; ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
