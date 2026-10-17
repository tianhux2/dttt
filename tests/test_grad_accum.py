# coding=utf-8
"""Gradient accumulation is exact: the step over N micro-batches applies the gradient of
``sum_m ce_m / sum_m w_m`` - the weighted combination of the micro-batch gradients with
their own mask-weight sums - and its diagnostics are the same combination.

    JAX_PLATFORMS=cpu .venv/bin/python tests/test_grad_accum.py
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")
os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")
os.environ.setdefault("TTT_ATTENTION", "xla")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from src.ttt.chunk_scan import Batch, LossConfig, loss_fn
from src.ttt.config import ModelConfig
from src.ttt.model import Model
from src.ttt.sharding import make_mesh
from src.ttt.train import compute_model, make_train_step

CFG = ModelConfig(
    vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
    num_attention_heads=4, num_key_value_heads=2, head_dim=8, suffix_len=1,
    chunk_size=8, window_chunks=1, mask_token_id=120, pad_token_id=0, eos_token_id=0,
)
N, B, T = 2, 4, 32
LCFG = LossConfig(cart_window=16, inner_lr=0.005, fused_ce_block=0, compute_dtype="float32", attn_chunk_batch=2)


def _leaves(t):
    return [np.asarray(x) for x in jax.tree_util.tree_leaves(eqx.filter(t, eqx.is_array))]


def _copy(t):
    """The train step donates its parameter buffers; hand it a fresh copy."""
    return jax.tree_util.tree_map(lambda x: jnp.array(x, copy=True), t)


def main():
    ndev = min(4, jax.device_count())
    mesh = make_mesh(jax.devices()[:ndev])
    jax.set_mesh(mesh)
    model = Model(CFG, jax.random.PRNGKey(0))
    params, static = eqx.partition(model, eqx.is_array)
    ids = jax.random.randint(jax.random.PRNGKey(1), (N, B, T), 1, 100)
    key, step = jax.random.PRNGKey(3), jnp.int32(5)
    failures = []

    # Reference: the N micro-batch gradients one by one, with the keys the scan draws,
    # combined with weights w_m = mask_count_m + 1e-6 (the denominators of their losses).
    keys = jax.random.split(key, N)
    def objective(p, b, k):
        return loss_fn(compute_model(p, static, mesh, "suffix,embed"), b, k, step, LCFG, mesh)
    gs, ws, ls, sts = [], [], [], []
    vg = jax.jit(jax.value_and_grad(objective, has_aux=True))
    for m in range(N):
        (l, st), g = vg(params, Batch(input_ids=ids[m]), keys[m])
        gs.append(_leaves(g)); ws.append(float(st["mask_count"]) + 1e-6); ls.append(float(l)); sts.append(st)
    wsum = sum(ws)
    ref_g = [sum(w * g[i] for w, g in zip(ws, gs)) / wsum for i in range(len(gs[0]))]
    ref_loss = sum(w * l for w, l in zip(ws, ls)) / wsum
    ref_inner = sum(float(s["inner_loss"]) for s in sts) / N

    # The accumulated step with SGD(1.0): params_new = params - grad.
    opt = optax.sgd(1.0)
    p_new, _, m = make_train_step(opt, LCFG, mesh, "suffix,embed", grad_accum=N)(
        static, _copy(params), opt.init(params), Batch(input_ids=_copy(ids)), _copy(key), step)
    got_g = [a - b for a, b in zip(_leaves(params), _leaves(p_new))]
    # Tolerance: the meta-gradient amplifies rounding-order differences between program
    # structures to ~1e-3 on the smallest leaves (eager vs jit of the SAME gradient differ
    # by that much); a wrong combination (mean instead of weighted, wrong denominator)
    # is off by percent to tens of percent.
    gmax = max(float(np.max(np.abs(b))) for b in ref_g)
    worst = max(float(np.max(np.abs(a - b))) / max(float(np.max(np.abs(b))), 1e-3 * gmax) for a, b in zip(got_g, ref_g))
    ok = worst < 5e-3 and abs(float(m["loss"]) - ref_loss) < 1e-5 and abs(float(m["inner_loss"]) - ref_inner) < 1e-5
    failures += [] if ok else [f"accum {N}: max rel grad err {worst:.2e}, loss {float(m['loss']):.6f} vs {ref_loss:.6f}"]
    print(f"[{'PASS' if ok else 'FAIL'}] grad_accum {N} == weighted combination of micro-batch gradients "
          f"(max rel err {worst:.1e}; loss {float(m['loss']):.5f} ref {ref_loss:.5f}; mask_count {float(m['mask_count']):.0f} = {wsum - N * 1e-6:.0f})")
    ok = abs(float(m["mask_count"]) - (wsum - N * 1e-6)) < 1e-2
    failures += [] if ok else ["mask_count is not the total over micro-batches"]

    # The diagnostics as for one batch of all N: main_loss (the training log's "loss") is sum CE /
    # sum weight like the loss - a mean of the micro-batch ratios would move with the split - and
    # the drift norms run over all sequences (their squares add), drift_rel being their ratio.
    ref_main = sum(w * float(s["main_loss"]) for w, s in zip(ws, sts)) / wsum
    d_ref = float(np.sqrt(sum(float(s["ttt_delta_norm"]) ** 2 for s in sts)))
    i_ref = float(np.sqrt(sum(float(s["ttt_init_norm"]) ** 2 for s in sts)))
    ok = (abs(float(m["main_loss"]) - ref_main) < 1e-5 and abs(float(m["ttt_delta_norm"]) - d_ref) < 1e-4 * d_ref
          and abs(float(m["ttt_drift_rel"]) - d_ref / i_ref) < 1e-4 * d_ref / i_ref)
    failures += [] if ok else [f"diagnostics: main_loss {float(m['main_loss']):.6f} vs {ref_main:.6f}, "
                               f"delta norm {float(m['ttt_delta_norm']):.4g} vs {d_ref:.4g}"]
    print(f"[{'PASS' if ok else 'FAIL'}] diagnostics as one batch: main_loss = the loss ({float(m['main_loss']):.5f}), "
          f"drift norms over all sequences (delta {float(m['ttt_delta_norm']):.4g} ref {d_ref:.4g})")

    # The gradient-noise-scale diagnostic: tr(S) and |G|^2 from the micro-batch norms.
    micro_sq = sum(sum(float(np.sum(x * x)) for x in g) for g in gs) / N
    full_sq = sum(float(np.sum(x * x)) for x in ref_g)
    tr_sigma = B * N * (micro_sq - full_sq) / (N - 1); signal_sq = (N * full_sq - micro_sq) / (N - 1)
    want = tr_sigma / signal_sq if signal_sq > 0 else float("nan")
    got = float(m["grad_noise_scale"])
    same = (np.isnan(got) and np.isnan(want)) or abs(got - want) <= 2e-2 * abs(want) + 1e-3
    ok = same and abs(float(m["grad_norm_micro"]) - np.sqrt(micro_sq)) < 1e-3 * np.sqrt(micro_sq)
    failures += [] if ok else [f"grad_noise_scale {got:.4g} vs {want:.4g}"]
    print(f"[{'PASS' if ok else 'FAIL'}] grad_noise_scale (B_simple) from the micro-batch norms   {got:.3g} seq (ref {want:.3g})")

    # grad_accum 1 is the unchanged path: identical to a direct value_and_grad.
    (l1, st1), g1 = jax.value_and_grad(objective, has_aux=True)(params, Batch(input_ids=ids[0]), key)
    p1, _, m1 = make_train_step(opt, LCFG, mesh, "suffix,embed", grad_accum=1)(
        static, _copy(params), opt.init(params), Batch(input_ids=_copy(ids[0])), _copy(key), step)
    g1 = _leaves(g1); gmax1 = max(float(np.max(np.abs(b))) for b in g1)
    d = max(float(np.max(np.abs((a - b) - c))) / max(float(np.max(np.abs(c))), 1e-3 * gmax1)
            for a, b, c in zip(_leaves(params), _leaves(p1), g1))
    ok = d < 5e-3 and abs(float(m1["loss"]) - float(l1)) < 1e-6
    failures += [] if ok else [f"grad_accum 1 differs from the direct gradient by {d:.2e}"]
    print(f"[{'PASS' if ok else 'FAIL'}] grad_accum 1 equals the direct step (max rel err {d:.1e}, rounding only)")

    # Non-uniform weights matter: with the loss masks differing between micro-batches the
    # plain mean of the micro-batch gradients is NOT the reference (guards the weighting).
    mean_g = [sum(g[i] for g in gs) / N for i in range(len(gs[0]))]
    dev = max(float(np.max(np.abs(a - b))) / max(float(np.max(np.abs(b))), 1e-3 * gmax) for a, b in zip(mean_g, ref_g))
    print(f"[info] unweighted mean of micro-batch gradients deviates from the exact one by {dev:.1e} "
          f"(weights {ws[0] - 1e-6:.0f} / {ws[1] - 1e-6:.0f})")

    print("ALL PASS" if not failures else "FAILURES: " + "; ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
