# coding=utf-8
"""The auxiliary-loss control: ``write_off`` keeps the inner stream (its CE) but never updates W,
``aux_causal_weight`` adds that CE to the outer objective.

    JAX_PLATFORMS=cpu .venv/bin/python tests/test_aux_causal.py
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

CFG = ModelConfig(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=8, suffix_len=2,
                  chunk_size=8, window_chunks=1, mask_token_id=120, pad_token_id=0, eos_token_id=0)
failures = []


def report(name, ok, detail=""):
    failures.append(name) if not ok else None
    print(f"[{'PASS' if ok else 'FAIL'}] {name:<70} {detail}")


def main():
    model = Model(CFG, jax.random.PRNGKey(0))
    ids = jax.random.randint(jax.random.PRNGKey(1), (3, 32), 1, 100)
    bb, key = Batch(input_ids=ids), jax.random.PRNGKey(2)
    leaves = lambda g: jax.tree_util.tree_leaves(eqx.filter(g, eqx.is_array))
    rel = lambda a, b: max(float(jnp.max(jnp.abs(x - y))) / (float(jnp.max(jnp.abs(x))) + 1e-12) for x, y in zip(leaves(a), leaves(b)))
    run = lambda c: eqx.filter_value_and_grad(lambda m: loss_fn(m, bb, key, jnp.int32(5), c)[0])(model)
    stats = lambda c: loss_fn(model, bb, key, jnp.int32(5), c)[1]
    for task, shift in (("causal", True), ("causal", False), ("mask", False)):
        base = LossConfig(cart_window=16, inner_lr=0.05, inner_mask_ratio=0.3, fused_ce_block=0, compute_dtype="float32",
                          attn_chunk_batch=2, inner_task=task, outer_shift=shift)
        l_w, g_w = run(base)
        l_off, g_off = run(dataclasses.replace(base, write_off=True))
        l_lr0, g_lr0 = run(dataclasses.replace(base, inner_lr=0.0))
        st_off = stats(dataclasses.replace(base, write_off=True))
        l_aux, g_aux = run(dataclasses.replace(base, write_off=True, aux_causal_weight=0.3))
        st_aux = stats(dataclasses.replace(base, write_off=True, aux_causal_weight=0.3))
        l_wa, g_wa = run(dataclasses.replace(base, aux_causal_weight=0.3))
        st_w = stats(base)
        ok = (abs(float(l_off - l_lr0)) < 1e-6 and rel(g_off, g_lr0) < 1e-6 and float(st_off["ttt_drift_rel"]) == 0.0
              and abs(float(l_aux) - (float(st_aux["main_loss"]) + 0.3 * float(st_aux["inner_loss"]))) < 1e-5
              and rel(g_off, g_aux) > 1e-3
              and abs(float(l_wa) - (float(st_w["main_loss"]) + 0.3 * float(st_w["inner_loss"]))) < 1e-5
              and rel(g_w, g_wa) > 1e-3 and float(l_w) != float(l_off))
        report(f"{task}{'+shift' if shift else ''}: write_off == inner_lr 0 (loss, grads, drift 0); aux = main + 0.3 inner; grads change",
               ok, f"loss {float(l_w):.4f} off {float(l_off):.4f} aux {float(l_aux):.4f}")
    base = LossConfig(fused_ce_block=0, compute_dtype="float32", inner_task="causal", outer_shift=True)
    for bad in (dict(write_off=True, no_write=True), dict(aux_causal_weight=0.3, no_write=True),
                dict(write_off=True, inner_from_outer=True), dict(write_off=True, inner_views=2)):
        try:
            loss_fn(model, bb, key, jnp.int32(5), dataclasses.replace(base, **bad)); ok = False
        except ValueError:
            ok = True
        report(f"rejects {bad}", ok)
    try:
        TrainConfig(write_off=True, no_write=True); ok = False
    except ValueError:
        ok = True
    TrainConfig(write_off=True, aux_causal_weight=0.3, inner_task="causal", outer_shift=True)
    report("TrainConfig validation", ok)
    print("ALL PASS" if not failures else "FAILURES: " + "; ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
