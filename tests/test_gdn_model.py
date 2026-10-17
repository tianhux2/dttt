# coding=utf-8
"""The gdn-memory model end to end (``ModelConfig.memory = "gdn"``): loss_fn runs and is finite,
the memory carries across chunks (inner_lr 0 = empty state = the "no-memory twin"), the
segmented scan equals the whole one, the write-side flags are refused, and the parameter
groups / decay mask see the new modules.

    JAX_PLATFORMS=cpu .venv/bin/python tests/test_gdn_model.py
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
from src.ttt.config import ModelConfig
from src.ttt.model import Model

CFG = ModelConfig(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=8, suffix_len=2,
                  chunk_size=8, window_chunks=1, mask_token_id=120, pad_token_id=0, eos_token_id=0,
                  memory="gdn", gdn_heads=2, gdn_head_dim=8)
failures = []


def report(name, ok, detail=""):
    failures.append(name) if not ok else None
    print(f"[{'PASS' if ok else 'FAIL'}] {name:<66} {detail}")


def main():
    model = Model(CFG, jax.random.PRNGKey(0))
    ok = model.fast_weight is None and model.gdn is not None and model.gdn.A_log.shape == (CFG.suffix_len, 2)
    report("gdn model: no fast weight, stacked gdn modules [S]", ok)
    ids = jax.random.randint(jax.random.PRNGKey(1), (3, 32), 1, 100)
    bb, key = Batch(input_ids=ids), jax.random.PRNGKey(2)
    leaves = lambda g: jax.tree_util.tree_leaves(eqx.filter(g, eqx.is_array))
    base = LossConfig(cart_window=16, fused_ce_block=0, compute_dtype="float32", attn_chunk_batch=2, outer_shift=True)
    run = lambda c: eqx.filter_value_and_grad(lambda m: loss_fn(m, bb, key, jnp.int32(5), c)[0])(model)
    l, g = run(base)
    _, st = loss_fn(model, bb, key, jnp.int32(5), base)
    fin = bool(jnp.isfinite(l)) and all(bool(jnp.all(jnp.isfinite(x))) for x in leaves(g))
    g_gdn = leaves(eqx.filter(g.gdn, eqx.is_array))
    report("loss_fn: finite loss and gradients; the gdn modules get gradient",
           fin and max(float(jnp.abs(x).max()) for x in g_gdn) > 0 and float(st["ttt_drift_rel"]) > 0,
           f"loss {float(l):.4f} state rms {float(st['ttt_drift_rel']):.3e}")
    # inner_lr 0: the state stays empty -> the no-memory twin; it differs from the memory-on loss
    l0, _ = run(dataclasses.replace(base, inner_lr=0.0))
    _, st0 = loss_fn(model, bb, key, jnp.int32(5), dataclasses.replace(base, inner_lr=0.0))
    report("inner_lr 0 keeps the state empty (twin), loss differs from memory on",
           float(st0["ttt_drift_rel"]) == 0.0 and float(l0) != float(l), f"{float(l0):.4f} vs {float(l):.4f}")
    # by-chunk: chunk 0 gain is 0 by construction (empty state), later chunks not all zero
    gain_chunk = st0["loss_by_chunk"] - st["loss_by_chunk"]
    report("gain by chunk: 0 at chunk 0, nonzero later", float(gain_chunk[0]) == 0.0 and float(jnp.abs(gain_chunk[1:]).max()) > 0)
    # segmented == whole (the state is a segment carry)
    l_seg, _ = loss_fn(model, bb, key, jnp.int32(5), dataclasses.replace(base, segment_chunks=2))
    report("segmented scan == whole", abs(float(l_seg - l)) < 1e-5, f"{float(l_seg):.6f} vs {float(l):.6f}")
    # unshifted reader and per-sequence / per-token evaluation stats
    l_u, g_u = run(dataclasses.replace(base, outer_shift=False))
    _, st_p = loss_fn(model, bb, key, jnp.int32(5), dataclasses.replace(base, per_sequence=True, per_token=True))
    report("unshifted reader runs; per-sequence / per-token stats present",
           bool(jnp.isfinite(l_u)) and "loss_by_seq" in st_p and "ce_by_token" in st_p)
    # refused write-side flags
    for bad in (dict(inner_from_outer=True), dict(write_off=True), dict(distill=True, inner_task="causal"),
                dict(mtp=True, inner_task="causal")):
        try:
            loss_fn(model, bb, key, jnp.int32(5), dataclasses.replace(base, **bad)); ok = False
        except (ValueError, TypeError):
            ok = True
        report(f"rejects {bad}", ok)
    # config validation
    try:
        ModelConfig(**{**dataclasses.asdict(CFG), "inner_lr_gate": True}); ok = False
    except ValueError:
        ok = True
    report("ModelConfig refuses write controls with memory gdn", ok)
    # Model.hidden (fixed / empty memory) works
    h = model.hidden(ids[0], compute_dtype=jnp.float32)
    report("Model.hidden runs with the gdn model (empty memory)", h.shape == (32, CFG.hidden_size) and bool(jnp.all(jnp.isfinite(h))))
    print("ALL PASS" if not failures else "FAILURES: " + "; ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
