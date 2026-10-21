# coding=utf-8
"""
Sequences shorter than the attention window (e.g. one 1024-token chunk under a window of 3 chunks): the window
slots before the sequence start are masked, so a window of 3 chunks gives the loss and gradient of a window that
just covers the sequence - for the plain write, the no-write baseline and the bound's fixed-t read; and several
segments shorter than the window are still refused (their carried window would be short).
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")
os.environ.setdefault("TTT_ATTENTION", "xla")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from src.ttt.chunk_scan import Batch, LossConfig, loss_fn
from src.ttt.config import ModelConfig
from src.ttt.model import Model

V, C = 128, 8
BASE = ModelConfig(
    vocab_size=V, hidden_size=32, intermediate_size=64, num_hidden_layers=3, num_attention_heads=4,
    num_key_value_heads=2, head_dim=8, suffix_len=1, chunk_size=C, window_chunks=3, mask_token_id=120,
    pad_token_id=0, eos_token_id=0)


def loss_and_grad(window: int, nc: int, lc: LossConfig):
    model = Model(dataclasses.replace(BASE, window_chunks=window), jax.random.PRNGKey(0))
    ids = jax.random.randint(jax.random.PRNGKey(1), (2, nc * C), 1, 110)
    f = lambda m: loss_fn(m, Batch(input_ids=ids), jax.random.PRNGKey(2), jnp.int32(3), lc)[0]
    loss, g = eqx.filter_value_and_grad(f)(model)
    return float(loss), np.concatenate([np.ravel(x) for x in jax.tree_util.tree_leaves(eqx.filter(g, eqx.is_array))])


def main():
    failures = []
    plain = LossConfig(fused_ce_block=0, compute_dtype="float32", outer_shift=True, inner_task="causal",
                       cart_window=16)
    cases = (("one chunk, write", 1, plain),
             ("two chunks, write", 2, plain),
             ("one chunk, no write", 1, dataclasses.replace(plain, no_write=True)),
             ("one chunk, fixed t 0.5", 1, dataclasses.replace(plain, outer_t=0.5, time_reweighting="none")))
    for name, nc, lc in cases:
        l3, g3 = loss_and_grad(3, nc, lc)
        ln, gn = loss_and_grad(nc, nc, lc)
        err = float(np.max(np.abs(g3 - gn)) / np.max(np.abs(gn)))
        ok = abs(l3 - ln) <= 1e-6 * abs(ln) and err < 1e-5 and np.isfinite(l3)
        failures += [] if ok else [f"{name}: loss {l3} vs {ln}, grad rel err {err:.1e}"]
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: window 3 == window {nc}   loss {l3:.6f} / {ln:.6f}, "
              f"grad max rel err {err:.1e}")

    # several segments shorter than the window: refused
    try:
        loss_and_grad(3, 4, dataclasses.replace(plain, segment_chunks=2))
        ok = False
    except ValueError:
        ok = True
    failures += [] if ok else ["segments shorter than the window were not refused"]
    print(f"[{'PASS' if ok else 'FAIL'}] 2 segments of 2 chunks under a window of 3 refused")

    print("ALL PASS" if not failures else "FAILURES: " + "; ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
