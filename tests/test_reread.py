# coding=utf-8
"""Re-read evaluation layout (``LossConfig.reread_chunks`` / ``score_from_chunk``): the first
n chunks are processed twice with the KV window restarted at the second pass, only the
tail is scored, and the outer masks of the tail are exactly those of the single pass, so
the re-read benefit is a paired difference.

    JAX_PLATFORMS=cpu .venv/bin/python tests/test_reread.py
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
from src.ttt.inner_masks import split_perm
from src.ttt.model import Model

CFG = ModelConfig(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=8, suffix_len=2,
                  chunk_size=8, window_chunks=1, mask_token_id=120, pad_token_id=0, eos_token_id=0)
C, NC, B = CFG.chunk_size, 4, 3
T = C * NC
BASE = LossConfig(cart_window=16, inner_lr=0.05, inner_mask_ratio=0.3, fused_ce_block=0,
                  compute_dtype="float32", attn_chunk_batch=2, per_sequence=True, per_token=True)
failures = []


def report(name, ok, detail=""):
    failures.append(name) if not ok else None
    print(f"[{'PASS' if ok else 'FAIL'}] {name:<62} {detail}")


def main():
    model = Model(CFG, jax.random.PRNGKey(0))
    ids = jax.random.randint(jax.random.PRNGKey(1), (B, T), 1, 100)
    key, step = jax.random.PRNGKey(2), jnp.int32(5)
    run = lambda cfg, batch=None: loss_fn(model, batch or Batch(input_ids=ids), key, step, cfg)
    n = 2
    ctl = dataclasses.replace(BASE, score_from_chunk=n, segment_chunks=n)
    rr = dataclasses.replace(ctl, reread_chunks=n)

    # 1. score_from_chunk == a host loss_mask on the same positions (bitwise: same key path)
    l_a, s_a = run(dataclasses.replace(BASE, score_from_chunk=n))
    l_b, s_b = run(BASE, Batch(input_ids=ids, loss_mask=jnp.broadcast_to(jnp.arange(T) >= n * C, (B, T))))
    ok = float(l_a) == float(l_b) and bool(jnp.all(s_a["ce_by_token"] == s_b["ce_by_token"]))
    report("score_from_chunk == host loss_mask (bitwise)", ok, f"{float(l_a):.5f} vs {float(l_b):.5f}")

    # 2. write off: the re-read tail is the single pass's tail (proves the window reset and the
    #    coordinate mapping; without the reset the second pass's first chunks would see pass 1)
    l0, s0 = run(dataclasses.replace(ctl, inner_lr=0.0))
    r0, t0 = run(dataclasses.replace(rr, inner_lr=0.0))
    shapes = t0["loss_by_chunk"].shape == (NC,) and t0["ce_by_token"].shape == (B, T) and t0["mask_by_token"].shape == (B, T)
    close = (abs(float(l0 - r0)) < 1e-5 and bool(jnp.allclose(s0["loss_by_chunk"], t0["loss_by_chunk"], atol=1e-5))
             and bool(jnp.allclose(s0["ce_by_token"][:, n * C:], t0["ce_by_token"][:, n * C:], atol=1e-5))
             and bool(jnp.all(s0["mask_by_token"] == t0["mask_by_token"]))
             and bool(jnp.allclose(s0["loss_by_seq"], t0["loss_by_seq"], atol=1e-5)))
    unscored = bool(jnp.all(t0["loss_by_chunk"][:n] == 0)) and bool(jnp.all(t0["mask_by_token"][:, :n * C] == 0))
    report("write off: reread tail == single-pass tail; shapes in original coordinates", shapes and close and unscored,
           f"loss {float(l0):.5f} vs {float(r0):.5f}")

    # 3. write on: the second write changes the tail, everything finite, W drifts more
    l1, s1 = run(ctl)
    r1, t1 = run(rr)
    ok = (abs(float(l1 - r1)) > 1e-7 and bool(jnp.isfinite(r1))
          and float(t1["ttt_drift_rel"]) != float(s1["ttt_drift_rel"]))
    report("write on: reread differs from the single pass, finite", ok,
           f"single {float(l1):.5f} reread {float(r1):.5f} drift {float(s1['ttt_drift_rel']):.2e} -> {float(t1['ttt_drift_rel']):.2e}")

    # 4. the causal and split tasks, host-side perms shifted for the duplicated block
    perm = jnp.asarray(np.stack([split_perm(np.asarray(r), "random", C, CFG.eos_token_id, CFG.pad_token_id, 0.5)
                                 for r in ids]))
    for name, kw, bb in (("causal", dict(inner_task="causal"), None),
                         ("split", dict(inner_task="split", inner_context_frac=0.5), Batch(input_ids=ids, inner_perm=perm))):
        lc0, sc0 = run(dataclasses.replace(ctl, inner_lr=0.0, **kw), bb)
        rc0, tc0 = run(dataclasses.replace(rr, inner_lr=0.0, **kw), bb)
        lc1, _ = run(dataclasses.replace(ctl, **kw), bb)
        rc1, _ = run(dataclasses.replace(rr, **kw), bb)
        ok = abs(float(lc0 - rc0)) < 1e-5 and abs(float(lc1 - rc1)) > 1e-7 and bool(jnp.isfinite(rc1))
        report(f"inner_task {name}: paired at lr 0, differs at lr > 0", ok, f"{float(lc0):.5f}/{float(rc0):.5f} -> {float(lc1):.5f}/{float(rc1):.5f}")

    # 5. errors: the reset must be a segment boundary
    for bad in (dict(reread_chunks=3, segment_chunks=2), dict(reread_chunks=2, segment_chunks=0)):
        try:
            run(dataclasses.replace(BASE, **bad)); ok = False
        except ValueError:
            ok = True
        report(f"rejects {bad}", ok)

    print("ALL PASS" if not failures else "FAILURES: " + "; ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
