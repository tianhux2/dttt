# coding=utf-8
"""The visible window (``ModelConfig.window_reach``, ``LossConfig.window_drop``) and the self-prior read
(``LossConfig.self_prior``) on a tiny random model:

1. ``key_validity`` with ``reach``: None is the old rule; ``r`` keeps the newest ``r`` existing window chunks (0: the
   own chunk only); a traced ``r`` gives the same mask;
2. the training scan of a model with ``window_reach`` 0 equals the full model's with ``window_drop`` 1 (the same
   weights and draws), and differs from the full window; ``window_drop`` past ``window_drop_until`` is the full window;
3. self-prior 1: the training scan's loss of chunks 0 and 1 equals the decoder's read of the chunk with the memory
   after the chunk's OWN write (``commit_block``), the KV window as before it; self-prior 0 is the plain scan;
4. meta-gradients are finite with self-prior 0.5 and window drop 0.5, and the self-prior changes them;
5. self-prior 1 over five chunks (the window full and moving from chunk 3 on): the training scan's loss of every
   chunk equals the decoder's read with the memory after the chunk's own write.

    JAX_PLATFORMS=cpu .venv/bin/python tests/test_window_self_prior.py
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
import numpy as np

from src.ttt.attention import key_validity
from src.ttt.chunk_scan import Batch, LossConfig, loss_fn
from src.ttt.config import ModelConfig
from src.ttt.generate import block_forward, commit_block, init_state
from src.ttt.model import Model
from src.ttt.objective import q_sample

BASE = dict(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
            num_key_value_heads=2, head_dim=8, suffix_len=2, chunk_size=8, window_chunks=2,
            mask_token_id=120, pad_token_id=0, eos_token_id=0)
C, Wn = BASE["chunk_size"], BASE["window_chunks"]
failures = []


def report(name, ok, detail=""):
    failures.append(name) if not ok else None
    print(f"[{'PASS' if ok else 'FAIL'}] {name:<78} {detail}")


def leaves(t):
    return jax.tree_util.tree_leaves(eqx.filter(t, eqx.is_array))


def main():
    # 1. key_validity with reach
    ok = True
    for c in range(5):
        old = np.concatenate([(c - Wn + np.arange(Wn * C) // C) >= 0, np.ones(C, bool)])
        ok &= np.array_equal(np.asarray(key_validity(jnp.int32(c), C, Wn)), old)
        for r in range(Wn + 1):
            want = old.copy()
            want[:Wn * C] &= (np.arange(Wn * C) // C) >= Wn - r
            got = np.asarray(key_validity(jnp.int32(c), C, Wn, r))
            traced = np.asarray(jax.jit(lambda rr: key_validity(jnp.int32(c), C, Wn, rr))(jnp.int32(r)))
            ok &= np.array_equal(got, want) and np.array_equal(traced, want)
        ok &= np.array_equal(np.asarray(key_validity(jnp.int32(c), C, Wn, 0)),
                             np.concatenate([np.zeros(Wn * C, bool), np.ones(C, bool)]))
    report("key_validity: reach None = old rule; reach r = newest r chunks; traced = static", bool(ok))

    # 2. window_reach 0 == window_drop 1; past window_drop_until the full window
    full = Model(ModelConfig(**BASE), jax.random.PRNGKey(0))
    reach0 = Model(ModelConfig(**BASE, window_reach=0), jax.random.PRNGKey(0))
    T = 4 * C
    ids = jax.random.randint(jax.random.PRNGKey(5), (2, T), 1, 100)
    key = jax.random.PRNGKey(11)
    base = LossConfig(time_reweighting="none", inner_lr=0.5, inner_task="causal", outer_shift=True, fused_ce_block=0,
                      compute_dtype="float32", attn_chunk_batch=2, outer_t=0.6)
    L = lambda m, c, step=5: float(loss_fn(m, Batch(input_ids=ids), key, jnp.int32(step), c)[0])
    l_full, l_r0 = L(full, base), L(reach0, base)
    l_drop = L(full, dataclasses.replace(base, window_drop=1.0, window_drop_until=10 ** 6))
    l_late = L(full, dataclasses.replace(base, window_drop=1.0, window_drop_until=3))
    report("window_reach 0 == window_drop 1; drop past window_drop_until == full window",
           abs(l_r0 - l_drop) < 1e-6 and abs(l_late - l_full) < 1e-6 and abs(l_full - l_r0) > 1e-5,
           f"{l_r0:.6f} / {l_drop:.6f}; full {l_full:.6f} / late {l_late:.6f}")

    # 3. self-prior 1 == the decoder reading each chunk with the memory after its own write
    model = full
    ids1 = ids[:1, :2 * C]
    _, key_mask, _ = jax.random.split(key, 3)
    x_t, _, t_mask = q_sample(key_mask, ids1, None, BASE["mask_token_id"], t=jnp.full((1,), 0.6))
    _, st0 = loss_fn(model, Batch(input_ids=ids1), key, jnp.int32(5), base)
    _, st1 = loss_fn(model, Batch(input_ids=ids1), key, jnp.int32(5), dataclasses.replace(base, self_prior=1.0))
    s0 = init_state(model, jnp.float32)
    commit = lambda c, st: commit_block(model, ids1[0, c * C:(c + 1) * C], jnp.arange(c * C, (c + 1) * C), st,
                                        jax.random.PRNGKey(0), 0.5, 1.0, 0.5, 0, jnp.float32, inner_task="causal")
    s1 = commit(0, s0)
    W2 = commit(1, s1).W

    def dec_loss(c, state):
        p = jnp.arange(c * C, (c + 1) * C)
        logits, _, _ = block_forward(model, x_t[0, c * C:(c + 1) * C], p, state, jnp.float32, pos_q=p + 1)
        tgt = jnp.concatenate([ids1[0, c * C + 1:(c + 1) * C], ids1[0, (c + 1) * C:(c + 1) * C + 1]])
        if tgt.shape[0] < C:
            tgt = jnp.concatenate([tgt, jnp.zeros((C - tgt.shape[0],), tgt.dtype)])
        m = jnp.concatenate([t_mask[0, c * C + 1:(c + 1) * C], t_mask[0, (c + 1) * C:(c + 1) * C + 1]])
        if m.shape[0] < C:
            m = jnp.concatenate([m, jnp.zeros((C - m.shape[0],), bool)])
        ce = -jnp.take_along_axis(jax.nn.log_softmax(logits, axis=-1), tgt[:, None], axis=-1)[:, 0]
        m = m.astype(jnp.float32)
        return float(jnp.sum(m * ce) / jnp.sum(m))
    d0_plain, d1_plain = dec_loss(0, s0), dec_loss(1, s1)
    d0_sp, d1_sp = dec_loss(0, s0._replace(W=s1.W)), dec_loss(1, s1._replace(W=W2))
    t0 = [float(x) for x in st0["loss_by_chunk"][:2]]
    t1 = [float(x) for x in st1["loss_by_chunk"][:2]]
    ok = (abs(t0[0] - d0_plain) < 2e-5 and abs(t0[1] - d1_plain) < 2e-5
          and abs(t1[0] - d0_sp) < 2e-5 and abs(t1[1] - d1_sp) < 2e-5 and abs(t1[1] - t0[1]) > 1e-6)
    report("self-prior 1: training scan == decoder with the chunk's own write; 0: the plain scan", ok,
           f"plain {t0} vs {d0_plain:.6f}, {d1_plain:.6f}; self-prior {t1} vs {d0_sp:.6f}, {d1_sp:.6f}")

    # 4. finite meta-gradients; the self-prior changes them
    params, static = eqx.partition(model, eqx.is_array)

    def grads(c):
        f = lambda p: loss_fn(eqx.combine(p, static), Batch(input_ids=ids), key, jnp.int32(5), c)[0]
        return jax.jit(jax.grad(f))(params)
    g0 = grads(base)
    g_sp = grads(dataclasses.replace(base, self_prior=0.5))
    g_wd = grads(dataclasses.replace(base, window_drop=0.5, window_drop_until=10 ** 6))
    fin = all(bool(jnp.all(jnp.isfinite(x))) for g in (g_sp, g_wd) for x in leaves(g))
    diff = max(float(jnp.max(jnp.abs(a - b))) for a, b in zip(leaves(g_sp), leaves(g0)))
    report("meta-gradients finite with self-prior 0.5 / window drop 0.5; self-prior changes them",
           fin and diff > 1e-8, f"max |g_sp - g0| {diff:.2e}")

    # 5. self-prior 1 over five chunks, the window moving: scan == decoder
    nc5 = 5
    ids5 = jax.random.randint(jax.random.PRNGKey(6), (1, nc5 * C), 1, 100)
    x5, _, tm5 = q_sample(key_mask, ids5, None, BASE["mask_token_id"], t=jnp.full((1,), 0.6))
    l_sp = [float(x) for x in loss_fn(model, Batch(input_ids=ids5), key, jnp.int32(5),
                                      dataclasses.replace(base, self_prior=1.0))[1]["loss_by_chunk"]]
    states = [init_state(model, jnp.float32)]
    for c in range(nc5):
        states.append(commit_block(model, ids5[0, c * C:(c + 1) * C], jnp.arange(c * C, (c + 1) * C), states[-1],
                                   jax.random.PRNGKey(0), 0.5, 1.0, 0.5, 0, jnp.float32, inner_task="causal"))

    def dec5(j):
        p = jnp.arange(j * C, (j + 1) * C)
        logits, _, _ = block_forward(model, x5[0, j * C:(j + 1) * C], p, states[j]._replace(W=states[j + 1].W),
                                     jnp.float32, pos_q=p + 1)
        tgt = jnp.pad(ids5[0, j * C + 1:(j + 1) * C + 1], (0, 1))[:C]          # the sequence's last slot: no target
        m = jnp.pad(tm5[0, j * C + 1:(j + 1) * C + 1], (0, 1))[:C].astype(jnp.float32)
        ce = -jnp.take_along_axis(jax.nn.log_softmax(logits, axis=-1), tgt[:, None], axis=-1)[:, 0]
        return float(jnp.sum(m * ce) / jnp.maximum(jnp.sum(m), 1e-6))
    err = max(abs(a - dec5(j)) for j, a in enumerate(l_sp))
    report("self-prior 1 over 5 chunks (moving window): training scan == decoder", err < 2e-5,
           f"max |scan - decoder| {err:.1e}")

    print("ALL PASS" if not failures else "FAILURES: " + "; ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
