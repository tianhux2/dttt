# coding=utf-8
"""Objective primitives against inline references."""
from __future__ import annotations

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import jax
import jax.numpy as jnp
import numpy as np

from src.ttt.objective import (cart_kernel, from_chunks, make_inner_mask, q_sample, time_weights,
                               to_chunks, weighted_ce)


def main():
    failures = []
    B, T, C, V, H = 4, 32, 8, 50, 16
    key = jax.random.PRNGKey(0)

    # 1. q_sample: masked exactly where t_mask, t_mask = (u < t) & loss_mask, key split order.
    ids = jax.random.randint(key, (B, T), 1, V)
    lm = jnp.ones((B, T), bool).at[0, :4].set(False)
    x_t, t, tm = q_sample(key, ids, lm, mask_token_id=V - 1)
    kt, ku = jax.random.split(key)
    t_ref = jax.random.uniform(kt, (B,)); u_ref = jax.random.uniform(ku, (B, T))
    tm_ref = (u_ref < t_ref[:, None]) & lm
    ok = (bool(jnp.all(tm == tm_ref)) and bool(jnp.all(jnp.where(tm, x_t == V - 1, x_t == ids)))
          and not bool(jnp.any(tm[0, :4])))
    failures += [] if ok else ["q_sample semantics"]
    print(f"[{'PASS' if ok else 'FAIL'}] q_sample: t_mask=(u<t)&loss_mask, masked where t_mask")

    # 2. cart_kernel values.
    k = cart_kernel(p=0.1, window=3)[0, 0]
    ref = np.array([0.5 * 0.1 * 0.9 ** (abs(d) - 1) if d else 0.0 for d in range(-3, 4)], np.float32)
    r = float(jnp.max(jnp.abs(k - ref)))
    ok = r < 1e-7
    failures += [] if ok else ["cart_kernel values"]
    print(f"[{'PASS' if ok else 'FAIL'}] cart_kernel values                max abs = {r:.1e}")

    # 3. time_weights: zero off t_mask in every mode; cart = conv of visibility.
    kern = cart_kernel(0.1, 4)
    for mode in ("cart", "linear", "original", "none"):
        w = time_weights(mode, t, tm, kern)
        ok = bool(jnp.all(w[~tm] == 0.0)) and bool(jnp.all(w[tm] >= 0.0))
        failures += [] if ok else [f"time_weights({mode}) nonzero off t_mask"]
    w = time_weights("cart", t, tm, kern)
    vis = (1.0 - tm.astype(jnp.float32))
    ref = jnp.stack([jnp.convolve(vis[b], kern[0, 0], mode="same") for b in range(B)]) * tm
    r = float(jnp.max(jnp.abs(w - ref)))
    ok = ok and r < 1e-5
    failures += [] if ok else ["cart weights != conv reference"]
    print(f"[{'PASS' if ok else 'FAIL'}] time_weights zero off mask; cart==conv  max abs = {r:.1e}")

    # 4. inner mask: pinned consumes the key directly; sampled splits it; pad never masked.
    chunk = jax.random.randint(key, (C,), 1, V).at[0].set(0)          # position 0 is pad
    f_pin = make_inner_mask(0.5, (0.0, 1.0), V - 1, pad_token_id=0)
    f_smp = make_inner_mask(-1.0, (0.3, 0.9), V - 1, pad_token_id=0)
    _, m_pin = f_pin(chunk, key)
    ref_pin = (jax.random.uniform(key, (C,)) < 0.5) & (chunk != 0)
    _, m_smp = f_smp(chunk, key)
    kr, kd = jax.random.split(key)
    ref_smp = (jax.random.uniform(kd, (C,)) < jax.random.uniform(kr, (), minval=0.3, maxval=0.9)) & (chunk != 0)
    ok = bool(jnp.all(m_pin == ref_pin)) and bool(jnp.all(m_smp == ref_smp)) and not bool(m_pin[0]) and not bool(m_smp[0])
    failures += [] if ok else ["inner mask key consumption / pad exclusion"]
    print(f"[{'PASS' if ok else 'FAIL'}] inner mask: pinned/sampled key use, pad excluded")
    # sampled ratio lands in range over many draws
    keys = jax.random.split(key, 2000)
    fr = jax.vmap(lambda kk: f_smp(jax.random.randint(kk, (256,), 1, V), kk)[1].mean())(keys)
    ok = 0.55 < float(fr.mean()) < 0.65 and float(fr.min()) > 0.2 and float(fr.max()) < 0.98
    failures += [] if ok else [f"sampled ratio distribution off: mean={float(fr.mean()):.3f}"]
    print(f"[{'PASS' if ok else 'FAIL'}] sampled ratio ~ U(0.3,0.9)        mean = {float(fr.mean()):.3f}")

    # 5. to_chunks / from_chunks inverse, chunk-major.
    x = jnp.arange(B * T * 3).reshape(B, T, 3)
    xc = to_chunks(x, C)
    ok = xc.shape == (T // C, B, C, 3) and bool(jnp.all(from_chunks(xc) == x)) and bool(jnp.all(xc[1, 2] == x[2, C:2 * C]))
    failures += [] if ok else ["to_chunks/from_chunks"]
    print(f"[{'PASS' if ok else 'FAIL'}] to_chunks / from_chunks           {xc.shape}")

    # 6. weighted_ce: fused == naive, both against an inline log_softmax.
    hid = jax.random.normal(key, (T, H)); lw = jax.random.normal(kt, (V, H)) * 0.1
    tg = jax.random.randint(ku, (T,), 0, V); wt = jax.random.uniform(key, (T,))
    ref = jnp.sum(-jax.nn.log_softmax(hid @ lw.T, -1)[jnp.arange(T), tg] * wt)
    r0 = abs(float(weighted_ce(hid, lw, tg, wt, 0)) - float(ref)) / float(ref)
    r1 = abs(float(weighted_ce(hid, lw, tg, wt, 16)) - float(ref)) / float(ref)
    ok = r0 < 1e-6 and r1 < 1e-5
    failures += [] if ok else [f"weighted_ce off: dense {r0:.1e} fused {r1:.1e}"]
    print(f"[{'PASS' if ok else 'FAIL'}] weighted_ce dense/fused == ref    {r0:.1e} / {r1:.1e}")

    # 7. q_sample row-wise keys: batch shape & seed decoupling
    eval_seed = 12345
    n_holdout = 128
    perm = np.random.default_rng(eval_seed).permutation(n_holdout)
    def make_batch(j_indices, shape):
        keys_j = jax.vmap(lambda j: jax.random.fold_in(jax.random.PRNGKey(eval_seed), j))(jnp.asarray(j_indices, dtype=jnp.int32))
        splits = jax.vmap(jax.random.split)(keys_j)
        keys_u, keys_row = splits[:, 0, :], splits[:, 1, :]
        u = jax.vmap(lambda k: jax.random.uniform(k, ()))(keys_u)
        ranks = jnp.asarray(perm[j_indices], dtype=jnp.float32)
        t = (ranks + u) / float(n_holdout)
        k_mask = jax.vmap(lambda k: jax.random.split(k, 3))(keys_row)[:, 1, :]
        ids = jnp.zeros(shape, dtype=jnp.int32)
        return q_sample(k_mask, ids, None, V - 1, t=t)

    # Sequence 35 in B=32 (batch 1, index 3) vs B=64 (batch 0, index 35)
    _, _, mask_32 = make_batch(np.arange(32, 64), (32, T))
    _, _, mask_64 = make_batch(np.arange(0, 64), (64, T))
    ok = bool(jnp.all(mask_32[3] == mask_64[35]))
    failures += [] if ok else ["q_sample row-wise batch decoupling"]
    print(f"[{'PASS' if ok else 'FAIL'}] q_sample row-wise batch/seed decoupled (B=32 vs B=64)")

    print("ALL PASS" if not failures else "FAILURES: " + "; ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
