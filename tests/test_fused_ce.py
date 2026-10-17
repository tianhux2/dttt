# coding=utf-8
"""Fused blockwise CE must match the naive log_softmax path in value and gradient."""
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import jax
import jax.numpy as jnp

from src.ttt.fused_ce import fused_weighted_ce, naive_weighted_ce

T, H, V = 64, 48, 1000          # V deliberately not a multiple of block_size
k = jax.random.split(jax.random.PRNGKey(0), 4)
hidden = jax.random.normal(k[0], (T, H), jnp.float32)
lm_w = jax.random.normal(k[1], (V, H), jnp.float32) * 0.05
tgt = jax.random.randint(k[2], (T,), 0, V)
w = jnp.where(jax.random.uniform(k[3], (T,)) < 0.4, jax.random.uniform(k[3], (T,)), 0.0)

fails = []
for block in (128, 256, 384):
    lf = float(fused_weighted_ce(hidden, lm_w, tgt, w, block))
    ln = float(naive_weighted_ce(hidden, lm_w, tgt, w))
    gh_f, gw_f = jax.grad(lambda h, m: fused_weighted_ce(h, m, tgt, w, block), argnums=(0, 1))(hidden, lm_w)
    gh_n, gw_n = jax.grad(lambda h, m: naive_weighted_ce(h, m, tgt, w), argnums=(0, 1))(hidden, lm_w)
    dl = abs(lf - ln) / (abs(ln) + 1e-9)
    dh = float(jnp.max(jnp.abs(gh_f - gh_n))) / (float(jnp.max(jnp.abs(gh_n))) + 1e-12)
    dw = float(jnp.max(jnp.abs(gw_f - gw_n))) / (float(jnp.max(jnp.abs(gw_n))) + 1e-12)
    ok = dl < 1e-5 and dh < 1e-4 and dw < 1e-4
    fails += [] if ok else [f"block={block}"]
    print(f"[{'PASS' if ok else 'FAIL'}] block={block:>4}  dloss={dl:.2e}  dgrad_hidden={dh:.2e}  dgrad_W={dw:.2e}")

print("\n" + ("ALL PASS" if not fails else "FAILURES: " + ",".join(fails)))
sys.exit(1 if fails else 0)
