# coding=utf-8
"""Gated DeltaNet memory (the linear-attention baseline): the rematerialised token scan equals
the plain loop in value and gradient; the state is a constant within a chunk; sizes.

    JAX_PLATFORMS=cpu .venv/bin/python tests/test_gdn.py
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import equinox as eqx
import jax
import jax.numpy as jnp

from src.ttt.gdn import GDNMemory

failures = []


def report(name, ok, detail=""):
    failures.append(name) if not ok else None
    print(f"[{'PASS' if ok else 'FAIL'}] {name:<62} {detail}")


def main():
    H, nh, dk, dv, C = 32, 4, 8, 8, 64
    m = GDNMemory(H, nh, dk, dv, 0.02, jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (C, H))
    S0 = jax.random.normal(jax.random.PRNGKey(2), (nh, dv, dk)) * 0.1
    S_scan = m.update(x, S0)
    S_ref = m.update_reference(x, S0)
    report("remat scan == reference loop (state)", bool(jnp.allclose(S_scan, S_ref, atol=1e-5, rtol=1e-5)),
           f"max abs {float(jnp.max(jnp.abs(S_scan - S_ref))):.2e}")
    # gradients w.r.t. the module and the input through both paths
    def f_scan(mm, xx): return jnp.sum(jnp.square(mm.update(xx, S0)))
    def f_ref(mm, xx): return jnp.sum(jnp.square(mm.update_reference(xx, S0)))
    g1 = eqx.filter_grad(f_scan)(m, x); g2 = eqx.filter_grad(f_ref)(m, x)
    l1 = jax.tree_util.tree_leaves(eqx.filter(g1, eqx.is_array)); l2 = jax.tree_util.tree_leaves(eqx.filter(g2, eqx.is_array))
    rel = max(float(jnp.max(jnp.abs(a - b))) / (float(jnp.max(jnp.abs(b))) + 1e-12) for a, b in zip(l1, l2))
    gx1 = jax.grad(lambda xx: f_scan(m, xx))(x); gx2 = jax.grad(lambda xx: f_ref(m, xx))(x)
    relx = float(jnp.max(jnp.abs(gx1 - gx2))) / (float(jnp.max(jnp.abs(gx2))) + 1e-12)
    report("remat scan == reference loop (grads: params, input)", rel < 1e-4 and relx < 1e-4, f"{rel:.1e} / {relx:.1e}")
    # decay / write strength ranges at init; zero state reads to zero; read shape
    alpha = jnp.exp(-jax.nn.softplus(m.dt(x) + m.dt_bias) * jnp.exp(m.A_log))
    o0 = m.read(x, m.state0())
    report("init: alpha in (0, 1), zero state reads to 0, read shape [C, H]",
           bool(jnp.all((alpha > 0) & (alpha < 1))) and float(jnp.max(jnp.abs(o0))) == 0.0 and o0.shape == (C, H),
           f"alpha mean {float(alpha.mean()):.3f}")
    # chunk carry: updating in two halves equals one pass
    S_a = m.update(x[: C // 2], S0); S_b = m.update(x[C // 2:], S_a)
    report("two half-chunk updates == one chunk update", bool(jnp.allclose(S_b, S_ref, atol=1e-5)))
    # a state written with a token's (k, v) reads that token's v back (delta rule at beta=1 with unit keys)
    print("ALL PASS" if not failures else "FAILURES: " + "; ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
