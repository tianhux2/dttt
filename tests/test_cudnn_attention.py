# coding=utf-8
"""
cuDNN attention (GPU): ``attend`` with ``impl="cudnn"`` equals the XLA reference - forward and
first-order gradients - for every key-validity pattern of the chunk window; the inner-loss
forward (cuDNN + XLA second order) gives the XLA meta-gradient; own-chunk masks and f32 inputs
take the XLA path.

    JAX_PLATFORMS=cuda .venv/bin/python tests/test_cudnn_attention.py      (GPU only)
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import jax
import jax.numpy as jnp

from src.ttt.attention import attend, attend_xla, key_validity, own_mask_matrix, read_mask_matrix, resolve_attention_impl


def rel(a, b):
    a, b = a.astype(jnp.float32), b.astype(jnp.float32)
    return float(jnp.max(jnp.abs(a - b))) / (float(jnp.max(jnp.abs(b))) + 1e-12)


def main():
    if jax.default_backend() != "gpu":
        print("[SKIP] cuDNN attention needs a GPU")
        return 0
    failures = []

    def report(name, ok, detail=""):
        failures.extend([] if ok else [name])
        print(f"[{'PASS' if ok else 'FAIL'}] {name:58s} {detail}")

    # The 1B geometry and a small one; chunks 0 .. W+1 cover every validity pattern
    # (window wholly invalid, partly valid, wholly valid).
    for C, W, nh, nkv, d in ((1024, 3, 16, 4, 128), (128, 1, 8, 2, 64)):
        n, K, scale = W + 2, (W + 1) * C, d ** -0.5
        ks = jax.random.split(jax.random.PRNGKey(C), 4)
        q = jax.random.normal(ks[0], (n, C, nh, d), jnp.bfloat16)
        k = jax.random.normal(ks[1], (n, K, nkv, d), jnp.bfloat16)
        v = jax.random.normal(ks[2], (n, K, nkv, d), jnp.bfloat16)
        g = jax.random.normal(ks[3], (n, C, nh, d), jnp.bfloat16)
        kv = jnp.stack([key_validity(jnp.int32(c), C, W) for c in range(n)])
        o_x, vjp_x = jax.vjp(lambda q, k, v: attend(q, k, v, kv, scale, nkv, impl="xla"), q, k, v)
        o_c, vjp_c = jax.vjp(lambda q, k, v: attend(q, k, v, kv, scale, nkv, impl="cudnn"), q, k, v)
        e = rel(o_c, o_x)
        report(f"C={C} W={W} heads {nh}/{nkv} d={d}: forward == xla", e < 2e-2, f"max rel {e:.1e}")
        e = max(rel(a, b) for a, b in zip(vjp_c(g), vjp_x(g)))
        report(f"C={C} W={W} heads {nh}/{nkv} d={d}: dq, dk, dv == xla", e < 2e-2, f"max rel {e:.1e}")
        # The inner streams' own-chunk patterns (causal: M2_shift's write; split) and the sub-block read's
        # block-causal clean stream as a cuDNN mask.
        for own_name in ("causal", f"split:{C // 2}", f"block:{C // 4}"):
            own = own_mask_matrix(C, own_name)
            o_x, vjp_x = jax.vjp(lambda q, k, v: attend(q, k, v, kv, scale, nkv, impl="xla", own_mask=own), q, k, v)
            o_c, vjp_c = jax.vjp(lambda q, k, v: attend(q, k, v, kv, scale, nkv, impl="cudnn", own_mask=own), q, k, v)
            e = max([rel(o_c, o_x)] + [rel(a, b) for a, b in zip(vjp_c(g), vjp_x(g))])
            report(f"C={C} W={W}: own chunk {own_name}: forward, dq, dk, dv == xla", e < 2e-2, f"max rel {e:.1e}")
        # The sub-block read: [window, the chunk's clean K/V, the read's own] with the [C, 2C] pattern, for the
        # full window and for window reach 0 (the 2C own keys first, then the valid window prefix).
        k2 = jax.random.normal(ks[1], (n, K + C, nkv, d), jnp.bfloat16)
        v2 = jax.random.normal(ks[2], (n, K + C, nkv, d), jnp.bfloat16)
        own = read_mask_matrix(C, C // 4)
        for reach in (None, 0):
            kv2 = jnp.stack([jnp.concatenate([key_validity(jnp.int32(c), C, W, reach), jnp.ones((C,), bool)])
                             for c in range(n)])
            o_x, vjp_x = jax.vjp(lambda q, k, v: attend(q, k, v, kv2, scale, nkv, impl="xla", own_mask=own), q, k2, v2)
            o_c, vjp_c = jax.vjp(lambda q, k, v: attend(q, k, v, kv2, scale, nkv, impl="cudnn", own_mask=own), q, k2, v2)
            e = max([rel(o_c, o_x)] + [rel(a, b) for a, b in zip(vjp_c(g), vjp_x(g))])
            report(f"C={C} W={W}: sub-block read [C, 2C], reach {reach}: forward, grads == xla", e < 2e-2,
                   f"max rel {e:.1e}")

    report("TTT_ATTENTION=auto resolves to cudnn on GPU", resolve_attention_impl("auto") == "cudnn",
           resolve_attention_impl("auto"))

    # The inner-loss forward is differentiated twice (reverse over reverse, as the
    # meta-gradient does): an inner gradient step, then the outer loss's gradient
    # through it.  cuDNN forward / first order + XLA second order == XLA throughout.
    t1, t2 = (jax.random.normal(jax.random.PRNGKey(s), q.shape, jnp.float32) for s in (5, 6))

    def meta(impl, own=None):
        def inner_loss(q, k, v):
            return jnp.sum(attend(q, k, v, kv, scale, nkv, impl=impl, inner=True, own_mask=own).astype(jnp.float32) * t1)

        def outer(q, k, v):
            gq, gk, gv = jax.grad(inner_loss, argnums=(0, 1, 2))(q, k, v)
            q2, k2, v2 = (x - 0.05 * g.astype(x.dtype) for x, g in ((q, gq), (k, gk), (v, gv)))
            return jnp.sum(attend(q2, k2, v2, kv, scale, nkv, impl=impl, inner=True, own_mask=own).astype(jnp.float32) * t2)
        return jax.value_and_grad(outer, argnums=(0, 1, 2))(q, k, v)
    for own_name in ("full", "causal"):
        own = own_mask_matrix(C, own_name)
        # The loss is a sum of ~2e5 terms of both signs: its bf16 error is relative to the summed
        # magnitude, not to the (cancelled) sum - against an f32 truth both paths are off by ~3e-5 of it.
        mag = float(jnp.sum(jnp.abs(attend(q, k, v, kv, scale, nkv, impl="xla", own_mask=own).astype(jnp.float32) * t2)))
        try:
            (lc, gc), (lx, gx) = meta("cudnn", own), meta("xla", own)
            e = max(rel(a, b) for a, b in zip(gc, gx))
            ok, detail = abs(float(lc) - float(lx)) <= 1e-3 * mag and e < 3e-2, \
                f"loss {float(lc):.4f} vs {float(lx):.4f} (of |terms| {mag:.0f}), grads max rel {e:.1e}"
        except Exception as ex:                                      # noqa: BLE001
            ok, detail = False, f"{type(ex).__name__}: {str(ex)[:120]}"
        report(f"inner=True, own chunk {own_name}: meta-gradient (2nd order) == xla", ok, detail)

    # The model calls attention under a vmap over sequences and a lax.map over chunks: the masked
    # cuDNN path merges the mapped axis into its batch axis (JAX's own rule cannot take the mask).
    own = own_mask_matrix(C, "causal")
    qs, ks2, vs, gs = (jnp.stack([x, x[::-1]]) for x in (q, k, v, g))
    kvs = jnp.stack([kv, kv[::-1]])
    t1s, t2s = jnp.stack([t1, t1[::-1]]), jnp.stack([t2, t2[::-1]])

    def vm(impl, inner):
        return lambda q, k, v: jax.vmap(lambda q, k, v, kv: attend(q, k, v, kv, scale, nkv, impl=impl, inner=inner,
                                                                    own_mask=own))(q, k, v, kvs)
    for inner in (False, True):
        try:
            o_x, vjp_x = jax.vjp(vm("xla", inner), qs, ks2, vs)
            o_c, vjp_c = jax.vjp(vm("cudnn", inner), qs, ks2, vs)
            e = max([rel(o_c, o_x)] + [rel(a, b) for a, b in zip(vjp_c(gs), vjp_x(gs))])
            ok, detail = e < 2e-2, f"max rel {e:.1e}"
        except Exception as ex:                                      # noqa: BLE001
            ok, detail = False, f"{type(ex).__name__}: {str(ex)[:120]}"
        report(f"under vmap, inner={inner}, own chunk causal: forward, dq, dk, dv == xla", ok, detail)

    def meta_v(impl):
        f = vm(impl, True)

        def inner_loss(q, k, v):
            return jnp.sum(f(q, k, v).astype(jnp.float32) * t1s)

        def outer(q, k, v):
            gq, gk, gv = jax.grad(inner_loss, argnums=(0, 1, 2))(q, k, v)
            q2, k2, v2 = (x - 0.05 * g_.astype(x.dtype) for x, g_ in ((q, gq), (k, gk), (v, gv)))
            return jnp.sum(f(q2, k2, v2).astype(jnp.float32) * t2s)
        return jax.value_and_grad(outer, argnums=(0, 1, 2))(qs, ks2, vs)
    mag = float(jnp.sum(jnp.abs(vm("xla", True)(qs, ks2, vs).astype(jnp.float32) * t2s)))
    try:
        (lc, gc), (lx, gx) = meta_v("cudnn"), meta_v("xla")
        e = max(rel(a, b) for a, b in zip(gc, gx))
        ok, detail = abs(float(lc) - float(lx)) <= 1e-3 * mag and e < 3e-2, \
            f"loss {float(lc):.4f} vs {float(lx):.4f} (of |terms| {mag:.0f}), grads max rel {e:.1e}"
    except Exception as ex:                                          # noqa: BLE001
        ok, detail = False, f"{type(ex).__name__}: {str(ex)[:120]}"
    report("under vmap, own chunk causal: meta-gradient (2nd order) == xla", ok, detail)

    # cuDNN takes half precision only: an f32 model (tests, f32 tools) falls back to XLA.
    q32, k32, v32 = (x.astype(jnp.float32) for x in (q, k, v))
    e = rel(attend(q32, k32, v32, kv, scale, nkv, impl="cudnn"), attend_xla(q32, k32, v32, kv, scale, nkv))
    report("f32 inputs take the XLA path", e == 0.0, f"max rel {e:.1e}")

    print("ALL PASS" if not failures else "FAILURES: " + "; ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
