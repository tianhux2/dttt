# coding=utf-8
"""
Rematerialisation must actually save memory, and must not change values.

The value half is what every other test checks.  The memory half is the one that
an earlier version silently failed: checkpointing only ``depth > 0`` left the root
scan holding ``f_0 * f_1`` residuals, so ``(8, 8)`` used MORE than a plain scan.
We pin it with XLA's compiled memory analysis on a body whose carry is large
enough for residuals to dominate.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import jax
import jax.numpy as jnp

from src.ttt.remat_scan import balanced_factors, live_carries, nested_remat_scan

STEPS = 64
CARRY = 256 * 1024  # 1 MiB f32 carry


def _step(carry, x):
    carry = jnp.tanh(carry * 1.001 + x)
    return carry, {"ce": jnp.sum(carry), "n": jnp.float32(1.0)}


def _grad_fn(factors, use_checkpoint):
    def f(init, xs):
        carry, ys = nested_remat_scan(_step, init, xs, factors, use_checkpoint)
        return ys["ce"] + jnp.sum(carry)
    return jax.jit(jax.grad(f))


def _plain_checkpointed(init, xs):
    carry, ys = jax.lax.scan(jax.checkpoint(_step), init, xs)
    return ys["ce"].sum() + jnp.sum(carry)


def _temp_bytes(fn, *args):
    return fn.lower(*args).compile().memory_analysis().temp_size_in_bytes


def main():
    failures = []
    init = jnp.zeros((CARRY,), jnp.float32)
    xs = jnp.linspace(0.0, 1.0, STEPS)[:, None] * jnp.ones((1, CARRY)) * 1e-3

    ref = jax.jit(jax.grad(_plain_checkpointed))
    g_ref = ref(init, xs)
    mem_plain = _temp_bytes(ref, init, xs)

    # 1. Values: every factorisation matches a plain checkpointed scan.
    for f in [(64,), (8, 8), (4, 4, 4), (2, 4, 8)]:
        g = _grad_fn(f, True)(init, xs)
        r = float(jnp.max(jnp.abs(g - g_ref))) / (float(jnp.max(jnp.abs(g_ref))) + 1e-12)
        ok = r < 1e-6
        failures += [] if ok else [f"factors={f} changed the gradient ({r:.2e})"]
        print(f"[{'PASS' if ok else 'FAIL'}] factors={str(f):<12} grad == plain scan   max rel = {r:.3e}")

    # 2. Memory: (8,8) must hold ~16 carries, not 64.  The old bug gave ~1.1x plain.
    mem_88 = _temp_bytes(_grad_fn((8, 8), True), init, xs)
    ratio = mem_88 / mem_plain
    ok = ratio < 0.4
    failures += [] if ok else [f"(8,8) temp memory is {ratio:.2f}x plain - remat is not saving memory"]
    print(f"[{'PASS' if ok else 'FAIL'}] (8,8) temp memory            {mem_88/2**20:6.1f} MiB vs plain {mem_plain/2**20:6.1f} MiB  ({ratio:.2f}x)")

    # More levels are NOT monotonically cheaper on a trivial body: each level adds a
    # recompute working set, and with a 1 MiB carry and a tanh step that overhead
    # dominates the carry saving (measured (4,4,4) = 34 MiB > (8,8) = 26 MiB).
    # `sum(factors)` counts carries; it is the right model only when the step body's
    # activations dwarf the carry, as they do for a 6-layer transformer chunk.
    # Assert only that deeper levels still beat the plain scan.
    mem_444 = _temp_bytes(_grad_fn((4, 4, 4), True), init, xs)
    ok = mem_444 < 0.6 * mem_plain
    failures += [] if ok else [f"(4,4,4) temp memory is {mem_444/mem_plain:.2f}x plain"]
    print(f"[{'PASS' if ok else 'FAIL'}] (4,4,4) < 0.6x plain         {mem_444/2**20:6.1f} MiB  ({mem_444/mem_plain:.2f}x)")

    # 3. balanced_factors: product n, and the documented C=256 floor.
    for n, lv, want in [(256, 2, (16, 16)), (256, 4, (4, 4, 4, 4)), (8, 2, (2, 4)), (32, 2, (4, 8))]:
        got = balanced_factors(n, lv)
        ok = got == want and live_carries(got) == sum(want)
        failures += [] if ok else [f"balanced_factors({n},{lv}) = {got}, want {want}"]
        print(f"[{'PASS' if ok else 'FAIL'}] balanced_factors({n:>3},{lv}) = {got}")

    # stack_ys: per-step outputs come out stacked like lax.scan's, at every depth.
    init, xs = jnp.zeros((64,)), jnp.linspace(-1, 1, STEPS)[:, None] * jnp.ones((STEPS, 64))
    _, ys_ref = jax.lax.scan(_step, init, xs)
    worst = 0.0
    for factors in ((64,), (8, 8), (4, 4, 4)):
        f = jax.jit(lambda i, x: nested_remat_scan(_step, i, x, factors, stack_ys=True))
        _, ys = f(init, xs)
        g = jax.jit(jax.grad(lambda i, x: jnp.sum(nested_remat_scan(_step, i, x, factors, stack_ys=True)[1]["ce"])))
        g_ref = jax.grad(lambda i, x: jnp.sum(jax.lax.scan(_step, i, x)[1]["ce"]))
        worst = max(worst, float(jnp.max(jnp.abs(ys["ce"] - ys_ref["ce"]))),
                    float(jnp.max(jnp.abs(g(init, xs) - g_ref(init, xs)))))
    ok = worst < 1e-5
    failures += [] if ok else [f"stack_ys differs from lax.scan ({worst:.2e})"]
    print(f"[{'PASS' if ok else 'FAIL'}] stack_ys == lax.scan (1/2/3 levels) max abs = {worst:.2e}")

    print("ALL PASS" if not failures else "FAILURES: " + "; ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
