# coding=utf-8
"""
N-level rematerialising scan for a chunk recurrence.

The chunk recurrence is ``carry_{c+1} = step(carry_c, x_c)`` over ``C`` steps, and
exact reverse-mode needs the carry at every point where the backward re-enters.
The number of carries held simultaneously is what decides whether a step fits::

    M_carry = L_live * bytes(carry)

A plain scan keeps every step's carry: ``L_live = C``.  Nesting ``L`` levels
with factors ``f_1..f_L`` (``prod(f_i) = C``) and checkpointing EVERY level gives
``L_live ~ sum(f_i)`` at ``~L``x recompute.  The sum is minimised at ``f_i = e``,
i.e. factors of 2..4.  For ``C = 256`` that is ``L_live = 16`` at 4x recompute.

Every level must be checkpointed, including the root.  An earlier version
checkpointed only ``depth > 0``; the root scan then retained ``f_0 * f_1`` step
residuals and ``(8, 8)`` used *more* memory than a plain checkpointed scan
(measured 74 MiB vs 66 MiB; 18 MiB when fixed).  ``tests/test_remat_scan.py``
pins the fixed behaviour by compiled temp memory, since value tests cannot see it.

Why not full Revolve/treeverse (~9 carries for ``C = 256``)?  Its schedule is
non-uniform, so the step function is inlined ``O(C)`` times; for a 24-layer body
the resulting program is not compile-tractable.  Nested scans keep one copy.

This is rematerialisation, NOT gradient truncation: the meta-gradient is exact.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Sequence, Tuple

import jax
import jax.numpy as jnp

Carry = Any
PyTree = Any


def balanced_factors(n: int, levels: int) -> Tuple[int, ...]:
    """Factor ``n`` into ``levels`` integers whose product is ``n`` and whose sum is small."""
    if levels <= 1:
        return (n,)
    factors = []
    remaining = n
    for lvl in range(levels, 1, -1):
        target = max(2, round(remaining ** (1.0 / lvl)))
        pick = None
        # Walk outwards from `target` to the nearest divisor of `remaining`.
        for delta in range(0, remaining):
            for cand in (target - delta, target + delta):
                if 2 <= cand <= remaining and remaining % cand == 0:
                    pick = cand
                    break
            if pick is not None:
                break
        if pick is None or pick == remaining:
            break
        factors.append(pick)
        remaining //= pick
    factors.append(remaining)
    return tuple(f for f in factors if f > 1) or (n,)


def live_carries(factors: Sequence[int]) -> int:
    """Peak number of simultaneously held carries for a factorisation."""
    return sum(int(f) for f in factors)


def nested_remat_scan(
    step_fn: Callable[[Carry, PyTree], Tuple[Carry, PyTree]],
    init_carry: Carry,
    xs: PyTree,
    factors: Sequence[int],
    use_checkpoint: bool = True,
    stack_ys: bool = False,
) -> Tuple[Carry, PyTree]:
    """``jax.lax.scan(step_fn, init_carry, xs)`` over ``prod(factors)`` steps with
    ``len(factors)`` levels of rematerialisation.

    ``step_fn(carry, x) -> (carry, ys)``.  ``ys`` may be any pytree of arrays; by
    default each leaf is **summed** over the step axis at every level (so nothing
    per-step is stacked, and per-step statistics come out as totals); with
    ``stack_ys`` they are stacked like ``lax.scan``'s (``[prod(factors), ...]``).
    ``None`` leaves in ``xs`` are passed through unchanged.
    """
    factors = tuple(int(f) for f in factors)
    total = math.prod(factors)
    lead = jax.tree_util.tree_leaves(xs)[0].shape[0]
    if lead != total:
        raise ValueError(f"xs leading axis {lead} != prod(factors) {total}")

    xs_r = jax.tree_util.tree_map(
        lambda x: None if x is None else x.reshape(*factors, *x.shape[1:]),
        xs,
        is_leaf=lambda x: x is None,
    )

    def _sum_ys(ys):
        if stack_ys:
            return ys
        return jax.tree_util.tree_map(lambda y: jnp.sum(y, axis=0), ys)

    def level_body(depth: int):
        if depth == len(factors) - 1:
            body = step_fn
        else:
            child = level_body(depth + 1)

            def body(carry, x):
                carry, ys = jax.lax.scan(child, carry, x)
                return carry, _sum_ys(ys)

        # Checkpoint EVERY level, root included - see the module docstring.
        return jax.checkpoint(body) if use_checkpoint else body

    carry, ys = jax.lax.scan(level_body(0), init_carry, xs_r)
    if stack_ys:
        return carry, jax.tree_util.tree_map(lambda y: y.reshape(total, *y.shape[len(factors):]), ys)
    return carry, _sum_ys(ys)
