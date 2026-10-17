# coding=utf-8
"""
The masked-diffusion objective and its inner-loop counterpart.

Both are the same functional: mask the input, predict the token at each masked
position (offset 0), score ONLY where masked.  Leak-freedom under bidirectional
attention comes from the masking, not from causality - position ``t`` can read
every other position, but its own answer was destroyed by the mask.

Outer loss (per sequence noise level ``t ~ U[0,1)``, diffusion time-weighting):

    L = sum_{chunks} sum_i w_i * CE_i / (sum_{chunks} |t_mask| + 1e-6)

Inner loss (per chunk, per sequence, own mask draw, plain mean):

    L_inner = mean_{i in inner_mask} CE_i

The inner mask ratio is SAMPLED per chunk per sequence from ``inner_mask_range``
(default skewed high).  Two reasons: the fast weight is read by an outer forward
whose noise level is ``U[0,1)``, so training it at one fixed ratio trains it where
it is never read; and a diffusion loss can only score masked positions, so the
ratio IS the supervised-token count.  ``inner_mask_ratio >= 0`` pins it instead,
consuming the key exactly as a pinned draw should so ablations are reproducible.
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array

from .fused_ce import fused_soft_ce, fused_weighted_ce, naive_soft_ce, naive_weighted_ce


# ----------------------------------------------------------------------------
# chunking
# ----------------------------------------------------------------------------
def to_chunks(x: Array, chunk: int) -> Array:
    """``[B, T, ...]`` -> ``[nc, B, C, ...]`` (chunk-major, so a scan iterates chunks)."""
    B, T = x.shape[:2]
    return jnp.swapaxes(x.reshape(B, T // chunk, chunk, *x.shape[2:]), 0, 1)


def from_chunks(x: Array) -> Array:
    """Inverse of ``to_chunks``: ``[nc, B, C, ...]`` -> ``[B, T, ...]``."""
    nc, B, C = x.shape[:3]
    return jnp.swapaxes(x, 0, 1).reshape(B, nc * C, *x.shape[3:])


# ----------------------------------------------------------------------------
# outer mask
# ----------------------------------------------------------------------------
def q_sample(key: Array, ids: Array, loss_mask: Optional[Array], mask_token_id: int,
             t: Optional[Array] = None) -> Tuple[Array, Array, Array]:
    """Diffusion forward noising.

    ``t ~ U[0,1)`` per sequence, ``u ~ U[0,1)`` per token, ``t_mask = (u < t) & loss_mask``.
    Supports row-wise keys ``[B, 2]`` for batch-shape/seed decoupled sampling, and explicit ``t [B]``.
    Returns ``(x_t [B,T], t [B], t_mask [B,T])``.
    """
    B, T = ids.shape
    if key.ndim == 2:
        split_v = jax.vmap(jax.random.split)(key)
        key_t, key_u = split_v[:, 0, :], split_v[:, 1, :]
        if t is None:
            t = jax.vmap(lambda k: jax.random.uniform(k, ()))(key_t)
        u = jax.vmap(lambda k: jax.random.uniform(k, (T,)))(key_u)
    else:
        key_t, key_u = jax.random.split(key)
        if t is None:
            t = jax.random.uniform(key_t, (B,))
        u = jax.random.uniform(key_u, (B, T))
    t_mask = u < t[:, None]
    if loss_mask is not None:
        t_mask = t_mask & loss_mask.astype(jnp.bool_)
    return jnp.where(t_mask, mask_token_id, ids), t, t_mask


def cart_kernel(p: float = 0.1, window: int = 512) -> Array:
    """CART time-reweighting kernel ``[1, 1, 2*window+1]``: ``0.5*p*(1-p)^(|d|-1)``, 0 at d=0.

    A token's weight rises with the density of VISIBLE neighbours around it.
    """
    d = jnp.abs(jnp.arange(-window, window + 1)).astype(jnp.float32)
    w = 0.5 * p * jnp.power(1.0 - p, jnp.maximum(d - 1.0, 0.0))
    w = jnp.where(d == 0, 0.0, w)
    return w[None, None, :]


def band_correlate(x: Array, kernel: Array) -> Array:
    """``out[b, t] = sum_d kernel[d + W] * x[b, t + d]`` for ``d`` in ``[-W, W]`` (zero
    padded) - a 1-D "SAME" cross-correlation with a ``[2W+1]`` kernel - as block
    matmuls: blocks of ``W`` positions, each output block reading its three
    neighbouring input blocks through a ``[3W, W]`` band matrix.  Numerically the
    same as ``lax.conv_general_dilated``, which XLA:TPU lowers through its
    convolution emitter; for this 1025-tap f32 shape the emitter took >2 min per
    fusion at 1 seq/chip (the whole step compiled in 26 min instead of 4)."""
    B, T = x.shape
    W = (kernel.shape[0] - 1) // 2
    nb = -(-T // W)
    xb = jnp.pad(x, ((0, 0), (0, nb * W - T))).reshape(B, nb, W)
    xp = jnp.pad(xb, ((0, 0), (1, 1), (0, 0)))                                     # [B, nb+2, W]
    x3 = jnp.concatenate([xp[:, :-2], xp[:, 1:-1], xp[:, 2:]], axis=-1)              # [B, nb, 3W]
    i = jnp.arange(3 * W)[:, None]                                                 # input offset in the 3 blocks
    o = jnp.arange(W)[None, :]                                                     # output offset in the middle block
    d = i - (o + W)                                                                # signed distance input - output
    band = jnp.where(jnp.abs(d) <= W, kernel[jnp.clip(d + W, 0, 2 * W)], 0.0)      # [3W, W]
    return jnp.einsum("bnk,ko->bno", x3, band, precision=jax.lax.Precision.HIGHEST).reshape(B, nb * W)[:, :T]


def time_weights(mode: str, t: Array, t_mask: Array, kernel: Optional[Array]) -> Array:
    """Per-token outer weights ``[B, T]`` (f32), zero off ``t_mask``."""
    tm = t_mask.astype(jnp.float32)
    if mode == "cart":
        w = band_correlate(1.0 - tm, kernel[0, 0])                       # density of visible neighbours
    elif mode == "linear":
        w = jnp.broadcast_to((1.0 - t)[:, None], tm.shape)
    elif mode == "original":
        w = jnp.broadcast_to((1.0 / jnp.maximum(t, 1e-6))[:, None], tm.shape)
    elif mode == "none":
        w = jnp.ones_like(tm)
    else:
        raise ValueError(f"time_reweighting {mode!r}")
    return w * tm


# ----------------------------------------------------------------------------
# inner mask
# ----------------------------------------------------------------------------
def span_mask(key: Array, n: int, ratio: Array, span: float) -> Array:
    """``[n]`` bool: spans of geometric length (mean ``span``) covering ~``ratio`` of the
    positions.  A span starts at position ``i`` with probability ``ratio / span``;
    position ``j`` is masked iff some span starting at ``i <= j`` reaches it
    (running max of the span ends).  Overlaps make the coverage slightly below ``ratio``.
    """
    k_start, k_len = jax.random.split(key)
    starts = jax.random.uniform(k_start, (n,)) < ratio / span
    lengths = 1 + jnp.floor(jnp.log(jax.random.uniform(k_len, (n,), minval=1e-6)) /
                            jnp.log(1.0 - 1.0 / span)).astype(jnp.int32)          # geometric, mean ~ span
    ends = jnp.where(starts, jnp.arange(n) + jnp.minimum(lengths, int(8 * span)), 0)
    return jax.lax.cummax(ends, axis=0) > jnp.arange(n)


def make_inner_mask(ratio: float, ratio_range: Tuple[float, float], mask_token_id: int,
                    pad_token_id: int, views: int = 1, span: float = 0.0, view_mode: str = "complement"
                    ) -> Callable[[Array, Array], Tuple[Array, Array]]:
    """Return ``f(chunk [C], key) -> (x_inner, inner_mask)``: ``[C]`` each for one view,
    ``[views, C]`` for more.

    Pinned (``ratio >= 0``): the key is used directly for the token draw.
    Sampled: ``k_ratio, k_draw = split(key)``; ``ratio ~ U(lo, hi)``.
    ``views = 2``: the second view is the complement of the first (every non-pad
    token is masked in exactly one view), so the averaged inner gradient covers the
    whole chunk without the per-chunk lottery of one draw.  ``span > 0``: contiguous
    spans of that mean length instead of i.i.d. tokens (``span_mask``).
    Pad tokens are never masked (note: in ``1b_e2e`` ``pad == eos``, so EOS is
    outer-masked but never inner-masked - documented, deliberate).
    """
    lo, hi = float(ratio_range[0]), float(ratio_range[1])
    pinned = ratio is not None and ratio >= 0.0
    if view_mode not in ("complement", "strata"):
        raise ValueError(f"inner view mode {view_mode!r}")
    if view_mode == "complement" and views not in (1, 2):
        raise ValueError(f"complementary inner views must be 1 or 2, got {views}")
    if view_mode == "strata" and views < 2:
        raise ValueError("stratified inner views need views >= 2")

    def strata(chunk: Array, key: Array) -> Tuple[Array, Array]:
        # ``views`` independent draws whose ratios stratify (lo, hi): view k (in a random
        # order) has r_k = lo + (hi - lo) * (k + u) / views with one u ~ U(0, 1) - the
        # write sees every masking density, as the reader's t ~ U does.
        k_u, k_perm, k_draw = jax.random.split(key, 3)
        u = jax.random.uniform(k_u, ())
        order = jax.random.permutation(k_perm, views)
        r = lo + (hi - lo) * (order.astype(jnp.float32) + u) / views                              # [V]
        not_pad = chunk != pad_token_id
        keys = jax.random.split(k_draw, views)
        if span > 0:
            ms = jax.vmap(lambda k, rr: span_mask(k, chunk.shape[0], rr, span))(keys, r) & not_pad
        else:
            ms = (jax.random.uniform(k_draw, (views,) + chunk.shape) < r[:, None]) & not_pad
        return jnp.where(ms, mask_token_id, chunk[None]), ms

    if view_mode == "strata":
        return strata

    def f(chunk: Array, key: Array) -> Tuple[Array, Array]:
        if pinned:
            r, k_draw = jnp.float32(ratio), key
        else:
            k_ratio, k_draw = jax.random.split(key)
            r = jax.random.uniform(k_ratio, (), minval=lo, maxval=hi)
        not_pad = chunk != pad_token_id
        if span > 0:
            m = span_mask(k_draw, chunk.shape[0], r, span) & not_pad
        else:
            m = (jax.random.uniform(k_draw, chunk.shape) < r) & not_pad
        if views == 1:
            return jnp.where(m, mask_token_id, chunk), m
        ms = jnp.stack([m, ~m & not_pad])
        return jnp.where(ms, mask_token_id, chunk[None]), ms

    return f


# ----------------------------------------------------------------------------
# cross-entropy
# ----------------------------------------------------------------------------
def weighted_ce(hidden: Array, lm_w: Array, targets: Array, weights: Array, block: int) -> Array:
    """``sum_i w_i * CE(logits_i, targets_i)`` over ``hidden [N, H]``.

    ``block > 0`` streams the vocabulary through ``fused_weighted_ce`` (never
    materialises ``[N, V]``); ``block == 0`` is the dense reference.
    """
    if block > 0:
        return fused_weighted_ce(hidden, lm_w, targets, weights, block)
    return naive_weighted_ce(hidden, lm_w, targets, weights)


def soft_weighted_ce(hidden: Array, lm_w: Array, target_ids: Array, target_probs: Array, weights: Array,
                     block: int) -> Array:
    """``sum_i w_i * sum_k q_ik * CE(logits_i, target_ik)`` - the soft cross-entropy
    against a sparse target distribution (``target_ids / target_probs [N, K]``, a
    teacher's renormalised top-k); ``block`` as in ``weighted_ce``."""
    if block > 0:
        return fused_soft_ce(hidden, lm_w, target_ids, target_probs, weights, block)
    return naive_soft_ce(hidden, lm_w, target_ids, target_probs, weights)
