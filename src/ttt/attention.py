# coding=utf-8
"""
Attention kernels for the chunk/window geometry.

Geometry: bidirectional within a chunk of ``C`` tokens, causal across chunks, a
window of the previous ``W`` chunks.  A chunk's keys are ``concat([clean K/V of
chunks [c-W, c-1], own K/V])`` = ``(W+1)*C`` keys, and every query in the chunk sees
the same key set - the mask is **key-only** (``key_valid [K]``, false for window
slots before the sequence start).  Functions here work on plain arrays for one
sequence with a leading ``n`` axis; ``model.Attention`` owns the projections.

Implementations (``TTT_ATTENTION`` = ``auto`` | ``flash`` | ``cudnn`` | ``xla``; ``auto``
is ``flash`` on TPU, ``cudnn`` on GPU, ``xla`` elsewhere):

* ``attend_xla``   - einsum + f32 softmax with an additive ``[K]`` key bias.  The
  exact reference on any backend; differentiable to any order.  (Not ``where`` on
  the score tensor: under the layer scan XLA hoisted the ``where``'s full-size
  constant operand out of every remat region and kept four 1 GB copies live.)
* ``attend_flash`` - Pallas TPU flash attention with ``segment_ids`` as the mask.
  GQA is expressed by folding the ``rep = nh/nkv`` query heads of one KV head into
  the query axis (a key-only mask permits it) instead of repeating K/V.  First
  order only.
* ``attend_cudnn`` - cuDNN fused attention on GPU, the mask as a per-chunk valid key
  count (keys reordered so the invalid ones come last).  First order only; on GH200
  at the 1B geometry 5.6x faster than ``attend_xla`` forward + backward.
* ``attend_prefix_splash`` - the chunk-parallel prefix on the splash kernel with a
  static ``[T, 2T]`` block mask (``ChunkWindowMask``); invalid blocks are skipped.
* the inner-loss forward (differentiated by ``jax.grad`` for the fast-weight update
  and again by the meta-gradient) uses ``attend_xla`` on TPU: the Pallas kernels
  have no second-order rule.  A ``custom_vjp`` with the flash forward and a
  blockwise plain-JAX backward was tried (commit ``2ebe2fa``): under the
  meta-gradient XLA stacks the per-block residuals anyway and fragments the
  backward into 8x more ops - 24.5 s/step against 13.2 s for the plain XLA path.
  On GPU it uses ``attend_cudnn_inner``: cuDNN forward and first-order backward
  (whole kernels, no blockwise residuals), and the second-order term from the XLA
  reference's double VJP.

Sharding: XLA's SPMD partitioner cannot split a Mosaic kernel, so under a mesh
context (``jax.set_mesh``) every Pallas call is wrapped in ``shard_map`` with fully
replicated specs; the batch ``vmap`` in ``chunk_scan`` carries
``spmd_axis_name=AXIS`` so that its axis is the one that is sharded.

Every XLA / Pallas call is ``jax.checkpoint``ed: it sits inside ``lax.map``/``scan``
bodies whose backward would otherwise stack one residual set per chunk (24 GB per layer
at 8K).  cuDNN calls are not: their residuals are the output and softmax statistics.
"""
from __future__ import annotations

import os
from functools import lru_cache, partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, custom_batching
from jax.sharding import PartitionSpec as P
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as _splash_mask

# Read once from TTT_ATTENTION; resolved lazily at trace time so importing this
# module never initialises a backend.
ATTENTION_IMPL = os.environ.get("TTT_ATTENTION", "auto")


def resolve_attention_impl(impl: Optional[str] = None) -> str:
    impl = ATTENTION_IMPL if impl is None else impl
    if impl == "auto":
        impl = {"tpu": "flash", "gpu": "cudnn"}.get(jax.default_backend(), "xla")
    if impl not in ("xla", "flash", "cudnn"):
        raise ValueError(f"unknown attention impl {impl!r} (xla | flash | cudnn | auto)")
    return impl


def _block_size(n: int, cap: int = 1024) -> int:
    """Largest multiple of 128 that divides ``n`` and is at most ``cap`` (Pallas block).

    1024 x 1024 blocks measured fastest at the 1B geometry; 2048 exceeds VMEM.
    """
    for b in (1024, 512, 256, 128):
        if b <= cap and n % b == 0:
            return b
    raise ValueError(f"flash attention needs sequence lengths that are multiples of 128, got {n}")


def _sharded(f, n_in: int):
    """Wrap a Pallas call in ``shard_map`` when a mesh context is set (see module doc)."""
    if jax.sharding.get_abstract_mesh().empty:
        return f
    return jax.shard_map(f, in_specs=(P(),) * n_in, out_specs=P(), check_vma=False)


# ----------------------------------------------------------------------------
# window geometry
# ----------------------------------------------------------------------------
def window_gather(x: Array, chunk: int, window: int, prev: Optional[Array] = None) -> Array:
    """``x [T, ...]`` (per-token, chunk-ordered) -> ``[nc, window*chunk, ...]``:
    for chunk ``c``, the tokens of chunks ``[c-window, c)``.  ``prev [window*chunk, ...]``
    is what precedes ``x`` (the last ``window`` chunks of the previous segment);
    zeros when absent (before the sequence start - masked by the key validity).
    """
    T = x.shape[0]
    nc, WC = T // chunk, window * chunk
    pad = jnp.zeros((WC,) + x.shape[1:], x.dtype) if prev is None else prev
    xp = jnp.concatenate([pad, x], axis=0)                              # [WC + T, ...]
    idx = jnp.arange(nc)[:, None] * chunk + jnp.arange(WC)[None, :]      # [nc, WC]
    return xp[idx]


def key_validity(chunk_idx: Array, chunk: int, window: int, reach=None) -> Array:
    """``[ (window+1)*chunk ]`` bool: window slot ``j`` is valid iff its chunk
    ``chunk_idx - window + j // chunk`` exists and, with ``reach`` (an int or a traced scalar,
    0 .. window), is one of the newest ``reach`` window chunks; own-chunk slots are always
    valid.  The window is stored oldest first, so the valid keys stay a suffix of the window
    (a prefix after ``_own_first``'s reordering: the cuDNN length mask still applies)."""
    WC = window * chunk
    j = jnp.arange(WC)
    win = (chunk_idx - window + j // chunk) >= 0
    if reach is not None:
        win = win & (j // chunk >= window - reach)
    return jnp.concatenate([win, jnp.ones((chunk,), jnp.bool_)])


def key_validity_all(nc: int, chunk: int, window: int, first=0, reach=None) -> Array:
    """``[nc, (window+1)*chunk]`` - ``key_validity`` for chunks ``first .. first+nc-1``
    (``first`` > 0 for a segment that is not the first of its sequence)."""
    return jax.vmap(lambda c: key_validity(c, chunk, window, reach))(first + jnp.arange(nc))


# ----------------------------------------------------------------------------
# own-chunk attention patterns of the inner (write) stream
# ----------------------------------------------------------------------------
def own_mask_matrix(chunk: int, own: str) -> Optional[np.ndarray]:
    """``[C, C]`` bool: may query slot ``i`` of a chunk attend to key slot ``j`` of the
    same chunk?  ``full`` (bidirectional; None), ``causal`` (``j <= i``: the inner
    stream holds clean tokens and slot i predicts slot i+1), ``split:K`` (slots
    ``< K`` are context that sees only context; target slots ``>= K`` see all context
    and targets ``<= i``), ``block:b`` (``j // b <= i // b``: the clean stream of a sub-block
    read, ``ModelConfig.read_block``).  Slots are physical positions after the host-side permutation."""
    if own == "full":
        return None
    i = np.arange(chunk)[:, None]; j = np.arange(chunk)[None, :]
    if own == "causal":
        return j <= i
    if own.startswith("split:"):
        K = int(own.split(":")[1])
        return (j < K) | ((i >= K) & (j <= i))
    if own.startswith("block:"):
        b = int(own.split(":")[1])
        return (j // b) <= (i // b)
    raise ValueError(f"own-chunk mode {own!r}")


def read_mask_matrix(chunk: int, block: int) -> np.ndarray:
    """``[C, 2C]`` bool: the sub-block read (``ModelConfig.read_block`` = b) over its two own-chunk key
    sets, ``[clean K/V of the chunk, the read's own (noisy) K/V]``: query slot ``i`` sees the clean slots
    of the earlier groups (``j // b < i // b``) and the noisy slots of its own group (``j // b == i // b``)."""
    i = np.arange(chunk)[:, None] // block
    j = np.arange(chunk)[None, :] // block
    return np.concatenate([j < i, j == i], axis=1)


def _own_mask_fn(own: str, chunk: int):
    """The same pattern as ``own_mask_matrix`` as a function of in-chunk slot indices
    (numpy or jnp), for the splash kernel's static mask."""
    if own == "full":
        return lambda qi, ki: qi == qi
    if own == "causal":
        return lambda qi, ki: ki <= qi
    if own.startswith("split:"):
        K = int(own.split(":")[1])
        return lambda qi, ki: (ki < K) | ((qi >= K) & (ki <= qi))
    if own.startswith("block:"):
        b = int(own.split(":")[1])
        return lambda qi, ki: (ki // b) <= (qi // b)
    raise ValueError(f"own-chunk mode {own!r}")


# ----------------------------------------------------------------------------
# xla (reference)
# ----------------------------------------------------------------------------
def attend_xla(q: Array, k: Array, v: Array, key_valid: Array, scale: float, kv_heads: int,
               own_mask: Optional[Array] = None, mask: Optional[Array] = None) -> Array:
    """``q [n, C, nh, d], k/v [n, K, nkv, d], key_valid [n, K]`` -> ``[n, C, nh, d]``.
    ``own_mask [C, M]`` restricts the queries' view of the LAST ``M`` keys (their own chunk; ``M = 2C``:
    the sub-block read's clean and noisy own keys, ``read_mask_matrix``);
    ``mask [C, K]`` restricts it over all keys (the cuDNN path's reordered key layout)."""
    n, C, nh, d = q.shape
    rep = nh // kv_heads
    qg = q.reshape(n, C, kv_heads, rep, d)
    s = jnp.einsum("ncgrd,nkgd->ngrck", qg, k, preferred_element_type=jnp.float32) * scale
    s = s + jnp.where(key_valid, 0.0, -1e30).astype(jnp.float32)[:, None, None, None, :]
    if own_mask is not None:
        Kn, M = k.shape[1], own_mask.shape[1]
        bias = jnp.concatenate([jnp.zeros((C, Kn - M), jnp.float32),
                                jnp.where(jnp.asarray(own_mask), 0.0, -1e30).astype(jnp.float32)], axis=1)
        s = s + bias[None, None, None, :, :]
    if mask is not None:
        s = s + jnp.where(jnp.asarray(mask), 0.0, -1e30).astype(jnp.float32)[None, None, None]
    p = jax.nn.softmax(s, axis=-1).astype(v.dtype)
    o = jnp.einsum("ngrck,nkgd->ncgrd", p, v, preferred_element_type=jnp.float32)
    return o.reshape(n, C, nh, d).astype(q.dtype)


# ----------------------------------------------------------------------------
# flash (Pallas, first order)
# ----------------------------------------------------------------------------
def _fold_gqa(q, k, v, kv_heads):
    """``q [n, C, nh, d], k/v [n, K, nkv, d]`` -> kernel layout ``[n, nkv, rep*C, d]``,
    ``[n, nkv, K, d]``: the rep query heads of one KV head become extra query rows
    (query head h uses KV head h // rep, so head h = g*rep + r)."""
    n, C, nh, d = q.shape
    rep = nh // kv_heads
    qh = q.reshape(n, C, kv_heads, rep, d).transpose(0, 2, 3, 1, 4).reshape(n, kv_heads, rep * C, d)
    return qh, jnp.swapaxes(k, 1, 2), jnp.swapaxes(v, 1, 2), rep


def _unfold_gqa(oh, C, rep):
    n, nkv, _, d = oh.shape
    return oh.reshape(n, nkv, rep, C, d).transpose(0, 3, 1, 2, 4).reshape(n, C, nkv * rep, d)


def _flash_block_sizes(q_len, K):
    from jax.experimental.pallas.ops.tpu import flash_attention as fa
    bq, bk = _block_size(q_len), _block_size(K)
    return fa.BlockSizes(
        block_q=bq, block_k_major=bk, block_k=bk, block_b=1,
        block_q_major_dkv=bq, block_k_major_dkv=bk, block_k_dkv=bk, block_q_dkv=bq,
        block_k_major_dq=bk, block_k_dq=bk, block_q_dq=bq)


def attend_flash(q: Array, k: Array, v: Array, key_valid: Array, scale: float, kv_heads: int) -> Array:
    """Same contract as ``attend_xla``; Pallas flash attention, first order only."""
    from jax.experimental.pallas.ops.tpu import flash_attention as fa
    n, C = q.shape[:2]
    qh, kh, vh, rep = _fold_gqa(q, k, v, kv_heads)
    seg = fa.SegmentIds(q=jnp.ones((n, rep * C), jnp.int32), kv=key_valid.astype(jnp.int32))
    o = fa.flash_attention(qh, kh, vh, segment_ids=seg, sm_scale=scale,
                           block_sizes=_flash_block_sizes(rep * C, k.shape[1]))
    return _unfold_gqa(o, C, rep)


# ----------------------------------------------------------------------------
# cudnn (GPU, first order)
# ----------------------------------------------------------------------------
def cudnn_supports(dtype, head_dim: int) -> bool:
    """cuDNN fused attention takes half-precision inputs and a head dim that is a multiple
    of 8, at most 256; anything else (an f32 model, tests) takes the XLA path."""
    return jnp.dtype(dtype) in (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float16)) and head_dim % 8 == 0 \
        and head_dim <= 256


def _own_first(x: Array, C: int, M: Optional[int] = None) -> Array:
    """``[n, W*C + M, ...]`` keys (window oldest first, the ``M`` own-chunk keys: ``C``, or ``2C`` for the
    sub-block read) -> (own keys, window newest first): the window slots that do not exist yet come last."""
    M = C if M is None else M
    n, W = x.shape[0], (x.shape[1] - M) // C
    win = x[:, :W * C].reshape(n, W, C, *x.shape[2:])[:, ::-1].reshape(n, W * C, *x.shape[2:])
    return jnp.concatenate([x[:, W * C:], win], axis=1)


def _cudnn_core(q, k, v, lengths, scale):
    return jax.nn.dot_product_attention(q, k, v, scale=scale, key_value_seq_lengths=lengths, implementation="cudnn")


class _Own:
    """An own-chunk pattern ``[C, M]`` (causal, split, block; ``[C, 2C]``: the sub-block read) as a static,
    hashable value: the masked cuDNN functions are built per pattern.  ``keys(K)``: the ``[C, K]`` mask over
    the keys in ``_own_first`` order - the own keys first, the window (all of it visible) after."""

    def __init__(self, m):
        self.m = np.asarray(m, bool)
        self._key = (self.m.shape, self.m.tobytes())

    def __hash__(self):
        return hash(self._key)

    def __eq__(self, other):
        return isinstance(other, _Own) and self._key == other._key

    def keys(self, K: int) -> np.ndarray:
        C, M = self.m.shape
        return np.concatenate([self.m, np.ones((C, K - M), bool)], axis=1)


def _merge_vmap(f):
    """``f(*arrays)``, arguments with a leading batch axis, made safe under vmap / lax.map: a
    mapped axis is merged into that batch axis and ``f`` runs once.  JAX's own cuDNN batching
    rule reshapes the mask to a full ``(B, N, T, S)`` bias, which a broadcast mask cannot take
    (the model calls attention under a vmap over sequences and a lax.map over chunks).
    Never differentiated: it only runs inside custom_vjp rules."""
    g = custom_batching.custom_vmap(f)

    @g.def_vmap
    def _rule(axis_size, in_batched, *args):
        args = [a if b else jnp.broadcast_to(a, (axis_size,) + a.shape) for a, b in zip(args, in_batched)]
        n = args[0].shape[1]
        out = g(*(a.reshape((axis_size * n,) + a.shape[2:]) for a in args))
        out = jax.tree_util.tree_map(lambda o: o.reshape((axis_size, n) + o.shape[1:]), out)
        return out, jax.tree_util.tree_map(lambda _: True, out)
    return g


@lru_cache(maxsize=None)
def _masked_fns(scale: float, own: _Own):
    """cuDNN attention with the own-chunk pattern ``own``: forward and first-order backward (it
    recomputes the forward - cuDNN's softmax statistics stay inside JAX's rule), vmap-safe."""
    def raw(q, k, v, lengths):
        return jax.nn.dot_product_attention(q, k, v, scale=scale, key_value_seq_lengths=lengths,
                                            mask=jnp.asarray(own.keys(k.shape[1]))[None, None], implementation="cudnn")

    def vjp(q, k, v, lengths, do):
        return jax.vjp(lambda q, k, v: raw(q, k, v, lengths), q, k, v)[1](do)
    return _merge_vmap(raw), _merge_vmap(vjp)


@partial(jax.custom_vjp, nondiff_argnums=(4, 5))
def _masked_attn(q, k, v, lengths, scale, own):
    return _masked_fns(scale, own)[0](q, k, v, lengths)


def _masked_attn_fwd(q, k, v, lengths, scale, own):
    return _masked_fns(scale, own)[0](q, k, v, lengths), (q, k, v, lengths)


def _masked_attn_bwd(scale, own, res, do):
    q, k, v, lengths = res
    return (*_masked_fns(scale, own)[1](q, k, v, lengths, do), None)


_masked_attn.defvjp(_masked_attn_fwd, _masked_attn_bwd)


def attend_cudnn(q: Array, k: Array, v: Array, key_valid: Array, scale: float, kv_heads: int,
                 own_mask: Optional[Array] = None) -> Array:
    """Same contract as ``attend_flash``; cuDNN fused attention, first order only.

    cuDNN takes a mask only as a full ``[C, K]`` bias, which it must read, while a
    valid key count lets it skip whole blocks.  Every ``key_valid`` here comes from
    ``key_validity``: the own chunk is valid and the window is valid from some chunk
    on (the chunks before the sequence start are not).  With the keys reordered to
    (own chunk, window newest first) - attention does not depend on the key order,
    RoPE is already applied - the valid keys are a prefix, of length
    ``sum(key_valid)``.  GQA is native (``nkv`` divides ``nh``).  An own-chunk pattern
    (``own_mask [C, C]``: the causal / split inner streams, the block-causal clean stream; ``[C, 2C]``: the
    sub-block read over the chunk's clean and noisy keys) goes in as a boolean mask over
    the reordered keys: exact (tests/test_cudnn_attention.py), 5x faster per call than the XLA
    path at 1B/8K, 1.5x the unmasked call; the M2_shift recipe trains with the same losses as on
    XLA at 26.9 instead of 35.2 s/step (4 GH200, 2 seq/GPU)."""
    C = q.shape[1]
    M = C if own_mask is None else own_mask.shape[1]
    kr, vr, lengths = _own_first(k, C, M), _own_first(v, C, M), jnp.sum(key_valid, axis=1, dtype=jnp.int32)
    if own_mask is None:
        return _cudnn_core(q, kr, vr, lengths, scale)
    return _masked_attn(q, kr, vr, lengths, scale, _Own(own_mask))


# The inner-loss forward is differentiated twice: by the inner update (first order) and
# again by the meta-gradient.  Forward and first-order backward run on cuDNN; the
# backward of that backward - the second-order term - differentiates the XLA reference
# (exact, materialises the score tensor once).  Reverse mode only: forward-mode
# derivatives through it need TTT_ATTENTION=xla.
def _xla_core(q, k, v, lengths, scale, kv_heads, mask=None):
    return attend_xla(q, k, v, jnp.arange(k.shape[1])[None, :] < lengths[:, None], scale, kv_heads, mask=mask)


def _fwd_call(q, k, v, lengths, scale, own):
    # The meta-gradient differentiates _inner_attn_fwd itself: a custom_vjp here, never the custom_vmap.
    return _cudnn_core(q, k, v, lengths, scale) if own is None else _masked_attn(q, k, v, lengths, scale, own)


@partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6))
def _inner_attn(q, k, v, lengths, scale, kv_heads, own):
    return _fwd_call(q, k, v, lengths, scale, own)


def _inner_attn_fwd(q, k, v, lengths, scale, kv_heads, own):
    return _fwd_call(q, k, v, lengths, scale, own), (q, k, v, lengths)


def _inner_attn_bwd(scale, kv_heads, own, res, do):
    q, k, v, lengths = res
    return (*_inner_attn_vjp(q, k, v, lengths, do, scale, kv_heads, own), None)


@partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7))
def _inner_attn_vjp(q, k, v, lengths, do, scale, kv_heads, own):
    if own is None:
        return jax.vjp(lambda q, k, v: _cudnn_core(q, k, v, lengths, scale), q, k, v)[1](do)
    return _masked_fns(scale, own)[1](q, k, v, lengths, do)


def _inner_attn_vjp_fwd(q, k, v, lengths, do, scale, kv_heads, own):
    return _inner_attn_vjp(q, k, v, lengths, do, scale, kv_heads, own), (q, k, v, lengths, do)


def _inner_attn_vjp_bwd(scale, kv_heads, own, res, cts):
    q, k, v, lengths, do = res
    mask = None if own is None else jnp.asarray(own.keys(k.shape[1]))
    ref = lambda q, k, v, do: jax.vjp(lambda q, k, v: _xla_core(q, k, v, lengths, scale, kv_heads, mask), q, k, v)[1](do)
    dq, dk, dv, ddo = jax.vjp(ref, q, k, v, do)[1](tuple(cts))
    return dq, dk, dv, None, ddo


_inner_attn.defvjp(_inner_attn_fwd, _inner_attn_bwd)
_inner_attn_vjp.defvjp(_inner_attn_vjp_fwd, _inner_attn_vjp_bwd)


def attend_cudnn_inner(q: Array, k: Array, v: Array, key_valid: Array, scale: float, kv_heads: int,
                       own_mask: Optional[Array] = None) -> Array:
    """``attend_cudnn`` for the inner-loss forward: second order through the XLA reference."""
    C = q.shape[1]
    return _inner_attn(q, _own_first(k, C), _own_first(v, C), jnp.sum(key_valid, axis=1, dtype=jnp.int32),
                       scale, kv_heads, None if own_mask is None else _Own(own_mask))


# ----------------------------------------------------------------------------
# splash prefix (static chunk-window block mask)
# ----------------------------------------------------------------------------
class ChunkWindowMask(_splash_mask._ComputableMask):
    """Static ``[T, WC + 2T]`` block mask of the chunk-parallel prefix attention.

    Keys are ``concat([incoming window (WC = W*C, present iff with_window),
    clean K (T), own-stream K (T)])``.  Query chunk ``c`` sees window slots
    ``s >= c`` (slot ``s`` holds chunk ``c_0 - W + s`` of the previous segment), the
    clean chunks ``[c-W, c-1]`` of this segment and its own chunk ``c`` - the same
    key set the suffix's shift window provides, expressed over whole segments.
    Blocks that align with chunks are fully valid or fully invalid, so the splash
    kernel skips the invalid ones outright (at 8K with W=7 that is 44% of all
    blocks).  ``mask_function`` must work on numpy (mask processing at trace time)
    and jnp (partial blocks inside the kernel).  Whether the window slots exist at
    all (first segment) is dynamic and expressed with ``SegmentIds``.
    """

    def __init__(self, T: int, chunk: int, window: int, with_window: bool = False, own: str = "full"):
        self.T, self.chunk, self.window, self.with_window, self.own = T, chunk, window, with_window, own
        WC = window * chunk if with_window else 0
        WCp = window_pad(WC)                    # the window is padded to a 128-multiple; pad slots invalid
        own_ok = _own_mask_fn(own, chunk)

        def f(q_ids, kv_ids):
            qc = q_ids // chunk
            win = kv_ids < WC
            clean = (kv_ids >= WCp) & (kv_ids < WCp + T)
            own_k = kv_ids >= WCp + T
            s = kv_ids // chunk
            kc = (kv_ids - WCp - own_k * T) // chunk
            prev = (kc < qc) & (qc - kc <= window)
            same = own_k & (kc == qc)
            return (win & (s >= qc)) | (clean & prev) | (same & own_ok(q_ids % chunk, (kv_ids - WCp - T) % chunk))

        super().__init__(shape=(T, WCp + 2 * T), mask_function=f)

    def _key(self):
        return (self.T, self.chunk, self.window, self.with_window, self.own)

    def __eq__(self, other):
        return isinstance(other, ChunkWindowMask) and self._key() == other._key()

    def __hash__(self):
        return hash(self._key())


_SPLASH_KERNELS: dict = {}


def window_pad(WC: int) -> int:
    """Key-axis length reserved for an incoming window of ``WC`` slots (128-multiple)."""
    return -(-WC // 128) * 128


def splash_kernel(T: int, chunk: int, window: int, num_heads: int, with_window: bool = False, own: str = "full"):
    """Splash attention kernel for ``ChunkWindowMask``; cached per geometry (the
    mask -> block-sparsity processing runs once per process)."""
    key = (T, chunk, window, num_heads, with_window, own)
    if key not in _SPLASH_KERNELS:
        from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as sk
        K = 2 * T + (window_pad(window * chunk) if with_window else 0)
        bq, bkv = _block_size(T), _block_size(K)
        bs = sk.BlockSizes(block_q=bq, block_kv=bkv, block_kv_compute=min(bkv, 512),
                           block_q_dkv=bq, block_kv_dkv=bkv, block_kv_dkv_compute=min(bkv, 512),
                           block_q_dq=bq, block_kv_dq=bkv)
        mask = _splash_mask.MultiHeadMask([ChunkWindowMask(T, chunk, window, with_window, own)] * num_heads)
        # The kernel holds the block-sparsity tables as arrays; build them as concrete
        # values even when first requested inside a trace, so the cache never holds
        # tracers of one jit and leaks them into the next.
        with jax.ensure_compile_time_eval():
            _SPLASH_KERNELS[key] = sk.make_splash_mha(mask, block_sizes=bs, head_shards=1, q_seq_shards=1)
    return _SPLASH_KERNELS[key]


def attend_prefix_splash(q: Array, k_clean: Array, k_own: Array, v_clean: Array, v_own: Array,
                         chunk: int, window: int, scale: float,
                         kv_win: Optional[Tuple[Array, Array]] = None,
                         win_valid: Optional[Array] = None, own: str = "full") -> Array:
    """One stream of the chunk-parallel prefix: ``q [T, nh, d]``, ``k*/v* [T, nkv, d]``
    -> ``[T, nh, d]``.  Keys are the whole clean sequence followed by the whole
    own-stream sequence; ``ChunkWindowMask`` selects the window.  No gathered
    window, no per-chunk loop, and invalid blocks cost nothing.

    ``kv_win`` (``[W*C, nkv, d]`` x2) is the clean K/V of the previous segment's last
    ``W`` chunks, prepended to the keys; ``win_valid [W*C]`` says which of its slots
    exist (none before the sequence start) and enters the kernel as segment ids.
    """
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as sk
    T, nh, _ = q.shape
    kernel = splash_kernel(T, chunk, window, nh, with_window=kv_win is not None, own=own)

    def f(q, kc, ko, vc, vo):
        qh = jnp.swapaxes(q, 0, 1) * scale                                # splash has no sm_scale
        kh = jnp.swapaxes(jnp.concatenate([kc, ko], axis=0), 0, 1)        # [nkv, 2T, d]
        vh = jnp.swapaxes(jnp.concatenate([vc, vo], axis=0), 0, 1)
        return jnp.swapaxes(kernel(qh, kh, vh), 0, 1)                     # [T, nh, d]

    def f_win(q, kc, ko, vc, vo, kw, vw, wv):
        WC = kw.shape[0]
        padn = window_pad(WC) - WC
        pad = lambda x: jnp.concatenate([x, jnp.zeros((padn,) + x.shape[1:], x.dtype)], axis=0) if padn else x
        qh = jnp.swapaxes(q, 0, 1) * scale
        kh = jnp.swapaxes(jnp.concatenate([pad(kw), kc, ko], axis=0), 0, 1)   # [nkv, WCp + 2T, d]
        vh = jnp.swapaxes(jnp.concatenate([pad(vw), vc, vo], axis=0), 0, 1)
        seg = sk.SegmentIds(q=jnp.ones((T,), jnp.int32),
                            kv=jnp.concatenate([wv.astype(jnp.int32), jnp.zeros((padn,), jnp.int32),
                                                jnp.ones((2 * T,), jnp.int32)]))
        return jnp.swapaxes(kernel(qh, kh, vh, segment_ids=seg), 0, 1)

    if kv_win is None:
        return jax.checkpoint(_sharded(f, 5))(q, k_clean, k_own, v_clean, v_own)
    return jax.checkpoint(_sharded(f_win, 8))(q, k_clean, k_own, v_clean, v_own, kv_win[0], kv_win[1], win_valid)


# ----------------------------------------------------------------------------
# dispatch
# ----------------------------------------------------------------------------
def attend(q: Array, k: Array, v: Array, key_valid: Array, scale: float, kv_heads: int,
           impl: Optional[str] = None, inner: bool = False, own_mask: Optional[Array] = None) -> Array:
    """``q [n, C, nh, d], k/v [n, K, nkv, d], key_valid [n, K]`` -> ``[n, C, nh, d]``.

    ``impl`` overrides ``TTT_ATTENTION``.  ``inner`` marks the inner-loss forward,
    which is differentiated twice: XLA, or on cuDNN ``attend_cudnn_inner`` (second
    order through the XLA reference; reverse mode only).  ``own_mask [C, C]`` (inner
    stream with a causal / split own-chunk pattern) runs on cuDNN as a mask and takes the
    XLA path on flash.
    """
    impl = resolve_attention_impl(impl)
    if impl == "cudnn" and not cudnn_supports(q.dtype, q.shape[-1]):
        impl = "xla"
    if impl == "flash" and (inner or own_mask is not None):
        impl = "xla"
    if impl == "xla":
        f = lambda q, k, v, kv: attend_xla(q, k, v, kv, scale, kv_heads, own_mask)
    elif impl == "cudnn":
        f = (lambda q, k, v, kv: (attend_cudnn_inner if inner else attend_cudnn)(q, k, v, kv, scale, kv_heads,
                                                                                  own_mask))
    else:
        f = _sharded(lambda q, k, v, kv: attend_flash(q, k, v, kv, scale, kv_heads), 4)
    if impl == "cudnn":
        # cuDNN's residuals are the output and the softmax statistics, not a score
        # tensor: recomputing its forward in the backward would buy no memory.
        return f(q, k, v, key_valid)
    return jax.checkpoint(f)(q, k, v, key_valid)
