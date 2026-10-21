# coding=utf-8
"""
The transformer.  Single sequence ``[T, ...]``; batching is ``eqx.filter_vmap`` outside.

Layout
------
* Every ``Linear`` weight is ``[in, out]`` and applied as ``x @ w``.
* Layers are **vmap-stacked**: one ``Block`` whose every array leaf has a leading
  layer axis.  The model holds two stacks, ``prefix: Block[L-S]`` (no fast weight)
  and ``suffix: Block[S]`` (fast weight), and iterates each with ``lax.scan``.  One
  compile trace per stack, one sharding spec per parameter kind.
* The fast weight ``W`` is a ``FastWeight[F]`` (three SwiGLU matrices per suffix
  layer; F = S unless ``suffix_plain_top`` leaves the top suffix layers plain, then
  the lowest F suffix layers carry it and the plain ones above run in a second scan,
  still chunk by chunk).  ``Model.fast_weight`` is the ``W_0`` template, an ordinary outer
  parameter; the LIVE per-sequence ``W`` is threaded into ``suffix_forward`` and
  lives in the training scan's carry.  The prime pre/post norms are outer
  parameters on the model, never in the carry.

Attention geometry
------------------
Bidirectional within a chunk, causal across chunks, window of ``W`` previous
chunks.  A chunk's keys are ``concat([clean K/V of chunks [c-W, c-1], own K/V])``,
so every query in a chunk sees the same key set and the mask is key-only.

The prefix runs **chunk-parallel**: prefix layers have no fast weight, so the
clean K/V of every chunk is a pure function of the clean tokens, and all three
input streams (outer-masked, clean, inner-masked) for all chunks can be computed
at once from one gathered window.  The suffix runs chunk-by-chunk (``W_c`` is
sequential) with a shift-window KV cache in the carry.  The kernels themselves
(XLA reference, Pallas flash/splash, the second-order-safe inner-loss variant)
live in ``attention.py``; ``Attention`` here owns only the projections.
"""
from __future__ import annotations

import math
import os
from typing import Any, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from . import attention
from .attention import key_validity, key_validity_all, own_mask_matrix, read_mask_matrix, resolve_attention_impl, window_gather  # noqa: F401 (re-exported)
from .config import ModelConfig
from .gdn import GDNMemory
from .remat_scan import balanced_factors, nested_remat_scan
from .sharding import constrain_local, replicate_in_context

# TTT_SUFFIX_LAYER_REMAT=0: inside the chunk-level remat, keep the suffix layers' activations
# instead of recomputing every layer once more in the backward.  The checkpoint is what fits
# a TPU v4 chip; on a 96 GB GH200 at 1B/8K, 2 seq/GPU, dropping it costs +10 GiB and saves
# 4.5% of the step.  Read once at import, like TTT_ATTENTION.
SUFFIX_LAYER_REMAT = os.environ.get("TTT_SUFFIX_LAYER_REMAT", "1") != "0"
_suffix_ck = jax.checkpoint if SUFFIX_LAYER_REMAT else (lambda f: f)

# Stream indices for the prefix's three parallel inputs.
OUTER, CLEAN, INNER = 0, 1, 2          # prefix streams; inner views occupy INNER, INNER+1, ...

# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def cast_floating(tree, dtype):
    """Cast every floating-point array leaf of ``tree`` to ``dtype``."""
    return jax.tree_util.tree_map(
        lambda x: x.astype(dtype) if eqx.is_inexact_array(x) else x, tree)


def stack_modules(make, keys: Array):
    """Build ``keys.shape[0]`` modules and stack every array leaf on a new axis 0."""
    return jax.vmap(make)(keys)


def slice_stack(stacked, sl: slice):
    """Slice the leading (layer) axis of every array leaf of a stacked module."""
    return jax.tree_util.tree_map(lambda x: x[sl], stacked)


def llama3_inv_freq(inv_freq: Array, scaling) -> Array:
    """Llama-3 RoPE scaling (HF ``_compute_llama3_parameters``): wavelengths shorter than
    ``original / high_freq_factor`` keep their frequency, longer than ``original / low_freq_factor``
    are slowed by ``factor``, the band between is interpolated."""
    s = dict(scaling)
    factor, low, high, old = (s["factor"], s["low_freq_factor"], s["high_freq_factor"],
                              s["original_max_position_embeddings"])
    wavelen = 2 * math.pi / inv_freq
    scaled = jnp.where(wavelen > old / low, inv_freq / factor, inv_freq)
    smooth = (old / wavelen - low) / (high - low)
    smoothed = (1 - smooth) * scaled / factor + smooth * scaled
    medium = ~(wavelen < old / high) & ~(wavelen > old / low)
    return jnp.where(medium, smoothed, scaled)


def rope_cos_sin(pos: Array, head_dim: int, theta: float, scaling=None) -> Tuple[Array, Array]:
    """Rotary tables for absolute positions ``pos [T]`` -> ``cos, sin [T, head_dim]`` (f32).
    ``scaling``: ``ModelConfig.rope_scaling`` (llama3), None for plain RoPE.

    No ``inv_freq`` parameter leaf: nothing to exclude from the optimiser.
    """
    inv_freq = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    if scaling is not None:
        inv_freq = llama3_inv_freq(inv_freq, scaling)
    freqs = pos.astype(jnp.float32)[:, None] * inv_freq[None, :]     # [T, d/2]
    emb = jnp.concatenate([freqs, freqs], axis=-1)                     # [T, d]
    return jnp.cos(emb), jnp.sin(emb)


def apply_rope(x: Array, cos: Array, sin: Array) -> Array:
    """``x [T, heads, d]`` rotated in f32, returned in ``x.dtype``."""
    x32 = x.astype(jnp.float32)
    half = x32.shape[-1] // 2
    rot = jnp.concatenate([-x32[..., half:], x32[..., :half]], axis=-1)
    out = x32 * cos[:, None, :] + rot * sin[:, None, :]
    return out.astype(x.dtype)


# ----------------------------------------------------------------------------
# primitives
# ----------------------------------------------------------------------------
class RMSNorm(eqx.Module):
    weight: Array
    eps: float = eqx.field(static=True)

    def __init__(self, dim: int, eps: float):
        self.weight = jnp.ones((dim,), jnp.float32)
        self.eps = eps

    def __call__(self, x: Array) -> Array:
        x32 = x.astype(jnp.float32)
        var = jnp.mean(jnp.square(x32), axis=-1, keepdims=True)
        return (self.weight * (x32 * jax.lax.rsqrt(var + self.eps))).astype(x.dtype)


class Linear(eqx.Module):
    """``[in, out]`` weight, ``x @ w (+ b)``; ``normal(std)`` init, zero bias."""
    weight: Array
    bias: Optional[Array]

    def __init__(self, in_dim: int, out_dim: int, std: float, key: Array, bias: bool = False):
        self.weight = jax.random.normal(key, (in_dim, out_dim), jnp.float32) * std
        self.bias = jnp.zeros((out_dim,), jnp.float32) if bias else None

    def __call__(self, x: Array) -> Array:
        y = x @ self.weight.astype(x.dtype)
        return y if self.bias is None else y + self.bias.astype(x.dtype)


def _act(name: str):
    return {"silu": jax.nn.silu, "gelu": jax.nn.gelu}[name]


class MLP(eqx.Module):
    """SwiGLU: ``down(act(gate(x)) * up(x))``."""
    gate: Linear
    up: Linear
    down: Linear
    act: Any = eqx.field(static=True)

    def __init__(self, dim: int, inter: int, act: str, std: float, key: Array):
        k1, k2, k3 = jax.random.split(key, 3)
        self.gate = Linear(dim, inter, std, k1)
        self.up = Linear(dim, inter, std, k2)
        self.down = Linear(inter, dim, std, k3)
        self.act = _act(act)

    def __call__(self, x: Array) -> Array:
        return self.down(self.act(self.gate(x)) * self.up(x))


class FastWeight(MLP):
    """The TTT fast weight: a SwiGLU MLP at ``ttt_intermediate_size``.

    Structurally an ``MLP``; a distinct type so the carry is self-describing.
    All three matrices are inner-updated, so none may be zero-initialised: with
    ``down = 0`` the gate/up gradients vanish identically and never recover.

    This is 3x LARGER per layer than a single ``[H, I]`` linear fast weight, not
    smaller - both scale as ``H*I``.  It is adopted for expressivity (a fast weight
    linear in its input cannot update the features it reads; 2608.30692 shows that
    degenerating to that form destroys extrapolation), not for memory.
    """


# ----------------------------------------------------------------------------
# attention
# ----------------------------------------------------------------------------
class Attention(eqx.Module):
    q: Linear
    k: Linear
    v: Linear
    o: Linear
    q_norm: Optional[RMSNorm]
    k_norm: Optional[RMSNorm]
    num_heads: int = eqx.field(static=True)
    kv_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    rope_theta: float = eqx.field(static=True)
    rope_scaling: Any = eqx.field(static=True)

    def __init__(self, cfg: ModelConfig, key: Array):
        H, nh, nkv, d = cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        k1, k2, k3, k4 = jax.random.split(key, 4)
        std, b = cfg.initializer_range, cfg.attention_bias
        self.q = Linear(H, nh * d, std, k1, bias=b)
        self.k = Linear(H, nkv * d, std, k2, bias=b)
        self.v = Linear(H, nkv * d, std, k3, bias=b)
        self.o = Linear(nh * d, H, std, k4)
        # QK-norm (Qwen3 / Gemma-3): per-head RMSNorm before RoPE.  It is the
        # mechanism that replaced q/k/v biases, and it matters more here than
        # usual because an SGD-updated fast weight writes into the residual.
        self.q_norm = RMSNorm(d, cfg.rms_norm_eps) if cfg.qk_norm else None
        self.k_norm = RMSNorm(d, cfg.rms_norm_eps) if cfg.qk_norm else None
        self.num_heads, self.kv_heads, self.head_dim, self.rope_theta = nh, nkv, d, cfg.rope_theta
        self.rope_scaling = cfg.rope_scaling

    def qkv(self, x: Array, pos: Array, pos_q: Optional[Array] = None) -> Tuple[Array, Array, Array]:
        """``x [T, H], pos [T]`` -> ``q [T, nh, d], k [T, nkv, d], v [T, nkv, d]`` (RoPE applied).
        ``pos_q``: the queries' positions when they differ from the keys' (the inner
        stream's shifted-target query "stands at" the position it predicts)."""
        T = x.shape[0]
        q = self.q(x).reshape(T, self.num_heads, self.head_dim)
        k = self.k(x).reshape(T, self.kv_heads, self.head_dim)
        v = self.v(x).reshape(T, self.kv_heads, self.head_dim)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        cos, sin = rope_cos_sin(pos, self.head_dim, self.rope_theta, self.rope_scaling)
        if pos_q is None:
            return apply_rope(q, cos, sin), apply_rope(k, cos, sin), v
        cos_q, sin_q = rope_cos_sin(pos_q, self.head_dim, self.rope_theta, self.rope_scaling)
        return apply_rope(q, cos_q, sin_q), apply_rope(k, cos, sin), v

    def attend(self, q: Array, k: Array, v: Array, key_valid: Array, impl: Optional[str] = None,
               inner: bool = False, own_mask: Optional[Array] = None) -> Array:
        """``q [n, C, nh, d], k/v [n, K, nkv, d], key_valid [n, K]`` -> ``[n, C, nh*d]``.
        Kernel selection and semantics: ``attention.attend``."""
        n, C = q.shape[:2]
        a = attention.attend(q, k, v, key_valid, self.scale, self.kv_heads, impl, inner, own_mask)
        return a.reshape(n, C, self.num_heads * self.head_dim)

    def attend_prefix(self, q: Array, k_clean: Array, k_own: Array, v_clean: Array, v_own: Array,
                      chunk: int, window: int, kv_win=None, win_valid=None, own: str = "full") -> Array:
        """One stream of the chunk-parallel prefix on the splash kernel: ``q [T, nh, d]``,
        ``k*/v* [T, nkv, d]`` -> ``[T, nh*d]``; see ``attention.attend_prefix_splash``."""
        a = attention.attend_prefix_splash(q, k_clean, k_own, v_clean, v_own, chunk, window, self.scale,
                                           kv_win, win_valid, own)
        return a.reshape(q.shape[0], self.num_heads * self.head_dim)

    @property
    def scale(self) -> float:
        return 1.0 / math.sqrt(self.head_dim)

    def out(self, a: Array) -> Array:
        return self.o(a)


# ----------------------------------------------------------------------------
# block
# ----------------------------------------------------------------------------
class Block(eqx.Module):
    attn_norm: RMSNorm
    attn: Attention
    mlp_norm: RMSNorm
    mlp: MLP

    def __init__(self, cfg: ModelConfig, key: Array):
        ka, km = jax.random.split(key)
        H = cfg.hidden_size
        self.attn_norm = RMSNorm(H, cfg.rms_norm_eps)
        self.attn = Attention(cfg, ka)
        self.mlp_norm = RMSNorm(H, cfg.rms_norm_eps)
        self.mlp = MLP(H, cfg.intermediate_size, cfg.hidden_act, cfg.initializer_range, km)

    def __call__(
        self,
        h: Array,                                   # [C, H]
        pos: Array,                                 # [C]
        kv_prev: Tuple[Array, Array],               # ([W*C, nkv, d], [W*C, nkv, d]) clean history
        key_valid: Array,                           # [(W+1)*C]
        prime: Optional[Tuple[RMSNorm, RMSNorm, FastWeight]] = None,
        inner: bool = False,
        pos_q: Optional[Array] = None,              # [C] query positions (inner stream, shifted targets)
        own_mask: Optional[Array] = None,           # [C, C] own-chunk pattern (attention.own_mask_matrix); [C, 2C]
                                                    # with own_kv (attention.read_mask_matrix)
        memory: Optional[Tuple[RMSNorm, GDNMemory, Array]] = None,   # (norm, module, state): the gdn memory
        update_memory: bool = False,                # also return the state written from this (clean) chunk
        return_prime: bool = False,                 # also return the fast weight's input and read contribution
        own_kv: Optional[Tuple[Array, Array]] = None,   # ([C, nkv, d] x2) the chunk's clean K/V: the sub-block read
    ):
        """One chunk through one layer.  Returns ``h`` and this chunk's own (rotated) K/V
        (and, with ``update_memory``, the memory state after this chunk; with ``return_prime``,
        ``(x, post_norm(W(x)))``: the fast weight's input and its read contribution, ``[C, H]`` each).

        ``inner``: this is the inner-loss forward (C), which the meta-gradient
        differentiates twice - see ``attention.attend``.
        """
        q, k, v = self.attn.qkv(self.attn_norm(h), pos, pos_q)
        if own_kv is None:
            K = jnp.concatenate([kv_prev[0], k], axis=0)
            V = jnp.concatenate([kv_prev[1], v], axis=0)
        else:
            # the sub-block read (ModelConfig.read_block): the window, the chunk's clean K/V, the read's own
            K = jnp.concatenate([kv_prev[0], own_kv[0].astype(k.dtype), k], axis=0)
            V = jnp.concatenate([kv_prev[1], own_kv[1].astype(v.dtype), v], axis=0)
            key_valid = jnp.concatenate([key_valid, jnp.ones((h.shape[0],), jnp.bool_)])
        a = self.attn.attend(q[None], K[None], V[None], key_valid[None], inner=inner, own_mask=own_mask)[0]
        h = h + self.attn.out(a)
        S_next = None
        if prime is not None:
            # The fast-weight residual sits BEFORE the base MLP (e2e transformer.py:213-227):
            # sequential, so the base MLP reads the post-prime stream.  The post-norm is
            # what makes inner_lr = 1.0 safe - it keeps the prime's contribution
            # scale-invariant to how far W has drifted along the sequence.
            norm, post_norm, W = prime
            x_p = norm(h)
            p_out = post_norm(W(x_p))
            h = h + p_out
            if return_prime:
                return h + self.mlp(self.mlp_norm(h)), (k, v), (x_p, p_out)
        elif memory is not None:
            # The gdn baseline: read the state the previous chunks wrote, at the same place;
            # the clean stream also writes this chunk into it for the next chunk (gdn.py).
            norm, gdn, S = memory
            xm = norm(h)
            h = h + gdn.read(xm, S)
            if update_memory:
                S_next = gdn.update(xm, S)
        h = h + self.mlp(self.mlp_norm(h))
        if update_memory:
            return h, (k, v), S_next
        return h, (k, v)

    def prefix_call(
        self,
        h3: Array,                                  # [3, T, H]  (outer, clean, inner)
        pos: Array,                                 # [T]
        chunk: int,
        window: int,
        key_valid: Array,                           # [nc, (W+1)*C]
        attn_chunk_batch: int,
        mlp_per_stream: bool = False,
        kv_win: Optional[Tuple[Array, Array]] = None,   # clean K/V preceding h3, [W*C, nkv, d] x2
        stream_pos: Optional[Array] = None,             # [n_streams, T] key positions per stream (None: pos)
        stream_pos_q: Optional[Array] = None,           # [n_streams, T] query positions per stream (None: keys')
        inner_own: str = "full",                        # own-chunk pattern of the inner streams (index >= 2)
        extra_full: int = 0,                            # trailing streams with the full pattern (distillation teacher)
        read_block: int = 0,                            # ModelConfig.read_block
    ) -> Tuple[Array, Tuple[Array, Array]]:         # (h3, clean K/V of the last W chunks)
        """All chunks of all three streams through one prefix layer, in parallel.

        Each stream's chunk ``c`` attends to the CLEAN stream's K/V of chunks
        ``[c-W, c-1]`` plus its own stream's K/V of chunk ``c`` - exactly what the
        sequential formulation does, without the carry.  ``h3`` may be one segment
        of a longer sequence: ``kv_win`` is then the clean K/V of the ``W`` chunks
        before it (``key_valid`` says which of those exist), and the returned window
        is the one the next segment needs.  Streams ``2 .. n-1-extra_full`` are inner
        streams (``inner_own``), the last ``extra_full`` bidirectional like the first two.
        ``read_block`` b > 0: the clean stream attends block-causally within its chunk and the
        outer stream reads the clean K/V of the earlier groups of b slots and its own group
        (``attention.read_mask_matrix``).
        """
        T = h3.shape[1]
        nc = T // chunk
        WC = window * chunk
        # One stream at a time and recomputed in the backward: the projections' f32
        # norm/RoPE intermediates are [T, H] / [T, nh, d] each (1-3 GB per stream at
        # 128K); vmapped, all three streams' were live, and a plain lax.map would
        # save them all as scan residuals.
        n_streams = h3.shape[0]
        n_inner = n_streams - 2 - extra_full
        sp = jnp.broadcast_to(pos[None], (n_streams, T)) if stream_pos is None else stream_pos
        spq = sp if stream_pos_q is None else stream_pos_q
        q3, k3, v3 = jax.lax.map(jax.checkpoint(lambda t: self.attn.qkv(self.attn_norm(t[0]), t[1], t[2])),
                                 (h3, sp, spq))
        owns = ["full", "full"] + [inner_own] * n_inner + ["full"] * extra_full
        own_mat = own_mask_matrix(chunk, inner_own)

        if resolve_attention_impl() == "flash":
            if read_block:
                raise NotImplementedError("read_block: the splash prefix has no sub-block read")
            # Splash kernel over whole sequences with the static chunk-window mask.
            win_valid = None if kv_win is None else key_valid[0, :WC]     # slots of the incoming window
            a3 = jnp.stack([self.attn.attend_prefix(q3[s], k3[CLEAN], k3[s], v3[CLEAN], v3[s], chunk, window,
                                                    kv_win, win_valid, owns[s])
                            for s in range(n_streams)])                      # [n_streams, T, nh*d]
        else:
            # Reference path: gathered clean window per chunk, chunked attention.
            k_prev, v_prev = (None, None) if kv_win is None else kv_win
            k_win = window_gather(k3[CLEAN], chunk, window, k_prev)          # [nc, W*C, nkv, d]
            v_win = window_gather(v3[CLEAN], chunk, window, v_prev)

            def one_stream(q, k, v, own, k_clean=None, v_clean=None):
                qc = q.reshape(nc, chunk, *q.shape[1:])
                by_chunk = lambda x: [] if x is None else [x.reshape(nc, chunk, *x.shape[1:])]
                K = jnp.concatenate([k_win] + by_chunk(k_clean) + by_chunk(k), axis=1)
                V = jnp.concatenate([v_win] + by_chunk(v_clean) + by_chunk(v), axis=1)
                kvld = key_valid if k_clean is None else jnp.concatenate([key_valid, jnp.ones((nc, chunk), jnp.bool_)],
                                                                        axis=1)
                # lax.map bounds the f32 score tensor to attn_chunk_batch chunks at a time;
                # cuDNN holds no score tensor, so it takes all chunks in one call.
                cudnn = resolve_attention_impl() == "cudnn" and attention.cudnn_supports(q.dtype, q.shape[-1])
                bs = nc if cudnn else attn_chunk_batch
                a = jax.lax.map(lambda t: self.attn.attend(t[0][None], t[1][None], t[2][None], t[3][None],
                                                           own_mask=own)[0],
                                (qc, K, V, kvld), batch_size=bs)
                return a.reshape(T, -1)

            # Sequential over the streams (not vmap): the f32 score tensor is the dominant
            # transient, and vmapping the streams would multiply it.  Outer and clean
            # streams (and the trailing extra ones) share the full own-chunk pattern;
            # inner streams may have their own.
            if read_block:
                # the sub-block read: the outer stream over the clean stream's own-chunk K/V too
                if extra_full:
                    raise NotImplementedError("read_block: no extra bidirectional streams (distillation)")
                parts = [one_stream(q3[OUTER], k3[OUTER], v3[OUTER], read_mask_matrix(chunk, read_block),
                                    k3[CLEAN], v3[CLEAN])[None],
                         one_stream(q3[CLEAN], k3[CLEAN], v3[CLEAN], own_mask_matrix(chunk, f"block:{read_block}"))[None]]
                if n_inner > 0:
                    parts.append(jax.lax.map(lambda t: one_stream(*t, own_mat),
                                             (q3[2:2 + n_inner], k3[2:2 + n_inner], v3[2:2 + n_inner])))
                a3 = jnp.concatenate(parts, axis=0)
            else:
                full_idx = np.asarray([0, 1] + list(range(2 + n_inner, n_streams)))
                a_full = jax.lax.map(lambda t: one_stream(*t, None), (q3[full_idx], k3[full_idx], v3[full_idx]))
                if n_inner > 0:
                    a_in = jax.lax.map(lambda t: one_stream(*t, own_mat),
                                       (q3[2:2 + n_inner], k3[2:2 + n_inner], v3[2:2 + n_inner]))
                    a3 = jnp.concatenate([a_full[:2], a_in, a_full[2:]], axis=0)   # [n_streams, T, nh*d]
                else:
                    a3 = a_full
        h3 = h3 + jax.vmap(self.attn.out)(a3)
        mlp = lambda x: self.mlp(self.mlp_norm(x))
        if mlp_per_stream:
            # One stream at a time, recomputed in the backward: the MLP's [T, I]
            # intermediates are 3.4 GB per stream at 128K.
            h3 = h3 + jax.lax.map(jax.checkpoint(mlp), h3)
        else:
            h3 = h3 + jax.vmap(mlp)(h3)
        return h3, (k3[CLEAN][T - WC:], v3[CLEAN][T - WC:])


# ----------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------
class Model(eqx.Module):
    cfg: ModelConfig = eqx.field(static=True)
    embed: Array                                    # [V, H]
    prefix: Block                                   # stacked [L-S]
    suffix: Block                                   # stacked [S]
    prime_norm: Optional[RMSNorm]                   # stacked [F]   outer params (None with memory gdn); F = cfg.fast_len
    prime_post_norm: Optional[RMSNorm]              # stacked [F]   outer params
    fast_weight: Optional[FastWeight]               # stacked [F]   W_0 template, outer param (None with memory gdn)
    mem_norm: Optional[RMSNorm]                     # stacked [S]   memory gdn: the norm before the state read / write
    gdn: Optional[GDNMemory]                        # stacked [S]   memory gdn: the Gated DeltaNet memory of each suffix layer
    inner_gate: Optional[Linear]                    # stacked [S]   H+2 -> 1: per-chunk multiplier on the inner lr
    inner_token_gate: Optional[Linear]              # H -> 1: per-token weight on the inner loss
    inner_decay_logit: Optional[Array]              # [S]           decay of W toward W_0 per chunk (see inner_loop)
    inner_precond: Optional[Tuple[Any, Any]]        # (rows, cols): log step-size factors per fast-weight matrix, [S, in] / [S, out]
    inner_step_log: Optional[Array]                 # [S]           log per-layer step multiplier (eta_l = lr * exp)
    inner_temp_log: Optional[Array]                 # []            log temperature of the inner CE
    inner_feat_head: Optional[Linear]               # H -> H: the inner stream's feature prediction (inner_loop)
    mtp_head: Optional[Linear]                      # H -> H before the shared LM head: the second target x_{i+2} (MTP)
    inner_kvb_value: Optional[Linear]               # stacked [S] H -> H: the value map of the key-value-binding write
    final_norm: RMSNorm
    lm_head: Optional[Array]                        # [V, H], None when tied

    def __init__(self, cfg: ModelConfig, key: Array):
        self.cfg = cfg
        k_embed, k_layers, k_prime, k_head = jax.random.split(key, 4)
        H, L, S = cfg.hidden_size, cfg.num_hidden_layers, cfg.suffix_len
        F = cfg.fast_len          # the suffix layers with a fast weight (= S unless suffix_plain_top)
        std = cfg.initializer_range

        self.embed = jax.random.normal(k_embed, (cfg.vocab_size, H), jnp.float32) * std
        # One stacked [L] build, then slice - so per-layer keys are those of an unsplit stack.
        layers = stack_modules(lambda k: Block(cfg, k), jax.random.split(k_layers, L))
        self.prefix = slice_stack(layers, slice(0, L - S))
        self.suffix = slice_stack(layers, slice(L - S, L))

        kp = jax.random.split(k_prime, F)
        if cfg.memory == "gdn":
            self.prime_norm = self.prime_post_norm = self.fast_weight = None
            self.mem_norm = stack_modules(lambda _: RMSNorm(H, cfg.rms_norm_eps), kp)
            self.gdn = stack_modules(lambda k: GDNMemory(H, cfg.gdn_heads, cfg.gdn_head_dim, cfg.gdn_head_dim, std, k), kp)
        else:
            self.prime_norm = stack_modules(lambda _: RMSNorm(H, cfg.rms_norm_eps), kp)
            self.prime_post_norm = stack_modules(lambda _: RMSNorm(H, cfg.rms_norm_eps), kp)
            self.fast_weight = stack_modules(
                lambda k: FastWeight(H, cfg.prime_intermediate_size, cfg.hidden_act, std, k), kp)
            self.mem_norm = self.gdn = None
        # Learned write controls (inner_loop.py): both gates start at exactly 1x (zero
        # weight and bias) so an untrained gate reproduces the plain write; the decay at
        # cfg.inner_decay.  The chunk gate reads [mean h_clean, inner loss / 8, mask ratio].
        self.inner_gate = stack_modules(lambda _: Linear(H + 2, 1, 0.0, k_prime, bias=True), kp) \
            if cfg.inner_lr_gate else None
        self.inner_token_gate = Linear(H, 1, 0.0, k_prime, bias=True) if cfg.inner_token_gate else None
        self.inner_decay_logit = jnp.full((F,), jax.scipy.special.logit(cfg.inner_decay / 0.1), jnp.float32) \
            if cfg.inner_decay > 0 else None
        # Learned inner optimiser (inner_loop.py), all at the identity: exp(0) = 1, temperature 1.
        if cfg.inner_optimizer:
            fw = eqx.filter(self.fast_weight, eqx.is_array)
            self.inner_precond = (jax.tree_util.tree_map(lambda w: jnp.zeros(w.shape[:-1], jnp.float32), fw),
                                  jax.tree_util.tree_map(lambda w: jnp.zeros(w.shape[:-2] + w.shape[-1:], jnp.float32), fw))
            self.inner_step_log = jnp.zeros((F,), jnp.float32)
            self.inner_temp_log = jnp.zeros((), jnp.float32)
        else:
            self.inner_precond = self.inner_step_log = self.inner_temp_log = None
        # Feature-prediction head (inner_loop.inner_objective): a random (non-zero) init, so
        # the feature term has a gradient w.r.t. W from the first step; its own key leaves
        # every other parameter's initialisation unchanged.
        self.inner_feat_head = Linear(H, H, std, jax.random.fold_in(k_prime, 7), bias=False) if cfg.inner_feat_head else None
        # Multi-token-prediction head: the final hidden state, mapped by an H x H matrix,
        # through the SHARED LM head predicts the token after the next (chunk_scan; an
        # auxiliary training target, dropped at inference).  Random init like the feature head.
        self.mtp_head = Linear(H, H, std, jax.random.fold_in(k_prime, 8), bias=False) if cfg.mtp_head else None
        # Value map of the key-value-binding write (inner_loop.WriteConfig.kvb_weight): one H x H
        # per suffix layer, its own key (fold_in 9) so every other initialisation is unchanged.
        self.inner_kvb_value = stack_modules(lambda k: Linear(H, H, std, k, bias=False),
                                             jax.random.split(jax.random.fold_in(k_prime, 9), F)) \
            if cfg.inner_kvb_head else None
        self.final_norm = RMSNorm(H, cfg.rms_norm_eps)
        # Tying shares ONE array by dropping lm_head, never by placing the same array
        # at two pytree positions (which become two leaves after any tree_map).
        self.lm_head = None if cfg.tie_word_embeddings else \
            jax.random.normal(k_head, (cfg.vocab_size, H), jnp.float32) * std

    # -- pieces ---------------------------------------------------------------
    def lm_head_weight(self) -> Array:
        return self.embed if self.lm_head is None else self.lm_head

    def embed_tokens(self, ids: Array) -> Array:
        # mode="clip": jnp.take's default fills out-of-range rows with NaN through a
        # full-size select; ids are validated against the vocabulary at load time.
        # Gathered from an f32 view: the backward scatter-adds every position's row
        # into its token's row, and on the bf16 compute copy that sum runs in bf16 - a
        # token seen thousands of times per step (the mask token) loses most of its
        # gradient (1B, 2x8192 on GH200: 53% on the mask row).  Same values out; the
        # callers cast them to the compute dtype.
        return jnp.take(self.embed.astype(jnp.float32), ids, axis=0, mode="clip")

    def logits(self, h: Array) -> Array:
        """``[.., H]`` -> ``[.., V]`` in at least f32 (bf16 logits into a f32 softmax
        move the initial-loss sanity check)."""
        dt = jnp.promote_types(h.dtype, jnp.float32)
        return h.astype(dt) @ self.lm_head_weight().astype(dt).T

    # -- prefix: chunk-parallel, three streams --------------------------------
    def prefix_forward(
        self,
        ids3: Array,                                # [3, T]
        pos: Array,                                 # [T]
        key_valid: Array,                           # [nc, (W+1)*C]
        attn_chunk_batch: int,
        compute_dtype=jnp.bfloat16,
        remat_prefix: bool = False,
        remat_levels: int = 1,
        kv_win: Optional[Tuple[Array, Array]] = None,   # per layer: clean K/V of the W chunks before ids3
        stream_pos: Optional[Array] = None,             # [n_streams, T] key positions per stream (None: pos)
        stream_pos_q: Optional[Array] = None,           # [n_streams, T] query positions per stream
        inner_own: str = "full",                        # own-chunk attention pattern of the inner streams
        extra_full: int = 0,                            # trailing bidirectional streams (distillation teacher)
    ) -> Tuple[Array, Tuple[Array, Array]]:         # (h3 [3, T, H], kv_win for the next segment [L-S, W*C, nkv, d] x2)
        """``remat_levels`` nests rematerialisation over the LAYER scan: the backward of
        a scan keeps every iteration's checkpointed input, i.e. ``L-S`` copies of
        ``[3, T, H]`` (27 GB per sequence at 128K); with two levels ``(3, 6)`` only
        ``3 + 6`` are live.  ``remat_prefix`` additionally checkpoints the whole stack
        so nothing but the embedding output survives the suffix phase, and makes the
        MLP recompute per stream (bounds its ``[T, I]`` transients).

        ``ids3`` may be one segment of a longer sequence (``chunk_scan``'s segmented
        scan): ``kv_win`` carries every prefix layer's clean K/V window across the
        segment boundary and ``key_valid`` is built from global chunk indices."""
        cfg = self.cfg
        L = cfg.num_hidden_layers - cfg.suffix_len
        h3 = constrain_local(jax.vmap(self.embed_tokens)(ids3).astype(compute_dtype))

        def body(h, xs):
            blk, win = xs
            blk = replicate_in_context(blk)         # gather this layer's weights, keep activations batch-sharded
            return blk.prefix_call(h, pos, cfg.chunk_size, cfg.window_chunks, key_valid, attn_chunk_batch,
                                   mlp_per_stream=remat_prefix, kv_win=win, stream_pos=stream_pos,
                                   stream_pos_q=stream_pos_q, inner_own=inner_own, extra_full=extra_full,
                                   read_block=cfg.read_block)

        def run(h):
            factors = balanced_factors(L, max(1, remat_levels))
            return nested_remat_scan(body, h, (self.prefix, kv_win), factors, stack_ys=True)

        return jax.checkpoint(run)(h3) if remat_prefix else run(h3)

    # -- suffix: one chunk, with the live fast weight -------------------------
    def suffix_forward(
        self,
        h: Array,                                   # [C, H]
        pos: Array,                                 # [C]
        kv: Tuple[Array, Array],                    # ([S, W*C, nkv, d], [S, W*C, nkv, d])
        W: Optional[FastWeight],                    # stacked [S], compute dtype (None with memory gdn)
        key_valid: Array,                           # [(W+1)*C]
        inner: bool = False,                        # the inner-loss forward (differentiated twice)
        pos_q: Optional[Array] = None,              # [C] query positions (inner stream, shifted targets)
        own_mask: Optional[Array] = None,           # [C, C] own-chunk pattern (inner stream); [C, 2C] with own_kv
        memory_state: Optional[Array] = None,       # [S, nh, dv, dk] the gdn states (memory gdn)
        update_memory: bool = False,                # memory gdn: also return the states written from this chunk
        return_prime: bool = False,                 # also return (x [S, C, H], post_norm(W(x)) [S, C, H]) per layer
        own_kv: Optional[Tuple[Array, Array]] = None,   # ([S, C, nkv, d] x2) the chunk's clean K/V: the sub-block read
    ):                                              # (h [C, H] normed, kv_own [S, C, nkv, d] x2[, states | prime])
        if self.gdn is not None:
            def body_m(h, xs):
                blk, mn, g, S_l, kv_l, ok_l = xs
                out = blk(h, pos, kv_l, key_valid, inner=inner, pos_q=pos_q, own_mask=own_mask,
                          memory=(mn, g, S_l), update_memory=update_memory, own_kv=ok_l)
                return out[0], out[1:]

            h, ys = jax.lax.scan(_suffix_ck(body_m), h, (self.suffix, self.mem_norm, self.gdn, memory_state, kv, own_kv))
            return (self.final_norm(h), ys[0], ys[1]) if update_memory else (self.final_norm(h), ys[0])

        def body(h, xs):
            blk, pn, ppn, w, kv_l, ok_l = xs
            return blk(h, pos, kv_l, key_valid, prime=(pn, ppn, w), inner=inner, pos_q=pos_q, own_mask=own_mask,
                       own_kv=ok_l)

        def body_p(h, xs):
            # also every fast-weight layer's input and read contribution ([F, C, H] each):
            # the key-value-binding write (inner_loop.inner_objective, WriteConfig.kvb_weight)
            blk, pn, ppn, w, kv_l, ok_l = xs
            h, kv_o, prime_io = blk(h, pos, kv_l, key_valid, prime=(pn, ppn, w), inner=inner, pos_q=pos_q,
                                    own_mask=own_mask, return_prime=True, own_kv=ok_l)
            return h, (kv_o, prime_io)

        def body_n(h, xs):
            # a plain suffix layer (fast_layers / suffix_plain_top): no memory term, but in the scan (its
            # input depends on W) and in the inner stream's path to the inner loss
            blk, kv_l, ok_l = xs
            return blk(h, pos, kv_l, key_valid, inner=inner, pos_q=pos_q, own_mask=own_mask, own_kv=ok_l)

        kv_parts, prime_parts = [], []
        own_runs = [r[2] for r in self._suffix_runs(own_kv)]          # own_kv sliced like the windows
        for (fast, blocks, kv_r, fsl), ok_r in zip(self._suffix_runs(kv), own_runs):
            if fast:
                xs = (blocks, fsl(self.prime_norm), fsl(self.prime_post_norm), fsl(W), kv_r, ok_r)
                if return_prime:
                    h, (kv_o, prime_io) = jax.lax.scan(_suffix_ck(body_p), h, xs)
                    prime_parts.append(prime_io)
                else:
                    h, kv_o = jax.lax.scan(_suffix_ck(body), h, xs)
            else:
                h, kv_o = jax.lax.scan(_suffix_ck(body_n), h, (blocks, kv_r, ok_r))
            kv_parts.append(kv_o)
        cat = lambda parts: parts[0] if len(parts) == 1 else jax.tree_util.tree_map(
            lambda *x: jnp.concatenate(x, axis=0), *parts)
        if return_prime:
            return self.final_norm(h), cat(kv_parts), cat(prime_parts)
        return self.final_norm(h), cat(kv_parts)

    def _suffix_runs(self, kv=None):
        """The suffix as runs of same-kind layers (``ModelConfig.suffix_runs``), bottom to top:
        ``[(fast, blocks, K/V windows, fsl)]``, ``fsl`` slicing a fast-weight stack to the run.  A run
        covering the whole suffix (the default layout) uses the stacks themselves, unsliced: the
        default path is the unchanged single scan."""
        S, F = self.cfg.suffix_len, self.cfg.fast_len
        out, f = [], 0
        for fast, a, b in self.cfg.suffix_runs:
            whole = (a, b) == (0, S)
            blocks = self.suffix if whole else slice_stack(self.suffix, slice(a, b))
            kv_r = kv if (kv is None or whole) else tuple(x[a:b] for x in kv)
            fa, fb = (f, f + b - a) if fast else (f, f)
            fsl = (lambda t: t) if (fa, fb) == (0, F) else (lambda t, fa=fa, fb=fb: slice_stack(t, slice(fa, fb)))
            out.append((fast, blocks, kv_r, fsl))
            f = fb
        return out

    # -- convenience: whole sequence, no TTT (tests, eval) --------------------
    def hidden(self, ids: Array, pos: Optional[Array] = None, W: Optional[FastWeight] = None,
               compute_dtype=jnp.float32, attn_chunk_batch: int = 8) -> Array:
        """Final-normed hidden states ``[T, H]`` for one sequence with the KV window
        applied and a FIXED fast weight (``W`` or the template).  Chunk-parallel end
        to end; the training path threads a per-chunk ``W`` through the scan instead."""
        cfg = self.cfg
        T = ids.shape[0]
        if T % cfg.chunk_size:
            raise ValueError(f"T={T} not a multiple of chunk_size={cfg.chunk_size}")
        nc = T // cfg.chunk_size
        if cfg.reach is not None and resolve_attention_impl() == "flash":
            raise NotImplementedError("window_reach: the splash prefix's static mask has the full window")
        if cfg.read_block:
            raise NotImplementedError("read_block: a fixed-memory forward of clean ids has no noisy read (use the "
                                      "training scan or the decoder)")
        pos = jnp.arange(T) if pos is None else pos
        kv_all = key_validity_all(nc, cfg.chunk_size, cfg.window_chunks, reach=cfg.reach)
        ids3 = jnp.stack([ids, ids, ids])
        h = self.prefix_forward(ids3, pos, kv_all, attn_chunk_batch, compute_dtype)[0][CLEAN]
        if self.gdn is not None:
            # memory gdn: an empty state reads to zero, so the fixed-memory forward is the
            # suffix without its memory term.
            return self._suffix_parallel(h, pos, None, kv_all, attn_chunk_batch)
        Wc = cast_floating(self.fast_weight if W is None else W, compute_dtype)
        # Suffix chunk-parallel with a fixed W: same as suffix_forward per chunk, but
        # the clean history of each chunk is gathered rather than carried.
        return self._suffix_parallel(h, pos, Wc, kv_all, attn_chunk_batch)

    def forward(self, ids: Array, **kw) -> Array:
        """Logits ``[T, V]``; see ``hidden``."""
        return self.logits(self.hidden(ids, **kw))

    def _suffix_parallel(self, h, pos, W, key_valid, attn_chunk_batch):
        cfg = self.cfg
        C, Wn = cfg.chunk_size, cfg.window_chunks
        nc = h.shape[0] // C

        def layer(h, xs):
            blk, prime = xs
            q, k, v = blk.attn.qkv(blk.attn_norm(h), pos)
            k_win, v_win = window_gather(k, C, Wn), window_gather(v, C, Wn)
            qc = q.reshape(nc, C, *q.shape[1:])
            K = jnp.concatenate([k_win, k.reshape(nc, C, *k.shape[1:])], axis=1)
            V = jnp.concatenate([v_win, v.reshape(nc, C, *v.shape[1:])], axis=1)
            a = jax.lax.map(lambda t: blk.attn.attend(t[0][None], t[1][None], t[2][None], t[3][None])[0],
                            (qc, K, V, key_valid), batch_size=attn_chunk_batch).reshape(h.shape[0], -1)
            h = h + blk.attn.out(a)
            if prime is not None:
                pn, ppn, w = prime
                h = h + ppn(w(pn(h)))
            h = h + blk.mlp(blk.mlp_norm(h))
            return h, None

        prime = None if W is None else (self.prime_norm, self.prime_post_norm, W)
        for fast, blocks, _, fsl in self._suffix_runs():
            h = jax.lax.scan(layer, h, (blocks, tuple(fsl(t) for t in prime) if fast and prime is not None else None))[0]
        return self.final_norm(h)
