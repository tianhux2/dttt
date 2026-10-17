# coding=utf-8
"""
The training objective over a batch: chunk-parallel prefix, then a rematerialised
scan over chunks for the suffix with the fast weight and KV window in the carry.

Per chunk the suffix runs THREE forwards on the same chunk with the same key set
and the same fast weight:

    A  outer-masked  -> weighted CE against the clean tokens         (the loss)
    B  clean         -> this chunk's K/V, shifted into the window
    C  inner-masked  -> grad w.r.t. W, one clipped SGD step

B and C are computed on the last chunk too, where nothing consumes them.  The
scan body is deliberately uniform: a ``lax.cond`` on the last chunk saved 1/nc of
B+C (a few percent of the step) but its backward materialises zero-filled
residuals of the untaken branch - one fast-weight-sized f32 buffer per leaf per
branch, 3.7 GB per chip at 8K - and that is what did not fit.  The unused last
update is one extra term in the drift statistic and nothing else.

The KV window is populated by CLEAN tokens because that is what inference sees
(decoded blocks are clean); the inner mask is an independent draw so the fast
weight gets a different corruption than the outer loss.  Both are why A/B/C
cannot be fused into one forward for a diffusion objective, unlike the AR case.

The prefix (layers without a fast weight) is computed once, chunk-parallel, for
all three streams - see ``model.Block.prefix_call``.

Two optional training-only additions to the write (``distill`` / ``mtp``):

    T  teacher      -> a fourth prefix stream, the chunk masked at a LOW ratio and read
                       bidirectionally; its suffix forward (with the current W, no
                       gradient) gives a top-k distribution per slot that the inner
                       stream's prediction of the same token is pulled towards (soft
                       CE next to the one-hot CE), where the teacher had it masked.
    MTP             -> ``mtp_head`` (H x H, then the shared LM head) predicts x_{i+2}:
                       in the outer loss only where that token is masked (leak-free
                       under bidirectional attention), in the causal write at every slot.

Segments (``segment_chunks``): a long sequence is processed as consecutive
segments of that many chunks by an outer scan, each segment exactly as a whole
sequence would be (prefix chunk-parallel over the segment, suffix chunk scan),
with three carries across the boundary: every prefix layer's clean KV window,
the suffix's KV window and the fast weight.  The result is identical to the
unsegmented computation; what changes is memory: the backward keeps one carry
per segment instead of per chunk / per layer over the whole sequence, so 128K
runs with plain per-segment + per-chunk checkpointing.  ``truncate_fast_weight``
additionally stops the meta-gradient at the fast weight on the boundary (the
inner-update chain then has the segment's length, as in the 8K stage); the KV
windows keep their gradient either way.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import NamedTuple, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.sharding import Mesh

from .inner_loop import (InnerExtras, InnerStats, WriteConfig, drift_stats, inner_lr, inner_objective, inner_update,
                         write_chunk)
from .model import (CLEAN, INNER, OUTER, FastWeight, Model, cast_floating, key_validity, key_validity_all, own_mask_matrix,
                    read_mask_matrix, resolve_attention_impl)
from .fused_ce import per_token_ce, topk_logprobs
from .objective import band_correlate, cart_kernel, from_chunks, make_inner_mask, q_sample, time_weights, to_chunks, weighted_ce
from .remat_scan import balanced_factors, nested_remat_scan
from .sharding import AXIS, batch_axis_name, constrain_batch_axis, constrain_replicated


class Batch(NamedTuple):
    input_ids: Array                              # [B, T] int32
    loss_mask: Optional[Array] = None             # [B, T] bool; None = all
    position_ids: Optional[Array] = None          # [B, T] int32; None = arange
    inner_mask: Optional[Array] = None            # [B, T] bool: host-chosen inner targets (inner_masks.py); None = random
    inner_perm: Optional[Array] = None            # [B, T] int32: per-chunk physical order of the inner stream (split task)
    outer_t: Optional[Array] = None               # [B] float32: optional explicit t per sequence (stratified eval / probe)


@dataclass(frozen=True)
class LossConfig:
    """Everything ``loss_fn`` needs that is not in ``ModelConfig``."""
    time_reweighting: str = "cart"
    cart_p: float = 0.1
    cart_window: int = 512
    outer_t: float = -1.0                 # >= 0: pin diffusion time t for evaluation / needle probing
    outer_shift: bool = False             # slot i predicts token i+1 (Dream-style objective shift)
    inner_lr: float = 1.0
    inner_clip: float = 1.0
    inner_lr_warmup_steps: int = 0
    inner_mask_range: Tuple[float, float] = (0.5, 1.0)
    inner_mask_ratio: float = -1.0
    second_order: bool = True
    remat_levels: int = 1
    remat_prefix: bool = False
    prefix_remat_levels: int = 1          # nested remat over the prefix layer scan (2 at 128K)
    segment_chunks: int = 0               # 0: the whole sequence is one segment
    truncate_fast_weight: bool = False    # stop the meta-gradient at W on segment boundaries
    inner_views: int = 1                  # 2: complementary inner masks, gradients averaged
    inner_view_mode: str = "complement"   # "strata": V independent draws with stratified mask ratios
    inner_views_sequential: bool = False  # V inner SGD steps per chunk (lr / V, clip / V each) instead of one averaged step
    inner_cart: bool = False              # weight the inner (mask-task) loss by the outer's CART visible-density weights
    inner_mask_span: float = 0.0          # > 0: span masking with this mean span length
    inner_update: str = "sgd"             # sgd | sign | muon2
    inner_task: str = "mask"              # mask (predict masked tokens) | causal (next token, clean chunk) | split (context/targets, any order)
    inner_from_outer: bool = False        # the write = the outer stream's (cart-weighted) loss gradient; no third prefix stream
    inner_feat: str = "off"               # off | add | only: feature-prediction term in the inner objective
    inner_feat_weight: float = 1.0        # its weight in "add"
    inner_update_frac: float = 1.0        # < 1: the write touches a fixed random fraction of W's entries (capacity probe)
    write: WriteConfig = WriteConfig()    # sub-chunk / surprise / key-value-binding / two-timescale causal writes
    write_plan: Tuple[int, ...] = ()      # evaluation: whole-chunk write steps per chunk (index = chunk, missing = 1, 0 = none)
    distill: bool = False                 # soft-target distillation of the write from a low-noise teacher view (see below)
    distill_t_range: Tuple[float, float] = (0.1, 0.3)   # the teacher view's mask ratio, sampled per chunk and sequence
    distill_weight: float = 1.0           # weight of the soft term next to the one-hot anchor CE
    distill_topk: int = 64                # tokens of the teacher distribution kept (unnormalised: the tail is dropped)
    distill_meta: bool = True             # False: no meta-gradient through the soft term (its write gradient is a constant)
    distill_warmup_steps: int = 0         # > 0: the soft term's weight ramps linearly from 0 over this many outer steps
    mtp: bool = False                     # second target x_{i+2} through mtp_head (outer where masked; the causal write everywhere)
    mtp_weight: float = 0.3               # its weight in the outer objective (0 at evaluation: the loss stays the main CE)
    inner_mtp_weight: float = 0.5         # its weight in the write
    inner_context_frac: float = 0.5       # split: share of each chunk that is context (never a target)
    no_write: bool = False                # no inner stream at all: W stays W_0 (a static parallel MLP)
    write_off: bool = False               # the inner stream is read (its CE is available as an auxiliary loss) but W is never updated
    aux_causal_weight: float = 0.0        # > 0: the inner stream's CE (before any write) joins the outer objective with this weight
    per_sequence: bool = False            # also return the loss per sequence (evaluation: paired noise)
    per_token: bool = False               # also return the CE per token and the outer mask (evaluation probes)
    score_from_chunk: int = 0             # outer loss only on chunks >= this (evaluation: score a tail)
    reread_chunks: int = 0                # evaluation: the first n chunks are read twice (W written twice on them)
                                          # before the rest; needs segment_chunks > 0 dividing n (the window reset
                                          # at the start of the second pass is a segment boundary)
    window_drop: float = 0.0              # training: the probability that a step sees no window (the own chunk only;
                                          # cross-chunk information through the memory alone), drawn per step
    window_drop_until: int = 0            # ... for steps below this (the last steps keep the full window)
    self_prior: float = 0.0               # training: the share of sequences whose chunks are read with the memory AFTER
                                          # their own write (write-then-read: the read learns to retrieve what the clean
                                          # causal write stored); the rest read as usual
    outer_views: int = 1                  # training: each chunk's noisy read done this many times, every view with its own
                                          # noise level and mask; the clean stream, the write and its meta-gradient are
                                          # shared, the CE of all views averaged per scored token
    fused_ce_block: int = 8192
    attn_chunk_batch: int = 2
    compute_dtype: str = "bfloat16"


class _Ys(NamedTuple):
    ce_sum: Array
    mask_count: Array
    inner: InnerStats
    ce_chunk: Array          # [nc] ce_sum placed at this chunk's slot (summed over the scan -> per-chunk profile)
    mask_chunk: Array        # [nc]
    ce_seq: Array            # [B] per-sequence CE sum (zeros unless cfg.per_sequence)
    mask_seq: Array          # [B]
    ce_tok: Array            # [B, nc, C] unweighted CE per token at this chunk's slot (cfg.per_token) else [1]
    mask_tok: Array          # [B, nc, C] outer mask at this chunk's slot (cfg.per_token) else [1]
    ce2_sum: Array           # outer MTP (x_{i+2}) weighted CE sum (0 without mtp)
    mask2_count: Array       # its masked-target count
    distill_mass: Array      # probability mass of the teacher's top-k, batch-summed (0 without distill)
    plan_ce: Array           # [2] inner CE before / after the write of the chunks a write plan writes > once, batch-summed


def feat_target_index(task: str, nc: int, B: int, C: int, perm_c: Optional[Array] = None) -> Array:
    """``[nc, B, C]`` chunk-local index of the clean-stream position each inner slot predicts:
    the slot itself (mask task), the next slot (causal), the next slot of the physical
    order (split; ``perm_c`` holds absolute positions)."""
    i = jnp.arange(C)
    if task == "mask":
        return jnp.broadcast_to(i[None, None], (nc, B, C))
    if task == "causal":
        return jnp.broadcast_to(jnp.minimum(i + 1, C - 1)[None, None], (nc, B, C))
    local = perm_c - (jnp.arange(nc) * C)[:, None, None]
    return jnp.concatenate([local[:, :, 1:], local[:, :, -1:]], axis=2)


def make_feat_target(hc: Array, tgt_idx: Array) -> Array:
    """``[C, H]`` feature targets of one chunk: the clean stream's suffix input ``hc [C, H]``
    gathered at ``tgt_idx [C]``, RMS-normalised over H (f32) and held constant (no
    meta-gradient through the target: the write must fit the representation, the
    representation must not be trained to be easy to fit)."""
    t = hc[tgt_idx].astype(jnp.float32)
    t = t * jax.lax.rsqrt(jnp.mean(jnp.square(t), axis=-1, keepdims=True) + 1e-6)
    return jax.lax.stop_gradient(t)


def shift_slots(x: Array, k: int) -> Tuple[Array, Array]:
    """``(y, valid)``: ``y[:, i] = x[:, i + k]`` along axis 1 where ``0 <= i + k < C`` (the
    edge slot copied, flagged invalid, elsewhere); ``valid [C]`` bool."""
    C = x.shape[1]
    i = jnp.arange(C)
    if k == 0:
        return x, jnp.ones((C,), jnp.bool_)
    if k > 0:
        return jnp.concatenate([x[:, k:], jnp.repeat(x[:, -1:], k, axis=1)], axis=1), i <= C - 1 - k
    return jnp.concatenate([jnp.repeat(x[:, :1], -k, axis=1), x[:, :k]], axis=1), i >= -k


def soft_targets(h_teacher: Array, lm_w: Array, topk: int, block: int) -> Tuple[Array, Array, Array]:
    """The teacher's top-k of every slot: ``(ids [B, C, K], q [B, C, K], mass [B])`` - the
    tokens, their probabilities UNnormalised (the dropped tail contributes nothing, so
    a flat teacher - the first steps from scratch - weighs almost nothing and a sharp one
    counts in full) and the mean mass the kept tokens cover; all held constant."""
    B, C, H = h_teacher.shape
    ids, logp = topk_logprobs(h_teacher.reshape(B * C, H), lm_w, topk, block or 8192)
    q = jnp.exp(logp)
    mass = jnp.mean(jnp.sum(q, axis=-1).reshape(B, C), axis=1)
    return ids.reshape(B, C, -1), q.reshape(B, C, -1), jax.lax.stop_gradient(mass)


def loss_fn(model: Model, batch: Batch, key: Array, step: Array, cfg: LossConfig,
            mesh: Optional[Mesh] = None) -> Tuple[Array, dict]:
    """Batch loss and diagnostics.  ``model`` is the (already cast) compute-dtype model."""
    mc = model.cfg
    use_gdn = mc.memory == "gdn"
    if use_gdn:
        # The gdn baseline has no inner stream and no fast weight: the two-stream (no_write)
        # layout, with the carried "W" slot holding the gdn states instead (see suffix_step).
        if cfg.inner_from_outer or cfg.write_off or cfg.distill or cfg.mtp or cfg.inner_feat != "off":
            raise ValueError("memory gdn: no inner stream (inner_from_outer / write_off / distill / mtp / inner_feat do not apply)")
        cfg = replace(cfg, no_write=True)
    ids = batch.input_ids
    B, T = ids.shape
    C, Wn, S = mc.chunk_size, mc.window_chunks, mc.suffix_len
    if T % C:
        raise ValueError(f"seq_len {T} not a multiple of chunk_size {C}")
    nc = T // C
    cdt = jnp.dtype(cfg.compute_dtype)
    pos = jnp.broadcast_to(jnp.arange(T)[None], (B, T)) if batch.position_ids is None else batch.position_ids

    # ---- keys: the split order is a contract (old-vs-new comparison depends on it)
    if key.ndim == 2:
        keys_split = jax.vmap(lambda k: jax.random.split(k, 3))(key)
        key_mask = keys_split[:, 1, :]
        key_inner = keys_split[:, 2, :]
        inner_keys = jax.vmap(lambda k: jax.random.split(k, nc))(key_inner).transpose(1, 0, 2)
    else:
        _, key_mask, key_inner = jax.random.split(key, 3)
        inner_keys = jax.random.split(key_inner, nc * B).reshape(nc, B, 2)

    # ---- outer objective
    loss_mask = batch.loss_mask
    if cfg.score_from_chunk:
        scored = (jnp.arange(T) >= cfg.score_from_chunk * C)[None, :]
        loss_mask = scored if loss_mask is None else (loss_mask.astype(jnp.bool_) & scored)
    t_in = batch.outer_t
    if cfg.outer_t >= 0:
        t_in = jnp.full((B,), cfg.outer_t, dtype=jnp.float32)
    kernel = cart_kernel(cfg.cart_p, cfg.cart_window) if cfg.time_reweighting == "cart" else None

    def noisy_view(k):
        """One noisy read's draw with key ``k``: ``(x_t, t, t_mask, w_raw, m_outer, weight)`` - the noised tokens,
        the noise level, the masked positions, the time weights, and the scored slots with their CE weights."""
        x_t, t, t_mask = q_sample(k, ids, loss_mask, mc.mask_token_id, t=t_in)
        tm_f = t_mask.astype(jnp.float32)
        if cfg.time_reweighting == "cart":
            w_raw = band_correlate(1.0 - tm_f, kernel[0, 0])
        elif cfg.time_reweighting == "linear":
            w_raw = jnp.broadcast_to((1.0 - t)[:, None], tm_f.shape)
        elif cfg.time_reweighting == "original":
            w_raw = jnp.broadcast_to((1.0 / jnp.maximum(t, 1e-6))[:, None], tm_f.shape)
        elif cfg.time_reweighting == "none":
            w_raw = jnp.ones_like(tm_f)
        else:
            raise ValueError(f"time_reweighting {cfg.time_reweighting!r}")
        if cfg.outer_shift:
            m_outer = jnp.pad(t_mask[:, 1:], ((0, 0), (0, 1)))
            w_shifted = jnp.pad(w_raw[:, 1:], ((0, 0), (0, 1)))
            weight = w_shifted * m_outer.astype(jnp.float32)
        else:
            m_outer = t_mask
            weight = w_raw * tm_f
        return x_t, t, t_mask, w_raw, m_outer, weight

    x_t, t, t_mask, w_raw, m_outer, weight = noisy_view(key_mask)
    tgt_outer = jnp.pad(ids[:, 1:], ((0, 0), (0, 1))) if cfg.outer_shift else ids
    # the extra noisy reads (outer_views): their own masks and noise levels, folded off the mask key (view 0 is the
    # single-view draw); read like view 0, scored on the same targets
    VO = cfg.outer_views
    if VO < 1:
        raise ValueError("outer_views >= 1")
    if VO > 1 and (cfg.distill or cfg.mtp or cfg.reread_chunks or cfg.inner_from_outer or cfg.per_sequence
                   or cfg.per_token or mc.read_block):
        raise ValueError("outer_views > 1 is a training option: not with distill / mtp / reread_chunks / inner_from_outer / "
                         "per-sequence or per-token scores / read_block")
    fold = (lambda k, v: jax.vmap(lambda kk: jax.random.fold_in(kk, v))(k)) if key_mask.ndim == 2 else jax.random.fold_in
    extra_views = [noisy_view(fold(key_mask, v)) for v in range(1, VO)]
    if cfg.mtp or cfg.distill:
        if cfg.no_write or cfg.inner_from_outer or cfg.inner_task == "split" or cfg.inner_views != 1:
            raise ValueError("distill / mtp need a single-view mask or causal write (not no_write / inner_from_outer / split)")
    if cfg.write_off and (cfg.no_write or cfg.inner_from_outer or cfg.distill or cfg.mtp or cfg.inner_views != 1
                          or cfg.inner_feat != "off"):
        raise ValueError("write_off keeps one plain inner stream (not no_write / inner_from_outer / distill / mtp / views / feat)")
    if cfg.aux_causal_weight > 0 and (cfg.no_write or cfg.inner_from_outer):
        raise ValueError("aux_causal_weight needs an inner stream (not no_write / inner_from_outer)")
    if cfg.write_plan and (not cfg.write.plain or cfg.inner_task != "causal" or cfg.inner_views != 1 or cfg.no_write
                           or cfg.write_off or cfg.inner_from_outer or cfg.reread_chunks or min(cfg.write_plan) < 0):
        raise ValueError("write_plan needs the plain causal write (one view, a write, no re-read) and steps >= 0")
    if not cfg.write.plain:
        cfg.write.check(C)
        if (cfg.inner_task != "causal" or cfg.inner_views != 1 or cfg.no_write or cfg.inner_from_outer or cfg.write_off
                or cfg.distill or cfg.mtp or cfg.inner_feat != "off" or cfg.inner_update_frac < 1.0):
            raise ValueError("the write schemes need the plain causal write (inner_task causal, one view, a write; "
                             "not with write_off / distill / mtp / inner_feat / inner_update_frac)")
        if cfg.write.kvb_weight > 0 and model.inner_kvb_value is None:
            raise ValueError("inner_kvb_weight needs a model with inner_kvb_head (configs/1b_e2e_kvb)")
    if not 0.0 <= cfg.window_drop <= 1.0 or not 0.0 <= cfg.self_prior <= 1.0:
        raise ValueError("window_drop and self_prior are probabilities")
    if cfg.self_prior > 0 and (use_gdn or cfg.no_write or cfg.write_off or cfg.inner_from_outer or cfg.write_plan
                               or cfg.reread_chunks or cfg.inner_views != 1):
        raise ValueError("self_prior reads a chunk after its own write: it needs a write (TTT, one view, no write plan / "
                         "re-read)")
    if (mc.reach is not None or cfg.window_drop > 0) and resolve_attention_impl() == "flash":
        raise NotImplementedError("window_reach / window_drop: the splash prefix's static mask has the full window")
    # Draws of the two training options, folded off the step key (the existing splits stay as they were).
    base_key = key if key.ndim == 1 else key[0]
    reach = mc.reach
    if cfg.window_drop > 0:
        drop = (jax.random.uniform(jax.random.fold_in(base_key, 7)) < cfg.window_drop) & (step < cfg.window_drop_until)
        reach = jnp.where(drop, 0, Wn if mc.reach is None else mc.reach)
    if cfg.self_prior > 0:
        if key.ndim == 2:
            sp_flags = jax.vmap(lambda k: jax.random.uniform(jax.random.fold_in(k, 8)) < cfg.self_prior)(key)
        else:
            sp_flags = jax.random.uniform(jax.random.fold_in(key, 8), (B,)) < cfg.self_prior
    else:
        sp_flags = None
    if cfg.mtp:
        # Multi-token prediction: slot i also predicts x_{i+2}, scored only where that
        # token is masked (it is in the input otherwise - bidirectional attention would
        # read it), CART-weighted at the target's position like the main target.
        if not cfg.outer_shift or cfg.inner_task != "causal":
            raise ValueError("mtp needs outer_shift and the causal write")
        if model.mtp_head is None:
            raise ValueError("mtp needs a model with mtp_head (configs/1b_e2e_mtp)")
        tgt_mtp = jnp.pad(ids[:, 2:], ((0, 0), (0, 2)))
        m_mtp = jnp.pad(t_mask[:, 2:], ((0, 0), (0, 2)))
        w_mtp = jnp.pad(w_raw[:, 2:], ((0, 0), (0, 2))) * m_mtp.astype(jnp.float32)
    else:
        tgt_mtp = m_mtp = w_mtp = None

    # ---- re-read layout (evaluation): the first n chunks are prepended once more, unscored,
    # with their own inner-mask draws; the KV window restarts at the second pass (``eff``
    # maps the physical chunk index to the one the validity rules see), so the second pass
    # is the ordinary computation except that W already holds the first pass's writes.
    # Drawn on the original [B, T] above, the outer masks of the scored tail are identical
    # to the single-pass run's (paired comparison).
    T0, nc0, n_rr = T, nc, cfg.reread_chunks
    host_mask, host_perm = batch.inner_mask, batch.inner_perm
    if n_rr:
        if cfg.segment_chunks <= 0 or n_rr % cfg.segment_chunks:
            raise ValueError("reread_chunks needs segment_chunks > 0 dividing it (the window reset is a segment boundary)")
        nC = n_rr * C
        dup = lambda a: jnp.concatenate([a[:, :nC], a], axis=1)
        zero = lambda a: jnp.concatenate([jnp.zeros_like(a[:, :nC]), a], axis=1)
        if cfg.distill:
            raise ValueError("reread_chunks with distill is not supported")
        x_t = jnp.concatenate([ids[:, :nC], x_t], axis=1)          # pass 1 is clean in the outer stream (unscored)
        ids, pos, m_outer, weight = dup(ids), dup(pos), zero(m_outer), zero(weight)
        tgt_outer = dup(tgt_outer)
        if cfg.mtp:
            tgt_mtp, m_mtp, w_mtp = dup(tgt_mtp), zero(m_mtp), zero(w_mtp)
        if host_mask is not None:
            host_mask = dup(host_mask)
        if host_perm is not None:                                   # absolute positions: the original block moves by nC
            host_perm = jnp.concatenate([host_perm[:, :nC], host_perm + nC], axis=1)
        if key_inner.ndim == 2:
            folded = jax.vmap(lambda k: jax.random.fold_in(k, 1))(key_inner)
            rr_inner = jax.vmap(lambda k: jax.random.split(k, n_rr))(folded).transpose(1, 0, 2)
        else:
            rr_inner = jax.random.split(jax.random.fold_in(key_inner, 1), n_rr * B).reshape(n_rr, B, 2)
        inner_keys = jnp.concatenate([rr_inner, inner_keys], axis=0)
        T, nc = T + nC, nc + n_rr
    eff = (lambda c: c - n_rr * (c >= n_rr)) if n_rr else (lambda c: c)
    plan_arr = jnp.asarray((tuple(cfg.write_plan) + (1,) * nc)[:nc], jnp.int32) if cfg.write_plan else None

    # ---- inner mask, drawn outside the scan so the prefix sees exactly what the suffix will
    # ---- the inner (write) task: what the C stream sees and what it must predict
    #   mask   : x_inner = tokens with the targets replaced by [MASK]; target_c = the tokens; w_c = the mask
    #   causal : x_inner = the clean chunk; slot i predicts slot i+1 (own-chunk attention causal)
    #   split  : x_inner = the clean chunk in a host-chosen physical order [context | targets];
    #            slots K-1 .. C-2 predict slots K .. C-1 (context rows see context only)
    # The last two also need the queries' positions (the predicted token's) and the
    # own-chunk pattern; ``pos_in_c`` / ``pos_q_c`` are their key / query positions.
    ids_all_c = to_chunks(ids, C)                                                             # [nc, B, C]
    own_mat = None
    pos_in_c, pos_q_c = to_chunks(pos, C), None
    if cfg.inner_task == "mask":
        if host_mask is not None:
            # Host-chosen targets (rare / document-aware rules): one view, deterministic per sequence.
            if cfg.inner_views != 1:
                raise ValueError("host-side inner masks support one view")
            inner_mask_c = to_chunks(host_mask, C)                                           # [nc, B, C]
            x_inner_c = jnp.where(inner_mask_c, mc.mask_token_id, ids_all_c)
        else:
            imask_fn = make_inner_mask(cfg.inner_mask_ratio, cfg.inner_mask_range, mc.mask_token_id, mc.pad_token_id,
                                       cfg.inner_views, cfg.inner_mask_span, cfg.inner_view_mode)
            x_inner_c, inner_mask_c = jax.vmap(jax.vmap(imask_fn))(ids_all_c, inner_keys)      # [nc, B, (V,) C]
        target_c = ids_all_c
        if cfg.inner_cart:
            # The write's loss weighted like the read's: CART density of visible neighbours
            # within the chunk (the mask stays boolean for the input construction).
            shp = inner_mask_c.shape
            flat = inner_mask_c.reshape(-1, C)
            inner_mask_c = time_weights("cart", None, flat, kernel if kernel is not None
                                        else cart_kernel(cfg.cart_p, cfg.cart_window)).reshape(shp)
    else:
        if cfg.inner_views != 1:
            raise ValueError("causal / split inner tasks are single-view")
        if cfg.inner_task == "split":
            if host_perm is None:
                raise ValueError("inner_task split needs Batch.inner_perm (host-side, inner_masks.split_perm)")
            perm_c = to_chunks(host_perm, C)                                                 # [nc, B, C] absolute positions
            K = int(round(cfg.inner_context_frac * C))
            own_mat = own_mask_matrix(C, f"split:{K}")
        else:
            perm_c = jnp.broadcast_to(jnp.arange(T).reshape(nc, 1, C), (nc, B, C))
            K = 0
            own_mat = own_mask_matrix(C, "causal")
        # gather the clean tokens / positions into the physical order (perm holds absolute positions)
        x_inner_c = jnp.take_along_axis(jnp.broadcast_to(ids[:, None, :], (B, nc, T)).swapaxes(0, 1),
                                        perm_c, axis=2)                                        # [nc, B, C]
        pos_in_c = jnp.take_along_axis(jnp.broadcast_to(pos[:, None, :], (B, nc, T)).swapaxes(0, 1), perm_c, axis=2)
        # slot i predicts the token at slot i+1: targets and query positions shifted by one
        target_c = jnp.concatenate([x_inner_c[:, :, 1:], x_inner_c[:, :, -1:]], axis=2)
        pos_q_c = jnp.concatenate([pos_in_c[:, :, 1:], pos_in_c[:, :, -1:]], axis=2)
        slot = jnp.arange(C)
        w = (slot >= max(K - 1, 0)) & (slot <= C - 2)                                          # predicting slots
        inner_mask_c = jnp.broadcast_to(w[None, None, :], (nc, B, C))
    V = cfg.inner_views
    if cfg.inner_feat != "off":
        if cfg.no_write or cfg.inner_from_outer:
            raise ValueError("inner_feat needs the inner stream (not no_write / inner_from_outer)")
        if model.inner_feat_head is None:
            raise ValueError("inner_feat needs a model with inner_feat_head (configs/1b_e2e_feat)")
    tgt_idx_c = feat_target_index(cfg.inner_task, nc, B, C, perm_c if cfg.inner_task == "split" else None) \
        if cfg.inner_feat != "off" else jnp.zeros((nc, B, C), jnp.int32)

    # ---- the distillation teacher: a fourth prefix stream, the chunk masked at a LOW
    # ratio (distill_t_range) and read bidirectionally - the reader near the end of its
    # denoising chain.  Its predictive distribution at the slots whose target it has
    # masked (top-k, unnormalised, held constant) becomes a soft target of the write
    # next to the one-hot CE: the write stores how seeing more of the document changes
    # the prediction.  Predicts like the outer stream (x_{i+1} at slot i under outer_shift).
    if cfg.distill:
        imask_T = make_inner_mask(-1.0, cfg.distill_t_range, mc.mask_token_id, mc.pad_token_id)
        if key_inner.ndim == 2:
            folded = jax.vmap(lambda k: jax.random.fold_in(k, 2))(key_inner)
            keys_T = jax.vmap(lambda k: jax.random.split(k, nc))(folded).transpose(1, 0, 2)
        else:
            keys_T = jax.random.split(jax.random.fold_in(key_inner, 2), nc * B).reshape(nc, B, 2)
        x_teacher_c, m_teacher_c = jax.vmap(jax.vmap(imask_T))(ids_all_c, keys_T)              # [nc, B, C]
    else:
        x_teacher_c = m_teacher_c = None
    # slot offsets: the student's target of slot i is x_{i + s_S}, the teacher's x_{i + s_T}
    s_S = 1 if cfg.inner_task == "causal" else 0
    s_T = 1 if cfg.outer_shift else 0
    # the write's MTP term (causal view: every slot may predict x_{i+2}, nothing is visible ahead)
    if cfg.mtp:
        slot = jnp.arange(C)
        mtp_tgt_c = jnp.concatenate([x_inner_c[:, :, 2:], x_inner_c[:, :, -2:]], axis=2)
        mtp_mask_c = jnp.broadcast_to((slot <= C - 3)[None, None, :], (nc, B, C)).astype(jnp.float32)
    else:
        mtp_tgt_c = mtp_mask_c = None

    # ---- segments
    n_seg_chunks = cfg.segment_chunks or nc
    if nc % n_seg_chunks:
        raise ValueError(f"{nc} chunks not a multiple of segment_chunks {n_seg_chunks}")
    if n_seg_chunks < Wn:
        raise ValueError(f"segment_chunks {n_seg_chunks} < window_chunks {Wn}: the carried window would be short")
    n_seg = nc // n_seg_chunks
    T_seg = n_seg_chunks * C
    nkv, d = mc.num_key_value_heads, mc.head_dim
    Lp = mc.num_hidden_layers - S

    # spmd_axis_name: the batch axis is the mesh axis, so a shard_map inside
    # (the Pallas attention kernel) sees its per-device slice of the batch.
    spmd = dict(spmd_axis_name=batch_axis_name(mesh)) if mesh is not None else {}
    lr_now = inner_lr(step, cfg.inner_lr, cfg.inner_lr_warmup_steps)
    # The soft term's weight, ramped from 0 when asked: from scratch the teacher is a weaker
    # predictor than the causal write for the first steps (it has not learned to use the
    # bidirectional context yet) and distilling it then destabilises the meta-gradient.
    distill_w = cfg.distill_weight
    if cfg.distill and cfg.distill_warmup_steps > 0:
        distill_w = cfg.distill_weight * jnp.minimum(1.0, step.astype(jnp.float32) / cfg.distill_warmup_steps)
    # With memory gdn the carried memory ``W`` is the gdn state [B, S, nh, dv, dk] (f32),
    # read by every stream and written by the clean forward (``update_memory``).
    def _sf(h, pos, kv, Wm, kvld, pq=None, update=False, prime=False, own_mask=None, own_kv=None):
        if use_gdn:
            return model.suffix_forward(h, pos, kv, None, kvld, pos_q=pq, memory_state=Wm, update_memory=update,
                                        own_mask=own_mask, own_kv=own_kv)
        return model.suffix_forward(h, pos, kv, Wm, kvld, pos_q=pq, return_prime=prime, own_mask=own_mask,
                                    own_kv=own_kv)
    # the key-value-binding write with clean-stream values reads the clean forward's per-layer inputs
    clean_prime = (not cfg.write.plain) and cfg.write.kvb_weight > 0 and cfg.write.kvb_value == "clean"
    # the sub-block read (ModelConfig.read_block): the clean stream block-causal, the outer stream over the
    # chunk's clean K/V (B's) and its own group; the masks static (numpy): cuDNN builds its kernels from them
    rb = mc.read_block
    if rb and cfg.distill:
        raise ValueError("read_block: no distillation teacher stream")
    clean_own = None if not rb else own_mask_matrix(C, f"block:{rb}")
    read_own = None if not rb else read_mask_matrix(C, rb)
    if cfg.outer_shift:
        fwd_outer = eqx.filter_vmap(lambda h, pos, kv, W, kvld, pq, ok: _sf(h, pos, kv, W, kvld, pq, own_mask=read_own,
                                                                            own_kv=ok),
                                    in_axes=(0, 0, 0, 0, None, 0, 0), **spmd)
    else:
        fwd_outer = eqx.filter_vmap(lambda h, pos, kv, W, kvld, ok: _sf(h, pos, kv, W, kvld, own_mask=read_own,
                                                                        own_kv=ok),
                                    in_axes=(0, 0, 0, 0, None, 0), **spmd)
    fwd_clean = eqx.filter_vmap(lambda h, pos, kv, W, kvld: _sf(h, pos, kv, W, kvld, update=use_gdn, prime=clean_prime,
                                                                own_mask=clean_own),
                                in_axes=(0, 0, 0, 0, None), **spmd)
    fwd_teacher = fwd_outer          # the teacher predicts like the outer stream (its query positions)

    def suffix_step(carry, xs):
        # no_write: W_0 is a constant, not a carry; memory gdn: the carry holds the gdn state
        kv, W = carry if (not cfg.no_write or use_gdn) else (carry, W0)
        c, ho, hc, hi, tok, wgt, tm, im, p, tgt, p_in, p_q, ti, ht, mt, tok2, wgt2, tm2, mtp_t, mtp_m, hv, wv, tmv = xs
        key_valid = key_validity(eff(c), C, Wn, reach)
        Wc = W if use_gdn else cast_floating(W, cdt)

        # A. outer loss: the read itself is computed after the write (C) below, so that the
        # self-prior sequences can read this chunk with the memory that already holds it
        p_q_outer = (p + 1) if cfg.outer_shift else None

        # T. the distillation teacher (forward only, held constant) -> the write's soft targets
        extras = None
        distill_mass = jnp.float32(0.0)
        if cfg.distill or cfg.mtp:
            soft_ids = soft_q = soft_mask = None
            if cfg.distill:
                if cfg.outer_shift:
                    hT, _ = fwd_teacher(ht, p, kv, Wc, key_valid, p_q_outer, None)
                else:
                    hT, _ = fwd_teacher(ht, p, kv, Wc, key_valid, None)
                hT = jax.lax.stop_gradient(hT)
                ids_k, q_k, mass = soft_targets(hT, model.lm_head_weight(), cfg.distill_topk, cfg.fused_ce_block)
                distill_mass = jnp.sum(mass)
                # align to the student's slots: slot i predicts x_{i + s_S}; the teacher's
                # prediction of that token stands at its slot i + s_S - s_T and counts
                # only where the teacher had it masked (mt[i + s_S])
                soft_ids, ok_a = shift_slots(ids_k, s_S - s_T)
                soft_q, _ = shift_slots(q_k, s_S - s_T)
                masked_t, ok_b = shift_slots(mt, s_S)
                first = im.reshape(B, -1, C)[:, 0] if im.ndim == 3 else im                     # [B, C] student slots
                soft_mask = (masked_t & (first > 0) & (ok_a & ok_b)[None, :]).astype(jnp.float32)
            extras = InnerExtras(soft_ids, soft_q, soft_mask, mtp_t, mtp_m)

        # B. clean -> next KV window (shift: drop the oldest chunk, append this one); with memory
        # gdn also the state written from this clean chunk (the "no-memory twin" of the
        # evaluation, inner_lr 0, keeps the state at zero)
        resB = fwd_clean(hc, p, kv, Wc, key_valid)
        own = resB[1]                                                                             # [B, S, C, nkv, d]
        kv_next = tuple(jnp.concatenate([cache[:, :, C:], o], axis=2) for cache, o in zip(kv, own))
        if use_gdn:
            mem_next = jnp.where(lr_now > 0, resB[2], W)

        # C. inner update (absent with no_write: W stays W_0 for the whole sequence)
        plan_ce = jnp.zeros((2,), jnp.float32)
        if cfg.no_write:
            W_next, st = (mem_next if use_gdn else W), InnerStats(*(jnp.float32(0.0) for _ in range(7)))
        elif cfg.write_off:
            # The inner stream is computed and its CE reported (and, with aux_causal_weight,
            # trained on by the shared weights) but W stays W_0: the "auxiliary loss without a
            # write" control that separates the write task's training signal from the write.
            pq = None if cfg.inner_task == "mask" else p_q
            f = lambda w, h, t, m, k, pp, q: inner_objective(w, model, h, t, m, k, pp, key_valid, cfg.fused_ce_block, cdt,
                                                             None, q, own_mat)[1][0]
            ce_b = eqx.filter_vmap(f, in_axes=(0, 0, 0, 0, 0, 0, None if pq is None else 0))(W, hi, tgt, im, kv, p_in, pq)
            W_next, st = W, InnerStats(jnp.sum(ce_b), *(jnp.float32(0.0) for _ in range(6)))
        elif cfg.inner_from_outer:
            # the write is the gradient of this chunk's OUTER objective (A's input, mask and
            # cart weights): write and read share one distribution, no third stream
            W_next, st = inner_update(W, model, ho, tok, wgt.astype(jnp.float32), kv, p, key_valid, lr_now,
                                      cfg.inner_clip, cfg.fused_ce_block, cdt, cfg.inner_update, h_clean=hc,
                                      pos_q=p_q_outer, frac=cfg.inner_update_frac)
        elif V > 1 and cfg.inner_views_sequential:
            # V SGD steps on the chunk, one view each (a stochastic descent on the chunk's
            # masked-diffusion objective at the reader's own operating point); lr / V and
            # clip / V per step so V identical gradients reproduce one step of lr, clip.
            feat_tgt = jax.vmap(make_feat_target)(hc, ti) if cfg.inner_feat != "off" else None
            def view_step(w, h_v, m_v):
                return inner_update(w, model, h_v, tgt, m_v, kv, p_in, key_valid, lr_now / V, cfg.inner_clip / V,
                                    cfg.fused_ce_block, cdt, cfg.inner_update, h_clean=hc,
                                    feat_tgt=feat_tgt, feat_mode=cfg.inner_feat, feat_weight=cfg.inner_feat_weight,
                                    frac=cfg.inner_update_frac)
            W_next, sts = W, []
            for v in range(V):
                W_next, st_v = jax.checkpoint(view_step)(W_next, hi[:, v], im[:, v])
                sts.append(st_v)
            st = InnerStats(*(sum(x) / V for x in zip(*sts)))
            if not cfg.second_order:
                W_next = jax.tree_util.tree_map(jax.lax.stop_gradient, W_next)
        else:
            feat_tgt = jax.vmap(make_feat_target)(hc, ti) if cfg.inner_feat != "off" else None
            if not cfg.write.plain:
                # sub-chunk / surprise / key-value-binding / two-timescale writes (inner_loop.write_chunk,
                # shared with generate.commit_block)
                W_next, st = write_chunk(W, model, hi, tgt, im, kv, p_in, key_valid, lr_now, cfg.inner_clip,
                                         cfg.fused_ce_block, cdt, cfg.inner_update, hc, p_q, own_mat, cfg.write,
                                         W0=W0, kvb_src=resB[2][0] if clean_prime else None)
            else:
                def plain_step(w):
                    return inner_update(w, model, hi, tgt, im, kv, p_in, key_valid, lr_now, cfg.inner_clip,
                                        cfg.fused_ce_block, cdt, cfg.inner_update, h_clean=hc,
                                        pos_q=None if cfg.inner_task == "mask" else p_q, own_mask=own_mat,
                                        feat_tgt=feat_tgt, feat_mode=cfg.inner_feat, feat_weight=cfg.inner_feat_weight,
                                        frac=cfg.inner_update_frac, extras=extras, distill_weight=distill_w,
                                        mtp_weight=cfg.inner_mtp_weight, distill_meta=cfg.distill_meta)
                if cfg.write_plan:
                    # Evaluation-time write plan: plan[c] whole-chunk steps on chunk c (0: chunk c is
                    # not written), the gradient recomputed each step; for the chunks written more than
                    # once, their own causal CE before the first and after the last step (what W holds
                    # of the chunk when read with the write's own keys).
                    k_c = plan_arr[c]
                    W1, st = plain_step(W)
                    Wk = jax.lax.fori_loop(1, jnp.maximum(k_c, 1), lambda _, w: plain_step(w)[0], W1)
                    W_next = jax.tree_util.tree_map(lambda a, b: jnp.where(k_c > 0, b, a), W, Wk)
                    post = jax.lax.cond(k_c > 1, lambda w: plain_step(w)[1].loss, lambda w: jnp.float32(0.0), Wk)
                    plan_ce = jnp.stack([st.loss, post]) * (k_c > 1).astype(jnp.float32)
                else:
                    W_next, st = plain_step(W)
            if not cfg.second_order:
                W_next = jax.tree_util.tree_map(jax.lax.stop_gradient, W_next)

        # A (continued). the read: W as it was before this chunk's write, or - self-prior sequences - after it
        if sp_flags is None:
            Wc_read = Wc
        else:
            W_read = jax.tree_util.tree_map(
                lambda a, b: jnp.where(sp_flags.reshape((B,) + (1,) * (a.ndim - 1)), b, a) if eqx.is_inexact_array(a) else a,
                W, W_next)
            Wc_read = cast_floating(W_read, cdt)
        if cfg.outer_shift:
            hA, _ = fwd_outer(ho, p, kv, Wc_read, key_valid, p_q_outer, own if rb else None)
        else:
            hA, _ = fwd_outer(ho, p, kv, Wc_read, key_valid, own if rb else None)
        if cfg.per_sequence:
            ce_seq = jax.vmap(lambda h, t, w: weighted_ce(h, model.lm_head_weight(), t, w.astype(jnp.float32),
                                                          cfg.fused_ce_block))(hA, tok, wgt)   # [B]
            ce_sum = jnp.sum(ce_seq)
            mask_seq = jnp.sum(tm.astype(jnp.float32), axis=1)
        else:
            ce_sum = weighted_ce(hA.reshape(B * C, -1), model.lm_head_weight(),
                                 tok.reshape(-1), wgt.reshape(-1).astype(jnp.float32), cfg.fused_ce_block)
            ce_seq = mask_seq = jnp.zeros((B,), jnp.float32)
        if cfg.mtp and cfg.mtp_weight > 0:
            h2 = model.mtp_head(hA)
            ce2_sum = weighted_ce(h2.reshape(B * C, -1), model.lm_head_weight(),
                                  tok2.reshape(-1), wgt2.reshape(-1).astype(jnp.float32), cfg.fused_ce_block)
            mask2_count = jnp.sum(tm2.astype(jnp.float32))
        else:
            ce2_sum = mask2_count = jnp.float32(0.0)
        # the extra noisy reads: the same window and memory, their own masks; their CE and scored slots add in
        mask_count = jnp.sum(tm.astype(jnp.float32))
        for k in range(VO - 1):
            if cfg.outer_shift:
                hV, _ = fwd_outer(hv[:, k], p, kv, Wc_read, key_valid, p_q_outer, None)
            else:
                hV, _ = fwd_outer(hv[:, k], p, kv, Wc_read, key_valid, None)
            ce_sum = ce_sum + weighted_ce(hV.reshape(B * C, -1), model.lm_head_weight(), tok.reshape(-1),
                                          wv[:, k].reshape(-1).astype(jnp.float32), cfg.fused_ce_block)
            mask_count = mask_count + jnp.sum(tmv[:, k].astype(jnp.float32))

        kv_next = constrain_batch_axis(kv_next, mesh)
        carry = kv_next if (cfg.no_write and not use_gdn) else (kv_next, constrain_batch_axis(W_next, mesh))
        slot = (jnp.arange(nc) == c).astype(jnp.float32)
        if cfg.per_token:
            ce_t = per_token_ce(hA.reshape(B * C, -1), model.lm_head_weight(), tok.reshape(-1),
                                cfg.fused_ce_block or 8192).reshape(B, C)
            ce_tok = ce_t[:, None, :] * slot[None, :, None]                              # [B, nc, C]
            mask_tok = tm.astype(jnp.float32)[:, None, :] * slot[None, :, None]
        else:
            ce_tok = mask_tok = jnp.zeros((1,), jnp.float32)
        return carry, _Ys(ce_sum, mask_count, st, ce_sum * slot, mask_count * slot, ce_seq, mask_seq, ce_tok, mask_tok,
                          ce2_sum, mask2_count, distill_mass, plan_ce)

    def segment(carry, xs):
        kvp, kvs, W = carry                           # prefix windows (None: single segment), suffix window, W
        (g, ids3, p, xi_c, im_c, wgt_c, tm_c, tgt_c, pin_c, pq_c, ti_c, tgt_o_c,
         mt_c, tok2_c, wgt2_c, tm2_c, mtp_t_c, mtp_m_c, wv_c, tmv_c) = xs   # segment index, [B, n_streams, T_seg], [B, T_seg], [nc_s, B, C] ...
        # ---- prefix: all chunks of the segment, all streams, in parallel
        kva = key_validity_all(n_seg_chunks, C, Wn, first=eff(g * n_seg_chunks), reach=reach)
        n_teacher = 1 if cfg.distill else 0
        if not cfg.outer_shift and (cfg.inner_task == "mask" or cfg.no_write or cfg.inner_from_outer):
            sp = spq = None
        else:
            p_in = from_chunks(pin_c) if (pin_c is not None and cfg.inner_task != "mask") else p
            p_q = from_chunks(pq_c) if (pq_c is not None and cfg.inner_task != "mask") else p
            p_q_outer = p + 1 if cfg.outer_shift else p
            # the extra noisy reads (last) take the read's key and query positions
            if cfg.no_write or cfg.inner_from_outer:
                sp = jnp.stack([p, p] + [p] * (VO - 1), axis=1)
                spq = jnp.stack([p_q_outer, p] + [p_q_outer] * (VO - 1), axis=1)
            elif V == 1:
                sp = jnp.stack([p, p, p_in] + [p] * (n_teacher + VO - 1), axis=1)
                spq = jnp.stack([p_q_outer, p, p_q] + [p_q_outer] * (n_teacher + VO - 1), axis=1)
            else:
                sp = jnp.stack([p, p] + [p_in] * V + [p] * (VO - 1), axis=1)
                spq = jnp.stack([p_q_outer, p] + [p_q] * V + [p_q_outer] * (VO - 1), axis=1)
        own = "full" if cfg.inner_task == "mask" else ("causal" if cfg.inner_task == "causal"
                                                       else f"split:{int(round(cfg.inner_context_frac * C))}")
        prefix = lambda i3, pp, kw, sp_, spq_: model.prefix_forward(
            i3, pp, kva, cfg.attn_chunk_batch, cdt, cfg.remat_prefix, cfg.prefix_remat_levels, kv_win=kw,
            stream_pos=sp_, stream_pos_q=spq_, inner_own=own, extra_full=n_teacher + VO - 1)
        ax_sp = None if sp is None else 0
        if kvp is None:
            h3, kvp_next = eqx.filter_vmap(lambda i3, pp, sp_, spq_: prefix(i3, pp, None, sp_, spq_),
                                           in_axes=(0, 0, ax_sp, ax_sp), **spmd)(ids3, p, sp, spq)
        else:
            h3, kvp_next = eqx.filter_vmap(prefix, in_axes=(0, 0, 0, ax_sp, ax_sp), **spmd)(ids3, p, kvp, sp, spq)
        h_outer = constrain_batch_axis(to_chunks(h3[:, OUTER], C), mesh, axis=1)          # [nc_s, B, C, H]
        h_clean = constrain_batch_axis(to_chunks(h3[:, CLEAN], C), mesh, axis=1)
        if cfg.no_write or cfg.inner_from_outer:
            h_inner = None
        elif V == 1:
            h_inner = constrain_batch_axis(to_chunks(h3[:, INNER], C), mesh, axis=1)
        else:   # [nc_s, B, V, C, H]: the batch axis stays second for the scan / vmap
            h_inner = constrain_batch_axis(jnp.stack([to_chunks(h3[:, INNER + v], C) for v in range(V)], axis=2),
                                           mesh, axis=1)
        h_teacher = constrain_batch_axis(to_chunks(h3[:, INNER + V], C), mesh, axis=1) if cfg.distill else None
        n_st = h3.shape[1]                                                                # [nc_s, B, VO-1, C, H]
        h_views = (constrain_batch_axis(jnp.stack([to_chunks(h3[:, n_st - VO + 1 + k], C) for k in range(VO - 1)],
                                                  axis=2), mesh, axis=1) if VO > 1 else None)
        # ---- suffix scan over the segment's chunks
        chunk_idx = g * n_seg_chunks + jnp.arange(n_seg_chunks)
        sxs = (chunk_idx, h_outer, h_clean, h_inner, tgt_o_c, wgt_c, tm_c, im_c, to_chunks(p, C), tgt_c, pin_c, pq_c, ti_c,
               h_teacher, mt_c, tok2_c, wgt2_c, tm2_c, mtp_t_c, mtp_m_c, h_views, wv_c, tmv_c)
        factors = balanced_factors(n_seg_chunks, max(1, cfg.remat_levels))
        if cfg.no_write and not use_gdn:
            kvs_next, ys = nested_remat_scan(suffix_step, kvs, sxs, factors)
            W_next = W
        else:
            (kvs_next, W_next), ys = nested_remat_scan(suffix_step, (kvs, W), sxs, factors)
        if cfg.truncate_fast_weight:
            W_next = jax.tree_util.tree_map(jax.lax.stop_gradient, W_next)
        kvp_next = None if kvp is None else constrain_batch_axis(kvp_next, mesh)
        return (kvp_next, kvs_next, W_next), ys

    # ---- carries at the sequence start
    kvp0 = tuple(jnp.zeros((B, Lp, Wn * C, nkv, d), cdt) for _ in range(2))
    kvs0 = tuple(jnp.zeros((B, S, Wn * C, nkv, d), cdt) for _ in range(2))
    if use_gdn:
        # the memory carry: every suffix layer's gdn state, empty at the sequence start
        W0 = jnp.zeros((B, S, mc.gdn_heads, mc.gdn_head_dim, mc.gdn_head_dim), jnp.float32)
    else:
        W0 = cast_floating(jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (B,) + x.shape), model.fast_weight),
                           jnp.float32)
    kvp0, kvs0, W0 = (constrain_batch_axis(x, mesh) for x in (kvp0, kvs0, W0))

    # the extra views' CE weights / scored slots per chunk, [nc, B, VO-1, C] (None with one view)
    views_c = lambda i: jnp.stack([to_chunks(v[i], C) for v in extra_views], axis=2) if extra_views else None

    # ---- per-segment inputs, segment axis first
    if cfg.no_write or cfg.inner_from_outer:
        inner_streams = []
    elif V == 1:
        inner_streams = [from_chunks(x_inner_c)]
    else:
        inner_streams = [from_chunks(x_inner_c[:, :, v]) for v in range(V)]
    teacher_streams = [from_chunks(x_teacher_c)] if cfg.distill else []
    view_streams = [v[0] for v in extra_views]                                            # the extra noisy reads, last
    ids3 = jnp.stack([x_t, ids, *inner_streams, *teacher_streams, *view_streams], axis=1)  # [B, 2+V(+1)(+VO-1), T]
    by_seg = lambda x: jnp.moveaxis(x.reshape(*x.shape[:-1], n_seg, T_seg), -2, 0)       # [.., T] -> [n_seg, .., T_seg]
    by_seg_c = lambda x: None if x is None else x.reshape(n_seg, n_seg_chunks, *x.shape[1:])   # [nc, ..] -> [n_seg, nc_s, ..]
    opt_c = lambda x: None if x is None else by_seg_c(to_chunks(x, C))
    xs = (jnp.arange(n_seg), by_seg(ids3), by_seg(pos), by_seg_c(x_inner_c), by_seg_c(inner_mask_c),
          by_seg_c(to_chunks(weight, C)), by_seg_c(to_chunks(m_outer, C)), by_seg_c(target_c), by_seg_c(pos_in_c),
          by_seg_c(pos_q_c if pos_q_c is not None else pos_in_c), by_seg_c(tgt_idx_c),
          by_seg_c(to_chunks(tgt_outer, C)),
          by_seg_c(m_teacher_c), opt_c(tgt_mtp), opt_c(w_mtp), opt_c(m_mtp), by_seg_c(mtp_tgt_c), by_seg_c(mtp_mask_c),
          by_seg_c(views_c(5)), by_seg_c(views_c(4)))
    if n_seg == 1:
        # The whole sequence at once: no prefix window carry, no outer scan (the 8K program).
        (_, _, W_f), ys = segment((None, kvs0, W0), jax.tree_util.tree_map(lambda x: x[0], xs))
    else:
        (_, _, W_f), ys = jax.lax.scan(jax.checkpoint(segment), (kvp0, kvs0, W0), xs)
        ys = jax.tree_util.tree_map(lambda y: jnp.sum(y, axis=0), ys)

    main_loss = ys.ce_sum / (ys.mask_count + 1e-6)
    mtp_loss = ys.ce2_sum / (ys.mask2_count + 1e-6)
    # The optimised objective adds the outer MTP term; every reported "loss" number other
    # than the training line's stays the main CE (evaluation runs with mtp_weight 0).
    loss = main_loss + cfg.mtp_weight * mtp_loss if cfg.mtp and cfg.mtp_weight > 0 else main_loss
    n_inner = nc * B
    n_plan = sum(1 for k in cfg.write_plan[:nc] if k > 1)
    if cfg.aux_causal_weight > 0:
        # The inner stream's mean CE (before any write) as an auxiliary training objective of
        # the shared weights - with write_off, the "is it just an auxiliary loss?" control.
        loss = loss + cfg.aux_causal_weight * ys.inner.loss / n_inner
    if use_gdn:
        # memory gdn: the "drift" is the RMS of the final states (they start at zero)
        mem_rms = jnp.sqrt(jnp.mean(jnp.square(W_f)))
        drift = {"ttt_drift_rel": mem_rms, "ttt_delta_norm": jnp.sqrt(jnp.sum(jnp.square(W_f))), "ttt_init_norm": jnp.float32(0.0)}
    else:
        drift = drift_stats(W_f, W0)
    stats = {
        **drift,
        "inner_loss": ys.inner.loss / n_inner,
        **({"inner_plan_pre": ys.plan_ce[0] / (B * n_plan), "inner_plan_post": ys.plan_ce[1] / (B * n_plan)}
           if n_plan else {}),
        "inner_grad_norm": ys.inner.grad_norm / n_inner,
        "inner_clipped_frac": ys.inner.clipped / n_inner,
        "inner_feat": ys.inner.feat / n_inner,
        "inner_distill": ys.inner.distill / n_inner,
        "inner_mtp": ys.inner.mtp / n_inner,
        "inner_kvb": ys.inner.kvb / n_inner,
        "main_loss": main_loss,
        "mtp_loss": mtp_loss,
        "mtp_mask_count": ys.mask2_count,
        "distill_mass": ys.distill_mass / n_inner,
        "inner_lr": lr_now,
        "masked_frac": ys.mask_count / (B * T0 * VO),
        "mask_count": ys.mask_count,
        # Loss by chunk position: whether later chunks (more history through the
        # window and the fast weight) are easier than the first.  Non-scalar; the
        # trainer logs it only from the evaluation.  Original coordinates (the re-read
        # block, never scored, is dropped).
        "loss_by_chunk": (ys.ce_chunk / (ys.mask_chunk + 1e-6))[n_rr:],
        "mask_by_chunk": ys.mask_chunk[n_rr:],
    }
    if cfg.per_sequence:
        # Replicated so every host can read it (evaluation reports a paired standard error).
        stats["loss_by_seq"] = constrain_replicated(ys.ce_seq / (ys.mask_seq + 1e-6), mesh)
        stats["mask_by_seq"] = constrain_replicated(ys.mask_seq, mesh)
    if cfg.per_token:
        ce_flat = ys.ce_tok.reshape(B, T)[:, n_rr * C:]
        mask_flat = ys.mask_tok.reshape(B, T)[:, n_rr * C:]
        if cfg.outer_shift:
            stats["ce_by_token"] = constrain_replicated(jnp.pad(ce_flat[:, :-1], ((0, 0), (1, 0))), mesh)
            stats["mask_by_token"] = constrain_replicated(jnp.pad(mask_flat[:, :-1], ((0, 0), (1, 0))), mesh)
        else:
            stats["ce_by_token"] = constrain_replicated(ce_flat, mesh)
            stats["mask_by_token"] = constrain_replicated(mask_flat, mesh)
    return loss, stats
