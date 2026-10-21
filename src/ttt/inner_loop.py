# coding=utf-8
"""
The test-time-training inner loop: one clipped step on the fast weight per chunk.

    g      = grad_W  mean_{i in inner_mask} CE( suffix(h_inner; W)_i , tokens_i )
             (averaged over the inner views when there are two)
    u      = direction(g)                       sgd: g | sign: g/sqrt(g^2+eps) | muon2: orthogonalised g
    u     <- u * min(1, clip / ||u||)          ONE global norm over all 3*S leaves
    W_l   <- W_l - eta_l * u_l - lambda_l (W_l - W0_l)      in f32, per suffix layer l

``eta_l = lr(step) * 2 sigmoid(gate_l([mean_c h_clean, sg(loss)/8, ratio]))`` when the
model has a chunk gate (else ``lr``): the chunk's content (the CLEAN stream's suffix
input - mask-independent, what inference has), its surprise (the inner loss before
the update, a feature under stop_gradient) and the realised mask ratio (the loss
scales with it).  With a token gate the inner CE is weighted per masked position by
``w_i = 2 sigmoid(u . h_clean_i)`` - a differentiable choice of WHAT to write.
``lambda_l = 0.1 sigmoid(decay_logit_l)`` when the model has a decay (else 0; not on
the main line).  Gates, decay and ``W0`` are outer parameters, so the meta-gradient
sets the write strength, focus and forgetting.  Every direction is rescaled to the
norm the clipped SGD step would have, so the rules differ only in direction.

Everything here is per sequence; ``inner_update`` vmaps it over the batch.  The
loss is computed with ``W`` cast to the model's compute dtype, and the update is
applied to the f32 carry.  The meta-gradient flows through this step unless
``second_order`` is off (handled by the caller with a ``stop_gradient``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from .fused_ce import per_token_ce
from .model import FastWeight, Model, cast_floating
from .objective import soft_weighted_ce, weighted_ce


class InnerStats(NamedTuple):
    """Per-chunk diagnostics, summed across the scan by ``nested_remat_scan``."""
    loss: Array          # inner loss, batch-summed
    grad_norm: Array     # pre-clip global norm, batch-summed
    clipped: Array       # number of sequences whose step was clipped
    feat: Array          # feature-prediction loss, batch-summed (0 without the feature task)
    distill: Array       # sum_k q_k log(q_k / p_k) over the teacher's top-k (~ KL when its mass is ~1), batch-summed
    mtp: Array           # second-target (x_{i+2}) CE of the write, batch-summed (0 without MTP)
    kvb: Array = 0.0     # key-value-binding term of the write (WriteConfig.kvb_weight), batch-summed


@dataclass(frozen=True)
class WriteConfig:
    """How the causal write of a chunk is carried out when it is more than one clipped step on
    the whole chunk (``write_chunk``).  The defaults are exactly that single step.

    * ``subchunks`` n: the chunk's write is n sequential clipped steps over its n consecutive
      sub-chunks (sub-chunk k's targets, its forward over slots ``[0, end_k)`` with the weights
      after steps 0..k-1) - finer granularity and n times the step budget per chunk
      (``subchunk_scale`` = 1/n keeps the budget of one step).
    * ``surprise_gamma`` > 0: slot weights ``(CE_i / mean CE)^gamma`` from the model's own loss
      before the step (held constant): the fixed-norm step goes to what the model cannot
      predict yet - the document-specific tokens.
    * ``kvb_weight`` > 0: + lambda * the key-value-binding term of every suffix layer,
      ``|| post_norm(W_l(x_l)) - theta_V,l v_l ||^2`` with the key = the fast weight's own input
      ``x_l`` (the read path is unchanged, the "simplified output rule" of TTT-KVB) and the value
      source ``v_l`` = the same input (``self``) or the CLEAN stream's input at the same
      position and layer (``clean``: bidirectional, it already knows the next token), mapped
      by a learned ``inner_kvb_value`` head: per-token bindings stored in the same W.
    * ``fast_frac`` f > 0: a two-timescale fast weight - a fixed fraction f of every W's
      hidden units is written by the n sub-chunk steps (fast: local detail, lr x
      ``fast_lr_scale``, relaxing toward W_0 by ``fast_decay`` after each chunk), the others by
      one step on the whole chunk (slow: the document prior).  Needs ``subchunks`` >= 2.
    * ``steps`` K > 1: K sequential clipped steps on the whole chunk, the gradient recomputed
      with the weights after the previous steps (each step lr x clip): the write fitted further
      to the chunk - an evaluation-time probe of how much more a chunk's write can store.
    * ``norm_alpha`` a > 0 with ``norm_cap`` c: the normalised step - an anchored, capped diagonal
      preconditioner from the chunk's own keys.  Per fast-weight layer, the hidden units' activation RMS
      r_u over the chunk's written slots and the input dimensions' RMS r_d (at the write's forward, held
      constant) give f_u = min(c, (max r / r_u)^a), f_d likewise; after the clip, down rows take f_u and
      gate / up entries min(f_d, f_u).  The dominant units keep the step (the document prior); weaker ones
      - the chunk's associations - take up to c times it (2026-10-05..08: one whole-chunk step; the listed
      two-channel split, whitening, novelty / top-k / rho^2 weights and two normalised steps per chunk were
      screened and removed; so were per-layer caps and the step without its mean-key component, 2026-10-11 / 12)."""
    subchunks: int = 1
    subchunk_scale: float = 1.0
    surprise_gamma: float = 0.0
    kvb_weight: float = 0.0
    kvb_value: str = "self"
    fast_frac: float = 0.0
    fast_lr_scale: float = 1.0
    fast_decay: float = 0.0
    steps: int = 1
    norm_alpha: float = 0.0
    norm_cap: float = 1.0

    @property
    def normalised(self) -> bool:
        return self.norm_alpha > 0

    @property
    def plain(self) -> bool:
        return self == WriteConfig()

    def check(self, chunk: int) -> None:
        if self.subchunks < 1 or chunk % self.subchunks:
            raise ValueError(f"inner_subchunks {self.subchunks} must divide the chunk size {chunk}")
        if not 0.0 < self.subchunk_scale <= 1.0 or self.surprise_gamma < 0 or self.kvb_weight < 0:
            raise ValueError("inner_subchunk_scale in (0, 1], inner_surprise_gamma >= 0, inner_kvb_weight >= 0")
        if self.kvb_value not in ("self", "clean"):
            raise ValueError(f"inner_kvb_value {self.kvb_value!r} (self | clean)")
        if not 0.0 <= self.fast_frac < 1.0 or not 0.0 <= self.fast_decay <= 1.0 or self.fast_lr_scale < 0:
            raise ValueError("inner_fast_frac in [0, 1), inner_fast_decay in [0, 1], inner_fast_lr_scale >= 0")
        if self.fast_frac > 0 and self.subchunks < 2:
            raise ValueError("inner_fast_frac needs inner_subchunks >= 2 (the fast units are written per sub-chunk)")
        if self.steps < 1 or (self.steps > 1 and (self.subchunks > 1 or self.fast_frac > 0)):
            raise ValueError("inner_write_steps >= 1, and > 1 only with whole-chunk steps (no sub-chunks / fast units)")
        if self.norm_alpha < 0 or self.norm_cap < 1.0 or (self.normalised and (
                self.subchunks > 1 or self.fast_frac > 0 or self.steps > 1)):
            raise ValueError("inner_norm_alpha >= 0, inner_norm_cap >= 1, and the normalised step alone, as one "
                             "whole-chunk step")


class InnerExtras(NamedTuple):
    """Side inputs of the inner objective's optional terms, one chunk of one sequence
    (``inner_update`` batches them); a term is off when its fields are ``None``.

    Distillation: ``soft_ids / soft_q [C, K]`` are the teacher's top-k tokens and their
    probabilities (unnormalised: the dropped tail contributes nothing) for the token slot
    i predicts, ``soft_mask [C]`` the slots that carry the soft term.  MTP: ``mtp_tgt [C]`` is slot i's second target
    (``x_{i+2}``), ``mtp_mask [C]`` the slots scored on it."""
    soft_ids: Optional[Array] = None
    soft_q: Optional[Array] = None
    soft_mask: Optional[Array] = None
    mtp_tgt: Optional[Array] = None
    mtp_mask: Optional[Array] = None


def inner_lr(step: Array, lr: float, warmup_steps: int, total_steps: int = 0,
             warmup_ratio: float = 0.0) -> Array:
    """Linear ``0.1*lr -> lr`` over ``warmup_steps`` OUTER steps, then flat.

    ``warmup_steps == 0`` disables the ramp (context-extension stage).
    ``warmup_steps < 0`` derives it from ``warmup_ratio * total_steps``.
    """
    if warmup_steps == 0:
        return jnp.float32(lr)
    w = warmup_steps if warmup_steps > 0 else max(1, int(total_steps * warmup_ratio))
    init = 0.1 * lr
    ramp = init + (lr - init) * (step.astype(jnp.float32) / w)
    return jnp.where(step < w, ramp, jnp.float32(lr))


def token_weights(model: Model, h_clean: Array) -> Array:
    """``[C]`` per-token weights of the inner loss from the clean hidden states (1.0
    without a token gate or at its zero initialisation)."""
    if model.inner_token_gate is None or h_clean is None:
        return jnp.ones((h_clean.shape[0] if h_clean is not None else 1,), jnp.float32)
    return 2.0 * jax.nn.sigmoid(model.inner_token_gate(h_clean.astype(jnp.float32))[:, 0])


def _bind_err_plain(w: FastWeight, ppn, x: Array, v: Array) -> Array:
    """``[C]``: one layer's read contribution ``post_norm(W(x))`` against the value ``v``, mean
    squared error over the hidden dimension."""
    return jnp.mean(jnp.square(ppn(w(x)).astype(jnp.float32) - v.astype(jnp.float32)), axis=-1)


@jax.custom_vjp
def _bind_err(w: FastWeight, ppn, x: Array, v: Array) -> Array:
    """``_bind_err_plain`` with a layer-local inner gradient, as in TTT-KVB: the binding of layer l
    moves W_l only - its key ``x`` (the layer's input, which depends on the earlier layers' W) and
    value ``v`` are inputs of the inner problem, so their cotangents are zero.  The backward rule is
    itself differentiated by the meta-gradient, which therefore still reaches the keys, the values
    and theta_V through the write."""
    return _bind_err_plain(w, ppn, x, v)


def _bind_err_fwd(w, ppn, x, v):
    return _bind_err_plain(w, ppn, x, v), (w, ppn, x, v)


def _bind_err_bwd(res, g):
    w, ppn, x, v = res
    _, vjp = jax.vjp(lambda w_: _bind_err_plain(w_, ppn, x, v), w)
    (gw,) = vjp(g)
    return gw, jax.tree_util.tree_map(jnp.zeros_like, ppn), jnp.zeros_like(x), jnp.zeros_like(v)


_bind_err.defvjp(_bind_err_fwd, _bind_err_bwd)


def inner_objective(W: FastWeight, model: Model, h_inner: Array, tokens: Array, inner_mask: Array,
                    kv: Tuple[Array, Array], pos: Array, key_valid: Array, fused_block: int,
                    compute_dtype, tok_w: Optional[Array] = None, pos_q: Optional[Array] = None,
                    own_mask: Optional[Array] = None, feat_tgt: Optional[Array] = None,
                    feat_mode: str = "off", feat_weight: float = 1.0, extras: Optional[InnerExtras] = None,
                    distill_weight: float = 1.0, mtp_weight: float = 1.0, soft_only: bool = False,
                    surprise_gamma: float = 0.0, kvb_weight: float = 0.0, kvb_src: Optional[Array] = None,
                    score_from: int = 0) -> Tuple[Array, Tuple[Array, Array, Array, Array, Array]]:
    """The inner objective of one chunk, differentiable w.r.t. ``W``:
    ``(objective, (ce, feat, distill, mtp, kvb))``; ``soft_only``: the objective is the
    distillation term alone (``distill_weight * soft``; the others are still reported).
    ``surprise_gamma`` / ``kvb_weight`` / ``kvb_src [S, C, H]``: the surprise weighting and the
    key-value-binding term of ``WriteConfig`` (``kvb_src`` None = the stream's own fast-weight
    inputs, else the given value sources, held constant).  ``score_from`` (static): the slots
    before it are forward context only (their weights must be zero) and skip the vocabulary
    projection and the binding term - a sub-chunk step of ``write_chunk``.

    ``ce`` is the weighted mean CE over the inner targets: ``tokens [C]`` are the targets of
    the C slots, ``inner_mask [C]`` (bool or float) the slots that count; ``tok_w [C]``
    (token gate) weights them further.  For the masked task the targets are the chunk's
    tokens at the masked slots; for the causal / split tasks the slots hold clean tokens in
    the physical (permuted) order, slot i predicts the token at slot i+1 (``tokens`` already
    shifted), its query stands at that token's position (``pos_q``) and ``own_mask``
    restricts the own-chunk attention.  ``h_inner [C, H]`` / ``inner_mask [C]`` for one
    view, ``[V, C, H]`` / ``[V, C]`` for several (mean of the per-view losses).

    ``feat`` is the feature-prediction term: the head's prediction from the same hidden
    state against ``feat_tgt [C, H]`` (the clean stream's RMS-normalised suffix input at the
    target slot, a constant), weighted mean over the slots of the squared error per
    dimension.  ``feat_mode``: "off" (objective = ce), "add" (ce + feat_weight * feat),
    "only" (feat).

    ``extras`` (``InnerExtras``) adds two further terms, each a weighted mean over its own
    slots and scaled by its weight: the soft cross-entropy against a teacher's top-k
    distribution of the slot's target (``distill``, reported as ``sum_k q_k log(q_k / p_k)``
    over that support, the KL to the teacher when its mass is ~1) and the CE of the
    slot's second target through ``mtp_head`` (``mtp``).  Both targets are constants (the teacher is ``stop_gradient``-ed by the
    caller); the anchor ``ce`` is kept."""
    Wc = cast_floating(W, compute_dtype)
    if feat_mode != "off" and (feat_tgt is None or model.inner_feat_head is None):
        raise ValueError("feature-prediction inner task needs feat_tgt and a model with inner_feat_head")
    if extras is not None and extras.mtp_tgt is not None and model.mtp_head is None:
        raise ValueError("the MTP inner term needs a model with mtp_head")
    if kvb_weight > 0 and model.inner_kvb_value is None:
        raise ValueError("the key-value-binding write needs a model with inner_kvb_head")
    if score_from and (feat_mode != "off" or extras is not None):
        raise ValueError("score_from supports the token-CE / surprise / binding objective only")
    tokens_all = tokens

    def one(h_in, m):
        # This forward's gradient is itself differentiated by the meta-gradient;
        # `inner` makes the attention take its second-order-safe (XLA) path.
        if kvb_weight > 0:
            h, _, (x_l, _) = model.suffix_forward(h_in, pos, kv, Wc, key_valid, inner=True, pos_q=pos_q,
                                                  own_mask=own_mask, return_prime=True)
        else:
            h, _ = model.suffix_forward(h_in, pos, kv, Wc, key_valid, inner=True, pos_q=pos_q, own_mask=own_mask)
        tokens, tw, ks = tokens_all, tok_w, kvb_src
        if score_from:
            h, m, tokens = h[score_from:], m[score_from:], tokens[score_from:]
            tw = None if tok_w is None else tok_w[score_from:]
            if kvb_weight > 0:
                x_l = x_l[:, score_from:]
                ks = None if kvb_src is None else kvb_src[:, score_from:]
        w = m.astype(jnp.float32) if tw is None else m.astype(jnp.float32) * tw
        if surprise_gamma > 0:
            # Surprise weighting: slot weights (CE_i / mean CE)^gamma of the model before the
            # step, held constant, rescaled to the same total - the step goes to the tokens the
            # model cannot yet predict (the document-specific ones).
            ce_i = jax.lax.stop_gradient(token_ce(h, model.lm_head_weight(), tokens, fused_block))
            w0sum = jnp.sum(w) + 1e-6
            rel = ce_i / (jnp.sum(w * ce_i) / w0sum + 1e-6)
            w = w * jnp.power(jnp.maximum(rel, 1e-6), surprise_gamma)
            w = w * (w0sum / (jnp.sum(w) + 1e-6))
        wsum = jnp.sum(w) + 1e-6
        if feat_mode != "off":
            pred = model.inner_feat_head(h).astype(jnp.float32)
            feat = jnp.sum(w * jnp.mean(jnp.square(pred - feat_tgt.astype(jnp.float32)), axis=-1)) / wsum
        else:
            feat = jnp.float32(0.0)
        h_mtp = model.mtp_head(h) if extras is not None and extras.mtp_tgt is not None else None
        if model.inner_temp_log is not None:                        # CE on logits / tau (learned temperature)
            h = h * jnp.exp(-model.inner_temp_log).astype(h.dtype)
            h_mtp = None if h_mtp is None else h_mtp * jnp.exp(-model.inner_temp_log).astype(h.dtype)
        lm_w = model.lm_head_weight()
        ce = weighted_ce(h, lm_w, tokens, w, fused_block) / wsum
        obj = ce if feat_mode == "off" else (feat if feat_mode == "only" else ce + feat_weight * feat)
        distill = mtp = jnp.float32(0.0)
        if extras is not None and extras.soft_ids is not None:
            wd = extras.soft_mask.astype(jnp.float32) if tok_w is None else extras.soft_mask.astype(jnp.float32) * tok_w
            wdsum = jnp.sum(wd) + 1e-6
            soft = soft_weighted_ce(h, lm_w, extras.soft_ids, extras.soft_q, wd, fused_block) / wdsum
            obj = obj + distill_weight * soft
            q = extras.soft_q.astype(jnp.float32)
            ent = -jnp.sum(q * jnp.log(jnp.maximum(q, 1e-30)), axis=-1)                       # -sum_k q_k log q_k per slot
            distill = soft - jnp.sum(wd * ent) / wdsum                                         # sum_k q_k log(q_k / p_k)
        if h_mtp is not None:
            wm = extras.mtp_mask.astype(jnp.float32) if tok_w is None else extras.mtp_mask.astype(jnp.float32) * tok_w
            mtp = weighted_ce(h_mtp, lm_w, extras.mtp_tgt, wm, fused_block) / (jnp.sum(wm) + 1e-6)
            obj = obj + mtp_weight * mtp
        kvb = jnp.float32(0.0)
        if kvb_weight > 0:
            # Key-value binding in every suffix layer: the fast weight's read contribution at the
            # slot (post_norm(W_l(x_l)), its key = its own input x_l) is fitted to the value
            # theta_V,l(v_l) - per-token bindings written into the same W as the E2E step, each
            # layer's term moving that layer's W only (_bind_err).
            src = x_l if ks is None else jax.lax.stop_gradient(ks)
            vals = jax.vmap(lambda lin, s: lin(s))(model.inner_kvb_value, src.astype(x_l.dtype))    # [S, C, H]
            err = jnp.mean(jax.vmap(_bind_err)(Wc, model.prime_post_norm, x_l, vals), axis=0)       # [C]
            kvb = jnp.sum(w * err) / wsum
            obj = obj + kvb_weight * kvb
        if soft_only:
            obj = distill_weight * soft
        return obj, ce, feat, distill, mtp, kvb

    if h_inner.ndim == 2:
        obj, ce, feat, distill, mtp, kvb = one(h_inner, inner_mask)
    else:
        obj, ce, feat, distill, mtp, kvb = (jnp.mean(x) for x in jax.vmap(one)(h_inner, inner_mask))
    return obj, (ce, feat, distill, mtp, kvb)


def token_ce(h: Array, lm_w: Array, tokens: Array, block: int) -> Array:
    """``[C]`` cross-entropy of every row (forward only; dense when ``block`` is 0)."""
    if block > 0:
        return per_token_ce(h, lm_w, tokens, block)
    lp = jax.nn.log_softmax(h.astype(jnp.float32) @ lm_w.astype(jnp.float32).T, axis=-1)
    return -jnp.take_along_axis(lp, tokens[:, None], axis=-1)[:, 0]


def inner_loss(W: FastWeight, model: Model, h_inner: Array, tokens: Array, inner_mask: Array,
               kv: Tuple[Array, Array], pos: Array, key_valid: Array, fused_block: int,
               compute_dtype, tok_w: Optional[Array] = None, pos_q: Optional[Array] = None,
               own_mask: Optional[Array] = None) -> Array:
    """The token-CE inner loss (``inner_objective`` with the feature task off)."""
    return inner_objective(W, model, h_inner, tokens, inner_mask, kv, pos, key_valid, fused_block,
                           compute_dtype, tok_w, pos_q, own_mask)[0]


def clip_global(g: FastWeight, clip: float) -> Tuple[FastWeight, Array]:
    """Scale the whole gradient tree so its GLOBAL norm is at most ``clip``.

    Returns ``(g_clipped, norm_before)``.  ``clip <= 0`` disables clipping.
    """
    sq = sum(jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(eqx.filter(g, eqx.is_array)))
    norm = jnp.sqrt(sq + 1e-8)
    if clip <= 0:
        return g, norm
    scale = jnp.minimum(1.0, clip / norm)
    return jax.tree_util.tree_map(lambda x: x * scale, g), norm


def _global_norm(tree) -> Array:
    return jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(eqx.filter(tree, eqx.is_array)))
                    + 1e-8)


# Newton-Schulz quintic coefficients (Muon).
_NS = (3.4445, -4.7750, 2.0315)


def orthogonalise2(g: Array) -> Array:
    """Two Newton-Schulz iterations (``inner_update muon2``): ``orthogonalise(g, 2)``."""
    return orthogonalise(g, 2)


def orthogonalise(g: Array, iters: int) -> Array:
    """``iters`` Newton-Schulz iterations toward the orthogonal factor of ``g [..., m, n]``,
    as a linear map of ``g`` whose matrix is held constant for the meta-gradient:
    ``X_{k+1} = (a I + b A_k + c A_k^2) X_k`` with ``A_k = X_k X_k^T`` (``m <= n``
    side), the products in bf16 with f32 accumulation as Muon does."""
    a, b, c = _NS
    tr = g.shape[-2] > g.shape[-1]
    x = jnp.swapaxes(g, -1, -2) if tr else g
    x = x / (jnp.linalg.norm(x, axis=(-2, -1), keepdims=True) + 1e-7)
    x = jax.lax.stop_gradient(x)
    m = x.shape[-2]
    eye = jnp.eye(m, dtype=jnp.float32)
    P = eye
    for _ in range(iters):
        xb = x.astype(jnp.bfloat16)
        A = jnp.matmul(xb, jnp.swapaxes(xb, -1, -2), preferred_element_type=jnp.float32)
        Pk = a * eye + b * A + c * jnp.matmul(A.astype(jnp.bfloat16), A.astype(jnp.bfloat16),
                                             preferred_element_type=jnp.float32)
        x = jnp.matmul(Pk.astype(jnp.bfloat16), xb, preferred_element_type=jnp.float32)
        P = jnp.matmul(Pk.astype(jnp.bfloat16), P.astype(jnp.bfloat16), preferred_element_type=jnp.float32)
    P = jax.lax.stop_gradient(P)
    y = jnp.matmul(P.astype(jnp.bfloat16), (jnp.swapaxes(g, -1, -2) if tr else g).astype(jnp.bfloat16),
                   preferred_element_type=jnp.float32)
    return jnp.swapaxes(y, -1, -2) if tr else y


def direction(g: FastWeight, rule: str) -> FastWeight:
    """The step direction for ``rule``; ``sign`` and ``muon2`` are rescaled to the
    gradient's global norm so that the clip below acts the same for every rule."""
    if rule == "sgd":
        return g
    if rule == "sign":
        f = lambda x: x * jax.lax.rsqrt(jnp.square(x) + 1e-12)
    elif rule == "muon2":
        f = orthogonalise2
    else:
        raise ValueError(f"inner_update {rule!r}")
    u = jax.tree_util.tree_map(f, g)
    return jax.tree_util.tree_map(lambda x: x * (_global_norm(g) / _global_norm(u)), u)


def inner_grad(W: FastWeight, model: Model, h_inner: Array, tokens: Array, inner_mask: Array,
               kv: Tuple[Array, Array], pos: Array, key_valid: Array, fused_block: int, compute_dtype,
               h_clean: Optional[Array] = None, pos_q: Optional[Array] = None,
               own_mask: Optional[Array] = None, feat_tgt: Optional[Array] = None,
               feat_mode: str = "off", feat_weight: float = 1.0, extras: Optional[InnerExtras] = None,
               distill_weight: float = 1.0, mtp_weight: float = 1.0, distill_meta: bool = True,
               surprise_gamma: float = 0.0, kvb_weight: float = 0.0, kvb_src: Optional[Array] = None,
               score_from: int = 0) -> Tuple[FastWeight, Tuple[Array, Array, Array, Array, Array]]:
    """One sequence: the gradient of the inner objective w.r.t. ``W`` and its terms
    ``(ce, feat, distill, mtp, kvb)`` - the first half of ``inner_step``."""
    if (model.inner_gate is not None or model.inner_token_gate is not None) and h_clean is None:
        raise ValueError("the model has write gates: inner_step needs h_clean")
    tok_w = token_weights(model, h_clean) if model.inner_token_gate is not None else None
    args = (model, h_inner, tokens, inner_mask, kv, pos, key_valid, fused_block, compute_dtype, tok_w, pos_q, own_mask,
            feat_tgt, feat_mode, feat_weight, extras)
    wkw = dict(surprise_gamma=surprise_gamma, kvb_weight=kvb_weight, kvb_src=kvb_src, score_from=score_from)
    soft_first_order = extras is not None and extras.soft_ids is not None and not distill_meta
    (_, (loss, feat, distill, mtp, kvb)), g = eqx.filter_value_and_grad(inner_objective, has_aux=True)(
        W, *args, 0.0 if soft_first_order else distill_weight, mtp_weight, **wkw)
    if soft_first_order:
        (_, (_, _, distill, _, _)), g_soft = eqx.filter_value_and_grad(inner_objective, has_aux=True)(
            W, *args, distill_weight, mtp_weight, True, **wkw)
        g = jax.tree_util.tree_map(lambda a, b: a + jax.lax.stop_gradient(b) if eqx.is_inexact_array(a) else a, g, g_soft)
    return g, (loss, feat, distill, mtp, kvb)


def unit_scale(u: FastWeight, scale: Tuple[Array, Array]) -> FastWeight:
    """The normalised step (``WriteConfig.norm_alpha``): ``scale = (f_units [S, I], f_dims [S, H])``;
    down rows times f_units, gate / up entries times min(f_dims, f_units)."""
    fu, fd = scale
    f_gu = jnp.minimum(fd[:, :, None], fu[:, None, :])                # [S, H, I]
    g, up, dn = u.gate.weight * f_gu, u.up.weight * f_gu, u.down.weight * fu[:, :, None]
    return eqx.tree_at(lambda t: (t.gate.weight, t.up.weight, t.down.weight), u, (g, up, dn))


def inner_apply(W: FastWeight, g: FastWeight, model: Model, lr: Array, clip: float, rule: str = "sgd",
                inner_mask: Optional[Array] = None, h_clean: Optional[Array] = None, loss: Optional[Array] = None,
                frac: float = 1.0, unit_mask: Optional[Array] = None,
                scale: Optional[Tuple[Array, Array]] = None) -> Tuple[FastWeight, Array]:
    """One sequence: the step of gradient ``g`` - ``(W', pre-clip norm)``; the second half of
    ``inner_step`` (unit mask, direction, preconditioner, write-dimension mask, clip, the
    per-layer step size and gate, decay).  ``inner_mask`` / ``h_clean`` / ``loss`` feed the
    write gate only.  ``scale``: the normalised step after the clip (``unit_scale``)."""
    if unit_mask is not None:
        g = mask_units(g, unit_mask)
    u = direction(g, rule)
    if model.inner_precond is not None:
        # learned factored preconditioner: g[l, i, j] * exp(a[l, i]) * exp(b[l, j]) per matrix
        rows, cols = model.inner_precond
        u = jax.tree_util.tree_map(
            lambda x, a, b: x * jnp.exp(a)[..., :, None] * jnp.exp(b)[..., None, :] if eqx.is_inexact_array(x) else x,
            u, rows, cols)
    if frac < 1.0:
        leaves, treedef = jax.tree_util.tree_flatten(eqx.filter(u, eqx.is_inexact_array))
        masks = [jax.random.bernoulli(jax.random.PRNGKey(7919 + i), frac, x.shape).astype(x.dtype) for i, x in enumerate(leaves)]
        um = jax.tree_util.tree_unflatten(treedef, [x * m for x, m in zip(leaves, masks)])
        um = eqx.combine(um, u)
        u = jax.tree_util.tree_map(lambda x: x * (_global_norm(u) / _global_norm(um)) if eqx.is_inexact_array(x) else x, um)
    u, norm = clip_global(u, clip)
    if scale is not None:
        u = unit_scale(u, scale)
    # per-layer step size: lr x learned gate of [chunk content, surprise, mask ratio]
    eta = lr
    if model.inner_step_log is not None:
        eta = eta * jnp.exp(model.inner_step_log)                                                   # [S], unbounded
    if model.inner_gate is not None:
        m = inner_mask.reshape(-1, inner_mask.shape[-1])[0]                                       # first view [C]
        ratio = jnp.mean((m > 0).astype(jnp.float32))
        feats = jnp.concatenate([jnp.mean(h_clean.astype(jnp.float32), axis=0),
                                 jnp.stack([jax.lax.stop_gradient(loss) / 8.0, ratio])])           # [H + 2]
        gate = jax.vmap(lambda lin: lin(feats)[0])(model.inner_gate)                              # [S]
        eta = lr * 2.0 * jax.nn.sigmoid(gate)
    eta_l = lambda x: eta if jnp.ndim(eta) == 0 else eta.reshape((-1,) + (1,) * (x.ndim - 1))
    if model.inner_decay_logit is not None:
        lam = (0.1 * jax.nn.sigmoid(model.inner_decay_logit)).reshape((-1,) + (1,) * 2)          # [S, 1, 1]
        W_new = jax.tree_util.tree_map(
            lambda p, d, p0: (p - eta_l(p) * d - lam * (p - p0.astype(jnp.float32))).astype(jnp.float32)
            if eqx.is_inexact_array(p) else p, W, u, model.fast_weight)
    else:
        W_new = jax.tree_util.tree_map(
            lambda p, d: (p - eta_l(p) * d).astype(jnp.float32) if eqx.is_inexact_array(p) else p, W, u)
    return W_new, norm


def inner_step(W: FastWeight, model: Model, h_inner: Array, tokens: Array, inner_mask: Array,
               kv: Tuple[Array, Array], pos: Array, key_valid: Array, lr: Array, clip: float,
               fused_block: int, compute_dtype, rule: str = "sgd",
               h_clean: Optional[Array] = None, pos_q: Optional[Array] = None,
               own_mask: Optional[Array] = None, feat_tgt: Optional[Array] = None,
               feat_mode: str = "off", feat_weight: float = 1.0, frac: float = 1.0,
               extras: Optional[InnerExtras] = None, distill_weight: float = 1.0,
               mtp_weight: float = 1.0, distill_meta: bool = True, unit_mask: Optional[Array] = None,
               surprise_gamma: float = 0.0, kvb_weight: float = 0.0,
               kvb_src: Optional[Array] = None, score_from: int = 0,
               scale: Optional[Tuple[Array, Array]] = None) -> Tuple[FastWeight, InnerStats]:
    """One sequence: ``W_f32 -> W_f32'`` plus stats (``inner_grad`` then ``inner_apply``).
    ``h_clean [C, H]`` (the clean stream's suffix input of the chunk) feeds the gates;
    required when the model has any.
    ``feat_tgt [C, H]`` / ``feat_mode`` / ``feat_weight``: the feature-prediction term,
    ``extras`` / ``distill_weight`` / ``mtp_weight`` the distillation and MTP terms
    (``inner_objective``); ``distill_meta`` False: the distillation term's gradient enters
    the write as a constant (no meta-gradient through it - a first-order soft term).
    ``frac < 1``: the write touches only a FIXED random ``frac`` of every fast-weight
    matrix's entries, the step rescaled to the norm it had (the write-dimension probe:
    what a narrower fast weight would lose, without retraining).  ``unit_mask [I]``: the
    step touches only these hidden units of every fast-weight MLP (gate / up columns, down
    rows) and is clipped as such (``write_chunk``'s two-timescale split);
    ``surprise_gamma`` / ``kvb_weight`` / ``kvb_src`` / ``score_from``: ``WriteConfig`` terms of the
    objective (``inner_objective``)."""
    g, (loss, feat, distill, mtp, kvb) = inner_grad(
        W, model, h_inner, tokens, inner_mask, kv, pos, key_valid, fused_block, compute_dtype, h_clean, pos_q,
        own_mask, feat_tgt, feat_mode, feat_weight, extras, distill_weight, mtp_weight, distill_meta,
        surprise_gamma, kvb_weight, kvb_src, score_from)
    W_new, norm = inner_apply(W, g, model, lr, clip, rule, inner_mask, h_clean, loss, frac, unit_mask, scale)
    return W_new, InnerStats(loss, norm, (norm > clip).astype(jnp.float32) if clip > 0 else jnp.float32(0.0),
                             feat, distill, mtp, kvb)


def inner_update(W: FastWeight, model: Model, h_inner: Array, tokens: Array, inner_mask: Array,
                 kv: Tuple[Array, Array], pos: Array, key_valid: Array, lr: Array, clip: float,
                 fused_block: int, compute_dtype, rule: str = "sgd",
                 h_clean: Optional[Array] = None, pos_q: Optional[Array] = None,
                 own_mask: Optional[Array] = None, feat_tgt: Optional[Array] = None,
                 feat_mode: str = "off", feat_weight: float = 1.0, frac: float = 1.0,
                 extras: Optional[InnerExtras] = None, distill_weight: float = 1.0,
                 mtp_weight: float = 1.0, distill_meta: bool = True, unit_mask: Optional[Array] = None,
                 surprise_gamma: float = 0.0, kvb_weight: float = 0.0,
                 kvb_src: Optional[Array] = None, score_from: int = 0,
                 scale: Optional[Tuple[Array, Array]] = None) -> Tuple[FastWeight, InnerStats]:
    """Batched ``inner_step``: ``W [B, S, ...]`` etc.  Stats are batch-summed.
    ``pos_q [B, C]`` batched like ``pos``; ``own_mask [C, C]`` shared; ``feat_tgt [B, C, H]``;
    every array of ``extras`` batched (``[B, C, ...]``); ``kvb_src [B, S, C, H]``; ``unit_mask [I]``
    shared; ``scale`` batched (``[B, S, I]``, ``[B, S, H]``)."""
    step = lambda w, h, t, m, k, p, hc, pq, ft, ex, ks, sc: inner_step(w, model, h, t, m, k, p, key_valid, lr, clip,
                                                                       fused_block, compute_dtype, rule, hc, pq,
                                                                       own_mask, ft, feat_mode, feat_weight, frac, ex,
                                                                       distill_weight, mtp_weight, distill_meta,
                                                                       unit_mask, surprise_gamma, kvb_weight, ks,
                                                                       score_from, sc)
    W_new, st = eqx.filter_vmap(step, in_axes=(0, 0, 0, 0, 0, 0, None if h_clean is None else 0,
                                               None if pos_q is None else 0, None if feat_tgt is None else 0,
                                               None if extras is None else 0, None if kvb_src is None else 0,
                                               None if scale is None else 0))(
        W, h_inner, tokens, inner_mask, kv, pos, h_clean, pos_q, feat_tgt, extras, kvb_src, scale)
    return W_new, InnerStats(*(jnp.sum(x) for x in st))


# ----------------------------------------------------------------------------
# the write of one chunk under a WriteConfig (training scan and decoding share it)
# ----------------------------------------------------------------------------
def fast_unit_mask(inter: int, frac: float) -> Array:
    """``[I]`` bool: the hidden units written per sub-chunk (the first ``round(frac * I)``;
    units are exchangeable at initialisation, so any fixed choice is equivalent)."""
    return jnp.arange(inter) < int(round(frac * inter))


def norm_scales(model: Model, W: FastWeight, h_inner: Array, kv: Tuple[Array, Array], pos: Array, key_valid: Array,
                pos_q: Array, own_mask, inner_mask: Array, compute_dtype, alpha: float,
                cap: float) -> Tuple[Array, Array]:
    """``(f_units [B, S, I], f_dims [B, S, H])`` of the normalised step (``WriteConfig.norm_alpha``), batched
    like ``write_chunk``: from the write's own forward (the inner stream, the weights before the step), per
    fast-weight layer the hidden units' and the input dimensions' RMS over the written slots r, and
    f = min(cap, (max r / r)^alpha) - 1 for the dominant ones.  Constants for the meta-gradient.
    ``own_mask`` goes to attention as given: static (numpy) in training, where cuDNN builds its masked kernels
    from it."""
    f32 = jnp.float32
    cap_a = jnp.asarray(cap, f32).reshape(-1, 1)                         # [1, 1]

    def one(w, h, k, p, pq, m):
        wc = cast_floating(w, compute_dtype)
        _, _, (x, _) = model.suffix_forward(h, p, k, wc, key_valid, pos_q=pq, own_mask=own_mask, return_prime=True)
        a = eqx.filter_vmap(lambda ww, z: ww.act(ww.gate(z)) * ww.up(z))(wc, x)      # [S, C, I]
        mw = m.astype(f32) / (jnp.sum(m.astype(f32)) + 1e-6)
        rms = lambda z: jnp.sqrt(jnp.einsum("c,scd->sd", mw, jnp.square(z.astype(f32))))
        f = lambda r: jnp.minimum(cap_a, (jnp.max(r, axis=-1, keepdims=True) / (r + 1e-12)) ** alpha)
        return f(rms(a)), f(rms(x))
    return tuple(jax.lax.stop_gradient(z) for z in eqx.filter_vmap(one)(W, h_inner, kv, pos, pos_q, inner_mask))


def mask_units(tree: FastWeight, unit_mask: Array) -> FastWeight:
    """Zero every entry of a fast-weight-shaped tree outside the hidden units ``unit_mask [I]``
    (gate / up weights ``[..., H, I]``: columns; down weight ``[..., I, H]``: rows)."""
    m = unit_mask.astype(jnp.float32)
    tree = eqx.tree_at(lambda t: t.gate.weight, tree, tree.gate.weight * m.astype(tree.gate.weight.dtype))
    tree = eqx.tree_at(lambda t: t.up.weight, tree, tree.up.weight * m.astype(tree.up.weight.dtype))
    return eqx.tree_at(lambda t: t.down.weight, tree, tree.down.weight * m[:, None].astype(tree.down.weight.dtype))


def write_chunk(W: FastWeight, model: Model, h_inner: Array, tokens: Array, inner_mask: Array,
                kv: Tuple[Array, Array], pos: Array, key_valid: Array, lr: Array, clip: float,
                fused_block: int, compute_dtype, rule: str, h_clean: Optional[Array], pos_q: Array,
                own_mask, wc: WriteConfig, W0: Optional[FastWeight] = None,
                kvb_src: Optional[Array] = None, remat: bool = True) -> Tuple[FastWeight, InnerStats]:
    """The causal write of one chunk under ``wc`` (batched like ``inner_update``: ``W [B, ...]``,
    ``h_inner [B, C, H]``, ``tokens / inner_mask / pos / pos_q [B, C]``, ``kvb_src [B, S, C, H]``,
    ``W0`` the batched W_0 for ``fast_decay``).  ``key_valid [(Wn+1)*C]`` is the chunk's key
    validity (window slots, then its own C slots).  A plain ``wc`` is exactly one
    ``inner_update``.  Sub-chunk k's step runs the chunk's inner forward over slots
    ``[0, end_k)`` only (causal own-chunk attention: nothing later can influence it) with the
    weights after steps 0..k-1, and scores sub-chunk k's slots (the vocabulary projection runs over
    those slots only, so n steps cost (n+1)/2 inner suffix forwards but one chunk of CE)."""
    base = dict(rule=rule, h_clean=h_clean, own_mask=own_mask)
    if wc.plain:
        return inner_update(W, model, h_inner, tokens, inner_mask, kv, pos, key_valid, lr, clip, fused_block,
                            compute_dtype, pos_q=pos_q, **base)
    if model.inner_gate is not None or model.inner_decay_logit is not None:
        raise ValueError("the write schemes are defined for the plain step: no write gate / learned decay")
    if wc.normalised:
        if wc.subchunks > 1 or wc.fast_frac > 0 or wc.steps > 1:
            raise ValueError("the normalised step: one whole-chunk step, alone")
        scale = norm_scales(model, W, h_inner, kv, pos, key_valid, pos_q, own_mask, inner_mask, compute_dtype,
                            wc.norm_alpha, wc.norm_cap)
        return inner_update(W, model, h_inner, tokens, inner_mask, kv, pos, key_valid, lr, clip, fused_block,
                            compute_dtype, pos_q=pos_q, kvb_weight=wc.kvb_weight,
                            kvb_src=kvb_src if wc.kvb_weight > 0 else None, surprise_gamma=wc.surprise_gamma,
                            scale=scale, **base)
    B, C = tokens.shape
    n = wc.subchunks
    own = own_mask                                                    # static (numpy) in training: cuDNN needs it so
    wkw = dict(surprise_gamma=wc.surprise_gamma)

    kvb_weight = wc.kvb_weight

    def full_step(w):
        return inner_update(w, model, h_inner, tokens, inner_mask, kv, pos, key_valid, lr, clip, fused_block,
                            compute_dtype, pos_q=pos_q, kvb_weight=kvb_weight,
                            kvb_src=kvb_src if kvb_weight > 0 else None, **base, **wkw)

    def sub_steps(w):
        sts = []
        Kw = key_valid.shape[0] - C                                   # window slots before the own chunk
        for k in range(n):
            s, e = k * C // n, (k + 1) * C // n
            sel = (jnp.arange(e) >= s).astype(jnp.float32)
            m_k = inner_mask[:, :e].astype(jnp.float32) * sel[None]
            ks = None if (kvb_src is None or kvb_weight <= 0) else kvb_src[:, :, :e]

            def one(w_, s_=s, e_=e, m_=m_k, ks_=ks):
                return inner_update(w_, model, h_inner[:, :e_], tokens[:, :e_], m_, kv, pos[:, :e_],
                                    key_valid[:Kw + e_], lr, clip * wc.subchunk_scale, fused_block, compute_dtype,
                                    rule=rule, h_clean=h_clean, pos_q=pos_q[:, :e_],
                                    own_mask=None if own is None else own[:e_, :e_],
                                    kvb_weight=kvb_weight, kvb_src=ks_, score_from=s_, **wkw)
            w, st = (jax.checkpoint(one) if remat else one)(w)
            sts.append(st)
        return w, InnerStats(*(sum(x) / n for x in zip(*sts)))

    def split_steps(w, fast):
        # Two timescales in one pass over the sub-chunks: every sub-step's gradient moves the fast
        # units at once (lr * fast_lr_scale, clip * subchunk_scale) and is accumulated for the slow
        # units, weighted by the sub-chunk's share of the chunk's targets; the slow units take one
        # clipped step at the chunk's end.  With the fast units frozen the accumulated gradient IS
        # the full-chunk gradient (the sub-chunk objectives partition the chunk's targets and the
        # causal forward over [0, e) is the full one's prefix), so the slow units see the chunk
        # once, at no cost beyond the sub-steps.
        Kw = key_valid.shape[0] - C
        lr_f = lr * wc.fast_lr_scale

        def seq(w, h, t, m, kv_, p, hc, pq, ks):
            mf = m.astype(jnp.float32)
            wtot = jnp.sum(mf) + 1e-6
            acc, sts = None, []
            for k in range(n):
                s, e = k * C // n, (k + 1) * C // n
                m_k = mf[:e] * (jnp.arange(e) >= s).astype(jnp.float32)

                def sub(w_, s_=s, e_=e, m_=m_k):
                    g, aux = inner_grad(w_, model, h[:e_], t[:e_], m_, kv_, p[:e_], key_valid[:Kw + e_], fused_block,
                                        compute_dtype, h_clean=hc, pos_q=None if pq is None else pq[:e_],
                                        own_mask=None if own is None else own[:e_, :e_],
                                        surprise_gamma=wc.surprise_gamma, kvb_weight=wc.kvb_weight,
                                        kvb_src=None if ks is None else ks[:, :e_], score_from=s_)
                    w_new, norm = inner_apply(w_, g, model, lr_f, clip * wc.subchunk_scale, rule, m_, hc, aux[0],
                                              unit_mask=fast)
                    share = jnp.sum(m_) / wtot
                    return w_new, jax.tree_util.tree_map(lambda x: x * share, mask_units(g, ~fast)), aux, norm
                w, g_slow, aux, norm = (jax.checkpoint(sub) if remat else sub)(w)
                acc = g_slow if acc is None else jax.tree_util.tree_map(jnp.add, acc, g_slow)
                sts.append((aux, norm, (norm > clip * wc.subchunk_scale).astype(jnp.float32)))
            loss = sum(a[0] for a, _, _ in sts) / n
            w, norm_s = inner_apply(w, acc, model, lr, clip, rule, mf, hc, loss, unit_mask=~fast)
            norm_f = sum(x for _, x, _ in sts) / n
            clip_f = sum(x for _, _, x in sts) / n
            return w, InnerStats(loss, (norm_f + norm_s) / 2, (clip_f + (norm_s > clip).astype(jnp.float32)) / 2,
                                 jnp.float32(0.0), jnp.float32(0.0), jnp.float32(0.0),
                                 sum(a[4] for a, _, _ in sts) / n)

        ks_all = kvb_src if wc.kvb_weight > 0 else None
        w, st = eqx.filter_vmap(seq, in_axes=(0, 0, 0, 0, 0, 0, None if h_clean is None else 0,
                                              None if pos_q is None else 0, None if ks_all is None else 0))(
            w, h_inner, tokens, inner_mask, kv, pos, h_clean, pos_q, ks_all)
        return w, InnerStats(*(jnp.sum(x) for x in st))

    if wc.fast_frac > 0:
        fast = fast_unit_mask(W.gate.weight.shape[-1], wc.fast_frac)
        W, st = split_steps(W, fast)
        if wc.fast_decay > 0:
            if W0 is None:
                raise ValueError("inner_fast_decay needs W0")
            relaxed = jax.tree_util.tree_map(
                lambda w, w0: (w0.astype(jnp.float32) + (1.0 - wc.fast_decay) * (w - w0.astype(jnp.float32)))
                if eqx.is_inexact_array(w) else w, W, W0)
            # keep the slow units, relax the fast ones
            keep = mask_units(jax.tree_util.tree_map(lambda x: x * 0 + 1 if eqx.is_inexact_array(x) else x, W),
                              fast)
            W = jax.tree_util.tree_map(lambda w, r, f: w * (1 - f) + r * f if eqx.is_inexact_array(w) else w,
                                       W, relaxed, keep)
        return W, st
    if n == 1 and wc.steps > 1:
        # K whole-chunk steps as a scan over the array leaves (one compiled step body)
        arrs, static = eqx.partition(W, eqx.is_array)

        def body(a, _):
            w_new, st = full_step(eqx.combine(a, static))
            return eqx.partition(w_new, eqx.is_array)[0], st
        arrs, sts = jax.lax.scan(jax.checkpoint(body) if remat else body, arrs, None, length=wc.steps)
        return eqx.combine(arrs, static), InnerStats(*(jnp.mean(x) for x in sts))
    if n == 1:
        return full_step(W)
    return sub_steps(W)


def drift_stats(W_final: FastWeight, W_0: FastWeight) -> dict:
    """``||W_C - W_0|| / ||W_0||`` over all leaves - the one number that says whether
    the fast weight is doing anything.  Compare against the band
    ``tools/memory_model.py`` predicts for the config, not a fixed one: the bound is
    ``n_chunks * clip * lr / ||W_0||`` and ``||W_0||`` scales as ``sqrt(params)``.
    """
    lf = jax.tree_util.tree_leaves(eqx.filter(W_final, eqx.is_array))
    l0 = jax.tree_util.tree_leaves(eqx.filter(W_0, eqx.is_array))
    d = jnp.sqrt(sum(jnp.sum(jnp.square(a.astype(jnp.float32) - b.astype(jnp.float32))) for a, b in zip(lf, l0)))
    i = jnp.sqrt(sum(jnp.sum(jnp.square(b.astype(jnp.float32))) for b in l0))
    return {"ttt_drift_rel": d / (i + 1e-12), "ttt_delta_norm": d, "ttt_init_norm": i}
