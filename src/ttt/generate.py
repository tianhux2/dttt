# coding=utf-8
"""
Block masked-diffusion decoding with test-time training (single sequence).

Generation proceeds in blocks of ``chunk_size`` tokens, left to right, through the
same sequential chunk machinery the training suffix uses - here for ALL layers,
since nothing is known ahead of time:

* a block starts fully masked (prompt positions inside it stay clean) and is
  unmasked over ``steps`` iterations: each iteration runs the model on the current
  block against the clean K/V window of the previous blocks and the live fast
  weight, samples every masked position, and commits the highest-confidence
  ``1/steps`` fraction (LLaDA-style low-confidence remasking);
* a finished block is then treated exactly as a clean chunk in training: its K/V
  enter the shift window (forward B) and one inner update is taken on an
  inner-masked copy of it (forward C) before the next block starts.

The per-layer state is ``(kv window [L, W*C, nkv, d] x2, fast weight W)``; K/V of
prefix layers are cached too, so a block costs one full forward per unmasking
iteration plus two (B, C) when it completes.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import NamedTuple, Optional, Tuple

import numpy as np

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from .attention import key_validity, own_mask_matrix, read_mask_matrix
from .inner_loop import WriteConfig, inner_update, write_chunk
from .model import FastWeight, Model, cast_floating
from .objective import make_inner_mask


class DecodeState(NamedTuple):
    kv: Tuple[Array, Array]      # clean K/V windows of every layer, [L, W*C, nkv, d] x2 (compute dtype)
    W: FastWeight                # live fast weight [S, ...], f32
    chunk: Array                 # index of the next block (int32 scalar)


def init_state(model: Model, compute_dtype=jnp.bfloat16) -> DecodeState:
    cfg = model.cfg
    if model.fast_weight is None:
        raise NotImplementedError("decoding with the gdn memory is not implemented (training / evaluation only)")
    shape = (cfg.num_hidden_layers, cfg.window_chunks * cfg.chunk_size, cfg.num_key_value_heads, cfg.head_dim)
    kv = (jnp.zeros(shape, compute_dtype), jnp.zeros(shape, compute_dtype))
    return DecodeState(kv, cast_floating(model.fast_weight, jnp.float32), jnp.int32(0))


def _split(model: Model, kv):
    """Prefix / suffix halves of the stacked per-layer K/V windows."""
    n_pre = model.cfg.num_hidden_layers - model.cfg.suffix_len
    return (kv[0][:n_pre], kv[1][:n_pre]), (kv[0][n_pre:], kv[1][n_pre:])


def block_forward(model: Model, ids: Array, pos: Array, state: DecodeState, compute_dtype=jnp.bfloat16,
                  pos_q: Optional[Array] = None, own_mask: Optional[Array] = None, return_prime: bool = False,
                  own_kv: Optional[Tuple[Array, Array]] = None):
    """One block ``ids [C]`` through all layers against the state's windows.

    Returns ``(logits [C, V] f32, h_suffix_in [C, H], kv_own [L, C, nkv, d] x2)``:
    the suffix input is what the inner update needs, the own K/V what the window
    shift needs.  ``pos_q`` / ``own_mask``: the inner stream of the causal / split tasks
    (queries at the predicted token's position, restricted own-chunk attention).
    ``return_prime``: a fourth element, the suffix layers' fast-weight inputs and read
    contributions ``([S, C, H], [S, C, H])`` (the key-value-binding write's clean values).
    ``own_kv``: the sub-block read (``ModelConfig.read_block``) - the block's clean K/V of every layer
    (``clean_forward``), read by the earlier groups' slots; a model with ``read_block`` reads only so, and
    its clean forward is ``clean_forward``.
    """
    cfg = model.cfg
    C = cfg.chunk_size
    if cfg.read_block and own_kv is None and own_mask is None:
        raise ValueError("read_block: the read needs the block's clean K/V (own_kv); the clean pass is clean_forward")
    if own_kv is not None and own_mask is None:
        own_mask = read_mask_matrix(C, cfg.read_block)                 # static (numpy): cuDNN masks need it
    key_valid = key_validity(state.chunk, C, cfg.window_chunks, cfg.reach)
    kv_pre, kv_suf = _split(model, state.kv)
    own_pre, own_suf = (None, None) if own_kv is None else _split(model, own_kv)
    h = model.embed_tokens(ids).astype(compute_dtype)

    def prefix_layer(h, xs):
        blk, kv_l, ok_l = xs
        return blk(h, pos, kv_l, key_valid, pos_q=pos_q, own_mask=own_mask, own_kv=ok_l)

    h, kv_own_pre = jax.lax.scan(prefix_layer, h, (model.prefix, kv_pre, own_pre))
    h_suffix_in = h
    Wc = cast_floating(state.W, compute_dtype)
    if return_prime:
        h, kv_own_suf, prime_io = model.suffix_forward(h, pos, kv_suf, Wc, key_valid, pos_q=pos_q, own_mask=own_mask,
                                                       return_prime=True, own_kv=own_suf)
    else:
        h, kv_own_suf = model.suffix_forward(h, pos, kv_suf, Wc, key_valid, pos_q=pos_q, own_mask=own_mask,
                                             own_kv=own_suf)
    kv_own = tuple(jnp.concatenate([a, b], axis=0) for a, b in zip(kv_own_pre, kv_own_suf))
    if return_prime:
        return model.logits(h).astype(jnp.float32), h_suffix_in, kv_own, prime_io
    return model.logits(h).astype(jnp.float32), h_suffix_in, kv_own


def clean_forward(model: Model, ids: Array, pos: Array, state: DecodeState, compute_dtype=jnp.bfloat16,
                  return_prime: bool = False):
    """The clean stream's forward of a block (training's forward B): ``block_forward`` with the clean
    own-chunk pattern - bidirectional, or with ``read_block`` b block-causal (a slot sees the clean slots of
    its own and the earlier groups of b), so that a group's clean K/V never depend on later tokens."""
    rb = model.cfg.read_block
    own = own_mask_matrix(model.cfg.chunk_size, f"block:{rb}") if rb else None     # static (numpy)
    return block_forward(model, ids, pos, state, compute_dtype, own_mask=own, return_prime=return_prime)


def block_tail(model: Model, ids: Array, pos: Array, state: DecodeState, compute_dtype=jnp.bfloat16) -> Array:
    """``[V]``: the logit the last slot of the complete block ``ids`` gives the next block's first token (the
    outer shift); with ``read_block`` the read is over the block's own clean K/V."""
    own = clean_forward(model, ids, pos, state, compute_dtype)[2] if model.cfg.read_block else None
    return block_forward(model, ids, pos, state, compute_dtype, pos_q=pos + 1, own_kv=own)[0][-1]


def commit_block(model: Model, clean_ids: Array, pos: Array, state: DecodeState, key: Array,
                 inner_lr: float, inner_clip: float, inner_mask_ratio: float,
                 fused_ce_block: int = 8192, compute_dtype=jnp.bfloat16, inner_views: int = 1,
                 inner_mask_span: float = 0.0, inner_update_rule: str = "sgd",
                 inner_mask: Optional[Array] = None, inner_task: str = "mask",
                 inner_perm: Optional[Array] = None, inner_context_frac: float = 0.5,
                 wc: WriteConfig = WriteConfig()) -> DecodeState:
    """A finished (clean) block becomes context: forward B shifts its K/V into the
    windows; forward C on an inner-masked copy updates the fast weight.

    ``inner_views``: 1 = one random mask (as in training with ``inner_views 1``);
    2 = the complementary pair (deterministic full coverage); the same rule, span
    and view count the checkpoint was trained with should be used.  ``wc``: the causal
    write's scheme the checkpoint was trained with (``inner_loop.write_chunk``, the same
    function the training scan calls)."""
    cfg = model.cfg
    C = cfg.chunk_size
    key_valid = key_validity(state.chunk, C, cfg.window_chunks, cfg.reach)
    clean_values = (not wc.plain) and wc.kvb_weight > 0 and wc.kvb_value == "clean"
    # B: clean K/V of every layer (and the clean suffix input, which feeds the write gates)
    if clean_values:
        _, h_clean, kv_own, (x_clean, _) = clean_forward(model, clean_ids, pos, state, compute_dtype, return_prime=True)
    else:
        _, h_clean, kv_own = clean_forward(model, clean_ids, pos, state, compute_dtype)
    kv_next = tuple(jnp.concatenate([cache[:, C:], own], axis=1) for cache, own in zip(state.kv, kv_own))
    # C: the write.  mask: an inner-masked copy; causal / split: the clean block in its
    # physical order with shifted targets - the same tasks as chunk_scan.loss_fn.
    _, kv_suf = _split(model, state.kv)
    pos_in, pos_q, own_mat, targets = pos, None, None, clean_ids
    if inner_task == "mask":
        if inner_mask is not None:                                   # host-chosen targets (inner_masks.py)
            x_inner = jnp.where(inner_mask, cfg.mask_token_id, clean_ids)
        else:
            x_inner, inner_mask = make_inner_mask(inner_mask_ratio, (inner_mask_ratio, inner_mask_ratio),
                                                  cfg.mask_token_id, cfg.pad_token_id, inner_views,
                                                  inner_mask_span)(clean_ids, key)
        if inner_views == 1:
            _, h_inner, _ = block_forward(model, x_inner, pos, state, compute_dtype)
        else:
            h_inner = jnp.stack([block_forward(model, x_inner[v], pos, state, compute_dtype)[1]
                                 for v in range(inner_views)])                       # [V, C, H]
        weights = inner_mask
    else:
        perm = jnp.arange(C) if inner_task == "causal" else (inner_perm - pos[0])   # block-relative order
        K = 0 if inner_task == "causal" else int(round(inner_context_frac * C))
        own_mat = own_mask_matrix(C, "causal" if inner_task == "causal" else f"split:{K}")   # static (numpy)
        x_inner, pos_in = clean_ids[perm], pos[perm]
        targets = jnp.concatenate([x_inner[1:], x_inner[-1:]])
        pos_q = jnp.concatenate([pos_in[1:], pos_in[-1:]])
        slot = jnp.arange(C)
        weights = (slot >= max(K - 1, 0)) & (slot <= C - 2)
        _, h_inner, _ = block_forward(model, x_inner, pos_in, state, compute_dtype, pos_q=pos_q, own_mask=own_mat)
    if not wc.plain:
        if inner_task != "causal":
            raise ValueError("the write schemes (WriteConfig) are defined for the causal write")
        W0b = jax.tree_util.tree_map(lambda x: x[None].astype(jnp.float32) if eqx.is_inexact_array(x) else x,
                                     model.fast_weight)
        W_next, _ = write_chunk(
            jax.tree_util.tree_map(lambda x: x[None], state.W), model, h_inner[None], targets[None],
            weights[None], tuple(k[None] for k in kv_suf), pos_in[None], key_valid, jnp.float32(inner_lr),
            inner_clip, fused_ce_block, compute_dtype, inner_update_rule, h_clean[None], pos_q[None], own_mat, wc,
            W0=W0b, kvb_src=x_clean[None] if clean_values else None, remat=False)
    else:
        W_next, _ = inner_update(
            jax.tree_util.tree_map(lambda x: x[None], state.W), model, h_inner[None], targets[None],
            weights[None], tuple(k[None] for k in kv_suf), pos_in[None], key_valid, jnp.float32(inner_lr),
            inner_clip, fused_ce_block, compute_dtype, inner_update_rule, h_clean=h_clean[None],
            pos_q=None if pos_q is None else pos_q[None], own_mask=own_mat)
    W_next = jax.tree_util.tree_map(lambda x: x[0], W_next)
    return DecodeState(kv_next, W_next, state.chunk + 1)


@dataclass(frozen=True)
class DecodeOpts:
    """How ``decode_block`` unmasks a block.  The defaults are the original sampler: ``steps``
    iterations, each committing the most confident ``1/(steps - i)`` of the unknown slots.

    * ``order`` "random": the slots to commit are drawn uniformly at random instead of by
      confidence (with temperature 1 and one slot per iteration: ancestral sampling).
    * ``gumbel_high``: temperature sampling with high-resolution Gumbel noise (``mode="high"``);
      the default low mode caps the noise near 16 in float32, lowering the effective temperature.
    * ``threshold`` > 0: every unknown slot whose token has probability >= threshold is committed,
      at least the single top-priority one (Fast-dLLM-style parallel decoding; a variable number of
      iterations, at most C - ``steps`` is ignored).
    * ``presence_penalty`` > 0: subtracted from the logits of tokens already present in the block.
    * ``remask_loop`` L > 0: after the block is complete, L iterations each re-mask ``remask_k``
      non-prompt slots, drawn with priority to low commit-time confidence, and re-predict them
      (a ReMDM-style corrector).
    * ``subblock`` b > 0: the block is filled left to right in groups of b slots (semi-autoregressive within the
      block): each group gets ``ceil(steps * b / C)`` iterations and only its slots are committed; the later
      groups stay masked meanwhile (the attention is still the block's own).
    * ``priority``: what "most confident" means - ``logp`` (log-probability of the sampled token, the default),
      ``margin`` (the gap between the two most probable tokens) or ``entropy`` (minus the predictive entropy)."""
    order: str = "confidence"
    gumbel_high: bool = False
    threshold: float = 0.0
    presence_penalty: float = 0.0
    remask_loop: int = 0
    remask_k: int = 16
    subblock: int = 0
    priority: str = "logp"


def decode_block(model: Model, ids: Array, known: Array, pos: Array, state: DecodeState, key: Array,
                 steps: int, temperature: float = 0.0, compute_dtype=jnp.bfloat16,
                 outer_shift: bool = False, prev_tail_logit: Optional[Array] = None,
                 opts: DecodeOpts = DecodeOpts()) -> Tuple[Array, Array]:
    """Unmask one block.  ``ids [C]`` carry the known tokens (prompt), ``known [C]``
    marks them; the rest is filled over ``steps`` iterations (``opts``: the sampler)."""
    cfg = model.cfg
    C = cfg.chunk_size
    if opts.order not in ("confidence", "random"):
        raise ValueError(f"DecodeOpts.order {opts.order!r}")
    if opts.priority not in ("logp", "margin", "entropy"):
        raise ValueError(f"DecodeOpts.priority {opts.priority!r} (logp | margin | entropy)")
    if opts.subblock < 0 or (opts.subblock and (C % opts.subblock or opts.threshold > 0)):
        raise ValueError(f"DecodeOpts.subblock {opts.subblock}: a divisor of the block size {C}, without threshold")
    rb = cfg.read_block
    if rb and (opts.threshold > 0 or opts.remask_loop > 0 or opts.subblock not in (0, rb) or not outer_shift):
        raise ValueError(f"read_block {rb}: its groups are decoded in order, with the outer shift (no threshold, "
                         f"remasking or other sub-block)")
    prompt = known
    pos_q = (pos + 1) if outer_shift else None

    def clean_kv(ids, known):
        """The block's clean K/V as decoded so far: what the sub-block read's earlier groups hold."""
        return clean_forward(model, jnp.where(known, ids, cfg.mask_token_id), pos, state, compute_dtype)[2]

    def predict(ids, known, k_sample, own=None):
        """Sample every slot given the known ones: (tokens, log-probability of each)."""
        x = jnp.where(known, ids, cfg.mask_token_id)
        logits, _, _ = block_forward(model, x, pos, state, compute_dtype, pos_q=pos_q, own_kv=own)
        logits = logits.at[:, cfg.mask_token_id].set(-jnp.inf)          # never emit the mask token
        if outer_shift:
            p0 = prev_tail_logit if prev_tail_logit is not None else jnp.zeros_like(logits[0])
            p0 = p0.at[cfg.mask_token_id].set(-jnp.inf)                  # nor from the previous block's tail
            pred_logits = jnp.concatenate([p0[None], logits[:-1]], axis=0)
        else:
            pred_logits = logits
        if opts.presence_penalty > 0:
            present = jnp.zeros((pred_logits.shape[-1],), jnp.float32).at[x].max(1.0).at[cfg.mask_token_id].set(0.0)
            pred_logits = pred_logits - opts.presence_penalty * present[None]
        if temperature > 0:
            sampled = jax.random.categorical(k_sample, pred_logits / temperature, axis=-1,
                                             mode="high" if opts.gumbel_high else None)
        else:
            sampled = jnp.argmax(pred_logits, axis=-1)
        conf = jnp.take_along_axis(jax.nn.log_softmax(pred_logits, axis=-1), sampled[:, None], axis=1)[:, 0]
        if opts.priority == "logp":
            score = conf
        else:
            logp = jax.nn.log_softmax(pred_logits.astype(jnp.float32), axis=-1)
            if opts.priority == "margin":
                top2 = jax.lax.top_k(logp, 2)[0]
                score = jnp.exp(top2[:, 0]) - jnp.exp(top2[:, 1])
            else:   # minus the predictive entropy
                score = jnp.sum(jnp.exp(logp) * jnp.where(jnp.isfinite(logp), logp, 0.0), axis=-1)
        return sampled, conf, score

    def select(known, score, k_sample, n_commit, allowed=None):
        """The ``n_commit`` unknown (and ``allowed``) slots of highest priority."""
        prio = score if opts.order == "confidence" else \
            jax.random.uniform(jax.random.fold_in(k_sample, 1), score.shape)
        closed = known if allowed is None else (known | ~allowed)
        prio = jnp.where(closed, -jnp.inf, prio)
        order = jnp.argsort(-prio)
        rank = jnp.empty_like(order).at[order].set(jnp.arange(C))
        return (rank < n_commit) & ~closed, prio

    def make_body(n_steps, allowed=None, own=None):
        def body(i, carry):
            ids, known, key, conf_at = carry
            key, k_sample = jax.random.split(key)
            sampled, conf, score = predict(ids, known, k_sample, own)
            # commit the top-priority 1/(n_steps - i) of what is still unknown (in the allowed slots)
            open_ = ~known if allowed is None else (~known & allowed)
            n_unknown = jnp.sum(open_)
            commit, _ = select(known, score, k_sample, jnp.ceil(n_unknown / (n_steps - i)).astype(jnp.int32), allowed)
            return jnp.where(commit, sampled, ids), known | commit, key, jnp.where(commit, conf, conf_at)
        return body

    def body_threshold(carry):
        ids, known, key, conf_at, it = carry
        key, k_sample = jax.random.split(key)
        sampled, conf, score = predict(ids, known, k_sample)
        top, _ = select(known, score, k_sample, 1)
        commit = ((conf >= jnp.log(opts.threshold)) & ~known) | top
        return jnp.where(commit, sampled, ids), known | commit, key, jnp.where(commit, conf, conf_at), it + 1

    conf_at = jnp.zeros((C,), jnp.float32)
    if opts.threshold > 0:
        ids, known, key, conf_at, _ = jax.lax.while_loop(
            lambda c: jnp.any(~c[1]) & (c[4] < C), body_threshold, (ids, known, key, conf_at, jnp.int32(0)))
    elif rb:
        # the sub-block read, group by group: group g's slots g*b .. g*b+b-1 predict tokens g*b+1 .. g*b+b (token 0:
        # the previous block's tail) reading the clean K/V of the tokens before g*b; ceil(steps * b / C) iterations
        steps_g = max(1, -(-steps * rb // C))
        tok = jnp.arange(C)

        def group(g, carry):
            allowed = ((tok >= g * rb + 1) & (tok <= (g + 1) * rb)) | ((tok == 0) & (g == 0))
            return jax.lax.fori_loop(0, steps_g, make_body(steps_g, allowed, clean_kv(carry[0], carry[1])), carry)
        ids, known, key, conf_at = jax.lax.fori_loop(0, C // rb, group, (ids, known, key, conf_at))
    elif opts.subblock:
        # semi-autoregressive within the block: group g's slots only, ceil(steps * b / C) iterations each
        b = opts.subblock
        steps_g = max(1, -(-steps * b // C))
        carry = (ids, known, key, conf_at)
        for g in range(C // b):
            allowed = (jnp.arange(C) // b) == g
            carry = jax.lax.fori_loop(0, steps_g, make_body(steps_g, allowed), carry)
        ids, known, key, conf_at = carry
    else:
        ids, known, key, conf_at = jax.lax.fori_loop(0, steps, make_body(steps), (ids, known, key, conf_at))

    if opts.remask_loop > 0:
        def remask(j, carry):
            ids, key, conf_at = carry
            key, k_pick, k_sample = jax.random.split(key, 3)
            # low commit-time confidence first, with Gumbel noise so the same slots are not revisited forever
            prio = jnp.where(prompt, -jnp.inf, -conf_at + jax.random.gumbel(k_pick, (C,)))
            order = jnp.argsort(-prio)
            rank = jnp.empty_like(order).at[order].set(jnp.arange(C))
            again = (rank < opts.remask_k) & ~prompt
            sampled, conf, _ = predict(ids, ~again, k_sample)
            return jnp.where(again, sampled, ids), key, jnp.where(again, conf, conf_at)
        ids, key, conf_at = jax.lax.fori_loop(0, opts.remask_loop, remask, (ids, key, conf_at))
    if outer_shift:
        tail_logit = block_tail(model, ids, pos, state, compute_dtype)
    else:
        tail_logit = jnp.zeros((cfg.vocab_size,), jnp.float32)
    return ids, tail_logit


@eqx.filter_jit
def _decode(model, ids, known, pos, state, key, steps, temperature, outer_shift=False, prev_tail_logit=None,
            opts=DecodeOpts()):
    return decode_block(model, ids, known, pos, state, key, steps, temperature,
                        outer_shift=outer_shift, prev_tail_logit=prev_tail_logit, opts=opts)


@eqx.filter_jit
def _commit(model, ids, pos, state, key, inner_lr, inner_clip, inner_mask_ratio, fused_ce_block,
            inner_views, inner_mask_span, inner_update_rule, inner_mask, inner_task, inner_perm, inner_context_frac,
            wc=WriteConfig()):
    return commit_block(model, ids, pos, state, key, inner_lr, inner_clip, inner_mask_ratio, fused_ce_block,
                        inner_views=inner_views, inner_mask_span=inner_mask_span, inner_update_rule=inner_update_rule,
                        inner_mask=inner_mask, inner_task=inner_task, inner_perm=inner_perm,
                        inner_context_frac=inner_context_frac, wc=wc)


def generate(model: Model, prompt: Array, max_new_tokens: int, key: Array, steps: int = 16,
             temperature: float = 0.0, inner_lr: float = 1.0, inner_clip: float = 1.0,
             inner_mask_ratio: float = 0.75, fused_ce_block: int = 8192, inner_views: int = 1,
             inner_mask_span: float = 0.0, inner_update_rule: str = "sgd",
             inner_mask_mode: str = "random", unigram=None, inner_target_frac: float = 0.5,
             inner_generic_floor: float = 0.1, inner_task: str = "mask",
             outer_shift: bool = False, wc: WriteConfig = WriteConfig()) -> Array:
    """``prompt [P]`` int32 -> ``[P + max_new_tokens]`` int32.

    Blocks fully inside the prompt are only committed (B + C); the block holding the
    prompt's end is decoded around its known tokens; later blocks are decoded from
    all-masked.  The last block is decoded in full and the result trimmed.
    """
    cfg = model.cfg
    C = cfg.chunk_size
    P = int(prompt.shape[0])
    total = P + max_new_tokens
    n_blocks = -(-total // C)
    ids_all = jnp.concatenate([prompt.astype(jnp.int32), jnp.full((n_blocks * C - P,), cfg.mask_token_id, jnp.int32)])
    known_all = jnp.arange(n_blocks * C) < P
    state = init_state(model)
    out = []
    tail_logit = jnp.zeros((cfg.vocab_size,), jnp.float32)
    for b in range(n_blocks):
        sl = slice(b * C, (b + 1) * C)
        pos = jnp.arange(b * C, (b + 1) * C)
        ids, known = ids_all[sl], known_all[sl]
        key, k_dec, k_inner = jax.random.split(key, 3)
        if not bool(known.all()):
            ids, tail_logit = _decode(model, ids, known, pos, state, k_dec, steps, temperature,
                                      outer_shift=outer_shift, prev_tail_logit=tail_logit)
        else:
            if outer_shift:
                tail_logit = block_tail(model, ids, pos, state, jnp.bfloat16)
        host_mask = host_perm = None
        prefix = np.concatenate([np.asarray(x) for x in out]) if out else None   # the document so far
        if inner_task == "split":
            from .inner_masks import split_perm
            host_perm = jnp.asarray(split_perm(np.asarray(ids), inner_mask_mode, C, cfg.eos_token_id, cfg.pad_token_id,
                                               1.0 - inner_target_frac, unigram, inner_generic_floor, seed=b,
                                               prefix=prefix)) + b * C
        elif inner_mask_mode != "random":
            # the same host-side rule as training
            from .inner_masks import inner_mask_rule
            host_mask = jnp.asarray(inner_mask_rule(np.asarray(ids), inner_mask_mode, C, cfg.eos_token_id,
                                                    cfg.pad_token_id, unigram, inner_target_frac,
                                                    inner_generic_floor, seed=b, prefix=prefix))
        state = _commit(model, ids, pos, state, k_inner, inner_lr, inner_clip, inner_mask_ratio, fused_ce_block,
                        inner_views, inner_mask_span, inner_update_rule, host_mask, inner_task, host_perm,
                        1.0 - inner_target_frac, wc)
        out.append(ids)
    return jnp.concatenate(out)[:total]


# ----------------------------------------------------------------------------
# CLI: python -m ttt.generate --ckpt bucket/checkpoints/1b_8k/step_00000499 --prompt "..."
# ----------------------------------------------------------------------------
def load_model(ckpt_dir: str, model_config: Optional[str] = None) -> Model:
    """Checkpoint weights on the default device (single-device inference)."""
    import json
    import numpy as np
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from . import checkpoint as ckpt
    from .config import ModelConfig
    if model_config is None and os.path.exists(os.path.join(ckpt_dir, "model_config.json")):
        model_config = os.path.join(ckpt_dir, "model_config.json")      # the exact config the checkpoint was saved with
    if model_config is None:
        cfg_path = os.path.join(ckpt_dir, "train_config.json")
        if not os.path.exists(cfg_path):
            cfg_path = os.path.join(os.path.dirname(ckpt_dir), "train_config.json")
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path) as f:
                    model_config = json.load(f).get("model_config", "configs/1b_e2e")
            except Exception:
                model_config = "configs/1b_e2e"
        else:
            model_config = "configs/1b_e2e"
    cfg = ModelConfig.load(model_config)
    mesh = Mesh(np.array(jax.devices()[:1]), ("d",))
    # A real (randomly initialised) model as the skeleton: load_weights keeps its
    # static fields and replaces every array leaf.
    skeleton = jax.jit(lambda k: Model(cfg, k), out_shardings=NamedSharding(mesh, P()))(jax.random.PRNGKey(0))
    shardings = jax.tree_util.tree_map(lambda x: NamedSharding(mesh, P()) if eqx.is_array(x) else None,
                                       eqx.filter(skeleton, eqx.is_array))
    return ckpt.load_weights(ckpt_dir, skeleton, shardings)


def main(argv=None):
    import argparse
    import json
    from transformers import AutoTokenizer

    # Pre-parse --ckpt to read default hyperparameters from train_config.json if present
    pre_ap = argparse.ArgumentParser(add_help=False)
    pre_ap.add_argument("--ckpt", default=None)
    pre_args, _ = pre_ap.parse_known_args(argv)
    ckpt_defaults = {}
    if pre_args.ckpt:
        cfg_path = os.path.join(pre_args.ckpt, "train_config.json")
        if not os.path.exists(cfg_path):
            cfg_path = os.path.join(os.path.dirname(pre_args.ckpt), "train_config.json")
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path) as f:
                    tcfg = json.load(f)
                for k in ("model_config", "inner_task", "inner_lr", "inner_clip", "inner_views",
                          "inner_mask_mode", "inner_target_frac", "outer_shift", "inner_subchunks",
                          "inner_subchunk_scale", "inner_surprise_gamma", "inner_kvb_weight", "inner_kvb_value",
                          "inner_fast_frac", "inner_fast_lr_scale", "inner_fast_decay"):
                    if k in tcfg:
                        ckpt_defaults[k] = tcfg[k]
            except Exception:
                pass

    ap = argparse.ArgumentParser(prog="python -m ttt.generate")
    ap.add_argument("--ckpt", required=True, help="checkpoint step directory")
    ap.add_argument("--model_config", default=ckpt_defaults.get("model_config", "configs/1b_e2e"))
    ap.add_argument("--prompt", default="The history of the Roman Empire")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--steps", type=int, default=16, help="unmasking iterations per block")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--inner_lr", type=float, default=ckpt_defaults.get("inner_lr", 1.0))
    ap.add_argument("--inner_clip", type=float, default=ckpt_defaults.get("inner_clip", 1.0))
    ap.add_argument("--inner_mask_ratio", type=float, default=0.75)
    ap.add_argument("--inner_views", type=int, default=ckpt_defaults.get("inner_views", 1), help="2: complementary mask pair per block")
    ap.add_argument("--inner_mask_span", type=float, default=0.0)
    ap.add_argument("--inner_update", default="sgd", choices=["sgd", "sign", "muon2"])
    ap.add_argument("--inner_mask_mode", default=ckpt_defaults.get("inner_mask_mode", "random"), choices=["random", "rare", "docaware"])
    ap.add_argument("--inner_target_frac", type=float, default=ckpt_defaults.get("inner_target_frac", 0.5))
    ap.add_argument("--inner_task", default=ckpt_defaults.get("inner_task", "mask"), choices=["mask", "causal", "split"])
    ap.add_argument("--outer_shift", type=lambda s: s.lower() in ("1", "true", "yes"),
                    default=ckpt_defaults.get("outer_shift", False))
    # the causal write's scheme (inner_loop.WriteConfig): as trained unless overridden
    for k, typ, dflt in (("inner_subchunks", int, 1), ("inner_subchunk_scale", float, 1.0),
                         ("inner_surprise_gamma", float, 0.0), ("inner_kvb_weight", float, 0.0),
                         ("inner_kvb_value", str, "self"), ("inner_fast_frac", float, 0.0),
                         ("inner_fast_lr_scale", float, 1.0), ("inner_fast_decay", float, 0.0)):
        ap.add_argument(f"--{k}", type=typ, default=ckpt_defaults.get(k, dflt))
    ap.add_argument("--unigram_path", default="bucket/data/tokens/unigram_llama3.i64")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    wc = WriteConfig(subchunks=a.inner_subchunks, subchunk_scale=a.inner_subchunk_scale,
                     surprise_gamma=a.inner_surprise_gamma, kvb_weight=a.inner_kvb_weight, kvb_value=a.inner_kvb_value,
                     fast_frac=a.inner_fast_frac, fast_lr_scale=a.inner_fast_lr_scale, fast_decay=a.inner_fast_decay)
    tok = AutoTokenizer.from_pretrained(a.model_config)
    model = cast_floating(load_model(a.ckpt, a.model_config), jnp.bfloat16)
    prompt = jnp.asarray(tok(a.prompt, add_special_tokens=False)["input_ids"], jnp.int32)
    out = generate(model, prompt, a.max_new_tokens, jax.random.PRNGKey(a.seed), steps=a.steps,
                   temperature=a.temperature, inner_lr=a.inner_lr, inner_clip=a.inner_clip,
                   inner_mask_ratio=a.inner_mask_ratio, inner_views=a.inner_views,
                   inner_mask_span=a.inner_mask_span, inner_update_rule=a.inner_update,
                   inner_mask_mode=a.inner_mask_mode, inner_target_frac=a.inner_target_frac,
                   inner_task=a.inner_task, outer_shift=a.outer_shift, wc=wc,
                   unigram=(np.fromfile(a.unigram_path, dtype=np.int64) if a.inner_mask_mode != "random"
                            and os.path.exists(a.unigram_path) else None))
    print(tok.decode(out[len(prompt):].tolist()))


if __name__ == "__main__":
    main()
