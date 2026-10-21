# coding=utf-8
"""
Jacobian lens (J-lens) for the chunked TTT / GDN masked-diffusion LM, and the memory read-out built on it.

Reference - anthropics/jacobian-lens, the companion code of "Verbalizable Representations Form a Global
Workspace in Language Models" (transformer-circuits.pub, 2026):

    lens_l(h) = unembed(J_l h),   unembed = lm_head . final_norm,   J_l = E[ dh_L / dh_l ],

h_L being the LAST layer's residual BEFORE the final norm, so that J = I at the last layer and the lens
there is the model's own output.  Taking h_L after the final norm and decoding with the LM head alone could
not reproduce the model even at the last layer: the RMSNorm Jacobian at h annihilates h itself,
J(h) h = eps / r(h)^2 * norm(h).

Adaptations to this model (the reference targets causal decoders):

* Attention is bidirectional inside a chunk, plus the clean K/V window of the previous chunks, so a
  source position influences every position of its chunk.  Estimator "sum" (the reference's
  reduction): one-hot cotangents at every valid target position at once; the gradient at a source
  position is summed over the targets and averaged over the valid sources.  Estimator "diag" (the
  reference's per-position variant): target = source only, on a subsample of positions (one
  backward batch per position).
* The lens is fitted in the READ condition of the outer stream: a chunk after the window has filled,
  the memory written from the previous chunks (the fast weight after its writes, or the gdn state),
  an i.i.d. mask on the chunk, query positions ``pos + 1`` under the outer shift (slot i predicts token
  i + 1).  Valid slots are those whose target is masked: the slots the training loss scores.
* Injection points: ``("out", l)`` is the residual after layer l, as in the reference.  ``("mem", s)``
  is, in suffix layer s, the residual right after the memory's read is added (after attention, before
  that layer's base MLP): the point where a memory contribution enters the network.

Memory contributions.  In the memory-on read forward, suffix layer s adds

    TTT: p_s = post_norm_s(W_s(norm_s(h)))        GDN: p_s = read_s(norm_s(h), S_s)

and the memory's own part is ``m_s = p_s(W) - p_s(W_0)`` on the same input (GDN: ``p_s(S) - p_s(0)``,
and an empty state reads exactly zero).  Its effect on the logits is read three ways: the exact
first-order effect in context (``direct_effect``, a JVP of the network above the injection point, all
positions at once), the lens (``decode_delta``: ``lm_head . J_norm(h_L) . J_bar m``, per position and
context-free), and the exact non-linear total (``read_forward`` with the memory and with it reset).
"""
from __future__ import annotations

import json
import os
from typing import NamedTuple, Optional, Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from src.ttt.attention import key_validity, own_mask_matrix, read_mask_matrix
from src.ttt.generate import _split, block_forward, clean_forward, commit_block, init_state
from src.ttt.inner_loop import WriteConfig, inner_lr as inner_lr_ramp
from src.ttt.model import Model, RMSNorm, cast_floating


class GDNState(NamedTuple):
    """The gdn model's counterpart of ``generate.DecodeState``."""
    kv: Tuple[Array, Array]      # clean K/V windows of every layer, [L, W*C, nkv, d] x2
    S: Array                     # gdn states of the suffix layers, [S, nh, dv, dk] f32
    chunk: Array                 # index of the next chunk (int32 scalar)


class Deltas(NamedTuple):
    """Additive injections of one read forward: ``out [L, C, H]`` after every layer, ``mem [S, C, H]``
    at every suffix layer's memory point."""
    out: Array
    mem: Array


class Read(NamedTuple):
    """One read forward.  ``h_L [C, H]``: the last layer's residual before the final norm; ``logits
    [C, V]``; the residual at every point, injections included (``out [L, C, H]``, ``mem [S, C, H]``);
    per suffix layer the memory's input ``mem_in``, its read ``mem_read`` and the read of the
    reference memory ``mem_ref`` (TTT: W_0; zeros when not asked for, and always for gdn)."""
    h_L: Array
    logits: Array
    out: Array
    mem: Array
    mem_in: Array
    mem_read: Array
    mem_ref: Array


class Transplant(NamedTuple):
    """Key transplant (``key_transplant.py``): at the slots ``sel [C]`` every fast-weight layer's memory
    contribution ``post_norm(W(x)) - post_norm(W_0(x))`` is taken at ``x_src [F, C, H]`` instead of the
    read's own input ``x``; the template's part ``post_norm(W_0(x))`` stays the read's.  With ``x_src`` the
    write's inputs at the matching slots (``write_keys``), the read queries the memory with the write's keys.
    ``x_src`` None: the read's own input.  ``scale``: the contribution times this (an oracle read gate that
    amplifies the memory at the selected slots only), a scalar or one per slot ``[C]``."""
    x_src: Optional[Array]
    sel: Array
    scale: Union[float, Array] = 1.0


class WriteSettings(NamedTuple):
    """How a checkpoint writes a clean chunk into its memory (TTT; the gdn update has no settings)."""
    inner_lr: float                  # the value of the inner-lr ramp at the checkpoint's step
    inner_clip: float = 1.0
    inner_task: str = "causal"
    inner_mask_ratio: float = 0.5    # the mask task only
    outer_shift: bool = True         # the read's query positions / slot targets, not the write
    fused_ce_block: int = 8192
    wc: WriteConfig = WriteConfig()
    rule: str = "sgd"                # the step's direction (TrainConfig.inner_update: sgd | sign | muon2)


def write_settings(ckpt_dir: str, **override) -> WriteSettings:
    """The checkpoint's own write: its inner-lr ramp at its step, clip, task, scheme and outer shift
    (``train_config.json`` / ``state.json`` of the step directory).  ``override``: None = keep."""
    from src.ttt.config import TrainConfig
    from src.ttt.train import write_config
    tc = TrainConfig.load(os.path.join(ckpt_dir, "train_config.json"))
    with open(os.path.join(ckpt_dir, "state.json")) as f:
        step = int(json.load(f)["step"])
    lr = float(inner_lr_ramp(jnp.int32(step), tc.inner_lr, tc.inner_lr_warmup_steps))
    ws = WriteSettings(lr, tc.inner_clip, tc.inner_task, tc.inner_mask_ratio if tc.inner_mask_ratio >= 0 else 0.5,
                       tc.outer_shift, tc.fused_ce_block, write_config(tc), tc.inner_update)
    return ws._replace(**{k: v for k, v in override.items() if v is not None})


# ----------------------------------------------------------------------------
# memory state: empty, write a clean chunk, reset
# ----------------------------------------------------------------------------
def init_memory(model: Model, dtype=jnp.float32):
    """No window, the memory at its template (TTT: W_0; gdn: zero states)."""
    if model.gdn is None:
        return init_state(model, dtype)
    cfg = model.cfg
    shape = (cfg.num_hidden_layers, cfg.window_chunks * cfg.chunk_size, cfg.num_key_value_heads, cfg.head_dim)
    S = jnp.zeros((cfg.suffix_len, cfg.gdn_heads, cfg.gdn_head_dim, cfg.gdn_head_dim), jnp.float32)
    return GDNState((jnp.zeros(shape, dtype), jnp.zeros(shape, dtype)), S, jnp.int32(0))


def reset_memory(model: Model, state):
    """The same windows with the memory reset (TTT: W_0; gdn: zero states) - the memory-off read."""
    if model.gdn is None:
        return state._replace(W=cast_floating(model.fast_weight, jnp.float32))
    return state._replace(S=jnp.zeros_like(state.S))


def commit(model: Model, clean_ids: Array, pos: Array, state, key: Array, ws: WriteSettings,
           compute_dtype=jnp.float32):
    """A clean chunk becomes context: its K/V shift into the windows and it is written into the memory.
    TTT: the checkpoint's own write (``generate.commit_block``, the training's forward B + write C).
    gdn: the clean forward's delta-rule update, as the training's clean stream writes it."""
    if model.gdn is None:
        return commit_block(model, clean_ids, pos, state, key, ws.inner_lr, ws.inner_clip, ws.inner_mask_ratio,
                            ws.fused_ce_block, compute_dtype, inner_update_rule=ws.rule, inner_task=ws.inner_task,
                            wc=ws.wc)
    C = model.cfg.chunk_size
    kv_own, S_next = _gdn_clean(model, clean_ids, pos, state, compute_dtype)
    kv_next = tuple(jnp.concatenate([cache[:, C:], own.astype(cache.dtype)], axis=1)
                    for cache, own in zip(state.kv, kv_own))
    return GDNState(kv_next, S_next, state.chunk + 1)


def _gdn_clean(model: Model, clean_ids: Array, pos: Array, state, compute_dtype):
    """The gdn model's clean forward of a chunk: ``(kv_own [L, C, nkv, d] x2, the states after its update)``
    (with ``read_block`` block-causal within the chunk, as the training's clean stream)."""
    cfg = model.cfg
    key_valid = key_validity(state.chunk, cfg.chunk_size, cfg.window_chunks, cfg.reach)
    own = own_mask_matrix(cfg.chunk_size, f"block:{cfg.read_block}") if cfg.read_block else None
    kv_pre, kv_suf = _split(model, state.kv)
    h = model.embed_tokens(clean_ids).astype(compute_dtype)
    h, kv_own_pre = jax.lax.scan(lambda h, xs: xs[0](h, pos, xs[1], key_valid, own_mask=own), h, (model.prefix, kv_pre))
    _, kv_own_suf, S_next = model.suffix_forward(h, pos, kv_suf, None, key_valid, memory_state=state.S,
                                                 update_memory=True, own_mask=own)
    return tuple(jnp.concatenate([a, b], axis=0) for a, b in zip(kv_own_pre, kv_own_suf)), S_next


def clean_own(model: Model, clean_ids: Array, pos: Array, state, compute_dtype=jnp.float32):
    """``kv_own [L, C, nkv, d] x2``: the clean forward of a chunk against ``state`` - the sub-block read's own
    clean K/V (``ModelConfig.read_block``; ``read_forward``'s ``own_kv``)."""
    if model.gdn is None:
        return clean_forward(model, clean_ids, pos, state, compute_dtype)[2]
    return _gdn_clean(model, clean_ids, pos, state, compute_dtype)[0]


def write_keys(model: Model, clean_ids: Array, pos: Array, state, ws: WriteSettings, compute_dtype=jnp.float32) -> Array:
    """The fast-weight layers' inputs ``[F, C, H]`` of the causal write of the clean chunk ``clean_ids``
    against ``state`` (before its write): ``commit_block``'s forward C, whose loss the write descends.
    Slot i holds token i and predicts token i + 1: these are the keys the write files the next tokens under."""
    if model.gdn is not None or ws.inner_task != "causal":
        raise ValueError("write_keys: the TTT memory's causal write only")
    C = model.cfg.chunk_size
    pos_q = jnp.concatenate([pos[1:], pos[-1:]])
    own = jnp.asarray(own_mask_matrix(C, "causal"))
    return block_forward(model, clean_ids, pos, state, compute_dtype, pos_q=pos_q, own_mask=own, return_prime=True)[3][0]


# ----------------------------------------------------------------------------
# the read forward with injections
# ----------------------------------------------------------------------------
def zero_deltas(model: Model, dtype=jnp.float32) -> Deltas:
    cfg = model.cfg
    C, H = cfg.chunk_size, cfg.hidden_size
    return Deltas(jnp.zeros((cfg.num_hidden_layers, C, H), dtype), jnp.zeros((cfg.fast_len, C, H), dtype))


def _read_prefix(model: Model, ids: Array, pos: Array, state, d_out_pre: Array, pos_q, compute_dtype, own_pre=None):
    """The prefix layers of the read: ``(h [C, H], residual after each layer [L-S, C, H])``.  ``own_pre``: the
    sub-block read's clean own-chunk K/V of the prefix layers."""
    cfg = model.cfg
    key_valid = key_validity(state.chunk, cfg.chunk_size, cfg.window_chunks, cfg.reach)
    kv_pre, _ = _split(model, state.kv)
    own_mask = None if own_pre is None else read_mask_matrix(cfg.chunk_size, cfg.read_block)

    def layer(h, xs):
        blk, kv_l, d, ok_l = xs
        h = blk(h, pos, kv_l, key_valid, pos_q=pos_q, own_mask=own_mask, own_kv=ok_l)[0] + d.astype(compute_dtype)
        return h, h

    return jax.lax.scan(layer, model.embed_tokens(ids).astype(compute_dtype), (model.prefix, kv_pre, d_out_pre, own_pre))


def _read_suffix(model: Model, h: Array, pos: Array, state, d_out_suf: Array, d_mem: Array, pos_q,
                 compute_dtype, ref: bool, transplant: Optional[Transplant] = None, decap: bool = False, own_suf=None,
                 den_W=None):
    """The suffix layers of the read, ``Block.__call__`` opened at the memory point:
    ``(h_L, out [S, C, H], mem [F, C, H], mem_in, mem_read, mem_ref)``.  The memory point and reads exist in
    the F fast-weight layers only (``ModelConfig.fast_mask``); plain suffix layers run without them.
    ``transplant``: the memory's contributions at its slots taken at its inputs (``Transplant``).
    ``decap``: the fast branch normalised by the TEMPLATE's RMS, ``gamma * W(x) / rms(W_0(x))``, instead of
    its own - equal at W = W_0, but the memory adds to the template's output instead of rotating it
    within a fixed norm (a read-time variant, the model was trained without it).
    ``den_W``: the same with the RMS of ANOTHER fast weight's output on the same input,
    ``gamma * W(x) / rms(W_den(x))`` (``memory_distance.py``: the memory read through the normalisation of a
    smaller memory - how much the later writes cost through the shared denominator alone)."""
    if (transplant is not None or decap or den_W is not None) and model.gdn is not None:
        raise ValueError("key transplant / decap / den_W: the TTT memory only")
    if sum((transplant is not None, decap, den_W is not None)) > 1:
        raise ValueError("transplant, decap and den_W: one at a time")
    cfg = model.cfg
    key_valid = key_validity(state.chunk, cfg.chunk_size, cfg.window_chunks, cfg.reach)
    _, kv_suf = _split(model, state.kv)
    read_own = None if own_suf is None else read_mask_matrix(cfg.chunk_size, cfg.read_block)

    def attention(blk, h, kv_l, ok_l):
        q, k, v = blk.attn.qkv(blk.attn_norm(h), pos, pos_q)
        if ok_l is None:
            K = jnp.concatenate([kv_l[0].astype(k.dtype), k], axis=0)
            V = jnp.concatenate([kv_l[1].astype(v.dtype), v], axis=0)
            kvld = key_valid
        else:                                                       # the sub-block read (Block.__call__'s own_kv)
            K = jnp.concatenate([kv_l[0].astype(k.dtype), ok_l[0].astype(k.dtype), k], axis=0)
            V = jnp.concatenate([kv_l[1].astype(v.dtype), ok_l[1].astype(v.dtype), v], axis=0)
            kvld = jnp.concatenate([key_valid, jnp.ones((h.shape[0],), jnp.bool_)])
        return h + blk.attn.out(blk.attn.attend(q[None], K[None], V[None], kvld[None], own_mask=read_own)[0])

    def tail(blk, h, p, dm, do):
        h = h + p + dm.astype(compute_dtype)
        h_mem = h
        h = h + blk.mlp(blk.mlp_norm(h)) + do.astype(compute_dtype)
        return h, h_mem

    if model.gdn is None:
        def layer(h, xs):
            blk, pn, ppn, w, w0, kv_l, do, dm, xt, ok_l, wd = xs
            h = attention(blk, h, kv_l, ok_l)
            x = pn(h)
            if decap or wd is not None:
                y0 = (w0 if decap else wd)(x).astype(jnp.float32)
                inv = jax.lax.rsqrt(jnp.mean(jnp.square(y0), axis=-1, keepdims=True) + ppn.eps)
                p = (ppn.weight * w(x).astype(jnp.float32) * inv).astype(x.dtype)
            else:
                p = ppn(w(x))
            p_ref = ppn(w0(x)) if ref else jnp.zeros_like(p)
            if transplant is not None:
                p0 = ppn(w0(x))
                m_t = (p - p0) if xt is None else (ppn(w(xt)) - ppn(w0(xt)))
                sc = jnp.asarray(transplant.scale, p.dtype)
                p = jnp.where(transplant.sel[:, None], p0 + (sc[:, None] if sc.ndim == 1 else sc) * m_t, p)
            h, h_mem = tail(blk, h, p, dm, do)
            return h, (h, h_mem, x, p, p_ref)

        def plain_layer(h, xs):
            blk, kv_l, do, ok_l = xs
            h = attention(blk, h, kv_l, ok_l)
            h = h + blk.mlp(blk.mlp_norm(h)) + do.astype(compute_dtype)
            return h, h

        W, W0 = cast_floating(state.W, compute_dtype), cast_floating(model.fast_weight, compute_dtype)
        Wd = None if den_W is None else cast_floating(den_W, compute_dtype)
        x_src = None if (transplant is None or transplant.x_src is None) else transplant.x_src.astype(compute_dtype)
        outs, mems = [], []
        own_runs = [r[2] for r in model._suffix_runs(own_suf)]
        for (fast, a, b), (_, blocks, kv_r, fsl), ok_r in zip(cfg.suffix_runs, model._suffix_runs(kv_suf), own_runs):
            if fast:
                h, ys = jax.lax.scan(layer, h, (blocks, fsl(model.prime_norm), fsl(model.prime_post_norm), fsl(W),
                                                fsl(W0), kv_r, d_out_suf[a:b], fsl(d_mem),
                                                None if x_src is None else fsl(x_src), ok_r,
                                                None if Wd is None else fsl(Wd)))
                outs.append(ys[0])
                mems.append(ys[1:])
            else:
                h, out_r = jax.lax.scan(plain_layer, h, (blocks, kv_r, d_out_suf[a:b], ok_r))
                outs.append(out_r)
        cat = lambda parts: parts[0] if len(parts) == 1 else jnp.concatenate(parts, axis=0)
        return (h, cat(outs)) + tuple(cat([m[i] for m in mems]) for i in range(4))

    # memory gdn: every suffix layer reads its state
    def layer(h, xs):
        blk, mn, g, S_l, kv_l, do, dm, ok_l = xs
        h = attention(blk, h, kv_l, ok_l)
        x = mn(h)
        p = g.read(x, S_l)
        h, h_mem = tail(blk, h, p, dm, do)
        return h, (h, h_mem, x, p, jnp.zeros_like(p))           # an empty state reads exactly zero

    h, ys = jax.lax.scan(layer, h, (model.suffix, model.mem_norm, model.gdn, state.S, kv_suf, d_out_suf, d_mem,
                                    own_suf))
    return (h,) + ys


def read_forward(model: Model, ids: Array, pos: Array, state, deltas: Optional[Deltas] = None,
                 outer_shift: bool = True, compute_dtype=jnp.float32, ref: bool = False,
                 transplant: Optional[Transplant] = None, decap: bool = False, own_kv=None, den_W=None) -> Read:
    """The outer stream's read of one chunk ``ids [C]`` (masked) against ``state`` (windows + memory),
    with additive injections; with zero injections it is ``generate.block_forward`` exactly
    (tested).  ``outer_shift``: query positions ``pos + 1`` (slot i predicts token i + 1).
    ``ref``: also the reference memory's read at every suffix layer (TTT: W_0 on the same input).
    ``transplant`` / ``decap`` / ``den_W``: see ``Transplant`` / ``_read_suffix``.  ``own_kv``: a ``read_block``
    model's read - the chunk's clean K/V of every layer (``clean_own``, against the same state)."""
    cfg = model.cfg
    if cfg.read_block and own_kv is None:
        raise ValueError("read_block: the read needs the chunk's clean K/V (own_kv = clean_own(...))")
    n_pre = cfg.num_hidden_layers - cfg.suffix_len
    d = zero_deltas(model, compute_dtype) if deltas is None else deltas
    pos_q = pos + 1 if outer_shift else None
    own_pre, own_suf = (None, None) if own_kv is None else _split(model, own_kv)
    h, out_pre = _read_prefix(model, ids, pos, state, d.out[:n_pre], pos_q, compute_dtype, own_pre)
    h_L, out_suf, mem, mem_in, mem_read, mem_ref = _read_suffix(model, h, pos, state, d.out[n_pre:], d.mem,
                                                                 pos_q, compute_dtype, ref, transplant, decap, own_suf,
                                                                 den_W)
    logits = model.logits(model.final_norm(h_L))
    return Read(h_L, logits, jnp.concatenate([out_pre, out_suf], axis=0), mem, mem_in, mem_read, mem_ref)


def target_fn(model: Model, ids: Array, pos: Array, state, scope: str = "suffix", outer_shift: bool = True,
              compute_dtype=jnp.float32):
    """``(f, primals)``: ``f(d_out, d_mem) -> h_L [C, H]`` over the injections of the points in scope.
    "suffix": the suffix layers' out and mem points (the prefix is run once, outside the graph);
    "all": every layer's out point and the suffix mem points."""
    cfg = model.cfg
    if cfg.read_block:
        raise NotImplementedError("read_block: the lens' target function has no sub-block read")
    n_pre = cfg.num_hidden_layers - cfg.suffix_len
    pos_q = pos + 1 if outer_shift else None
    zeros = zero_deltas(model, compute_dtype)
    if scope == "suffix":
        h_pre = jax.lax.stop_gradient(_read_prefix(model, ids, pos, state, zeros.out[:n_pre], pos_q, compute_dtype)[0])

        def f(d_out_suf, d_mem):
            return _read_suffix(model, h_pre, pos, state, d_out_suf, d_mem, pos_q, compute_dtype, False)[0]
        return f, (zeros.out[n_pre:], zeros.mem)
    if scope != "all":
        raise ValueError(f"scope {scope!r}: suffix | all")

    def f(d_out, d_mem):
        return read_forward(model, ids, pos, state, Deltas(d_out, d_mem), outer_shift, compute_dtype).h_L
    return f, (zeros.out, zeros.mem)


def scope_points(model: Model, scope: str):
    """The points ``f`` of ``target_fn`` differentiates, in the order of its outputs:
    ``(out points, mem points)``."""
    cfg = model.cfg
    n_pre = cfg.num_hidden_layers - cfg.suffix_len
    outs = range(n_pre, cfg.num_hidden_layers) if scope == "suffix" else range(cfg.num_hidden_layers)
    return [("out", l) for l in outs], [("mem", s) for s in range(cfg.fast_len)]


# ----------------------------------------------------------------------------
# estimators: one example -> per point the [H_out, H_in] average Jacobian
# ----------------------------------------------------------------------------
def _dim_batches(H: int, dim_batch: int) -> Array:
    """``[nb, b]`` output dimensions, the last batch padded by repeating dimension 0 (dropped later)."""
    nb = -(-H // dim_batch)
    dims = jnp.arange(nb * dim_batch)
    return jnp.where(dims < H, dims, 0).reshape(nb, dim_batch)


def jacobian_sum(f, primals, valid: Array, dim_batch: int = 32):
    """The reference's reduction: ``J[i, :] = mean_{s valid} sum_{t valid} dh_L[t, i] / dd[s, :]`` for
    every injection array of ``primals`` (each ``[P, C, H]``).  Returns a tuple of ``[P, H, H]`` f32
    (rows = output dimensions of h_L, columns = the injection's dimensions)."""
    h_L, vjp = jax.vjp(f, *primals)
    C, H = h_L.shape
    tgt = valid.astype(h_L.dtype)[:, None]
    src = valid.astype(jnp.float32) / jnp.maximum(jnp.sum(valid.astype(jnp.float32)), 1.0)

    def rows(dims):
        ct = jax.nn.one_hot(dims, H, dtype=h_L.dtype)[:, None, :] * tgt[None]              # [b, C, H]
        grads = jax.vmap(vjp)(ct)
        return tuple(jnp.einsum("bpch,c->pbh", g.astype(jnp.float32), src) for g in grads)  # [P, b, H]

    blocks = jax.lax.map(rows, _dim_batches(H, dim_batch))                                   # [nb, P, b, H]
    return tuple(jnp.moveaxis(b, 0, 1).reshape(b.shape[1], -1, H)[:, :H] for b in blocks)


def jacobian_diag(f, primals, positions: Array, dim_batch: int = 32):
    """The per-position variant: ``J[i, :] = mean_{p in positions} dh_L[p, i] / dd[p, :]``
    (``positions [K]`` int).  Same layout as ``jacobian_sum``."""
    h_L, vjp = jax.vjp(f, *primals)
    C, H = h_L.shape
    dims = _dim_batches(H, dim_batch)

    def at(p):
        sel = jax.nn.one_hot(p, C, dtype=h_L.dtype)                                         # [C]

        def rows(dims):
            ct = jax.nn.one_hot(dims, H, dtype=h_L.dtype)[:, None, :] * sel[None, :, None]
            grads = jax.vmap(vjp)(ct)
            return tuple(jnp.einsum("bpch,c->pbh", g.astype(jnp.float32), sel.astype(jnp.float32)) for g in grads)

        blocks = jax.lax.map(rows, dims)
        return tuple(jnp.moveaxis(b, 0, 1).reshape(b.shape[1], -1, H)[:, :H] for b in blocks)

    per = jax.lax.map(at, positions)                                                         # [K, P, H, H] each
    return tuple(jnp.mean(x, axis=0) for x in per)


# ----------------------------------------------------------------------------
# decoding
# ----------------------------------------------------------------------------
def norm_jvp(norm: RMSNorm, h: Array, v: Array) -> Array:
    """The exact first-order change of ``norm(h)`` along ``v``, row by row:
    ``g * (v / r - h (h . v) / (d r^3))``, ``r = sqrt(mean(h^2) + eps)``."""
    h32, v32 = h.astype(jnp.float32), v.astype(jnp.float32)
    d = h32.shape[-1]
    r2 = jnp.mean(jnp.square(h32), axis=-1, keepdims=True) + norm.eps
    r = jnp.sqrt(r2)
    return norm.weight * (v32 / r - h32 * jnp.sum(h32 * v32, axis=-1, keepdims=True) / (d * r2 * r))


def decode_residual(model: Model, J: Array, h: Array) -> Array:
    """The lens of the reference: the residual ``h [.., H]`` at a point -> logits ``[.., V]``,
    ``lm_head(final_norm(J h))``.  At ``("out", L-1)`` J = I and this is the model's output."""
    return model.logits(model.final_norm(h.astype(jnp.float32) @ jnp.asarray(J, jnp.float32).T))


def decode_delta(model: Model, J: Array, m: Array, h_L: Optional[Array] = None) -> Array:
    """A perturbation ``m [.., H]`` at a point -> the first-order change of the logits ``[.., V]``:
    transported by J, then through the final norm's exact Jacobian at ``h_L`` (the position's own
    final residual); without ``h_L`` only the norm's gain (a context-free direction read-out)."""
    u = m.astype(jnp.float32) @ jnp.asarray(J, jnp.float32).T
    dn = norm_jvp(model.final_norm, h_L, u) if h_L is not None else model.final_norm.weight * u
    return model.logits(dn)


def direct_effect(model: Model, ids: Array, pos: Array, state, tangent: Deltas, outer_shift: bool = True,
                  compute_dtype=jnp.float32) -> Array:
    """The exact first-order change of the logits ``[C, V]`` when the injections move along ``tangent``
    (every point and position at once, attention between positions included): a JVP of the read."""
    zeros = zero_deltas(model, compute_dtype)

    def f(d_out, d_mem):
        return read_forward(model, ids, pos, state, Deltas(d_out, d_mem), outer_shift, compute_dtype).h_L

    h_L, dh = jax.jvp(f, (zeros.out, zeros.mem), (tangent.out.astype(compute_dtype), tangent.mem.astype(compute_dtype)))
    return model.logits(norm_jvp(model.final_norm, h_L, dh))


def mem_tangent(model: Model, m: Array, layer: Optional[int] = None, dtype=jnp.float32) -> Deltas:
    """The tangent that adds the memory contributions ``m [S, C, H]`` at the mem points (one suffix
    layer's only, with ``layer``)."""
    z = zero_deltas(model, dtype)
    mem = m.astype(dtype) if layer is None else z.mem.at[layer].set(m[layer].astype(dtype))
    return Deltas(z.out, mem)


def read_mask(clean: Array, ratio: float, key: Array, mask_id: int, outer_shift: bool = True):
    """The read condition's corruption of one chunk: i.i.d. mask at ``ratio``.  Returns ``(ids,
    masked [C] bool, valid [C] bool)``, valid = the slots the training loss scores (the target -
    token i + 1 under the outer shift, token i otherwise - is masked; under the shift the last slot's
    target lies in the next chunk and is never valid)."""
    masked = jax.random.uniform(key, clean.shape) < ratio
    ids = jnp.where(masked, mask_id, clean)
    if outer_shift:
        valid = jnp.concatenate([masked[1:], jnp.zeros((1,), bool)])
    else:
        valid = masked
    return ids, masked, valid
