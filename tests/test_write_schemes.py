# coding=utf-8
"""The causal write's schemes (inner_loop.WriteConfig / write_chunk): sub-chunk writes, surprise
weighting, key-value binding in the fast weight, the two-timescale split of W and K whole-chunk
steps, the normalised step - their mechanics, the plain default staying bitwise
unchanged, and TRAINING == DECODING: the loss the training scan reports for chunk 1 equals the loss the decoder
gets after committing chunk 0 with the same scheme (generate.commit_block calls the same write_chunk).  In the
training scan every own-chunk mask reaches attention static (numpy): on the GPU cuDNN builds its masked kernels
from it, and a traced mask fails there only.

    JAX_PLATFORMS=cpu .venv/bin/python tests/test_write_schemes.py
"""
from __future__ import annotations

import dataclasses
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")
os.environ.setdefault("TTT_ATTENTION", "xla")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from src.ttt.attention import key_validity, own_mask_matrix
from src.ttt.chunk_scan import Batch, LossConfig, loss_fn
from src.ttt.config import ModelConfig
from src.ttt.generate import block_forward, commit_block, init_state
from src.ttt.inner_loop import (WriteConfig, _bind_err_plain, fast_unit_mask, inner_grad, inner_objective,
                                 inner_update, mask_units, token_ce, write_chunk)
from src.ttt.model import Model, cast_floating
from src.ttt.objective import q_sample

CFG = ModelConfig(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=8, suffix_len=2,
                  chunk_size=8, window_chunks=1, mask_token_id=120, pad_token_id=0, eos_token_id=0,
                  inner_kvb_head=True)
C, Wn, S, H = CFG.chunk_size, CFG.window_chunks, CFG.suffix_len, CFG.hidden_size
failures = []


def report(name, ok, detail=""):
    failures.append(name) if not ok else None
    print(f"[{'PASS' if ok else 'FAIL'}] {name:<78} {detail}")


def leaves(t):
    return jax.tree_util.tree_leaves(eqx.filter(t, eqx.is_array))


def maxdiff(a, b):
    return max(float(jnp.max(jnp.abs(x - y))) for x, y in zip(leaves(a), leaves(b)))


def chunk_inputs(model, ids_c, seed=3):
    """One causal-write chunk for B=1: the inner stream's prefix output, targets, weights,
    positions, a random window and its validity - what the training scan hands the write."""
    B = 1
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    pos = jnp.arange(C, 2 * C)
    kv = (jax.random.normal(k1, (B, S, Wn * C, 2, 8)) * 0.3, jax.random.normal(k2, (B, S, Wn * C, 2, 8)) * 0.3)
    kvd = key_validity(jnp.int32(1), C, Wn)
    own = jnp.asarray(own_mask_matrix(C, "causal"))
    h_in = model.embed_tokens(ids_c)[None].astype(jnp.float32) * 30.0
    tgt = jnp.concatenate([ids_c[1:], ids_c[-1:]])[None]
    w = (jnp.arange(C) <= C - 2)[None]
    pq = jnp.concatenate([pos[1:], pos[-1:]])[None]
    return h_in, tgt, w, kv, pos[None], kvd, pq, own


def main():
    model = Model(CFG, jax.random.PRNGKey(0))
    W0 = jax.tree_util.tree_map(lambda x: x[None].astype(jnp.float32) if eqx.is_inexact_array(x) else x,
                                model.fast_weight)
    ids_c = jax.random.randint(jax.random.PRNGKey(1), (C,), 1, 100)
    h_in, tgt, w, kv, pos, kvd, pq, own = chunk_inputs(model, ids_c)
    common = (model, h_in, tgt, w, kv, pos, kvd, jnp.float32(0.5), 1.0, 0, jnp.float32, "sgd", None, pq, own)

    # 1. the plain scheme is exactly one inner_update
    W_a, _ = write_chunk(W0, *common, WriteConfig())
    W_b, _ = inner_update(W0, model, h_in, tgt, w, kv, pos, kvd, jnp.float32(0.5), 1.0, 0, jnp.float32, "sgd",
                          h_clean=None, pos_q=pq, own_mask=own)
    report("plain WriteConfig == one inner_update (bitwise)", maxdiff(W_a, W_b) == 0.0)

    # 1b. K whole-chunk steps == K explicit inner_updates, the gradient recomputed each time
    W_k, st_k = write_chunk(W0, *common, WriteConfig(steps=3), remat=False)
    W_kr, _ = write_chunk(W0, *common, WriteConfig(steps=3))
    W_e, losses = W0, []
    for _ in range(3):
        W_e, st_e = inner_update(W_e, model, h_in, tgt, w, kv, pos, kvd, jnp.float32(0.5), 1.0, 0, jnp.float32, "sgd",
                                 h_clean=None, pos_q=pq, own_mask=own)
        losses.append(float(st_e.loss))
    report("steps 3 == three explicit steps (gradient recomputed), inner loss falls, remat identical",
           maxdiff(W_k, W_e) < 1e-6 and maxdiff(W_k, W_kr) < 1e-6 and losses[0] > losses[1] > losses[2]
           and abs(float(st_k.loss) - np.mean(losses)) < 1e-5,
           f"inner loss {' '.join(f'{x:.4f}' for x in losses)}, max diff {maxdiff(W_k, W_e):.1e}")

    # 2. sub-chunk mechanics: the truncated forward over slots [0, e) equals the full chunk's first e
    #    slots (causal own-chunk attention), and write_chunk(n) == n explicit truncated steps
    Wc = cast_floating(model.fast_weight, jnp.float32)
    full, _ = model.suffix_forward(h_in[0], pos[0], (kv[0][0], kv[1][0]), Wc, kvd, inner=True, pos_q=pq[0], own_mask=own)
    ok = True
    for e in (2, 4, 6):
        part, _ = model.suffix_forward(h_in[0, :e], pos[0, :e], (kv[0][0], kv[1][0]), Wc, kvd[:Wn * C + e], inner=True,
                                       pos_q=pq[0, :e], own_mask=own[:e, :e])
        ok &= bool(jnp.allclose(part, full[:e], atol=1e-5))
    report("truncated inner forward over [0, e) == the full chunk's first e slots", ok)
    n = 2
    W_n, _ = write_chunk(W0, *common, WriteConfig(subchunks=n), remat=False)
    W_r = W0
    for k in range(n):
        s, e = k * C // n, (k + 1) * C // n
        m_k = w[:, :e].astype(jnp.float32) * (jnp.arange(e) >= s)[None]
        W_r, _ = inner_update(W_r, model, h_in[:, :e], tgt[:, :e], m_k, kv, pos[:, :e], kvd[:Wn * C + e],
                              jnp.float32(0.5), 1.0, 0, jnp.float32, "sgd", pos_q=pq[:, :e], own_mask=own[:e, :e])
    report("subchunks 2 == two explicit truncated steps; differs from one step",
           maxdiff(W_n, W_r) < 1e-6 and maxdiff(W_n, W_a) > 1e-6, f"|dW n2 - n1| {maxdiff(W_n, W_a):.2e}")
    # the sub-steps score only their own slots (score_from): == the unsliced steps, also with the
    # surprise weights and the binding term (value source: another tensor, to cover its slicing)
    n = 4
    ksrc = jax.random.normal(jax.random.PRNGKey(4), (1, S, C, CFG.hidden_size))
    kw = dict(surprise_gamma=1.0, kvb_weight=1.0)
    W_n, _ = write_chunk(W0, *common, WriteConfig(subchunks=n, surprise_gamma=1.0, kvb_weight=1.0), kvb_src=ksrc,
                         remat=False)
    W_r = W0
    for k in range(n):
        s, e = k * C // n, (k + 1) * C // n
        m_k = w[:, :e].astype(jnp.float32) * (jnp.arange(e) >= s)[None]
        W_r, _ = inner_update(W_r, model, h_in[:, :e], tgt[:, :e], m_k, kv, pos[:, :e], kvd[:Wn * C + e],
                              jnp.float32(0.5), 1.0, 0, jnp.float32, "sgd", pos_q=pq[:, :e], own_mask=own[:e, :e],
                              kvb_src=ksrc[:, :, :e], **kw)
    report("sub-steps scoring only their slots == unsliced steps (surprise + binding)", maxdiff(W_n, W_r) < 1e-6,
           f"max |dW| {maxdiff(W_n, W_r):.1e}")
    # ... and score_from really drops the slots before it: their weights are ignored
    ones = jnp.ones((C,), jnp.float32)
    obj = lambda m_, sf: float(inner_objective(Wc, model, h_in[0], tgt[0], m_, (kv[0][0], kv[1][0]), pos[0], kvd, 0,
                                               jnp.float32, pos_q=pq[0], own_mask=own, surprise_gamma=1.0,
                                               kvb_weight=1.0, kvb_src=ksrc[0], score_from=sf)[0])
    o_all, o_sf, o_zero = obj(ones, 0), obj(ones, 4), obj(ones * (jnp.arange(C) >= 4), 0)
    report("score_from ignores the slots before it (== zero weights there, != all slots)",
           abs(o_sf - o_zero) < 1e-6 and abs(o_sf - o_all) > 1e-4, f"{o_sf:.6f} / {o_zero:.6f} / all {o_all:.6f}")
    # the step budget: with subchunk_scale 1/n every step is clipped to clip/n
    W_s, _ = write_chunk(W0, model, h_in, tgt, w, kv, pos, kvd, jnp.float32(1.0), 1.0, 0, jnp.float32, "sgd", None, pq,
                         own, WriteConfig(subchunks=4, subchunk_scale=0.25))
    dnorm = float(jnp.sqrt(sum(jnp.sum(jnp.square(a - b)) for a, b in zip(leaves(W_s), leaves(W0)))))
    report("subchunk_scale 1/n keeps the chunk's total step within one clip", dnorm <= 1.0 + 1e-4, f"|dW| {dnorm:.3f}")

    # 3. surprise weighting: gamma 0 == plain; gamma > 0 == a manual weighted step; token_ce == dense CE
    W_g0, _ = write_chunk(W0, *common, WriteConfig(surprise_gamma=0.0))
    W_g1, _ = write_chunk(W0, *common, WriteConfig(surprise_gamma=1.0))
    h_out, _ = model.suffix_forward(h_in[0], pos[0], (kv[0][0], kv[1][0]), Wc, kvd, inner=True, pos_q=pq[0], own_mask=own)
    ce_i = token_ce(h_out, model.lm_head_weight(), tgt[0], 0)
    lp = jax.nn.log_softmax(h_out @ model.lm_head_weight().T, axis=-1)
    ce_ref = -jnp.take_along_axis(lp, tgt[0][:, None], axis=-1)[:, 0]
    m = w[0].astype(jnp.float32)
    wt = m * (ce_i / (jnp.sum(m * ce_i) / jnp.sum(m)))
    W_man, _ = inner_update(W0, model, h_in, tgt, (wt * (jnp.sum(m) / jnp.sum(wt)))[None], kv, pos, kvd, jnp.float32(0.5),
                            1.0, 0, jnp.float32, "sgd", pos_q=pq, own_mask=own)
    report("surprise: gamma 0 == plain, gamma 1 == the CE-weighted step, token_ce == dense CE",
           maxdiff(W_g0, W_a) == 0.0 and maxdiff(W_g1, W_man) < 1e-5 and maxdiff(W_g1, W_a) > 1e-6
           and bool(jnp.allclose(ce_i, ce_ref, atol=1e-5)))

    # 4. key-value binding: weight 0 == plain; > 0 changes the write; the value head gets a meta-gradient
    W_k0, _ = write_chunk(W0, *common, WriteConfig(kvb_weight=0.0))
    W_k1, st_k = write_chunk(W0, *common, WriteConfig(kvb_weight=1.0))

    def down(m_):
        Wx, _ = write_chunk(W0, m_, *common[1:], WriteConfig(kvb_weight=1.0))
        return jnp.sum(jnp.square(Wx.down.weight))
    g_head = eqx.filter_grad(down)(model).inner_kvb_value.weight
    report("kvb: weight 0 == plain, weight 1 changes the write, the value head gets a meta-gradient",
           maxdiff(W_k0, W_a) == 0.0 and maxdiff(W_k1, W_a) > 1e-6 and float(jnp.abs(g_head).max()) > 0
           and float(st_k.kvb) > 0, f"kvb term {float(st_k.kvb):.3f}")
    # layer-local, as in TTT-KVB: the binding term's inner gradient (weight 1 minus weight 0) equals
    # the per-layer gradient with every layer's key and value held constant (no path through x_l
    # into the earlier layers' W); the meta-gradient still reaches the keys (prime_norm)
    kvs = (kv[0][0], kv[1][0])
    gargs = (model, h_in[0], tgt[0], w[0], kvs, pos[0], kvd, 0, jnp.float32)
    g_on = inner_grad(Wc, *gargs, pos_q=pq[0], own_mask=own, kvb_weight=1.0)[0]
    g_off = inner_grad(Wc, *gargs, pos_q=pq[0], own_mask=own, kvb_weight=0.0)[0]
    g_kvb = jax.tree_util.tree_map(lambda a, b: a - b, g_on, g_off)
    _, _, (xl, _) = model.suffix_forward(h_in[0], pos[0], kvs, Wc, kvd, inner=True, pos_q=pq[0], own_mask=own,
                                         return_prime=True)
    xl = jax.lax.stop_gradient(xl)
    vl = jax.vmap(lambda lin, s_: lin(s_))(model.inner_kvb_value, xl)
    m0 = w[0].astype(jnp.float32)
    ref = lambda Wx: jnp.sum(m0 * jnp.mean(jax.vmap(_bind_err_plain)(Wx, model.prime_post_norm, xl, vl), axis=0)) \
        / (jnp.sum(m0) + 1e-6)
    g_ref = eqx.filter_grad(ref)(Wc)
    scale = max(float(jnp.abs(x).max()) for x in leaves(g_ref))

    # second order through the keys: d/d(prime_norm) of <binding term's inner gradient, R> != 0
    R = [jax.random.normal(jax.random.fold_in(jax.random.PRNGKey(9), i), x.shape) for i, x in enumerate(leaves(g_ref))]

    def proj(m_):
        a_ = inner_grad(Wc, m_, *gargs[1:], pos_q=pq[0], own_mask=own, kvb_weight=1.0)[0]
        b_ = inner_grad(Wc, m_, *gargs[1:], pos_q=pq[0], own_mask=own, kvb_weight=0.0)[0]
        return sum(jnp.sum((x - y) * r) for x, y, r in zip(leaves(a_), leaves(b_), R))
    d_pn = float(jnp.max(jnp.abs(eqx.filter_grad(proj)(model).prime_norm.weight)))
    report("kvb is layer-local (inner gradient == per-layer, keys/values constant); keys get a meta-gradient",
           maxdiff(g_kvb, g_ref) < 1e-4 * scale and scale > 0 and d_pn > 0,
           f"|g_kvb - g_ref| {maxdiff(g_kvb, g_ref):.1e} (scale {scale:.1e}), |dg_prime_norm| {d_pn:.1e}")
    plain_model = Model(dataclasses.replace(CFG, inner_kvb_head=False), jax.random.PRNGKey(0))
    try:
        write_chunk(W0, plain_model, *common[1:], WriteConfig(kvb_weight=1.0)); ok = False
    except ValueError:
        ok = True
    report("kvb refuses a model without inner_kvb_head", ok)

    # 5. two-timescale split: the fast units follow the sub-steps; the slow units take one clipped
    #    step on the accumulated sub-step gradients (== the full-chunk step when the fast units are frozen)
    inter = CFG.intermediate_size
    fast = fast_unit_mask(inter, 0.5)
    n = 4
    wcf = WriteConfig(subchunks=n, fast_frac=0.5)
    W_f, _ = write_chunk(W0, *common, wcf, W0=W0, remat=False)
    W_x = W0                                                  # explicit: fast-masked sub-steps
    for k in range(n):
        s, e = k * C // n, (k + 1) * C // n
        m_k = w[:, :e].astype(jnp.float32) * (jnp.arange(e) >= s)[None]
        W_x, _ = inner_update(W_x, model, h_in[:, :e], tgt[:, :e], m_k, kv, pos[:, :e], kvd[:Wn * C + e],
                              jnp.float32(0.5), 1.0, 0, jnp.float32, "sgd", pos_q=pq[:, :e], own_mask=own[:e, :e],
                              unit_mask=fast)
    d_slow = mask_units(jax.tree_util.tree_map(lambda a, b: a - b, W_f, W0), ~fast)
    slow_norm = float(jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in leaves(d_slow))))
    W_f0, _ = write_chunk(W0, *common, dataclasses.replace(wcf, fast_lr_scale=0.0), W0=W0, remat=False)
    W_slow_only, _ = inner_update(W0, model, h_in, tgt, w, kv, pos, kvd, jnp.float32(0.5), 1.0, 0, jnp.float32, "sgd",
                                  pos_q=pq, own_mask=own, unit_mask=~fast)
    W_fd, _ = write_chunk(W0, *common, dataclasses.replace(wcf, fast_decay=1.0), W0=W0, remat=False)
    nz = lambda t: max(float(jnp.abs(x).max()) for x in leaves(t))
    ok_fast = maxdiff(mask_units(W_f, fast), mask_units(W_x, fast)) < 1e-6
    ok_slow = 0 < slow_norm <= 0.5 * 1.0 + 1e-5 and maxdiff(W_f0, W_slow_only) < 1e-6
    ok_decay = (nz(mask_units(jax.tree_util.tree_map(lambda a, b: a - b, W_fd, W0), fast)) < 1e-6
                and maxdiff(mask_units(W_fd, ~fast), mask_units(W_f, ~fast)) < 1e-6)
    report("fast split: fast units == the fast-masked sub-steps; slow = one clipped step, == the full-chunk "
           "step with the fast units frozen; decay 1 resets the fast units", ok_fast and ok_slow and ok_decay,
           f"fast {ok_fast} slow {ok_slow} (|dW_slow| {slow_norm:.3f}, frozen {maxdiff(W_f0, W_slow_only):.1e}) "
           f"decay {ok_decay}")

    # 6. the configuration checks
    bad = []
    for wc in (WriteConfig(subchunks=3), WriteConfig(fast_frac=0.5), WriteConfig(kvb_value="x"),
               WriteConfig(subchunk_scale=0.0), WriteConfig(steps=0), WriteConfig(steps=2, subchunks=2),
               WriteConfig(norm_alpha=1.0, norm_cap=4.0, steps=2)):
        try:
            wc.check(C); bad.append(wc)
        except ValueError:
            pass
    report("WriteConfig.check refuses bad settings", not bad, str(bad))

    # 7. TRAINING == DECODING, per scheme: loss_fn's chunk-1 loss (shifted reader, unweighted) equals
    #    the decoder's after commit_block(chunk 0) with the same scheme
    T = 2 * C
    ids = jax.random.randint(jax.random.PRNGKey(5), (1, T), 1, 100)
    key = jax.random.PRNGKey(11)
    base = LossConfig(time_reweighting="none", inner_lr=0.5, inner_task="causal", outer_shift=True, fused_ce_block=0,
                      compute_dtype="float32", attn_chunk_batch=2, outer_t=0.6)
    _, key_mask, _ = jax.random.split(key, 3)
    x_t, _, t_mask = q_sample(key_mask, ids, None, CFG.mask_token_id, t=jnp.full((1,), 0.6))
    schemes = {"plain": WriteConfig(), "subchunks 2": WriteConfig(subchunks=2),
               "subchunks 4 scale .25": WriteConfig(subchunks=4, subchunk_scale=0.25),
               "surprise 1": WriteConfig(surprise_gamma=1.0), "steps 2": WriteConfig(steps=2),
               "kvb self": WriteConfig(kvb_weight=0.5),
               "kvb clean": WriteConfig(kvb_weight=0.5, kvb_value="clean"),
               "fast split": WriteConfig(subchunks=2, fast_frac=0.5, fast_decay=0.3),
               "all": WriteConfig(subchunks=2, surprise_gamma=0.5, kvb_weight=0.3, kvb_value="clean", fast_frac=0.25),
               "normalised": WriteConfig(norm_alpha=1.0, norm_cap=4.0)}
    for name, wc in schemes.items():
        _, st = loss_fn(model, Batch(input_ids=ids), key, jnp.int32(5), dataclasses.replace(base, write=wc))
        l_train = float(st["loss_by_chunk"][1])
        state = init_state(model, jnp.float32)
        state = commit_block(model, ids[0, :C], jnp.arange(C), state, jax.random.PRNGKey(0), 0.5, 1.0, 0.5, 0,
                             jnp.float32, inner_task="causal", wc=wc)
        p1 = jnp.arange(C, 2 * C)
        logits, _, _ = block_forward(model, x_t[0, C:], p1, state, jnp.float32, pos_q=p1 + 1)
        tgt1 = jnp.concatenate([ids[0, C + 1:], jnp.zeros((1,), ids.dtype)])
        m1 = jnp.concatenate([t_mask[0, C + 1:], jnp.zeros((1,), bool)]).astype(jnp.float32)
        ce1 = -jnp.take_along_axis(jax.nn.log_softmax(logits, axis=-1), tgt1[:, None], axis=-1)[:, 0]
        l_dec = float(jnp.sum(m1 * ce1) / jnp.sum(m1))
        report(f"training scan == decoder after one committed chunk: {name}", abs(l_train - l_dec) < 2e-5,
               f"{l_train:.6f} vs {l_dec:.6f}")
        jax.clear_caches()          # many distinct programs: free the executables (LLVM code memory on CPU)

    # 8. loss_fn: every scheme finite with finite meta-gradients; inner_lr 0 is scheme-independent
    ids4 = jax.random.randint(jax.random.PRNGKey(7), (2, 4 * C), 1, 100)
    b4 = Batch(input_ids=ids4)
    cfg4 = LossConfig(cart_window=16, inner_lr=0.05, inner_task="causal", outer_shift=True, fused_ce_block=0,
                      compute_dtype="float32", attn_chunk_batch=2)
    l_ref0 = loss_fn(model, b4, key, jnp.int32(5), dataclasses.replace(cfg4, inner_lr=0.0))[0]
    from src.ttt import attention as A
    attend_xla0 = A.attend_xla
    for name, wc in schemes.items():
        c = dataclasses.replace(cfg4, write=wc)
        static = []                                     # per attention call with an own-chunk mask: numpy?

        def spy(*args, **kw):
            om = args[6] if len(args) > 6 else kw.get("own_mask")
            if om is not None:
                static.append(isinstance(om, np.ndarray))
            return attend_xla0(*args, **kw)
        A.attend_xla = spy
        try:
            l, g = eqx.filter_value_and_grad(lambda m_: loss_fn(m_, b4, key, jnp.int32(5), c)[0])(model)
        finally:
            A.attend_xla = attend_xla0
        fin = bool(jnp.isfinite(l)) and all(bool(jnp.all(jnp.isfinite(x))) for x in leaves(g))
        l0 = loss_fn(model, b4, key, jnp.int32(5), dataclasses.replace(c, inner_lr=0.0))[0]
        l_seg = loss_fn(model, b4, key, jnp.int32(5), dataclasses.replace(c, segment_chunks=2))[0]
        report(f"loss_fn {name}: finite loss / gradients, inner_lr 0 == plain, segmented == whole, own masks static",
               fin and abs(float(l0 - l_ref0)) < 1e-6 and abs(float(l_seg - l)) < 1e-5 and bool(static) and all(static),
               f"loss {float(l):.6f} (inner_lr 0: {float(l_ref0):.6f}); static masks {sum(static)}/{len(static)}")
        jax.clear_caches()

    # 9. loss_fn refuses the schemes outside the plain causal write
    for bad_kw in (dict(inner_task="mask"), dict(inner_views=2, inner_task="causal"), dict(write_off=True),
                   dict(mtp=True)):
        c = dataclasses.replace(cfg4, write=WriteConfig(subchunks=2), **bad_kw)
        try:
            loss_fn(model, b4, key, jnp.int32(5), c); ok = False
        except ValueError:
            ok = True
        report(f"loss_fn refuses write schemes with {bad_kw}", ok)

    # 10. evaluation-time write plans: all ones == no plan, all zeros == no write; changing chunk c's
    #     steps leaves the loss of chunks <= c unchanged and changes the later ones; more steps lower
    #     the planned chunk's own CE (the diagnostic), whose value before the first step does not move
    cfgp = dataclasses.replace(cfg4, inner_lr=0.5)
    run = lambda plan, **kw: loss_fn(model, b4, key, jnp.int32(5), dataclasses.replace(cfgp, write_plan=plan, **kw))[1]
    lb = lambda st: np.asarray(st["loss_by_chunk"])
    st0, st1, stz, stn = run(()), run((1, 1, 1, 1)), run((0, 0, 0, 0)), run((), inner_lr=0.0)
    st2, st4, sts = run((1, 2, 1, 1)), run((1, 4, 1, 1)), run((1, 1, 0, 1))
    ok_eq = np.allclose(lb(st1), lb(st0), atol=1e-6) and np.allclose(lb(stz), lb(stn), atol=1e-6)
    ok_c = (np.allclose(lb(st4)[:2], lb(st0)[:2], atol=1e-6) and np.abs(lb(st4)[2:] - lb(st0)[2:]).min() > 1e-6
            and np.allclose(lb(sts)[:3], lb(st0)[:3], atol=1e-6) and abs(lb(sts)[3] - lb(st0)[3]) > 1e-6)
    pre2, post2, pre4, post4 = (float(st2["inner_plan_pre"]), float(st2["inner_plan_post"]),
                                float(st4["inner_plan_pre"]), float(st4["inner_plan_post"]))
    ok_d = abs(pre2 - pre4) < 1e-6 and post4 < post2 < pre2 and "inner_plan_pre" not in st1
    try:
        run((1, 2, 1, 1), write=WriteConfig(subchunks=2)); ok_r = False
    except ValueError:
        ok_r = True
    report("write plan: ones == none, zeros == no write, only later chunks move, own CE falls with steps",
           ok_eq and ok_c and ok_d and ok_r, f"own CE {pre2:.4f} -> {post2:.4f} (2) / {post4:.4f} (4)")

    print("ALL PASS" if not failures else "FAILURES: " + "; ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
