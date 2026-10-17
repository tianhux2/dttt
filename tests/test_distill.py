# coding=utf-8
"""Soft-target distillation of the write from a low-noise teacher view, and the
multi-token (x_{i+2}) target (chunk_scan / inner_loop / fused_ce).

    JAX_PLATFORMS=cpu .venv/bin/python tests/test_distill.py
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

from src.ttt.chunk_scan import Batch, LossConfig, loss_fn, shift_slots, soft_targets
from src.ttt.config import ModelConfig
from src.ttt.fused_ce import fused_soft_ce, fused_weighted_ce, naive_soft_ce, topk_logprobs
from src.ttt.inner_loop import InnerExtras, inner_objective, inner_step
from src.ttt.model import Model, key_validity
from src.ttt.objective import q_sample

CFG = ModelConfig(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=8, suffix_len=2,
                  chunk_size=8, window_chunks=1, mask_token_id=120, pad_token_id=0, eos_token_id=0,
                  mtp_head=True)
C, NC, B = CFG.chunk_size, 4, 3
T = C * NC
S = CFG.suffix_len
failures = []


def report(name, ok, detail=""):
    failures.append(name) if not ok else None
    print(f"[{'PASS' if ok else 'FAIL'}] {name:<70} {detail}")


def leaves(g):
    return jax.tree_util.tree_leaves(eqx.filter(g, eqx.is_array))


def finite(l, g):
    return bool(jnp.isfinite(l)) and all(bool(jnp.all(jnp.isfinite(x))) for x in leaves(g))


def rel(g1, g2):
    return max(float(jnp.max(jnp.abs(a - b))) / (float(jnp.max(jnp.abs(a))) + 1e-12) for a, b in zip(leaves(g1), leaves(g2)))


def main():
    # 1. the fused soft CE against the dense reference: value, gradients, second order, one-hot limit
    Tn, H, V, K = 40, 24, 300, 5
    k = jax.random.split(jax.random.PRNGKey(0), 6)
    hid = jax.random.normal(k[0], (Tn, H), jnp.float32)
    lm = jax.random.normal(k[1], (V, H), jnp.float32) * 0.05
    tid = jax.random.randint(k[2], (Tn, K), 0, V)
    q = jax.nn.softmax(jax.random.normal(k[4], (Tn, K)), axis=-1)
    w = jnp.where(jax.random.uniform(k[3], (Tn,)) < 0.5, jax.random.uniform(k[3], (Tn,)), 0.0)
    v = jax.random.normal(k[5], (Tn, H))
    ok, det = True, []
    for block in (64, 128):                                            # 300 = 4 x 64 + 44 (padding), 2 x 128 + 44
        lf, ln = float(fused_soft_ce(hid, lm, tid, q, w, block)), float(naive_soft_ce(hid, lm, tid, q, w))
        gf = jax.grad(lambda h, m: fused_soft_ce(h, m, tid, q, w, block), argnums=(0, 1))(hid, lm)
        gn = jax.grad(lambda h, m: naive_soft_ce(h, m, tid, q, w), argnums=(0, 1))(hid, lm)
        hv_f = jax.jvp(lambda h: jax.grad(lambda h: fused_soft_ce(h, lm, tid, q, w, block))(h), (hid,), (v,))[1]
        hv_n = jax.jvp(lambda h: jax.grad(lambda h: naive_soft_ce(h, lm, tid, q, w))(h), (hid,), (v,))[1]
        d = [abs(lf - ln) / abs(ln)] + [float(jnp.max(jnp.abs(a - b))) / float(jnp.max(jnp.abs(b))) for a, b in zip(gf, gn)] \
            + [float(jnp.max(jnp.abs(hv_f - hv_n))) / float(jnp.max(jnp.abs(hv_n)))]
        ok &= max(d) < 1e-4
        det.append(f"block {block}: {max(d):.1e}")
    l1 = float(fused_soft_ce(hid, lm, tid[:, :1], jnp.ones((Tn, 1)), w, 64))
    l0 = float(fused_weighted_ce(hid, lm, tid[:, 0], w, 64))
    ok &= l1 == l0
    report("fused_soft_ce == dense soft CE (value, grads, HVP); K=1 == one-hot", ok, "; ".join(det))

    # 2. streaming top-k: exact ids and log-probs, with and without a padded last block
    full = jax.nn.log_softmax(hid @ lm.T, axis=-1)
    rv, ri = jax.lax.top_k(full, 7)
    ok = True
    for block in (64, 100, 512):
        ids, lp = topk_logprobs(hid, lm, 7, block)
        ok &= bool(jnp.all(ids == ri)) and float(jnp.max(jnp.abs(lp - rv))) < 1e-5
    g = jax.grad(lambda h: jnp.sum(soft_targets(h[None], lm, 7, 64)[1]) + jnp.sum(soft_targets(h[None], lm, 7, 64)[2]))(hid)
    ok &= bool(jnp.all(g == 0))
    report("topk_logprobs exact (padded blocks too); soft_targets carry no gradient", ok)

    # 3. slot alignment helper
    x = jnp.arange(2 * 5).reshape(2, 5)
    y1, v1 = shift_slots(x, 1); y2, v2 = shift_slots(x, -1); y0, v0 = shift_slots(x, 0)
    ok = (bool(jnp.all(y1[:, :-1] == x[:, 1:])) and bool(jnp.all(v1 == jnp.array([1, 1, 1, 1, 0], bool)))
          and bool(jnp.all(y2[:, 1:] == x[:, :-1])) and bool(jnp.all(v2 == jnp.array([0, 1, 1, 1, 1], bool)))
          and bool(jnp.all(y0 == x)) and bool(jnp.all(v0)))
    report("shift_slots: y[i] = x[i + k], edge flagged invalid", ok)

    # 4. inner_objective: extras None == plain; the soft / MTP terms enter as documented
    model = Model(CFG, jax.random.PRNGKey(0))
    key = jax.random.PRNGKey(7)
    h_in = jax.random.normal(key, (C, CFG.hidden_size))
    tok = jax.random.randint(key, (C,), 1, 100)
    imask = jnp.arange(C) <= C - 2
    kv = (jax.random.normal(jax.random.PRNGKey(3), (S, C, 2, 8)) * 0.1, jax.random.normal(jax.random.PRNGKey(4), (S, C, 2, 8)) * 0.1)
    kvv = key_validity(jnp.int32(1), C, 1)
    p1 = jnp.arange(C, 2 * C)
    Wf = model.fast_weight
    Kd = 4
    sid = jax.random.randint(jax.random.PRNGKey(8), (C, Kd), 1, 100)
    sq = jax.nn.softmax(jax.random.normal(jax.random.PRNGKey(9), (C, Kd)), axis=-1)
    smask = (jnp.arange(C) % 2 == 0).astype(jnp.float32)
    mtp_t = jnp.roll(tok, -1)
    mtp_m = (jnp.arange(C) <= C - 3).astype(jnp.float32)
    args = (Wf, model, h_in, tok, imask, kv, p1, kvv, 0, jnp.float32)
    o_plain, (ce_plain, _, d0, m0, _) = inner_objective(*args)
    o_none, _ = inner_objective(*args, extras=None)
    ex_d = InnerExtras(sid, sq, smask, None, None)
    o_d, (ce_d, _, kl_d, _, _) = inner_objective(*args, extras=ex_d, distill_weight=0.5)
    ex_m = InnerExtras(None, None, None, mtp_t, mtp_m)
    o_m, (ce_m, _, _, mtp_m_v, _) = inner_objective(*args, extras=ex_m, mtp_weight=0.25)
    ex_dm = InnerExtras(sid, sq, smask, mtp_t, mtp_m)
    o_dm, (ce_dm, _, kl_dm, mtp_dm, _) = inner_objective(*args, extras=ex_dm, distill_weight=0.5, mtp_weight=0.25)
    # reference soft CE and MTP CE from the suffix forward
    h_out, _ = model.suffix_forward(h_in, p1, kv, Wf, kvv, inner=True)
    lp = jax.nn.log_softmax(h_out @ model.lm_head_weight().T, axis=-1)
    soft_ref = jnp.sum(smask * -jnp.sum(sq * jnp.take_along_axis(lp, sid, axis=-1), axis=-1)) / (jnp.sum(smask) + 1e-6)
    ent = -jnp.sum(sq * jnp.log(sq), axis=-1)
    kl_ref = soft_ref - jnp.sum(smask * ent) / (jnp.sum(smask) + 1e-6)
    lp2 = jax.nn.log_softmax(model.mtp_head(h_out) @ model.lm_head_weight().T, axis=-1)
    mtp_ref = jnp.sum(mtp_m * -jnp.take_along_axis(lp2, mtp_t[:, None], axis=-1)[:, 0]) / (jnp.sum(mtp_m) + 1e-6)
    ok = (float(o_plain) == float(o_none) == float(ce_plain) and float(d0) == 0.0 and float(m0) == 0.0
          and abs(float(o_d - (ce_d + 0.5 * soft_ref))) < 1e-5 and abs(float(kl_d - kl_ref)) < 1e-5 and float(kl_d) >= -1e-6
          and float(ce_d) == float(ce_plain)
          and abs(float(o_m - (ce_m + 0.25 * mtp_ref))) < 1e-5 and abs(float(mtp_m_v - mtp_ref)) < 1e-5
          and abs(float(o_dm - (ce_dm + 0.5 * soft_ref + 0.25 * mtp_ref))) < 1e-5)
    report("inner_objective: extras off == plain; soft term = weighted soft CE (KL reported); MTP term", ok,
           f"ce {float(ce_plain):.3f} kl {float(kl_d):.3f} mtp {float(mtp_ref):.3f}")

    # 5. mtp_head is an outer parameter of the write: it gets a meta-gradient through the inner step
    def down(m_, ex):
        Wn, _ = inner_step(m_.fast_weight, m_, h_in, tok, imask, kv, p1, kvv, jnp.float32(1.0), 1.0, 0, jnp.float32, extras=ex)
        return jnp.sum(jnp.square(Wn.down.weight))
    g_m = eqx.filter_grad(lambda m: down(m, ex_m))(model).mtp_head.weight
    g_0 = eqx.filter_grad(lambda m: down(m, None))(model).mtp_head.weight
    report("mtp_head gets a meta-gradient through the write (none without the term)",
           float(jnp.abs(g_m).max()) > 0 and float(jnp.abs(g_0).max()) == 0)

    # 6. loss_fn: every student x shift combination runs, finite, the distillation changes the
    #    gradient; distill_weight 0 reproduces the plain run; stats in range
    ids = jax.random.randint(jax.random.PRNGKey(1), (B, T), 1, 100)
    bb = Batch(input_ids=ids)
    base = LossConfig(cart_window=16, inner_lr=0.05, inner_mask_ratio=0.3, fused_ce_block=0, compute_dtype="float32",
                      attn_chunk_batch=2, distill_topk=6, distill_t_range=(0.2, 0.5))
    key_l = jax.random.PRNGKey(2)

    def run(cfg, batch=bb):
        return eqx.filter_value_and_grad(lambda m: loss_fn(m, batch, key_l, jnp.int32(5), cfg)[0])(model)

    for name, kw in (("causal+shift", dict(inner_task="causal", outer_shift=True)),
                     ("causal", dict(inner_task="causal")),
                     ("mask+shift", dict(outer_shift=True)),
                     ("mask", dict())):
        l_ref, g_ref = run(dataclasses.replace(base, **kw))
        cfg_d = dataclasses.replace(base, distill=True, **kw)
        l_d, g_d = run(cfg_d)
        _, st = loss_fn(model, bb, key_l, jnp.int32(5), cfg_d)
        l_0, g_0 = run(dataclasses.replace(cfg_d, distill_weight=0.0))
        kl, mass = float(st["inner_distill"]), float(st["distill_mass"])
        ok = (finite(l_d, g_d) and rel(g_ref, g_d) > 1e-6 and abs(kl) < 10.0 and 0.0 < mass <= 1.0
              and abs(float(l_0 - l_ref)) < 1e-6 and rel(g_ref, g_0) < 1e-5)
        report(f"loss_fn distill {name}: finite, changes the gradient, weight 0 == plain", ok,
               f"loss {float(l_d):.4f} kl {kl:.3f} mass {mass:.2f} |w0-plain| {abs(float(l_0 - l_ref)):.1e}")
        jax.clear_caches()          # many distinct programs: free the executables (LLVM code memory on CPU)

    # 6b. first-order soft term: the same write (loss / stats), a different meta-gradient; the
    #     warm-up ramp: weight 0 at step 0 (== plain), full after distill_warmup_steps
    cfg_d = dataclasses.replace(base, distill=True, inner_task="causal", outer_shift=True)
    l_ref, g_ref = run(dataclasses.replace(base, inner_task="causal", outer_shift=True))
    l_m, g_m = run(cfg_d)
    l_f, g_f = run(dataclasses.replace(cfg_d, distill_meta=False))
    cfg_w = dataclasses.replace(cfg_d, distill_warmup_steps=10)
    l_w0 = loss_fn(model, bb, key_l, jnp.int32(0), cfg_w)[0]
    l_p0 = loss_fn(model, bb, key_l, jnp.int32(0), dataclasses.replace(base, inner_task="causal", outer_shift=True))[0]
    l_w10 = loss_fn(model, bb, key_l, jnp.int32(10), cfg_w)[0]
    l_m10 = loss_fn(model, bb, key_l, jnp.int32(10), cfg_d)[0]
    ok = (abs(float(l_m - l_f)) < 1e-6 and rel(g_m, g_f) > 1e-4 and rel(g_ref, g_f) > 1e-6
          and abs(float(l_w0 - l_p0)) < 1e-6 and abs(float(l_w10 - l_m10)) < 1e-6)
    report("distill_meta false: same write, different meta-gradient; warm-up ramp 0 -> full", ok,
           f"|meta-fo| {rel(g_m, g_f):.2e} |w0-plain| {abs(float(l_w0 - l_p0)):.1e}")
    jax.clear_caches()

    # 7. the teacher's mask draw does not disturb the student's draws: the outer mask and the
    #    per-token stats of the plain run are reproduced with distill on (paired evaluation)
    cfg_p = dataclasses.replace(base, inner_task="causal", outer_shift=True, per_token=True, per_sequence=True)
    _, st_p = loss_fn(model, bb, key_l, jnp.int32(5), cfg_p)
    _, st_d = loss_fn(model, bb, key_l, jnp.int32(5), dataclasses.replace(cfg_p, distill=True, distill_weight=0.0))
    ok = bool(jnp.all(st_p["mask_by_token"] == st_d["mask_by_token"])) and \
        bool(jnp.allclose(st_p["ce_by_token"], st_d["ce_by_token"], atol=1e-5)) and \
        bool(jnp.allclose(st_p["loss_by_seq"], st_d["loss_by_seq"], atol=1e-6))
    report("distill leaves the outer mask and the per-token / per-sequence stats unchanged (weight 0)", ok)
    jax.clear_caches()

    # 8. MTP: the objective adds the outer term; the reported main loss and its mask are the shift
    #    objective's; the second target is scored only where x_{i+2} is masked; the meta-gradient
    #    of the plain parameters is unchanged with both MTP weights 0
    cfg_s = dataclasses.replace(base, inner_task="causal", outer_shift=True)
    cfg_m = dataclasses.replace(cfg_s, mtp=True, mtp_weight=0.3, inner_mtp_weight=0.5)
    l_s, g_s = run(cfg_s)
    l_m, g_m = run(cfg_m)
    _, st_m = loss_fn(model, bb, key_l, jnp.int32(5), cfg_m)
    _, key_mask, _ = jax.random.split(key_l, 3)
    _, _, t_mask = q_sample(key_mask, ids, None, CFG.mask_token_id)
    n2 = float(jnp.sum(t_mask[:, 2:]))
    l_m0, g_m0 = run(dataclasses.replace(cfg_m, mtp_weight=0.0, inner_mtp_weight=0.0))
    checks = [finite(l_m, g_m), abs(float(l_m - (st_m["main_loss"] + 0.3 * st_m["mtp_loss"]))) < 1e-6,
              float(st_m["mtp_mask_count"]) == n2, 0.0 < float(st_m["mtp_loss"]) < 10.0,
              0.0 < float(st_m["inner_mtp"]) < 10.0, abs(float(l_m0 - l_s)) < 1e-6,
              rel(g_s, g_m0) < 1e-5, float(jnp.abs(g_m.mtp_head.weight).max()) > 0]
    ok = all(checks)
    report("loss_fn mtp: loss = main + 0.3 mtp; MTP mask = sum t_mask[2:]; weights 0 == plain shift run", ok,
           f"main {float(st_m['main_loss']):.4f} mtp {float(st_m['mtp_loss']):.4f} inner_mtp {float(st_m['inner_mtp']):.3f}"
           + ("" if ok else f" checks {checks}"))
    jax.clear_caches()

    # 9. distill + MTP together, and the segmented scan (32K program shape) gives the same loss
    cfg_dm = dataclasses.replace(cfg_m, distill=True)
    l_dm, g_dm = run(cfg_dm)
    l_seg, _ = loss_fn(model, bb, key_l, jnp.int32(5), dataclasses.replace(cfg_dm, segment_chunks=2))
    l_fo, g_fo = run(dataclasses.replace(cfg_dm, second_order=False))
    ok = finite(l_dm, g_dm) and abs(float(l_seg - l_dm)) < 1e-5 and finite(l_fo, g_fo)
    report("distill + mtp: finite; segmented == whole; first-order runs", ok, f"loss {float(l_dm):.4f} seg {float(l_seg):.4f}")

    # 10. guards
    plain = Model(dataclasses.replace(CFG, mtp_head=False), jax.random.PRNGKey(0))
    for name, m_, bad in (("mtp without the head", plain, dict(mtp=True, inner_task="causal", outer_shift=True)),
                          ("mtp without shift", model, dict(mtp=True, inner_task="causal")),
                          ("mtp with the mask task", model, dict(mtp=True, outer_shift=True)),
                          ("distill with no_write", model, dict(distill=True, no_write=True)),
                          ("distill with inner_from_outer", model, dict(distill=True, inner_from_outer=True)),
                          ("distill with two views", model, dict(distill=True, inner_views=2))):
        try:
            loss_fn(m_, bb, key_l, jnp.int32(5), dataclasses.replace(base, **bad)); ok = False
        except ValueError:
            ok = True
        report(f"rejects {name}", ok)

    print("ALL PASS" if not failures else "FAILURES: " + "; ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
