# coding=utf-8
"""Feature-prediction inner task: the inner stream predicts (through a learned H x H head)
the RMS-normalised, stop-gradient suffix input of the CLEAN stream at the target slot,
next to / instead of the token CE.

    JAX_PLATFORMS=cpu .venv/bin/python tests/test_inner_feat.py
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

from src.ttt.chunk_scan import Batch, LossConfig, feat_target_index, loss_fn, make_feat_target
from src.ttt.config import ModelConfig
from src.ttt.inner_loop import inner_loss, inner_objective, inner_step
from src.ttt.inner_masks import split_perm
from src.ttt.model import CLEAN, Model, key_validity, key_validity_all
from src.ttt.objective import to_chunks

CFG = ModelConfig(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=8, suffix_len=2,
                  chunk_size=8, window_chunks=1, mask_token_id=120, pad_token_id=0, eos_token_id=0,
                  inner_feat_head=True)
C, W, NC, B = CFG.chunk_size, CFG.window_chunks, 4, 3
T = C * NC
S = CFG.suffix_len
failures = []


def report(name, ok, detail=""):
    failures.append(name) if not ok else None
    print(f"[{'PASS' if ok else 'FAIL'}] {name:<62} {detail}")


def main():
    model = Model(CFG, jax.random.PRNGKey(0))
    ids = jax.random.randint(jax.random.PRNGKey(1), (B, T), 1, 100)

    # 1. target slot index per task
    idx_c = feat_target_index("causal", NC, B, C)
    idx_m = feat_target_index("mask", NC, B, C)
    perm = jnp.asarray(np.stack([split_perm(np.asarray(r), "random", C, CFG.eos_token_id, CFG.pad_token_id, 0.5) for r in ids]))
    perm_c = to_chunks(perm, C)
    idx_s = feat_target_index("split", NC, B, C, perm_c)
    i = jnp.arange(C)
    ok = (bool(jnp.all(idx_c == jnp.minimum(i + 1, C - 1))) and bool(jnp.all(idx_m == i))
          and idx_s.shape == (NC, B, C)
          and bool(jnp.all(idx_s[:, :, :-1] == (perm_c - (jnp.arange(NC) * C)[:, None, None])[:, :, 1:])))
    report("feat_target_index: causal i+1, mask i, split = local perm shifted", ok)

    # 2. the target: RMS 1, stop-gradient, aligned with the clean stream at slot i+1 (causal)
    pos = jnp.arange(T)
    ids3 = jnp.stack([ids[0], ids[0], ids[0]])
    kva = key_validity_all(NC, C, W)
    h3, _ = model.prefix_forward(ids3, pos, kva, 2, jnp.float32, stream_pos=jnp.stack([pos] * 3),
                                 stream_pos_q=jnp.stack([pos] * 3), inner_own="causal")
    hc = to_chunks(h3[CLEAN][None], C)[:, 0]                                              # [NC, C, H]
    tgt = jax.vmap(make_feat_target)(hc, idx_c[:, 0])                                     # [NC, C, H]
    rms = jnp.sqrt(jnp.mean(tgt ** 2, axis=-1))
    ref = h3[CLEAN].reshape(NC, C, -1)
    ref = ref / jnp.sqrt(jnp.mean(ref.astype(jnp.float32) ** 2, axis=-1, keepdims=True) + 1e-6)
    aligned = bool(jnp.allclose(tgt[:, :-1], ref[:, 1:], atol=1e-5))
    g = jax.grad(lambda h: jnp.sum(jax.vmap(make_feat_target)(h, idx_c[:, 0])))(hc)
    report("make_feat_target: RMS 1, stop-gradient, causal slot i+1 of the clean stream",
           bool(jnp.allclose(rms, 1.0, atol=5e-3)) and aligned and bool(jnp.all(g == 0)))

    # 3. inner_objective: off == inner_loss bitwise, feat 0; add == ce + lam * feat, feat ~ 1 at init
    key = jax.random.PRNGKey(7)
    h_in = jax.random.normal(key, (C, CFG.hidden_size))
    tok = jax.random.randint(key, (C,), 1, 100)
    imask = jax.random.uniform(jax.random.PRNGKey(2), (C,)) < 0.6
    kv = (jax.random.normal(jax.random.PRNGKey(3), (S, C, 2, 8)) * 0.1, jax.random.normal(jax.random.PRNGKey(4), (S, C, 2, 8)) * 0.1)
    kvv = key_validity(jnp.int32(1), C, 1)
    p1 = jnp.arange(C, 2 * C)
    ft = tgt[1]
    Wf = model.fast_weight
    l_plain = inner_loss(Wf, model, h_in, tok, imask, kv, p1, kvv, 0, jnp.float32)
    o_off, (ce_off, f_off, _, _, _) = inner_objective(Wf, model, h_in, tok, imask, kv, p1, kvv, 0, jnp.float32, feat_tgt=ft, feat_mode="off")
    o_add, (ce_add, f_add, _, _, _) = inner_objective(Wf, model, h_in, tok, imask, kv, p1, kvv, 0, jnp.float32, feat_tgt=ft, feat_mode="add", feat_weight=0.5)
    o_only, (ce_only, f_only, _, _, _) = inner_objective(Wf, model, h_in, tok, imask, kv, p1, kvv, 0, jnp.float32, feat_tgt=ft, feat_mode="only")
    ok = (float(o_off) == float(l_plain) and float(f_off) == 0.0 and abs(float(o_add - (ce_add + 0.5 * f_add))) < 1e-6
          and 0.5 < float(f_add) < 1.5 and float(ce_add) == float(l_plain) and float(o_only) == float(f_only))
    report("inner_objective: off == inner_loss; add == ce + lam*feat (feat ~ 1 at init); only", ok,
           f"ce {float(ce_add):.4f} feat {float(f_add):.4f}")

    # 4. the head is an outer parameter: it gets a meta-gradient in add mode, none in off mode
    def down(model_, mode):
        Wn, _ = inner_step(model_.fast_weight, model_, h_in, tok, imask, kv, p1, kvv, jnp.float32(1.0), 1.0, 0, jnp.float32,
                           feat_tgt=ft, feat_mode=mode)
        return jnp.sum(jnp.square(Wn.down.weight))
    g_add = eqx.filter_grad(lambda m: down(m, "add"))(model).inner_feat_head.weight
    g_off = eqx.filter_grad(lambda m: down(m, "off"))(model).inner_feat_head.weight
    report("head gets a meta-gradient in add mode, none in off mode",
           float(jnp.abs(g_add).max()) > 0 and float(jnp.abs(g_off).max()) == 0)

    # 5. loss_fn: every task x mode runs, finite, changes the gradient; stats carry inner_feat
    base = LossConfig(cart_window=16, inner_lr=0.05, inner_mask_ratio=0.3, fused_ce_block=0, compute_dtype="float32", attn_chunk_batch=2)
    def run(cfg, batch):
        return eqx.filter_value_and_grad(lambda m: loss_fn(m, batch, jax.random.PRNGKey(2), jnp.int32(5), cfg)[0], has_aux=False)(model)
    def rel(g1, g2):
        return max(float(jnp.max(jnp.abs(a - b))) / (float(jnp.max(jnp.abs(a))) + 1e-12)
                   for a, b in zip(jax.tree_util.tree_leaves(eqx.filter(g1, eqx.is_array)), jax.tree_util.tree_leaves(eqx.filter(g2, eqx.is_array))))
    bb = Batch(input_ids=ids); bs = Batch(input_ids=ids, inner_perm=perm)
    l_ref, g_ref = run(dataclasses.replace(base, inner_task="causal"), bb)
    for name, kw, batch in (("causal+add", dict(inner_task="causal", inner_feat="add"), bb),
                            ("causal+only", dict(inner_task="causal", inner_feat="only"), bb),
                            ("mask+add", dict(inner_feat="add"), bb),
                            ("split+add", dict(inner_task="split", inner_context_frac=0.5, inner_feat="add"), bs),
                            ("causal+add first-order", dict(inner_task="causal", inner_feat="add", second_order=False), bb)):
        l, g = run(dataclasses.replace(base, **kw), batch)
        _, st = loss_fn(model, batch, jax.random.PRNGKey(2), jnp.int32(5), dataclasses.replace(base, **kw))
        fin = bool(jnp.isfinite(l)) and all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree_util.tree_leaves(eqx.filter(g, eqx.is_array)))
        feat = float(st["inner_feat"])
        ok = fin and 0.3 < feat < 1.7 and (name == "mask+add" or rel(g_ref, g) > 1e-6)
        report(f"loss_fn {name}: finite, inner_feat stat, gradient differs", ok, f"loss {float(l):.4f} feat {feat:.3f}")

    # 6. guards
    for bad in (dict(inner_feat="add", no_write=True), dict(inner_feat="add", inner_from_outer=True)):
        try:
            loss_fn(model, bb, jax.random.PRNGKey(2), jnp.int32(5), dataclasses.replace(base, **bad)); ok = False
        except ValueError:
            ok = True
        report(f"rejects {bad}", ok)
    plain = Model(dataclasses.replace(CFG, inner_feat_head=False), jax.random.PRNGKey(0))
    try:
        loss_fn(plain, bb, jax.random.PRNGKey(2), jnp.int32(5), dataclasses.replace(base, inner_feat="add")); ok = False
    except ValueError:
        ok = True
    report("rejects inner_feat without the head", ok)

    print("ALL PASS" if not failures else "FAILURES: " + "; ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
