# coding=utf-8
"""
Invariants of the training objective: every memory/speed lever is a LAYOUT
change - same loss, same gradients.  Reference = unsharded, remat_levels=1.

Run on CPU (8 forced devices) or on one pinned TPU node:
    JAX_PLATFORMS=cpu .venv/bin/python tests/test_exact_equivalence.py
"""
from __future__ import annotations

import dataclasses
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")
# TPU defaults f32 matmuls to bf16 passes; these compare differently-ordered
# reductions, so the default turns an exact invariant into a ~3e-3 false failure.
os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")
os.environ.setdefault("TTT_ATTENTION", "xla")   # exact reference path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import equinox as eqx
import jax
import jax.numpy as jnp

from src.ttt.chunk_scan import Batch, LossConfig, loss_fn
from src.ttt.config import ModelConfig
from src.ttt.model import Model
from src.ttt.sharding import make_mesh

CFG = ModelConfig(
    vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
    num_attention_heads=4, num_key_value_heads=2, head_dim=8, suffix_len=1,
    chunk_size=8, window_chunks=1, mask_token_id=120, pad_token_id=0, eos_token_id=0,
)
B, T = 8, 64                                                     # 8 chunks
BASE = LossConfig(cart_window=32, inner_lr=0.005, inner_mask_ratio=0.30, fused_ce_block=0,
                  compute_dtype="float32", attn_chunk_batch=2)


def _setup():
    model = Model(CFG, jax.random.PRNGKey(0))
    batch = Batch(input_ids=jax.random.randint(jax.random.PRNGKey(1), (B, T), 1, 100))
    ndev = min(8, jax.device_count())
    mesh = make_mesh(jax.devices()[:ndev])
    return model, batch, mesh


def _run(model, batch, mesh=None, **kw):
    cfg = dataclasses.replace(BASE, **kw)
    def f(m):
        return loss_fn(m, batch, jax.random.PRNGKey(2), jnp.int32(5), cfg, mesh)[0]
    return eqx.filter_value_and_grad(f)(model)


def _max_rel(g1, g2):
    worst = 0.0
    for a, b in zip(jax.tree_util.tree_leaves(eqx.filter(g1, eqx.is_array)),
                    jax.tree_util.tree_leaves(eqx.filter(g2, eqx.is_array))):
        worst = max(worst, float(jnp.max(jnp.abs(a - b))) / (float(jnp.max(jnp.abs(a))) + 1e-12))
    return worst


def main():
    model, batch, mesh = _setup()
    failures = []
    with mesh:
        l_ref, g_ref = _run(model, batch, mesh=None)

        def check(name, tol, **kw):
            l, g = _run(model, batch, **kw)
            r = _max_rel(g_ref, g)
            ok = r < tol
            failures.append(name) if not ok else None
            print(f"[{'PASS' if ok else 'FAIL'}] {name:<36} max rel dgrad = {r:.3e}  (dloss={abs(float(l)-float(l_ref)):.1e})")

        check("carry sharding invariant", 1e-4, mesh=mesh)
        for lv in (2, 3, 4):
            check(f"remat_levels={lv} invariant", 1e-4, mesh=mesh, remat_levels=lv)
        for blk in (32, 48):
            check(f"fused CE block={blk} invariant", 2e-3, mesh=mesh, fused_ce_block=blk)
        for acb in (1, 8):
            check(f"attn_chunk_batch={acb} invariant", 1e-4, mesh=mesh, attn_chunk_batch=acb)
        check("remat_prefix invariant", 1e-4, mesh=mesh, remat_prefix=True)
        check("prefix_remat_levels=2 invariant", 1e-4, mesh=mesh, prefix_remat_levels=2)
        check("prefix levels=2 + remat_prefix", 1e-4, mesh=mesh, prefix_remat_levels=2, remat_prefix=True)
        check("all levers composed", 2e-3, mesh=mesh, remat_levels=3, fused_ce_block=32,
              remat_prefix=True, attn_chunk_batch=8)
        # Segmented scan: 8 chunks as 2 x 4 and 4 x 2 (window 2) is the same computation.
        for sc in (4, 2):
            check(f"segment_chunks={sc} invariant", 1e-4, mesh=mesh, segment_chunks=sc)
        check("segments + remat levers", 2e-3, mesh=mesh, segment_chunks=4, remat_levels=2,
              prefix_remat_levels=2, remat_prefix=True, fused_ce_block=32)

        # Truncating the fast-weight meta-gradient at segment boundaries keeps the
        # loss (a forward-only change) and changes the gradient.
        l_tr, g_tr = _run(model, batch, mesh=mesh, segment_chunks=4, truncate_fast_weight=True)
        r = _max_rel(g_ref, g_tr)
        ok = abs(float(l_tr) - float(l_ref)) < 1e-5 and r > 1e-6
        failures.append("truncate_fast_weight") if not ok else None
        print(f"[{'PASS' if ok else 'FAIL'}] {'truncate_fast_weight':<36} dloss = {abs(float(l_tr)-float(l_ref)):.1e}, dgrad = {r:.3e}")

        # Second order must actually be present: dropping it must change the gradient.
        _, g_fo = _run(model, batch, mesh=mesh, second_order=False)
        so = _max_rel(g_ref, g_fo)
        gn = float(jnp.sqrt(sum(jnp.sum(x**2) for x in jax.tree_util.tree_leaves(eqx.filter(g_ref, eqx.is_array)))))
        ok = so > 1e-6 and gn > 0
        failures.append("second-order alive") if not ok else None
        print(f"[{'PASS' if ok else 'FAIL'}] {'second-order alive':<36} |SO-FO| rel = {so:.3e}  (|grad| = {gn:.4f})")

        # Write options (off by default, so everything above is the production
        # program): each runs, is finite, and changes the gradient.
        for name, kw in (("inner_views=2", dict(inner_views=2)), ("inner_mask_span=4", dict(inner_mask_span=4.0)),
                         ("inner_update=sign", dict(inner_update="sign")), ("inner_update=muon2", dict(inner_update="muon2"))):
            l_o, g_o = _run(model, batch, mesh=mesh, **kw)
            r = _max_rel(g_ref, g_o)
            fin = bool(jnp.isfinite(l_o)) and all(bool(jnp.all(jnp.isfinite(x)))
                                                  for x in jax.tree_util.tree_leaves(eqx.filter(g_o, eqx.is_array)))
            ok = fin and r > 1e-6
            failures.append(f"option {name}") if not ok else None
            print(f"[{'PASS' if ok else 'FAIL'}] option {name:<28} finite, dgrad = {r:.3e}  (loss {float(l_o):.4f})")
        # Learned gate and decay: extra outer parameters that receive gradient.
        mg = Model(dataclasses.replace(CFG, inner_lr_gate=True, inner_decay=0.02), jax.random.PRNGKey(0))
        l_g, g_g = _run(mg, batch, mesh=mesh)
        gn_gate = float(jnp.abs(g_g.inner_gate.weight).max()) + float(jnp.abs(g_g.inner_gate.bias).max())
        gn_dec = float(jnp.abs(g_g.inner_decay_logit).max())
        ok = bool(jnp.isfinite(l_g)) and gn_gate > 0 and gn_dec > 0
        failures.append("gate/decay gradients") if not ok else None
        print(f"[{'PASS' if ok else 'FAIL'}] {'gate + decay receive gradient':<36} |dgate| {gn_gate:.2e} |ddecay| {gn_dec:.2e}")

        # no_write (no inner stream at all) is inner_lr = 0 with fewer FLOPs; per_sequence
        # (evaluation) is a pure re-grouping of the same CE.
        l_nw, g_nw = _run(model, batch, mesh=mesh, no_write=True)
        l_z, g_z = _run(model, batch, mesh=mesh, inner_lr=0.0)
        r = _max_rel(g_z, g_nw)
        ok = abs(float(l_nw) - float(l_z)) < 1e-5 and r < 1e-4
        failures.append("no_write == inner_lr 0") if not ok else None
        print(f"[{'PASS' if ok else 'FAIL'}] {'no_write == inner_lr 0':<36} dloss = {abs(float(l_nw)-float(l_z)):.1e}, dgrad = {r:.3e}")
        l_ps, g_ps = _run(model, batch, mesh=mesh, per_sequence=True)
        r = _max_rel(g_ref, g_ps)
        ok = abs(float(l_ps) - float(l_ref)) < 1e-5 and r < 1e-4
        failures.append("per_sequence invariant") if not ok else None
        print(f"[{'PASS' if ok else 'FAIL'}] {'per_sequence invariant':<36} dloss = {abs(float(l_ps)-float(l_ref)):.1e}, dgrad = {r:.3e}")

        # Host-side inner masks: a Batch with inner_mask runs, is finite, and differs from the
        # random draw; the same mask twice gives the same loss (no device randomness left).
        import numpy as np
        from src.ttt.inner_masks import inner_mask_rule
        ids_np = np.asarray(batch.input_ids)
        hm = np.stack([inner_mask_rule(r, "docaware", CFG.chunk_size, CFG.eos_token_id, CFG.pad_token_id,
                                       np.bincount(ids_np.ravel(), minlength=CFG.vocab_size).astype(np.int64))
                       for r in ids_np])
        bh = Batch(input_ids=batch.input_ids, inner_mask=jnp.asarray(hm))
        l_h1, g_h = _run(model, bh, mesh=mesh)
        l_h2, _ = _run(model, bh, mesh=mesh)
        r = _max_rel(g_ref, g_h)
        ok = bool(jnp.isfinite(l_h1)) and float(l_h1) == float(l_h2) and r > 1e-6 and 0.3 < hm.mean() < 0.7
        failures.append("host-side inner mask") if not ok else None
        print(f"[{'PASS' if ok else 'FAIL'}] {'host-side inner mask (docaware)':<36} loss {float(l_h1):.4f} (rerun identical), dgrad vs random = {r:.3e}, coverage {hm.mean():.2f}")

        # The inner loop must do something: pinning W changes the loss.
        l_frozen, _ = _run(model, batch, mesh=mesh, inner_lr=0.0)
        ok = abs(float(l_frozen) - float(l_ref)) > 1e-7
        failures.append("inner loop alive") if not ok else None
        print(f"[{'PASS' if ok else 'FAIL'}] {'inner loop alive':<36} dloss(lr=0) = {abs(float(l_frozen)-float(l_ref)):.3e}")

    print("ALL PASS" if not failures else "FAILURES: " + "; ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
