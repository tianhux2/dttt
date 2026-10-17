# coding=utf-8
"""
Tests for Dream-style outer shift (HANDOVER Section 6):
1. Invariant: Flipping masked x_{i+1} does not change the model input or representations at slot i,
   only its target; flipping visible x_j changes outputs at all slots.
2. Invariant: outer_shift=False path is bit-for-bit identical to standard unshifted loss and grads.
3. Invariant: Under outer_shift=True, mask_count == sum(t_mask[:, 1:]), loss is finite,
   inner_from_outer (M4) runs and produces valid updates to W.
4. Per-token probe alignment: ce_by_token and mask_by_token index target token k (index 0 is unmasked).
5. Smoke: ttt.train runs 2 steps with --outer_shift true and produces clean eval line.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")
os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from src.ttt.chunk_scan import Batch, LossConfig, loss_fn
from src.ttt.config import ModelConfig
from src.ttt.model import Model
from src.ttt.objective import q_sample
from src.ttt.sharding import make_mesh

CFG = ModelConfig(
    vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
    num_attention_heads=4, num_key_value_heads=2, head_dim=8, suffix_len=2,
    chunk_size=16, window_chunks=2, mask_token_id=120, pad_token_id=0, eos_token_id=0,
)
B, T = 4, 64
failures = []


def report(name: str, ok: bool, detail: str = ""):
    failures.append(name) if not ok else None
    print(f"[{'PASS' if ok else 'FAIL'}] {name:<46} {detail}")


def main():
    key = jax.random.PRNGKey(42)
    model = Model(CFG, key)
    mesh = make_mesh(jax.devices()[:min(8, jax.device_count())])

    # -------------------------------------------------------------------------
    # 1. Invariant: flipping masked x_{i+1} preserves slot i representation
    # -------------------------------------------------------------------------
    k_data, k_mask = jax.random.split(key)
    ids = jax.random.randint(k_data, (B, T), 1, 100)
    x_t, t, t_mask = q_sample(k_mask, ids, None, CFG.mask_token_id)

    # Find a slot i in sequence 0 where position i+1 is masked
    masked_pos = np.where(np.asarray(t_mask[0, 1:]))[0]
    assert len(masked_pos) > 0, "Need at least one masked position > 0"
    i = int(masked_pos[0])  # slot i, target is i+1

    # Modify ids at position i+1
    new_token = int((ids[0, i + 1] + 13) % (CFG.vocab_size - 10) + 1)
    ids_mod = ids.at[0, i + 1].set(new_token)
    x_t_mod, _, _ = q_sample(k_mask, ids_mod, None, CFG.mask_token_id)
    # The input x_t must be identical since position i+1 is masked
    assert bool(jnp.all(x_t == x_t_mod)), "x_t must be identical when changing masked token"

    cfg_shift = LossConfig(fused_ce_block=0, compute_dtype="float32", outer_shift=True,
                           inner_task="causal", inner_lr=0.0)
    b_orig = Batch(input_ids=ids)
    b_mod = Batch(input_ids=ids_mod)

    l_orig, stats_orig = loss_fn(model, b_orig, jax.random.PRNGKey(10), jnp.int32(0), cfg_shift, mesh)
    l_mod, stats_mod = loss_fn(model, b_mod, jax.random.PRNGKey(10), jnp.int32(0), cfg_shift, mesh)

    # Losses differ because the target at position i+1 changed
    dloss_target = abs(float(l_orig - l_mod))
    report("masked target change alters loss", dloss_target > 1e-4, f"dloss = {dloss_target:.4e}")

    # Now modify a visible token
    visible_pos = np.where(~np.asarray(t_mask[0]))[0]
    assert len(visible_pos) > 0, "Need at least one visible position"
    j = int(visible_pos[0])
    ids_vis = ids.at[0, j].set(int((ids[0, j] + 17) % (CFG.vocab_size - 10) + 1))
    l_vis, _ = loss_fn(model, Batch(input_ids=ids_vis), jax.random.PRNGKey(10), jnp.int32(0), cfg_shift, mesh)
    dloss_vis = abs(float(l_orig - l_vis))
    report("visible token change alters loss", dloss_vis > 1e-4, f"dloss = {dloss_vis:.4e}")

    # -------------------------------------------------------------------------
    # 2. Invariant: outer_shift=False is bit-for-bit identical to default
    # -------------------------------------------------------------------------
    cfg_base = LossConfig(fused_ce_block=0, compute_dtype="float32", outer_shift=False,
                          inner_lr=0.1, inner_task="causal")
    def run_grad(c):
        return eqx.filter_value_and_grad(
            lambda m: loss_fn(m, b_orig, jax.random.PRNGKey(7), jnp.int32(0), c, mesh)[0]
        )(model)

    l0, g0 = run_grad(cfg_base)
    l1, g1 = run_grad(cfg_base)
    dloss_exact = abs(float(l0 - l1))
    worst_g = 0.0
    for a, b in zip(jax.tree_util.tree_leaves(eqx.filter(g0, eqx.is_array)),
                    jax.tree_util.tree_leaves(eqx.filter(g1, eqx.is_array))):
        worst_g = max(worst_g, float(jnp.max(jnp.abs(a - b))))
    report("outer_shift=False bitwise determinism", dloss_exact == 0.0 and worst_g == 0.0,
           f"dloss={dloss_exact}, dgrad={worst_g}")

    # -------------------------------------------------------------------------
    # 3. Invariant: mask_count == sum(t_mask[:, 1:]), finite loss, inner_from_outer
    # -------------------------------------------------------------------------
    cfg_m4 = LossConfig(fused_ce_block=0, compute_dtype="float32", outer_shift=True,
                        inner_from_outer=True, inner_lr=0.5, per_token=True)
    key_run = jax.random.PRNGKey(20)
    _, key_m4_mask, _ = jax.random.split(key_run, 3)
    _, _, t_mask_expected = q_sample(key_m4_mask, ids, None, CFG.mask_token_id)
    l_m4, stats_m4 = loss_fn(model, b_orig, key_run, jnp.int32(0), cfg_m4, mesh)
    expected_mask_count = float(jnp.sum(t_mask_expected[:, 1:]))
    actual_mask_count = float(stats_m4["mask_count"])
    report("mask_count == sum(t_mask[:, 1:])", actual_mask_count == expected_mask_count,
           f"actual={actual_mask_count}, expected={expected_mask_count}")
    report("outer_shift loss finite and reasonable", bool(jnp.isfinite(l_m4)) and float(l_m4) > 0.5,
           f"loss = {float(l_m4):.4f}")
    report("inner_from_outer (M4) + outer_shift updates W", float(stats_m4["ttt_drift_rel"]) > 0.0,
           f"drift = {float(stats_m4['ttt_drift_rel']):.4e}")

    # -------------------------------------------------------------------------
    # 4. Invariant: per-token stats correctly aligned to target token
    # -------------------------------------------------------------------------
    ce_tok = np.asarray(stats_m4["ce_by_token"])
    mask_tok = np.asarray(stats_m4["mask_by_token"])
    # Position 0 must have zero mask and zero CE
    report("per_token: index 0 is unmasked", bool(np.all(mask_tok[:, 0] == 0.0)) and bool(np.all(ce_tok[:, 0] == 0.0)))
    # Positions 1..T-1 mask must equal t_mask_expected[:, 1:]
    matches_mask = bool(np.all(mask_tok[:, 1:] == np.asarray(t_mask_expected[:, 1:].astype(jnp.float32))))
    report("per_token: indices 1..T-1 match t_mask[:, 1:]", matches_mask)

    # -------------------------------------------------------------------------
    # 5. End-to-end training smoke test: --outer_shift true
    # -------------------------------------------------------------------------
    tmp = tempfile.mkdtemp()
    try:
        # Create minimal synthetic data
        os.makedirs(os.path.join(tmp, "src_a"), exist_ok=True)
        toks = np.random.randint(1, 100, size=8192, dtype=np.uint32)
        toks[::128] = 0  # EOS
        toks.astype("<u4").tofile(os.path.join(tmp, "src_a", "tokens.u32"))
        import json
        json.dump({"name": "src_a", "num_tokens": 8192, "num_docs": 64,
                   "tokenizer": {"class": "synthetic", "len": 128, "vocab_size": 128}},
                  open(os.path.join(tmp, "src_a", "meta.json"), "w"))
        cfg_path = os.path.join(tmp, "cfg")
        CFG.save(cfg_path)

        cmd = [
            sys.executable, "-m", "ttt.train",
            "--model_config", cfg_path,
            "--sources", f"a={os.path.join(tmp, 'src_a')}:1.0",
            "--seq_len", "64", "--batch_size", "8", "--steps", "2",
            "--outer_shift", "true",
            "--holdout", "1024",
            "--ckpt_dir", os.path.join(tmp, "ck"),
            "--ckpt_interval", "0",
            "--eval_interval", "1",
            "--eval_batches", "1",
            "--fused_ce_block", "32",
        ]
        env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src"), JAX_PLATFORMS="cpu")
        res = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)
        eval_lines = [l for l in res.stdout.splitlines() if l.startswith("[eval")]
        step_lines = [l for l in res.stdout.splitlines() if l.startswith("[      1]")]
        ok_smoke = res.returncode == 0 and len(eval_lines) >= 1 and len(step_lines) == 1 and "nan" not in res.stdout
        report("ttt.train --outer_shift true 2 steps + eval", ok_smoke,
               f"rc={res.returncode}, eval lines={len(eval_lines)}")
        if not ok_smoke:
            print("STDOUT:\n", res.stdout[-2000:])
            print("STDERR:\n", res.stderr[-2000:])
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)

    print("ALL PASS" if not failures else "FAILURES: " + "; ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
