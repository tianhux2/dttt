# coding=utf-8
"""The 2-D mesh (``--fsdp_size``: data parallel x FSDP, the 8-GPU runs) trains like the 1-D mesh:
one optimiser step with 2 micro-batches on 8 devices as {'fsdp': 8} and as {'dp': 2, 'fsdp': 4}
gives the same loss and the same updated parameters (up to the order of the reductions).

    JAX_PLATFORMS=cpu .venv/bin/python tests/test_mesh_2d.py
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")
os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")
os.environ.setdefault("TTT_ATTENTION", "xla")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from src.ttt.chunk_scan import Batch, LossConfig
from src.ttt.config import ModelConfig
from src.ttt.model import Model
from src.ttt.sharding import make_mesh, param_shardings
from src.ttt.train import make_train_step

CFG = ModelConfig(
    vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
    num_attention_heads=4, num_key_value_heads=2, head_dim=8, suffix_len=1,
    chunk_size=8, window_chunks=1, mask_token_id=120, pad_token_id=0, eos_token_id=0,
)
LCFG = LossConfig(cart_window=16, inner_lr=0.005, fused_ce_block=0, attn_chunk_batch=2,
                  inner_task="causal", outer_shift=True, compute_dtype="float32")
N, B, T = 2, 8, 64


def main():
    failures = []

    def report(name, ok, detail=""):
        failures.extend([] if ok else [name])
        print(f"[{'PASS' if ok else 'FAIL'}] {name:64s} {detail}")

    if jax.device_count() < 8:
        print("SKIP: needs 8 devices"); return 0
    model = Model(CFG, jax.random.PRNGKey(0))
    params0, static = eqx.partition(model, eqx.is_array)
    ids = jax.random.randint(jax.random.PRNGKey(1), (N, B, T), 1, 100)
    key = jax.random.PRNGKey(3)
    opt = optax.sgd(1.0)
    res = {}
    for name, fsdp in (("1-D", 0), ("2-D", 4)):
        mesh = make_mesh(jax.devices()[:8], fsdp)
        jax.set_mesh(mesh)
        params = jax.device_put(params0, param_shardings(params0, mesh))
        new, _, m = make_train_step(opt, LCFG, mesh, "suffix,embed", grad_accum=N)(
            static, params, opt.init(params), Batch(input_ids=jnp.array(ids, copy=True)), key, jnp.int32(5))
        res[name] = (float(m["loss"]), [np.asarray(x, np.float64) for x in jax.tree_util.tree_leaves(new)],
                     dict(mesh.shape))
    (l1, p1, s1), (l2, p2, s2) = res["1-D"], res["2-D"]
    report(f"mesh shapes {s1} and {s2}", s1 == {"fsdp": 8} and s2 == {"dp": 2, "fsdp": 4})
    report("same loss", abs(l1 - l2) <= 1e-6 * abs(l1), f"{l1:.7f} vs {l2:.7f}")
    d = max(float(np.max(np.abs(a - b))) / max(float(np.max(np.abs(a))), 1e-12) for a, b in zip(p1, p2))
    report("same updated parameters", d < 1e-5, f"max rel diff {d:.1e}")
    print("ALL PASS" if not failures else "FAILURES: " + "; ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
