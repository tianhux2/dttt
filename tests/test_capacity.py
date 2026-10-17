# coding=utf-8
"""TTT capacity tools: magnitude pruning of a wider checkpoint's fast weight into a narrower
model (``checkpoint.prune_fast_weight`` / ``load_weights``) and the write-dimension probe
(``inner_update_frac``: the write touches a fixed random fraction of W, norm preserved).

    JAX_PLATFORMS=cpu .venv/bin/python tests/test_capacity.py
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

from src.ttt.checkpoint import prune_fast_weight
from src.ttt.chunk_scan import Batch, LossConfig, loss_fn
from src.ttt.config import ModelConfig
from src.ttt.inner_loop import inner_step
from src.ttt.model import Model, key_validity

CFG = ModelConfig(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=8, suffix_len=2,
                  chunk_size=8, window_chunks=1, mask_token_id=120, pad_token_id=0, eos_token_id=0)
C, S = CFG.chunk_size, CFG.suffix_len
failures = []


def report(name, ok, detail=""):
    failures.append(name) if not ok else None
    print(f"[{'PASS' if ok else 'FAIL'}] {name:<62} {detail}")


def main():
    model = Model(CFG, jax.random.PRNGKey(0))

    # 1. pruning keeps the most important units, in order, with consistent shapes
    fw = model.fast_weight
    g, u, dn = (np.asarray(x) for x in (fw.gate.weight, fw.up.weight, fw.down.weight))
    imp = np.linalg.norm(g, axis=1) * np.linalg.norm(u, axis=1) * np.linalg.norm(dn, axis=2)
    keep = np.sort(np.argsort(-imp, axis=1)[:, :16], axis=1)
    pr = prune_fast_weight(fw, 16)
    ok = (pr.gate.weight.shape == (S, 32, 16) and pr.up.weight.shape == (S, 32, 16) and pr.down.weight.shape == (S, 16, 32)
          and all(np.array_equal(np.asarray(pr.gate.weight)[s], g[s][:, keep[s]]) for s in range(S))
          and all(np.array_equal(np.asarray(pr.down.weight)[s], dn[s][keep[s], :]) for s in range(S))
          and bool(np.all(imp[np.arange(S)[:, None], keep].min(axis=1) >= np.sort(imp, axis=1)[:, -16].min())))
    report("prune_fast_weight: top-16 units by |gate||up||down|, order kept, shapes", ok)
    # the pruned fast weight runs inside a narrow model
    narrow = Model(dataclasses.replace(CFG, ttt_intermediate_size=16), jax.random.PRNGKey(1))
    narrow = eqx.tree_at(lambda m: m.fast_weight, narrow, jax.tree_util.tree_map(jnp.asarray, pr))
    ids = jax.random.randint(jax.random.PRNGKey(1), (2, 4 * C), 1, 100)
    l, st = loss_fn(narrow, Batch(input_ids=ids), jax.random.PRNGKey(2), jnp.int32(5),
                    LossConfig(cart_window=16, inner_lr=0.05, fused_ce_block=0, compute_dtype="float32", attn_chunk_batch=2))
    report("narrow model with the pruned fast weight runs", bool(jnp.isfinite(l)), f"loss {float(l):.4f}")

    # 2. write-dimension probe: frac 1 unchanged; frac 0.25 touches ~25 % of the entries, keeps the norm
    key = jax.random.PRNGKey(7)
    h_in = jax.random.normal(key, (C, CFG.hidden_size)); tok = jax.random.randint(key, (C,), 1, 100)
    imask = jax.random.uniform(jax.random.PRNGKey(2), (C,)) < 0.6
    kv = (jax.random.normal(jax.random.PRNGKey(3), (S, C, 2, 8)) * 0.1, jax.random.normal(jax.random.PRNGKey(4), (S, C, 2, 8)) * 0.1)
    kvv = key_validity(jnp.int32(1), C, 1); pos = jnp.arange(C, 2 * C)
    W = model.fast_weight
    args = (model, h_in, tok, imask, kv, pos, kvv, jnp.float32(1.0), 0.0, 0, jnp.float32)   # clip off
    W1, _ = inner_step(W, *args)
    W1b, _ = inner_step(W, *args, frac=1.0)
    Wq, _ = inner_step(W, *args, frac=0.25)
    d1 = [np.asarray(a - b) for a, b in zip(jax.tree_util.tree_leaves(eqx.filter(W1, eqx.is_array)), jax.tree_util.tree_leaves(eqx.filter(W, eqx.is_array)))]
    dq = [np.asarray(a - b) for a, b in zip(jax.tree_util.tree_leaves(eqx.filter(Wq, eqx.is_array)), jax.tree_util.tree_leaves(eqx.filter(W, eqx.is_array)))]
    same = all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(jax.tree_util.tree_leaves(eqx.filter(W1, eqx.is_array)), jax.tree_util.tree_leaves(eqx.filter(W1b, eqx.is_array))))
    touched = sum(int(np.count_nonzero(x)) for x in dq) / sum(x.size for x in dq)
    n1, nq = np.sqrt(sum(float(np.sum(x ** 2)) for x in d1)), np.sqrt(sum(float(np.sum(x ** 2)) for x in dq))
    ok = same and 0.2 < touched < 0.3 and abs(n1 - nq) / n1 < 1e-4
    report("inner_update_frac: 1.0 bitwise unchanged; 0.25 touches ~25 %, norm preserved", ok,
           f"touched {touched:.3f}, norms {n1:.4f} vs {nq:.4f}")
    # the same entries every call (a fixed subspace)
    Wq2, _ = inner_step(W, model, h_in * 1.5, tok, imask, kv, pos, kvv, jnp.float32(1.0), 0.0, 0, jnp.float32, frac=0.25)
    dq2 = [np.asarray(a - b) for a, b in zip(jax.tree_util.tree_leaves(eqx.filter(Wq2, eqx.is_array)), jax.tree_util.tree_leaves(eqx.filter(W, eqx.is_array)))]
    ok = all(np.array_equal(x != 0, y != 0) for x, y in zip(dq, dq2))
    report("inner_update_frac: the touched entries are a fixed subspace", ok)

    print("ALL PASS" if not failures else "FAILURES: " + "; ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
