# coding=utf-8
"""
The stacked-layer scans equal explicit per-layer loops, and the chunk-parallel
prefix equals a sequential chunk-by-chunk formulation with a carried KV window.

(b) is the load-bearing one: it is what lets 18 of 24 layers leave the chunk
recurrence.  If the gathered-window formulation and the sequential carry ever
disagree, everything downstream is wrong in a way no invariance test can see.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")
os.environ.setdefault("TTT_ATTENTION", "xla")   # exact reference path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import equinox as eqx
import jax
import jax.numpy as jnp

from src.ttt.config import ModelConfig
from src.ttt.model import CLEAN, INNER, OUTER, Block, Model, key_validity, key_validity_all, slice_stack

CFG = ModelConfig(
    vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
    num_attention_heads=4, num_key_value_heads=2, head_dim=8, suffix_len=2,
    chunk_size=8, window_chunks=2, mask_token_id=120, pad_token_id=0, eos_token_id=0,
)
T = 64
NC, C, W = T // CFG.chunk_size, CFG.chunk_size, CFG.window_chunks


def _rel(a, b):
    return float(jnp.max(jnp.abs(a - b))) / (float(jnp.max(jnp.abs(b))) + 1e-12)


def _sequential_prefix(m: Model, ids3, pos):
    """Reference: run the prefix chunk by chunk with an explicit shift-window KV,
    exactly as the old scan did - clean stream writes the cache, all three read it."""
    nkv, d = CFG.num_key_value_heads, CFG.head_dim
    h3 = jax.vmap(m.embed_tokens)(ids3).astype(jnp.float32)          # [3, T, H]
    out = jnp.zeros_like(h3)
    for li in range(CFG.prefix_len):
        blk = slice_stack(m.prefix, slice(li, li + 1))
        blk = jax.tree_util.tree_map(lambda x: x[0], blk)             # unstack one layer
        cache = (jnp.zeros((W * C, nkv, d)), jnp.zeros((W * C, nkv, d)))
        cur = h3
        nxt = jnp.zeros_like(h3)
        for c in range(NC):
            sl = slice(c * C, (c + 1) * C)
            kv_valid = key_validity(jnp.int32(c), C, W)
            for s in (OUTER, CLEAN, INNER):
                h_s, kv_s = blk(cur[s, sl], pos[sl], cache, kv_valid)
                nxt = nxt.at[s, sl].set(h_s)
                if s == CLEAN:
                    kv_clean = kv_s
            cache = tuple(jnp.concatenate([cch[C:], own], axis=0) for cch, own in zip(cache, kv_clean))
        h3 = nxt
    return h3


def main():
    failures = []
    m = Model(CFG, jax.random.PRNGKey(0))
    key = jax.random.PRNGKey(1)
    ids = jax.random.randint(key, (T,), 1, 100)
    pos = jnp.arange(T)
    # three genuinely different streams
    k1, k2 = jax.random.split(key)
    ids3 = jnp.stack([
        jnp.where(jax.random.uniform(k1, (T,)) < 0.4, CFG.mask_token_id, ids),
        ids,
        jnp.where(jax.random.uniform(k2, (T,)) < 0.6, CFG.mask_token_id, ids),
    ])
    kva = key_validity_all(NC, C, W)

    # (a) suffix_forward (lax.scan over the stack) == Python loop over per-layer slices.
    h0 = jax.random.normal(jax.random.PRNGKey(3), (C, CFG.hidden_size))
    nkv, d = CFG.num_key_value_heads, CFG.head_dim
    kv = (jax.random.normal(jax.random.PRNGKey(4), (CFG.suffix_len, W * C, nkv, d)) * 0.1,
          jax.random.normal(jax.random.PRNGKey(5), (CFG.suffix_len, W * C, nkv, d)) * 0.1)
    kvv = key_validity(jnp.int32(3), C, W)
    h_scan, kvo_scan = m.suffix_forward(h0, pos[:C], kv, m.fast_weight, kvv)
    h_loop = h0
    kvo_loop = []
    for li in range(CFG.suffix_len):
        one = lambda t: jax.tree_util.tree_map(lambda x: x[li], t)
        blk, pn, ppn, w = one(m.suffix), one(m.prime_norm), one(m.prime_post_norm), one(m.fast_weight)
        h_loop, kvo = blk(h_loop, pos[:C], (kv[0][li], kv[1][li]), kvv, prime=(pn, ppn, w))
        kvo_loop.append(kvo)
    h_loop = m.final_norm(h_loop)
    r = max(_rel(h_scan, h_loop),
            _rel(kvo_scan[0], jnp.stack([k for k, _ in kvo_loop])),
            _rel(kvo_scan[1], jnp.stack([v for _, v in kvo_loop])))
    ok = r < 1e-6
    failures += [] if ok else [f"suffix scan != per-layer loop ({r:.2e})"]
    print(f"[{'PASS' if ok else 'FAIL'}] suffix scan == per-layer loop      max rel = {r:.3e}")

    # (b) THE ONE THAT MATTERS: chunk-parallel prefix == sequential carried-KV prefix,
    #     for all three streams (clean writes the cache, outer/inner read it).
    h_par, kv_last = m.prefix_forward(ids3, pos, kva, attn_chunk_batch=2, compute_dtype=jnp.float32)
    h_seq = _sequential_prefix(m, ids3, pos)
    for s, name in ((CLEAN, "clean"), (OUTER, "outer"), (INNER, "inner")):
        r = _rel(h_par[s], h_seq[s])
        ok = r < 1e-6
        failures += [] if ok else [f"prefix {name} stream: parallel != sequential ({r:.2e})"]
        print(f"[{'PASS' if ok else 'FAIL'}] prefix parallel == sequential ({name:<5})  max rel = {r:.3e}")

    # (b') attn_chunk_batch is a pure memory knob.  lax.map regroups chunks, which
    #      changes XLA fusion and reduction order, so expect f32-epsilon noise, not 0.
    worst = 0.0
    for b in (1, 4, 8):
        worst = max(worst, _rel(m.prefix_forward(ids3, pos, kva, attn_chunk_batch=b, compute_dtype=jnp.float32)[0], h_par))
    ok = worst < 1e-6
    failures += [] if ok else [f"attn_chunk_batch changed the prefix output ({worst:.2e})"]
    print(f"[{'PASS' if ok else 'FAIL'}] attn_chunk_batch 1/4/8 invariant   max rel = {worst:.3e}")

    # (b'') the window binds PER LAYER.  Through depth the receptive field grows by W
    #       chunks per layer (chunk 1's K/V at layer 1 already encodes chunk 0), so the
    #       end-to-end prefix is NOT window-limited - and must not be.  Test one layer:
    #       perturb chunk 0, chunk W+1's output must be exactly unchanged, chunk W's not.
    blk0 = jax.tree_util.tree_map(lambda x: x[0], m.prefix)
    h3f = jax.vmap(m.embed_tokens)(ids3).astype(jnp.float32)
    ids3_p = ids3.at[CLEAN, :C].set(jnp.where(ids3[CLEAN, :C] == 5, 6, 5))
    h3f_p = jax.vmap(m.embed_tokens)(ids3_p).astype(jnp.float32)
    o, _ = blk0.prefix_call(h3f, pos, C, W, kva, 2)
    o_p, _ = blk0.prefix_call(h3f_p, pos, C, W, kva, 2)
    far = float(jnp.max(jnp.abs(o_p[CLEAN, (W + 1) * C:(W + 2) * C] - o[CLEAN, (W + 1) * C:(W + 2) * C])))
    near = float(jnp.max(jnp.abs(o_p[CLEAN, W * C:(W + 1) * C] - o[CLEAN, W * C:(W + 1) * C])))
    ok = far == 0.0 and near > 0.0
    failures += [] if ok else [f"window does not bind per layer: far={far:.2e} near={near:.2e}"]
    print(f"[{'PASS' if ok else 'FAIL'}] window binds per layer (W={W})     beyond={far:.1e} within={near:.1e}")

    # (c) prefix output is independent of the fast weight.
    m2 = eqx.tree_at(lambda mm: mm.fast_weight,
                     m, jax.tree_util.tree_map(lambda x: x + 1.0, m.fast_weight))
    r = float(jnp.max(jnp.abs(m2.prefix_forward(ids3, pos, kva, 2, jnp.float32)[0] - h_par)))
    ok = r == 0.0
    failures += [] if ok else ["prefix depends on the fast weight"]
    print(f"[{'PASS' if ok else 'FAIL'}] prefix independent of W            max abs = {r:.3e}")

    # (d) segments: the prefix over the second half of the sequence, given the first
    #     half's carried K/V windows and global chunk indices, equals the second half
    #     of the whole-sequence prefix (every layer, all three streams).
    nc = ids3.shape[1] // C
    half = nc // 2
    kva_lo = key_validity_all(half, C, W)
    kva_hi = key_validity_all(half, C, W, first=half)
    _, win = m.prefix_forward(ids3[:, :half * C], pos[:half * C], kva_lo, 2, jnp.float32)
    h_hi, _ = m.prefix_forward(ids3[:, half * C:], pos[half * C:], kva_hi, 2, jnp.float32, kv_win=win)
    r = _rel(h_hi, h_par[:, half * C:])
    ok = r < 1e-6
    failures += [] if ok else [f"segmented prefix != whole ({r:.2e})"]
    print(f"[{'PASS' if ok else 'FAIL'}] segmented prefix == whole          max rel = {r:.3e}")
    # ... and the carried windows are the last W chunks' clean K/V of the whole run.
    r = max(_rel(kv_last[0], m.prefix_forward(ids3[:, half * C:], pos[half * C:], kva_hi, 2, jnp.float32, kv_win=win)[1][0]),
            _rel(kv_last[1], m.prefix_forward(ids3[:, half * C:], pos[half * C:], kva_hi, 2, jnp.float32, kv_win=win)[1][1]))
    ok = r < 1e-6
    failures += [] if ok else [f"segmented prefix window != whole ({r:.2e})"]
    print(f"[{'PASS' if ok else 'FAIL'}] segmented prefix window == whole   max rel = {r:.3e}")

    print("ALL PASS" if not failures else "FAILURES: " + "; ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
