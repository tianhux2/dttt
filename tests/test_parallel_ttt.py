# coding=utf-8
"""
The parallel-MLP fast weight: carry shape, placement on the top suffix_len layers,
no-op when absent, three-residual composition, every matrix gets gradient, and
the prime norms are outer params rather than carry.
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
from src.ttt.model import FastWeight, Model, key_validity

L, S, T, H, I = 4, 2, 16, 32, 64
CFG = ModelConfig(
    vocab_size=128, hidden_size=H, intermediate_size=I, num_hidden_layers=L,
    num_attention_heads=4, num_key_value_heads=2, head_dim=8, suffix_len=S,
    chunk_size=8, window_chunks=1, mask_token_id=120, pad_token_id=0, eos_token_id=0,
)


def main():
    failures = []
    m = Model(CFG, jax.random.PRNGKey(0))
    ids = jax.random.randint(jax.random.PRNGKey(2), (T,), 1, 100)
    W = m.fast_weight

    # 1. Carry: 3 x suffix_len leaves, gate/up [H, I], down [I, H], stacked on axis 0.
    leaves = jax.tree_util.tree_leaves(eqx.filter(W, eqx.is_array))
    shapes = sorted({x.shape for x in leaves})
    ok = isinstance(W, FastWeight) and len(leaves) == 3 and shapes == [(S, H, I), (S, I, H)]
    failures += [] if ok else [f"carry wrong: {len(leaves)} leaves, {shapes}"]
    print(f"[{'PASS' if ok else 'FAIL'}] carry = FastWeight[S] 3 leaves    {shapes}")

    # 2. Which layers carry it: prefix stack has L-S layers, suffix has S.
    np_, ns = m.prefix.attn.q.weight.shape[0], m.suffix.attn.q.weight.shape[0]
    ok = (np_, ns) == (L - S, S) and m.prime_norm.weight.shape == (S, H)
    failures += [] if ok else [f"stacks are {np_}/{ns}, want {L-S}/{S}"]
    print(f"[{'PASS' if ok else 'FAIL'}] prefix[{np_}] + suffix[{ns}] = {L} layers")

    # 3. The per-layer forward with prime=None is bit-identical to a plain block.
    blk = jax.tree_util.tree_map(lambda x: x[0], m.suffix)
    h = jax.random.normal(jax.random.PRNGKey(5), (8, H))
    pos = jnp.arange(8)
    kv0 = (jnp.zeros((8, 2, 8)), jnp.zeros((8, 2, 8)))
    kvv = key_validity(jnp.int32(0), 8, 1)
    a, _ = blk(h, pos, kv0, kvv)
    b, _ = blk(h, pos, kv0, kvv, prime=None)
    r = float(jnp.max(jnp.abs(a - b)))
    ok = r == 0.0
    failures += [] if ok else ["prime=None is not a no-op"]
    print(f"[{'PASS' if ok else 'FAIL'}] prime=None is a no-op            max abs = {r:.3e}")

    # 4. Three-residual composition matches a hand-written reference:
    #    h += attn(ln1(h)); h += post_norm(W(pre_norm(h))); h += mlp(ln2(h)).
    one = lambda t: jax.tree_util.tree_map(lambda x: x[0], t)
    pn, ppn, w = one(m.prime_norm), one(m.prime_post_norm), one(W)
    out, _ = blk(h, pos, kv0, kvv, prime=(pn, ppn, w))
    q, k, v = blk.attn.qkv(blk.attn_norm(h), pos)
    K, V = jnp.concatenate([kv0[0], k]), jnp.concatenate([kv0[1], v])
    ref = h + blk.attn.out(blk.attn.attend(q[None], K[None], V[None], kvv[None])[0])
    ref = ref + ppn(w(pn(ref)))
    ref = ref + blk.mlp(blk.mlp_norm(ref))
    r = float(jnp.max(jnp.abs(out - ref))) / (float(jnp.max(jnp.abs(ref))) + 1e-12)
    ok = r < 1e-6
    failures += [] if ok else ["forward != three-residual reference"]
    print(f"[{'PASS' if ok else 'FAIL'}] 3-residual composition           max rel = {r:.3e}")

    # 5. THE ONE THAT MATTERS: every one of the three matrices gets a real gradient
    #    (a zero-initialised down_proj would make dL/dgate = dL/dup = 0 forever).
    def f(w):
        return jnp.sum(m.forward(ids, W=w) ** 2)
    g = eqx.filter_grad(f)(W)
    mags = [float(jnp.max(jnp.abs(x))) for x in jax.tree_util.tree_leaves(eqx.filter(g, eqx.is_array))]
    ok = len(mags) == 3 and all(v > 0 for v in mags)
    failures += [] if ok else [f"zero gradient on some fast-weight matrix: {mags}"]
    print(f"[{'PASS' if ok else 'FAIL'}] all 3 matrices get gradient      min |g| = {min(mags):.3e}")

    # 6. The prime norms are outer params, not part of the FastWeight carry type.
    ok = not any(isinstance(x, type(m.prime_norm)) for x in jax.tree_util.tree_leaves(W, is_leaf=lambda x: isinstance(x, eqx.Module)))
    ok = ok and m.prime_norm.weight.shape == (S, H) and m.prime_post_norm.weight.shape == (S, H)
    failures += [] if ok else ["prime norms leaked into the carry"]
    print(f"[{'PASS' if ok else 'FAIL'}] prime norms are outer params")

    print("ALL PASS" if not failures else "FAILURES: " + "; ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
