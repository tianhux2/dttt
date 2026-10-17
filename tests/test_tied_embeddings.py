# coding=utf-8
"""
Tied input/output embeddings: one leaf, correct logits, gradient accumulates both
paths, fused CE agrees on the tied weight, and the lookup's gradient is summed in f32
under the bf16 compute copy.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")
os.environ.setdefault("TTT_ATTENTION", "xla")   # exact reference path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from src.ttt.config import ModelConfig
from src.ttt.fused_ce import fused_weighted_ce
from src.ttt.model import Model, cast_floating

V, H, T = 128, 32, 16
BASE = ModelConfig(
    vocab_size=V, hidden_size=H, intermediate_size=64, num_hidden_layers=4,
    num_attention_heads=4, num_key_value_heads=2, head_dim=8, suffix_len=1,
    chunk_size=8, window_chunks=1, mask_token_id=120, pad_token_id=0, eos_token_id=0,
)


def main():
    failures = []
    ids = jax.random.randint(jax.random.PRNGKey(1), (T,), 1, V)
    tgt = jax.random.randint(jax.random.PRNGKey(3), (T,), 1, V)
    tied = Model(dataclasses.replace(BASE, tie_word_embeddings=True), jax.random.PRNGKey(0))
    untied = Model(dataclasses.replace(BASE, tie_word_embeddings=False), jax.random.PRNGKey(0))

    # 1. Exactly one [V,H] leaf when tied, two when untied.
    n = lambda m: sum(1 for x in jax.tree_util.tree_leaves(eqx.filter(m, eqx.is_array)) if x.shape == (V, H))
    ok = n(tied) == 1 and n(untied) == 2
    failures += [] if ok else [f"leaf count tied={n(tied)} untied={n(untied)}"]
    print(f"[{'PASS' if ok else 'FAIL'}] one [V,H] leaf when tied      tied={n(tied)} untied={n(untied)}")

    # 2. Tied logits are exactly h @ embed.T.
    ok = tied.lm_head is None and tied.lm_head_weight() is tied.embed and untied.lm_head is not None
    failures += [] if ok else ["lm_head_weight() wiring wrong"]
    print(f"[{'PASS' if ok else 'FAIL'}] lm_head_weight() wiring")
    h = jax.random.normal(jax.random.PRNGKey(7), (T, H))
    r = float(jnp.max(jnp.abs(tied.logits(h) - h @ tied.embed.T)))
    ok = r == 0.0
    failures += [] if ok else ["tied logits != h @ embed.T"]
    print(f"[{'PASS' if ok else 'FAIL'}] tied logits == h @ embed.T   max abs = {r:.3e}")

    # 3. THE ONE THAT MATTERS: with one leaf used twice, the gradient must be the SUM
    #    of the lookup-path and unembed-path contributions.  Exact test: give an UNTIED
    #    twin lm_head == embed; its two grad leaves are the two paths.
    def loss_of(m):
        lg = m.forward(ids)
        return -jnp.mean(jax.nn.log_softmax(lg, axis=-1)[jnp.arange(T), tgt])
    twin = eqx.tree_at(lambda m: m.lm_head, untied, tied.embed)
    twin = eqx.tree_at(lambda m: m.embed, twin, tied.embed)
    dl = abs(float(loss_of(twin)) - float(loss_of(tied)))
    g_tied = eqx.filter_grad(loss_of)(tied).embed
    g_twin = eqx.filter_grad(loss_of)(twin)
    g_sum = g_twin.embed + g_twin.lm_head
    r = float(jnp.max(jnp.abs(g_tied - g_sum))) / (float(jnp.max(jnp.abs(g_tied))) + 1e-12)
    ok = dl < 1e-6 and r < 1e-6
    failures += [] if ok else [f"tied grad != lookup + unembed (dloss={dl:.2e}, rel={r:.2e})"]
    print(f"[{'PASS' if ok else 'FAIL'}] tied grad == lookup+unembed  max rel = {r:.3e} (dloss={dl:.2e})")

    # 4. Fused CE agrees with dense on the tied weight, value and gradient.
    w = jnp.ones((T,), jnp.float32)
    hid = lambda m: m.hidden(ids)
    def dense(m):
        lg = m.logits(hid(m)); return -jnp.sum(jax.nn.log_softmax(lg, -1)[jnp.arange(T), tgt] * w)
    def fused(m):
        return fused_weighted_ce(hid(m), m.lm_head_weight(), tgt, w, 32)
    ld, gd = eqx.filter_value_and_grad(dense)(tied)
    lf, gf = eqx.filter_value_and_grad(fused)(tied)
    dl = abs(float(ld) - float(lf)) / (abs(float(ld)) + 1e-8)
    dg = max(float(jnp.max(jnp.abs(a - b))) / (float(jnp.max(jnp.abs(a))) + 1e-8)
             for a, b in zip(jax.tree_util.tree_leaves(eqx.filter(gd, eqx.is_array)),
                             jax.tree_util.tree_leaves(eqx.filter(gf, eqx.is_array))))
    ok = dl < 1e-5 and dg < 2e-3
    failures += [] if ok else [f"fused CE != dense on tied (dloss={dl:.2e} dgrad={dg:.2e})"]
    print(f"[{'PASS' if ok else 'FAIL'}] fused CE == dense (tied)     dloss = {dl:.3e}  dgrad = {dg:.3e}")

    # 5. The lookup's gradient under the bf16 compute copy (train.compute_model casts the
    #    whole model) is summed in f32.  The mask token sits at half of N positions, as in
    #    training, with rows sharing a mean direction: a bf16 sum stops growing after a few
    #    hundred of them (tens of percent off); f32 leaves one bf16 rounding of the result.
    N = 8192
    ids_n = jnp.where(jax.random.bernoulli(jax.random.PRNGKey(11), 0.5, (N,)), BASE.mask_token_id,
                      jax.random.randint(jax.random.PRNGKey(12), (N,), 1, V))
    cot = (1.0 + jax.random.normal(jax.random.PRNGKey(13), (N, H))) / N
    params, static = eqx.partition(tied, eqx.is_array)

    def lookup(p):
        m = cast_floating(eqx.combine(p, static), jnp.bfloat16)
        return jnp.sum(m.embed_tokens(ids_n).astype(jnp.float32) * cot)
    g = np.asarray(jax.grad(lookup)(params).embed, np.float64)
    exact = np.zeros((V, H))
    np.add.at(exact, np.asarray(ids_n), np.asarray(cot, np.float64))
    k = BASE.mask_token_id
    r = float(np.linalg.norm(g[k] - exact[k]) / np.linalg.norm(exact[k]))
    ok = r < 1e-2
    failures += [] if ok else [f"lookup gradient not summed in f32 (mask row rel err {r:.2e})"]
    print(f"[{'PASS' if ok else 'FAIL'}] lookup grad summed in f32    mask row x{int((ids_n == k).sum())}: rel err = {r:.3e}")

    print("ALL PASS" if not failures else "FAILURES: " + "; ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
