# coding=utf-8
"""
Leak tests of the causal / split inner tasks: the representation that predicts a token
never had access to that token.

The inner stream holds the CLEAN chunk in a physical order (identity for ``causal``,
[context | targets] for ``split``); slot i predicts the token at slot i+1.  Flipping the
token at slot j must leave every output at slots < j bitwise unchanged (they may only
see slots <= i < j), and, for ``split``, every context slot unchanged whatever target is
flipped.  Flipping a token of a later chunk must not change the current chunk at all.
Also: with the full own-chunk pattern and unshifted queries, the inner stream of the
prefix equals the clean stream bitwise.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")
os.environ.setdefault("TTT_ATTENTION", "xla")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import jax
import jax.numpy as jnp
import numpy as np

from src.ttt.attention import own_mask_matrix
from src.ttt.config import ModelConfig
from src.ttt.model import CLEAN, INNER, Model, cast_floating, key_validity, key_validity_all

CFG = ModelConfig(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=8, suffix_len=2,
                  chunk_size=8, window_chunks=1, mask_token_id=120, pad_token_id=0, eos_token_id=0)
C, W, NC = CFG.chunk_size, CFG.window_chunks, 4
T = C * NC
failures = []


def report(name, ok, detail=""):
    failures.append(name) if not ok else None
    print(f"[{'PASS' if ok else 'FAIL'}] {name:<58} {detail}")


def inner_outputs(model, ids, perm, own):
    """Per-slot logits of the inner stream for the whole sequence: prefix (3 streams, the
    inner one permuted with shifted queries) then the suffix chunk by chunk with the clean
    window of forward B (fixed W = W_0)."""
    pos = jnp.arange(T)
    x_in = ids[perm]; p_in = pos[perm]
    # shifted query positions within each chunk (slot i stands at slot i+1's position)
    pq = p_in.reshape(NC, C); pq = jnp.concatenate([pq[:, 1:], pq[:, -1:]], axis=1).reshape(T)
    ids3 = jnp.stack([ids, ids, x_in]); sp = jnp.stack([pos, pos, p_in]); spq = jnp.stack([pos, pos, pq])
    kva = key_validity_all(NC, C, W)
    h3, _ = model.prefix_forward(ids3, pos, kva, 2, jnp.float32, stream_pos=sp, stream_pos_q=spq, inner_own=own)
    Wc = cast_floating(model.fast_weight, jnp.float32)
    own_mat = None if own == "full" else jnp.asarray(own_mask_matrix(C, own))
    S, nkv, d = CFG.suffix_len, CFG.num_key_value_heads, CFG.head_dim
    kv = (jnp.zeros((S, W * C, nkv, d)), jnp.zeros((S, W * C, nkv, d)))
    outs = []
    for c in range(NC):
        sl = slice(c * C, (c + 1) * C)
        kvv = key_validity(jnp.int32(c), C, W)
        _, own_kv = model.suffix_forward(h3[CLEAN, sl], pos[sl], kv, Wc, kvv)                 # B: clean K/V
        h, _ = model.suffix_forward(h3[INNER, sl], p_in[sl], kv, Wc, kvv, inner=True,
                                    pos_q=None if own == "full" else pq[sl], own_mask=own_mat)
        outs.append(model.logits(h))
        kv = tuple(jnp.concatenate([cache[:, C:], o], axis=1) for cache, o in zip(kv, own_kv))
    return jnp.stack(outs)                                                                    # [NC, C, V]


def main():
    model = Model(CFG, jax.random.PRNGKey(0))
    ids = jax.random.randint(jax.random.PRNGKey(1), (T,), 1, 100)
    ident = jnp.arange(T)

    # 0. full pattern, unshifted queries: inner stream == clean stream (sanity of the plumbing)
    pos = jnp.arange(T); kva = key_validity_all(NC, C, W)
    h3, _ = model.prefix_forward(jnp.stack([ids, ids, ids]), pos, kva, 2, jnp.float32,
                                 stream_pos=jnp.stack([pos, pos, pos]), stream_pos_q=None, inner_own="full")
    report("full pattern: inner stream == clean stream (bitwise)", bool(jnp.all(h3[INNER] == h3[CLEAN])))

    rng = np.random.default_rng(0)
    for own, perm in (("causal", ident), ("split:4", None)):
        if perm is None:                                    # [context (4, natural order) | targets (4, shuffled)]
            perm = np.zeros(T, np.int32)
            for c in range(NC):
                tgt = rng.choice(C, size=4, replace=False); ctx = np.setdiff1d(np.arange(C), tgt)
                perm[c * C:(c + 1) * C] = c * C + np.concatenate([np.sort(ctx), rng.permutation(tgt)])
            perm = jnp.asarray(perm)
        K = 0 if own == "causal" else 4
        base = inner_outputs(model, ids, perm, own)
        leak = later_ok = ctx_ok = True
        detail = []
        for c in (1, 2):                                    # a chunk with a window and one after
            for j in (max(K, 1), C - 2, C - 1):             # flip the token at physical slot j of chunk c
                ids2 = ids.at[perm[c * C + j]].set((ids[perm[c * C + j]] + 17) % 100 + 1)
                out2 = inner_outputs(model, ids2, perm, own)
                same_before = bool(jnp.all(out2[c, :j] == base[c, :j]))           # slots < j untouched
                changed_after = bool(jnp.any(out2[c, j:] != base[c, j:]))          # its own slot sees it
                earlier_chunks = bool(jnp.all(out2[:c] == base[:c]))
                leak &= same_before; later_ok &= earlier_chunks
                if own != "causal":
                    ctx_ok &= bool(jnp.all(out2[c, :K] == base[c, :K])) if j >= K else True
                detail.append(f"c{c}j{j}:{'ok' if same_before else 'LEAK'}{'' if changed_after else '?'}")
        report(f"{own}: outputs before a flipped slot bitwise unchanged", leak, " ".join(detail))
        report(f"{own}: earlier chunks unchanged by a later chunk's flip", later_ok)
        if own != "causal":
            report(f"{own}: context slots unchanged when a target is flipped", ctx_ok)
        # the visible side: flipping a CONTEXT / earlier slot must change later predictions
        j0 = 0
        ids3_ = ids.at[perm[1 * C + j0]].set((ids[perm[1 * C + j0]] + 23) % 100 + 1)
        out3 = inner_outputs(model, ids3_, perm, own)
        report(f"{own}: a visible earlier token does change later slots", bool(jnp.any(out3[1, 1:] != base[1, 1:])))

    # loss_fn level: causal / split / outer-reuse run, are finite and change the gradient
    # against the masked task (own process: the equivalence suite's many programs exhaust
    # the CPU JIT's code memory).
    import dataclasses
    import equinox as eqx
    from src.ttt.chunk_scan import Batch, LossConfig, loss_fn
    from src.ttt.inner_masks import split_perm
    B = 4
    ids_b = jax.random.randint(jax.random.PRNGKey(5), (B, T), 1, 100)
    base = LossConfig(cart_window=32, inner_lr=0.005, inner_mask_ratio=0.3, fused_ce_block=0,
                      compute_dtype="float32", attn_chunk_batch=2)
    def run(cfg, batch):
        return eqx.filter_value_and_grad(lambda m: loss_fn(m, batch, jax.random.PRNGKey(2), jnp.int32(5), cfg)[0])(model)
    def rel(g1, g2):
        return max(float(jnp.max(jnp.abs(a - b))) / (float(jnp.max(jnp.abs(a))) + 1e-12)
                   for a, b in zip(jax.tree_util.tree_leaves(eqx.filter(g1, eqx.is_array)),
                                   jax.tree_util.tree_leaves(eqx.filter(g2, eqx.is_array))))
    l_ref, g_ref = run(base, Batch(input_ids=ids_b))
    perm = jnp.asarray(np.stack([split_perm(np.asarray(r), "random", C, CFG.eos_token_id, CFG.pad_token_id, 0.5)
                                 for r in ids_b]))
    for name, kw, bb in (("inner_task=causal", dict(inner_task="causal"), Batch(input_ids=ids_b)),
                         ("inner_task=split", dict(inner_task="split", inner_context_frac=0.5),
                          Batch(input_ids=ids_b, inner_perm=perm)),
                         ("inner_from_outer", dict(inner_from_outer=True), Batch(input_ids=ids_b))):
        l_o, g_o = run(dataclasses.replace(base, **kw), bb)
        r = rel(g_ref, g_o)
        fin = bool(jnp.isfinite(l_o)) and all(bool(jnp.all(jnp.isfinite(x)))
                                              for x in jax.tree_util.tree_leaves(eqx.filter(g_o, eqx.is_array)))
        report(f"loss_fn {name}: finite, differs from the masked task", fin and r > 1e-6,
               f"dgrad = {r:.3e}, loss {float(l_o):.4f}")

    print("ALL PASS" if not failures else "FAILURES: " + "; ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
