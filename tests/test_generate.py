# coding=utf-8
"""
Block decoding with test-time training (``ttt.generate``):
* the sequential all-layer chunk machinery reproduces the chunk-parallel reference
  (``Model.forward``) logits chunk by chunk when the fast weight is frozen;
* ``generate`` keeps the prompt, fills every masked position with in-vocabulary
  tokens, is deterministic for a key, and moves the fast weight when inner_lr > 0.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import jax
import jax.numpy as jnp

from src.ttt.config import ModelConfig
from src.ttt.generate import block_forward, commit_block, generate, init_state
from src.ttt.model import Model

CFG = ModelConfig(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=8, suffix_len=2,
                  chunk_size=16, window_chunks=2, mask_token_id=120, pad_token_id=0, eos_token_id=0)
failures = []


def report(name, ok, detail=""):
    failures.append(name) if not ok else None
    print(f"[{'PASS' if ok else 'FAIL'}] {name:<46} {detail}")


def main():
    model = Model(CFG, jax.random.PRNGKey(0))
    C, nc = CFG.chunk_size, 5
    ids = jax.random.randint(jax.random.PRNGKey(1), (nc * C,), 1, 100)

    # 1. sequential windows == chunk-parallel reference, fast weight frozen (inner_lr = 0)
    ref = model.forward(ids)                                            # [T, V] with W = W_0
    state = init_state(model, jnp.float32)
    worst = 0.0
    for c in range(nc):
        blk, pos = ids[c * C:(c + 1) * C], jnp.arange(c * C, (c + 1) * C)
        logits, _, _ = block_forward(model, blk, pos, state, jnp.float32)
        worst = max(worst, float(jnp.max(jnp.abs(logits - ref[c * C:(c + 1) * C]))) / float(jnp.max(jnp.abs(ref))))
        state = commit_block(model, blk, pos, state, jax.random.PRNGKey(c), 0.0, 1.0, 0.5, 0, jnp.float32)
    report("sequential blocks == chunk-parallel logits", worst < 1e-4, f"max rel = {worst:.2e}")

    # 2. generate: prompt kept, no masks left, in vocabulary, deterministic, TTT moves W
    prompt = ids[:C + 5]
    out = generate(model, prompt, 2 * C - 5, jax.random.PRNGKey(3), steps=4, inner_lr=0.5, fused_ce_block=0)
    out2 = generate(model, prompt, 2 * C - 5, jax.random.PRNGKey(3), steps=4, inner_lr=0.5, fused_ce_block=0)
    report("output shape and prompt preserved", out.shape == (3 * C,) and bool(jnp.all(out[:C + 5] == prompt)),
           f"shape {out.shape}")
    report("no mask tokens, ids in vocabulary",
           bool(jnp.all(out != CFG.mask_token_id)) and bool(jnp.all((out >= 0) & (out < CFG.vocab_size))))
    report("deterministic for a key", bool(jnp.all(out == out2)))
    st = init_state(model, jnp.float32)
    st2 = commit_block(model, ids[:C], jnp.arange(C), st, jax.random.PRNGKey(9), 1.0, 1.0, 0.5, 0, jnp.float32)
    moved = sum(float(jnp.sum(jnp.abs(a - b))) for a, b in zip(jax.tree_util.tree_leaves(st2.W), jax.tree_util.tree_leaves(st.W)))
    report("commit_block updates the fast weight", moved > 0 and int(st2.chunk) == 1, f"|dW|_1 = {moved:.3e}")

    # 3. write options at decode time: the complementary pair and the other update
    #    rules commit, move W, and generate keeps its contract with them.
    st3 = commit_block(model, ids[:C], jnp.arange(C), st, jax.random.PRNGKey(9), 1.0, 1.0, 0.5, 0, jnp.float32,
                       inner_views=2, inner_update_rule="muon2", inner_mask_span=4.0)
    moved3 = sum(float(jnp.sum(jnp.abs(a - b))) for a, b in zip(jax.tree_util.tree_leaves(st3.W), jax.tree_util.tree_leaves(st.W)))
    out3 = generate(model, prompt, 2 * C - 5, jax.random.PRNGKey(3), steps=4, inner_lr=0.5, fused_ce_block=0,
                    inner_views=2, inner_update_rule="sign")
    report("views=2 / muon2 / span commit; generate with views=2 + sign",
           moved3 > 0 and out3.shape == (3 * C,) and bool(jnp.all(out3[:C + 5] == prompt))
           and bool(jnp.all(out3 != CFG.mask_token_id)), f"|dW|_1 = {moved3:.3e}")

    # 4. host-side (document-aware) inner masks at decode time: deterministic, prompt kept.
    import numpy as np
    uni = np.bincount(np.asarray(ids), minlength=CFG.vocab_size).astype(np.int64)
    out4 = generate(model, prompt, 2 * C - 5, jax.random.PRNGKey(3), steps=4, inner_lr=0.5, fused_ce_block=0,
                    inner_mask_mode="docaware", unigram=uni)
    out5 = generate(model, prompt, 2 * C - 5, jax.random.PRNGKey(3), steps=4, inner_lr=0.5, fused_ce_block=0,
                    inner_mask_mode="docaware", unigram=uni)
    report("docaware masks at decode: deterministic, prompt kept",
           bool(jnp.all(out4 == out5)) and bool(jnp.all(out4[:C + 5] == prompt)) and bool(jnp.all(out4 != CFG.mask_token_id)))

    # 5. causal and split inner tasks at decode time: commit runs, moves W, deterministic.
    for task in ("causal", "split"):
        o1 = generate(model, prompt, 2 * C - 5, jax.random.PRNGKey(3), steps=4, inner_lr=0.5, fused_ce_block=0,
                      inner_task=task, inner_mask_mode="random", unigram=uni)
        o2 = generate(model, prompt, 2 * C - 5, jax.random.PRNGKey(3), steps=4, inner_lr=0.5, fused_ce_block=0,
                      inner_task=task, inner_mask_mode="random", unigram=uni)
        report(f"inner_task {task} at decode: deterministic, prompt kept",
               bool(jnp.all(o1 == o2)) and bool(jnp.all(o1[:C + 5] == prompt)) and bool(jnp.all(o1 != CFG.mask_token_id)))

    # sampler options (DecodeOpts): the default is the original loop (a copy of it below, bitwise);
    # threshold / random order / presence penalty / remasking complete the block, keep the prompt,
    # never emit [MASK], and remasking changes the result
    from src.ttt.generate import DecodeOpts, decode_block

    def legacy(ids, known, pos, state, key, steps):
        def body(i, carry):
            ids, known, key = carry
            key, k_sample = jax.random.split(key)
            logits, _, _ = block_forward(model, jnp.where(known, ids, CFG.mask_token_id), pos, state, jnp.float32)
            logits = logits.at[:, CFG.mask_token_id].set(-jnp.inf)
            sampled = jax.random.categorical(k_sample, logits, axis=-1)
            conf = jnp.take_along_axis(jax.nn.log_softmax(logits, axis=-1), sampled[:, None], axis=1)[:, 0]
            n_commit = jnp.ceil(jnp.sum(~known) / (steps - i)).astype(jnp.int32)
            conf = jnp.where(known, -jnp.inf, conf)
            order = jnp.argsort(-conf)
            rank = jnp.empty_like(order).at[order].set(jnp.arange(C))
            commit = (rank < n_commit) & ~known
            return jnp.where(commit, sampled, ids), known | commit, key
        return jax.lax.fori_loop(0, steps, body, (ids, known, key))[0]

    st0 = init_state(model, jnp.float32)
    pos0 = jnp.arange(C)
    known0 = jnp.arange(C) < 5
    ids0 = jnp.where(known0, jax.random.randint(jax.random.PRNGKey(3), (C,), 1, 100), CFG.mask_token_id)
    k = jax.random.PRNGKey(9)
    ref = legacy(ids0, known0, pos0, st0, k, 4)
    new, _ = decode_block(model, ids0, known0, pos0, st0, k, 4, 1.0, jnp.float32)
    report("DecodeOpts default == the original sampler (bitwise)", bool(jnp.all(ref == new)))
    ok = True
    outs = {}
    for name, o in {"threshold": DecodeOpts(threshold=0.05), "random": DecodeOpts(order="random", gumbel_high=True),
                    "presence": DecodeOpts(presence_penalty=2.0), "remask": DecodeOpts(remask_loop=6, remask_k=4),
                    "subblock": DecodeOpts(subblock=C // 2), "margin": DecodeOpts(priority="margin"),
                    "entropy": DecodeOpts(priority="entropy"),
                    "subblock_margin": DecodeOpts(subblock=C // 4, priority="margin")}.items():
        out, _ = decode_block(model, ids0, known0, pos0, st0, k, 4, 1.0, jnp.float32, outer_shift=True,
                              prev_tail_logit=jnp.zeros((CFG.vocab_size,)).at[CFG.mask_token_id].set(50.0), opts=o)
        outs[name] = out
        ok &= bool(jnp.all(out[:5] == ids0[:5])) and bool(jnp.all(out != CFG.mask_token_id))
    base, _ = decode_block(model, ids0, known0, pos0, st0, k, 4, 1.0, jnp.float32, outer_shift=True,
                           prev_tail_logit=jnp.zeros((CFG.vocab_size,)).at[CFG.mask_token_id].set(50.0))
    ok &= bool(jnp.all(base != CFG.mask_token_id)) and not bool(jnp.all(outs["remask"] == base))
    report("sampler options: complete, prompt kept, no [MASK] (also from a [MASK]-peaked tail logit)", ok)

    print("ALL PASS" if not failures else "FAILURES: " + "; ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
