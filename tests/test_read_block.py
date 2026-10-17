# coding=utf-8
"""The sub-block read (``ModelConfig.read_block``) on tiny random models:

1. ``read_block`` = the chunk size is the plain model: the training loss, the decoder's read logits and the committed
   state agree with ``read_block`` 0;
2. the training scan's loss of chunks 0 - 2 equals the decoder reading each chunk with the clean K/V of its own clean
   tokens (``clean_forward``, block-causal) after the previous chunks' commits;
3. no look-ahead: a group's read logits do not move when the clean tokens from the group on, or the noisy tokens
   outside the group, change; a group's clean K/V do not move when later tokens change;
4. ``decode_block`` decodes group by group (prompt kept, no [MASK], deterministic) and ``generate`` runs;
5. the meta-gradient is finite; the j_lens read equals the decoder's (TTT) and runs for gdn; memory_distance and
   context_learning run on a read_block model.

    JAX_PLATFORMS=cpu .venv/bin/python tests/test_read_block.py
"""
from __future__ import annotations

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

from src.ttt.chunk_scan import Batch, LossConfig, loss_fn
from src.ttt.config import ModelConfig
from src.ttt.generate import (DecodeOpts, block_forward, clean_forward, commit_block, decode_block, generate,
                              init_state)
from src.ttt.model import Model
from src.ttt.objective import q_sample

BASE = dict(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
            num_key_value_heads=2, head_dim=8, suffix_len=2, chunk_size=16, window_chunks=2,
            mask_token_id=120, pad_token_id=0, eos_token_id=0)
C, RB = BASE["chunk_size"], 4
F32 = jnp.float32
failures = []


def report(name, ok, detail=""):
    failures.append(name) if not ok else None
    print(f"[{'PASS' if ok else 'FAIL'}] {name:84s} {detail}")


def leaves(t):
    return jax.tree_util.tree_leaves(eqx.filter(t, eqx.is_array))


def maxdiff(a, b):
    return max(float(jnp.max(jnp.abs(x - y))) for x, y in zip(leaves(a), leaves(b)))


def main():
    plain = Model(ModelConfig(**BASE), jax.random.PRNGKey(0))
    whole = Model(ModelConfig(**BASE, read_block=C), jax.random.PRNGKey(0))
    rbm = Model(ModelConfig(**BASE, read_block=RB), jax.random.PRNGKey(0))
    ids = jax.random.randint(jax.random.PRNGKey(5), (2, 4 * C), 1, 100)
    key = jax.random.PRNGKey(11)
    lc = LossConfig(time_reweighting="none", inner_lr=0.5, inner_task="causal", outer_shift=True, fused_ce_block=0,
                    compute_dtype="float32", attn_chunk_batch=2, outer_t=0.6)
    L = lambda m: float(loss_fn(m, Batch(input_ids=ids), key, jnp.int32(5), lc)[0])

    # 1. read_block = C is the plain model
    l0, lC, lb = L(plain), L(whole), L(rbm)
    st0 = init_state(plain, F32)
    blk, pos = ids[0, :C], jnp.arange(C)
    lg0 = block_forward(plain, blk, pos, st0, F32, pos_q=pos + 1)[0]
    lgC = block_forward(whole, blk, pos, st0, F32, pos_q=pos + 1, own_kv=clean_forward(whole, blk, pos, st0, F32)[2])[0]
    s0 = commit_block(plain, blk, pos, st0, jax.random.PRNGKey(1), 0.5, 1.0, 0.5, 0, F32, inner_task="causal")
    sC = commit_block(whole, blk, pos, st0, jax.random.PRNGKey(1), 0.5, 1.0, 0.5, 0, F32, inner_task="causal")
    e_lg, e_st = float(jnp.max(jnp.abs(lg0 - lgC))), maxdiff(s0, sC)
    report("read_block = chunk == read_block 0: training loss, decoder logits, committed state",
           abs(l0 - lC) < 1e-5 and e_lg < 1e-4 and e_st < 1e-5 and abs(lb - l0) > 1e-4,
           f"loss {l0:.6f} / {lC:.6f} (read_block {RB}: {lb:.6f}); logits {e_lg:.1e}; state {e_st:.1e}")

    # 2. the training scan == the decoder's sub-block read, chunk by chunk
    ids1 = ids[:1, :3 * C]
    _, key_mask, _ = jax.random.split(key, 3)
    x_t, _, t_mask = q_sample(key_mask, ids1, None, BASE["mask_token_id"], t=jnp.full((1,), 0.6))
    _, st = loss_fn(rbm, Batch(input_ids=ids1), key, jnp.int32(5), lc)
    scan = [float(x) for x in st["loss_by_chunk"][:3]]

    def dec_loss(c, state):
        p = jnp.arange(c * C, (c + 1) * C)
        own = clean_forward(rbm, ids1[0, c * C:(c + 1) * C], p, state, F32)[2]
        logits = block_forward(rbm, x_t[0, c * C:(c + 1) * C], p, state, F32, pos_q=p + 1, own_kv=own)[0]
        tgt = jnp.concatenate([ids1[0, c * C + 1:(c + 1) * C], ids1[0, (c + 1) * C:(c + 1) * C + 1]])
        m = jnp.concatenate([t_mask[0, c * C + 1:(c + 1) * C], t_mask[0, (c + 1) * C:(c + 1) * C + 1]])
        if tgt.shape[0] < C:
            tgt, m = jnp.concatenate([tgt, jnp.zeros((1,), tgt.dtype)]), jnp.concatenate([m, jnp.zeros((1,), bool)])
        ce = -jnp.take_along_axis(jax.nn.log_softmax(logits, axis=-1), tgt[:, None], axis=-1)[:, 0]
        m = m.astype(F32)
        return float(jnp.sum(m * ce) / jnp.sum(m))
    state, dec = init_state(rbm, F32), []
    for c in range(3):
        dec.append(dec_loss(c, state))
        state = commit_block(rbm, ids1[0, c * C:(c + 1) * C], jnp.arange(c * C, (c + 1) * C), state,
                             jax.random.PRNGKey(0), 0.5, 1.0, 0.5, 0, F32, inner_task="causal")
    e = max(abs(a - b) for a, b in zip(scan, dec))
    report("training scan == decoder's sub-block read (clean K/V of the chunk's own clean tokens)", e < 2e-5,
           f"scan {np.round(scan, 6).tolist()} vs {np.round(dec, 6).tolist()}")

    # 3. no look-ahead
    st1 = commit_block(rbm, ids[0, :C], jnp.arange(C), init_state(rbm, F32), jax.random.PRNGKey(0), 0.5, 1.0, 0.5, 0,
                       F32, inner_task="causal")
    p = jnp.arange(C, 2 * C)
    clean = ids[0, C:2 * C]
    noisy = jnp.where(jnp.arange(C) % 3 == 0, BASE["mask_token_id"], clean)
    other = jax.random.randint(jax.random.PRNGKey(9), (C,), 1, 100)

    def read(cl, nz):
        return block_forward(rbm, nz, p, st1, F32, pos_q=p + 1, own_kv=clean_forward(rbm, cl, p, st1, F32)[2])[0]
    base = read(clean, noisy)
    kv_base = clean_forward(rbm, clean, p, st1, F32)[2]
    worst_r, worst_k = 0.0, 0.0
    for g in range(C // RB):
        grp = (jnp.arange(C) // RB) == g
        later = jnp.arange(C) >= g * RB
        alt = read(jnp.where(later, other, clean), jnp.where(grp, noisy, other))
        worst_r = max(worst_r, float(jnp.max(jnp.abs(alt[grp] - base[grp]))))
        after = jnp.arange(C) >= (g + 1) * RB
        kv_alt = clean_forward(rbm, jnp.where(after, other, clean), p, st1, F32)[2]
        worst_k = max(worst_k, max(float(jnp.max(jnp.abs(a[:, :(g + 1) * RB] - b[:, :(g + 1) * RB])))
                                   for a, b in zip(kv_alt, kv_base)))
    moved = float(jnp.max(jnp.abs(read(clean, noisy) - read(clean, jnp.where(jnp.arange(C) < RB, other, noisy)))))
    report("no look-ahead: group reads ignore later clean / other groups' noisy tokens; clean K/V block-causal",
           worst_r < 1e-6 and worst_k < 1e-6 and moved > 1e-4, f"read {worst_r:.1e}, K/V {worst_k:.1e}; moves {moved:.1e}")

    # 4. decoding group by group
    known0 = jnp.arange(C) < 5
    ids0 = jnp.where(known0, ids[0, C:2 * C], BASE["mask_token_id"])
    tail = jnp.zeros((BASE["vocab_size"],)).at[BASE["mask_token_id"]].set(50.0)
    o1, t1 = decode_block(rbm, ids0, known0, p, st1, jax.random.PRNGKey(4), 8, 1.0, F32, outer_shift=True,
                          prev_tail_logit=tail)
    o2, _ = decode_block(rbm, ids0, known0, p, st1, jax.random.PRNGKey(4), 8, 1.0, F32, outer_shift=True,
                         prev_tail_logit=tail, opts=DecodeOpts(priority="margin"))
    o3, _ = decode_block(rbm, ids0, known0, p, st1, jax.random.PRNGKey(4), 8, 1.0, F32, outer_shift=True,
                         prev_tail_logit=tail)
    ok = (bool(jnp.all(o1[:5] == ids0[:5])) and bool(jnp.all(o1 != BASE["mask_token_id"]))
          and bool(jnp.all(o2 != BASE["mask_token_id"])) and bool(jnp.all(o1 == o3)) and bool(jnp.isfinite(t1).all()))
    try:
        decode_block(rbm, ids0, known0, p, st1, jax.random.PRNGKey(4), 8, 1.0, F32, outer_shift=True,
                     opts=DecodeOpts(remask_loop=2))
        ok = False                                                    # remasking is refused
    except ValueError:
        pass
    out = generate(rbm, ids[0, :C + 5], 2 * C - 5, jax.random.PRNGKey(3), steps=4, inner_lr=0.5, fused_ce_block=0,
                   inner_task="causal", outer_shift=True)
    ok &= out.shape == (3 * C,) and bool(jnp.all(out[:C + 5] == ids[0, :C + 5])) and bool(jnp.all(out != BASE["mask_token_id"]))
    report("decode_block group by group: prompt kept, no [MASK], deterministic; generate runs", ok)

    # 5. gradients; static masks; the j_lens read; the tools
    params, static = eqx.partition(rbm, eqx.is_array)
    g = jax.jit(jax.grad(lambda pr: loss_fn(eqx.combine(pr, static), Batch(input_ids=ids), key, jnp.int32(5), lc)[0]))(params)
    report("meta-gradient finite with read_block", all(bool(jnp.all(jnp.isfinite(x))) for x in leaves(g)))
    # every own-chunk mask reaches attention static (numpy): on the GPU cuDNN builds its masked kernels from
    # it, and a traced one fails there only (the XLA path here takes either)
    from src.ttt import attention as A
    attend_xla0, masks = A.attend_xla, []

    def spy(*args, **kw):
        om = args[6] if len(args) > 6 else kw.get("own_mask")
        if om is not None:
            masks.append((om.shape, isinstance(om, np.ndarray)))
        return attend_xla0(*args, **kw)
    A.attend_xla = spy
    try:
        eqx.filter_value_and_grad(lambda m_: loss_fn(m_, Batch(input_ids=ids), key, jnp.int32(5), lc)[0])(rbm)
        n_scan = len(masks)
        eqx.filter_jit(lambda m_: decode_block(m_, ids0, known0, p, st1, jax.random.PRNGKey(4), 8, 1.0, F32,
                                               outer_shift=True, prev_tail_logit=tail))(rbm)
        commit_block(rbm, ids[0, C:2 * C], p, st1, jax.random.PRNGKey(0), 0.5, 1.0, 0.5, 0, F32, inner_task="causal")
    finally:
        A.attend_xla = attend_xla0
    shapes = {s for s, _ in masks}
    report("own-chunk masks static (numpy) in the training scan, the decoder and the commit",
           bool(masks) and all(ok_ for _, ok_ in masks) and (C, 2 * C) in shapes and (C, C) in shapes,
           f"{sum(ok_ for _, ok_ in masks)}/{len(masks)} static ({n_scan} in the scan), shapes {sorted(shapes)}")

    from tools.j_lens import core
    own = core.clean_own(rbm, clean, p, st1, F32)
    lg_core = core.read_forward(rbm, noisy, p, st1, None, True, F32, own_kv=own).logits
    e = float(jnp.max(jnp.abs(lg_core - base)))
    gdn = Model(ModelConfig(**BASE, read_block=RB, memory="gdn", gdn_heads=2, gdn_head_dim=8), jax.random.PRNGKey(2))
    gs = core.commit(gdn, ids[0, :C], jnp.arange(C), core.init_memory(gdn, F32), jax.random.PRNGKey(0),
                     core.WriteSettings(0.5), F32)
    lg_g = core.read_forward(gdn, noisy, p, gs, None, True, F32, own_kv=core.clean_own(gdn, clean, p, gs, F32)).logits
    lgdn = float(loss_fn(gdn, Batch(input_ids=ids), key, jnp.int32(5), lc)[0])
    report("j_lens read == decoder read (TTT); gdn read and training loss finite", e < 1e-5
           and bool(jnp.isfinite(lg_g).all()) and np.isfinite(lgdn), f"max |diff| {e:.1e}; gdn loss {lgdn:.4f}")

    from tools.j_lens import context_learning as CL
    from tools.j_lens import memory_distance as MD
    seqs = [np.random.default_rng(i).integers(1, 100, 8 * C).astype(np.int32) for i in range(2)]
    ws = core.WriteSettings(0.5, inner_task="causal", fused_ce_block=0)
    o_md, _ = MD.run(rbm, seqs, ws, [4], 8, 2, "alt", 7, F32, log=lambda *a, **k: None)
    o_cl, _ = CL.run(rbm, [(f"s{i}", x) for i, x in enumerate(seqs)], ws, [6], 3, F32, CL.Tok("/nonexistent", 128),
                     tasks=("span",), sizes=dict(span_len=8, span_cue=2), log=lambda *a, **k: None)
    report("memory_distance and context_learning run on a read_block model",
           all(np.isfinite(v).all() for v in o_md.values()) and np.isfinite(o_cl["ce"]).all())

    print("ALL PASS" if not failures else "FAILURES: " + "; ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
