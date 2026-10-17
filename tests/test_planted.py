# coding=utf-8
"""Planted recall: the training-stream augmentation (data.Plant / plant_copy / Sampler) and the
held-out probe (probes.planted_probe_batch, loss_fn with a probe loss_mask at outer_t = 1).

    JAX_PLATFORMS=cpu .venv/bin/python tests/test_planted.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")
os.environ.setdefault("TTT_ATTENTION", "xla")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import jax
import jax.numpy as jnp
import numpy as np

from src.ttt.chunk_scan import Batch, LossConfig, loss_fn
from src.ttt.config import ModelConfig
from src.ttt.data import Plant, Sampler, parse_sources, plant_copy, sources_hash
from src.ttt.model import Model
from src.ttt.probes import planted_chunks, planted_probe_batch, recall_train_batch

C, WN, T = 8, 1, 64                      # chunk 8, window 1 chunk, 8 chunks per sequence
V, EOS = 128, 0
failures = []


def report(name, ok, detail=""):
    failures.append(name) if not ok else None
    print(f"[{'PASS' if ok else 'FAIL'}] {name:<78} {detail}")


def synthetic_source(d, n_tokens, seed, eos_rate=0.02):
    os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(seed)
    toks = rng.integers(1, 100, size=n_tokens, dtype=np.uint32)
    toks[rng.random(n_tokens) < eos_rate] = EOS
    toks.astype("<u4").tofile(os.path.join(d, "tokens.u32"))
    json.dump({"name": os.path.basename(d), "num_tokens": int(n_tokens), "num_docs": 1,
               "tokenizer": {"class": "synthetic", "len": V, "vocab_size": V}}, open(os.path.join(d, "meta.json"), "w"))
    return d


def main():
    # 1. plant_copy: the span lands >= window + 1 chunks later, nothing else changes, no EOS spliced
    ok, far = True, True
    for s in range(200):
        rng = np.random.default_rng(s)
        row = np.random.default_rng(1000 + s).integers(1, 100, T).astype(np.int32)
        row[np.random.default_rng(2000 + s).random(T) < 0.05] = EOS
        orig = row.copy()
        r = plant_copy(row, rng, 2, 6, C, WN, eos=EOS)
        if r is None:
            ok &= bool((row == orig).all())
            continue
        a, b, L = r
        changed = np.flatnonzero(row != orig)
        ok &= bool((row[b:b + L] == orig[a:a + L]).all()) and 2 <= L <= 6
        ok &= bool(((changed >= b) & (changed < b + L)).all())
        ok &= not (orig[a:a + L] == EOS).any() and not (orig[b:b + L] == EOS).any()
        far &= (b // C) >= (a // C) + WN + 1
    report("plant_copy: exact copy, only the target changes, target >= window + 1 chunks later, no EOS", ok and far)

    tmp = tempfile.mkdtemp()
    src = synthetic_source(os.path.join(tmp, "a"), 200_000, 1)
    sources = parse_sources(f"a={src}:1.0")
    plant = Plant(0.5, 2, 6, C, WN)
    kw = dict(holdout_tokens=16 * T, eos=EOS)
    s0 = Sampler(sources, T, 8, 7, **kw)
    s1 = Sampler(sources, T, 8, 7, plant=plant, **kw)
    s1b = Sampler(sources, T, 8, 7, plant=plant, **kw)
    ranks = [Sampler(sources, T, 8, 7, rank=r, world=2, plant=plant, **kw) for r in range(2)]

    # 2. Sampler: planting is deterministic, per global sample (ranks agree), changes only planted
    #    spans at about frac of the rows, and enters the data hash only when active
    det = all(np.array_equal(s1.batch(st), s1b.batch(st)) for st in range(5))
    glob = all(np.array_equal(s1.global_batch(st), np.concatenate([r_.batch(st) for r_ in ranks])) for st in range(5))
    n_rows = n_planted = 0
    only_span = True
    for st in range(40):
        x0, x1 = s0.batch(st), s1.batch(st)
        for r0, r1 in zip(x0, x1):
            n_rows += 1
            d = np.flatnonzero(r0 != r1)
            if len(d):
                n_planted += 1
                only_span &= d.max() - d.min() < 6 and (d.min() // C) >= WN + 1
    frac = n_planted / n_rows
    report("Sampler plant: deterministic, ranks == global, only a span changes, ~frac of the rows",
           det and glob and only_span and 0.35 < frac < 0.55, f"planted {frac:.2f} (frac 0.5; identical copies are invisible)")
    h_old = sources_hash(sources, 7, T, 8, 16 * T)
    report("data hash: unchanged without planting, different with it", s0.hash == h_old and s1.hash != s0.hash)
    ev_same = all(np.array_equal(s0.eval_batch(0, k), s1.eval_batch(0, k)) for k in range(2))
    report("held-out batches are never planted", ev_same)

    # 3. the probe batch: span of chunk 1 -> chunk window + 3 (beyond), also -> the chunk before it
    #    (within); cue visible, the rest (block) or every other one (alt) scored; paired draws
    local = s0.eval_batch(0, 0)
    jg = list(range(len(local)))
    L, cue = 6, 2
    cs, ct, cw = planted_chunks(T // C, WN)
    ids_b, lm_b = planted_probe_batch(local, jg, 12345, C, WN, L, cue, "beyond", "block", eos=EOS)
    ids_w, lm_w = planted_probe_batch(local, jg, 12345, C, WN, L, cue, "within", "block", eos=EOS)
    ids_a, lm_a = planted_probe_batch(local, jg, 12345, C, WN, L, cue, "beyond", "alt", eos=EOS)
    ok = (cs, ct, cw) == (1, WN + 3, WN + 2) and np.array_equal(ids_b, ids_a) and np.array_equal(lm_b, lm_w)
    for r in range(len(local)):
        pos = np.flatnonzero(lm_b[r])
        b0 = pos.min() - cue
        span = ids_b[r, b0:b0 + L]
        a_hits = [a for a in range(cs * C, (cs + 1) * C - L + 1) if np.array_equal(local[r, a:a + L], span)]
        ok &= len(pos) == L - cue and pos.max() == b0 + L - 1 and b0 // C == ct and len(a_hits) >= 1
        ok &= int(lm_a[r].sum()) == (L - cue) // 2 and set(np.flatnonzero(lm_a[r])) <= set(pos)
        wd = np.flatnonzero(ids_w[r] != ids_b[r])
        ok &= len(wd) == 0 or (wd.min() // C == cw and wd.max() // C == cw)
        ok &= bool((ids_b[r, :b0] == local[r, :b0]).all() and (ids_b[r, b0 + L:] == local[r, b0 + L:]).all())
    report("probe batch: chunk 1 -> chunk window+3, within copy in window+2, cue visible, masks, paired", ok)
    ids_n, lm_n = planted_probe_batch(local, jg, 12345, C, WN, L, cue, "none", "block", eos=EOS)
    report("probe 'none' control: sequence untouched, same scored positions as 'beyond'",
           np.array_equal(ids_n, local) and np.array_equal(lm_n, lm_b))
    ids_2, _ = planted_probe_batch(local, jg, 12345, C, WN, L, cue, "beyond", "block", eos=EOS)
    report("probe batch deterministic (same draws for every checkpoint)", np.array_equal(ids_b, ids_2))
    try:
        planted_chunks(4, 3); refused = False
    except ValueError:
        refused = True
    report("probe refuses sequences too short for window + 4 chunks", refused)

    # 3b. recall-only training batches: deterministic per (seed, j); one EOS-free copy per row from a
    #     chunk cs into a chunk >= cs + window + 1; nothing else changes; the scored positions are the
    #     copy's tail after the cue - all of it (block) or every other one (alt), about half the rows each
    rr = np.random.default_rng(5).integers(1, 100, (64, T)).astype(np.int32)
    rr[3, :] = EOS                                             # no EOS-free placement: unchanged, unscored
    jr = list(range(100, 164))
    ids_t, lm_t = recall_train_batch(rr, jr, 99, C, WN, 4, 6, 2, eos=EOS)
    ids_t2, lm_t2 = recall_train_batch(rr, jr, 99, C, WN, 4, 6, 2, eos=EOS)
    ok = np.array_equal(ids_t, ids_t2) and np.array_equal(lm_t, lm_t2)
    ok &= np.array_equal(ids_t[3], rr[3]) and not lm_t[3].any()
    n_block = n_rows = 0
    for r in range(len(rr)):
        pos = np.flatnonzero(lm_t[r])
        if r == 3:
            continue
        n_rows += 1
        block = len(pos) > 1 and (np.diff(pos) == 1).all()
        n_block += int(block)
        b = pos.min() - 2 - (0 if block else 1)
        Ls = [pos.max() - b + 1] if block else [pos.max() - b + 1, pos.max() - b + 2]
        found = False
        for L in Ls:
            if not 4 <= L <= 6 or (b + L - 1) // C != b // C:
                continue
            outside = np.ones(T, bool); outside[b:b + L] = False
            srcs = [a for a in range(0, (b // C - WN) * C) if a // C == (a + L - 1) // C
                    and np.array_equal(rr[r, a:a + L], ids_t[r, b:b + L])]
            found |= bool(np.array_equal(ids_t[r, outside], rr[r, outside]) and srcs and not (ids_t[r, b:b + L] == EOS).any())
        ok &= found and (block or (np.diff(pos) == 2).all())
    report("recall training batch: deterministic, one copy beyond the window, only its tail scored (block / alt)",
           ok and 0.3 < n_block / n_rows < 0.7, f"block rows {n_block}/{n_rows}")

    # 4. loss_fn: loss_mask + outer_t 1 masks exactly the probe positions; the write changes their CE
    cfg = ModelConfig(vocab_size=V, hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8, suffix_len=2, chunk_size=C, window_chunks=WN,
                      mask_token_id=V - 1, pad_token_id=EOS, eos_token_id=EOS)
    model = Model(cfg, jax.random.PRNGKey(0))
    lc = LossConfig(inner_lr=0.5, inner_task="causal", outer_shift=True, fused_ce_block=0, compute_dtype="float32",
                    attn_chunk_batch=2, per_token=True, per_sequence=True, time_reweighting="none", cart_window=16)
    bt = Batch(input_ids=jnp.asarray(ids_b), loss_mask=jnp.asarray(lm_b), outer_t=jnp.ones((len(ids_b),), jnp.float32))
    key = jax.random.PRNGKey(3)
    _, st_on = loss_fn(model, bt, key, jnp.int32(10), lc)
    _, st_off = loss_fn(model, bt, key, jnp.int32(10), LossConfig(**{**lc.__dict__, "inner_lr": 0.0}))
    m = np.asarray(st_on["mask_by_token"]) > 0.5
    ce_on, ce_off = np.asarray(st_on["ce_by_token"]), np.asarray(st_off["ce_by_token"])
    ok = np.array_equal(m, lm_b) and np.isfinite(ce_on[m]).all() and np.abs(ce_on[m] - ce_off[m]).max() > 1e-6
    report("loss_fn: exactly the probe positions masked (outer_t 1), write on != off", ok,
           f"scored {int(m.sum())}, mean |on - off| {np.abs(ce_on[m] - ce_off[m]).mean():.2e}")

    print("ALL PASS" if not failures else "FAILURES: " + "; ".join(failures))


if __name__ == "__main__":
    main()
