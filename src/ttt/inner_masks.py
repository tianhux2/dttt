# coding=utf-8
"""
Deterministic inner-mask rules computed on the host (numpy) and shipped with the batch.

The write of the fast weight is driven by the tokens the inner stream has to predict.
``random`` (the default, drawn on the device from the step key) hides an i.i.d. subset.
The rules here choose the targets from the tokens themselves, identically in training,
evaluation and decoding:

* ``rare``     - per chunk, the ``target_frac`` least frequent tokens (corpus unigram
                 counts), ties broken by a position hash.
* ``docaware`` - per chunk, tokens whose 3-gram already occurred earlier in the same
                 document (past only: since the last EOS, within what is known so far)
                 rank first, then the rare tercile, then the rest; a ``generic_floor``
                 fraction of the target slots is reserved for hash-chosen generic tokens so
                 the write is never blind to them.

Pad/EOS positions are never targets.  The rules need the unigram table
(``python -m ttt.data unigram``); without it ``rare`` degrades to the position hash.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

MODES = ("random", "rare", "docaware")


def _pos_hash(n: int, seed: int) -> np.ndarray:
    """``[n]`` pseudo-random floats in [0, 1), a pure function of (position, seed)."""
    x = (np.arange(n, dtype=np.uint64) + np.uint64(seed) * np.uint64(0x9E3779B97F4A7C15)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    x ^= x >> np.uint64(33); x *= np.uint64(0xFF51AFD7ED558CCD); x ^= x >> np.uint64(33)
    return (x & np.uint64(0xFFFFFF)).astype(np.float64) / float(1 << 24)


def _trigram_recurs(tokens: np.ndarray, eos: int, prefix: Optional[np.ndarray] = None) -> np.ndarray:
    """``[T]`` bool: the 3-gram ending at each position occurred earlier in the same
    document.  ``prefix`` = tokens known before ``tokens`` (decoding: the committed blocks)."""
    seq = tokens.astype(np.int64) if prefix is None else np.concatenate([prefix.astype(np.int64), tokens.astype(np.int64)])
    n = len(seq); off = 0 if prefix is None else len(prefix)
    h = seq.copy()
    h = h * 1_000_003 + np.concatenate([[-7], seq[:-1]])
    h = h * 1_000_003 + np.concatenate([[-7, -7], seq[:-2]])
    doc = np.cumsum(seq == eos)                                   # document id per position
    key = (h & 0x7FFFFFFFFFFFFFFF) * 4099 + doc                    # same 3-gram in the same document
    order = np.lexsort((np.arange(n), key))
    ks = key[order]
    seen_before = np.zeros(n, bool)
    same = ks[1:] == ks[:-1]
    seen_before[order[1:][same]] = True                           # every occurrence but the first
    seen_before[:2] = False
    return seen_before[off:]


def inner_mask_rule(tokens: np.ndarray, mode: str, chunk: int, eos: int, pad: int,
                    unigram: Optional[np.ndarray] = None, target_frac: float = 0.5,
                    generic_floor: float = 0.1, seed: int = 0,
                    prefix: Optional[np.ndarray] = None) -> np.ndarray:
    """``[T]`` bool mask of the inner targets for one sequence (``T`` a multiple of ``chunk``)."""
    T = len(tokens)
    if mode == "random":
        raise ValueError("random masks are drawn on the device")
    k = int(round(target_frac * chunk))
    hash01 = _pos_hash(T, seed)
    score = hash01 * 0.1                                          # tie-break / mild randomness
    if unigram is not None:
        counts = unigram[np.clip(tokens, 0, len(unigram) - 1)].astype(np.float64)
        rank = np.argsort(np.argsort(counts))                     # 0 = rarest in this sequence
        score += 1.0 - rank / max(T - 1, 1)                       # rarer -> higher
        if mode == "docaware":
            total = unigram.sum(); order = np.argsort(-unigram); before = np.cumsum(unigram[order]) - unigram[order]
            rare_ids = np.zeros(len(unigram), bool); rare_ids[order[before >= 2 * total / 3]] = True
            score += 1.0 * rare_ids[np.clip(tokens, 0, len(unigram) - 1)]
    if mode == "docaware":
        score += 3.0 * _trigram_recurs(tokens, eos, prefix)
    # generic floor: a hash-chosen slice of positions is forced into the target set
    if generic_floor > 0:
        score += 10.0 * (hash01 < generic_floor * target_frac)
    score = np.where((tokens == pad) | (tokens == eos), -np.inf, score)
    mask = np.zeros(T, bool)
    for c in range(T // chunk):
        sl = slice(c * chunk, (c + 1) * chunk)
        idx = np.argsort(-score[sl], kind="stable")[:k]
        idx = idx[np.isfinite(score[sl][idx])]
        mask[c * chunk + idx] = True
    return mask


def split_perm(tokens: np.ndarray, mode: str, chunk: int, eos: int, pad: int, context_frac: float = 0.5,
               unigram: Optional[np.ndarray] = None, generic_floor: float = 0.1, seed: int = 0,
               prefix: Optional[np.ndarray] = None) -> np.ndarray:
    """``[T]`` int32 physical order of the inner stream for the split task: per chunk
    ``[context (natural order) | targets (random order)]`` as absolute positions.  The
    targets are chosen by ``inner_mask_rule`` (``mode`` = ``docaware`` / ``rare``; ``random``
    = a hash-chosen set, XLNet-style partial prediction); their count is ``C - K`` with
    ``K = round(context_frac * C)``; their order comes from a generator seeded by
    ``seed`` and the tokens themselves (deterministic per sequence)."""
    T = len(tokens)
    K = int(round(context_frac * chunk))
    if mode == "random":
        h = _pos_hash(T, seed)
        targets = np.zeros(T, bool)
        for c in range(T // chunk):
            sl = slice(c * chunk, (c + 1) * chunk)
            idx = np.argsort(-h[sl], kind="stable")[: chunk - K]
            targets[c * chunk + idx] = True
    else:
        targets = inner_mask_rule(tokens, mode, chunk, eos, pad, unigram, 1.0 - K / chunk, generic_floor, seed, prefix)
    rng = np.random.default_rng([seed, int(np.bitwise_xor.reduce(tokens.astype(np.int64)[: min(T, 64)]) & 0x7FFFFFFF)])
    perm = np.zeros(T, np.int32)
    for c in range(T // chunk):
        base = c * chunk
        t = np.flatnonzero(targets[base:base + chunk])
        ctx = np.setdiff1d(np.arange(chunk), t)
        # pad/eos-only chunks can leave fewer than C-K targets: fill from the context's tail
        while len(t) < chunk - K:
            t = np.append(t, ctx[-1]); ctx = ctx[:-1]
        perm[base:base + chunk] = base + np.concatenate([ctx, rng.permutation(t)])
    return perm
