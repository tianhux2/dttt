# coding=utf-8
"""
Evaluation probes: what kind of token does the fast weight help?

Given the per-token cross-entropy of the write-on and write-off twins on held-out
sequences, label every scored token on the host and aggregate the write gain
(CE_off - CE_on) per class:

* P1 recurrence - has the token's 4-gram (the token and its three predecessors)
  occurred earlier in the SAME document?  ``beyond``: yes, but every earlier
  occurrence lies before the attention window of the token's chunk (only the fast
  weight can know it); ``within``: an occurrence lies inside the window (attention
  can find it); ``novel``: never before.
* P2 rarity - the token's unigram-frequency bin (terciles by token mass over the
  training streams: ``frequent`` / ``mid`` / ``rare``).
* P3 neighbours - are the token's left / right neighbours visible (v) or masked (m) in the
  outer stream?  The causal write is keyed on the clean previous token, which the shifted
  read-out slot holds only when that token is visible; ``Lv_Rm`` and ``Lm_Rv`` have the
  same noise-level distribution (share ~ t(1-t)), so their gains compare the two directions.

Everything here is numpy on ``[T]`` token arrays; the document bounds come from the
stream around the sequence (documents can start long before it, so up to
``lookback`` tokens before the sequence are hashed too).
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

P1_CLASSES = ("beyond", "within", "novel", "pre_seq")
P2_CLASSES = ("frequent", "mid", "rare")
P3_CLASSES = ("Lv_Rv", "Lv_Rm", "Lm_Rv", "Lm_Rm")


def rarity_bins(counts: np.ndarray) -> np.ndarray:
    """``[vocab]`` int8 bin per token id: 0 frequent, 1 mid, 2 rare - terciles of the
    TOKEN MASS (each bin holds ~1/3 of all occurrences), unseen ids are rare."""
    order = np.argsort(-counts.astype(np.int64), kind="stable")
    sorted_counts = counts[order].astype(np.float64)
    before = np.cumsum(sorted_counts) - sorted_counts               # mass of the more frequent ids
    total = sorted_counts.sum() if sorted_counts.sum() > 0 else 1.0
    bins = np.where(before < total / 3, 0, np.where(before < 2 * total / 3, 1, 2)).astype(np.int8)
    out = np.empty(len(counts), np.int8)
    out[order] = bins
    out[counts == 0] = 2
    return out


def _ngram_hash(tokens: np.ndarray, n: int = 4) -> np.ndarray:
    """``[len(tokens)]`` int64 hash of the n-gram ENDING at each position (positions
    ``< n-1`` get -1)."""
    t = tokens.astype(np.int64)
    h = np.zeros_like(t)
    for k in range(n):
        h = h * 1_000_003 + np.concatenate([np.full(k, -7, np.int64), t[:len(t) - k]])   # shifted copies
    h = h & 0x7FFFFFFFFFFFFFFF
    h[: n - 1] = -1
    return h


def recurrence_classes(context: np.ndarray, seq_start: int, seq_len: int, doc_starts: np.ndarray,
                       chunk: int, window_chunks: int, n: int = 4) -> np.ndarray:
    """``[seq_len]`` int8 P1 class of every position of the sequence
    ``context[seq_start : seq_start+seq_len]``.  ``context`` is the stream around the
    sequence (its index 0 is stream position ``base``; ``doc_starts`` are the context
    indices where documents start, i.e. the position after each EOS, plus 0).

    0 beyond: earlier occurrence in same document within [seq_start, window_start)
    1 within: earlier occurrence in same document inside the attention window
    2 novel: never before in this document
    3 pre_seq: earlier occurrence exists in same document, but ONLY before seq_start (model window/W unreachable)
    Window of position i (sequence offset p, chunk c = p // chunk): stream positions
    >= max(seq_start, seq_start + (c - window_chunks) * chunk).
    """
    h = _ngram_hash(context, n)
    L = len(context)
    # previous occurrence of the same hash (or -1): sort by (hash, position)
    order = np.lexsort((np.arange(L), h))
    hs, ps = h[order], order
    prev = np.full(L, -1, np.int64)
    same = hs[1:] == hs[:-1]
    prev[ps[1:][same]] = ps[:-1][same]
    # document start of every position
    ds = np.sort(doc_starts)
    doc_of = ds[np.searchsorted(ds, np.arange(L), side="right") - 1]
    cls = np.full(seq_len, 2, np.int8)  # default novel
    for p in range(seq_len):
        i = seq_start + p
        if h[i] < 0:
            continue
        j = prev[i]
        if j < 0 or j < doc_of[i]:
            continue                                        # no earlier occurrence in this document
        win = seq_start + (p // chunk - window_chunks) * chunk
        within = False
        beyond = False
        pre_seq = False
        while j >= doc_of[i] and j >= 0:
            if j >= win and j >= seq_start:
                within = True
                break
            elif j >= seq_start:
                beyond = True
            else:
                pre_seq = True
            j = prev[j]
        if within:
            cls[p] = 1
        elif beyond:
            cls[p] = 0
        elif pre_seq:
            cls[p] = 3
        else:
            cls[p] = 2
    return cls


def neighbour_classes(scored: np.ndarray) -> np.ndarray:
    """``[B, T]`` P3 class of every position from the outer mask ``scored [B, T]`` (True =
    masked): 2 * (left neighbour masked) + (right neighbour masked); a missing neighbour
    (sequence edge) counts as masked."""
    s = scored.astype(bool)
    left = np.ones_like(s); left[:, 1:] = s[:, :-1]
    right = np.ones_like(s); right[:, :-1] = s[:, 1:]
    return (2 * left + right).astype(np.int8)


def aggregate(ce_on: np.ndarray, ce_off: np.ndarray, scored: np.ndarray, labels: np.ndarray,
              n_classes: int, chunk: int) -> Dict[str, np.ndarray]:
    """Per class: token count, mean CE (write on), mean gain, and the per-sequence mean
    gain (for a standard error over sequences); plus gain per class x chunk position.
    ``ce_* [B, T]``, ``scored [B, T]`` bool (outer-masked positions), ``labels [B, T]``."""
    B, T = ce_on.shape
    nc = T // chunk
    out = {"n": np.zeros(n_classes, np.int64), "loss": np.zeros(n_classes), "gain": np.zeros(n_classes),
           "seq_gain": np.full((n_classes, B), np.nan), "gain_by_chunk": np.zeros((n_classes, nc)),
           "n_by_chunk": np.zeros((n_classes, nc), np.int64)}
    g = ce_off - ce_on
    pos_chunk = np.broadcast_to((np.arange(T) // chunk)[None], (B, T))
    for k in range(n_classes):
        m = scored & (labels == k)
        n = int(m.sum())
        out["n"][k] = n
        if n:
            out["loss"][k] = float(ce_on[m].mean()); out["gain"][k] = float(g[m].mean())
            per_seq_n = m.sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                out["seq_gain"][k] = np.where(per_seq_n > 0, (g * m).sum(axis=1) / np.maximum(per_seq_n, 1), np.nan)
            for c in range(nc):
                mc = m & (pos_chunk == c)
                out["n_by_chunk"][k, c] = int(mc.sum())
                if mc.any():
                    out["gain_by_chunk"][k, c] = float(g[mc].mean())
    return out


def se_over_sequences(seq_gain: np.ndarray) -> float:
    v = seq_gain[~np.isnan(seq_gain)]
    return float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")


# ----------------------------------------------------------------------------
# planted recall: can the model reproduce a span it saw beyond its attention window?
# ----------------------------------------------------------------------------
PLANT_PLACEMENTS = ("beyond", "within", "none")
PLANT_MODES = ("block", "alt")


def planted_chunks(n_chunks: int, window: int, target: Optional[int] = None) -> Tuple[int, int, int]:
    """``(source, target, in-window copy)`` chunk indices of the planted-recall probe: the span is
    taken from chunk 1 and planted in chunk ``window + 3`` (two chunks beyond the attention window
    of the source), or in chunk ``target`` (``window + 3`` or later: a longer distance); the "within"
    placement also copies it into the chunk just before the target, inside the target's window
    (attention can copy it: the ceiling)."""
    ct = window + 3 if target is None else int(target)
    if ct < window + 3:
        raise ValueError(f"planted recall: the target chunk {ct} must be >= window + 3 = {window + 3}")
    if n_chunks < ct + 1:
        raise ValueError(f"planted recall needs >= {ct + 1} chunks (window {window}), the sequence has {n_chunks}")
    return 1, ct, ct - 1


def planted_probe_batch(local: np.ndarray, j_global: Sequence[int], seed: int, chunk: int, window: int,
                        length: int, cue: int, placement: str, mode: str,
                        eos: Optional[int] = None, tries: int = 8,
                        target: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Host side of the planted-recall probe for one local batch ``local [B, T]`` of held-out
    sequences (global indices ``j_global``): returns ``(ids [B, T] int32, loss_mask [B, T] bool)``.
    ``target``: the copy's chunk (``planted_chunks``; default ``window + 3``) - the span, its offsets
    within the chunks and the scored slots are the same draws for every target.

    Per sequence (draws keyed on ``(seed, 7, j)``, identical for every placement and mode, so all
    variants and all checkpoints are paired): a span of ``length`` tokens at a random offset in chunk
    1 is copied to a random offset in chunk ``window + 3``; "within" also copies it into chunk
    ``window + 2`` (inside the target's attention window); "none" plants nothing and scores the same
    positions of the natural text (the control: the memory's gain there is its generic prior).  The first ``cue`` tokens of the planted
    copy stay visible; ``mode`` "block" scores (masks) all the rest, "alt" every other one of them
    (each scored token keeps a visible predecessor).  Evaluate with ``outer_t = 1``: exactly the
    scored positions are masked, everything else is clean.  Offsets avoid EOS when possible."""
    if placement not in PLANT_PLACEMENTS or mode not in PLANT_MODES:
        raise ValueError(f"placement {placement!r} / mode {mode!r}")
    if not 0 < cue < length <= chunk:
        raise ValueError(f"need 0 < cue ({cue}) < length ({length}) <= chunk ({chunk})")
    B, T = local.shape
    cs, ct, cw = planted_chunks(T // chunk, window, target)
    ids = np.array(local, dtype=np.int32, copy=True)
    lm = np.zeros((B, T), bool)
    for r, j in enumerate(j_global):
        rng = np.random.default_rng([seed, 7, int(j)])
        for _ in range(tries):
            a = cs * chunk + int(rng.integers(0, chunk - length + 1))
            b = ct * chunk + int(rng.integers(0, chunk - length + 1))
            w = cw * chunk + int(rng.integers(0, chunk - length + 1))
            if eos is None or not ((local[r, a:a + length] == eos).any() or (local[r, b:b + length] == eos).any()
                                   or (local[r, w:w + length] == eos).any()):
                break
        span = local[r, a:a + length]
        if placement != "none":                  # "none": the same positions, nothing planted (control)
            ids[r, b:b + length] = span
        if placement == "within":
            ids[r, w:w + length] = span
        scored = np.arange(b + cue, b + length)
        lm[r, scored if mode == "block" else scored[1::2]] = True
    return ids, lm


def recall_train_batch(local: np.ndarray, j_global: Sequence[int], seed: int, chunk: int, window: int,
                       len_min: int, len_max: int, cue: int, eos: Optional[int] = None,
                       tries: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """Recall-only training batch (``--train_recall``): the probe's task on training sequences, with
    random geometry.  Per sequence (draws keyed on ``(seed, 11, j)``): a span of ``len_min..len_max``
    tokens from a random chunk ``cs`` is copied into a random chunk ``ct >= cs + window + 1`` (beyond
    the attention window), avoiding EOS in both; its first ``cue`` tokens stay visible and the rest is
    scored as a block or alternately (each with probability 1/2).  Returns ``(ids, loss_mask)``; train
    with ``outer_t = 1`` so that exactly the scored positions are masked.  A sequence without an
    EOS-free placement after ``tries`` draws is left unchanged and unscored."""
    B, T = local.shape
    nc = T // chunk
    if nc < window + 2 or not 0 < cue < len_min <= len_max <= chunk:
        raise ValueError(f"recall_train_batch: need >= window + 2 chunks and 0 < cue < len_min <= len_max <= chunk")
    ids = np.array(local, dtype=np.int32, copy=True)
    lm = np.zeros((B, T), bool)
    for r, j in enumerate(j_global):
        rng = np.random.default_rng([seed, 11, int(j)])
        block = rng.random() < 0.5
        for _ in range(tries):
            L = int(rng.integers(len_min, len_max + 1))
            cs = int(rng.integers(0, nc - window - 1))
            ct = int(rng.integers(cs + window + 1, nc))
            a = cs * chunk + int(rng.integers(0, chunk - L + 1))
            b = ct * chunk + int(rng.integers(0, chunk - L + 1))
            if eos is None or not ((local[r, a:a + L] == eos).any() or (local[r, b:b + L] == eos).any()):
                ids[r, b:b + L] = local[r, a:a + L]
                scored = np.arange(b + cue, b + L)
                lm[r, scored if block else scored[1::2]] = True
                break
    return ids, lm
