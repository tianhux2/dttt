# coding=utf-8
"""Tests for evaluation probes (probes.py): P1 recurrence classes including pre_seq, P3 neighbour classes."""
from __future__ import annotations

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from src.ttt.probes import P1_CLASSES, P3_CLASSES, aggregate, neighbour_classes, recurrence_classes


def main():
    failures = []
    chunk = 64
    window_chunks = 2  # attention window = 2 chunks = 128 tokens
    seq_len = 256     # 4 chunks: chunk 0 (0..63), chunk 1 (64..127), chunk 2 (128..191), chunk 3 (192..255)
    seq_start = 100   # lookback is context[0:100], sequence is context[100:356]

    # Construct context of length 400
    # context[0:100]: before sequence (lookback)
    # context[100:356]: the sequence (offsets 0..255)
    # doc_starts: e.g. document starts at 0
    #
    # Pattern A: tokens [10, 11, 12, 13] (4-gram hash)
    # Put Pattern A at:
    # 1. offset 10 (before sequence, inside document): context[10:14] = [10, 11, 12, 13]
    # And at chunk 3 of sequence: offset seq_start + 200: context[300:304] = [10, 11, 12, 13]
    # At position p = 203 (chunk 3, window = [seq_start + (3 - 2)*64, ...] = [seq_start + 64, ...]):
    # Since Pattern A only appeared at offset 10 (< seq_start), it should be classified as PRE_SEQ (class 3)!

    context = np.arange(500, 900, dtype=np.int64)  # distinct tokens by default
    # Doc starts at 0
    doc_starts = np.array([0], dtype=np.int64)

    # 1. Test pre_seq: only appears before seq_start (inside Document 2, doc start is 51, seq_start is 100)
    context[60:64] = [10, 11, 12, 13]
    context[seq_start + 200 : seq_start + 204] = [10, 11, 12, 13]

    # 2. Test beyond: appears at sequence chunk 0 (seq_start + 10), and repeated at sequence chunk 3 (seq_start + 220)
    # For p = 223 (chunk 3): window starts at seq_start + (3 - 2)*64 = seq_start + 64.
    # Earlier occurrence is at seq_start + 10, which is >= seq_start and < window_start.
    # So it should be BEYOND (class 0)!
    context[seq_start + 10 : seq_start + 14] = [20, 21, 22, 23]
    context[seq_start + 220 : seq_start + 224] = [20, 21, 22, 23]

    # 3. Test within: appears at chunk 2 (seq_start + 140), and repeated at chunk 3 (seq_start + 240)
    # For p = 243 (chunk 3): window starts at seq_start + 64. Earlier occurrence is at seq_start + 140 >= window_start.
    # So it should be WITHIN (class 1)!
    context[seq_start + 140 : seq_start + 144] = [30, 31, 32, 33]
    context[seq_start + 240 : seq_start + 244] = [30, 31, 32, 33]

    # 4. Test document boundary: pattern appears at offset 5 (in doc 1), but doc 2 starts at 51.
    # Then repeated at seq_start + 50 (in doc 2).
    # Even though it appeared before seq_start, it was in a PREVIOUS document.
    # So it should be NOVEL (class 2)!
    context[5:9] = [40, 41, 42, 43]
    context[seq_start + 50 : seq_start + 54] = [40, 41, 42, 43]
    doc_starts_multi = np.array([0, 51], dtype=np.int64)

    cls = recurrence_classes(context, seq_start, seq_len, doc_starts_multi, chunk, window_chunks, n=4)

    # Verify positions:
    # Position 203: 4-gram ending at seq_start + 203 -> pre_seq (3)
    p_pre = cls[203]
    ok_pre = p_pre == 3
    failures += [] if ok_pre else [f"pre_seq classification failed: got {p_pre}, expected 3"]
    print(f"[{'PASS' if ok_pre else 'FAIL'}] P1 pre_seq: earlier occurrence only before seq_start (class {p_pre})")

    # Position 223: 4-gram ending at seq_start + 223 -> beyond (0)
    p_beyond = cls[223]
    ok_beyond = p_beyond == 0
    failures += [] if ok_beyond else [f"beyond classification failed: got {p_beyond}, expected 0"]
    print(f"[{'PASS' if ok_beyond else 'FAIL'}] P1 beyond: earlier occurrence in [seq_start, win_start) (class {p_beyond})")

    # Position 243: 4-gram ending at seq_start + 243 -> within (1)
    p_within = cls[243]
    ok_within = p_within == 1
    failures += [] if ok_within else [f"within classification failed: got {p_within}, expected 1"]
    print(f"[{'PASS' if ok_within else 'FAIL'}] P1 within: earlier occurrence inside attention window (class {p_within})")

    # Position 53: 4-gram ending at seq_start + 53 -> novel (2) because doc start is 51
    p_novel = cls[53]
    ok_novel = p_novel == 2
    failures += [] if ok_novel else [f"novel across doc boundary failed: got {p_novel}, expected 2"]
    print(f"[{'PASS' if ok_novel else 'FAIL'}] P1 novel across doc boundary: earlier occurrence in prior doc (class {p_novel})")

    # 5. Test aggregate with 4 classes
    B = 2
    ce_on = np.ones((B, seq_len), dtype=np.float32)
    ce_off = np.full((B, seq_len), 2.0, dtype=np.float32)
    scored = np.ones((B, seq_len), dtype=bool)
    labels = np.full((B, seq_len), 2, dtype=np.int8)  # default novel
    labels[:, 203] = 3
    labels[:, 223] = 0
    labels[:, 243] = 1

    agg = aggregate(ce_on, ce_off, scored, labels, len(P1_CLASSES), chunk)
    ok_agg = (len(agg["n"]) == 4 and agg["n"][3] == B and agg["n"][0] == B and
              agg["n"][1] == B and agg["gain"][3] == 1.0)
    failures += [] if ok_agg else ["P1 aggregate with 4 classes failed"]
    print(f"[{'PASS' if ok_agg else 'FAIL'}] P1 aggregate: 4 classes properly aggregated (n={agg['n']})")

    # P3: 2 * (left neighbour masked) + (right neighbour masked); a missing neighbour counts as masked
    sc = np.array([[0, 1, 1, 0, 1, 0], [1, 0, 1, 1, 1, 1]], bool)
    lab3 = neighbour_classes(sc)
    ok_p3 = (lab3.tolist() == [[3, 1, 2, 3, 0, 3], [2, 3, 1, 3, 3, 3]] and len(P3_CLASSES) == 4
             and P3_CLASSES[1] == "Lv_Rm" and P3_CLASSES[2] == "Lm_Rv")
    agg3 = aggregate(np.zeros(sc.shape), np.ones(sc.shape), sc, lab3, len(P3_CLASSES), 3)
    ok_p3 &= agg3["n"].tolist() == [1, 2, 2, 3] and float(agg3["gain"][1]) == 1.0
    failures += [] if ok_p3 else ["P3 neighbour classes"]
    print(f"[{'PASS' if ok_p3 else 'FAIL'}] P3 neighbour classes: left/right visibility of scored tokens ({lab3.tolist()}, n={agg3['n'].tolist()})")

    print("ALL PASS" if not failures else "FAILURES: " + "; ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
