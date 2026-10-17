# coding=utf-8
"""Data identity across machines and batch splits: the path-free data hash over the step batch,
the legacy (pre-2026-10) hash and the resume check built on it, and evaluation by step batches."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.ttt.config import TrainConfig
from src.ttt.data import Sampler, Source, doc_windows, sources_hash
from src.ttt.train import check_resume_data

EOS, T, H = 7, 32, 20000


def _stream(d, seed):
    """A token stream with documents of 1..5000 tokens (several longer than the 64-sequence lookback)."""
    os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(seed)
    lens = np.where(rng.random(300) < 0.1, rng.integers(2100, 5000, 300), rng.integers(1, 200, 300))
    toks = np.concatenate([np.concatenate([rng.integers(10, 99, L - 1), [EOS]]) for L in lens]).astype("<u4")
    toks.tofile(os.path.join(d, "tokens.u32"))
    json.dump({"name": os.path.basename(d), "num_tokens": int(len(toks)), "num_docs": int(len(lens)),
               "tokenizer": {"len": 128, "eos_token_id": EOS}}, open(os.path.join(d, "meta.json"), "w"))
    return Source(os.path.basename(d).rsplit("_", 1)[0], d, 0.0, int(len(toks)), 128, EOS)


def _v1(sources, paths, seed, seq_len, batch, holdout, extra):
    """The data hash exactly as data.sources_hash computed it before 2026-10."""
    return hashlib.sha256(json.dumps(
        [(s.name, os.path.abspath(p), round(s.weight, 9), s.num_tokens) for s, p in zip(sources, paths)]
        + [seed, seq_len, batch, holdout] + (list(extra) if extra else []), sort_keys=True).encode()).hexdigest()[:16]


def main():
    failures = []
    tmp = tempfile.mkdtemp()
    a = _stream(os.path.join(tmp, "new", "a_0123456789abcdef"), 1)
    b = _stream(os.path.join(tmp, "new", "b_0123456789abcdef"), 2)
    srcs = [Source(a.name, a.path, 0.7, a.num_tokens, 128, EOS), Source(b.name, b.path, 0.3, b.num_tokens, 128, EOS)]
    wins = [doc_windows(s.path, T, H) for s in srcs]
    smp = lambda G, A, s=srcs, seed=5: Sampler(s, T, G, seed, holdout_tokens=H, eos=EOS, windows=wins, accum=A)

    # 1. the hash counts sequences per optimiser step, not the split
    ok = smp(8, 4).hash == smp(32, 1).hash == smp(16, 2).hash and smp(8, 4).hash not in (smp(8, 2).hash, smp(16, 1).hash)
    failures += [] if ok else ["step-batch hash"]
    print(f"[{'PASS' if ok else 'FAIL'}] data hash: 8 x 4 == 32 x 1 == 16 x 2, != 8 x 2 and 16 x 1")

    # 2. path-free: the same streams in another directory keep the hash; the legacy form does not
    moved = []
    for s in srcs:
        d = os.path.join(tmp, "elsewhere", os.path.basename(s.path))
        shutil.copytree(s.path, d)
        moved.append(Source(s.name, d, s.weight, s.num_tokens, 128, EOS))
    m = Sampler(moved, T, 32, 5, holdout_tokens=H, eos=EOS, windows=wins)
    ok = m.hash == smp(32, 1).hash and m.legacy_hash([s.path for s in moved], 32) != m.legacy_hash([s.path for s in srcs], 32)
    failures += [] if ok else ["path-free hash"]
    print(f"[{'PASS' if ok else 'FAIL'}] data hash: unchanged by moving the streams; the legacy form changes")

    # 3. the legacy form is the old formula, bit for bit (it must keep reproducing the TPU checkpoints)
    s32 = smp(32, 1)
    old_paths = [os.path.join("/old/root", "bucket/data", os.path.basename(s.path)) for s in srcs]
    ok = s32.legacy_hash(old_paths, 32) == _v1(s32.sources, old_paths, 5, T, 32, H, s32._extra)
    failures += [] if ok else ["legacy formula"]
    print(f"[{'PASS' if ok else 'FAIL'}] legacy hash == the pre-2026-10 formula")

    # 4. resuming a legacy checkpoint: recomputed from its recorded (relative) paths and --resume_legacy_root
    ck = os.path.join(tmp, "ckpt", "step_00000099")
    os.makedirs(ck)
    rel = [os.path.join("bucket/data", os.path.basename(s.path)) for s in srcs]
    json.dump({"sources": ",".join(f"{s.name}={p}:{w}" for s, p, w in zip(srcs, rel, (0.7, 0.3))),
               "batch_size": 32}, open(os.path.join(ck, "train_config.json"), "w"))
    st = {"step": 99, "data_hash": _v1(s32.sources, [os.path.join("/old/root", p) for p in rel], 5, T, 32, H, s32._extra)}

    def resume(sampler, root, state=st):
        try:
            check_resume_data(ck, state, sampler, TrainConfig(resume_legacy_root=root), log=lambda *_: None)
            return True
        except RuntimeError:
            return False
    ok = (resume(smp(8, 4), "/old/root") and resume(smp(32, 1), "/old/root") and not resume(smp(8, 4), "")
          and not resume(smp(8, 2), "/old/root") and not resume(smp(8, 4, seed=6), "/old/root")
          and not resume(smp(8, 4, s=[srcs[0], Source(b.name, b.path, 0.4, b.num_tokens, 128, EOS)]), "/old/root"))
    failures += [] if ok else ["legacy resume"]
    print(f"[{'PASS' if ok else 'FAIL'}] legacy resume: accepted at 8 x 4 and 32 x 1 with the old root; refused "
          f"without it, at 16 per step, with another seed or weights")

    # 5. a checkpoint with the new hash: any split of the same step batch, nothing else
    st2 = {"step": 99, "data_hash": smp(32, 1).hash, "data_hash_version": 2}
    ok = resume(smp(8, 4), "", st2) and resume(smp(16, 2), "", st2) and not resume(smp(16, 1), "", st2)
    failures += [] if ok else ["v2 resume"]
    print(f"[{'PASS' if ok else 'FAIL'}] resume (hash version 2): 8 x 4 and 16 x 2 accepted for 32 x 1, 16 x 1 refused")

    # 6. evaluation by step batches: micro-batches k*A .. k*A+A-1 hold the sequences, document
    #    lengths, beyond-window counts and document starts of step batch k
    g, e = smp(8, 4), smp(32, 1)
    ok = g.eval_step_batches(0) == e.eval_batches(0) >= 1
    for k in range(g.eval_step_batches(0)):
        micro = [g.eval_doc_info(0, k * 4 + i, EOS, 3 * 8) for i in range(4)]
        whole = e.eval_doc_info(0, k, EOS, 3 * 8)
        ok &= all(np.array_equal(np.concatenate([mi[j] for mi in micro]), whole[j]) for j in range(3))
        ok &= np.array_equal(np.concatenate([g.eval_batch(0, k * 4 + i) for i in range(4)]), e.eval_batch(0, k))
    failures += [] if ok else ["eval step batches"]
    print(f"[{'PASS' if ok else 'FAIL'}] evaluation: 4 micro-batches of 8 == one step batch of 32 "
          f"(sequences and document info, {g.eval_step_batches(0)} step batches)")

    print("ALL PASS" if not failures else "FAILURES: " + "; ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
