# coding=utf-8
"""On-demand checkpoints and clean stops of a running job: a TRIGGER_SAVE file makes the trainer
write a checkpoint at the end of the current step (outside the checkpoint interval) and remove
the file; TRIGGER_STOP does the same and exits cleanly; the stopped run resumes at the next step.

    JAX_PLATFORMS=cpu .venv/bin/python tests/test_trigger.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import numpy as np

V = 128


def _source(d, n, seed):
    os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(seed)
    t = rng.integers(1, 100, size=n, dtype=np.uint32)
    t[rng.random(n) < 0.02] = 0
    t.astype("<u4").tofile(os.path.join(d, "tokens.u32"))
    json.dump({"name": os.path.basename(d), "num_tokens": int(n), "num_docs": 1,
               "tokenizer": {"class": "synthetic", "len": V, "vocab_size": V}}, open(os.path.join(d, "meta.json"), "w"))


def main():
    tmp = tempfile.mkdtemp()
    src = _source(os.path.join(tmp, "a"), 60000, 1)
    cfg = os.path.join(tmp, "cfg"); os.makedirs(cfg)
    json.dump(dict(vocab_size=V, hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
                   num_key_value_heads=2, head_dim=8, suffix_len=1, chunk_size=8, window_chunks=1,
                   mask_token_id=V - 1, pad_token_id=0, eos_token_id=0), open(os.path.join(cfg, "config.json"), "w"))
    trig = os.path.join(tmp, "trig"); os.makedirs(trig)
    ck = os.path.join(tmp, "ck")
    common = [sys.executable, "-m", "ttt.train", "--model_config", cfg, "--sources", f"a={os.path.join(tmp, 'a')}:1.0",
              "--seq_len", "64", "--batch_size", "4", "--lr", "1e-3", "--warmup_steps", "2", "--fused_ce_block", "32",
              "--log_interval", "1", "--attn_chunk_batch", "2", "--cart_window", "16", "--holdout", "4096",
              "--eval_batches", "0", "--ckpt_dir", ck, "--ckpt_interval", "0"]
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src"), TTT_TRIGGER_DIR=trig)
    failures = []
    # A stale trigger must not fire at start-up.
    open(os.path.join(trig, "TRIGGER_STOP"), "w").close()
    p = subprocess.Popen(common + ["--steps", "40"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = []
    fired_save = fired_stop = False
    t0 = time.time()
    while True:
        line = p.stdout.readline()
        if not line:
            if p.poll() is not None:
                break
            continue
        lines.append(line.rstrip())
        if line.startswith("[      1]") and not fired_save:
            open(os.path.join(trig, "TRIGGER_SAVE"), "w").close(); fired_save = True
        if line.startswith("[      4]") and not fired_stop:
            open(os.path.join(trig, "TRIGGER_STOP"), "w").close(); fired_stop = True
        if time.time() - t0 > 600:
            p.kill(); failures.append("timeout"); break
    out = "\n".join(lines)
    steps = [l for l in lines if l.startswith("[      ")]
    dirs = sorted(os.listdir(ck)) if os.path.exists(ck) else []
    ok = (p.returncode == 0 and "[ttt] triggered checkpoint ->" in out and "triggered stop after step" in out
          and any(d.startswith("step_0000000") for d in dirs) and len(steps) < 40
          and not os.path.exists(os.path.join(trig, "TRIGGER_SAVE")) and not os.path.exists(os.path.join(trig, "TRIGGER_STOP")))
    failures += [] if ok else [f"trigger run: rc={p.returncode} dirs={dirs} steps={len(steps)}"]
    print(f"[{'PASS' if ok else 'FAIL'}] TRIGGER_SAVE writes a checkpoint mid-run, TRIGGER_STOP saves and exits   "
          f"checkpoints {dirs}, {len(steps)} of 40 steps logged")
    if not ok:
        print(out[-2500:])
    # The stopped run resumes at the step after its last checkpoint.
    last = int(dirs[-1].split("_")[1]) if dirs else -1
    r = subprocess.run(common + ["--steps", str(last + 3), "--resume", "auto"], env=env, capture_output=True, text=True)
    ok2 = r.returncode == 0 and f"at step {last + 1}" in r.stdout and f"[      {last + 1}]" in r.stdout
    failures += [] if ok2 else [f"resume: rc={r.returncode} {r.stdout[-800:]} {r.stderr[-800:]}"]
    print(f"[{'PASS' if ok2 else 'FAIL'}] --resume auto continues at step {last + 1}")
    shutil.rmtree(tmp, ignore_errors=True)
    print("ALL PASS" if not failures else "FAILURES: " + "; ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
