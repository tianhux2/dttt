# coding=utf-8
"""
``tools/elbo_eval.py`` around ``measure``, on CPU with a tiny checkpoint trained here:

* ``sequences``: by default the checkpoint's sources under its ``data_root`` and ``pg19_test``, ``n_per_source``
  each in source order, consecutive ``seq_len`` slices of the held-out tail; the same on every call;
* ``train_config``: the checkpoint's configuration, at another length with one segment;
* the command line end to end, at the checkpoint's length and at ``--seq_len``: the arrays' shapes, the summary
  equal to the arrays (NELBO = the mean over t of the CE per masked token), the length recorded.
"""
from __future__ import annotations

import dataclasses
import json
import os
import shutil
import subprocess
import sys
import tempfile

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8 "
                                   "--xla_cpu_enable_concurrency_optimized_scheduler=false")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import numpy as np

V, T, B = 128, 64, 8
PG19 = "pg19_test_a4dfccb3b3ac65f9"


def _synthetic_source(d, n_tokens, seed):
    os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(seed)
    toks = rng.integers(1, 100, size=n_tokens, dtype=np.uint32)
    toks[rng.random(n_tokens) < 0.02] = 0          # a few EOS
    toks.astype("<u4").tofile(os.path.join(d, "tokens.u32"))
    json.dump({"name": os.path.basename(d), "num_tokens": int(n_tokens), "num_docs": 1,
               "tokenizer": {"class": "synthetic", "len": V, "vocab_size": V}}, open(os.path.join(d, "meta.json"), "w"))
    return toks


def _check(failures, ok, name, detail=""):
    failures += [] if ok else [name]
    print(f"[{'PASS' if ok else 'FAIL'}] {name}   {detail}")


def _elbo(args):
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "elbo_eval.py"), *args], capture_output=True,
                       text=True, cwd=ROOT, timeout=1500)
    if r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-3000:])
    return r


def main():
    from tools.elbo_eval import sequences, train_config
    failures = []
    tmp = tempfile.mkdtemp()
    cfg_dir = os.path.join(tmp, "cfg"); os.makedirs(cfg_dir)
    json.dump(dict(vocab_size=V, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                   num_attention_heads=4, num_key_value_heads=2, head_dim=8, suffix_len=1,
                   chunk_size=8, window_chunks=1, mask_token_id=V - 1, pad_token_id=0, eos_token_id=0),
              open(os.path.join(cfg_dir, "config.json"), "w"))
    data = os.path.join(tmp, "data")
    streams = {n: _synthetic_source(os.path.join(data, n), 20000, i + 1) for i, n in enumerate(("a", "b", PG19))}
    holdout = 32 * T
    train = ["--model_config", cfg_dir, "--data_root", data, "--sources", "a=a:0.6,b=b:0.4", "--seq_len", str(T),
             "--batch_size", str(B), "--steps", "2", "--lr", "1e-3", "--warmup_steps", "2", "--inner_lr_warmup_steps",
             "2", "--fused_ce_block", "32", "--attn_chunk_batch", "2", "--cart_window", "16", "--seed", "7",
             "--holdout", str(holdout), "--inner_task", "causal", "--outer_shift", "true", "--eval_batches", "1",
             "--ckpt_dir", os.path.join(tmp, "run"), "--ckpt_interval", "2"]
    r = subprocess.run([sys.executable, "-m", "ttt.train", *train], capture_output=True, text=True, timeout=1500,
                       env=dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src")))
    ckpt = os.path.join(tmp, "run", "step_00000001")
    _check(failures, r.returncode == 0 and os.path.exists(os.path.join(ckpt, "DONE")), "a tiny checkpoint trains")
    if r.returncode:
        print(r.stdout[-2000:]); print(r.stderr[-3000:])
        print("FAILURES: " + "; ".join(failures))
        return 1

    # 1. the default sequences: the training sources under data_root, then PG-19 test; held-out, consecutive
    tc = train_config(ckpt)
    rows = sequences(tc, None, 2, 0)
    names = [n for n, _ in rows]
    ok = names == ["a", "a", "b", "b", "pg19_test", "pg19_test"] and all(len(x) == T for _, x in rows)
    for n, src in (("a", "a"), ("b", "b"), ("pg19_test", PG19)):
        toks, mine = streams[src], [x for m, x in rows if m == n]
        tail0 = len(toks) - holdout
        starts = [o for o in range(tail0, len(toks) - T + 1) if np.array_equal(toks[o:o + T], mine[0])]
        ok = ok and len(starts) >= 1 and np.array_equal(toks[starts[0] + T:starts[0] + 2 * T], mine[1])
    again = sequences(tc, None, 2, 0)
    ok = ok and all(m == n and np.array_equal(x, y) for (m, x), (n, y) in zip(rows, again))
    _check(failures, ok, "sequences: sources under data_root + pg19_test, consecutive held-out slices, repeatable",
           str(names))

    # 2. train_config at another length: one segment, everything else the checkpoint's
    tc2 = train_config(ckpt, 2 * T)
    ok = (tc2.seq_len == 2 * T and tc2.segment_chunks == 0
          and dataclasses.replace(tc2, seq_len=tc.seq_len, segment_chunks=tc.segment_chunks) == tc)
    ok = ok and all(len(x) == 2 * T for _, x in sequences(tc2, None, 2, 0))
    _check(failures, ok, "train_config: --seq_len gives that length in one segment, the rest unchanged")

    # 3. the command line, at the checkpoint's length and at twice it
    res = {}
    for tag, extra in (("T", []), ("2T", ["--seq_len", str(2 * T)])):
        out = os.path.join(tmp, f"elbo_{tag}.npz")
        r = _elbo(["--ckpt", ckpt, "--n_per_source", "2", "--n_t", "2", "--batch", "2", "--out", out, *extra])
        res[tag] = np.load(out) if r.returncode == 0 else None
    ok = res["T"] is not None and res["2T"] is not None
    if ok:
        z = res["T"]
        s, meta = json.loads(str(z["summary"])), json.loads(str(z["meta"]))
        ce, cnt = z["ce"], z["count"]
        nelbo = (ce.sum(-1) / cnt.sum(-1)).mean(0)
        ok = (ce.shape == cnt.shape == (2, 2, 6) and list(z["sources"]) == names and np.isfinite(ce).all()
              and np.allclose(s["nelbo"], nelbo, rtol=1e-6) and sorted(s["by_source"]) == ["a", "b", "pg19_test"]
              and meta["seq_len"] == T and json.loads(str(res["2T"]["meta"]))["seq_len"] == 2 * T
              and res["2T"]["count"].sum() > 1.5 * cnt.sum())
        detail = f"NELBO (write, no write) {s['nelbo'][0]:.3f} {s['nelbo'][1]:.3f}"
    else:
        detail = "a run failed"
    _check(failures, ok, "command line: shapes, summary == arrays, --seq_len recorded and scored", detail)

    shutil.rmtree(tmp, ignore_errors=True)
    print("ALL PASS" if not failures else "FAILURES: " + "; ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
