# coding=utf-8
"""
Launching from run configurations, on CPU (8 emulated devices):

* ``config.load_run_config``: ``base`` (a file or a list), ``#`` comments, unknown keys and cycles refused;
  ``--config`` under the flags given; ``data_root`` for relative stream directories;
* the warmup-stable-decay schedule: its values, and bit for bit the old cosine schedule when lr == end_lr
  (every run so far) - the same optimiser steps;
* ``--continue_from``: on the same data a branch is the trunk continued, BITWISE; on new data (another
  length) the stream starts at its first batch, the data offset survives an interruption, and the stopped
  and the uninterrupted stage end BITWISE equal; the method's write switched on at a branch trains; the
  last step is always checkpointed; the evaluation runs at the continued step; the same data at another step
  batch is refused (the stream would start again at its first sample);
* a run from a config file (``data_root`` + relative streams) equals the same run from flags, BITWISE.
"""
from __future__ import annotations

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


def _synthetic_source(d, n_tokens, seed):
    os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(seed)
    toks = rng.integers(1, 100, size=n_tokens, dtype=np.uint32)
    toks[rng.random(n_tokens) < 0.02] = 0          # a few EOS
    toks.astype("<u4").tofile(os.path.join(d, "tokens.u32"))
    json.dump({"name": os.path.basename(d), "num_tokens": int(n_tokens), "num_docs": 1,
               "tokenizer": {"class": "synthetic", "len": V, "vocab_size": V}}, open(os.path.join(d, "meta.json"), "w"))
    return d


def _run(args):
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src"))
    r = subprocess.run([sys.executable, "-m", "ttt.train", *args], capture_output=True, text=True, env=env, timeout=1500)
    if r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-3000:])
    return r


def _load(d):
    from src.ttt.sharded_io import ShardedFileReader
    import equinox as eqx, jax
    from src.ttt.config import ModelConfig
    from src.ttt.model import Model
    cfg = ModelConfig.load(os.path.join(d, "model_config.json"))
    skel = eqx.filter(Model(cfg, jax.random.PRNGKey(0)), eqx.is_array)
    with ShardedFileReader(os.path.join(d, "model.eqx")) as f:
        p = eqx.tree_deserialise_leaves(f, jax.tree_util.tree_map(np.asarray, skel))
    return jax.tree_util.tree_leaves(p)


def _same(d1, d2):
    if not (os.path.exists(os.path.join(d1, "DONE")) and os.path.exists(os.path.join(d2, "DONE"))):
        return False, float("nan")
    a, b = _load(d1), _load(d2)
    return all(np.array_equal(x, y) for x, y in zip(a, b)), max(float(np.max(np.abs(x - y))) for x, y in zip(a, b))


def _lines(r, prefix):
    return [l for l in r.stdout.splitlines() if l.startswith(prefix)]


def _check(failures, ok, name, detail=""):
    failures += [] if ok else [name]
    print(f"[{'PASS' if ok else 'FAIL'}] {name}   {detail}")


def unit(failures, tmp):
    import optax
    from src.ttt.config import TrainConfig, load_run_config
    from src.ttt.data import split_sources
    from src.ttt.train import lr_schedule, parse_args

    # 1. base chains (a file, then a list), comments, the child wins; unknown keys and cycles are refused
    d = os.path.join(tmp, "runs"); os.makedirs(d)
    w = lambda name, obj: json.dump(obj, open(os.path.join(d, name), "w"))
    w("base.json", {"#": "the common settings", "seq_len": 128, "lr": 1e-3, "steps": 10, "sources": "a=a:1"})
    w("arm.json", {"base": "base.json", "inner_lr": 0.82, "steps": 20, "#steps": "TODO"})
    w("long.json", {"seq_len": 512, "segment_chunks": 8})
    w("stage.json", {"base": ["arm.json", "long.json"], "steps": 30})
    s = load_run_config(os.path.join(d, "stage.json"))
    ok = s == {"seq_len": 512, "lr": 1e-3, "steps": 30, "sources": "a=a:1", "inner_lr": 0.82, "segment_chunks": 8}
    _check(failures, ok, "load_run_config: base file and list, comments, the child wins", str(s))
    w("typo.json", {"base": "base.json", "inner_lrr": 1.0})
    w("cyc_a.json", {"base": "cyc_b.json"}); w("cyc_b.json", {"base": "cyc_a.json"})
    errs = []
    for name in ("typo.json", "cyc_a.json"):
        try:
            load_run_config(os.path.join(d, name)); errs.append("")
        except ValueError as e:
            errs.append(str(e))
    ok = "inner_lrr" in errs[0] and "cycle" in errs[1]
    _check(failures, ok, "unknown key (named) and a base cycle are refused", " | ".join(e[:60] for e in errs))

    # 2. --config under the flags; JSON 1 for a float and 2e1 for an int land in their field types
    w("types.json", {"base": "stage.json", "inner_lr": 1, "steps": 2e1, "outer_shift": True,
                     "inner_mask_range": [0.4, 0.9]})
    c = parse_args(["--config", os.path.join(d, "types.json"), "--seed", "8", "--lr", "2e-3"])
    ok = (c.seed == 8 and c.lr == 2e-3 and c.seq_len == 512 and c.steps == 20 and isinstance(c.steps, int)
          and c.inner_lr == 1.0 and isinstance(c.inner_lr, float) and c.outer_shift is True
          and c.inner_mask_range == (0.4, 0.9) and c.segment_chunks == 8)
    ok = ok and parse_args(["--seed", "8"]) == TrainConfig(seed=8)
    _check(failures, ok, "--config under the flags; field types; no --config = the defaults")

    # 3. data_root: relative stream directories (and the unigram table) under it, absolute ones kept
    sp = split_sources("a=x/a:0.6,b=/abs/b:0.4", "/data")
    tc = TrainConfig(data_root="/data", unigram_path="u.i64")
    ok = (sp == [("a", "/data/x/a", 0.6), ("b", "/abs/b", 0.4)] and split_sources("a=x/a:1") == [("a", "x/a", 1.0)]
          and tc.data_path(tc.unigram_path) == "/data/u.i64" and tc.data_path("/abs/u") == "/abs/u"
          and TrainConfig().data_path("bucket/u") == "bucket/u")
    _check(failures, ok, "data_root resolves relative streams and the unigram table", str(sp))

    # 4. WSD: 0 -> lr over the warm-up, flat, linear to end_lr over [decay_start, decay_end], flat after
    tc = TrainConfig(lr=1e-3, end_lr=1e-4, warmup_steps=100, decay_start=1000, decay_end=2000)
    f = lr_schedule(tc)
    got = [float(f(s)) for s in (0, 50, 100, 999, 1000, 1500, 2000, 5000)]
    want = [0.0, 5e-4, 1e-3, 1e-3, 1e-3, 5.5e-4, 1e-4, 1e-4]
    ok = all(abs(g - x) < 1e-9 for g, x in zip(got, want))
    flat = lr_schedule(TrainConfig(lr=1e-3, end_lr=1e-5, warmup_steps=100))
    ok = ok and float(flat(100)) == float(flat(10 ** 6)) == np.float32(1e-3)
    _check(failures, ok, "WSD schedule values; no decay = flat after the warm-up", " ".join(f"{g:.2e}" for g in got))
    # the GPU recipe (lr = end_lr, cosine): the same float32 rates at every step, so the same optimiser steps
    old = optax.warmup_cosine_decay_schedule(init_value=0.0, peak_value=1e-3, warmup_steps=5000,
                                             decay_steps=50_000, end_value=1e-3)
    new = lr_schedule(TrainConfig(lr=1e-3, end_lr=1e-3, warmup_steps=5000))
    steps = np.concatenate([np.arange(0, 6000), np.arange(6000, 60_000, 997)]).astype(np.int32)
    import jax
    a, b = np.asarray(jax.vmap(old)(steps)), np.asarray(jax.vmap(new)(steps))
    ok = a.dtype == b.dtype and np.array_equal(a, b)
    _check(failures, ok, "WSD == the old cosine schedule bit for bit when lr == end_lr",
           f"{len(steps)} steps, max diff {float(np.max(np.abs(a - b))):.1e}")

    # 5. validation: decay bounds; continue_from with init_from / an explicit resume path
    bad = [dict(decay_start=50, decay_end=200, warmup_steps=100), dict(decay_start=300, decay_end=300),
           dict(decay_end=300), dict(continue_from="x", init_from="y"), dict(continue_from="x", resume="/p"),
           dict(continue_from="auto"), dict(continue_from="ckpt/a/step_00000009/", ckpt_dir="ckpt/a")]
    refused = []
    for kw in bad:
        try:
            TrainConfig(**kw); refused.append(False)
        except ValueError:
            refused.append(True)
    ok = all(refused) and TrainConfig(continue_from="x", resume="auto").continue_from == "x"
    _check(failures, ok, "decay bounds, continue_from exclusivity and its own ckpt_dir validated", str(refused))


def training(failures, tmp):
    cfg_dir = os.path.join(tmp, "cfg"); os.makedirs(cfg_dir)
    json.dump(dict(vocab_size=V, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                   num_attention_heads=4, num_key_value_heads=2, head_dim=8, suffix_len=1,
                   chunk_size=8, window_chunks=1, mask_token_id=V - 1, pad_token_id=0, eos_token_id=0),
              open(os.path.join(cfg_dir, "config.json"), "w"))
    data = os.path.join(tmp, "data")
    src_a = _synthetic_source(os.path.join(data, "a"), 20000, 1)
    src_b = _synthetic_source(os.path.join(data, "b"), 20000, 2)
    common = dict(model_config=cfg_dir, seq_len=T, batch_size=B, lr=1e-3, warmup_steps=2, inner_lr_warmup_steps=2,
                  fused_ce_block=32, log_interval=1, attn_chunk_batch=2, cart_window=16, seed=7, holdout=32 * T,
                  inner_task="causal", outer_shift=True, eval_batches=1)
    flags = lambda d: sum((["--" + k, json.dumps(v) if isinstance(v, bool) else str(v)] for k, v in d.items()), [])
    runs = os.path.join(tmp, "runs"); os.makedirs(runs, exist_ok=True)
    trunk_cfg = os.path.join(runs, "trunk.json")
    json.dump({**common, "data_root": data, "sources": "a=a:0.6,b=b:0.4", "#sources": "relative to data_root"},
              open(trunk_cfg, "w"))
    ck = lambda name, step: os.path.join(tmp, name, f"step_{step:08d}")

    # A. a config file with data_root and relative streams == the same run from flags with absolute paths
    r_f = _run(["--config", trunk_cfg, "--steps", "4", "--ckpt_dir", os.path.join(tmp, "trunk"), "--ckpt_interval", "2"])
    r_c = _run(flags(common) + ["--sources", f"a={src_a}:0.6,b={src_b}:0.4", "--steps", "2",
                                "--ckpt_dir", os.path.join(tmp, "flags"), "--ckpt_interval", "2"])
    same, worst = _same(ck("trunk", 1), ck("flags", 1))
    ok = r_f.returncode == 0 and r_c.returncode == 0 and same
    _check(failures, ok, "--config (data_root, relative streams) == flags (absolute paths), BITWISE", f"max abs {worst:.1e}")
    if r_f.returncode:
        return

    # B. a branch on the same data continues the trunk's stream: == the uninterrupted trunk, BITWISE;
    #    its evaluation runs at the continued step
    r = _run(["--config", trunk_cfg, "--steps", "4", "--ckpt_dir", os.path.join(tmp, "branch"), "--ckpt_interval", "2",
              "--continue_from", ck("trunk", 1)])
    same, worst = _same(ck("trunk", 3), ck("branch", 3))
    ok = (r.returncode == 0 and same and "continuing" in r.stdout and "the continued run's stream, read on" in r.stdout
          and len(_lines(r, "[eval      2] loss")) == 1 and not _lines(r, "[eval      0]"))
    _check(failures, ok, "continue_from on the same data == the trunk continued, BITWISE; eval at the start step",
           f"max abs {worst:.1e}")

    # C. the next stage on another length: a new stream from its first batch (data offset = the start step),
    #    WSD decaying across it, the last step saved although 8 % 3 != 0
    stage = ["--config", trunk_cfg, "--seq_len", str(2 * T), "--ckpt_interval", "3", "--continue_from", ck("trunk", 3),
             "--decay_start", "4", "--decay_end", "8", "--end_lr", "1e-4"]
    r_s = _run(stage + ["--steps", "8", "--ckpt_dir", os.path.join(tmp, "stage")])
    st = json.load(open(os.path.join(ck("stage", 7), "state.json"))) if r_s.returncode == 0 else {}
    lr6 = [l for l in _lines(r_s, "[      6]") if "lr 5.50e-04" in l]
    ok = (r_s.returncode == 0 and "a new stream" in r_s.stdout and "its first batch at step 4" in r_s.stdout
          and st.get("data_offset") == 4 and os.path.exists(os.path.join(ck("stage", 5), "DONE")) and len(lr6) == 1)
    _check(failures, ok, "continue_from on a new length: new stream at step 4, offset saved, WSD lr, last step saved",
           f"offset {st.get('data_offset')}")

    # D. the same stage stopped after step 5 and started again with the same command: it resumes ITS OWN
    #    checkpoint (not the continued run's), restores the data offset and ends BITWISE equal to C
    r1 = _run(stage + ["--steps", "6", "--ckpt_dir", os.path.join(tmp, "stage_stop")])
    r2 = _run(stage + ["--steps", "8", "--ckpt_dir", os.path.join(tmp, "stage_stop")])
    same, worst = _same(ck("stage", 7), ck("stage_stop", 7))
    ok = r1.returncode == 0 and r2.returncode == 0 and "resumed" in r2.stdout and "continuing" not in r2.stdout and same
    _check(failures, ok, "interrupted stage resumes its own checkpoint, == uninterrupted, BITWISE", f"max abs {worst:.1e}")

    # E. the method switched on at a branch: normalised write + self-prior from the trunk's checkpoint
    r = _run(["--config", trunk_cfg, "--steps", "6", "--ckpt_dir", os.path.join(tmp, "method"), "--ckpt_interval", "2",
              "--continue_from", ck("trunk", 3), "--inner_norm_alpha", "1", "--inner_norm_cap", "128",
              "--inner_lr", "0.82", "--self_prior", "0.1"])
    l5 = _lines(r, "[      5]")
    ok = r.returncode == 0 and len(l5) == 1 and "write[" in l5[0] and "nan" not in l5[0]
    _check(failures, ok, "the method's write switched on at a branch trains", l5[0][:80] if l5 else "")

    # F'. the same data at another step batch is refused: the new stream would start again at the first sample
    r = _run(["--config", trunk_cfg, "--steps", "6", "--ckpt_dir", os.path.join(tmp, "batch16"),
              "--continue_from", ck("trunk", 3), "--batch_size", str(2 * B)])
    ok = r.returncode != 0 and "another step batch" in (r.stderr + r.stdout)
    _check(failures, ok, "continue_from on the same data at another step batch is refused")

    # F. a continue_from that is not a complete checkpoint is refused before training
    r = _run(["--config", trunk_cfg, "--steps", "6", "--ckpt_dir", os.path.join(tmp, "bad"),
              "--continue_from", os.path.join(tmp, "trunk", "step_00000002")])
    ok = r.returncode != 0 and "not a complete checkpoint" in (r.stderr + r.stdout)
    _check(failures, ok, "continue_from without a DONE marker is refused")


def main():
    failures = []
    tmp = tempfile.mkdtemp()
    unit(failures, tmp)
    training(failures, tmp)
    shutil.rmtree(tmp, ignore_errors=True)
    print("ALL PASS" if not failures else "FAILURES: " + "; ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
