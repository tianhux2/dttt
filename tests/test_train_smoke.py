# coding=utf-8
"""
End-to-end on CPU: train 6 steps; separately train 3, checkpoint, resume for 3 more.
The resumed run's parameters must equal the uninterrupted run's BITWISE - one
assertion that covers data order, optimiser state, step counter and the f32
checkpoint round-trip at once.  Also: init_from starts at step 0 with fresh
optimiser; a changed data config refuses to resume; --probe_memory reports and exits.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

os.environ.setdefault("JAX_PLATFORMS", "cpu")
# 8 emulated devices; XLA:CPU's concurrency-optimised scheduler intermittently stalls a collective
# of the 4-view sequential program (the 40 s rendezvous watchdog then aborts it; seen at e8dfd15
# too, 4 of 5 runs on a loaded host; 3 of 3 pass without that scheduler)
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


def _run(args, extra_env=None):
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src"), **(extra_env or {}))
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


def main():
    failures = []
    tmp = tempfile.mkdtemp()
    cfg_dir = os.path.join(tmp, "cfg"); os.makedirs(cfg_dir)
    json.dump(dict(vocab_size=V, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                   num_attention_heads=4, num_key_value_heads=2, head_dim=8, suffix_len=1,
                   chunk_size=8, window_chunks=1, mask_token_id=V - 1, pad_token_id=0, eos_token_id=0),
              open(os.path.join(cfg_dir, "config.json"), "w"))
    src_a = _synthetic_source(os.path.join(tmp, "a"), 4000, 1)
    src_b = _synthetic_source(os.path.join(tmp, "b"), 4000, 2)
    common = ["--model_config", cfg_dir, "--sources", f"a={src_a}:0.6,b={src_b}:0.4",
              "--seq_len", str(T), "--batch_size", str(B), "--lr", "1e-3", "--warmup_steps", "2",
              "--inner_lr_warmup_steps", "2", "--fused_ce_block", "32", "--log_interval", "1",
              "--attn_chunk_batch", "2", "--cart_window", "16", "--seed", "7", "--holdout", str(16 * T)]

    # A: 6 uninterrupted steps, checkpoints at steps 2 and 5.
    ck_a = os.path.join(tmp, "ck_a")
    r = r_a = _run(common + ["--steps", "6", "--ckpt_dir", ck_a, "--ckpt_interval", "3"])
    ok = r.returncode == 0 and os.path.exists(os.path.join(ck_a, "step_00000005", "DONE"))
    failures += [] if ok else ["run A failed"]
    losses = [l for l in r.stdout.splitlines() if l.startswith("[") and not l.startswith("[eval") and "] loss " in l]
    print(f"[{'PASS' if ok else 'FAIL'}] run A: 6 steps, 2 checkpoints   {losses[0][:60] if losses else ''}")
    if not ok:
        print("FAILURES: " + "; ".join(failures)); return 1

    # evaluation: baseline at step 0 and at every checkpoint, finite, with the no-TTT twin.
    evals = [l for l in r.stdout.splitlines() if l.startswith("[eval") and " no-ttt " in l]
    detail = [l for l in r.stdout.splitlines() if l.startswith("[eval") and "gain by chunk " in l and "per source" not in l]
    ok = (len(evals) == 3 and len(detail) == 3 and all("gain" in l and "(se " in l and "nan" not in l for l in evals)
          and all(" a " in l and "±" in l for l in detail))
    failures += [] if ok else [f"eval lines: {evals} {detail}"]
    print(f"[{'PASS' if ok else 'FAIL'}] eval at 0 + every checkpoint      {evals[0][:70] if evals else ''}")

    # loss at step 0 ~ ln(V) under uniform weighting is checked in Tier 2; here: finite & moving.
    l0 = float(losses[0].split("loss")[1].split()[0]); l5 = float(losses[-1].split("loss")[1].split()[0])
    ok = np.isfinite(l0) and np.isfinite(l5) and l0 != l5
    failures += [] if ok else [f"loss not moving: {l0} -> {l5}"]
    print(f"[{'PASS' if ok else 'FAIL'}] loss finite and moving          {l0:.4f} -> {l5:.4f}")

    # B: resume from step 2, run to 6 -> must land BITWISE on A's step-5 weights.
    ck_b = os.path.join(tmp, "ck_b")
    r = _run(common + ["--steps", "6", "--ckpt_dir", ck_b, "--ckpt_interval", "3",
                       "--resume", os.path.join(ck_a, "step_00000002")])
    ok = r.returncode == 0 and "resumed" in r.stdout and os.path.exists(os.path.join(ck_b, "step_00000005", "DONE"))
    if ok:
        la, lb = _load(os.path.join(ck_a, "step_00000005")), _load(os.path.join(ck_b, "step_00000005"))
        ok = all(np.array_equal(x, y) for x, y in zip(la, lb))
        worst = max(float(np.max(np.abs(x - y))) for x, y in zip(la, lb))
    else:
        worst = float("nan")
    failures += [] if ok else [f"resumed run != uninterrupted run (max abs {worst})"]
    print(f"[{'PASS' if ok else 'FAIL'}] resume == uninterrupted, BITWISE  max abs = {worst:.1e}")

    # C: init_from -> step 0, and a changed data config refuses --resume.
    r = _run(common + ["--steps", "1", "--ckpt_dir", os.path.join(tmp, "ck_c"), "--ckpt_interval", "0",
                       "--init_from", os.path.join(ck_a, "step_00000005")])
    ok = r.returncode == 0 and "fresh optimiser" in r.stdout and "[      0]" in r.stdout
    failures += [] if ok else ["init_from did not start at step 0 with fresh optimiser"]
    print(f"[{'PASS' if ok else 'FAIL'}] init_from: weights only, step 0")
    r = _run(common[:3] + [f"a={src_a}:0.5,b={src_b}:0.5"] + common[4:] +
             ["--steps", "6", "--ckpt_dir", os.path.join(tmp, "ck_d"), "--resume", os.path.join(ck_a, "step_00000002")])
    ok = r.returncode != 0 and "data configuration changed" in (r.stderr + r.stdout)
    failures += [] if ok else ["resume with changed data mix was not refused"]
    print(f"[{'PASS' if ok else 'FAIL'}] resume refuses a changed data mix")

    # D: --keep_checkpoints prunes: 6 steps, interval 2 -> steps 1,3,5 written, only 3 and 5 kept.
    ck_f = os.path.join(tmp, "ck_f")
    r = _run(common + ["--steps", "6", "--ckpt_dir", ck_f, "--ckpt_interval", "2", "--keep_checkpoints", "2"])
    kept = sorted(d for d in os.listdir(ck_f)) if r.returncode == 0 and os.path.isdir(ck_f) else []
    ok = kept == ["step_00000003", "step_00000005"]
    failures += [] if ok else [f"keep_checkpoints kept {kept}"]
    print(f"[{'PASS' if ok else 'FAIL'}] keep_checkpoints prunes older     {kept}")

    # F: the segmented scan through the CLI (2 segments, truncated): runs, and the
    #    step-0 loss (a forward-only quantity) equals run A's.
    r = _run(common + ["--steps", "1", "--ckpt_dir", os.path.join(tmp, "ck_g"), "--ckpt_interval", "0",
                       "--segment_chunks", str(T // 8 // 2), "--truncate_fast_weight", "true"])
    first = lambda out: next((ln for ln in out.splitlines() if ln.startswith("[      0]")), "")
    la, lg = first(r_a.stdout), first(r.stdout)
    ok = r.returncode == 0 and lg and la.split("loss")[1].split()[0] == lg.split("loss")[1].split()[0]
    failures += [] if ok else [f"segmented run: rc={r.returncode} '{lg}' vs '{la}'"]
    print(f"[{'PASS' if ok else 'FAIL'}] segment_chunks via CLI, step-0 loss == whole")

    # G: write options through the CLI, and --init_from into a model that has the
    #    learned gate + decay the checkpoint was saved without (lenient load).
    cfg_g = os.path.join(tmp, "cfg_gate"); os.makedirs(cfg_g)
    json.dump({**json.load(open(os.path.join(cfg_dir, "config.json"))), "inner_lr_gate": True, "inner_decay": 0.02,
               "inner_optimizer": True},
              open(os.path.join(cfg_g, "config.json"), "w"))
    r = _run(["--model_config", cfg_g] + common[2:] +
             ["--steps", "2", "--ckpt_dir", os.path.join(tmp, "ck_h"), "--ckpt_interval", "0",
              "--init_from", os.path.join(ck_a, "step_00000005"), "--inner_views", "2",
              "--inner_update", "muon2", "--inner_mask_span", "4"])
    ok = r.returncode == 0 and "keeping its initialisation" in r.stdout and "[      1]" in r.stdout
    failures += [] if ok else [f"write options / lenient init_from: rc={r.returncode} {r.stderr[-300:]}"]
    print(f"[{'PASS' if ok else 'FAIL'}] views 2 + muon2 + span via CLI; init_from without gate/decay")

    # H: --no_write baseline (no inner stream) and the eval-time inner-lr sweep.
    r = _run(common + ["--steps", "1", "--ckpt_dir", os.path.join(tmp, "ck_i"), "--ckpt_interval", "0",
                       "--no_write", "true"])
    ok_nw = r.returncode == 0 and "[      0]" in r.stdout and "drift 0.00e+00" in r.stdout \
        and all("gain +0.0000" in l for l in r.stdout.splitlines() if l.startswith("[eval") and " no-ttt " in l)
    r = _run(common + ["--steps", "1", "--ckpt_dir", os.path.join(tmp, "ck_j"), "--ckpt_interval", "0",
                       "--eval_inner_lr_scales", "0.5,1,2"])
    sweep = [l for l in r.stdout.splitlines() if l.startswith("[eval") and "lr scale" in l]
    ok = ok_nw and r.returncode == 0 and len(sweep) == 1 and "x0.5" in sweep[0] and "x2" in sweep[0]
    failures += [] if ok else [f"no_write / lr sweep: {ok_nw} {sweep} {r.stderr[-300:]}"]
    print(f"[{'PASS' if ok else 'FAIL'}] --no_write baseline; --eval_inner_lr_scales sweep")

    # I: evaluation probes (per-token CE, recurrence and rarity classes) on a planted source:
    #    documents with a phrase repeated far apart must yield a non-empty "beyond" class.
    src_p = os.path.join(tmp, "planted"); os.makedirs(src_p)
    rng = np.random.default_rng(11)
    parts = []
    for d in range(12):
        doc = rng.integers(1, 100, size=3000, dtype=np.uint32)
        for s in range(40):
            phrase = rng.integers(1, 100, size=5, dtype=np.uint32)
            doc[s * 64 : s * 64 + 5] = phrase
            doc[s * 64 + 30 : s * 64 + 35] = phrase
        parts.append(np.concatenate([doc, [0]]))
    toks_p = np.concatenate(parts).astype("<u4"); toks_p.tofile(os.path.join(src_p, "tokens.u32"))
    json.dump({"name": "planted", "num_tokens": int(len(toks_p)), "num_docs": 12, "sep": "eos",
               "tokenizer": {"class": "synthetic", "len": V, "vocab_size": V, "eos_token_id": 0}},
              open(os.path.join(src_p, "meta.json"), "w"))
    uni = os.path.join(tmp, "unigram.i64"); np.bincount(toks_p.astype(np.int64), minlength=V).astype(np.int64).tofile(uni)
    r = _run(["--model_config", cfg_dir, "--sources", f"p={src_p}:1.0", "--seq_len", str(T), "--batch_size", str(B),
              "--steps", "1", "--ckpt_dir", os.path.join(tmp, "ck_k"), "--ckpt_interval", "0", "--holdout", "4000",
              "--fused_ce_block", "32", "--cart_window", "16", "--attn_chunk_batch", "2", "--seed", "7",
              "--eval_probes", "true", "--unigram_path", uni, "--wandb_project=",
              "--eval_sources", f"ood={src_b}:1.0"])                   # an evaluation-only source
    p1 = [l for l in r.stdout.splitlines() if "probe P1 pooled" in l]
    ood = [l for l in r.stdout.splitlines() if l.startswith("[eval") and " no-ttt " in l and " ood " in l]
    p2 = [l for l in r.stdout.splitlines() if "probe P2 pooled" in l]
    import re as _re
    n_beyond = int(_re.search(r"beyond n=(\d+)", p1[0]).group(1)) if p1 else 0
    ok = (r.returncode == 0 and len(p1) >= 1 and len(p2) >= 1 and n_beyond >= 3 and "loss nan" not in p1[0] + p2[0]
          and len(ood) >= 1)
    failures += [] if ok else [f"probes: rc={r.returncode} {p1[:1]} {p2[:1]} ood={len(ood)} {r.stderr[-400:]}"]
    print(f"[{'PASS' if ok else 'FAIL'}] eval probes P1/P2 + eval-only source   {p1[0][:100] if p1 else ''}")

    # J: host-side inner masks through the CLI (docaware with the planted unigram table).
    r = _run(common + ["--steps", "2", "--ckpt_dir", os.path.join(tmp, "ck_l"), "--ckpt_interval", "0",
                       "--inner_mask_mode", "docaware", "--unigram_path", uni])
    ok = r.returncode == 0 and "[      1]" in r.stdout and "loss nan" not in " ".join(l for l in r.stdout.splitlines() if l.startswith("[      1]"))
    failures += [] if ok else [f"inner_mask_mode docaware: rc={r.returncode} {r.stderr[-300:]}"]
    print(f"[{'PASS' if ok else 'FAIL'}] --inner_mask_mode docaware trains")

    # K: all inner tasks through the CLI: causal and split train; causal without the inner
    # stream (inner_from_outer) trains too.
    for task in ("causal", "split", "from_outer"):
        extra = ["--inner_from_outer", "true"] if task == "from_outer" else ["--inner_task", task]
        r = _run(common + ["--steps", "2", "--ckpt_dir", os.path.join(tmp, f"ck_{task}"), "--ckpt_interval", "0"] + extra)
        l1 = [l for l in r.stdout.splitlines() if l.startswith("[      1]")]
        ok = r.returncode == 0 and len(l1) == 1 and "loss nan" not in l1[0]
        failures += [] if ok else [f"inner_task {task}: rc={r.returncode} {l1[:1]} {r.stderr[-300:]}"]
    print(f"[{'PASS' if ok else 'FAIL'}] --inner_task causal / split / --inner_from_outer train   causal: {l1[0][:60] if l1 else ''}")

    # L: grad_accum > 1: micro-batches loop inside train_step, batch_size is the PER-DEVICE
    # micro-batch, masks stay correctly sharded across micro-batches and devices - including
    # the split task's host-side perms (stacked [N, B, T]); the tokens/s line must count
    # both micro-batches.  The data stream is indexed in micro-batches, so a resumed run
    # continues at step*N: the two checkpointed halves must equal the uninterrupted run.
    ok = True; info = []
    for extra in ([], ["--inner_task", "split", "--inner_mask_mode", "docaware", "--unigram_path", uni]):
        r = _run(common + ["--steps", "2", "--ckpt_dir", os.path.join(tmp, f"ck_ga{len(extra)}"), "--ckpt_interval", "0",
                           "--grad_accum", "2"] + extra)
        l1 = [l for l in r.stdout.splitlines() if l.startswith("[      1]")]
        ok &= r.returncode == 0 and len(l1) == 1 and "loss nan" not in l1[0] and "x 2 micro-batches" in r.stdout
        info.append(l1[0][:70] if l1 else r.stderr[-300:])
    ck_g = os.path.join(tmp, "ck_g")
    r_full = _run(common + ["--steps", "4", "--ckpt_dir", ck_g, "--ckpt_interval", "2", "--grad_accum", "2"])
    lines_full = [l for l in r_full.stdout.splitlines() if l.startswith("[") and "] loss " in l and not l.startswith("[eval")]
    ck_h = os.path.join(tmp, "ck_h")
    r_a = _run(common + ["--steps", "2", "--ckpt_dir", ck_h, "--ckpt_interval", "2", "--grad_accum", "2"])
    r_b = _run(common + ["--steps", "4", "--ckpt_dir", ck_h, "--ckpt_interval", "2", "--grad_accum", "2", "--resume", "auto"])
    lines_b = [l for l in r_b.stdout.splitlines() if l.startswith("[") and "] loss " in l and not l.startswith("[eval")]
    same = (r_full.returncode == 0 and r_a.returncode == 0 and r_b.returncode == 0 and len(lines_full) == 4 and len(lines_b) == 2
            and [l.split("loss")[1].split()[0] for l in lines_full[2:]] == [l.split("loss")[1].split()[0] for l in lines_b])
    ok &= same
    failures += [] if ok else [f"grad_accum: {info} resume-equal={same} {(lines_full[2:], lines_b)}"]
    print(f"[{'PASS' if ok else 'FAIL'}] --grad_accum 2 trains (plain, split+docaware); resume continues at step*N   {info[0][:60]}")

    # M: multi-view sequential inner writes with stratified ratios and CART-weighted inner loss.
    r = _run(common + ["--steps", "2", "--ckpt_dir", os.path.join(tmp, "ck_mv"), "--ckpt_interval", "0",
                       "--inner_views", "4", "--inner_view_mode", "strata", "--inner_views_sequential", "true",
                       "--inner_cart", "true", "--inner_mask_range", "0.0", "1.0"])
    l1 = [l for l in r.stdout.splitlines() if l.startswith("[      1]")]
    ok = r.returncode == 0 and len(l1) == 1 and "nan" not in l1[0]
    failures += [] if ok else [f"multi-view sequential: rc={r.returncode} {r.stderr[-300:]}"]
    print(f"[{'PASS' if ok else 'FAIL'}] --inner_views 4 strata sequential + inner_cart trains   {l1[0][:60] if l1 else ''}")

    # N: re-read evaluation (eval-only): the reread program scores the tail after reading the
    # first chunk twice; the log line carries the paired benefit and its SE, and the probes
    # are reported for the re-read write too.
    r = _run(common + ["--steps", "1", "--ckpt_dir", os.path.join(tmp, "ck_rr"), "--ckpt_interval", "0",
                       "--init_from", os.path.join(ck_a, "step_00000005"), "--eval_only", "true",
                       "--eval_reread_chunks", "1", "--eval_probes", "true", "--unigram_path", uni])
    rr = [l for l in r.stdout.splitlines() if l.startswith("[eval") and "reread(n=1)" in l]
    rp = [l for l in r.stdout.splitlines() if "reread probe P1 pooled" in l]
    ok = r.returncode == 0 and len(rr) == 1 and "(se " in rr[0] and "nan" not in rr[0].split("|")[0] and len(rp) == 1
    failures += [] if ok else [f"eval_reread: rc={r.returncode} {rr[:1]} {r.stderr[-300:]}"]
    print(f"[{'PASS' if ok else 'FAIL'}] --eval_reread_chunks: paired benefit line + reread probes   {rr[0][:80] if rr else ''}")

    # O: feature-prediction inner task: a model config with the head loads a plain checkpoint
    # leniently (the head starts fresh), trains with causal + add, logs the feature term.
    cfg_f = os.path.join(tmp, "cfg_feat"); os.makedirs(cfg_f)
    json.dump({**json.load(open(os.path.join(cfg_dir, "config.json"))), "inner_feat_head": True},
              open(os.path.join(cfg_f, "config.json"), "w"))
    r = _run(["--model_config", cfg_f] + common[2:] + ["--steps", "2", "--ckpt_dir", os.path.join(tmp, "ck_feat"),
                                                       "--ckpt_interval", "0", "--init_from", os.path.join(ck_a, "step_00000005"),
                                                       "--inner_task", "causal", "--inner_feat", "add"])
    l1 = [l for l in r.stdout.splitlines() if l.startswith("[      1]")]
    ok = (r.returncode == 0 and len(l1) == 1 and "nan" not in l1[0] and " feat " in l1[0]
          and "has no inner_feat_head" in r.stdout)
    failures += [] if ok else [f"inner_feat: rc={r.returncode} {l1[:1]} {r.stderr[-300:]}"]
    print(f"[{'PASS' if ok else 'FAIL'}] --inner_feat add (causal) trains from a plain checkpoint   {l1[0][:70] if l1 else ''}")

    # Q: the distillation write (a fourth prefix stream, no new parameters) trains and evaluates
    # from the plain config; the MTP head loads a plain checkpoint leniently and trains with the
    # shift objective; the log line carries both terms and the eval line is the main CE.
    r = _run(common + ["--steps", "2", "--ckpt_dir", os.path.join(tmp, "ck_dist"), "--ckpt_interval", "2",
                       "--inner_task", "causal", "--outer_shift", "true", "--distill", "true",
                       "--distill_topk", "8", "--eval_interval", "2"])
    l1 = [l for l in r.stdout.splitlines() if l.startswith("[      1]")]
    ev = [l for l in r.stdout.splitlines() if l.startswith("[eval      2] loss")]
    ok = (r.returncode == 0 and len(l1) == 1 and "nan" not in l1[0] and " distill " in l1[0]
          and len(ev) == 1 and "nan" not in ev[0])
    failures += [] if ok else [f"distill: rc={r.returncode} {l1[:1]} {ev[:1]} {r.stderr[-300:]}"]
    print(f"[{'PASS' if ok else 'FAIL'}] --distill true (causal + shift) trains and evaluates   {l1[0][:90] if l1 else ''}")
    cfg_m = os.path.join(tmp, "cfg_mtp"); os.makedirs(cfg_m)
    json.dump({**json.load(open(os.path.join(cfg_dir, "config.json"))), "mtp_head": True},
              open(os.path.join(cfg_m, "config.json"), "w"))
    r = _run(["--model_config", cfg_m] + common[2:] + ["--steps", "2", "--ckpt_dir", os.path.join(tmp, "ck_mtp"),
                                                       "--ckpt_interval", "0", "--init_from", os.path.join(ck_a, "step_00000005"),
                                                       "--inner_task", "causal", "--outer_shift", "true", "--mtp", "true",
                                                       "--distill", "true", "--distill_topk", "8", "--eval_interval", "2"])
    l1 = [l for l in r.stdout.splitlines() if l.startswith("[      1]")]
    ev = [l for l in r.stdout.splitlines() if l.startswith("[eval      2] loss")]
    ok = (r.returncode == 0 and len(l1) == 1 and "nan" not in l1[0] and " mtp " in l1[0] and " distill " in l1[0]
          and "has no mtp_head" in r.stdout and len(ev) == 1 and "nan" not in ev[0])
    failures += [] if ok else [f"mtp: rc={r.returncode} {l1[:1]} {ev[:1]} {r.stderr[-300:]}"]
    print(f"[{'PASS' if ok else 'FAIL'}] --mtp true + --distill true train from a plain checkpoint   {l1[0][:90] if l1 else ''}")
    r = _run(common + ["--steps", "1", "--ckpt_dir", os.path.join(tmp, "ck_mtp_bad"), "--ckpt_interval", "0",
                       "--mtp", "true"])
    ok = r.returncode != 0 and "mtp needs outer_shift" in (r.stderr + r.stdout)
    failures += [] if ok else ["--mtp without --outer_shift was accepted"]
    print(f"[{'PASS' if ok else 'FAIL'}] --mtp without --outer_shift / causal is refused")

    # R: the causal write's schemes through the CLI: sub-chunk + surprise + two-timescale split from
    # the plain config; the key-value-binding head loads a plain checkpoint leniently and trains with
    # clean values; both evaluate; the log line names the scheme.
    r = _run(common + ["--steps", "2", "--ckpt_dir", os.path.join(tmp, "ck_ws"), "--ckpt_interval", "0",
                       "--inner_task", "causal", "--outer_shift", "true", "--inner_subchunks", "2",
                       "--inner_surprise_gamma", "1.0", "--inner_fast_frac", "0.5", "--inner_fast_decay", "0.1",
                       "--eval_interval", "2"])
    l1 = [l for l in r.stdout.splitlines() if l.startswith("[      1]")]
    ev = [l for l in r.stdout.splitlines() if l.startswith("[eval      2] loss")]
    ok = r.returncode == 0 and len(l1) == 1 and "nan" not in l1[0] and "write[n2" in l1[0] and len(ev) == 1 and "nan" not in ev[0]
    failures += [] if ok else [f"write schemes: rc={r.returncode} {l1[:1]} {ev[:1]} {r.stderr[-300:]}"]
    print(f"[{'PASS' if ok else 'FAIL'}] --inner_subchunks 2 + surprise + fast split train and evaluate   {l1[0][:90] if l1 else ''}")
    cfg_k = os.path.join(tmp, "cfg_kvb"); os.makedirs(cfg_k)
    json.dump({**json.load(open(os.path.join(cfg_dir, "config.json"))), "inner_kvb_head": True},
              open(os.path.join(cfg_k, "config.json"), "w"))
    r = _run(["--model_config", cfg_k] + common[2:] + ["--steps", "2", "--ckpt_dir", os.path.join(tmp, "ck_kvb"),
                                                       "--ckpt_interval", "0", "--init_from", os.path.join(ck_a, "step_00000005"),
                                                       "--inner_task", "causal", "--outer_shift", "true",
                                                       "--inner_kvb_weight", "0.5", "--inner_kvb_value", "clean",
                                                       "--eval_interval", "2"])
    l1 = [l for l in r.stdout.splitlines() if l.startswith("[      1]")]
    ev = [l for l in r.stdout.splitlines() if l.startswith("[eval      2] loss")]
    ok = (r.returncode == 0 and len(l1) == 1 and "nan" not in l1[0] and " kvb " in l1[0]
          and "has no inner_kvb_value" in r.stdout and len(ev) == 1 and "nan" not in ev[0])
    failures += [] if ok else [f"kvb: rc={r.returncode} {l1[:1]} {ev[:1]} {r.stderr[-300:]}"]
    print(f"[{'PASS' if ok else 'FAIL'}] --inner_kvb_weight (clean values) trains from a plain checkpoint   {l1[0][:90] if l1 else ''}")
    r = _run(common + ["--steps", "1", "--ckpt_dir", os.path.join(tmp, "ck_ws_bad"), "--ckpt_interval", "0",
                       "--inner_subchunks", "2"])
    ok = r.returncode != 0 and "write schemes" in (r.stderr + r.stdout)
    failures += [] if ok else ["--inner_subchunks with the mask task was accepted"]
    print(f"[{'PASS' if ok else 'FAIL'}] write schemes with the mask task are refused")

    # S: planted recall - the training stream with planted spans trains, and the evaluation's probe
    # reports both masking modes (gain, ceiling, recovered share) and saves eval_<step>_planted.npz.
    logp = os.path.join(tmp, "log_planted")
    r = _run(common + ["--steps", "2", "--ckpt_dir", os.path.join(tmp, "ck_pl"), "--ckpt_interval", "0",
                       "--inner_task", "causal", "--outer_shift", "true", "--plant_frac", "0.5",
                       "--plant_len_min", "2", "--plant_len_max", "6", "--eval_interval", "2",
                       "--eval_planted_recall", "true", "--eval_planted_len", "6", "--eval_planted_cue", "2",
                       "--log_dir", logp])
    pl = [l for l in r.stdout.splitlines() if l.startswith("[eval      2] planted recall") and "recovered" in l]
    modes = {m for m in ("block", "alt") if any(f"planted recall {m} " in l for l in pl)}
    npz_ok = os.path.exists(os.path.join(logp, "eval_2_planted.npz"))
    ok = r.returncode == 0 and modes == {"block", "alt"} and npz_ok and not any("nan" in l for l in pl)
    failures += [] if ok else [f"planted: rc={r.returncode} {pl[:2]} npz {npz_ok} {r.stderr[-300:]}"]
    print(f"[{'PASS' if ok else 'FAIL'}] --plant_frac trains; --eval_planted_recall reports both modes + npz   {pl[0][:90] if pl else ''}")

    # P: a narrower fast weight loads a wider checkpoint by magnitude pruning and trains.
    cfg_n = os.path.join(tmp, "cfg_narrow"); os.makedirs(cfg_n)
    json.dump({**json.load(open(os.path.join(cfg_dir, "config.json"))), "ttt_intermediate_size": 16},
              open(os.path.join(cfg_n, "config.json"), "w"))
    r = _run(["--model_config", cfg_n] + common[2:] + ["--steps", "2", "--ckpt_dir", os.path.join(tmp, "ck_nar"),
                                                       "--ckpt_interval", "0", "--init_from", os.path.join(ck_a, "step_00000005")])
    l1 = [l for l in r.stdout.splitlines() if l.startswith("[      1]")]
    ok = r.returncode == 0 and len(l1) == 1 and "nan" not in l1[0] and "fast weight narrowed 64 -> 16" in r.stdout
    failures += [] if ok else [f"narrow init_from: rc={r.returncode} {l1[:1]} {r.stderr[-300:]}"]
    print(f"[{'PASS' if ok else 'FAIL'}] narrower fast weight: init_from prunes the checkpoint's and trains")

    # U: evaluation by step batches: 16 sequences x 1 and 8 x 2 micro-batches (8 devices) score the
    # same held-out sequences and report the same losses - a step batch's micro-batches combine as
    # one batch (sum CE / sum weight), not as a mean of their ratios, which would move with the split.
    # A holdout of 32 sequences, so that every source has a whole step batch of 16 to score.
    ev = {}
    for bs, ga in ((2 * B, 1), (B, 2)):
        r = _run(common + ["--steps", "1", "--ckpt_dir", os.path.join(tmp, f"ck_u{ga}"), "--ckpt_interval", "0",
                           "--batch_size", str(bs), "--grad_accum", str(ga), "--holdout", str(32 * T)])
        ev[ga] = [l for l in r.stdout.splitlines() if l.startswith("[eval      0] loss")] if r.returncode == 0 else []
    ok = len(ev[1]) == 1 and ev[1] == ev[2] and " a 0.00 " not in ev[1][0]
    failures += [] if ok else [f"evaluation by step batches: {ev}"]
    print(f"[{'PASS' if ok else 'FAIL'}] evaluation: 16 x 1 and 8 x 2 micro-batches report the same losses   "
          f"{(ev[1] or [''])[0][:60]}")

    # E: --probe_memory compiles, prints XLA's memory analysis and exits without training.
    r = _run(common + ["--steps", "3", "--ckpt_dir", os.path.join(tmp, "ck_e"), "--ckpt_interval", "0",
                       "--probe_memory", "true"])
    ok = r.returncode == 0 and "[probe] total per chip" in r.stdout and "] loss " not in r.stdout
    failures += [] if ok else ["probe_memory did not report"]
    print(f"[{'PASS' if ok else 'FAIL'}] probe_memory reports and exits   "
          f"{[l for l in r.stdout.splitlines() if l.startswith('[probe] total')][:1]}")

    shutil.rmtree(tmp, ignore_errors=True)
    print("ALL PASS" if not failures else "FAILURES: " + "; ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
