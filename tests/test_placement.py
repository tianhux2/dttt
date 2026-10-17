# coding=utf-8
"""
Fast weights in a chosen subset of the suffix layers (``ModelConfig.suffix_plain_top`` and
``ModelConfig.fast_layers``), CPU:
1. configuration checks, and the parameter shapes: the suffix stack keeps every suffix layer, the
   fast-weight stacks only the F fast-weight layers;
2. exact equivalence with a reference built from the default layout: every suffix layer with a fast
   weight, the plain positions with their post-norm scale set to zero.  Their memory term is then exactly
   zero whatever W, their inner gradients are zero (so the clip norm is unchanged), and the two models
   compute the same function, the same writes and the same outer loss:
   - the chunk-parallel forward with a fixed W (``Model.hidden``);
   - ``loss_fn`` with the causal write and the outer shift (the M2_shift recipe), second order: the
     loss and the gradient of every shared parameter; the reference's zeroed fast weights get none;
   both for plain layers above a fast block (suffix_plain_top) and for interleaved plain layers;
3. the meta-gradient reaches every fast layer's W_0 and every plain block;
4. end to end on the training command line (interleaved layout): 4 uninterrupted steps == 2 steps,
   checkpoint, resume for 2 more, bitwise; evaluation lines finite.
"""
from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import tempfile

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")
os.environ.setdefault("TTT_ATTENTION", "xla")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from src.ttt.chunk_scan import Batch, LossConfig, loss_fn
from src.ttt.config import ModelConfig
from src.ttt.model import Model

BASE = dict(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=6, num_attention_heads=4,
            num_key_value_heads=2, head_dim=8, suffix_len=4, chunk_size=8, window_chunks=1, mask_token_id=120,
            pad_token_id=0, eos_token_id=0)
LAYOUTS = {"plain top (fast 2-3, plain 4-5)": ModelConfig(**BASE, suffix_plain_top=2),
           "interleaved (fast 2, 4, 5, plain 3)": ModelConfig(**BASE, fast_layers=(2, 4, 5))}
CFG_R = ModelConfig(**BASE)                              # reference: fast weights in layers 2-5
failures = []


def report(name, ok, detail=""):
    failures.append(name) if not ok else None
    print(f"[{'PASS' if ok else 'FAIL'}] {name:<78} {detail}")


def rel(a, b):
    a, b = jnp.asarray(a), jnp.asarray(b)
    return float(jnp.max(jnp.abs(a - b)) / (jnp.max(jnp.abs(b)) + 1e-30))


def reference(model):
    """The default-layout model computing the same function: ``model``'s parameters, its fast weights at the
    fast positions, the other suffix positions with a fast weight whose post-norm scale is zero."""
    cfg = model.cfg
    pos = jnp.asarray([i for i, f in enumerate(cfg.fast_mask) if f])
    ref = Model(CFG_R, jax.random.PRNGKey(0))
    put = lambda r, p: r.at[pos].set(p) if eqx.is_array(r) else r
    fw = jax.tree_util.tree_map(put, ref.fast_weight, model.fast_weight)
    pn = jax.tree_util.tree_map(put, ref.prime_norm, model.prime_norm)
    ppn_w = jnp.zeros_like(ref.prime_post_norm.weight).at[pos].set(model.prime_post_norm.weight)
    ppn = eqx.tree_at(lambda n: n.weight, ref.prime_post_norm, ppn_w)
    ref = eqx.tree_at(lambda m: (m.fast_weight, m.prime_norm, m.prime_post_norm), ref, (fw, pn, ppn))
    return eqx.tree_at(lambda m: (m.embed, m.prefix, m.suffix, m.final_norm), ref,
                       (model.embed, model.prefix, model.suffix, model.final_norm)), pos


def check_layout(name, cfg):
    F, S = cfg.fast_len, cfg.suffix_len
    plain = [i for i, f in enumerate(cfg.fast_mask) if not f]
    model = Model(cfg, jax.random.PRNGKey(0))
    shapes_ok = (model.suffix.mlp.gate.weight.shape[0] == S and model.prefix.mlp.gate.weight.shape[0] == 6 - S
                 and all(x.shape[0] == F for x in jax.tree_util.tree_leaves(eqx.filter(
                     (model.fast_weight, model.prime_norm, model.prime_post_norm), eqx.is_array))))
    report(f"{name}: shapes (suffix [S], fast-weight stacks [F])", shapes_ok, f"S {S}, F {F}, runs {cfg.suffix_runs}")
    ref, pos = reference(model)
    NC = 4
    T = cfg.chunk_size * NC
    ids = jax.random.randint(jax.random.PRNGKey(1), (T,), 1, 100)
    h_p, h_r = model.hidden(ids), ref.hidden(ids)
    report(f"{name}: hidden() == reference", rel(h_p, h_r) < 1e-6, f"max rel {rel(h_p, h_r):.1e}")
    W_moved = jax.tree_util.tree_map(lambda w: w * 1.7 if eqx.is_array(w) else w, model.fast_weight)
    W_ref = jax.tree_util.tree_map(lambda r, p: r.at[pos].set(p) if eqx.is_array(r) else r, ref.fast_weight, W_moved)
    h_p2, h_r2 = model.hidden(ids, W=W_moved), ref.hidden(ids, W=W_ref)
    report(f"{name}: hidden(W) == reference with a moved fast weight, and it matters",
           rel(h_p2, h_r2) < 1e-6 and rel(h_p2, h_p) > 1e-4, f"max rel {rel(h_p2, h_r2):.1e}")

    B = 2
    ids_b = jax.random.randint(jax.random.PRNGKey(5), (B, T), 1, 100)
    lc = LossConfig(cart_window=16, inner_lr=0.5, inner_clip=1.0, inner_task="causal", outer_shift=True,
                    fused_ce_block=0, compute_dtype="float32", attn_chunk_batch=2)
    run = lambda m, c: eqx.filter_value_and_grad(
        lambda mm: loss_fn(mm, Batch(input_ids=ids_b), jax.random.PRNGKey(2), jnp.int32(3), c), has_aux=True)(m)
    (l_p, st_p), g_p = run(model, lc)
    (l_r, st_r), g_r = run(ref, lc)
    at = lambda t: jax.tree_util.tree_map(lambda x: x[pos], t)
    shared = [("embed", lambda g: g.embed, lambda g: g.embed), ("prefix", lambda g: g.prefix, lambda g: g.prefix),
              ("suffix blocks", lambda g: g.suffix, lambda g: g.suffix),
              ("final norm", lambda g: g.final_norm, lambda g: g.final_norm),
              ("fast weight W_0", lambda g: g.fast_weight, lambda g: at(g.fast_weight)),
              ("prime norm", lambda g: g.prime_norm, lambda g: at(g.prime_norm)),
              ("prime post-norm", lambda g: g.prime_post_norm, lambda g: at(g.prime_post_norm))]
    worst, names = 0.0, []
    for nm, fp, fr in shared:
        for a, b in zip(jax.tree_util.tree_leaves(eqx.filter(fp(g_p), eqx.is_array)),
                        jax.tree_util.tree_leaves(eqx.filter(fr(g_r), eqx.is_array))):
            r = rel(a, b)
            if r > 1e-5:
                names.append(nm)
            worst = max(worst, r)
    zero = max(float(jnp.max(jnp.abs(x[jnp.asarray(plain)]))) for x in
               jax.tree_util.tree_leaves(eqx.filter(g_r.fast_weight, eqx.is_array)))
    report(f"{name}: loss_fn (causal write, shift, 2nd order) loss == reference",
           abs(float(l_p) - float(l_r)) < 1e-6 * abs(float(l_r)), f"{float(l_p):.6f} vs {float(l_r):.6f}")
    report(f"{name}: every shared parameter's gradient == reference", worst < 1e-5, f"max rel {worst:.1e} {names}")
    report(f"{name}: the reference's zeroed fast weights get no gradient; same inner CE",
           zero == 0.0 and rel(st_p["inner_loss"], st_r["inner_loss"]) < 1e-6, f"max |g| {zero:.1e}")
    per_fast = [float(jnp.max(jnp.abs(x[i]))) for i in range(F) for x in
                jax.tree_util.tree_leaves(eqx.filter(g_p.fast_weight, eqx.is_array))]
    per_plain = [float(jnp.max(jnp.abs(x[i]))) for i in plain for x in
                 jax.tree_util.tree_leaves(eqx.filter(g_p.suffix, eqx.is_array))]
    (_, _), g_0 = run(model, dataclasses.replace(lc, inner_lr=0.0))
    d_write = rel(jax.tree_util.tree_leaves(eqx.filter(g_p.suffix, eqx.is_array))[0],
                  jax.tree_util.tree_leaves(eqx.filter(g_0.suffix, eqx.is_array))[0])
    report(f"{name}: meta-gradient reaches every W_0 layer and plain block; the write changes it",
           min(per_fast) > 0 and min(per_plain) > 0 and d_write > 1e-6,
           f"min |g| fast {min(per_fast):.1e}, plain {min(per_plain):.1e}, d(write) {d_write:.1e}")


def main():
    # 1. configuration
    for bad in (dict(suffix_plain_top=4), dict(suffix_plain_top=-1), dict(suffix_plain_top=1, memory="gdn"),
                dict(suffix_plain_top=1, inner_kvb_head=True), dict(fast_layers=(3, 4, 5)), dict(fast_layers=(2, 2, 5)),
                dict(fast_layers=(2, 6)), dict(fast_layers=(2, 4), suffix_plain_top=1),
                dict(fast_layers=(2, 4), memory="gdn")):
        try:
            ModelConfig(**BASE, **bad)
            report(f"config rejects {bad}", False)
        except ValueError:
            report(f"config rejects {bad}", True)
    c = ModelConfig(**BASE, fast_layers=[5, 2, 4])
    report("config: fast_layers from a JSON list -> sorted tuple, hashable; mask / runs / indices",
           c.fast_layers == (2, 4, 5) and hash(c) is not None and c.fast_mask == (True, False, True, True)
           and c.suffix_runs == ((True, 0, 1), (False, 1, 2), (True, 2, 4)) and c.fast_layer_indices == (2, 4, 5)
           and ModelConfig(**BASE).suffix_runs == ((True, 0, 4),))

    # 2-3. equivalence with the reference, meta-gradient
    for name, cfg in LAYOUTS.items():
        check_layout(name, cfg)

    # 4. the training command line (interleaved): uninterrupted == checkpoint + resume, bitwise
    tmp = tempfile.mkdtemp()
    cfg_dir = os.path.join(tmp, "cfg"); os.makedirs(cfg_dir)
    json.dump(dict(BASE, vocab_size=128, mask_token_id=127, fast_layers=[2, 4, 5]), open(os.path.join(cfg_dir, "config.json"), "w"))
    src = os.path.join(tmp, "a"); os.makedirs(src)
    np.random.default_rng(1).integers(1, 100, size=6000, dtype=np.uint32).astype("<u4").tofile(os.path.join(src, "tokens.u32"))
    json.dump({"name": "a", "num_tokens": 6000, "num_docs": 1, "tokenizer": {"class": "synthetic", "len": 128, "vocab_size": 128}},
              open(os.path.join(src, "meta.json"), "w"))
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src"),
               XLA_FLAGS="--xla_force_host_platform_device_count=8 --xla_cpu_enable_concurrency_optimized_scheduler=false")
    T = BASE["chunk_size"] * 4
    common = ["--model_config", cfg_dir, "--sources", f"a={src}:1.0", "--seq_len", str(T), "--batch_size", "8",
              "--lr", "1e-3", "--warmup_steps", "2", "--inner_lr_warmup_steps", "2", "--fused_ce_block", "32",
              "--log_interval", "1", "--attn_chunk_batch", "2", "--cart_window", "16", "--seed", "7",
              "--holdout", str(16 * T), "--inner_task", "causal", "--outer_shift", "true"]
    runp = lambda args: subprocess.run([sys.executable, "-m", "ttt.train", *common, *args], capture_output=True,
                                       text=True, env=env, timeout=1500)
    ra = runp(["--steps", "4", "--ckpt_dir", os.path.join(tmp, "ck_a"), "--ckpt_interval", "2"])
    rb = runp(["--steps", "2", "--ckpt_dir", os.path.join(tmp, "ck_b"), "--ckpt_interval", "2"])
    rc = runp(["--steps", "4", "--ckpt_dir", os.path.join(tmp, "ck_b"), "--ckpt_interval", "2", "--resume", "auto"]) \
        if rb.returncode == 0 else rb
    ok = ra.returncode == 0 and rc.returncode == 0
    if not ok:
        for r in (ra, rb, rc):
            print(r.stdout[-1500:]); print(r.stderr[-2500:])
    same = False
    if ok:
        from src.ttt.sharded_io import ShardedFileReader

        def leaves(d):
            skel = eqx.filter(Model(ModelConfig.load(os.path.join(d, "model_config.json")), jax.random.PRNGKey(0)), eqx.is_array)
            with ShardedFileReader(os.path.join(d, "model.eqx")) as f:
                return jax.tree_util.tree_leaves(eqx.tree_deserialise_leaves(f, jax.tree_util.tree_map(np.asarray, skel)))
        la, lb = leaves(os.path.join(tmp, "ck_a", "step_00000003")), leaves(os.path.join(tmp, "ck_b", "step_00000003"))
        same = len(la) == len(lb) and all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(la, lb))
    evals = [l for l in ra.stdout.splitlines() if l.startswith("[eval") and " no-ttt " in l]
    layout = [l for l in ra.stdout.splitlines() if "fast weights in layers [2, 4, 5], plain [3]" in l]
    report("train CLI (interleaved): 4 steps == 2 + checkpoint + resume 2, bitwise", ok and same)
    report("train CLI: layout logged, eval lines finite",
           bool(layout) and bool(evals) and all("nan" not in l for l in evals), evals[0][:80] if evals else "")

    print(f"\n{'ALL PASSED' if not failures else f'{len(failures)} FAILED: {failures}'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
