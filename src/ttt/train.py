# coding=utf-8
"""
Training entry point: ``python -m ttt.train --config runs/trunk_8k.json [--flag value ...]``

Every CLI flag is a ``TrainConfig`` field; ``--config`` reads them from a run configuration
(``config.load_run_config``) and the flags given replace its values.  The loop is deliberately small: build
or load, then ``for step: batch -> train_step -> log / checkpoint``.  Everything
that computes lives in ``chunk_scan.loss_fn``; everything that persists lives in
``checkpoint``; everything that reads data lives in ``data``.

Multi-host: ``jax.distributed.initialize()`` must be the FIRST JAX call.  Every
process runs this file; the sampler gives each host its slice of the global batch
by construction, so no cross-host data traffic exists.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import signal
import socket
import re
import sys
import time
from typing import Any, Optional, Tuple

# Must precede any JAX import that could initialise a backend.
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", os.path.expanduser("~/.jax_cache"))
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "10")

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding, PartitionSpec as P

from . import checkpoint as ckpt
from .chunk_scan import Batch, LossConfig, loss_fn
from .inner_loop import WriteConfig
from .config import ModelConfig, TrainConfig, load_run_config
from .data import MaskMaker, Plant, Prefetcher, Sampler, check_vocab, parse_sources, split_sources
from .model import Model, cast_floating
from .sharding import AXIS, batch_axis_name, constrain_replicated, make_mesh, param_shardings


# ----------------------------------------------------------------------------
# On-demand checkpoints and clean stops of a running job (no restart, no lost steps)
# ----------------------------------------------------------------------------
# Any of these, on ANY host of the job, takes effect at the end of the current step:
#   touch <trigger_dir>/TRIGGER_SAVE   -> a checkpoint is written (all processes), the file removed
#   touch <trigger_dir>/TRIGGER_STOP   -> the same, then the run exits cleanly (`--resume auto` continues it)
#   kill -USR1 <pid> / kill -USR2 <pid> -> the same two by signal, on the host of that process
# <trigger_dir> is $TTT_TRIGGER_DIR, else the repo's log/ directory (present on every pod host).
# The per-process flags are all-gathered once per step, so one host's file is enough.
# docs/launch.md, section 6.
TRIGGER_DIR = os.environ.get("TTT_TRIGGER_DIR") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "log")
TRIGGER_SAVE, TRIGGER_STOP = 1, 2
_signal_flag = [0]


def install_trigger_signals():
    def handler(signum, _frame):
        _signal_flag[0] = max(_signal_flag[0], TRIGGER_STOP if signum == signal.SIGUSR2 else TRIGGER_SAVE)
    for sig in (signal.SIGUSR1, signal.SIGUSR2):
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):        # not the main thread / unsupported platform
            pass


def poll_trigger() -> int:
    """0 (nothing), TRIGGER_SAVE or TRIGGER_STOP from this process's files and signals."""
    flag = _signal_flag[0]
    if os.path.exists(os.path.join(TRIGGER_DIR, "TRIGGER_STOP")):
        flag = TRIGGER_STOP
    elif os.path.exists(os.path.join(TRIGGER_DIR, "TRIGGER_SAVE")):
        flag = max(flag, TRIGGER_SAVE)
    return flag


def clear_trigger():
    _signal_flag[0] = 0
    for name in ("TRIGGER_SAVE", "TRIGGER_STOP"):
        try:
            os.remove(os.path.join(TRIGGER_DIR, name))
        except OSError:
            pass


# ----------------------------------------------------------------------------
# CLI: argparse generated from TrainConfig
# ----------------------------------------------------------------------------
def _json_value(default, v):
    """A run-config value in its field's type: JSON writes 1 for 1.0 and 2e5 for 200000."""
    if isinstance(default, float) and isinstance(v, int) and not isinstance(v, bool):
        return float(v)
    if isinstance(default, int) and not isinstance(default, bool) and isinstance(v, float) and v.is_integer():
        return int(v)
    return v


def parse_args(argv=None) -> TrainConfig:
    """The defaults, replaced by the run configuration of ``--config runs/x.json`` (``config.load_run_config``),
    replaced by the flags given."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="")
    run = pre.parse_known_args(argv)[0].config
    ap = argparse.ArgumentParser(prog="python -m ttt.train",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--config", default="", help="run configuration (JSON of these fields; 'base' inherits)")
    defaults = {}
    for f in dataclasses.fields(TrainConfig):
        defaults[f.name] = f.default
        kw: dict = {"default": f.default, "dest": f.name}
        if f.type is bool or f.default is False or f.default is True:
            ap.add_argument(f"--{f.name}", type=lambda s: s.lower() in ("1", "true", "yes"), **kw)
        elif f.name in ("inner_mask_range", "distill_t_range"):
            ap.add_argument(f"--{f.name}", type=type(f.default[0]), nargs=2, metavar=("LO", "HI"), **kw)
        else:
            ap.add_argument(f"--{f.name}", type=type(f.default), **kw)
    if run:
        ap.set_defaults(**{k: _json_value(defaults[k], v) for k, v in load_run_config(run).items()})
    d = vars(ap.parse_args(argv))
    d.pop("config")
    for name in ("inner_mask_range", "distill_t_range"):
        d[name] = tuple(d[name])
    return TrainConfig(**d)


# ----------------------------------------------------------------------------
# optimiser
# ----------------------------------------------------------------------------
def lr_schedule(cfg: TrainConfig):
    """Warmup-stable-decay over GLOBAL steps (a branch or the next stage continues the count): 0 -> lr over
    ``warmup_steps``, flat, then linearly to ``end_lr`` from ``decay_start`` to ``decay_end``, flat after it
    (``decay_start`` -1: flat to the end)."""
    warm = optax.linear_schedule(0.0, cfg.lr, max(1, cfg.warmup_steps))
    if cfg.decay_start < 0:
        return warm
    return optax.join_schedules([warm, optax.linear_schedule(cfg.lr, cfg.end_lr, cfg.decay_end - cfg.decay_start)],
                                [cfg.decay_start])


def make_optimizer(cfg: TrainConfig):
    sched = lr_schedule(cfg)
    no_decay = re.compile(cfg.no_decay_pattern) if cfg.no_decay_pattern else None

    # A CALLABLE mask: the partitioned param tree has None at non-array leaves, and a
    # prebuilt mask tree is the easiest thing here to get structurally wrong.
    def decay_mask(params):
        return jax.tree_util.tree_map_with_path(
            lambda p, x: eqx.is_array(x) and not (no_decay and no_decay.search(jax.tree_util.keystr(p))),
            params)

    chain = [optax.clip_by_global_norm(cfg.grad_clip),
             optax.adamw(sched, b1=cfg.adam_b1, b2=cfg.adam_b2, weight_decay=cfg.weight_decay, mask=decay_mask)]
    if cfg.fast_weight_lr_scale != 1.0:
        # The W_0 template's whole update scaled, as if its lr were: its gradient is mostly noise (B_simple ~1200
        # sequences against ~66 for the model).  Only when asked: the default optimiser state keeps its structure.
        fast = re.compile(r"(^|\.)fast_weight(\.|$)")
        chain.append(optax.masked(optax.scale(cfg.fast_weight_lr_scale), lambda params: jax.tree_util.tree_map_with_path(
            lambda p, x: eqx.is_array(x) and bool(fast.search(jax.tree_util.keystr(p))), params)))
    return optax.chain(*chain), sched


def check_resume_data(d: str, st: dict, sampler: Sampler, cfg: TrainConfig, log=print) -> None:
    """Refuse to resume the checkpoint at ``d`` on another training stream.

    ``data_hash_version`` 2 (written since 2026-10) is path-free and counts the sequences per
    optimiser step.  Version 1 hashed every stream's absolute path and the micro-batch size, so
    a copy of the data elsewhere (or 16 x 4 for 64 x 1) never matched: it is recomputed in its
    own form from this run's streams - their token counts and window indexes - with the
    checkpoint's recorded stream paths, relative ones joined to ``--resume_legacy_root`` (the
    directory that run started in; default: this one).  Equal means the same names, weights,
    token counts, windows, seed, length, holdout and planting; the sequences per step are
    compared directly."""
    want, version = st.get("data_hash"), st.get("data_hash_version", 1)
    if version >= 2:
        got = sampler.hash
    else:
        with open(os.path.join(d, "train_config.json")) as f:
            old = json.load(f)
        old_step = old["batch_size"] * old.get("grad_accum", 1)
        if old_step != sampler.step_batch:
            raise RuntimeError(f"checkpoint {d}: {old_step} sequences per step, this run {sampler.step_batch}")
        recorded = {name: path for name, path, _ in split_sources(old["sources"])}
        root = cfg.resume_legacy_root or os.getcwd()
        paths = [os.path.join(root, recorded.get(s.name, s.path)) for s in sampler.sources]
        got = sampler.legacy_hash(paths, old["batch_size"])
    if got != want:
        raise RuntimeError(f"data configuration changed since checkpoint ({want} != {got}, hash version "
                           f"{version}); use --init_from instead")
    log(f"[ttt] resume data check: hash {want} (version {version}) reproduced")


def write_config(cfg: TrainConfig) -> WriteConfig:
    """The causal write's scheme (``inner_loop.WriteConfig``) from the flat CLI flags."""
    return WriteConfig(subchunks=cfg.inner_subchunks, subchunk_scale=cfg.inner_subchunk_scale,
                       surprise_gamma=cfg.inner_surprise_gamma, kvb_weight=cfg.inner_kvb_weight,
                       kvb_value=cfg.inner_kvb_value, fast_frac=cfg.inner_fast_frac,
                       fast_lr_scale=cfg.inner_fast_lr_scale, fast_decay=cfg.inner_fast_decay,
                       steps=cfg.inner_write_steps, norm_alpha=cfg.inner_norm_alpha, norm_cap=cfg.inner_norm_cap)


def at_scale(lc: LossConfig, sc: float) -> LossConfig:
    """``lc`` with the inner lr scaled by ``sc`` (an evaluation twin); the no-write twin (0)
    takes one write step and no write plan whatever ``--inner_write_steps`` / ``--eval_write_plan``
    say - a zero step leaves W exactly unchanged, so more steps would only cost time."""
    lc = dataclasses.replace(lc, inner_lr=sc * lc.inner_lr)
    return dataclasses.replace(lc, write=dataclasses.replace(lc.write, steps=1), write_plan=()) if sc == 0 else lc


def write_plan(cfg: TrainConfig, nc: int) -> Tuple[int, ...]:
    """``--eval_write_plan "c:k,..."`` as whole-chunk write steps per chunk (``LossConfig.write_plan``;
    chunks not named keep one step)."""
    if not cfg.eval_write_plan:
        return ()
    plan = [1] * nc
    for x in cfg.eval_write_plan.split(","):
        if x.strip():
            c, k = (int(v) for v in x.split(":"))
            if c >= nc:
                raise ValueError(f"eval_write_plan: chunk {c} >= {nc} chunks per sequence")
            plan[c] = k
    return tuple(plan)


def loss_config(cfg: TrainConfig) -> LossConfig:
    return LossConfig(
        time_reweighting=cfg.time_reweighting, cart_p=cfg.cart_p, cart_window=cfg.cart_window,
        outer_t=cfg.outer_t, outer_shift=cfg.outer_shift,
        inner_lr=cfg.inner_lr, inner_clip=cfg.inner_clip, inner_lr_warmup_steps=cfg.inner_lr_warmup_steps,
        inner_mask_range=cfg.inner_mask_range, inner_mask_ratio=cfg.inner_mask_ratio,
        second_order=cfg.second_order, remat_levels=cfg.remat_levels, remat_prefix=cfg.remat_prefix,
        prefix_remat_levels=cfg.prefix_remat_levels, segment_chunks=cfg.segment_chunks,
        truncate_fast_weight=cfg.truncate_fast_weight, inner_views=cfg.inner_views,
        inner_view_mode=cfg.inner_view_mode, inner_views_sequential=cfg.inner_views_sequential, inner_cart=cfg.inner_cart,
        inner_mask_span=cfg.inner_mask_span, inner_update=cfg.inner_update, no_write=cfg.no_write,
        inner_task=cfg.inner_task, inner_context_frac=1.0 - cfg.inner_target_frac, inner_from_outer=cfg.inner_from_outer,
        inner_feat=cfg.inner_feat, inner_feat_weight=cfg.inner_feat_weight, inner_update_frac=cfg.inner_update_frac,
        distill=cfg.distill, distill_t_range=cfg.distill_t_range, distill_weight=cfg.distill_weight,
        distill_topk=cfg.distill_topk, distill_meta=cfg.distill_meta, distill_warmup_steps=cfg.distill_warmup_steps,
        mtp=cfg.mtp, mtp_weight=cfg.mtp_weight,
        inner_mtp_weight=cfg.inner_mtp_weight, write_off=cfg.write_off, aux_causal_weight=cfg.aux_causal_weight,
        write=write_config(cfg),
        window_drop=cfg.window_drop, window_drop_until=int(cfg.window_drop_frac * cfg.steps), self_prior=cfg.self_prior,
        outer_views=cfg.outer_views,
        fused_ce_block=cfg.fused_ce_block, attn_chunk_batch=cfg.attn_chunk_batch, compute_dtype="bfloat16")


# ----------------------------------------------------------------------------
# evaluation: held-out sequences, fixed masks, with and without the inner loop
# ----------------------------------------------------------------------------
def make_eval_step(lcfg: LossConfig, mesh, replicate_weights: str):
    @eqx.filter_jit
    def eval_step(static_model, params, batch: Batch, key, step):
        return loss_fn(compute_model(params, static_model, mesh, replicate_weights), batch, key, step, lcfg, mesh)
    return eval_step


def evaluate(eval_steps, static, params, groups, cfg, to_device, key, step, log,
             rank: int = 0, world: int = 1, mesh=None, is_main: bool = True, tag: str = ""):
    """Held-out loss per source (weighted total), the loss profile by chunk position,
    and the same with the inner loop disabled (``inner_lr = 0``) and at other
    multiples of the inner lr: ``eval_steps`` maps ``scale -> step function``; scale
    1 is the trained setting and scale 0 the no-write twin.  ``gain = loss(0) −
    loss(1)`` is the WRITE ABLATION (what the fast-weight updates buy this model),
    reported overall, per source and per chunk position, with the paired standard
    error over sequences.  Masks are fixed per batch, so numbers are comparable
    across checkpoints and runs.  ``tag`` (e.g. " t=1.0") marks a variant evaluation: it
    is appended to every ``[eval N`` line, inserted into the wandb keys (``eval t=1.0/...``)
    and the per-sequence file name (``eval_<step>_t1.0.npz``)."""
    if tag:
        log0, prefix = log, f"[eval {step:>6}]"
        log = lambda *a, **k: log0(*(x.replace(prefix, f"[eval {step:>6}{tag}]", 1) if isinstance(x, str) else x for x in a), **k)
    # keys: inner-lr scales (floats; 1 = trained, 0 = the no-write twin) and "reread"
    scales = sorted(eval_steps, key=lambda k: (isinstance(k, str), 0.0 if isinstance(k, str) else k))
    lab = lambda k: k if isinstance(k, str) else f"x{k:g}"
    z = 0.0 if 0.0 in eval_steps else 1.0                          # no-write twin (absent with --no_write)
    mcfg = ModelConfig.load(cfg.model_config)
    nc = cfg.seq_len // mcfg.chunk_size
    window_tokens = mcfg.window_chunks * mcfg.chunk_size
    out = {}
    tot = {k: 0.0 for k in scales}; chunk = {k: np.zeros(nc) for k in scales}; wsum = 0.0
    gains = []                                                     # per-sequence paired gains, all sources
    rr_gains = []                                                  # per-sequence paired re-read benefits
    probe_batches = []                                             # (source, batch, ce_on, ce_off, scored)
    probe_rr = []                                                  # the same with the re-read write on
    buckets = ((0, 1, "0"), (1, 1024, "1-1K"), (1024, 3072, "1-3K"), (3072, 1 << 62, ">=3K"))
    by_bucket = {b[2]: [] for b in buckets}                        # (loss, gain) per sequence, all sources
    src_bucket = {}                                                # (source, bucket) -> [(loss, gain)]
    sources = [src for _, srcs, _ in groups for src in srcs]      # all sources, training ones first
    pool_names = {src.name for _, srcs, weighted in groups if weighted for src in srcs}   # pooled probes: the mix
    probe_ctx = {}                                                 # source name -> (sampler, index)
    flat = [(smp, si, src, weighted) for smp, srcs, weighted in groups for si, src in enumerate(srcs)]
    all_eval_sources = []
    all_eval_doc_ids = []
    all_eval_doc_lens = []
    all_eval_beyond = []
    all_eval_losses = {sc: [] for sc in scales}
    all_eval_gains = []
    all_eval_masks = []
    for smp, si, src, weighted in flat:
        sampler = smp
        probe_ctx[src.name] = (smp, si)
        n_holdout = max(1, int(sampler.n_eval_seq[si]))
        perm = np.random.default_rng(cfg.eval_seed).permutation(n_holdout)
        n_steps = sampler.eval_step_batches(si)                   # batch_size x grad_accum sequences each
        n_b = sampler.accum * (n_steps if not weighted else min(cfg.eval_batches, n_steps))
        acc = {k: 0.0 for k in scales}; ch = {k: np.zeros(nc) for k in scales}; g_src = []; rr_src = []
        # A step batch's loss is that of its micro-batches taken as one batch - sum CE / sum weight,
        # as in training - so it does not depend on the split; then the mean over step batches.
        part = {k: [0.0, 0.0, np.zeros(nc), np.zeros(nc)] for k in scales}
        for k in range(n_b):
            j_start = k * sampler.G + sampler.rank * sampler.local
            j_indices = [(j_start + j) % n_holdout for j in range(sampler.local)]
            keys_j = jax.vmap(lambda j: jax.random.fold_in(jax.random.PRNGKey(cfg.eval_seed), j))(jnp.asarray(j_indices, dtype=jnp.int32))
            splits = jax.vmap(jax.random.split)(keys_j)
            keys_u = splits[:, 0, :]
            keys_row = splits[:, 1, :]
            u = jax.vmap(lambda k: jax.random.uniform(k, ()))(keys_u)
            ranks = jnp.asarray(perm[j_indices], dtype=jnp.float32)
            t_local = np.asarray((ranks + u) / float(n_holdout))

            b = to_device(sampler.eval_batch(si, k), outer_t=t_local)
            if mesh is not None:
                key_sharding = NamedSharding(mesh, P(batch_axis_name(mesh), None))
                ek = jax.make_array_from_process_local_data(key_sharding, np.asarray(keys_row), (cfg.batch_size, 2))
            else:
                ek = keys_row

            seq = {}; tok = {}
            for sc in scales:
                l, st = eval_steps[sc](static, params, b, ek, jnp.int32(step))
                pt, mc, mch = part[sc], float(st["mask_count"]), np.asarray(st["mask_by_chunk"], np.float64)
                pt[0] += float(l) * mc; pt[1] += mc
                pt[2] += np.asarray(st["loss_by_chunk"], np.float64) * mch; pt[3] += mch
                if (k + 1) % sampler.accum == 0:
                    acc[sc] += pt[0] / max(pt[1], 1e-6); ch[sc] += pt[2] / np.maximum(pt[3], 1e-6)
                    part[sc] = [0.0, 0.0, np.zeros(nc), np.zeros(nc)]
                if "loss_by_seq" in st:
                    seq[sc] = np.asarray(st["loss_by_seq"])
                    if sc == 1.0 and "mask_by_seq" in st:
                        all_eval_masks.append(np.asarray(st["mask_by_seq"]))
                if "ce_by_token" in st and sc in (z, 1.0, "reread"):
                    tok[sc] = (np.asarray(st["ce_by_token"]), np.asarray(st["mask_by_token"]) > 0.5)
            if cfg.eval_probes and z in tok and 1.0 in tok and z != 1.0:
                probe_batches.append((src.name, k, tok[1.0][0], tok[z][0], tok[1.0][1]))
                if "reread" in tok:
                    probe_rr.append((src.name, k, tok["reread"][0], tok[z][0], tok["reread"][1]))
            if "reread" in seq and 1.0 in seq:
                rr_src.append(seq[1.0] - seq["reread"])              # benefit of the second read, per sequence
            doc_len, beyond, doc_id = sampler.eval_doc_info(si, k, mcfg.eos_token_id, window_tokens)
            if z in seq and 1.0 in seq and z != 1.0:
                g_src.append(seq[z] - seq[1.0])
                for j in range(len(beyond)):
                    name = next(b[2] for b in buckets if b[0] <= beyond[j] < b[1])
                    rec = (float(seq[1.0][j]), float(seq[z][j] - seq[1.0][j]), int(beyond[j]))
                    by_bucket[name].append(rec); src_bucket.setdefault((src.name, name), []).append(rec)
            all_eval_sources.extend([src.name] * len(doc_id))
            all_eval_doc_ids.append(doc_id)
            all_eval_doc_lens.append(doc_len)
            all_eval_beyond.append(beyond)
            for sc in scales:
                if sc in seq:
                    all_eval_losses[sc].append(seq[sc])
            if z in seq and 1.0 in seq:
                all_eval_gains.append(seq[z] - seq[1.0])
        for sc in scales:
            acc[sc] /= max(1, n_b // sampler.accum); ch[sc] /= max(1, n_b // sampler.accum)
            if weighted:                                           # eval-only sources are reported, not averaged
                tot[sc] += src.weight * acc[sc]; chunk[sc] += src.weight * ch[sc]
        if weighted:
            wsum += src.weight
        out[f"eval/{src.name}/loss"] = acc[1.0]; out[f"eval/{src.name}/loss_nottt"] = acc[z]
        for sc in scales:
            if sc not in (z, 1.0):
                out[f"eval/{src.name}/loss@{lab(sc)}"] = acc[sc]
        out[f"eval/{src.name}/gain"] = acc[z] - acc[1.0]
        if rr_src:
            g = np.concatenate(rr_src)
            out[f"eval/{src.name}/reread_gain"] = acc[1.0] - acc["reread"]
            out[f"eval/{src.name}/reread_gain_se"] = float(g.std(ddof=1) / np.sqrt(len(g)))
            if weighted:
                rr_gains.append((src.weight, g))
        for i in range(nc):
            out[f"eval/{src.name}/gain_chunk{i}"] = float(ch[z][i] - ch[1.0][i])
        if g_src:
            g = np.concatenate(g_src)
            out[f"eval/{src.name}/gain_se"] = float(g.std(ddof=1) / np.sqrt(len(g)))
            if weighted:
                gains.append((src.weight, g))
    for sc in scales:
        tot[sc] /= wsum; chunk[sc] /= wsum
    out.update({"eval/loss": tot[1.0], "eval/loss_nottt": tot[z], "eval/ttt_gain": tot[z] - tot[1.0]})
    out.update({f"eval/chunk{i}": float(v) for i, v in enumerate(chunk[1.0])})
    out.update({f"eval/gain_chunk{i}": float(chunk[z][i] - chunk[1.0][i]) for i in range(nc)})
    for sc in scales:
        if sc not in (z, 1.0):
            out[f"eval/loss@{lab(sc)}"] = tot[sc]
    if rr_gains:
        var = sum((w / wsum) ** 2 * g.var(ddof=1) / len(g) for w, g in rr_gains)
        out["eval/reread_gain"] = tot[1.0] - tot["reread"]; out["eval/reread_gain_se"] = float(np.sqrt(var))
        out.update({f"eval/reread_chunk{i}": float(chunk[1.0][i] - chunk["reread"][i]) for i in range(nc)})
    if gains:
        # SE of the source-weighted mean gain: weights sum to wsum, sequences independent.
        var = sum((w / wsum) ** 2 * g.var(ddof=1) / len(g) for w, g in gains)
        out["eval/ttt_gain_se"] = float(np.sqrt(var))
    log(f"[eval {step:>6}] loss {tot[1.0]:.4f}  no-ttt {tot[z]:.4f}  gain {tot[z] - tot[1.0]:+.4f}"
        + (f" (se {out['eval/ttt_gain_se']:.4f})" if gains else "") + " | "
        f"by chunk " + " ".join(f"{v:.2f}" for v in chunk[1.0]) + " | "
        + " ".join(f"{src.name} {out[f'eval/{src.name}/loss']:.2f}" for src in sources))
    log(f"[eval {step:>6}] gain by chunk " + " ".join(f"{v:+.3f}" for v in chunk[z] - chunk[1.0]) + " | "
        + " ".join(f"{src.name} {out[f'eval/{src.name}/gain']:+.3f}"
                   + (f"±{out[f'eval/{src.name}/gain_se']:.3f}" if f"eval/{src.name}/gain_se" in out else "")
                   for src in sources)
        + (" | lr scale " + " ".join(f"{lab(sc)} {tot[sc]:.4f}" for sc in scales if sc not in (z, 1.0, "reread"))
           if len([sc for sc in scales if sc not in (z, 1.0, "reread")]) else ""))
    # Gain by chunk position per source: whether W accumulates out-of-window content
    # (chunks >= window) on the sources that have any.
    log(f"[eval {step:>6}] gain by chunk per source: " + " | ".join(
        f"{src.name} " + " ".join(f"{out[f'eval/{src.name}/gain_chunk{i}']:+.3f}" for i in range(nc)) for src in sources))
    if probe_batches:
        out.update(_probes(probe_batches, probe_ctx, sources, mcfg, cfg, step, log, pool=pool_names))
    if "reread" in eval_steps:
        # Re-read benefit: single pass minus re-read on the scored tail (paired), and the
        # write ablation of both (does the second write add to the first?).
        n = cfg.eval_reread_chunks
        log(f"[eval {step:>6}] reread(n={n}): loss {tot['reread']:.4f}  single {tot[1.0]:.4f}  "
            f"benefit {tot[1.0] - tot['reread']:+.4f}" + (f" (se {out['eval/reread_gain_se']:.4f})" if rr_gains else "")
            + f" | ablation single {tot[z] - tot[1.0]:+.4f} reread {tot[z] - tot['reread']:+.4f} | by chunk "
            + " ".join(f"{v:+.3f}" for v in chunk[1.0] - chunk["reread"]) + " | "
            + " ".join(f"{src.name} {out.get(f'eval/{src.name}/reread_gain', float('nan')):+.4f}"
                       + (f"±{out[f'eval/{src.name}/reread_gain_se']:.4f}" if f"eval/{src.name}/reread_gain_se" in out else "")
                       for src in sources))
        if probe_rr:
            out.update(_probes(probe_rr, probe_ctx, sources, mcfg, cfg, step, log, label="reread probe",
                               key_prefix="eval/probe_reread", pool=pool_names))
    # Length buckets: which documents the write helps (the guard against a mix that buys
    # write gain at the cost of general ability).
    if any(by_bucket.values()):
        def fmt(recs):
            a = np.array(recs)
            return f"n={len(a)} loss {a[:, 0].mean():.3f} gain {a[:, 1].mean():+.4f} beyond {a[:, 2].mean() / 1024:.1f}K"
        for name, recs in by_bucket.items():
            if recs:
                a = np.array(recs)
                out[f"eval/bucket/{name}/loss"] = float(a[:, 0].mean()); out[f"eval/bucket/{name}/gain"] = float(a[:, 1].mean())
                out[f"eval/bucket/{name}/n"] = len(a)
        log(f"[eval {step:>6}] by beyond window: " + " | ".join(f"{n}: {fmt(r)}" for n, r in by_bucket.items() if r))
        log(f"[eval {step:>6}] by source x beyond: " + " | ".join(
            f"{s_}/{b_} {np.array(r)[:, 1].mean():+.4f} (n={len(r)})" for (s_, b_), r in sorted(src_bucket.items())))
    # Persistence of per-sequence evaluations and document clusters
    eval_dir = cfg.log_dir or os.environ.get("TTT_LOG_DIR", "")
    if eval_dir and is_main:
        os.makedirs(eval_dir, exist_ok=True)
        save_dict = {
            "sources": np.array(all_eval_sources),
            "doc_ids": np.concatenate(all_eval_doc_ids) if all_eval_doc_ids else np.array([], dtype=np.int64),
            "doc_lens": np.concatenate(all_eval_doc_lens) if all_eval_doc_lens else np.array([], dtype=np.int64),
            "beyond": np.concatenate(all_eval_beyond) if all_eval_beyond else np.array([], dtype=np.int64),
        }
        for sc, l_list in all_eval_losses.items():
            if l_list:
                save_dict[f"loss_{sc}"] = np.concatenate(l_list)
        if all_eval_gains:
            save_dict["gain"] = np.concatenate(all_eval_gains)
        if all_eval_masks:
            save_dict["mask"] = np.concatenate(all_eval_masks)          # scored tokens per sequence (token-weighted means)
        np.savez_compressed(os.path.join(eval_dir, f"eval_{step}{tag.replace(' ', '_').replace('=', '')}.npz"), **save_dict)
    # Other inner-lr scales: the loss by chunk and per source, as gains against the trained scale.
    for sc in scales:
        if sc in (z, 1.0, "reread"):
            continue
        log(f"[eval {step:>6}] {lab(sc)} vs x1: by chunk "
            + " ".join(f"{v:+.3f}" for v in chunk[1.0] - chunk[sc]) + " | "
            + " ".join(f"{src.name} {out[f'eval/{src.name}/loss'] - out[f'eval/{src.name}/loss@{lab(sc)}']:+.3f}" for src in sources))
    if tag:
        out = {k.replace("eval/", f"eval{tag}/", 1): v for k, v in out.items()}
    return out


def evaluate_planted(planted_steps, static, params, groups, cfg, to_device, step, log, mesh=None,
                     is_main: bool = True) -> dict:
    """Planted-recall probe (``probes.planted_probe_batch``): in every held-out sequence a span of
    chunk 1 is copied into chunk ``window + 3``, beyond the attention window - and, as the ceiling,
    also into the chunk before it, inside the window, where attention can copy it - and the copy's
    tail after the cue is masked (``outer_t`` 1: nothing else is).  Mean CE of that tail with the
    memory on (scale 1) and off (scale 0), both placements, both masking modes: the memory's recall
    gain ``off - on`` beyond the window, the copy ceiling ``beyond off - within off`` (what attention
    gets from seeing the span) and the share of it recovered.  The same sequences and draws for every
    checkpoint (paired across runs; ``eval_<step>_planted.npz``)."""
    from . import probes as PR
    mcfg = ModelConfig.load(cfg.model_config)
    C, Wn = mcfg.chunk_size, mcfg.window_chunks
    scales = sorted(planted_steps)
    per, names = {}, []                       # (placement, mode, scale) -> [per-sequence mean CE]; source per sequence
    plan_ce = []                              # write plan: (before, after) own CE of the chunks written > once
    for smp, srcs, _ in groups:
        for si, src in enumerate(srcs):
            for k in range(smp.accum * min(cfg.eval_planted_batches, smp.eval_step_batches(si))):
                local = smp.eval_batch(si, k)
                j0 = k * smp.G + smp.rank * smp.local
                j_global = [j0 + j for j in range(smp.local)]
                keys = jax.vmap(lambda j: jax.random.fold_in(jax.random.PRNGKey(cfg.eval_seed), j))(
                    jnp.asarray(j_global, dtype=jnp.int32))
                ek = keys if mesh is None else jax.make_array_from_process_local_data(
                    NamedSharding(mesh, P(batch_axis_name(mesh), None)), np.asarray(keys), (cfg.batch_size, 2))
                names.extend([src.name] * smp.G)
                for pl in PR.PLANT_PLACEMENTS:
                    for mode in PR.PLANT_MODES:
                        ids, lm = PR.planted_probe_batch(local, j_global, cfg.eval_seed, C, Wn, cfg.eval_planted_len,
                                                         cfg.eval_planted_cue, pl, mode, eos=mcfg.eos_token_id)
                        b = to_device(ids, outer_t=np.ones(len(ids), np.float32), loss_mask=lm)
                        for sc in scales:
                            _, st = planted_steps[sc](static, params, b, ek, jnp.int32(step))
                            ce = np.asarray(st["ce_by_token"], np.float64)
                            m = np.asarray(st["mask_by_token"]) > 0.5
                            per.setdefault((pl, mode, sc), []).append((ce * m).sum(1) / np.maximum(m.sum(1), 1))
                            if "inner_plan_pre" in st and pl == "beyond" and mode == "alt" and sc == 1.0:
                                plan_ce.append((float(st["inner_plan_pre"]), float(st["inner_plan_post"])))
    out = {}
    if not per:
        return out
    if plan_ce:
        # what W holds of the chunks the plan writes more than once, read with the write's own keys
        pre, post = np.mean(plan_ce, axis=0)
        out.update({"eval/plan_own_ce_before": float(pre), "eval/plan_own_ce_after": float(post)})
        log(f"[eval {step:>6}] write plan {cfg.eval_write_plan}: own causal CE of the chunks written > once "
            f"{pre:.3f} -> {post:.3f} after their steps")
    arr = {kk: np.concatenate(v) for kk, v in per.items()}
    nm = np.array(names)
    for mode in PR.PLANT_MODES:
        b_on, w_on = arr[("beyond", mode, 1.0)], arr[("within", mode, 1.0)]
        line = (f"[eval {step:>6}] planted recall {mode} (n={len(b_on)}, span {cfg.eval_planted_len}, cue "
                f"{cfg.eval_planted_cue}): beyond on {b_on.mean():.3f}")
        out[f"eval/planted_{mode}_beyond"], out[f"eval/planted_{mode}_within"] = float(b_on.mean()), float(w_on.mean())
        if 0.0 in scales:
            b_off, w_off = arr[("beyond", mode, 0.0)], arr[("within", mode, 0.0)]
            g = b_off - b_on
            se = float(g.std(ddof=1) / np.sqrt(len(g))) if len(g) > 1 else float("nan")
            n_on, n_off = arr[("none", mode, 1.0)], arr[("none", mode, 0.0)]
            rec_g = g - (n_off - n_on)                     # recall = gain on the copy - gain on unplanted text
            rse = float(rec_g.std(ddof=1) / np.sqrt(len(rec_g))) if len(rec_g) > 1 else float("nan")
            out.update({f"eval/planted_{mode}_recall": float(rec_g.mean()), f"eval/planted_{mode}_recall_se": rse})
            ceil = float(b_off.mean() - w_off.mean())
            # the share is only meaningful when seeing the span helps attention at all (a model that
            # cannot copy has no ceiling to recover)
            rec = float(g.mean() / ceil) if ceil > 0.01 else float("nan")
            line += (f" off {b_off.mean():.3f} gain {g.mean():+.4f} (se {se:.4f}) | within on {w_on.mean():.3f} off "
                     f"{w_off.mean():.3f} | copy ceiling {ceil:+.3f} recovered "
                     + (f"{100 * rec:.1f}%" if np.isfinite(rec) else "n/a (no copy ceiling)")
                     + f" | unplanted gain {float((n_off - n_on).mean()):+.4f} -> recall {float(rec_g.mean()):+.4f} (se {rse:.4f})"
                     + (f" = {100 * float(rec_g.mean()) / ceil:.1f}% of ceiling" if ceil > 0.01 else ""))
            out.update({f"eval/planted_{mode}_gain": float(g.mean()), f"eval/planted_{mode}_gain_se": se,
                        f"eval/planted_{mode}_ceiling": ceil})
            if np.isfinite(rec):
                out[f"eval/planted_{mode}_recovered"] = rec
            log(line)
            log(f"[eval {step:>6}] planted recall {mode} gain by source: " + " | ".join(
                f"{s_} {g[nm == s_].mean():+.3f}" for s_ in dict.fromkeys(names)))
        else:
            log(line + f" | within on {w_on.mean():.3f}")
    eval_dir = cfg.log_dir or os.environ.get("TTT_LOG_DIR", "")
    if eval_dir and is_main:
        os.makedirs(eval_dir, exist_ok=True)
        np.savez_compressed(os.path.join(eval_dir, f"eval_{step}_planted.npz"), sources=nm,
                            **{f"{pl}_{mode}_x{sc:g}": v for (pl, mode, sc), v in arr.items()})
    return out


def _probes(batches, probe_ctx, sources, mcfg, cfg, step, log, label="probe", key_prefix="eval/probe", pool=None) -> dict:
    """P1 recurrence / P2 rarity classes of every scored held-out token (``probes.py``),
    aggregated per source and pooled, logged (``label``) and returned as wandb keys
    (``key_prefix``).  ``pool``: the source names that enter the pooled numbers (the training
    mix; evaluation-only sources are not capped by ``--eval_batches`` and would dominate a
    token-weighted pool - pg19_test alone was 46 % of the pooled 'beyond' class)."""
    from . import probes as P
    C, W = mcfg.chunk_size, mcfg.window_chunks
    nc = cfg.seq_len // C
    bins = None
    uni = cfg.data_path(cfg.unigram_path)
    if uni and os.path.exists(uni):
        counts = np.fromfile(uni, dtype=np.int64)
        if len(counts) >= mcfg.vocab_size:
            bins = P.rarity_bins(counts[:mcfg.vocab_size])
    out = {}
    pooled = {"P1": [], "P2": [], "P3": []}
    per_src = {}
    for sname, k, ce_on, ce_off, scored in batches:
        sampler, si = probe_ctx[sname]
        ctx = sampler.eval_context(si, k, mcfg.eos_token_id, lookback=131072)
        B, T = ce_on.shape
        lab1 = np.zeros((B, T), np.int8); lab2 = np.zeros((B, T), np.int8)
        toks = np.zeros((B, T), np.int64)
        for j, (c_tokens, off, dstarts) in enumerate(ctx[:B]):
            lab1[j] = P.recurrence_classes(c_tokens, off, T, dstarts, C, W)
            toks[j] = c_tokens[off:off + T]
        if bins is not None:
            lab2 = bins[np.clip(toks, 0, len(bins) - 1)]
        in_pool = pool is None or sname in pool
        a1 = P.aggregate(ce_on, ce_off, scored, lab1, len(P.P1_CLASSES), C)
        per_src.setdefault(("P1", sname), []).append(a1)
        if in_pool:
            pooled["P1"].append(a1)
        if bins is not None:
            a2 = P.aggregate(ce_on, ce_off, scored, lab2, len(P.P2_CLASSES), C)
            per_src.setdefault(("P2", sname), []).append(a2)
            if in_pool:
                pooled["P2"].append(a2)
        a3 = P.aggregate(ce_on, ce_off, scored, P.neighbour_classes(scored), len(P.P3_CLASSES), C)
        per_src.setdefault(("P3", sname), []).append(a3)
        if in_pool:
            pooled["P3"].append(a3)

    def merge(aggs):
        n = sum(a["n"] for a in aggs)
        loss = sum(a["loss"] * a["n"] for a in aggs) / np.maximum(n, 1)
        gain = sum(a["gain"] * a["n"] for a in aggs) / np.maximum(n, 1)
        seq = np.concatenate([a["seq_gain"] for a in aggs], axis=1)
        nbc = sum(a["n_by_chunk"] for a in aggs)
        gbc = sum(a["gain_by_chunk"] * a["n_by_chunk"] for a in aggs) / np.maximum(nbc, 1)
        return n, loss, gain, seq, gbc

    for probe, classes in (("P1", P.P1_CLASSES), ("P2", P.P2_CLASSES), ("P3", P.P3_CLASSES)):
        if not pooled[probe]:
            continue
        n, loss, gain, seq, gbc = merge(pooled[probe])
        parts = []
        for ci, name in enumerate(classes):
            se = P.se_over_sequences(seq[ci])
            out[f"{key_prefix}/{probe}/{name}/n"] = int(n[ci]); out[f"{key_prefix}/{probe}/{name}/loss"] = float(loss[ci])
            out[f"{key_prefix}/{probe}/{name}/gain"] = float(gain[ci]); out[f"{key_prefix}/{probe}/{name}/se"] = se
            parts.append(f"{name} n={int(n[ci])} loss {loss[ci]:.3f} gain {gain[ci]:+.4f}±{se:.4f}")
        if probe == "P3":
            # direction of the write's key: left-only minus right-only visible neighbour (same t
            # distribution), SE over the sequences that have both classes
            d = seq[1] - seq[2]
            d = d[~np.isnan(d)]
            dse = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
            out[f"{key_prefix}/P3/left_minus_right"] = float(gain[1] - gain[2]); out[f"{key_prefix}/P3/left_minus_right_se"] = dse
            parts.append(f"Lv_Rm - Lm_Rv {gain[1] - gain[2]:+.4f}±{dse:.4f}")
        log(f"[eval {step:>6}] {label} {probe} pooled: " + " | ".join(parts))
        log(f"[eval {step:>6}] {label} {probe} gain by chunk: " + " | ".join(
            f"{name} " + " ".join(f"{v:+.3f}" for v in gbc[ci]) for ci, name in enumerate(classes)))
        for src in sources:
            aggs = per_src.get((probe, src.name))
            if not aggs:
                continue
            n, loss, gain, seq, _ = merge(aggs)
            for ci, name in enumerate(classes):
                out[f"{key_prefix}/{probe}/{src.name}/{name}/gain"] = float(gain[ci])
                out[f"{key_prefix}/{probe}/{src.name}/{name}/n"] = int(n[ci])
            log(f"[eval {step:>6}] {label} {probe} {src.name}: " + " | ".join(
                f"{name} n={int(n[ci])} loss {loss[ci]:.3f} gain {gain[ci]:+.4f}" for ci, name in enumerate(classes)))
    return out


# ----------------------------------------------------------------------------
# the step
# ----------------------------------------------------------------------------
def compute_model(params, static_model, mesh, replicate_weights: str):
    """The bf16 compute copy of the model for one step.

    The suffix stack is used once per chunk per forward (A, B, C, their
    rematerialisations and the inner backward) and the tied embedding once per
    chunk in each CE: under plain FSDP every use re-gathered the sharded weights.
    Gather the bf16 copies once (0.8 + 0.5 GB per chip at 1B) and keep them; the
    f32 master copy and optimiser state stay sharded.  Prefix layers are gathered
    per layer inside their scan.
    """
    m = cast_floating(eqx.combine(params, static_model), jnp.bfloat16)
    rep = {x.strip() for x in replicate_weights.split(",") if x.strip()}
    if rep - {"suffix", "embed"}:
        raise ValueError(f"replicate_weights: unknown entries {rep - {'suffix', 'embed'}}")
    where = []
    if "suffix" in rep:
        where.append((lambda m: (m.suffix, m.mem_norm, m.gdn)) if m.gdn is not None
                     else (lambda m: (m.suffix, m.prime_norm, m.prime_post_norm, m.fast_weight)))
    if "embed" in rep:
        where.append((lambda m: (m.embed,)) if m.lm_head is None else (lambda m: (m.embed, m.lm_head)))
    if "suffix" in rep and m.inner_feat_head is not None:
        where.append(lambda m: (m.inner_feat_head,))
    if "suffix" in rep and m.mtp_head is not None:
        where.append(lambda m: (m.mtp_head,))
    if "suffix" in rep and m.inner_kvb_value is not None:
        where.append(lambda m: (m.inner_kvb_value,))
    if where:
        m = eqx.tree_at(lambda m: sum((w(m) for w in where), ()), m,
                        replace_fn=lambda x: constrain_replicated(x, mesh))
    return m


NORM_STATS = ("ttt_delta_norm", "ttt_init_norm", "ttt_drift_rel")


def make_train_step(optimizer, lcfg: LossConfig, mesh, replicate_weights: str = "suffix,embed",
                    grad_accum: int = 1):
    """One optimiser step.  With ``grad_accum`` > 1 the batch leaves carry a leading
    micro-batch axis ``[N, B, T]`` and the step scans over them: the batch loss is
    ``sum(ce) / sum(mask weight)`` over all N, so each micro-batch gradient (of
    ``ce_m / n_m``) enters with weight ``n_m / sum n`` - the gradient of one N-times
    larger batch, at the memory of one.  ``grad_accum == 1`` is the unchanged path."""
    @eqx.filter_jit(donate="all-except-first")
    def train_step(static_model, params, opt_state, batch: Batch, key, step):
        def objective(p, b, k):
            return loss_fn(compute_model(p, static_model, mesh, replicate_weights), b, k, step, lcfg, mesh)

        if grad_accum == 1:
            (loss, stats), grads = jax.value_and_grad(objective, has_aux=True)(params, batch, key)
        else:
            keys = jax.random.split(key, grad_accum)
            first = jax.tree_util.tree_map(lambda x: x[0], batch)
            st0 = jax.eval_shape(lambda: objective(params, first, keys[0])[1])
            zeros = lambda t: jax.tree_util.tree_map(lambda a: jnp.zeros(a.shape, a.dtype), t)
            # The gradient accumulator must stay sharded like the parameters (a replicated
            # f32 copy would be 5 GB per chip at 1B); a scan carry keeps its initial layout.
            shard = param_shardings(params, mesh) if mesh is not None else None
            acc0 = jax.tree_util.tree_map(lambda a, sh: jax.lax.with_sharding_constraint(jnp.zeros(a.shape, a.dtype), sh),
                                          params, shard) if shard is not None else zeros(params)

            def body(acc, xs):
                b, k = xs
                (loss_m, st), g = jax.value_and_grad(objective, has_aux=True)(params, b, k)
                w = st["mask_count"] + 1e-6                     # the micro-batch loss's denominator
                acc_g, acc_w, acc_loss, acc_sq, acc_st = acc
                acc_g = jax.tree_util.tree_map(lambda a, x: a + w * x, acc_g, g)
                # summed below as: the losses weighted by their denominators, the norms squared
                st = dict(st, main_loss=st["main_loss"] * w,
                          mtp_loss=st["mtp_loss"] * (st["mtp_mask_count"] + 1e-6),
                          **{n: jnp.square(st[n]) for n in NORM_STATS})
                acc_st = jax.tree_util.tree_map(lambda a, x: a + x, acc_st, st)
                return (acc_g, acc_w + w, acc_loss + w * loss_m, acc_sq + optax.global_norm(g) ** 2, acc_st), None

            (grads, wsum, loss, micro_sq, stats), _ = jax.lax.scan(
                body, (acc0, jnp.zeros(()), jnp.zeros(()), jnp.zeros(()), zeros(st0)), (batch, keys))
            grads = jax.tree_util.tree_map(lambda x: x / wsum, grads)
            loss = loss / wsum
            # The diagnostics as for one batch of all N: per-chunk / per-sequence means are the
            # mean over the (equal) micro-batches; counts are totals; a loss is sum CE / sum
            # weight; the drift norms run over all sequences (their squares add).
            tot = stats
            stats = {k: v / grad_accum for k, v in tot.items()}
            stats["mask_count"], stats["mtp_mask_count"] = tot["mask_count"], tot["mtp_mask_count"]
            stats["main_loss"] = tot["main_loss"] / wsum
            stats["mtp_loss"] = tot["mtp_loss"] / (tot["mtp_mask_count"] + grad_accum * 1e-6)
            d, i = jnp.sqrt(tot["ttt_delta_norm"]), jnp.sqrt(tot["ttt_init_norm"])
            stats["ttt_delta_norm"], stats["ttt_init_norm"] = d, i
            # TTT: ||W - W0|| / ||W0||; the GDN memory (no W0) reports the RMS of its states
            stats["ttt_drift_rel"] = jnp.where(i > 0, d / (i + 1e-12), jnp.sqrt(tot["ttt_drift_rel"] / grad_accum))
            # Gradient noise scale (McCandlish et al. 2018, "B_simple") from the N
            # micro-batch gradients of B_m sequences each: E|g_B|^2 = |G|^2 + tr(S)/B for a
            # batch of B, so the micro (B_m) and full (N*B_m) norms give tr(S) and |G|^2.
            # The critical batch, in SEQUENCES; a per-step estimate, average it over steps.
            N, Bm = grad_accum, batch.input_ids.shape[1]
            micro_sq = micro_sq / N
            full_sq = optax.global_norm(grads) ** 2
            tr_sigma = Bm * N * (micro_sq - full_sq) / (N - 1)
            signal_sq = (N * full_sq - micro_sq) / (N - 1)
            stats["grad_norm_micro"] = jnp.sqrt(micro_sq)
            # NaN when the two-norm estimate of |G|^2 is not positive (a noisy step): the
            # offline estimate is the ratio of the step-AVERAGED norms, not the average ratio.
            stats["grad_noise_scale"] = jnp.where(signal_sq > 0, tr_sigma / jnp.where(signal_sq > 0, signal_sq, 1.0), jnp.nan)
        gnorm = optax.global_norm(grads)
        finite = jnp.isfinite(gnorm)
        # Gradient norm by parameter group (where a spike lives): embedding, prefix layers,
        # suffix layers, the fast-weight template, everything else (norms, heads, write controls).
        mem = (lambda g: (g.gdn,)) if grads.gdn is not None else (lambda g: (g.fast_weight,))   # the memory group
        groups = {"embed": lambda g: (g.embed,), "prefix": lambda g: (g.prefix,), "suffix": lambda g: (g.suffix,),
                  "fast_weight": mem}
        rest = eqx.tree_at(lambda g: (g.embed, g.prefix, g.suffix) + mem(g), grads, replace_fn=lambda _: None)
        gn = {f"gnorm_{k}": optax.global_norm(eqx.filter(f(grads), eqx.is_array)) for k, f in groups.items()}
        gn["gnorm_other"] = optax.global_norm(eqx.filter(rest, eqx.is_array))
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, {"loss": loss, "grad_norm": gnorm, "finite": finite, **gn, **stats}

    return train_step


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def _multi_host_tpu() -> bool:
    """A TPU host that has not been pinned to its own chips (``TPU_HOST_BOUNDS``)
    belongs to the whole slice: every host must start together and join through
    ``jax.distributed.initialize``.  Decided from the device files, not from
    ``jax.default_backend()``, which would itself initialise the backend first."""
    platforms = os.environ.get("JAX_PLATFORMS", "").lower()
    return (platforms == "" or "tpu" in platforms) and os.environ.get("TPU_HOST_BOUNDS") is None \
        and os.path.exists("/dev/accel0")


def main(argv=None):
    if os.environ.get("TTT_COORDINATOR"):
        # The launcher (docs/launch.md) names the coordinator and numbers the hosts.  On Cloud TPU
        # jax.process_index() is still assigned by the TPU runtime from the slice
        # topology, which is why every process announces itself below.
        local_dev_ids = None
        if "TTT_LOCAL_DEVICE_IDS" in os.environ:
            local_dev_ids = [int(x) for x in os.environ["TTT_LOCAL_DEVICE_IDS"].split(",")]
        elif "CUDA_VISIBLE_DEVICES" in os.environ:
            local_dev_ids = list(range(len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))))
        # TTT_HEARTBEAT_TIMEOUT_S: how long a process may stop answering the coordination service
        # before every process is aborted (JAX default 100 s).  A process frozen in a shared-
        # filesystem outage (the /taiga metadata server, 10-03, ~12 min) otherwise kills the run.
        jax.distributed.initialize(coordinator_address=os.environ["TTT_COORDINATOR"],
                                   num_processes=int(os.environ["TTT_NUM_PROCESSES"]),
                                   process_id=int(os.environ["TTT_PROCESS_ID"]),
                                   local_device_ids=local_dev_ids,
                                   heartbeat_timeout_seconds=int(os.environ.get("TTT_HEARTBEAT_TIMEOUT_S", "100")))
    elif os.environ.get("COORDINATOR_ADDRESS") or int(os.environ.get("NUM_PROCESSES", "1")) > 1 or _multi_host_tpu():
        jax.distributed.initialize()

    cfg = parse_args(argv)
    install_trigger_signals()
    clear_trigger()                       # a stale trigger file must not fire on the first step
    mcfg = ModelConfig.load(cfg.model_config)
    rank, world = jax.process_index(), jax.process_count()
    is_main = rank == 0
    # Every process announces itself, so no per-host log is ever empty and the
    # host that carries the training log is identifiable from any of them.
    print(f"[ttt] process {rank}/{world} on {socket.gethostname()}: "
          f"{'main - training log follows' if is_main else 'worker - quiet'}", flush=True)
    # Flush every line: pod logs are files, and block-buffered stdout would hide
    # the header, the probe result, and the last steps before a crash.
    log = (lambda *a, **k: print(*a, **k, flush=True)) if is_main else (lambda *a, **k: None)

    devices = jax.devices()
    mesh = make_mesh(devices, cfg.fsdp_size)
    jax.set_mesh(mesh)          # mesh context: model.Attention wraps Pallas calls in shard_map
    ndev = len(devices)
    if cfg.batch_size % ndev:
        raise ValueError(f"batch_size {cfg.batch_size} must be divisible by {ndev} devices")
    if cfg.grad_accum < 1:
        raise ValueError(f"grad_accum {cfg.grad_accum} must be >= 1")
    log(f"[ttt] {world} hosts x {ndev // world} chips = {ndev} devices (mesh {dict(mesh.shape)}); "
        f"batch {cfg.batch_size} ({cfg.batch_size // ndev}/chip)"
        + (f" x {cfg.grad_accum} micro-batches = {cfg.batch_size * cfg.grad_accum} sequences/step" if cfg.grad_accum > 1 else "")
        + f", seq {cfg.seq_len}, steps {cfg.steps}")
    log(f"[ttt] model {cfg.model_config}: L={mcfg.num_hidden_layers} H={mcfg.hidden_size} "
        f"suffix={mcfg.suffix_len}"
        + (f" (fast weights in layers {list(mcfg.fast_layer_indices)}, plain "
           f"{[l for l in range(mcfg.prefix_len, mcfg.num_hidden_layers) if l not in mcfg.fast_layer_indices]})"
           if not all(mcfg.fast_mask) else "")
        + f" chunk={mcfg.chunk_size} window={mcfg.window_chunks}")
    # The full configuration, so a log is self-describing (checkpoints carry it too).
    log("[ttt] config " + " ".join(f"{k}={v}" for k, v in dataclasses.asdict(cfg).items()))

    # ---- model: init directly sharded on device, or load
    key = jax.random.PRNGKey(cfg.seed)
    skeleton = jax.eval_shape(lambda: Model(mcfg, key))
    shardings = param_shardings(skeleton, mesh)
    with mesh:
        model = jax.jit(lambda k: Model(mcfg, k), out_shardings=shardings)(key)
    n_params = sum(int(np.prod(x.shape)) for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))
    log(f"[ttt] {n_params/1e9:.3f}B params")

    optimizer, sched = make_optimizer(cfg)
    params, static = eqx.partition(model, eqx.is_array)
    opt_state = optimizer.init(params)
    start_step = 0

    # This run's own newest checkpoint first (--continue_from implies --resume auto), then the run it continues.
    resume_dir = ckpt.resolve_resume(cfg.ckpt_dir, "auto" if cfg.continue_from else cfg.resume)
    cont_dir = None if resume_dir or not cfg.continue_from else ckpt.resolve_resume(cfg.ckpt_dir, cfg.continue_from)
    if resume_dir or cont_dir:
        from_dir = resume_dir or cont_dir
        st = ckpt.read_state(from_dir)
        model = ckpt.load_weights(from_dir, model, shardings)
        params, static = eqx.partition(model, eqx.is_array)
        opt_state = ckpt.load_opt(from_dir, opt_state, mesh)
        start_step = int(st["step"]) + 1
        log(f"[ttt] resumed {from_dir} at step {start_step}" if resume_dir else
            f"[ttt] continuing {from_dir} at step {start_step}: its weights, optimiser state and step; "
            f"this run's data and settings")
    elif cfg.init_from:
        model = ckpt.load_weights(cfg.init_from, model, shardings)
        params, static = eqx.partition(model, eqx.is_array)
        log(f"[ttt] initialised weights from {cfg.init_from}; fresh optimiser and schedule")

    # ---- data
    sources = parse_sources(cfg.sources, cfg.data_root)
    check_vocab(sources, mcfg.vocab_size)
    plant = None
    if cfg.plant_frac > 0:
        if cfg.plant_len_max > mcfg.chunk_size or cfg.seq_len // mcfg.chunk_size < mcfg.window_chunks + 2:
            raise ValueError(f"plant: span <= chunk_size ({mcfg.chunk_size}) and >= window_chunks + 2 chunks per sequence")
        plant = Plant(cfg.plant_frac, cfg.plant_len_min, cfg.plant_len_max, mcfg.chunk_size, mcfg.window_chunks)
    windows = None
    if cfg.doc_windows:
        from .data import windows_path
        paths = [windows_path(s.path, cfg.seq_len, cfg.holdout) for s in sources]
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"doc_windows: build the window index first (python -m ttt.data windows): {missing}")
        windows = [np.load(p) for p in paths]
    sampler = Sampler(sources, cfg.seq_len, cfg.batch_size, cfg.seed, rank=rank, world=world,
                      holdout_tokens=cfg.holdout, eos=mcfg.eos_token_id, plant=plant, windows=windows,
                      accum=cfg.grad_accum)
    # Step s trains on the sampler's batch s - data_offset.  A continued run keeps the checkpoint's stream when its
    # data is the same (a branch reads on where the trunk stopped) and otherwise starts the new stream at its first
    # batch (a stage of another length or mix: no sequence repeated or skipped inside the stage); checkpoints carry it.
    data_offset = 0
    if resume_dir:
        check_resume_data(resume_dir, st, sampler, cfg, log)
        data_offset = int(st.get("data_offset", 0))
    elif cont_dir:
        same = st.get("data_hash_version", 1) >= 2 and st.get("data_hash") == sampler.hash
        if not same and st.get("data_hash_version", 1) >= 2:
            with open(os.path.join(cont_dir, "train_config.json")) as f:
                old = json.load(f)
            old_step = old["batch_size"] * old.get("grad_accum", 1)
            if old_step != sampler.step_batch and st.get("data_hash") == sampler.hash_at(old_step):
                raise ValueError(f"--continue_from {cont_dir}: the same data at another step batch ({old_step} -> "
                                 f"{sampler.step_batch} sequences) would read the stream again from its first sample; "
                                 "keep batch_size x grad_accum")
        data_offset = int(st.get("data_offset", 0)) if same else start_step
        log(f"[ttt] data: " + ("the continued run's stream, read on" if same else
                               f"a new stream (hash {sampler.hash}), its first batch at step {start_step}"))
    log(f"[ttt] sources: " + ", ".join(f"{s.name}({s.num_tokens/1e9:.2f}B, w={s.weight:.2f})" for s in sources))
    batch_axis = batch_axis_name(mesh)
    batch_sharding = NamedSharding(mesh, P(batch_axis, None))

    mask_maker = MaskMaker(cfg.inner_mask_mode, mcfg.chunk_size, mcfg.eos_token_id, mcfg.pad_token_id,
                           cfg.data_path(cfg.unigram_path), cfg.inner_target_frac, cfg.inner_generic_floor, cfg.seed,
                           cfg.inner_task)
    if mask_maker.active and cfg.inner_mask_mode != "random" and mask_maker.unigram is None:
        log(f"[ttt] inner_mask_mode {cfg.inner_mask_mode}: no unigram table at {cfg.data_path(cfg.unigram_path)}, "
            f"rarity falls back to the position hash")
    if cfg.eval_probes and not os.path.exists(cfg.data_path(cfg.unigram_path)):
        log(f"[ttt] eval_probes: no unigram table at {cfg.data_path(cfg.unigram_path)}, the rarity probe (P2) is skipped")

    def to_device(local: np.ndarray, host: Optional[np.ndarray] = None, outer_t: Optional[np.ndarray] = None,
                  loss_mask: Optional[np.ndarray] = None) -> Batch:
        """``local [B_local, T]`` (one batch) or ``[N, B_local, T]`` (N micro-batches of one
        optimiser step) -> a batch-sharded ``Batch``; ``loss_mask`` (host [B_local, T] bool):
        only these positions may be masked / scored (the planted-recall probe)."""
        micro = local.ndim == 3
        shard = NamedSharding(mesh, P(None, batch_axis, None)) if micro else batch_sharding
        shape = ((local.shape[0],) if micro else ()) + (cfg.batch_size, cfg.seq_len)
        ids = jax.make_array_from_process_local_data(shard, local, shape)
        t_arr = None
        if outer_t is not None:
            t_shard = NamedSharding(mesh, P(batch_axis))
            t_arr = jax.make_array_from_process_local_data(t_shard, outer_t.astype(np.float32), (cfg.batch_size,))
        lm_arr = None if loss_mask is None else jax.make_array_from_process_local_data(
            shard, loss_mask.astype(np.bool_), shape)
        if host is None and mask_maker.active:
            host = np.stack([mask_maker(x) for x in local]) if micro else mask_maker(local)
        if host is None:
            return Batch(input_ids=ids, outer_t=t_arr, loss_mask=lm_arr)
        arr = jax.make_array_from_process_local_data(
            shard, host.astype(np.int32 if cfg.inner_task == "split" else np.bool_), shape)
        if cfg.inner_task == "split":
            return Batch(input_ids=ids, inner_perm=arr, outer_t=t_arr, loss_mask=lm_arr)
        return Batch(input_ids=ids, inner_mask=arr, outer_t=t_arr, loss_mask=lm_arr)

    # recall-only training (--train_recall): uniform weights over the planted copy's scored tail
    lc_train = dataclasses.replace(loss_config(cfg), time_reweighting="none") if cfg.train_recall else loss_config(cfg)
    train_step = make_train_step(optimizer, lc_train, mesh, cfg.replicate_weights, cfg.grad_accum)
    eval_every = cfg.eval_interval or cfg.ckpt_interval        # 0: only the step-0 baseline
    eval_steps = None
    eval_steps_fixed = {}                                        # fixed t -> {scale: step}
    eval_groups = [(sampler, sources, True)]
    if cfg.eval_sources and cfg.holdout:
        # Evaluation-only sources (e.g. an out-of-distribution long-document corpus): their
        # held-out tail is scored and reported per source, never trained on or averaged in.
        ev_sources = parse_sources(cfg.eval_sources, cfg.data_root)
        check_vocab(ev_sources, mcfg.vocab_size)
        eval_groups.append((Sampler(ev_sources, cfg.seq_len, cfg.batch_size, cfg.seed, rank=rank, world=world,
                                    holdout_tokens=cfg.holdout, eos=mcfg.eos_token_id, eval_only=True,
                                    accum=cfg.grad_accum), ev_sources, False))
    if cfg.holdout and cfg.eval_batches:
        lc = dataclasses.replace(loss_config(cfg), per_sequence=True, per_token=cfg.eval_probes, mtp_weight=0.0,
                                 aux_causal_weight=0.0, write_plan=write_plan(cfg, cfg.seq_len // mcfg.chunk_size),
                                 window_drop=0.0, self_prior=0.0, outer_views=1)
        if cfg.eval_reread_chunks:
            # Re-read evaluation: every program (the scales and the re-read) scores the same
            # tail with the same segment layout, so the comparison is exactly paired.
            n = cfg.eval_reread_chunks
            seg = lc.segment_chunks or n
            if n % seg:
                raise ValueError(f"eval_reread_chunks {n} must be a multiple of segment_chunks {seg}")
            s0 = n if cfg.eval_score_from_chunk < 0 else cfg.eval_score_from_chunk
            lc = dataclasses.replace(lc, score_from_chunk=s0, segment_chunks=seg)
        scales = {1.0} | {float(x) for x in cfg.eval_inner_lr_scales.split(",") if x.strip()}
        if not cfg.no_write:
            scales.add(0.0)
        eval_steps = {sc: make_eval_step(at_scale(lc, sc), mesh,
                                         cfg.replicate_weights) for sc in scales}
        if cfg.eval_reread_chunks:
            eval_steps["reread"] = make_eval_step(dataclasses.replace(lc, reread_chunks=cfg.eval_reread_chunks),
                                                  mesh, cfg.replicate_weights)
        # Fixed-noise evaluations (unweighted CE at a pinned t, write on and off): the CART /
        # t~U headline under-weights the high-noise tokens the memory serves; t = 1 is the
        # regime where only the window and the memory carry information.
        for t_fixed in [float(x) for x in cfg.eval_fixed_t.split(",") if x.strip()]:
            lc_t = dataclasses.replace(lc, outer_t=t_fixed, time_reweighting="none", per_token=False, reread_chunks=0,
                                       score_from_chunk=0)
            eval_steps_fixed[t_fixed] = {sc: make_eval_step(at_scale(lc_t, sc), mesh,
                                                            cfg.replicate_weights) for sc in ((0.0, 1.0) if not cfg.no_write else (1.0,))}
    planted_steps = None
    if cfg.eval_planted_recall and eval_steps is not None:
        from .probes import planted_chunks
        planted_chunks(cfg.seq_len // mcfg.chunk_size, mcfg.window_chunks)          # refuses too-short sequences
        if cfg.eval_planted_len > mcfg.chunk_size:
            raise ValueError(f"eval_planted_len {cfg.eval_planted_len} > chunk_size {mcfg.chunk_size}")
        # per-token CE of the masked tail only (outer_t 1 and the probe's loss_mask fix the mask)
        lc_p = dataclasses.replace(loss_config(cfg), per_sequence=True, per_token=True, mtp_weight=0.0,
                                   aux_causal_weight=0.0, time_reweighting="none", outer_t=-1.0,
                                   write_plan=write_plan(cfg, cfg.seq_len // mcfg.chunk_size),
                                   window_drop=0.0, self_prior=0.0, outer_views=1)
        planted_steps = {sc: make_eval_step(at_scale(lc_p, sc), mesh,
                                            cfg.replicate_weights) for sc in ((0.0, 1.0) if not cfg.no_write else (1.0,))}

    # ---- probe: compile, print XLA's memory analysis, exit
    if cfg.probe_memory:
        b = to_device(sampler.batch(0))
        lowered = train_step.lower(static, params, opt_state, b, key, jnp.int32(0))
        compiled = lowered.compile().compiled          # equinox wrapper -> jax.stages.Compiled
        ma = compiled.memory_analysis()
        log(f"[probe] argument {ma.argument_size_in_bytes/2**30:.2f} GiB  output {ma.output_size_in_bytes/2**30:.2f} GiB  "
            f"temp {ma.temp_size_in_bytes/2**30:.2f} GiB  generated {ma.generated_code_size_in_bytes/2**20:.1f} MiB")
        log(f"[probe] total per chip ~ {(ma.argument_size_in_bytes + ma.output_size_in_bytes + ma.temp_size_in_bytes)/2**30:.2f} GiB")
        return 0

    # ---- wandb (main process only)
    wb = None
    if cfg.wandb_project and is_main:
        import wandb
        # Continue the checkpoint's wandb run only when this run writes to the same
        # checkpoint directory; a branch (resume into another ckpt_dir) is a new run.
        same_run = resume_dir and os.path.dirname(os.path.abspath(resume_dir)) == os.path.abspath(cfg.ckpt_dir)
        wb = wandb.init(project=cfg.wandb_project, config={**dataclasses.asdict(cfg), **dataclasses.asdict(mcfg)},
                        resume="allow", id=st.get("wandb_run_id") if same_run else None)

    # ---- loop
    # The data stream is indexed in micro-batches of batch_size: optimiser step s consumes
    # sampler batches s*N .. s*N+N-1, so "N micro-batches for S steps" reads exactly the tokens
    # of "1 for S*N steps", and the step's sequences depend only on batch_size * N (16 x 4 reads
    # what 64 x 1 reads; the data hash counts that product).
    N = cfg.grad_accum
    prefetch = Prefetcher(sampler, (start_step - data_offset) * N, max(cfg.prefetch, N),
                          mask_maker if mask_maker.active else None)
    save_state = lambda: {"data_offset": data_offset, **({"wandb_run_id": wb.id} if wb else {})}
    overflow = 0
    t_last = time.time()
    last_log_step = start_step - 1        # s/step = elapsed / steps since the previous log line
    tokens_per_step = cfg.batch_size * cfg.seq_len * N
    def run_eval(step):
        if eval_steps is None:
            return
        ev = evaluate(eval_steps, static, params, eval_groups, cfg, to_device, key, step, log,
                      rank=rank, world=world, mesh=mesh, is_main=is_main)
        for t_fixed, steps_t in eval_steps_fixed.items():
            ev.update(evaluate(steps_t, static, params, eval_groups, cfg, to_device, key, step, log,
                               rank=rank, world=world, mesh=mesh, is_main=is_main, tag=f" t={t_fixed:g}"))
        if planted_steps is not None:
            ev.update(evaluate_planted(planted_steps, static, params, eval_groups, cfg, to_device, step, log,
                                       mesh=mesh, is_main=is_main))
        if wb:
            wb.log(ev, step=step)

    try:
        if cfg.eval_only:
            run_eval(start_step)
            return 0
        if (start_step == 0 or cont_dir) and cfg.eval_step0:
            run_eval(start_step)                      # the untrained baseline, or the continued run's last state
        profile_stop = -1
        for step in range(start_step, cfg.steps):
            # Profiling (process 0 only): two steady-state steps right after start-up,
            # and again whenever a file named TRIGGER appears in --profile_dir, so a
            # running pod job can be traced without a restart.
            if cfg.profile_dir and is_main:
                trigger = os.path.join(cfg.profile_dir, "TRIGGER")
                if step == start_step + 2 or (step % cfg.log_interval == 0 and os.path.exists(trigger)):
                    if os.path.exists(trigger):
                        os.remove(trigger)
                    jax.block_until_ready(params)
                    jax.profiler.start_trace(cfg.profile_dir, create_perfetto_trace=True)
                    profile_stop = step + 2
                elif step == profile_stop:
                    jax.block_until_ready(params)
                    jax.profiler.stop_trace()
                    log(f"[ttt] profile (steps {step - 2}-{step - 1}) -> {cfg.profile_dir}")
            ds = step - data_offset                   # the sampler's batch of this step
            if cfg.train_recall:
                from .probes import recall_train_batch
                local, host = prefetch.get(ds)
                j0 = ds * sampler.G + sampler.rank * sampler.local
                ids_r, lm_r = recall_train_batch(local, range(j0, j0 + len(local)), cfg.seed, mcfg.chunk_size,
                                                 mcfg.window_chunks, cfg.plant_len_min, cfg.plant_len_max,
                                                 cfg.train_recall_cue, eos=mcfg.eos_token_id)
                b = to_device(ids_r, host, outer_t=np.ones(len(ids_r), np.float32), loss_mask=lm_r)
            elif N == 1:
                b = to_device(*prefetch.get(ds))
            else:
                micro = [prefetch.get(ds * N + m) for m in range(N)]
                b = to_device(np.stack([x for x, _ in micro]),
                              None if micro[0][1] is None else np.stack([h for _, h in micro]))
            # Stateless per-step key: a resumed run must draw exactly the masks the
            # uninterrupted run drew at this step.  A running `key, sk = split(key)`
            # would start over at the resume point.
            sk = jax.random.fold_in(key, step)
            params, opt_state, m = train_step(static, params, opt_state, b, sk, jnp.int32(step))

            if step % cfg.log_interval == 0 or step == cfg.steps - 1:
                m = {k: float(v) for k, v in m.items() if getattr(v, "ndim", 0) == 0}
                now = time.time()
                # Per-step time over the steps actually elapsed since the last log line (the
                # final line covers steps - 1 - last_log_step steps, not log_interval; dividing
                # by log_interval under-reported it by 10 %).  The first line after a start or
                # resume therefore includes compilation, as the --log_interval 1 runs always did.
                dt = (now - t_last) / max(1, step - last_log_step)
                t_last, last_log_step = now, step
                overflow += int(not m["finite"])
                row = {"step": step, "lr": float(sched(step)), "step_time": dt,
                       "tokens_per_s": tokens_per_step / max(dt, 1e-9), "overflow": overflow, **m}
                # The logged loss is the main CE (comparable across arms); with MTP the optimised
                # objective (main + mtp_weight x mtp) is the wandb "loss" and the line shows both terms.
                log(f"[{step:>7}] loss {m['main_loss']:.4f}  inner {m['inner_loss']:.4f}  "
                    f"drift {m['ttt_drift_rel']:.2e}  gnorm {m['grad_norm']:.3f}  "
                    f"lr {row['lr']:.2e}  {dt:.3f}s/step  {row['tokens_per_s']/1e3:.0f}k tok/s"
                    + (f"  noise {m['grad_noise_scale']:.0f} seq" if "grad_noise_scale" in m else "")
                    + (f"  feat {m['inner_feat']:.3f}" if cfg.inner_feat != "off" else "")
                    + (f"  distill {m['inner_distill']:.3f}" if cfg.distill else "")
                    + (f"  mtp {m['mtp_loss']:.3f}/{m['inner_mtp']:.3f}" if cfg.mtp else "")
                    + (f"  aux x{cfg.aux_causal_weight:g}" if cfg.aux_causal_weight > 0 else "")
                    + (f"  kvb {m['inner_kvb']:.3f}" if cfg.inner_kvb_weight > 0 else "")
                    + (f"  write[n{cfg.inner_subchunks} s{cfg.inner_subchunk_scale:g} g{cfg.inner_surprise_gamma:g}"
                       f" f{cfg.inner_fast_frac:g}" + (f" k{cfg.inner_write_steps}" if cfg.inner_write_steps > 1 else "")
                       + "]" if not write_config(cfg).plain else "")
                    + (f"  gn[e/p/s/w/o] {m['gnorm_embed']:.2f}/{m['gnorm_prefix']:.2f}/{m['gnorm_suffix']:.2f}/"
                       f"{m['gnorm_fast_weight']:.2f}/{m['gnorm_other']:.2f}" if m["grad_norm"] > 2.0 or step < 2 else ""))
                if wb:
                    wb.log(row, step=step)

            # every ckpt_interval steps and the last step (the next stage continues from it)
            if cfg.ckpt_interval and ((step + 1) % cfg.ckpt_interval == 0 or step == cfg.steps - 1):
                d = ckpt.save(cfg.ckpt_dir, step, eqx.combine(params, static), opt_state, mcfg, cfg, sampler.hash,
                              keep=cfg.keep_checkpoints, extra_state=save_state())
                log(f"[ttt] checkpoint -> {d}  (keeping the newest {cfg.keep_checkpoints})")
                saved_here = True
            else:
                saved_here = False
            # On-demand checkpoint / clean stop (see the trigger helpers at the top): every
            # process polls its own host, the flags are combined across hosts once per step.
            flag = poll_trigger()
            if world > 1:
                # Combined across hosts through the two collectives every pod run already
                # exercises: a batch-sharded global array (as the evaluation keys) gathered like
                # a checkpoint (`checkpoint._to_host`).  A bare jnp scalar under the mesh
                # context is a global non-addressable array that the gather refuses.
                per_dev = np.full((jax.local_device_count(),), flag, np.int32)
                flag_arr = jax.make_array_from_process_local_data(NamedSharding(mesh, P(batch_axis_name(mesh))), per_dev, (ndev,))
                flag = int(np.max(multihost_utils.process_allgather(flag_arr, tiled=True)))
            if flag:
                if not saved_here:
                    d = ckpt.save(cfg.ckpt_dir, step, eqx.combine(params, static), opt_state, mcfg, cfg, sampler.hash,
                                  keep=cfg.keep_checkpoints, extra_state=save_state())
                    log(f"[ttt] triggered checkpoint -> {d}")
                clear_trigger()
                if flag == TRIGGER_STOP:
                    log(f"[ttt] triggered stop after step {step}: checkpoint written, exiting (--resume auto continues)")
                    break
            if eval_every and ((step + 1) % eval_every == 0 or step == cfg.steps - 1):
                run_eval(step + 1)
    finally:
        prefetch.close()
        if wb:
            wb.finish()
        if os.environ.get("TTT_COORDINATOR"):
            try:
                jax.distributed.shutdown()
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
