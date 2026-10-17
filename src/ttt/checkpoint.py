# coding=utf-8
"""
Checkpoints: f32 master weights, optimiser state, and enough metadata to resume.

Layout::

    {ckpt_dir}/step_{N:08d}/
        model.eqx.part*     eqx.tree_serialise_leaves of the array leaves, f32
        opt.eqx.part*       optax state
        model_config.json   train_config.json
        state.json          {step, data_hash, data_hash_version, git, timestamp}
        DONE                written last; a directory without it is ignored

Two ways in:

* ``resume(dir)``     - same run: step, optimiser, data position all continue.
                        The data hash must match or we refuse (a different data
                        order would silently re-see or skip sequences).
* ``init_from(dir)``  - weights only: step 0, fresh optimiser, fresh schedule.
                        This is what a context-extension stage wants; resuming
                        with ``--steps 1250`` from step 50 000 would run zero steps.

Weights are saved in f32.  The previous stack saved bf16 and reloaded it as the
f32 master, rounding every parameter to 8 mantissa bits on every resume while
the AdamW moments stayed f32.
"""
from __future__ import annotations

import dataclasses
import json
import os
import shutil
import subprocess
import time
from typing import Any, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding

from .sharded_io import ShardedFileReader, ShardedFileWriter


def _git() -> dict:
    """Commit of the running code.  Worker hosts get the code by rsync without .git,
    so the launcher passes TTT_GIT_COMMIT / TTT_GIT_BRANCH / TTT_GIT_DIRTY."""
    if os.environ.get("TTT_GIT_COMMIT"):
        return {"commit": os.environ["TTT_GIT_COMMIT"], "branch": os.environ.get("TTT_GIT_BRANCH", ""),
                "dirty": os.environ.get("TTT_GIT_DIRTY", "0") == "1"}
    try:
        run = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, timeout=5).stdout.strip()
        return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
                "dirty": bool(run("status", "--porcelain"))}
    except Exception:  # noqa: BLE001 - metadata only
        return {}


def step_dir(ckpt_dir: str, step: int) -> str:
    return os.path.join(ckpt_dir, f"step_{step:08d}")


def latest_complete(ckpt_dir: str) -> Optional[str]:
    """Newest ``step_*`` directory that has a ``DONE`` marker, or None."""
    if not os.path.isdir(ckpt_dir):
        return None
    done = [d for d in os.listdir(ckpt_dir)
            if d.startswith("step_") and os.path.exists(os.path.join(ckpt_dir, d, "DONE"))]
    return os.path.join(ckpt_dir, max(done)) if done else None


def _to_host(tree):
    """Gather sharded arrays to fully-replicated host numpy (every process gets the whole tree)."""
    def one(x):
        if isinstance(x, jax.Array) and len(x.sharding.device_set) > 1:
            return np.asarray(multihost_utils.process_allgather(x, tiled=True))
        return np.asarray(x) if isinstance(x, jax.Array) else x
    return jax.tree_util.tree_map(one, tree)


def prune(ckpt_dir: str, keep: int) -> list:
    """Delete all but the newest ``keep`` complete checkpoints; returns what was removed.
    Checkpoints are 15 GB each at 1B, and the bucket is not a free archive."""
    if keep <= 0 or not os.path.isdir(ckpt_dir):
        return []
    done = sorted(d for d in os.listdir(ckpt_dir)
                  if d.startswith("step_") and os.path.exists(os.path.join(ckpt_dir, d, "DONE")))
    removed = []
    for d in done[:-keep]:
        shutil.rmtree(os.path.join(ckpt_dir, d), ignore_errors=True)
        removed.append(d)
    return removed


def save(ckpt_dir: str, step: int, model, opt_state, model_cfg, train_cfg, data_hash: str,
         keep: int = 0, extra_state: Optional[dict] = None) -> str:
    """Write a complete checkpoint, then keep only the newest ``keep`` (0 = all).
    ``extra_state`` (e.g. the wandb run id) is merged into ``state.json`` so a
    resume can continue it.  All processes must call this (collective gathers)."""
    d = step_dir(ckpt_dir, step)
    params = _to_host(eqx.filter(model, eqx.is_array))
    opt = _to_host(opt_state)
    if jax.process_index() == 0:
        os.makedirs(d, exist_ok=True)
        with ShardedFileWriter(os.path.join(d, "model.eqx")) as f:
            eqx.tree_serialise_leaves(f, params)
        with ShardedFileWriter(os.path.join(d, "opt.eqx")) as f:
            eqx.tree_serialise_leaves(f, opt)
        model_cfg.save(os.path.join(d, "model_config.json"))
        train_cfg.save(os.path.join(d, "train_config.json"))
        with open(os.path.join(d, "state.json"), "w") as f:
            json.dump({"step": int(step), "data_hash": data_hash, "data_hash_version": 2, "git": _git(), "time": time.time(),
                       **(extra_state or {})}, f, indent=2)
        with open(os.path.join(d, "DONE"), "w") as f:
            f.write("ok\n")
        prune(ckpt_dir, keep)
    multihost_utils.sync_global_devices(f"ckpt_{step}")
    return d


def _place(tree, target_shardings):
    """Host numpy -> device arrays with the given shardings (process-local callback)."""
    def one(x, sh):
        if sh is None or not hasattr(x, "shape"):
            return x
        return jax.make_array_from_callback(x.shape, sh, lambda idx: x[idx])
    return jax.tree_util.tree_map(one, tree, target_shardings)


def _host_like(tree):
    """Host placeholders with the leaves' shapes and dtypes.  Never ``np.asarray`` a
    live leaf: on a pod the skeleton's arrays span devices of other hosts and
    cannot be fetched."""
    return jax.tree_util.tree_map(
        lambda x: np.zeros(x.shape, x.dtype) if hasattr(x, "shape") else x, tree)


# Optional model fields added after a checkpoint was written keep the skeleton's
# initialisation when the saved model_config does not enable them (init_from only:
# a resume also needs their optimiser moments, which the checkpoint cannot have).
_OPTIONAL = {"inner_gate": ("inner_lr_gate", False), "inner_token_gate": ("inner_token_gate", False),
             "inner_decay_logit": ("inner_decay", 0.0), "inner_precond": ("inner_optimizer", False),
             "inner_step_log": ("inner_optimizer", False), "inner_temp_log": ("inner_optimizer", False),
             "inner_feat_head": ("inner_feat_head", False), "mtp_head": ("mtp_head", False),
             "inner_kvb_value": ("inner_kvb_head", False)}


def _saved_without(d: str, model_skeleton):
    """Names of optional fields the checkpoint at ``d`` was saved without."""
    p = os.path.join(d, "model_config.json")
    if not os.path.exists(p):
        return []
    with open(p) as f:
        saved = json.load(f)
    return [name for name, (key, default) in _OPTIONAL.items()
            if getattr(model_skeleton, name, None) is not None and saved.get(key, default) == default]


def _saved_fast_width(d: str) -> Optional[int]:
    p = os.path.join(d, "model_config.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        saved = json.load(f)
    return saved.get("ttt_intermediate_size") or saved.get("intermediate_size")


def prune_fast_weight(fw, inter: int):
    """Narrow a stacked SwiGLU fast weight ``[S, H, I] / [S, I, H]`` to ``inter`` hidden units
    per layer, keeping the units of largest importance ``|gate_j| * |up_j| * |down_j|``
    (magnitude pruning; the kept units stay in their original order).  Host arrays."""
    g, u, dn = (np.asarray(x) for x in (fw.gate.weight, fw.up.weight, fw.down.weight))
    imp = np.linalg.norm(g, axis=1) * np.linalg.norm(u, axis=1) * np.linalg.norm(dn, axis=2)    # [S, I]
    keep = np.sort(np.argsort(-imp, axis=1)[:, :inter], axis=1)                                 # [S, inter]
    take = lambda x, ax: np.stack([np.take(x[s], keep[s], axis=ax - 1) for s in range(x.shape[0])])
    fw = eqx.tree_at(lambda m: m.gate.weight, fw, take(g, 2))
    fw = eqx.tree_at(lambda m: m.up.weight, fw, take(u, 2))
    return eqx.tree_at(lambda m: m.down.weight, fw, take(dn, 1))


def load_weights(d: str, model_skeleton, shardings) -> Any:
    """Weights only, placed with ``shardings`` (a pytree of NamedSharding matching the model).
    A checkpoint whose fast weight is WIDER than the skeleton's (``ttt_intermediate_size``)
    is narrowed by magnitude pruning of its hidden units (``prune_fast_weight``)."""
    missing = _saved_without(d, model_skeleton)
    skel, shard = model_skeleton, shardings
    for name in missing:                    # read into the checkpoint's own structure
        skel = eqx.tree_at(lambda m: getattr(m, name), skel, None)
        shard = eqx.tree_at(lambda m: getattr(m, name), shard, None)
    saved_w = _saved_fast_width(d)
    new_w = model_skeleton.cfg.prime_intermediate_size
    narrow = model_skeleton.fast_weight is not None and saved_w is not None and saved_w != new_w
    if narrow:
        if saved_w < new_w:
            raise ValueError(f"checkpoint fast weight ({saved_w}) narrower than the model's ({new_w})")
        from .model import FastWeight, stack_modules
        cfg = model_skeleton.cfg
        wide = stack_modules(lambda k: FastWeight(cfg.hidden_size, saved_w, cfg.hidden_act, cfg.initializer_range, k),
                             jax.random.split(jax.random.PRNGKey(0), cfg.fast_len))
        skel = eqx.tree_at(lambda m: m.fast_weight, skel, wide)
        shard = eqx.tree_at(lambda m: m.fast_weight, shard, jax.tree_util.tree_map(lambda _: None, wide))
    params_skel = eqx.filter(skel, eqx.is_array)
    with ShardedFileReader(os.path.join(d, "model.eqx")) as f:
        params = eqx.tree_deserialise_leaves(f, _host_like(params_skel))
    params = jax.tree_util.tree_map(lambda x: x.astype(np.float32) if np.issubdtype(x.dtype, np.floating) else x, params)
    if narrow:
        pruned = prune_fast_weight(params.fast_weight, new_w)
        params = eqx.tree_at(lambda m: m.fast_weight, params, pruned)
        shard = eqx.tree_at(lambda m: m.fast_weight, shard, shardings.fast_weight)
        skel = eqx.tree_at(lambda m: m.fast_weight, skel, model_skeleton.fast_weight)
        print(f"[ttt] checkpoint {d}: fast weight narrowed {saved_w} -> {new_w} hidden units per layer "
              f"(magnitude pruning)", flush=True)
    params = _place(params, eqx.filter(shard, lambda s: isinstance(s, NamedSharding)))
    model = eqx.combine(params, skel)
    for name in missing:                    # ... then put the new fields back from the skeleton
        model = eqx.tree_at(lambda m: getattr(m, name), model, getattr(model_skeleton, name),
                            is_leaf=lambda x: x is None)
        print(f"[ttt] checkpoint {d} has no {name}: keeping its initialisation", flush=True)
    return model


def load_opt(d: str, opt_skeleton, mesh) -> Any:
    """Optimiser state, placed like the fresh ``opt_skeleton`` - sharded leaves keep
    their sharding, everything else (optax's step counter, scalars) is replicated.
    Leaving a scalar on one device while the params span the mesh makes jit refuse."""
    from jax.sharding import PartitionSpec as P
    rep = NamedSharding(mesh, P())
    shardings = jax.tree_util.tree_map(
        lambda x: x.sharding if isinstance(getattr(x, "sharding", None), NamedSharding) else rep,
        opt_skeleton)
    with ShardedFileReader(os.path.join(d, "opt.eqx")) as f:
        opt = eqx.tree_deserialise_leaves(f, _host_like(opt_skeleton))
    return _place(opt, shardings)


def read_state(d: str) -> dict:
    with open(os.path.join(d, "state.json")) as f:
        return json.load(f)


def resolve_resume(ckpt_dir: str, spec: str) -> Optional[str]:
    """``"auto"`` -> newest complete checkpoint under ``ckpt_dir`` (or None); else the path."""
    if not spec:
        return None
    if spec == "auto":
        return latest_complete(ckpt_dir)
    if not os.path.exists(os.path.join(spec, "DONE")):
        raise FileNotFoundError(f"{spec} is not a complete checkpoint (no DONE marker)")
    return spec
