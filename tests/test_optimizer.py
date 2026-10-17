# coding=utf-8
"""Outer optimiser: callable decay mask, schedule endpoints, no-decay leaves get zero decay, the template's lr scale."""
from __future__ import annotations

import dataclasses
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import re

import equinox as eqx
import jax
import jax.numpy as jnp

from src.ttt.config import ModelConfig, TrainConfig
from src.ttt.model import Model
from src.ttt.train import make_optimizer

CFG = ModelConfig(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=8, suffix_len=1,
                  chunk_size=8, window_chunks=1, mask_token_id=120, pad_token_id=0, eos_token_id=0,
                  attention_bias=True)
TC = TrainConfig(lr=1e-3, end_lr=1e-5, warmup_steps=100, decay_start=500, decay_end=1000, steps=1000, weight_decay=0.1)


def main():
    failures = []
    m = Model(CFG, jax.random.PRNGKey(0))
    params, _ = eqx.partition(m, eqx.is_array)
    opt, sched = make_optimizer(TC)

    # 1. THE LIKELY FAILURE: optax accepts the callable mask against the partitioned tree.
    try:
        state = opt.init(params); ok = True
    except Exception as e:  # noqa: BLE001
        ok, state = False, None
        failures.append(f"opt.init: {type(e).__name__}: {e}")
    print(f"[{'PASS' if ok else 'FAIL'}] optax accepts callable decay mask")
    if not ok:
        print("FAILURES: " + "; ".join(failures)); return 1

    # 2. Zero grads at step 0: masked leaves get exactly 0 update, others -lr0*wd*p.
    zero = jax.tree_util.tree_map(jnp.zeros_like, params)
    upd, _ = opt.update(zero, state, params)
    lr0 = float(sched(0))
    nd = re.compile(TC.no_decay_pattern)
    e_nodecay = e_decay = 0.0
    for (path, p), u in zip(jax.tree_util.tree_flatten_with_path(params)[0], jax.tree_util.tree_leaves(upd)):
        n = jax.tree_util.keystr(path)
        if nd.search(n):
            e_nodecay = max(e_nodecay, float(jnp.max(jnp.abs(u))))
        else:
            want = -lr0 * TC.weight_decay * p
            e_decay = max(e_decay, float(jnp.max(jnp.abs(u - want))) / (float(jnp.max(jnp.abs(want))) + 1e-12))
    ok = e_nodecay == 0.0 and e_decay < 1e-5
    failures += [] if ok else [f"decay mask: nodecay {e_nodecay:.1e} decay_rel {e_decay:.1e}"]
    print(f"[{'PASS' if ok else 'FAIL'}] masked=0 decay, unmasked=-lr*wd*p     {e_nodecay:.1e} / {e_decay:.1e}")

    # 3. Which leaves are exempt: embed, every norm, biases - and nothing else.
    exempt = sorted({jax.tree_util.keystr(p) for p, _ in jax.tree_util.tree_flatten_with_path(params)[0]
                     if nd.search(jax.tree_util.keystr(p))})
    ok = any("embed" in e for e in exempt) and any("norm" in e for e in exempt) and any("bias" in e for e in exempt) \
        and not any(".weight" in e and "norm" not in e for e in exempt)
    failures += [] if ok else [f"exempt set wrong: {exempt[:6]}"]
    print(f"[{'PASS' if ok else 'FAIL'}] exempt = embed|norm|bias only         {len(exempt)} leaves")

    # 4. Schedule endpoints (warmup-stable-decay; the values in between: test_run_config).
    s0, sw, st_ = float(sched(0)), float(sched(TC.warmup_steps)), float(sched(TC.decay_end))
    ok = s0 == 0.0 and abs(sw - TC.lr) < 1e-9 and abs(st_ - TC.end_lr) < 1e-9
    failures += [] if ok else [f"schedule {s0} {sw} {st_}"]
    print(f"[{'PASS' if ok else 'FAIL'}] sched 0 / warmup / decay end          {s0:.1e} {sw:.1e} {st_:.1e}")

    # 5. fast_weight_lr_scale: the W_0 template's first update (Adam step + decay) times the scale, every other
    #    leaf's unchanged; the default adds no transformation (the optimiser state keeps its structure).
    g = jax.tree_util.tree_map(lambda p: jax.random.normal(jax.random.PRNGKey(1), p.shape, p.dtype), params)

    def first_update(tc):
        o, _ = make_optimizer(tc)
        return o.update(g, o.init(params), params)[0]
    u1, uh = first_update(TC), first_update(dataclasses.replace(TC, fast_weight_lr_scale=0.5))
    e_fw = e_rest = 0.0
    n_fw = 0
    for (path, a), b in zip(jax.tree_util.tree_flatten_with_path(u1)[0], jax.tree_util.tree_leaves(uh)):
        if re.search(r"(^|\.)fast_weight(\.|$)", jax.tree_util.keystr(path)):
            n_fw += 1
            e_fw = max(e_fw, float(jnp.max(jnp.abs(b - 0.5 * a))) / (float(jnp.max(jnp.abs(a))) + 1e-12))
        else:
            e_rest = max(e_rest, float(jnp.max(jnp.abs(b - a))))
    plain = len(make_optimizer(TC)[0].init(params)) == 2
    ok = n_fw > 0 and e_fw < 1e-6 and e_rest == 0.0 and plain
    failures += [] if ok else [f"fast_weight_lr_scale: {n_fw} leaves, rel {e_fw:.1e}, rest {e_rest:.1e}, plain {plain}"]
    print(f"[{'PASS' if ok else 'FAIL'}] fast_weight_lr_scale 0.5: W_0 halved     {n_fw} leaves, rel {e_fw:.1e}, rest {e_rest:.1e}")

    print("ALL PASS" if not failures else "FAILURES: " + "; ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
