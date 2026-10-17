# coding=utf-8
"""The TPU prefix path on CPU: ``prefix_call``'s splash kernel over the static chunk-window mask, run in Pallas
interpret mode, against the reference path (gathered window, XLA attention) - for the research recipe (causal write,
outer shift, CART, second order) and the training options the GPU branch added.  The suffix's flash kernel has no
interpret mode; it is the TPU-era kernel, fed only the key validity, so here it is the XLA reference on both sides.

1. loss and meta-gradient, splash prefix == reference: the plain recipe; the normalised write; the self-prior; two
   noisy reads (``outer_views``: extra bidirectional prefix streams); segments (the window carried across);
2. the options the splash prefix cannot express are refused, not run wrong: ``window_reach``, ``window_drop`` and
   ``read_block``.

    JAX_PLATFORMS=cpu .venv/bin/python tests/test_splash_prefix_cpu.py
"""
from __future__ import annotations

import dataclasses
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as sk

_make = sk.make_splash_mha
sk.make_splash_mha = lambda *a, **k: _make(*a, **{**k, "interpret": True})      # the TPU kernel, interpreted

from src.ttt import attention as A
from src.ttt.chunk_scan import Batch, LossConfig, loss_fn
from src.ttt.config import ModelConfig
from src.ttt.inner_loop import WriteConfig
from src.ttt.model import Model

A.attend_flash = lambda q, k, v, kv, scale, kv_heads: A.attend_xla(q, k, v, kv, scale, kv_heads)   # suffix: reference

# the smallest shapes the splash kernel accepts (128-multiples), window 2
CFG = dict(vocab_size=256, hidden_size=128, intermediate_size=256, num_hidden_layers=4, num_attention_heads=4,
           num_key_value_heads=2, head_dim=32, suffix_len=2, chunk_size=64, window_chunks=2,
           mask_token_id=250, pad_token_id=0, eos_token_id=0)
failures = []


def report(name, ok, detail=""):
    failures.append(name) if not ok else None
    print(f"[{'PASS' if ok else 'FAIL'}] {name:72s} {detail}")


def leaves(t):
    return jax.tree_util.tree_leaves(eqx.filter(t, eqx.is_inexact_array))


def run(impl, model, lc, ids, key):
    A.ATTENTION_IMPL = impl
    params, static = eqx.partition(model, eqx.is_array)
    f = lambda pr: loss_fn(eqx.combine(pr, static), Batch(input_ids=ids), key, jnp.int32(5), lc)[0]
    return jax.value_and_grad(f)(params)


def main():
    model = Model(ModelConfig(**CFG), jax.random.PRNGKey(0))
    ids = jax.random.randint(jax.random.PRNGKey(3), (2, 4 * CFG["chunk_size"]), 1, 240)
    key = jax.random.PRNGKey(11)
    base = LossConfig(time_reweighting="cart", inner_lr=0.5, inner_task="causal", outer_shift=True, fused_ce_block=0,
                      compute_dtype="float32", attn_chunk_batch=2, second_order=True)
    cases = {"research recipe (causal write, outer shift, CART, 2nd order)": base,
             "normalised write (alpha 1, cap 4)": dataclasses.replace(base, write=WriteConfig(norm_alpha=1.0, norm_cap=4.0)),
             "self-prior 0.5": dataclasses.replace(base, self_prior=0.5),
             "two noisy reads (outer_views 2)": dataclasses.replace(base, outer_views=2),
             "segments of 2 chunks (the window carried across)": dataclasses.replace(base, segment_chunks=2)}
    for name, lc in cases.items():
        l_s, g_s = run("flash", model, lc, ids, key)
        l_x, g_x = run("xla", model, lc, ids, key)
        scale = max(float(jnp.max(jnp.abs(b))) for b in leaves(g_x))
        err = max(float(jnp.max(jnp.abs(a - b))) for a, b in zip(leaves(g_s), leaves(g_x)))
        report(f"splash prefix == reference: {name}", abs(float(l_s) - float(l_x)) < 1e-5 and err < 1e-4 * scale,
               f"loss {float(l_s):.6f} / {float(l_x):.6f}; grad max diff {err:.1e} (scale {scale:.1e})")

    refused = []
    for name, m, lc in (("window_reach 0", Model(ModelConfig(**CFG, window_reach=0), jax.random.PRNGKey(0)), base),
                        ("window_drop 0.5", model, dataclasses.replace(base, window_drop=0.5, window_drop_until=100)),
                        ("read_block 32", Model(ModelConfig(**CFG, read_block=32), jax.random.PRNGKey(0)), base)):
        A.ATTENTION_IMPL = "flash"
        try:
            loss_fn(m, Batch(input_ids=ids), key, jnp.int32(5), lc)
        except NotImplementedError:
            refused.append(name)
    report("window_reach / window_drop / read_block refused on the splash prefix", len(refused) == 3,
           f"refused: {refused}")

    print("ALL PASS" if not failures else "FAILURES: " + "; ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
