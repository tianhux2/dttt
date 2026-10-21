# coding=utf-8
"""
Llama-3 RoPE scaling (``ModelConfig.rope_scaling``): the inverse frequencies match an independent numpy
implementation of HF's ``_compute_llama3_parameters`` in all three bands and at both band edges; without scaling
the rotary tables are bitwise those of the plain formula; the configuration validates, stays hashable and
round-trips through its JSON file.
"""
from __future__ import annotations

import math
import os
import sys
import tempfile

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import dataclasses

import jax.numpy as jnp
import numpy as np

from src.ttt.config import ModelConfig
from src.ttt.model import Model, llama3_inv_freq, rope_cos_sin

LLAMA = {"rope_type": "llama3", "factor": 32.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
         "original_max_position_embeddings": 8192}
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def ref_llama3(inv: np.ndarray, s: dict) -> np.ndarray:
    """HF's formula, element by element in float64."""
    old, low, high, factor = (s["original_max_position_embeddings"], s["low_freq_factor"], s["high_freq_factor"],
                              s["factor"])
    out = []
    for f in inv.astype(np.float64):
        wl = 2 * math.pi / f
        if wl < old / high:
            out.append(f)
        elif wl > old / low:
            out.append(f / factor)
        else:
            sm = (old / wl - low) / (high - low)
            out.append((1 - sm) * f / factor + sm * f)
    return np.asarray(out)


def main():
    failures = []

    # 1. head_dim 64, theta 5e5, factor 32 over 8192 positions: all three bands occur and match the reference.
    d, theta = 64, 500000.0
    inv = 1.0 / (theta ** (np.arange(0, d, 2, dtype=np.float32) / d))
    wl = 2 * np.pi / inv.astype(np.float64)
    bands = (int((wl < 2048).sum()), int(((wl >= 2048) & (wl <= 8192)).sum()), int((wl > 8192).sum()))
    got = np.asarray(llama3_inv_freq(jnp.asarray(inv), tuple(sorted(LLAMA.items()))), np.float64)
    err = float(np.max(np.abs(got - ref_llama3(inv, LLAMA)) / ref_llama3(inv, LLAMA)))
    ok = err < 1e-6 and min(bands) > 0
    failures += [] if ok else [f"llama3 inv_freq: rel err {err:.2e}, bands {bands}"]
    print(f"[{'PASS' if ok else 'FAIL'}] llama3 inv_freq vs reference   bands kept/smoothed/scaled {bands}: "
          f"max rel err = {err:.2e}")

    # 2. The band edges (wavelength old/high and old/low) and just beyond them: continuous, as the reference.
    edges = np.array([2048 * 0.999, 2048.0, 2048 * 1.001, 8192 * 0.999, 8192.0, 8192 * 1.001])
    inv_e = (2 * np.pi / edges).astype(np.float32)
    got_e = np.asarray(llama3_inv_freq(jnp.asarray(inv_e), tuple(sorted(LLAMA.items()))), np.float64)
    ref_e = ref_llama3(inv_e, LLAMA)
    err_e = float(np.max(np.abs(got_e - ref_e) / ref_e))
    ok = err_e < 1e-5
    failures += [] if ok else [f"band edges: rel err {err_e:.2e}"]
    print(f"[{'PASS' if ok else 'FAIL'}] band edges                     max rel err = {err_e:.2e}")

    # 3. Without scaling: bitwise the plain formula (the code before rope_scaling existed).
    pos = jnp.arange(0, 131072, 97)
    cos, sin = rope_cos_sin(pos, 128, 1_000_000.0)
    inv0 = 1.0 / (1_000_000.0 ** (jnp.arange(0, 128, 2, dtype=jnp.float32) / 128))
    fr = pos.astype(jnp.float32)[:, None] * inv0[None, :]
    emb = jnp.concatenate([fr, fr], axis=-1)
    ok = bool(jnp.array_equal(cos, jnp.cos(emb)) and jnp.array_equal(sin, jnp.sin(emb)))
    cos_s, _ = rope_cos_sin(pos, 128, 1_000_000.0, tuple(sorted(LLAMA.items())))
    ok = ok and not bool(jnp.array_equal(cos_s, cos))
    failures += [] if ok else ["rope without scaling is not bitwise the plain formula"]
    print(f"[{'PASS' if ok else 'FAIL'}] no scaling == plain RoPE bitwise (and scaling changes it)")

    # 4. The configuration: a scaling given as a dict is held as sorted pairs (hashable), round-trips, reaches
    #    Attention; 1b_e2e has none; bad scalings are refused.
    old = ModelConfig.load(os.path.join(REPO, "configs", "1b_e2e"))
    cfg = dataclasses.replace(old, rope_theta=500000.0, rope_scaling=LLAMA)
    ok = dict(cfg.rope_scaling) == LLAMA and isinstance(hash(cfg), int)
    with tempfile.TemporaryDirectory() as tmp:
        cfg.save(os.path.join(tmp, "config.json"))
        back = ModelConfig.load(tmp)
    ok = ok and back == cfg
    tiny = dataclasses.replace(cfg, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                               num_key_value_heads=2, head_dim=16, suffix_len=1, ttt_intermediate_size=32,
                               vocab_size=256, mask_token_id=250, chunk_size=8)
    m = Model(tiny, __import__("jax").random.PRNGKey(0))
    ok = ok and dict(m.prefix.attn.rope_scaling) == LLAMA
    ok = ok and old.rope_scaling is None
    for bad in ({**LLAMA, "rope_type": "yarn"}, {k: v for k, v in LLAMA.items() if k != "factor"},
                {**LLAMA, "low_freq_factor": 4.0, "high_freq_factor": 1.0}):
        try:
            dataclasses.replace(cfg, rope_scaling=bad)
            ok = False
        except ValueError:
            pass
    failures += [] if ok else ["rope_scaling configuration handling"]
    print(f"[{'PASS' if ok else 'FAIL'}] config: loads, hashable, round-trips, reaches Attention; 1b_e2e none; "
          f"bad scalings refused")

    print("ALL PASS" if not failures else "FAILURES: " + "; ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
