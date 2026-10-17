# coding=utf-8
"""
From-scratch initialisation.

Every matrix ~ normal(initializer_range), biases zero, norms one, no RoPE leaf,
and the initial CE on MASKED positions is ln(V) - the one number that catches a
broken init, broken tying, or a broken mask token before any pod time is spent.
"""
from __future__ import annotations

import os
import subprocess
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import equinox as eqx
import jax
import jax.numpy as jnp

from src.ttt.config import ModelConfig
from src.ttt.model import Model

STD = 0.02
CFG = ModelConfig(
    vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
    num_attention_heads=4, num_key_value_heads=2, head_dim=16, suffix_len=1,
    chunk_size=8, window_chunks=1, mask_token_id=500, pad_token_id=0, eos_token_id=0,
    initializer_range=STD, attention_bias=True, tie_word_embeddings=True,
)


def main():
    failures = []
    m = Model(CFG, jax.random.PRNGKey(0))
    named = jax.tree_util.tree_flatten_with_path(eqx.filter(m, eqx.is_array))[0]

    # 1. Every weight matrix has std ~ initializer_range (stacked leaves included).
    bad = []
    for path, x in named:
        n = jax.tree_util.keystr(path)
        if n.endswith(".bias") or "norm" in n.lower():
            continue
        sd = float(jnp.std(x))
        if not 0.85 * STD < sd < 1.15 * STD:
            bad.append(f"{n}: {sd:.4f}")
    ok = not bad
    failures += [] if ok else ["weights off initializer_range: " + "; ".join(bad[:3])]
    print(f"[{'PASS' if ok else 'FAIL'}] all weights std ~ {STD}      {len(named)} leaves, {len(bad)} off")

    # 2. The embedding specifically (Equinox's default was std 1.0 - 50x too large).
    es = float(jnp.std(m.embed))
    ok = 0.85 * STD < es < 1.15 * STD
    failures += [] if ok else [f"embedding std {es:.4f}"]
    print(f"[{'PASS' if ok else 'FAIL'}] embedding std ~ {STD}        {es:.5f}")

    # 3. Biases exactly 0, norm gains exactly 1.
    nb = [jax.tree_util.keystr(p) for p, x in named
          if jax.tree_util.keystr(p).endswith(".bias") and float(jnp.max(jnp.abs(x))) != 0.0]
    nn_ = [jax.tree_util.keystr(p) for p, x in named
           if "norm" in jax.tree_util.keystr(p).lower() and float(jnp.max(jnp.abs(x - 1.0))) != 0.0]
    ok = not nb and not nn_
    failures += [] if ok else [f"bias/norm init wrong: {nb[:2]} {nn_[:2]}"]
    print(f"[{'PASS' if ok else 'FAIL'}] biases 0, norms 1")

    # 4. No RoPE parameter leaf exists to be trained or protected.
    ok = not any("inv_freq" in jax.tree_util.keystr(p) or "rope" in jax.tree_util.keystr(p).lower()
                 for p, _ in named)
    failures += [] if ok else ["a RoPE table is a parameter leaf"]
    print(f"[{'PASS' if ok else 'FAIL'}] no RoPE parameter leaf")

    # 5. Initial CE ~ ln(V) on MASKED positions - as training scores them.  On
    #    clean positions a tied model reads ~ln(V)-1 at init (h_t ~ e_{ids[t]}, so
    #    the self-logit is ~|e|^2): a property to know, not a bug.
    T = 64
    ids = jax.random.randint(jax.random.PRNGKey(2), (T,), 1, CFG.vocab_size)
    logits = m.forward(jnp.full_like(ids, CFG.mask_token_id))
    ce = float(-jnp.mean(jax.nn.log_softmax(logits, axis=-1)[jnp.arange(T), ids]))
    ref = float(jnp.log(jnp.array(CFG.vocab_size)))
    ok = abs(ce - ref) < 0.05
    failures += [] if ok else [f"initial CE {ce:.4f} != ln(V) {ref:.4f}"]
    print(f"[{'PASS' if ok else 'FAIL'}] initial CE ~ ln(V) on masked  {ce:.4f} vs {ref:.4f}")

    # 6. Determinism across PYTHONHASHSEED (no hash()-derived keys anywhere).
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    snippet = (
        "import os,sys;sys.path.insert(0,%r);os.environ['JAX_PLATFORMS']='cpu';"
        "import jax,jax.numpy as jnp;from src.ttt.config import ModelConfig;from src.ttt.model import Model;"
        "c=ModelConfig(vocab_size=512,hidden_size=64,intermediate_size=128,num_hidden_layers=4,"
        "num_attention_heads=4,num_key_value_heads=2,head_dim=16,suffix_len=1,chunk_size=8,window_chunks=1,"
        "mask_token_id=500,pad_token_id=0,eos_token_id=0);"
        "m=Model(c,jax.random.PRNGKey(0));print(float(jnp.sum(m.embed**2)))"
    ) % root
    outs = []
    for seed in ("0", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        r = subprocess.run([sys.executable, "-c", snippet], capture_output=True, text=True, env=env)
        outs.append(r.stdout.strip().splitlines()[-1] if r.stdout.strip() else f"ERR {r.stderr[-120:]}")
    ok = outs[0] == outs[1] and not outs[0].startswith("ERR")
    failures += [] if ok else [f"init differs across PYTHONHASHSEED: {outs}"]
    print(f"[{'PASS' if ok else 'FAIL'}] init stable vs PYTHONHASHSEED  {outs[0][:18]}")

    print("ALL PASS" if not failures else "FAILURES: " + "; ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
