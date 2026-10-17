# coding=utf-8
"""The key transplant (``tools/j_lens/key_transplant.py``, ``core.Transplant`` / ``core.write_keys``) on a
tiny random model:

1. transplanting the read's own inputs at every slot is the plain read;
2. with the memory at W_0 a transplant changes nothing (the memory's part is zero at any input);
3. ``write_keys`` runs the write's forward: its logits give the write's inner loss (``inner_loop.inner_loss``);
4. the planted pairing: every scored read slot maps to a source slot with the same token and next token;
5. the command line runs end to end and reports every variant.

    JAX_PLATFORMS=cpu .venv/bin/python tests/test_key_transplant.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TTT_ATTENTION", "xla")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_default_matmul_precision", "highest")

from src.ttt import probes as PR
from src.ttt.attention import key_validity, own_mask_matrix
from src.ttt.config import ModelConfig
from src.ttt.generate import _split, block_forward
from src.ttt.inner_loop import inner_loss
from src.ttt.model import Model
from tools.j_lens import core
from tools.j_lens.key_transplant import match

CFG = dict(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
           num_key_value_heads=2, head_dim=8, suffix_len=2, chunk_size=32, window_chunks=3,
           mask_token_id=127, pad_token_id=0, eos_token_id=0)
C, T, LEN, CUE = 32, 256, 16, 4
F32 = jnp.float32


def main():
    failures = []

    def report(name, ok, detail=""):
        failures.extend([] if ok else [name])
        print(f"[{'PASS' if ok else 'FAIL'}] {name:72s} {detail}")

    model = Model(ModelConfig(**CFG), jax.random.PRNGKey(0))
    ws = core.WriteSettings(0.3, inner_task="causal", fused_ce_block=0)
    seq = np.random.default_rng(1).integers(1, 100, T).astype(np.int32)
    st = core.init_memory(model, F32)
    for c in range(4):                                   # four chunks written, the fifth read
        st = core.commit(model, jnp.asarray(seq[c * C:(c + 1) * C]), jnp.arange(c * C, (c + 1) * C), st,
                         jax.random.PRNGKey(c), ws, F32)
    pos = jnp.arange(4 * C, 5 * C)
    ids, _, _ = core.read_mask(jnp.asarray(seq[4 * C:5 * C]), 0.5, jax.random.PRNGKey(9), model.cfg.mask_token_id)

    # 1. the read's own inputs transplanted everywhere: the plain read
    plain = core.read_forward(model, ids, pos, st, None, True, F32, ref=True)
    tx = core.Transplant(plain.mem_in, jnp.ones(C, bool))
    d = float(jnp.max(jnp.abs(core.read_forward(model, ids, pos, st, None, True, F32, transplant=tx).logits - plain.logits)))
    report("transplanting the read's own inputs is the plain read", d < 1e-4, f"max |dlogit| {d:.1e}")

    # 2. memory at W_0: any transplant changes nothing
    st0 = core.reset_memory(model, st)
    rnd = core.Transplant(jax.random.normal(jax.random.PRNGKey(5), plain.mem_in.shape), jnp.ones(C, bool))
    a = core.read_forward(model, ids, pos, st0, None, True, F32).logits
    b = core.read_forward(model, ids, pos, st0, None, True, F32, transplant=rnd).logits
    d = float(jnp.max(jnp.abs(a - b)))
    report("with the memory at W_0 a transplant changes nothing", d < 1e-5, f"max |dlogit| {d:.1e}")
    d = float(jnp.max(jnp.abs(core.read_forward(model, ids, pos, st, None, True, F32, transplant=rnd).logits - plain.logits)))
    report("... while with the written memory it does", d > 1e-3, f"max |dlogit| {d:.1e}")

    # 3. write_keys is the write's forward (the forward whose loss the write descends)
    clean, posc = jnp.asarray(seq[4 * C:5 * C]), jnp.arange(4 * C, 5 * C)
    keys = core.write_keys(model, clean, posc, st, ws, F32)
    pos_q = jnp.concatenate([posc[1:], posc[-1:]])
    own = jnp.asarray(own_mask_matrix(C, "causal"))
    logits, h_in, _, _ = block_forward(model, clean, posc, st, F32, pos_q=pos_q, own_mask=own, return_prime=True)
    tgt = jnp.concatenate([clean[1:], clean[-1:]])
    lp = jax.nn.log_softmax(logits, axis=-1)
    ce_fwd = float(-jnp.mean(jnp.take_along_axis(lp, tgt[:, None], axis=-1)[:-1, 0]))
    w = jnp.arange(C) <= C - 2
    ce_write = float(inner_loss(st.W, model, h_in, tgt, w, _split(model, st.kv)[1], posc,
                                key_validity(st.chunk, C, model.cfg.window_chunks), 0, F32, pos_q=pos_q, own_mask=own))
    report("write_keys' forward gives the write's inner loss", abs(ce_fwd - ce_write) < 1e-5 * abs(ce_write),
           f"{ce_fwd:.6f} vs {ce_write:.6f}; keys {tuple(keys.shape)}")

    # 4. the planted pairing
    cs, ct, cw = PR.planted_chunks(T // C, model.cfg.window_chunks)
    ok, n = True, 0
    for j, mode in enumerate(("alt", "block", "alt")):
        s = np.random.default_rng(10 + j).integers(1, 100, T).astype(np.int32)
        pid, lm = PR.planted_probe_batch(s[None], [j], 12345, C, model.cfg.window_chunks, LEN, CUE, "beyond", mode, eos=0)
        sel, src = match(pid[0], lm[0], C, cs, ct, LEN, CUE, mode)
        rs, ss = np.flatnonzero(sel) + ct * C, src[sel] + cs * C
        ok &= sel.sum() == lm[0].sum() and all(pid[0][r] == pid[0][q] and pid[0][r + 1] == pid[0][q + 1]
                                               for r, q in zip(rs, ss))
        n += int(sel.sum())
    report("planted pairing: same token and same next token at every scored slot", bool(ok), f"{n} slots")

    # 5. the command line, end to end
    tmp = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmp, "cfg"))
    json.dump(CFG, open(os.path.join(tmp, "cfg", "config.json"), "w"))
    out = os.path.join(tmp, "kt.npz")
    env = dict(os.environ, JAX_PLATFORMS="cpu", TTT_ATTENTION="xla")
    r = subprocess.run([sys.executable, "-m", "tools.j_lens.key_transplant", "--random_init", "--model_config",
                        os.path.join(tmp, "cfg"), "--inner_lr", "0.3", "--synthetic", "100", "--n_per_source", "3",
                        "--seq_len", str(T), "--length", str(LEN), "--cue", str(CUE), "--out", out],
                       capture_output=True, text=True, env=env, cwd=ROOT, timeout=1200)
    ok = r.returncode == 0 and os.path.exists(out)
    if ok:
        z = np.load(out)
        s = json.loads(str(z["summary"]))
        ok = (z["sums"].shape == (3, 10) and np.all(np.isfinite(z["sums"])) and z["counts"].sum() > 0
              and all(k in s for k in ("ceiling", "gain_on", "gain_on_tx", "recall_one_tx", "transplant_effect")))
    report("command line: every variant, finite", ok, "" if ok else (r.stdout[-800:] + r.stderr[-1500:]))
    if ok:
        print("\n".join("    " + l for l in r.stdout.splitlines() if l.startswith("[kt]") and "sequence" not in l))

    print("ALL PASS" if not failures else "FAILURES: " + "; ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
