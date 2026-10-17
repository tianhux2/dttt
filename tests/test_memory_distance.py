# coding=utf-8
"""The planted probe at a chosen target chunk (``probes.planted_chunks`` / ``planted_probe_batch``, ``target``) and
the recall-over-distance measurement (``tools/j_lens/memory_distance.py``) on a tiny random model:

1. ``planted_chunks``: the default target is ``window + 3``; a later target moves only the copy; targets before
   ``window + 3`` or past the sequence are refused;
2. ``planted_probe_batch``: target ``window + 3`` is the default exactly; a later target plants the same span at
   the same offset in its own chunk, the scored slots shifted by whole chunks, and ``match`` pairs it;
3. the one write pass: at the farther target, the copy's own-key gain, the ceiling and the isolated source write's
   gain equal an independent computation (a fresh pass over the planted sequence, the in-window copy written for
   the ceiling, W_0 + the source chunk's write for the isolated memory);
4. the command line runs end to end: every number finite, one row per distance, the memory drifts further with
   distance.

    JAX_PLATFORMS=cpu .venv/bin/python tests/test_memory_distance.py
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

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_default_matmul_precision", "highest")

from src.ttt import probes as PR
from src.ttt.config import ModelConfig
from src.ttt.model import Model, cast_floating
from tools.j_lens import core
from tools.j_lens.key_transplant import match
from tools.j_lens.memory_distance import KEYS, run

CFG = dict(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
           num_key_value_heads=2, head_dim=8, suffix_len=2, chunk_size=32, window_chunks=3,
           mask_token_id=127, pad_token_id=0, eos_token_id=0)
C, NC, LEN, CUE, SEED = 32, 16, 16, 4, 12345
T = C * NC
F32 = jnp.float32


def main():
    failures = []

    def report(name, ok, detail=""):
        failures.extend([] if ok else [name])
        print(f"[{'PASS' if ok else 'FAIL'}] {name:72s} {detail}")

    Wn = CFG["window_chunks"]

    # 1. planted_chunks
    refused = 0
    for bad in (Wn + 2, NC):
        try:
            PR.planted_chunks(NC, Wn, bad)
        except ValueError:
            refused += 1
    ok = (PR.planted_chunks(NC, Wn) == (1, Wn + 3, Wn + 2) == PR.planted_chunks(NC, Wn, Wn + 3)
          and PR.planted_chunks(NC, Wn, 10) == (1, 10, 9) and refused == 2)
    report("planted_chunks: default window + 3, a later target, bad targets refused", ok)

    # 2. planted_probe_batch at a later target
    seq = np.random.default_rng(1).integers(1, 100, T).astype(np.int32)
    ok = True
    for p in PR.PLANT_PLACEMENTS:
        d_ids, d_lm = PR.planted_probe_batch(seq[None], [0], SEED, C, Wn, LEN, CUE, p, "alt", eos=0)
        t_ids, t_lm = PR.planted_probe_batch(seq[None], [0], SEED, C, Wn, LEN, CUE, p, "alt", eos=0, target=Wn + 3)
        ok &= np.array_equal(d_ids, t_ids) and np.array_equal(d_lm, t_lm)
    ids6, lm6 = PR.planted_probe_batch(seq[None], [0], SEED, C, Wn, LEN, CUE, "beyond", "alt", eos=0)
    ids10, lm10 = PR.planted_probe_batch(seq[None], [0], SEED, C, Wn, LEN, CUE, "beyond", "alt", eos=0, target=10)
    s6, s10 = np.flatnonzero(lm6[0]), np.flatnonzero(lm10[0])
    ok &= np.array_equal(s10, s6 + (10 - (Wn + 3)) * C)
    b = s6[0] - CUE - 1 - (Wn + 3) * C                                # the copy's offset in its chunk (alt)
    ok &= np.array_equal(ids10[0, 10 * C + b:10 * C + b + LEN], ids6[0, (Wn + 3) * C + b:(Wn + 3) * C + b + LEN])
    ok &= np.array_equal(ids10[0, :10 * C], seq[:10 * C]) and np.array_equal(ids10[0, 11 * C:], seq[11 * C:])
    sel, src = match(ids10[0], lm10[0], C, 1, 10, LEN, CUE, "alt")
    ok &= sel.sum() == len(s10)
    report("planted_probe_batch: target window + 3 is the default; a later target moves the copy only", bool(ok))

    # 3. the one write pass against an independent computation at the farther target
    model = Model(ModelConfig(**CFG), jax.random.PRNGKey(0))
    ws = core.WriteSettings(0.3, inner_task="causal", fused_ce_block=0)
    out, targets = run(model, [seq], ws, [5, 9], LEN, CUE, "alt", SEED, F32, log=lambda *_: None)
    t = 10
    commit = lambda ids, c, st: core.commit(model, jnp.asarray(ids[c * C:(c + 1) * C]), jnp.arange(c * C, (c + 1) * C),
                                            st, jax.random.PRNGKey(0), ws, F32)
    pl = {p: PR.planted_probe_batch(seq[None], [0], SEED, C, Wn, LEN, CUE, p, "alt", eos=0, target=t)
          for p in ("beyond", "within")}
    ib, lb = pl["beyond"][0][0], pl["beyond"][1][0]
    iw, lw = pl["within"][0][0], pl["within"][1][0]
    st = core.init_memory(model, F32)
    for c in range(t - 1):
        if c == 1:
            W_before = st.W
        st = commit(ib, c, st)
        if c == 1:
            W_after = st.W
    st_w = commit(iw, t - 1, st)
    st = commit(ib, t - 1, st)
    sel, _ = match(ib, lb, C, 1, t, LEN, CUE, "alt")
    pos = jnp.arange(t * C, (t + 1) * C)

    def ce(ids, lm, state):
        chunk, mk = ids[t * C:(t + 1) * C], lm[t * C:(t + 1) * C]
        lg = core.read_forward(model, jnp.asarray(np.where(mk, CFG["mask_token_id"], chunk)), pos, state, None, True,
                               F32).logits
        tgt = jnp.asarray(np.concatenate([chunk[1:], chunk[-1:]]))
        return np.asarray(-jnp.take_along_axis(jax.nn.log_softmax(lg, axis=-1), tgt[:, None], axis=-1)[:, 0], np.float64)
    off = ce(ib, lb, core.reset_memory(model, st))
    gain_own = float((off - ce(ib, lb, st))[sel].sum())
    ceiling = float((off - ce(iw, lw, core.reset_memory(model, st_w)))[sel].sum())
    W0 = cast_floating(model.fast_weight, F32)
    W_iso = jax.tree_util.tree_map(lambda x, y, z: x + (y - z) if eqx.is_inexact_array(x) else x, W0, W_after, W_before)
    gain_iso = float((off - ce(ib, lb, st._replace(W=W_iso)))[sel].sum())
    di = targets.index(t)
    got = (out["gain_copy"][0, di, KEYS.index("own")], out["ceiling"][0, di], out["gain_copy"][0, di, KEYS.index("isolated")])
    want = (gain_own, ceiling, gain_iso)
    err = max(abs(g - w) / max(abs(w), 1e-3) for g, w in zip(got, want))
    report("one write pass == a fresh pass at the farther target (own gain, ceiling, isolated)", err < 1e-4,
           f"max rel diff {err:.1e}; {got} vs {want}")

    # 4. the command line
    tmp = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmp, "cfg"))
    json.dump(CFG, open(os.path.join(tmp, "cfg", "config.json"), "w"))
    out_p = os.path.join(tmp, "md.npz")
    env = dict(os.environ, JAX_PLATFORMS="cpu", TTT_ATTENTION="xla")
    r = subprocess.run([sys.executable, "-m", "tools.j_lens.memory_distance", "--random_init", "--model_config",
                        os.path.join(tmp, "cfg"), "--inner_lr", "0.3", "--synthetic", "100", "--n_per_source", "2",
                        "--seq_len", str(T), "--distances", "5,9,14", "--length", str(LEN), "--cue", str(CUE),
                        "--out", out_p], capture_output=True, text=True, env=env, cwd=ROOT, timeout=1500)
    ok = r.returncode == 0 and os.path.exists(out_p)
    if ok:
        z = np.load(out_p)
        s = json.loads(str(z["summary"]))
        dr = [row["drift"] for row in s["rows"]]
        ok = (all(np.all(np.isfinite(z[k])) for k in ("gain_copy", "gain_none", "ceiling", "drift"))
              and len(s["rows"]) == 3 and [row["distance"] for row in s["rows"]] == [5, 9, 14]
              and dr[0] > 0 and dr[0] < dr[1] < dr[2])
    report("command line: finite, one row per distance, the drift grows with distance", ok,
           "" if ok else (r.stdout[-800:] + r.stderr[-1500:]))
    if ok:
        print("\n".join("    " + l for l in r.stdout.splitlines() if l.startswith("[md]") and "sequence" not in l))

    print("ALL PASS" if not failures else "FAILURES: " + "; ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
