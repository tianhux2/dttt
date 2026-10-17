# coding=utf-8
"""``tools/j_lens/doc_switch.py`` on a tiny random model and synthetic streams:

1. the switch sequences: whole held-out documents only, one EOS exactly at the switch, cross and same pairs;
2. at the switch's first chunk the reset memory is W_0 with the carry state's windows: reset == off;
3. with the inner lr 0 nothing is written: carry == reset == off at every chunk; with a write, A's content changes
   B's reads (carry != reset) and the reset variant is undefined before the switch;
4. the command line end to end with passkeys asked in B (finite numbers, the summary's rows).

    JAX_PLATFORMS=cpu .venv/bin/python tests/test_doc_switch.py
"""
from __future__ import annotations

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

import jax
import jax.numpy as jnp
import numpy as np

from src.ttt.config import ModelConfig
from src.ttt.model import Model
from tools.j_lens import core
from tools.j_lens.doc_switch import run, summarise, switch_sequences

BASE = dict(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
            num_key_value_heads=2, head_dim=8, suffix_len=2, chunk_size=8, window_chunks=2,
            mask_token_id=120, pad_token_id=0, eos_token_id=0)
failures = []


def report(name, ok, detail=""):
    failures.append(name) if not ok else None
    print(f"[{'PASS' if ok else 'FAIL'}] {name:<92} {detail}")


def stream(d, docs):
    """A token stream of ``docs`` (EOS 0 after each), from a leading EOS so every document is whole."""
    os.makedirs(d)
    np.concatenate([[0]] + [np.concatenate([x, [0]]) for x in docs]).astype("<u4").tofile(os.path.join(d, "tokens.u32"))
    return d


def main():
    rng = np.random.default_rng(0)
    C = BASE["chunk_size"]
    n_a, n_b = 4, 3
    with tempfile.TemporaryDirectory() as tmp:
        # 1. the sequences
        docs = {s: [rng.integers(1, 100, int(n)).astype(np.int32) for n in (40, 5, 50, 33, 60)] for s in ("a", "b")}
        srcs = [(s, stream(os.path.join(tmp, s), docs[s])) for s in ("a", "b")]
        seqs, kinds = switch_sequences(srcs, n_a, n_b, C, 1, 10 ** 6, 0)
        long_docs = {s: [x for x in docs[s] if len(x) >= max(n_a, n_b) * C] for s in docs}
        A0, B_cross, B_same = long_docs["a"][0], long_docs["b"][0], long_docs["a"][1]
        seq_cross, seq_same = seqs[0][1], seqs[1][1]
        ok = (len(seqs) == 4 and kinds == ["cross", "same", "cross", "same"] and [n for n, _ in seqs][:2] == ["a>b", "a>a"]
              and all(len(s) == (n_a + n_b) * C and np.flatnonzero(s == 0).tolist() == [n_a * C - 1] for _, s in seqs)
              and np.array_equal(seq_cross[:n_a * C - 1], A0[:n_a * C - 1])
              and np.array_equal(seq_cross[n_a * C:], B_cross[:n_b * C])
              and np.array_equal(seq_same[n_a * C:], B_same[:n_b * C]))
        report("switch sequences: whole documents (the 5-token one skipped), one EOS at the switch, cross / same", ok,
               f"{len(seqs)} sequences, EOS at {np.flatnonzero(seq_cross == 0).tolist()}")

        # 2. / 3. run
        model = Model(ModelConfig(**BASE), jax.random.PRNGKey(1))
        ws = core.WriteSettings(0.5, fused_ce_block=0)
        quiet = lambda *a, **k: None
        o = run(model, seqs, ws, n_a, [0.4, 0.8], 3, jnp.float32, log=quiet)
        ce = o["ce"]
        first = np.abs(ce[:, n_a, 1] - ce[:, n_a, 2]).max()
        later = np.abs(ce[:, n_a + 1:, 0] - ce[:, n_a + 1:, 1]).max()
        ok = (first < 1e-4 and later > 1e-4 and np.isnan(ce[:, :n_a, 1]).all() and np.isfinite(ce[:, n_a:]).all()
              and np.isfinite(ce[:, :, [0, 2]]).all() and (o["count"] > 0).all())
        report("at the switch reset == off; afterwards A's content changes B's reads; reset undefined before", ok,
               f"|reset - off| at the switch {first:.1e}, max |carry - reset| later {later:.1e}")
        o0 = run(model, seqs, ws._replace(inner_lr=0.0), n_a, [0.4, 0.8], 3, jnp.float32, log=quiet)
        d0 = max(np.abs(o0["ce"][:, :, 0] - o0["ce"][:, :, 2]).max(),
                 np.abs(o0["ce"][:, n_a:, 0] - o0["ce"][:, n_a:, 1]).max())
        report("inner lr 0: nothing written, carry == reset == off at every chunk", d0 < 1e-5, f"max diff {d0:.1e}")
        s = summarise(o, kinds, n_a, n_boot=50)
        offs = sorted({r["offset"] for r in s["rows"] if r["kind"] == "all"})
        b_row = next(r for r in s["rows"] if r["kind"] == "cross" and r["offset"] == 1)
        ok = (offs == list(range(-n_a, n_b)) and {"stale", "gain", "net"} <= set(b_row)
              and abs(b_row["stale"][0] - (b_row["carry"] - b_row["reset"])) < 1e-9)
        report("summary: one row per offset and kind, stale = carry - reset", ok, f"offsets {offs}")

        # 4. the command line, with passkeys asked in B
        cfg_dir = os.path.join(tmp, "cfg")
        os.makedirs(cfg_dir)
        json.dump(dict(BASE, chunk_size=512, window_chunks=3), open(os.path.join(cfg_dir, "config.json"), "w"))
        big = {s: [rng.integers(1, 100, 3000).astype(np.int32) for _ in range(3)] for s in ("x", "y")}
        spec = ",".join(f"{s}={stream(os.path.join(tmp, 'big_' + s), big[s])}:1.0" for s in ("x", "y"))
        out = os.path.join(tmp, "switch.npz")
        r = subprocess.run([sys.executable, "-m", "tools.j_lens.doc_switch", "--random_init", "--model_config", cfg_dir,
                            "--inner_lr", "0.5", "--sources", spec, "--holdout", "100000", "--n_a", "5", "--n_b", "3",
                            "--n_per_pair", "1", "--t", "0.5", "--targets", "7", "--tasks", "key1", "--out", out],
                           cwd=ROOT, env=dict(os.environ, JAX_PLATFORMS="cpu", TTT_ATTENTION="xla", PYTHONPATH=ROOT),
                           capture_output=True, text=True, timeout=1800)
        ok = r.returncode == 0 and os.path.exists(out)
        detail = "" if ok else (r.stdout[-400:] + r.stderr[-1200:])
        if ok:
            z = np.load(out, allow_pickle=True)
            sm = json.loads(str(z["summary"]))
            # cl_ce [n, tasks, targets, placements (beyond, within, none), (on, off)]: the l2r read never reads
            # "within, on" (context_learning.read_l2r leaves it NaN), every other cell is a finite CE sum
            read = np.ones(z["cl_ce"].shape, bool) if "cl_ce" in z.files else None
            if read is not None:
                read[:, :, :, 1, 0] = False
            ok = (z["ce"].shape == (4, 8, 3, 1) and np.isfinite(z["ce"][:, 5:]).all() and read is not None
                  and np.isfinite(z["cl_ce"][read]).all()
                  and len(sm["passkeys"]["rows"]) == 1 and sm["passkeys"]["rows"][0]["task"] == "key1")
            detail = (f"ce {z['ce'].shape} finite from the switch {bool(np.isfinite(z['ce'][:, 5:]).all())}, "
                      f"cl_ce read cells finite {read is not None and bool(np.isfinite(z['cl_ce'][read]).all())}, "
                      f"passkey rows {len(sm['passkeys']['rows'])}")
        report("command line: switch reads and passkeys asked in B, finite", ok, detail)
    print("ALL PASS" if not failures else "FAILURES: " + "; ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
