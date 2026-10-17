# coding=utf-8
"""``tools/j_lens/context_learning.py``:

1. the task builders with the Llama 3 tokenizer: the scored tokens of a lexicon test are its nonce words (the
   Veltan twin of the English line), a ``same`` test sentence is a taught one and a ``new`` one is not; the cipher's
   scored letters are the taught substitution of new words whose letters were all taught twice; a passkey test
   scores the six digits of a taught fact; the token ids equal the whole text's; every block fits the 1B geometry
   (chunk 1024, window 3);
2. the layout: the teaching block ends with chunk TEACH_END - 1 (beyond) or right before the target (within), the
   rest is the natural text; the target chunk holds the test block, scored where the test is;
3. ``run`` on tiny random TTT and gdn models: every number equals an independent computation (each placement's
   sequence written from scratch, chunk by chunk, then the target chunk read with the memory on and reset), the
   shared natural pass and the one teaching pass over both targets notwithstanding; with the inner lr at 0 the TTT
   memory never moves (on == off exactly);
4. the command line on a tiny random model (``--random_init --synthetic``): finite numbers, one row per task and
   target.

    JAX_PLATFORMS=cpu .venv/bin/python tests/test_context_learning.py
"""
from __future__ import annotations

import json
import os
import re
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

from src.ttt.config import ModelConfig
from src.ttt.model import Model
from tools.j_lens import context_learning as CL
from tools.j_lens import core

BASE = dict(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
            num_key_value_heads=2, head_dim=8, suffix_len=2, chunk_size=128, window_chunks=2,
            mask_token_id=127, pad_token_id=0, eos_token_id=0)
SMALL = dict(lex_pairs=2, lex_reps=(1, 2), cipher_teach=10, cipher_test=1, span_len=12, span_cue=4)
TOKENIZER = os.path.join(ROOT, "configs", "1b_e2e", "tokenizer.json")
F32 = jnp.float32


def main():
    failures = []

    def report(name, ok, detail=""):
        failures.extend([] if ok else [name])
        print(f"[{'PASS' if ok else 'FAIL'}] {name:76s} {detail}")

    # 1. builders with the real tokenizer
    if os.path.exists(TOKENIZER):
        tok = CL.Tok(TOKENIZER, 128256)
        dec = lambda ids: tok.t.decode([int(i) for i in ids])
        ok_lex = ok_ci = ok_tok = ok_fit = ok_key = True
        for si in range(40):
            foreign = np.random.default_rng(si).integers(1000, 2000, 4096)
            for fam in CL.FAMILIES:
                rng = np.random.default_rng([7, list(CL.FAMILIES).index(fam), si])
                teach, tests = CL.build(fam, rng, tok, foreign, 128001, CL.SIZES)
                ok_fit &= len(teach) <= 3 * 1024 and all(len(x) <= 1023 for x, _ in tests.values())
                if fam == "span":
                    continue
                text = dec(teach)
                ok_tok &= list(tok.encode(text)) == list(teach)
                for k, (x, sc) in tests.items():
                    ok_tok &= list(tok.encode(dec(x))) == list(x)
                    if fam.startswith("lex"):
                        lines = dec(x).strip("\n").split("\n")
                        runs = np.split(np.flatnonzero(sc), np.flatnonzero(np.diff(np.flatnonzero(sc)) > 1) + 1)
                        words = [dec(x[r]).strip() for r in runs]
                        for (en, ve), w in zip(zip(lines[0::2], lines[1::2]), words):
                            e, v = en[len("English: "):].split(), ve[len("Veltan: "):].split()
                            diff = [(a, b) for a, b in zip(e, v) if a != b]
                            ok_lex &= len(e) == len(v) and len(diff) == 1 and diff[0][1].strip(".,") == w
                            ok_lex &= (en in text) == k.endswith("same")       # taught sentence or a new one
                    elif fam.startswith("key"):
                        val = dec(x[sc]).strip()
                        cue = dec(x[~sc]).strip("\n").split(" is ")[0]
                        ok_key &= val.isdigit() and len(val) == 6 and sc.sum() == 2 and f"{cue} is {val}." in text
                        ok_key &= text.count("The special magic number") == int(fam[3:])
                    else:
                        code = dict(re.findall(r" ([a-z]) = ([a-z])", text))
                        taught = [ln.split(":")[0] for ln in text.strip("\n").split("\n")]
                        letters = [dec([i]).strip() for i in x[sc]]
                        want = [code[a] for ln in dec(x).strip("\n").split("\n") for a in ln.split(":")[0]]
                        ok_ci &= letters == want and len(letters) == sum(len(ln.split(":")[0])
                                                                         for ln in dec(x).strip("\n").split("\n"))
                        ok_ci &= all(ln.split(":")[0] not in taught for ln in dec(x).strip("\n").split("\n"))
                        ok_ci &= all("".join(taught).count(a) >= 2 for ln in dec(x).strip("\n").split("\n")
                                     for a in ln.split(":")[0])
        report("lexicon: scored = the Veltan nonce word; same taught, new not", bool(ok_lex))
        report("cipher: scored = the taught substitution of new, twice-taught letters", bool(ok_ci))
        report("passkeys: scored = the six digits (two tokens) of a taught fact; 1 / 4 facts taught", bool(ok_key))
        report("token ids == the whole text's; blocks fit chunk 1024 / window 3", bool(ok_tok and ok_fit))
    else:
        print(f"[SKIP] builders: no {TOKENIZER}")

    # 2. layout
    C, Wn, T = BASE["chunk_size"], BASE["window_chunks"], 10 * 128
    seq = np.random.default_rng(0).integers(1, 100, T).astype(np.int32)
    teach, test, sc = np.full(150, 101, np.int32), np.full(20, 102, np.int32), np.arange(20) % 3 == 0
    wr = CL.written(seq, teach, C, Wn, 6)
    x, s = CL.target_chunk(seq, test, sc, 5, C, 6)
    end = CL.TEACH_END * C
    ok = (np.array_equal(wr["beyond"][end - 150:end], teach) and np.array_equal(wr["beyond"][:end - 150], seq[:end - 150])
          and np.array_equal(wr["beyond"][end:], seq[end:]) and np.array_equal(wr["within"][6 * C - 150:6 * C], teach)
          and np.array_equal(wr["within"][:6 * C - 150], seq[:6 * C - 150]) and np.array_equal(wr["none"], seq)
          and np.array_equal(x[5:25], test) and np.array_equal(x[:5], seq[6 * C:6 * C + 5])
          and np.array_equal(s[5:25], sc) and s.sum() == sc.sum())
    try:
        CL.written(seq, np.zeros(2 * C + 1, np.int32), C, Wn, 6)
        ok = False                                                    # a block wider than the window is refused
    except ValueError:
        pass
    report("layout: teaching blocks beyond / within, test block in the target chunk", bool(ok))

    # 3. run == an independent computation; inner lr 0: on == off
    seqs = [(f"s{i}", np.random.default_rng(10 + i).integers(1, 100, T).astype(np.int32)) for i in range(2)]
    targets, seed = [6, 8], 3
    tok_toy = CL.Tok("/nonexistent", BASE["vocab_size"])
    for name, cfg in (("ttt", ModelConfig(**BASE)), ("gdn", ModelConfig(**BASE, memory="gdn", gdn_heads=2, gdn_head_dim=8))):
        model = Model(cfg, jax.random.PRNGKey(1))
        ws = core.WriteSettings(0.5, inner_task="causal", fused_ce_block=0)
        commit = eqx.filter_jit(lambda m, ids, pos, st: core.commit(m, ids, pos, st, jax.random.PRNGKey(0), ws, F32))
        logits = eqx.filter_jit(lambda m, ids, pos, st: core.read_forward(m, ids, pos, st, None, True, F32).logits)
        out, tg = CL.run(model, seqs, ws, targets, seed, F32, tok_toy, sizes=SMALL, log=lambda *a, **k: None)
        worst, fin = 0.0, bool(np.isfinite(out["ce"]).all() and (out["count"] > 0).all())
        for si in range(len(seqs)):
            for fam in ("span", "lex8", "cipher"):
                rng = np.random.default_rng([seed, list(CL.FAMILIES).index(fam), si])
                teach, tests = CL.build(fam, rng, tok_toy, seqs[(si + 1) % 2][1], 0, dict(CL.SIZES, **SMALL))
                off = int(rng.integers(1, C - max(len(x) for x, _ in tests.values()) + 1))
                k = sorted(tests)[0]
                ti = list(CL.TASKS).index(k)
                for gi, t in enumerate(tg):
                    x, s = CL.target_chunk(seqs[si][1], *tests[k], off, C, t)
                    inp = jnp.asarray(np.where(s, cfg.mask_token_id, x))
                    tgt, sel = jnp.asarray(np.concatenate([x[1:], x[-1:]])), np.concatenate([s[1:], [False]])
                    for pi, p in enumerate(CL.PLACEMENTS):
                        ids = CL.written(seqs[si][1], teach, C, Wn, t)[p]
                        st = core.init_memory(model, F32)
                        for c in range(t):
                            st = commit(model, jnp.asarray(ids[c * C:(c + 1) * C]), jnp.arange(c * C, (c + 1) * C), st)
                        for mi, stm in enumerate((st, core.reset_memory(model, st))):
                            lg = logits(model, inp, jnp.arange(t * C, (t + 1) * C), stm)
                            ce = -jnp.take_along_axis(jax.nn.log_softmax(lg, -1), tgt[:, None], -1)[:, 0]
                            want = float(np.asarray(ce, np.float64)[sel].sum())
                            worst = max(worst, abs(out["ce"][si, ti, gi, pi, mi] - want) / max(abs(want), 1.0))
        report(f"{name}: run == every placement written from scratch (span, lex8, cipher)", fin and worst < 1e-5,
               f"max rel {worst:.1e}")
        if name == "ttt":
            out0, _ = CL.run(model, seqs, ws._replace(inner_lr=0.0), targets, seed, F32, tok_toy, sizes=SMALL,
                             log=lambda *a, **k: None)
            report("ttt, inner lr 0: on == off for every task, target and placement",
                   bool(np.array_equal(out0["ce"][..., 0], out0["ce"][..., 1])))

    # 4. the command line
    tmp = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmp, "cfg"))
    json.dump(dict(BASE, chunk_size=512, window_chunks=3), open(os.path.join(tmp, "cfg", "config.json"), "w"))
    out = os.path.join(tmp, "cl.npz")
    r = subprocess.run([sys.executable, "-m", "tools.j_lens.context_learning", "--random_init", "--model_config",
                        os.path.join(tmp, "cfg"), "--inner_lr", "0.5", "--synthetic", "100", "--n_per_source", "2",
                        "--seq_len", "4096", "--targets", "7", "--tasks", "span,cipher", "--out", out],
                       capture_output=True, text=True, cwd=ROOT, timeout=1500,
                       env=dict(os.environ, JAX_PLATFORMS="cpu", TTT_ATTENTION="xla"))
    ok = r.returncode == 0 and os.path.exists(out)
    if ok:
        z = np.load(out)
        s = json.loads(str(z["summary"]))
        ok = (np.isfinite(z["ce"]).all() and len(s["rows"]) == 2 and {x["task"] for x in s["rows"]} == {"span", "cipher"}
              and all(np.isfinite(x["recall"][0]) for x in s["rows"]))
    report("command line: span + cipher at target 7, finite", bool(ok), "" if ok else (r.stdout[-600:] + r.stderr[-1500:]))

    print("ALL PASS" if not failures else "FAILURES: " + "; ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
