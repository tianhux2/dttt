#!/usr/bin/env python3
# coding=utf-8
"""
Paired comparison of runs on the post-training evaluation suite (docs/evaluation.md): for every
pair A-B the differences A - B with paired standard errors, section by section -

    8K        eval_<step>.npz (the training-time evaluation): the mix-weighted headline (write on), the no-write twin
              and the write gain, document-cluster bootstrap SE (tools/paired_diff.py); pg19_test on its own
    32K       eval32k/eval_0.npz: the same at 32K
    distance  distance_32k.npz, distance_128k.npz: recall of a planted span by distance, % of the copy ceiling (own
              read, and isolated), and the memory's general gain (nats); bootstrap over sequences
    context   cl_32k.npz, cl_128k.npz: share = recall / ceiling (%) per task and distance, and the answers exactly
              right under the memory (l2r read); bootstrap over sequences
    bound     elbo.npz: the likelihood bound with the write on and off, and the memory's part (off - on)
    noise     noise_<tag>.json: B_simple with one and two reads, as recorded (not paired)

A run is NAME=DIR[@STEP] (STEP overrides --step for its 8K file): DIR holds eval_<step>.npz and the post-training
files, either itself or in its newest posteval_*/.  Sequences are aligned per source
in order (a run with another layout may score fewer of a source; the shared ones are compared).  A missing file
skips its section.  Pure numpy.

    python tools/round_compare.py --step 179000 M=log/branch_method P=log/branch_plain --pairs M-P
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paired_diff import cluster_boot, mix_stat, weights_of  # noqa: E402

N_BOOT = 2000
MD_KEYS = ("own", "write", "ctx", "isolated")      # distance npz: gain_copy / gain_none [..., key]


def post_dir(d):
    p = sorted(glob.glob(os.path.join(d, "posteval_*")), key=os.path.getmtime)
    return p[-1] if p else d


def npz(path):
    if not os.path.exists(path):
        return None
    with np.load(path, allow_pickle=True) as f:
        return {k: f[k] for k in f.files}


def summary(d):
    return json.loads(str(d["summary"]))


def pm(a, b, se, f=".1f"):
    return f"{a:{f}} / {b:{f}} ({a - b:+{f}} ± {se:{f}})"


def align(ref, other):
    """Indices into ``other`` of ``ref``'s sequences (per source, in order); -1 where ``other`` has fewer."""
    idx = np.full(len(ref["sources"]), -1)
    for s in np.unique(ref["sources"]):
        r, o = np.where(ref["sources"] == s)[0], np.where(other["sources"] == s)[0]
        n = min(len(r), len(o))
        idx[r[:n]] = o[:n]
    return idx


def evals(A, B, W, label):
    ib = align(A, B)
    keep = ib >= 0
    per_seq = lambda d, sel: {k: v[sel] for k, v in d.items() if getattr(v, "ndim", 0) and len(v) == len(d["sources"])}
    a, b = per_seq(A, keep), per_seq(B, ib[keep])
    n_other = int(np.sum((a["doc_ids"] != b["doc_ids"]) | (a["doc_lens"] != b["doc_lens"])))
    src, docs = a["sources"], a["doc_ids"]
    tokw = a["mask"].astype(np.float64) if "mask" in a else np.ones(len(src))
    cells = []
    for key in ("loss_1.0", "loss_0.0", "gain"):
        m, _, se = mix_stat((a[key] - b[key]).astype(np.float64), src, docs, W, n_boot=N_BOOT, tokw=tokw)
        cells.append(f"{m:+.4f} ± {se:.4f}")
    pg = src == "pg19_test"
    for key in ("loss_1.0", "gain"):
        if not pg.any():
            cells.append("n/a")
            continue
        x, w = (a[key] - b[key]).astype(np.float64)[pg], tokw[pg]
        cells.append(f"{np.sum(x * w) / np.sum(w):+.4f} ± {cluster_boot(x, docs[pg], n=N_BOOT, w=w):.4f}")
    note = f" ({n_other} with another document id or length)" if n_other else ""
    return f"| {label} | {keep.sum()} of {len(keep)}{note} | " + " | ".join(cells) + " |"


def distance(A, B, label, rng):
    if not np.array_equal(A["count"], B["count"]):
        return [f"| {label} | scored slots differ: not paired |"]
    boots = rng.integers(0, A["count"].shape[0], (N_BOOT, A["count"].shape[0]))

    def pct(d, di, k, idx=slice(None)):
        g = (d["gain_copy"][:, di, MD_KEYS.index(k)] - d["gain_none"][:, di, MD_KEYS.index(k)])[idx]
        return 100 * g.sum(-1) / d["ceiling"][idx, di].sum(-1)

    def gen(d, di, idx=slice(None)):
        return (d["gen_off"][idx, di] - d["gen_ce"][idx, di]).sum(-1) / d["count_gen"][idx, di].sum(-1)
    rows = []
    for di, dist in enumerate(summary(A)["distances"]):
        cells = [pm(pct(A, di, k), pct(B, di, k), (pct(A, di, k, boots) - pct(B, di, k, boots)).std(ddof=1))
                 for k in ("own", "isolated")]
        cells.append(pm(gen(A, di), gen(B, di), (gen(A, di, boots) - gen(B, di, boots)).std(ddof=1), ".4f"))
        rows.append(f"| {label} | {dist} | " + " | ".join(cells) + " |")
    return rows


def exact(d, ti, gi):
    """Per sequence, the share of its answers exactly right under the memory (beyond, on) - context_learning's."""
    out = []
    for si in range(d["l2r_ok"].shape[0]):
        ok, item = d["l2r_ok"][si, ti, gi, 0, 0], d["l2r_item"][si, ti]
        v = item >= 0
        out.append(np.mean([np.all(ok[v & (item == i)] == 1) for i in np.unique(item[v])]) if v.any() else np.nan)
    return np.asarray(out)


def context(A, B, label, rng):
    ma, mb = json.loads(str(A["meta"])), json.loads(str(B["meta"]))
    if ma["tasks"] != mb["tasks"] or ma["targets"] != mb["targets"] or not np.array_equal(A["count"], B["count"]):
        return [f"| {label} | tasks, targets or scored tokens differ: not paired |"]
    S = A["count"].shape[0]
    boots = rng.integers(0, S, (N_BOOT, S))
    dist = iter(r["distance"] for r in summary(A)["rows"])
    rows = []
    for ti, task in enumerate(ma["tasks"]):
        w = A["count"][:, ti]
        for gi in range(len(ma["targets"])):
            def parts(d):
                per = d["ce"][:, ti, gi] / np.maximum(w, 1)[:, None, None]     # [S, placement, (on, off)]
                b, wi, nn = per[:, 0], per[:, 1], per[:, 2]
                return (b[:, 1] - b[:, 0]) - (nn[:, 1] - nn[:, 0]), b[:, 1] - wi[:, 1]

            def share(rc, idx=slice(None)):
                return 100 * (w[idx] * rc[0][idx]).sum(-1) / (w[idx] * rc[1][idx]).sum(-1)
            pa, pb = parts(A), parts(B)
            cells = [pm(share(pa), share(pb), (share(pa, boots) - share(pb, boots)).std(ddof=1))]
            if "l2r_ok" in A and "l2r_ok" in B:
                ea, eb = exact(A, ti, gi), exact(B, ti, gi)
                v = ~(np.isnan(ea) | np.isnan(eb))
                x = (ea - eb)[v]
                cells.append(pm(ea[v].mean(), eb[v].mean(), x.std(ddof=1) / np.sqrt(len(x)), ".3f"))
            rows.append(f"| {label} | {task} | {next(dist)} | " + " | ".join(cells) + " |")
    return rows


def bound(A, B, label, rng):
    if not np.array_equal(A["count"], B["count"]):
        return f"| {label} | scored tokens differ: not paired |"
    boots = rng.integers(0, A["ce"].shape[-1], (N_BOOT, A["ce"].shape[-1]))

    def nelbo(d, idx=slice(None)):                     # [scale] or [scale, boot]: the midpoint rule over t
        return (d["ce"][..., idx].sum(-1) / d["count"][..., idx].sum(-1)).mean(0)
    na, nb, ba, bb = nelbo(A), nelbo(B), nelbo(A, boots), nelbo(B, boots)
    cells = [pm(na[s], nb[s], (ba[s] - bb[s]).std(ddof=1), ".4f") for s in (0, 1)]
    cells.append(pm(na[1] - na[0], nb[1] - nb[0], ((ba[1] - ba[0]) - (bb[1] - bb[0])).std(ddof=1), ".4f"))
    return f"| {label} | " + " | ".join(cells) + " |"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", help="NAME=DIR[@STEP]")
    ap.add_argument("--pairs", required=True, help="comma-separated A-B: the differences A - B")
    ap.add_argument("--step", type=int, required=True, help="the training-time evaluation (eval_<step>.npz)")
    a = ap.parse_args(argv)
    runs = {}
    for x in a.runs:
        name, rest = x.split("=", 1)
        d, _, st = rest.partition("@")
        runs[name] = (d, post_dir(d), int(st) if st else a.step)
    pairs = [tuple(p.split("-")) for p in a.pairs.split(",")]
    for p in pairs:
        if len(p) != 2 or not set(p) <= set(runs):
            sys.exit(f"--pairs: {'-'.join(p)} - names are {sorted(runs)}")
    rng = np.random.default_rng(0)
    print("runs: " + "; ".join(f"{n} = {d} (8K at {st}; post-training files in {p})" for n, (d, p, st) in runs.items()))
    W = weights_of(runs[pairs[0][0]][0])
    print(f"mix weights (from {pairs[0][0]}'s config): " + ", ".join(f"{k} {v:g}" for k, v in W.items()))
    lab = lambda x, y: f"{x} − {y}"

    for title, get in (("8K held-out", lambda r: npz(os.path.join(r[0], f"eval_{r[2]}.npz"))),
                       ("32K held-out", lambda r: npz(os.path.join(r[1], "eval32k", "eval_0.npz")))):
        rows = [evals(A, B, W, lab(x, y)) for x, y in pairs
                for A, B in [(get(runs[x]), get(runs[y]))] if A is not None and B is not None]
        if rows:
            print(f"\n### {title} (nats; headline = write on; gain = twin - headline)\n")
            print("| pair | sequences | headline (mix) | twin (mix) | write gain (mix) | pg19 headline | pg19 gain |")
            print("|:--|--:|--:|--:|--:|--:|--:|")
            print("\n".join(rows))
    for tag in ("32k", "128k"):
        rows = [r for x, y in pairs for A, B in [(npz(os.path.join(runs[x][1], f"distance_{tag}.npz")),
                                                  npz(os.path.join(runs[y][1], f"distance_{tag}.npz")))]
                if A is not None and B is not None for r in distance(A, B, lab(x, y), rng)]
        if rows:
            print(f"\n### Recall over distance, {tag} (% of the copy ceiling: A / B (A − B ± SE))\n")
            print("| pair | distance (chunks) | own read | isolated | general gain (nats) |")
            print("|:--|--:|--:|--:|--:|")
            print("\n".join(rows))
    for tag in ("32k", "128k"):
        rows = [r for x, y in pairs for A, B in [(npz(os.path.join(runs[x][1], f"cl_{tag}.npz")),
                                                  npz(os.path.join(runs[y][1], f"cl_{tag}.npz")))]
                if A is not None and B is not None for r in context(A, B, lab(x, y), rng)]
        if rows:
            print(f"\n### Learning from context beyond the window, {tag} (A / B (A − B ± SE))\n")
            print("| pair | task | distance (chunks) | share = recall / ceiling (%) | exact under the memory |")
            print("|:--|:--|--:|--:|--:|")
            print("\n".join(rows))
    rows = [bound(A, B, lab(x, y), rng) for x, y in pairs
            for A, B in [(npz(os.path.join(runs[x][1], "elbo.npz")), npz(os.path.join(runs[y][1], "elbo.npz")))]
            if A is not None and B is not None]
    if rows:
        print("\n### Likelihood bound (nats per token; A / B (A − B ± SE))\n")
        print("| pair | write on | write off | memory's part (off − on) |")
        print("|:--|--:|--:|--:|")
        print("\n".join(rows))
    noise = [(n, f) for n, (d, p, st) in runs.items() for f in sorted(glob.glob(os.path.join(p, "noise_*.json")))]
    if noise:
        print("\n### Gradient noise (B_simple in sequences, all parameters; not paired)\n")
        print("| run | file | step | one read | two reads |")
        print("|:--|:--|--:|--:|--:|")
        for n, f in noise:
            r = json.load(open(f))
            bs = {v: r["by_views"].get(v, {}).get("all", {}).get("b_simple", float("nan")) for v in ("1", "2")}
            print(f"| {n} | {os.path.basename(f)} | {r.get('step')} | {bs['1']:.1f} | {bs['2']:.1f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
