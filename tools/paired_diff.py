"""Paired differences between arms from their per-sequence evaluation arrays (eval_<step>.npz).

    tools/paired_diff.py --step 1000 A=log/20260919_070357 B=log/20260919_101420 [C=...]

For every pair (later arm minus the first): the mix-weighted headline (loss with the write on),
the no-write twin and the write ablation, each with the naive paired SE (independent sequences)
and a document-cluster bootstrap SE (documents resampled within source; sequences of one document
are not independent; a source drawn from a single document has no spread between documents to
resample, so its sequences are resampled instead).  Means and SEs are token-weighted when the npz
carries mask counts, per source as in the mix.  Source weights are read from the run's `[ttt] config` line (the recipe's
mix); evaluation-only sources (weight 0) are reported per source but excluded from the mix.
"""
import argparse
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def weights_of(run_dir):
    txt = open(os.path.join(run_dir, "train_main.log"), errors="replace").read()
    m = re.search(r"\[ttt\] config .*?sources=(\S+)", txt)
    w = {}
    for item in m.group(1).split(","):
        name, rest = item.split("=", 1)
        w[name] = float(rest.rsplit(":", 1)[1])
    return w


def load(run_dir, step):
    d = np.load(os.path.join(run_dir, f"eval_{step}.npz"), allow_pickle=True)
    return {k: d[k] for k in d.files}


def wse(x, w):
    """SE of the w-weighted mean of independent x."""
    p = w / w.sum()
    return float(np.sqrt(np.sum(p ** 2 * (x - np.sum(p * x)) ** 2) * len(x) / max(len(x) - 1, 1)))


def cluster_boot(x, docs, n=2000, seed=12345, w=None):
    """SE of the (w-weighted) mean of x under resampling of documents (clusters)."""
    udocs = np.unique(docs)
    if len(udocs) < 2:
        return float("nan")
    w = np.ones(len(x)) if w is None else w
    idx = {u: np.where(docs == u)[0] for u in udocs}
    rng = np.random.default_rng(seed)
    means = np.empty(n)
    for i in range(n):
        pick = rng.choice(udocs, size=len(udocs), replace=True)
        sel = np.concatenate([idx[u] for u in pick])
        means[i] = np.sum(x[sel] * w[sel]) / np.sum(w[sel])
    return float(means.std(ddof=1))


def mix_stat(diff, sources, docs, weights, n_boot=2000, tokw=None):
    """Mix-weighted mean of per-sequence differences, naive SE and cluster-bootstrap SE.
    ``tokw`` (scored tokens per sequence, from the npz 'mask' array when present) makes the
    per-source mean token-weighted like the logged headline; None = sequence-weighted."""
    srcs = [s for s in np.unique(sources) if weights.get(str(s), 0) > 0]
    wsum = sum(weights[str(s)] for s in srcs)
    tw = np.ones_like(diff) if tokw is None else tokw.astype(np.float64)
    wmean = lambda x, w: float(np.sum(x * w) / np.sum(w))
    mean = sum(weights[str(s)] * wmean(diff[sources == s], tw[sources == s]) for s in srcs) / wsum
    naive_var = sum((weights[str(s)] / wsum) ** 2 * wse(diff[sources == s], tw[sources == s]) ** 2 for s in srcs)
    # cluster bootstrap: resample documents within every source, recompute the weighted mean
    rng = np.random.default_rng(12345)
    per = {}
    for s in srcs:
        m = sources == s
        units = docs[m] if len(np.unique(docs[m])) > 1 else np.arange(m.sum())    # one document: its sequences
        udocs = np.unique(units)
        per[s] = (diff[m], tw[m], udocs, {u: np.where(units == u)[0] for u in udocs})
    boots = np.empty(n_boot)
    for i in range(n_boot):
        tot = 0.0
        for s in srcs:
            x, w, udocs, idx = per[s]
            sel = np.concatenate([idx[u] for u in rng.choice(udocs, size=len(udocs), replace=True)])
            tot += weights[str(s)] * wmean(x[sel], w[sel])
        boots[i] = tot / wsum
    return mean, float(np.sqrt(naive_var)), float(boots.std(ddof=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+", help="NAME=log/dir")
    ap.add_argument("--step", type=int, default=1000)
    a = ap.parse_args()
    arms = [(x.split("=", 1)[0], x.split("=", 1)[1]) for x in a.arms]
    data = {n: load(d if os.path.isabs(d) else os.path.join(ROOT, d), a.step) for n, d in arms}
    weights = weights_of(arms[0][1] if os.path.isabs(arms[0][1]) else os.path.join(ROOT, arms[0][1]))
    ref_name = arms[0][0]
    ref = data[ref_name]
    for n in data:
        assert np.array_equal(data[n]["sources"], ref["sources"]) and np.array_equal(data[n]["doc_ids"], ref["doc_ids"]), \
            f"{n}: evaluation set differs from {ref_name}"
    # token-weighted (as the logged headline) only when every arm carries mask counts, so the
    # per-arm rows and the differences use one functional
    tokw = ref.get("mask") if all("mask" in d for d in data.values()) else None
    print(f"step {a.step}; reference {ref_name}; mix weights of {len([s for s in weights if weights[s] > 0])} training sources; "
          f"{len(ref['sources'])} sequences, {len(np.unique(ref['doc_ids']))} documents; "
          f"{'token-weighted (as the logged headline)' if tokw is not None else 'sequence-weighted (no mask counts in this npz)'}")
    print("\n| arm − ref | headline (write on) | no-write twin | write ablation |")
    print("|:--|--:|--:|--:|")
    for n, d in data.items():
        if n == ref_name:
            continue
        cells = []
        for key in ("loss_1.0", "loss_0.0", "gain"):
            diff = d[key] - ref[key]
            m, se_n, se_c = mix_stat(diff, ref["sources"], ref["doc_ids"], weights, tokw=tokw)
            cells.append(f"{m:+.4f} (naive ±{se_n:.4f}, cluster ±{se_c:.4f}, {abs(m)/se_c if se_c > 0 else float('nan'):.1f}σ)")
        print(f"| {n} − {ref_name} | " + " | ".join(cells) + " |")
    print("\n| arm | headline | twin | ablation (mix-weighted, own cluster SE) |")
    print("|:--|--:|--:|--:|")
    for n, d in data.items():
        cells = []
        for key in ("loss_1.0", "loss_0.0", "gain"):
            m, se_n, se_c = mix_stat(d[key], d["sources"], d["doc_ids"], weights, tokw=None if tokw is None else d["mask"])
            cells.append(f"{m:.4f} ±{se_c:.4f}")
        print(f"| {n} | " + " | ".join(cells) + " |")
    print(f"\nper-source paired headline differences (arm − ref), {'token' if tokw is not None else 'sequence'}-weighted, "
          f"naive ± / cluster ±:")
    for n, d in data.items():
        if n == ref_name:
            continue
        parts = []
        for s in np.unique(ref["sources"]):
            m = ref["sources"] == s
            diff = (d["loss_1.0"][m] - ref["loss_1.0"][m]).astype(np.float64)
            w = np.ones(m.sum()) if tokw is None else tokw[m].astype(np.float64)
            parts.append(f"{s} {np.sum(diff * w) / np.sum(w):+.4f} ±{wse(diff, w):.4f}/±{cluster_boot(diff, ref['doc_ids'][m], w=w):.4f} "
                         f"(n={m.sum()}, docs={len(np.unique(ref['doc_ids'][m]))})")
        print(f"  {n}: " + "; ".join(parts))


if __name__ == "__main__":
    sys.exit(main())
