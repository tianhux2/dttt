# coding=utf-8
"""
Reproduction check between two evaluations of the same weights: per-sequence eval files of one
checkpoint on two machines (e.g. a TPU run's log/<run>/eval_4000.npz and its re-evaluation on GPU).  First that
both scored the same sequences (sources, document starts and lengths, tokens beyond the window: equal arrays),
then per array the means and the per-sequence differences (mean, quantiles of |B - A|, correlation), per source
for the losses.

    python tools/compare_eval.py log/<run>/eval_4000.npz log/<rerun>/eval_4000.npz
    python tools/compare_eval.py log/<run>/eval_4000_planted.npz log/<rerun>/eval_4000_planted.npz
"""
import sys

import numpy as np

IDS = ("sources", "doc_ids", "doc_lens", "beyond")


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    a, b = (np.load(p, allow_pickle=True) for p in sys.argv[1:])
    same = True
    for k in IDS:
        if k in a.files or k in b.files:
            eq = k in a.files and k in b.files and a[k].shape == b[k].shape and np.array_equal(a[k], b[k])
            same &= eq
            print(f"  {k:12s} {'identical' if eq else 'DIFFERENT'}")
    if not same:
        print("the two files did not score the same sequences: the differences below are not paired")
    print(f"  {'array':16s} {'mean A':>9s} {'mean B':>9s} {'mean B-A':>10s} {'|d| p50':>9s} {'p99':>9s} {'max':>9s} {'corr':>7s}")
    for k in sorted(set(a.files) & set(b.files) - set(IDS)):
        x, y = np.asarray(a[k], np.float64), np.asarray(b[k], np.float64)
        if x.shape != y.shape or x.ndim != 1:
            continue
        d = np.abs(y - x)
        r = np.corrcoef(x, y)[0, 1] if x.std() > 0 and y.std() > 0 else float("nan")
        print(f"  {k:16s} {x.mean():9.5f} {y.mean():9.5f} {(y - x).mean():+10.5f} {np.median(d):9.5f} "
              f"{np.quantile(d, 0.99):9.5f} {d.max():9.5f} {r:7.4f}")
    if "sources" in a.files and same:
        for k in [k for k in ("loss_1.0", "loss_0.0") if k in a.files and k in b.files]:
            src = a["sources"]
            row = " | ".join(f"{s} {(b[k][src == s] - a[k][src == s]).mean():+.5f}" for s in dict.fromkeys(src))
            print(f"  {k} B-A by source: {row}")


if __name__ == "__main__":
    main()
