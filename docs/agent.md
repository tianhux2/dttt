# Agent: the switch and batch runs

**Goal.** Train three runs to step 10000, measure the gradient noise of their final checkpoints, compare their
evaluations, and write the results to `log/agent_report.md`.

| | config | devices | from |
|:--|:--|--:|:--|
| P, plain | `runs/trunk_8k_10k.json` | 64 | scratch |
| M, method from step 5000 | `runs/method_5k_10k.json` | 64 | P's `ckpt/trunk_8k/step_00004999` |
| B, step batch 256 | `runs/trunk_8k_b256.json` | 128 | scratch |

**1. Check the data** the three runs need, against [data.md](data.md), sections 1 and 6: under `data/tokens`, the
12 streams of `runs/trunk_8k.json` with `windows_8192_4194304.npy`, `pg19_test` and `unigram_llama3.i64`. Rebuild a
missing or wrong stream as in section 3 (window indexes: section 5), under the directory name `runs/` uses, and check
that it matches section 1's documents and tokens exactly. Do not prepare anything else, and do not edit code or run
configurations.

**2. Run.** Start each command on all hosts as in [launch.md](launch.md), section 4 (or 5 in Docker), appending to
the logs, and link process 0's log to `log/<run>/train_main.log`. After a crash, run the same command again.

```bash
python -u -m ttt.train --config runs/trunk_8k_10k.json --steps 5000   # P to step_00004999 (use 64 gpu)
python -u -m ttt.train --config runs/trunk_8k_10k.json                # P on to 10000 (use 64 gpu)
python -u -m ttt.train --config runs/method_5k_10k.json               # M, once step_00004999/DONE exists (use 64 gpu)
python -u -m ttt.train --config runs/trunk_8k_b256.json               # B, any time (use 128 gpu)
```

P checkpoints every 2000 steps, so it must stop at 5000 first: that writes `step_00004999`, where M starts.

**3. Noise** of each final checkpoint, one GPU each:

```bash
for d in trunk_8k method_5k_10k trunk_8k_b256; do mkdir -p log/$d/posteval_final
  JAX_PLATFORMS=cuda python tools/grad_noise.py --ckpt ckpt/$d/step_00009999 --n_seq 32 --n_masks 4 --views 2 \
      --out log/$d/posteval_final/noise_step9999.json
done
```

**4. Compare** (paired: the same held-out sequences and masks):

```bash
python tools/round_compare.py --step 10000 M=log/method_5k_10k P=log/trunk_8k --pairs M-P
python tools/round_compare.py --step 10000 B=log/trunk_8k_b256 P=log/trunk_8k --pairs B-P        # same steps
python tools/round_compare.py --step 10000 B=log/trunk_8k_b256@5000 P=log/trunk_8k --pairs B-P   # same tokens
```

Also the evaluations at the earlier steps, and planted recall from the `[eval <step>] planted recall` lines. M's
training loss includes the self-prior reads, so compare M and P through the evaluations, not the training loss.

**5. Report:** the setup (commit, devices, `s/step`, restarts), the tables of step 4, B_simple with one and two
reads per checkpoint, and the training loss and `gnorm` of each run. Differences below about 0.002 nats are noise.
Report facts.
