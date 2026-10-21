# Evaluation

Every comparison in the study is paired: the arms see the same sequences, masks and noise levels, and a result is
the difference between arms with a paired standard error.

## 1. During training (8K runs)

Every `eval_interval` steps, at the starting step and at the last step, process 0 logs and saves to
`log_dir/eval_<step>.npz`:

- **The held-out loss** of every training source: `eval_batches` step batches of 128 sequences from the last
  4,194,304 tokens of the stream. Every sequence has a fixed noise level and mask (`eval_seed`), so numbers are
  comparable across steps and arms. The headline is the mix-weighted loss with the write on. `no-ttt` is the same
  with the write off (`inner_lr` 0), and `gain = no-ttt − loss` is what the memory buys. It is reported per source,
  per chunk position, by document length, and with standard errors over sequences. PG-19 test
  (`eval_sources`) is scored on all of its sequences and reported on its own, never mixed into the headline.
- **Probes** (`eval_probes`, `probes.py`): the gain per token class.
  - P1, recurrence: the token's context occurs earlier beyond the window, within it, not at all, or before the
    sequence.
  - P2, rarity: frequent, mid or rare, by `unigram_llama3.i64`.
  - P3: whether the left and right neighbours are visible.
- **Planted recall** (`eval_planted_recall`): a 64-token span of chunk 1 is copied beyond the window. Its first 16
  tokens stay visible as the cue, and the rest is masked, in blocks or alternately. The probe measures how much
  of it the memory recovers, with the write on and off, against the ceiling of the same copy inside the window
  (`eval_<step>_planted.npz`).

In the 32K and 128K stages this evaluation is off (`eval_batches: 0`). A step batch is 128 sequences, and the
holdout holds only 127 sequences of 32K (31 of 128K) per source.

The log lines start with `[eval <step>]`: `loss ... no-ttt ... gain ...`, then per source, by chunk, the probes and
planted recall.

## 2. After training

The same suite on the last checkpoint of both arms. These tools run as one process on one host.

- Unset `TTT_COORDINATOR TTT_NUM_PROCESSES TTT_PROCESS_ID TTT_LOCAL_DEVICE_IDS NUM_PROCESSES COORDINATOR_ADDRESS`
  first. An inherited `TTT_COORDINATOR` made the training entry point wait for the other processes; on GPU it failed
  with `KeyError: TTT_PROCESS_ID`.
- On a host of a multi-host TPU slice, the process must be confined to the host's own chips, or libtpu waits for
  the whole slice. On a v4 pod host this was `TPU_HOST_BOUNDS=1,1,1 TPU_CHIPS_PER_HOST_BOUNDS=2,2,1`; check the
  equivalent for the slice first. Alternatively, copy the checkpoint to a single-host machine.
- `JAX_PLATFORMS` must be set; the memory tools were always run with `TTT_ATTENTION=xla`.

Example for the method's arm. The final checkpoint is `C=ckpt/method_128k/step_00181239`, and the results go to
`R=log/branch_method/posteval_final`, where `round_compare.py` looks for them:

```bash
T=data/tokens
PROBE="gutenberg=$T/cp_gutenberg_6d1ade79103fe750:1,pre1929=$T/cp_pre1929_fdbe1eb1616b3ee7:1,arxiv=$T/cp_arxiv_5752128469516ca7:1,fineweb_edu_long=$T/fineweb_edu_long_dc_1961668477d8a501:1,dclm_long=$T/dclm_long_dc_17baf11a547ae088:1,starcoder_repo=$T/starcoder_repo_34497f38b4da24f3:1,open_web_math_long=$T/open_web_math_long_63b0e44605eca383:1,pg19_test=$T/pg19_test_a4dfccb3b3ac65f9:1"
mkdir -p $R/eval32k
# the held-out loss at 32K; the arm's 8K branch configuration gives the mix and the write settings
python -m ttt.train --config runs/branch_method.json --continue_from= --resume= --init_from $C --eval_only true \
    --seq_len 32768 --segment_chunks 8 --doc_windows false --batch_size <devices> --eval_batches 2 \
    --inner_lr_warmup_steps 0 --eval_planted_recall false --log_dir $R/eval32k --ckpt_dir /tmp/ck32k --ckpt_interval 0 \
    --wandb_project= > $R/eval32k/train_main.log 2>&1
# recall of a planted span by distance, at 32K and 128K
TTT_ATTENTION=xla python -m tools.j_lens.memory_distance --ckpt $C --sources "$PROBE" --seq_len 32768 \
    --n_per_source 8 --distances 5,9,17,30 --tag method_32k --out $R/distance_32k.npz
TTT_ATTENTION=xla python -m tools.j_lens.memory_distance --ckpt $C --sources "$PROBE" --seq_len 131072 \
    --n_per_source 4 --distances 5,30,65,126 --tag method_128k --out $R/distance_128k.npz
# context learning in generation mode: passkeys and a new lexicon
TTT_ATTENTION=xla python -m tools.j_lens.context_learning --ckpt $C --sources "$PROBE" --seq_len 32768 \
    --n_per_source 8 --targets 8,20 --tasks key1,key4,lex8_new --read l2r --tag method_32k --out $R/cl_32k.npz
TTT_ATTENTION=xla python -m tools.j_lens.context_learning --ckpt $C --sources "$PROBE" --seq_len 131072 \
    --n_per_source 4 --targets 64,127 --tasks key1,key4 --read l2r --tag method_128k --out $R/cl_128k.npz
# the likelihood bound with the write on and off, and the gradient noise scale: both run at the checkpoint's own
# length (elbo_eval unless --seq_len) and, by default, mix, so they use the 8K branch's last checkpoint
C8=ckpt/branch_method/step_00178999
TTT_ATTENTION=xla python tools/elbo_eval.py --ckpt $C8 --sources "$PROBE" --n_per_source 6 --tag method --out $R/elbo.npz
python tools/grad_noise.py --ckpt $C8 --n_seq 16 --n_masks 4 --views 2 --out $R/noise_step178999.json
```

- `--inner_lr_warmup_steps 0` matters: with `init_from`, the evaluation runs at step 0, where the write's warm-up
  would cut the inner learning rate to a tenth.
- The memory tools read the write settings (`inner_norm_alpha`, `inner_norm_cap`, `inner_lr`) from the
  checkpoint. The 32K evaluation takes them from the run configuration, so it uses the arm's own branch
  configuration.

What each tool measures:

| tool | measures |
|:--|:--|
| 32K held-out loss (`ttt.train --eval_only`) | the training-time evaluation at 32K, the write on and off |
| `tools/j_lens/memory_distance.py` | recall of a planted span by distance in chunks, as a share of the copy ceiling, and the memory's general gain |
| `tools/j_lens/context_learning.py` | facts taught early and asked later (passkeys, a new lexicon), read left to right as in generation |
| `tools/elbo_eval.py` | the likelihood bound (NELBO) over noise levels, and the memory's part of it |
| `tools/grad_noise.py` | the critical batch `B_simple = tr Σ / |G|²` in sequences, split into mask and data noise, for one and two reads |
| `tools/j_lens/doc_switch.py` | a document switch: document A then document B, B read with A's memory carried, reset or off; passkeys taught in A and asked in B |

## 3. Paired comparison

```bash
python tools/round_compare.py --step 179000 M=log/branch_method P=log/branch_plain --pairs M-P
```

Each run is `NAME=DIR[@STEP]`. `DIR` holds `train_main.log` (it supplies the mix weights) and the 8K evaluation
`eval_<step>.npz`; the post-training files are in its newest `posteval_*` directory. The tool prints the
differences with paired standard errors for every section that has files on both sides:
- 8K and 32K losses, with document-cluster bootstrap errors;
- recall by distance and context learning, bootstrapped over sequences;
- the likelihood bound;
- the noise scales, as recorded.

Two arms trained with the same recipe and different seeds differed by about 0.001–0.002 nats in loss and 1–3
points in recall. A difference smaller than that is noise.
