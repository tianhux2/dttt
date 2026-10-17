# Configuration

Every training run is described by a run configuration in `runs/`: a JSON object of `TrainConfig` fields
(`src/ttt/config.py`). `python -m ttt.train --help` lists every field with its default. The model's shape is a
separate file, `ModelConfig`, named by `model_config` (`configs/1b_e2e`, [architecture.md](architecture.md)).

## 1. Run configuration files

- `"base"`: a file or a list of files, relative to this one. They are read first, in order, and this file's keys
  replace theirs. `stage_32k.json` and `stage_128k.json` are partial files that only make sense in such a list.
- Keys starting with `#` are comments; `TODO(x)` in a comment marks a value that `x` still has to decide
  (section 6). Unknown keys are errors, so a typo cannot silently fall back to a default.
- Flags on the command line replace the file's values: `--config runs/x.json --steps 100`. Write an empty value as
  `--flag=` (for example `--continue_from=`): a bare `""` is lost when the command goes through ssh.
- A checkpoint's `train_config.json` is itself a run configuration.

## 2. The chain of runs

Steps are global: a branch or a stage continues the step count of the run it continues.

| run | starts from | ends at step | data |
|:--|:--|--:|:--|
| `trunk_8k` | scratch | 162000 | 8K, the main mix |
| `branch_plain`, `branch_method` | `ckpt/trunk_8k/step_00161999` | 179000 | 8K, the main mix |
| `plain_32k`, `method_32k` | their branch's `step_00178999` | 181000 | 32K, mostly books |
| `plain_128k`, `method_128k` | their 32K stage's `step_00180999` | 181240 | 128K, mostly books |

- The arms differ only in `inner_norm_alpha`, `inner_norm_cap`, `inner_lr` and `self_prior` (`branch_method.json`).
  The data order and the per-step random keys are the same, so the arms are compared pair by pair.
- `tiny_smoke.json`: 20 steps of a tiny model on the same data, the first test of a new machine.

## 3. Starting from a checkpoint

| field | loads | step | data | use |
|:--|:--|:--|:--|:--|
| `resume: auto` | weights, optimiser state | continues | must be what the checkpoint was written with (checked by hash) | restarting a run |
| `continue_from: <step dir>` | weights, optimiser state | continues | may differ (length, mix) | a branch, the next stage |
| `init_from: <step dir>` | weights only | starts at 0 | any | evaluation (`eval_only`), fine-tuning with a new schedule |

- `continue_from` resumes the run's own newest checkpoint when there is one, so the same command restarts it. It
  needs a complete checkpoint (a `DONE` file), in another run's `ckpt_dir`, and cannot be combined with `init_from`
  or `resume: <path>`.
- On the same data, a continued run reads the stream on from where the checkpoint stopped: `branch_plain` is
  exactly the trunk continued. On other data (another length or mix) it starts that stream at its first batch.
  Every checkpoint's `state.json` records this position as `data_offset`.
- The step batch (`batch_size × grad_accum`) cannot change on the same data: such a continuation is refused,
  because the stream would start again at its first sample.
- A continued run first evaluates its starting point (`eval_step0`, the default). The last step of every run is
  checkpointed and evaluated, whatever `ckpt_interval` and `eval_interval` are.

## 4. Learning rate

Warmup-stable-decay over global steps (`train.lr_schedule`): `lr` rises linearly from 0 over `warmup_steps`, stays
flat, falls linearly to `end_lr` from `decay_start` to `decay_end`, and stays there. `decay_start: -1` (the default)
means no decay. It replaced a cosine schedule; for the earlier GPU runs, which had `lr` equal to `end_lr`, the two
are bit for bit the same.

The trunk is flat at 1e-3. The branches decay from step 162000 to 181240, the end of the 128K stage: the decay runs
through the 8K part (89 % of its steps) and both long stages.

## 5. The fields of the main study

`runs/trunk_8k.json`, and what the later runs change.

**Model and data**

| field | value | meaning |
|:--|:--|:--|
| `model_config` | `configs/1b_e2e` | the model's shape |
| `data_root` | `data/tokens` | relative stream directories and `unigram_path` are read under it |
| `sources` | 12 streams | `name=dir:weight,...`; the weights sum to 1 ([data.md](data.md)) |
| `doc_windows` | true | every sequence is one window inside one document; no packing |
| `seq_len` | 8192 | tokens per sequence, 8 chunks |

**Batch and length of the run**

| field | value | meaning |
|:--|:--|:--|
| `batch_size` | 128 | sequences per micro-batch, over all devices: one per device |
| `grad_accum` | 1 | micro-batches per step; the step batch is 128 × 8192 ≈ 1.05M tokens |
| `steps` | 162000 | the global step at which the run ends (170B tokens) |
| `seed` | 42 | initialisation, data order and every step's masks |

**Optimiser**

| field | value | meaning |
|:--|:--|:--|
| `lr`, `warmup_steps` | 1e-3, 5000 | the peak learning rate and its linear warm-up |
| `end_lr`, `decay_start`, `decay_end` | 1e-4, −1, −1 | the decay; set in the branches |
| `weight_decay`, `grad_clip` | 0.1, 1.0 | AdamW decay (not on embeddings, norms, biases, write controls); global gradient-norm clip |

**The write**

| field | value | meaning |
|:--|:--|:--|
| `inner_task` | causal | the write predicts the next token of the clean chunk |
| `inner_lr`, `inner_clip` | 1.0, 1.0 | the write's step size and its norm clip |
| `inner_lr_warmup_steps` | 5000 | the step size ramps from 0.1× to 1× over these outer steps |
| `second_order` | true | the meta-gradient flows through the write |
| `inner_norm_alpha`, `inner_norm_cap` | 0, 1 (trunk); 1, 128 (method) | the normalised write; 0 is the plain write |
| `self_prior` | 0 (trunk); 0.1 (method) | the share of sequences read after their own chunk's write |

**The outer loss**

| field | value | meaning |
|:--|:--|:--|
| `time_reweighting`, `cart_p`, `cart_window` | cart, 0.1, 512 | weight of a masked token by its visible neighbours: kernel `0.5·p·(1−p)^(|d|−1)` over ±512 positions |
| `outer_shift` | true | the slot of token `i` predicts token `i+1` |
| `outer_views` | 1 | noisy reads of every chunk per step (2: two masks sharing the clean stream and the write) |

**Memory and speed**

| field | value | meaning |
|:--|:--|:--|
| `remat_levels` | 1 | rematerialisation levels over the chunk scan |
| `remat_prefix`, `prefix_remat_levels` | false, 1 | checkpoint the prefix stack (32K and longer) |
| `segment_chunks` | 0 (8K); 8 (32K, 128K) | process the sequence in segments of this many chunks |
| `truncate_fast_weight` | false; true at 128K | stop the meta-gradient at the fast weight on segment boundaries |
| `fused_ce_block` | 8192 | vocabulary entries per block of the cross-entropy |
| `attn_chunk_batch` | 2 (8K); 1 (long) | chunks whose attention scores the XLA path holds at once |
| `fsdp_size` | 0 | FSDP over all devices; N > 0: a `dp × fsdp` mesh with groups of N |
| `replicate_weights` | `suffix,embed` (default) | bf16 copies gathered once per step |

**Checkpoints**

| field | value | meaning |
|:--|:--|:--|
| `ckpt_dir` | `ckpt/<run>` | one directory shared by every host |
| `ckpt_interval` | 1000 (8K); 500 (32K); 40 (128K) | steps between checkpoints; the last step is always saved |
| `keep_checkpoints` | 0 | 0 keeps all (15 GB each); N keeps the newest N |
| `resume` | auto | resume the newest complete checkpoint of `ckpt_dir` |
| `continue_from` | (branches and stages) | the checkpoint a branch or a stage continues |

**Evaluation and logs** ([evaluation.md](evaluation.md))

| field | value | meaning |
|:--|:--|:--|
| `log_dir` | `log/<run>` | where process 0 writes `eval_<step>.npz` |
| `eval_sources` | PG-19 test | evaluation-only sources, reported separately |
| `eval_batches` | 2 (8K); 0 (32K, 128K) | held-out step batches per training source per evaluation; 0 turns evaluation off |
| `eval_interval` | 2000 | steps between evaluations; step 0 (or the continued step) and the last step too |
| `eval_probes`, `unigram_path` | true, `unigram_llama3.i64` | per-token classes: recurrence, rarity, visible neighbours |
| `eval_planted_recall`, `eval_planted_batches`, `eval_planted_len`, `eval_planted_cue` | true, 1, 64, 16 | the planted-recall probe |
| `holdout` | 4194304 (default) | tokens at the end of every stream never trained on |
| `log_interval` | 10 (default) | steps between log lines |
| `wandb_project` | empty | the W&B project; empty for none |

**Options outside the main line.** They stay at their defaults: `distill*`, `mtp*`, `inner_kvb_*`, `inner_feat*`,
`inner_subchunks`, `inner_subchunk_scale`, `inner_fast_*`, `inner_surprise_gamma`, `inner_write_steps`,
`inner_update` (sgd), `inner_views` (1), `inner_mask_mode` (random), `window_drop*`, `plant_*`, `train_recall*`,
`write_off`, `aux_causal_weight`, `inner_from_outer`, `fast_weight_lr_scale` (1). They were screened out; the code
keeps them to reproduce earlier results. `no_write` is the no-memory baseline. `memory: gdn` in a model
configuration is the Gated DeltaNet baseline architecture.

## 6. Open values

The files hold the current best values; these decide them:

| value | now | decided by |
|:--|:--|:--|
| `batch_size` | 128, one sequence per device: **decided** | measured on GH200: the critical batch is 93 (plain) to 129 (method) sequences after 1.3B tokens and grows with training; a larger batch is not faster on 128 devices |
| `outer_views` | 1 | two reads cost +18 % per step and break even now; re-measure with `tools/grad_noise.py` at ~10B tokens and switch to 2 once the one-read critical batch exceeds ~180 |
| `sources` (mix) | the v2 weights | **no packing** (decided 2026-10-17): single-document windows only, with enough long documents downloaded that no 8K source is read more than ~3 times ([data.md](data.md), section 7) |
| `steps`, `decay_*` | 170B trunk + 17.8B + 8.4B + 4.0B tokens | the token budget |
| `end_lr` | 1e-4 | the pilot |
| the method's switch | at the branch point, at once | the pilot (an earlier switch, a ramp of α; a ramp needs code) |
| `fsdp_size` | 0 | first-day throughput; a 2-D mesh is not yet laid out by the physical topology |
| `truncate_fast_weight` at 128K | true | the pilot |
| `wandb_project`, `keep_checkpoints` | none, all | the owner |
