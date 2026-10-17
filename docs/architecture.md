# Architecture

The model is a 1.32B-parameter block masked-diffusion language model. Its top layers carry a fast weight that is
trained at test time: every 1024-token chunk writes into it with one gradient step, and the following chunks read
it. Training is end to end: the outer loss differentiates through the writes (the meta-gradient), so the slow
weights learn both to predict and to use the memory.

## 1. The model (`configs/1b_e2e`)

| | |
|:--|:--|
| layers | 24: 18 prefix layers, then 6 suffix layers that carry the fast weight |
| width | hidden 2048, MLP 4352 (SwiGLU), 16 query heads and 4 key-value heads of 128, QK-norm, RoPE θ = 10⁶ |
| vocabulary | Llama-3 tokenizer, 128,256 tokens; tied input and output embeddings; mask token 128002, EOS 128001 |
| chunk | 1024 tokens = the diffusion block = the TTT chunk |
| attention | bidirectional inside a chunk, causal across chunks, over a window of the 3 previous chunks (clean keys and values) |
| fast weight | in each suffix layer, a SwiGLU MLP of the same shape (gate and up 2048→4352, down 4352→2048); its initial value `W_0` is an ordinary outer parameter |

Everything beyond the 3-chunk window reaches a chunk only through the fast weight. At 8K (8 chunks), half of the
chunks have context outside their window; at 128K almost all of it is.

## 2. One training sequence

The prefix layers have no fast weight, so they run once, chunk-parallel, over three input streams of the whole
sequence. The suffix then scans the chunks in order (`chunk_scan.py`), with the fast weight `W` and the key-value
window in the scan's carry. For chunk `c`, with the fast weight `W_c` written by chunks `0..c-1`:

| stream | input | what it does |
|:--|:--|:--|
| A, outer read | the chunk with a fraction `t ~ U[0,1)` of its tokens masked | predicts the masked tokens with `W_c`: the loss |
| B, clean | the clean chunk | its keys and values enter the window of the next chunks |
| C, inner write | the clean chunk, read causally (left to right) | its next-token loss gives `grad_W`; one clipped SGD step gives `W_{c+1}` |

**The write** (`inner_loop.py`): `W_{c+1} = W_c − η · clip(∇_W L_inner)`, one global norm clip over the fast
weights of all suffix layers. `η` is `inner_lr`, ramped from 0.1× to 1× over `inner_lr_warmup_steps`; the clip is
`inner_clip`.

**The outer loss** (`objective.py`): the cross-entropy of the masked tokens of stream A, weighted per token by CART.
A token's weight grows with the number of visible neighbours around it (kernel `0.5·p·(1−p)^(|d|−1)` over ±512
positions). With `outer_shift` the slot of token `i` predicts token `i+1`, as in next-token prediction.

**The meta-gradient**: with `second_order` the outer gradient flows back through every write into `W_0` and the
slow weights. A first-order version is about 30 % faster per step but measured 0.092 nats worse in loss, and its
memory gain fell from 0.041 to 0.006.

**Why three streams, unlike TTT for autoregressive models**: the read is a denoising task on a masked chunk, the
window must hold what inference has (clean tokens), and the write predicts the next token on clean text. The
three cannot share one forward.

## 3. The method under study

The main study compares plain TTT with two additions to it, switched on together in `runs/branch_method.json`:

- **Normalised write** (`inner_norm_alpha`, `inner_norm_cap`; `inner_loop.norm_scales` and `unit_scale`): per
  fast-weight layer, the RMS `r` of every hidden unit and every input dimension over the written slots gives a
  factor `f = min(cap, (max r / r)^α)`. Down-projection rows are scaled by the unit's factor, gate and up entries by
  the smaller of the two factors, after the clip. Rarely used units get larger updates, so a few dominant units no
  longer take the whole step. Settings: α 1, cap 128, `inner_lr` 0.82.
- **Self-prior** (`self_prior` 0.1): a tenth of the sequences read every chunk after its own write, so the read
  learns to retrieve what the clean causal write stored.

Both are switched on late: plain TTT first (the trunk), the method in the last part (the branch). Switched on from
scratch, the normalised write cost +0.04 nats at step 500, because its amplification also scales the backward and
causes gradient spikes early. Switched on at step 2200 of 2500 on GPU, it cost +0.007 at 8K and multiplied the
planted recall and the passkey shares by about ten.

## 4. A training step

`train.make_train_step`: the loss and its gradient for the step batch (`batch_size` sequences, or
`grad_accum` micro-batches of them, summed exactly), global-norm clipping, AdamW (β 0.9 / 0.95, weight decay 0.1,
not on embeddings, norms, biases or write controls), and the learning rate of `train.lr_schedule`: warm-up,
stable, then linear decay, over global steps. Per-step random keys are `fold_in(PRNGKey(seed), step)`, so a resumed
run draws exactly the masks the uninterrupted one would have.

The logged loss is the outer loss. `inner` is the write's loss, `drift` is `||W − W_0|| / ||W_0||`, and `gn[...]`
splits the gradient norm into the embedding, prefix, suffix, fast-weight template and the rest.

## 5. Parallelism and memory

- **Devices** (`sharding.py`): FSDP over one mesh axis (`fsdp_size 0`), or a two-dimensional `dp × fsdp` mesh
  (`fsdp_size N`: parameters sharded inside groups of N devices, replicated across the groups). The batch is split
  over all devices, whole sequences per device. There is no tensor or sequence parallelism, so a step needs at
  least one sequence per device.
- **Replicated compute copies** (`replicate_weights suffix,embed`): the bf16 copies of the suffix layers and the
  embedding are gathered once per step instead of at every use.
- **Rematerialisation**: `remat_levels` over the chunk scan; `remat_prefix` and `prefix_remat_levels` for 32K and
  longer; `TTT_SUFFIX_LAYER_REMAT` (environment) inside the chunk.
- **Long sequences** (`segment_chunks`): the sequence is processed as segments of that many chunks, with the
  prefix windows, the suffix window and the fast weight carried across the boundaries. The result equals the
  unsegmented one; the backward keeps one carry per segment. `truncate_fast_weight` also stops the meta-gradient at
  `W` on segment boundaries. From a random initialisation, the full 128K meta-gradient blew up (gradient norm
  1.6e11), so the 128K stage truncates.
- **Attention kernels** (`attention.py`, `TTT_ATTENTION`): Pallas flash and splash on TPU, cuDNN on GPU, an XLA
  reference elsewhere. The write's forward is differentiated twice, so on TPU it uses the XLA kernel, which has a
  second-order rule.
- **Cross-entropy** over the 128K vocabulary in blocks of `fused_ce_block` vocabulary entries, never materialising
  the full logits (`fused_ce.py`).

## 6. Data and the sampler

`data.Sampler` is a pure function of `(seed, step)`. A sample's source is drawn by weight, and the sample is the
next entry of that source's per-epoch permutation of its training windows. With `doc_windows`, a window lies inside
one document (`windows_<T>_<holdout>.npy`). Each host reads only its own share of the step batch. The last 4,194,304
tokens of every stream are never trained on: they are the held-out evaluation set. The data hash (sources, weights,
token counts, window indexes, seed, length, step batch, holdout; no paths) protects a resume from a changed data
configuration. Details: [data.md](data.md).

## 7. Code map

| file | contents |
|:--|:--|
| `src/ttt/config.py` | `ModelConfig`, `TrainConfig`, run configurations (`load_run_config`) |
| `src/ttt/train.py` | the entry point: arguments, optimiser and schedule, the step, evaluation, checkpoints, the loop |
| `src/ttt/model.py` | the transformer: stacked prefix and suffix blocks, the fast weight, attention projections |
| `src/ttt/chunk_scan.py` | the training objective: chunk-parallel prefix, then the scan over chunks with the three streams |
| `src/ttt/inner_loop.py` | the write: inner loss, clipped step, normalised step, write schemes |
| `src/ttt/objective.py` | masking, CART weighting, the inner mask |
| `src/ttt/attention.py` | attention kernels (XLA, Pallas flash / splash, cuDNN) |
| `src/ttt/fused_ce.py` | blockwise cross-entropy |
| `src/ttt/remat_scan.py` | nested rematerialised scans |
| `src/ttt/sharding.py` | meshes and parameter shardings |
| `src/ttt/data.py` | token streams, window indexes, the sampler, prefetching, the data command line |
| `src/ttt/checkpoint.py` | saving and loading checkpoints |
| `src/ttt/probes.py` | evaluation probes: recurrence and rarity classes, planted recall |
| `src/ttt/generate.py` | block-wise generation |
| `src/ttt/gdn.py` | the Gated DeltaNet memory (`memory: gdn`), a baseline architecture |
| `src/ttt/inner_masks.py`, `sharded_io.py` | host-side inner masks (off the main line), sharded file I/O |
| `tools/` | data preparation and decontamination, post-training evaluation, paired comparison |
| `tests/` | CPU tests, one file per feature |

The code also contains options that were screened out: distillation, multi-token prediction, key-value binding,
feature prediction, sub-chunk and two-timescale writes, window drop, planted training. Their defaults are off and
the main study does not use them ([config.md](config.md)).
