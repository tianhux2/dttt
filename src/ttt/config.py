# coding=utf-8
"""
Configuration: two frozen dataclasses with JSON round-trip.

``ModelConfig`` uses Hugging Face field names where an equivalent exists, so a
config written for this package reads naturally next to any HF ``config.json``.
Unknown keys are an ERROR on load, not a warning: a typo'd field silently taking
its default is exactly how the previous stack trained with the wrong mask token.
"""
from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field, fields
from typing import Any, Dict, Optional, Tuple


def _load_json(path: str) -> Dict[str, Any]:
    if os.path.isdir(path):
        path = os.path.join(path, "config.json")
    with open(path) as f:
        return json.load(f)


# Options removed after their screens, with the value that meant "off": a saved configuration that has them off
# still loads; one that used them cannot be reproduced and is refused.
_REMOVED = {"inner_assoc_scale": 1.0, "inner_prior_units": "", "inner_prior_k": 0, "inner_prior_dims": 0,
            "inner_norm_whiten": False, "self_prior_delay": [0, 0]}


def _check_keys(cls, d: Dict[str, Any]) -> Dict[str, Any]:
    """``d`` without the removed options that are off; unknown keys and removed options in use are errors."""
    known = {f.name for f in fields(cls)}
    used = sorted(k for k in set(d) - known if k in _REMOVED and d[k] != _REMOVED[k])
    if used:
        raise ValueError(f"{cls.__name__}: removed options in use: {', '.join(f'{k}={d[k]!r}' for k in used)}")
    d = {k: v for k, v in d.items() if k not in _REMOVED or k in known}
    unknown = sorted(set(d) - known)
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown keys {unknown}. Known: {sorted(known)}")
    return d


def _from_dict(cls, d: Dict[str, Any]):
    return cls(**_check_keys(cls, d))


def load_run_config(path: str, _seen: Tuple[str, ...] = ()) -> Dict[str, Any]:
    """A run configuration (``python -m ttt.train --config runs/x.json``): a JSON object of ``TrainConfig``
    fields.  ``"base"``, a file or a list of files relative to this one, is read first (in order) and overridden
    key by key, so a branch or a later stage names only what differs from its base; keys starting with ``#``
    are comments.  Unknown keys are errors, as in a saved configuration (a saved ``train_config.json`` is itself
    a run config)."""
    path = os.path.abspath(path)
    if path in _seen:
        raise ValueError(f"run config {path}: 'base' cycle")
    with open(path) as f:
        d = json.load(f)
    base = d.pop("base", [])
    out: Dict[str, Any] = {}
    for b in [base] if isinstance(base, str) else base:
        out.update(load_run_config(os.path.join(os.path.dirname(path), b), _seen + (path,)))
    out.update({k: v for k, v in d.items() if not k.startswith("#")})
    return _check_keys(TrainConfig, out)


@dataclass(frozen=True)
class ModelConfig:
    # --- architecture (HF names) ---
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    rope_scaling: Optional[Tuple[Tuple[str, Any], ...]] = None   # HF rope_scaling, llama3 only: rope_type, factor,
                                            # low_freq_factor, high_freq_factor, original_max_position_embeddings;
                                            # held as sorted (key, value) pairs (hashable), saved as a dict. None: plain RoPE
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True
    attention_bias: bool = False
    qk_norm: bool = True
    # --- test-time training ---
    suffix_len: int = 6                     # top `suffix_len` layers run chunk by chunk (the suffix) ...
    suffix_plain_top: int = 0               # ... and all but the top `suffix_plain_top` of them carry the fast weight:
                                            # 0 = the fast weight in every suffix layer; e.g. suffix_len 12, plain 6 on
                                            # 24 layers = fast weights in layers 12-17, plain layers 18-23 above them
                                            # (they stay in the suffix: the inner loss is read at the top, end to end)
    fast_layers: Optional[Tuple[int, ...]] = None   # any set of fast-weight layers instead (absolute indices, e.g.
                                            # [16, 18, 20, 21, 22, 23]): the lowest must be the suffix's first layer
                                            # (suffix_len = num_hidden_layers - min); the others in the suffix are plain
    ttt_intermediate_size: Optional[int] = None   # None -> intermediate_size
    inner_lr_gate: bool = False             # learned per-layer, per-chunk multiplier on the inner lr (from h_clean, surprise, ratio)
    inner_token_gate: bool = False          # learned per-token weights on the inner loss (from h_clean)
    inner_decay: float = 0.0                # > 0: learned decay of W toward W_0 per chunk, this is its init (off the main line)
    inner_optimizer: bool = False           # learned inner optimiser: factored per-parameter step sizes, per-layer step, loss temperature
    inner_feat_head: bool = False                # H -> H head on the inner stream: feature-prediction inner task
    mtp_head: bool = False                       # H -> H head before the shared LM head: second target x_{i+2} (MTP)
    inner_kvb_head: bool = False                 # per-suffix-layer H -> H value map of the key-value-binding write
    memory: str = "ttt"                          # cross-chunk memory of the suffix layers: ttt (fast weight W) | gdn (Gated DeltaNet state)
    gdn_heads: int = 16                          # gdn: heads per layer
    gdn_head_dim: int = 128                      # gdn: key = value dim per head (state nh x d x d per layer)
    # --- chunking / attention window ---
    chunk_size: int = 1024                  # diffusion block == TTT chunk == KV write granularity
    window_chunks: int = 7                  # each chunk attends to the previous `window_chunks` chunks
    window_reach: int = -1                  # >= 0: ... of which only the newest `window_reach` are visible (0: the own
                                            # chunk only, cross-chunk information through the memory alone); the KV
                                            # window keeps `window_chunks` chunks.  -1: all of them
    read_block: int = 0                     # > 0: the read in sub-blocks of this many slots, left to right within the
                                            # chunk (BD3-LM's blocks inside the TTT chunk): slot group g reads the
                                            # clean K/V of the earlier groups - the clean stream attends
                                            # block-causally - and its own group's noisy tokens.  0: the whole chunk
    # --- special tokens ---
    mask_token_id: int = 0
    pad_token_id: int = 0
    eos_token_id: int = 0
    bos_token_id: Optional[int] = None

    def __post_init__(self):
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be a multiple of num_key_value_heads")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for RoPE")
        if self.rope_scaling is not None:
            rs = dict(self.rope_scaling)                         # a JSON dict -> hashable sorted pairs
            need = {"rope_type", "factor", "low_freq_factor", "high_freq_factor", "original_max_position_embeddings"}
            if rs.get("rope_type") != "llama3" or set(rs) != need:
                raise ValueError(f"rope_scaling {rs}: llama3 only, with exactly the keys {sorted(need)}")
            if not (rs["factor"] >= 1 and 0 < rs["low_freq_factor"] < rs["high_freq_factor"]
                    and rs["original_max_position_embeddings"] > 0):
                raise ValueError(f"rope_scaling {rs}: factor >= 1, 0 < low_freq_factor < high_freq_factor, "
                                 "original_max_position_embeddings > 0")
            object.__setattr__(self, "rope_scaling", tuple(sorted(rs.items())))
        if not 0 < self.suffix_len <= self.num_hidden_layers:
            raise ValueError("suffix_len must be in (0, num_hidden_layers]")
        if not 0 <= self.suffix_plain_top < self.suffix_len:
            raise ValueError("suffix_plain_top must be in [0, suffix_len)")
        if self.suffix_plain_top and (self.memory == "gdn" or self.inner_kvb_head):
            raise ValueError("suffix_plain_top > 0: TTT only, without the key-value-binding head")
        if self.fast_layers is not None:
            fl = tuple(sorted(int(l) for l in self.fast_layers))
            object.__setattr__(self, "fast_layers", fl)          # a JSON list -> a hashable tuple
            if len(set(fl)) != len(fl) or not fl or fl[-1] >= self.num_hidden_layers:
                raise ValueError(f"fast_layers {fl}: distinct layer indices below num_hidden_layers")
            if fl[0] != self.num_hidden_layers - self.suffix_len:
                raise ValueError(f"fast_layers {fl}: the lowest must be the suffix's first layer "
                                 f"(suffix_len = {self.num_hidden_layers - fl[0]})")
            if self.suffix_plain_top or self.memory == "gdn" or self.inner_kvb_head:
                raise ValueError("fast_layers: TTT only, without suffix_plain_top or the key-value-binding head")
        if self.mask_token_id >= self.vocab_size:
            raise ValueError(f"mask_token_id {self.mask_token_id} >= vocab_size {self.vocab_size}")
        if not 0.0 <= self.inner_decay < 0.1:
            raise ValueError("inner_decay must be in [0, 0.1) (parameterised as 0.1 * sigmoid)")
        if self.hidden_act not in ("silu", "gelu"):
            raise ValueError(f"unsupported hidden_act {self.hidden_act!r}")
        if self.memory not in ("ttt", "gdn"):
            raise ValueError(f"memory {self.memory!r} (ttt | gdn)")
        if not -1 <= self.window_reach <= self.window_chunks:
            raise ValueError(f"window_reach {self.window_reach}: -1 (all) or 0 .. window_chunks ({self.window_chunks})")
        if self.read_block < 0 or (self.read_block and self.chunk_size % self.read_block):
            raise ValueError(f"read_block {self.read_block}: 0 or a divisor of chunk_size {self.chunk_size}")
        if self.memory == "gdn" and (self.inner_lr_gate or self.inner_token_gate or self.inner_decay > 0
                                     or self.inner_optimizer or self.inner_feat_head or self.inner_kvb_head):
            raise ValueError("memory gdn has no fast weight: the write controls / feature head do not apply")

    # derived
    @property
    def prefix_len(self) -> int:
        return self.num_hidden_layers - self.suffix_len

    @property
    def fast_len(self) -> int:
        """Suffix layers with a fast weight (``fast_mask``)."""
        return sum(self.fast_mask)

    @property
    def fast_mask(self) -> Tuple[bool, ...]:
        """Per suffix layer (bottom to top): does it carry a fast weight."""
        if self.fast_layers is not None:
            return tuple(self.prefix_len + i in self.fast_layers for i in range(self.suffix_len))
        return (True,) * (self.suffix_len - self.suffix_plain_top) + (False,) * self.suffix_plain_top

    @property
    def fast_layer_indices(self) -> Tuple[int, ...]:
        """The fast-weight layers' absolute indices, bottom to top (the order of the fast-weight stacks)."""
        return tuple(self.prefix_len + i for i, f in enumerate(self.fast_mask) if f)

    @property
    def suffix_runs(self) -> Tuple[Tuple[bool, int, int], ...]:
        """The suffix as maximal runs of same-kind layers: (fast, first suffix index, end), bottom to top."""
        m, runs, a = self.fast_mask, [], 0
        for i in range(1, len(m) + 1):
            if i == len(m) or m[i] != m[a]:
                runs.append((m[a], a, i))
                a = i
        return tuple(runs)

    @property
    def prime_intermediate_size(self) -> int:
        return self.ttt_intermediate_size or self.intermediate_size

    @property
    def reach(self) -> Optional[int]:
        """The visible window chunks for ``key_validity`` (None: all ``window_chunks``)."""
        return None if self.window_reach < 0 else self.window_reach

    @property
    def window_len(self) -> int:
        """Tokens of clean history a chunk attends to (its own chunk excluded)."""
        return self.window_chunks * self.chunk_size

    @classmethod
    def load(cls, path: str) -> "ModelConfig":
        return _from_dict(cls, _load_json(path))

    def save(self, path: str) -> None:
        d = dataclasses.asdict(self)
        if d["rope_scaling"] is not None:
            d["rope_scaling"] = dict(d["rope_scaling"])
        with open(path, "w") as f:
            json.dump(d, f, indent=2)


@dataclass(frozen=True)
class TrainConfig:
    # --- what to train on ---
    model_config: str = "configs/1b_e2e"
    sources: str = ""                       # "name=dir:weight,name=dir:weight"
    data_root: str = ""                     # relative stream directories (sources, eval_sources) and a relative
                                            # unigram_path are under this directory ("": the working directory)
    seq_len: int = 8192
    batch_size: int = 64                    # global
    grad_accum: int = 1                     # micro-batches of batch_size per optimiser step (exact: sum ce / sum weight)
    steps: int = 50_000
    seed: int = 42                          # init AND data order
    plant_frac: float = 0.0                 # planted recall: this fraction of the training sequences gets one span
                                            # copied from an early chunk to beyond its attention window (data.Plant)
    plant_len_min: int = 32                 # planted span length range (tokens)
    plant_len_max: int = 128
    train_recall: bool = False              # recall-only training: every sequence gets one planted copy beyond the
                                            # window (plant_len_min..max, random chunks) and the outer loss is ONLY
                                            # its tail after train_recall_cue tokens, masked alt or block (t = 1)
    train_recall_cue: int = 16
    doc_windows: bool = False               # training sequences = single-document windows (data.doc_windows, built by
                                            # `python -m ttt.data windows`): no packing across documents
    # --- outer optimiser ---
    lr: float = 1e-3
    end_lr: float = 1e-5
    warmup_steps: int = 5000
    decay_start: int = -1                   # warmup-stable-decay over global steps: 0 -> lr over warmup_steps, flat,
    decay_end: int = -1                     # then linearly to end_lr from decay_start to decay_end (-1: no decay)
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    adam_b1: float = 0.9
    adam_b2: float = 0.95
    no_decay_pattern: str = r"embed|norm|bias|inner_|A_log"   # no AdamW decay on these (path regex; inner_ = write controls, A_log = gdn decay)
    fast_weight_lr_scale: float = 1.0       # the W_0 template's updates (Adam step and weight decay) times this: its
                                            # gradient is the noisiest group's (tools/grad_noise.py); 1 = no extra step
    # --- inner loop ---
    inner_lr: float = 1.0
    inner_clip: float = 1.0
    inner_lr_warmup_steps: int = 5000       # 0 disables the 0.1*lr -> lr ramp
    inner_mask_range: Tuple[float, float] = (0.5, 1.0)
    inner_mask_ratio: float = -1.0          # >= 0 pins the ratio (bit-reproducible ablations)
    second_order: bool = True
    # --- objective ---
    time_reweighting: str = "cart"          # cart | linear | original | none
    cart_p: float = 0.1
    cart_window: int = 512
    # --- memory / speed ---
    remat_levels: int = 1
    remat_prefix: bool = False              # checkpoint the whole prefix stack + per-stream MLP (32K+)
    prefix_remat_levels: int = 1            # nested remat over the prefix layer scan (2 at 128K: (3, 6))
    segment_chunks: int = 0                 # process the sequence in segments of this many chunks (0: whole)
    truncate_fast_weight: bool = False      # stop the meta-gradient at W on segment boundaries
    inner_views: int = 1                    # 2: complementary inner masks, gradients averaged
    inner_view_mode: str = "complement"     # "strata": V independent draws with stratified mask ratios
    inner_views_sequential: bool = False    # V sequential inner SGD steps per chunk (lr / V, clip / V each)
    inner_cart: bool = False                # CART-weight the inner mask-task loss like the outer loss
    inner_mask_span: float = 0.0            # > 0: span masking with this mean span length
    inner_update: str = "sgd"               # sgd | sign | muon2 (direction of the inner step)
    inner_task: str = "mask"                # mask | causal | split  (what the write predicts; chunk_scan.LossConfig)
    inner_from_outer: bool = False          # the write = the outer loss's gradient (no third stream, -15% step)
    inner_mask_mode: str = "random"         # random | rare | docaware (host-side deterministic targets, inner_masks.py)
    inner_target_frac: float = 0.5          # target fraction per chunk for the host-side rules
    inner_generic_floor: float = 0.1        # share of the target slots reserved for hash-chosen generic tokens
    no_write: bool = False                  # baseline: no inner stream, W stays W_0 (static parallel MLP)
    write_off: bool = False                 # keep the inner stream (its CE) but never update W (aux-loss control)
    aux_causal_weight: float = 0.0          # > 0: add the inner stream's CE to the outer objective with this weight
    inner_feat: str = "off"                 # off | add | only: the inner stream also / instead predicts the clean
                                            # stream's suffix input at the target slot (needs inner_feat_head)
    inner_feat_weight: float = 1.0          # weight of the feature term in 'add'
    inner_update_frac: float = 1.0          # < 1: the write touches a fixed random fraction of W (capacity probe, eval)
    # --- write schemes of the causal write (inner_loop.WriteConfig; defaults = one step per chunk) ---
    inner_subchunks: int = 1                # n sequential clipped steps per chunk, one per sub-chunk
    inner_subchunk_scale: float = 1.0       # clip of each sub-step relative to inner_clip (1/n: the budget of one step)
    inner_surprise_gamma: float = 0.0       # > 0: slot weights (CE_i / mean CE)^gamma of the model before the step
    inner_kvb_weight: float = 0.0           # > 0: + key-value-binding term in every suffix layer (needs inner_kvb_head)
    inner_kvb_value: str = "self"           # the binding's value source: self (the stream's own input) | clean
    inner_fast_frac: float = 0.0            # > 0: this fraction of W's hidden units is written per sub-chunk (fast),
                                            # the rest once per chunk (slow); needs inner_subchunks >= 2
    inner_fast_lr_scale: float = 1.0        # lr of the fast writes relative to inner_lr
    inner_fast_decay: float = 0.0           # the fast units relax toward W_0 by this fraction after every chunk
    inner_write_steps: int = 1              # K sequential clipped steps on the whole chunk (gradient recomputed; eval probe)
    inner_norm_alpha: float = 0.0           # > 0: the normalised step (WriteConfig.norm_alpha): f = min(cap, (max r / r)^alpha)
    inner_norm_cap: float = 1.0             # its cap
    window_drop: float = 0.0                # training: per step, the probability of no window (own chunk only; LossConfig)
    window_drop_frac: float = 0.9           # ... over the first this share of the steps (the rest keep the full window)
    self_prior: float = 0.0                 # training: share of sequences read after their own chunk's write (LossConfig)
    outer_views: int = 1                    # training: noisy reads of each chunk per step, each with its own mask and noise
                                            # level, sharing the clean stream and the write (LossConfig.outer_views)
    distill: bool = False                   # the write also matches a low-noise view's predictive distribution
                                            # (a fourth, bidirectional prefix stream masked at distill_t_range; its
                                            # top-k held constant; soft CE where its target is masked; chunk_scan)
    distill_t_range: Tuple[float, float] = (0.1, 0.3)   # the teacher view's mask ratio, sampled per chunk
    distill_weight: float = 1.0             # weight of the soft term next to the one-hot anchor CE
    distill_topk: int = 64                  # tokens of the teacher distribution kept (unnormalised)
    distill_meta: bool = True               # False: the soft term's write gradient is a constant (no meta-gradient)
    distill_warmup_steps: int = 0           # > 0: ramp the soft term's weight from 0 over this many steps
    mtp: bool = False                       # second target x_{i+2} through mtp_head: outer loss where it is masked
                                            # and the causal write at every slot (needs outer_shift, inner_task causal)
    mtp_weight: float = 0.3                 # weight of the outer MTP term (the reported loss stays the main CE)
    inner_mtp_weight: float = 0.5           # weight of the MTP term in the write
    fused_ce_block: int = 8192              # 0 -> dense CE (tests)
    attn_chunk_batch: int = 2               # bounds the prefix's f32 score tensor (XLA path)
    replicate_weights: str = "suffix,embed" # bf16 compute copies gathered once per step, not per use ("" = none)
    # --- checkpoints / logging ---
    ckpt_dir: str = "bucket/checkpoints"
    ckpt_interval: int = 1000
    keep_checkpoints: int = 3               # newest N kept on disk (0 = all); 15 GB each at 1B
    resume: str = ""                        # "auto" | path
    resume_legacy_root: str = ""            # resuming a checkpoint written before 2026-10: the directory its
                                            # relative stream paths were relative to (default: the current one)
    init_from: str = ""                     # weights only, fresh schedule
    continue_from: str = ""                 # a complete step_* directory of another run: its weights, optimiser
                                            # state and step, continued with this run's data and settings (a branch,
                                            # the next stage); once this run has a checkpoint of its own, that one is
                                            # resumed instead (implies resume auto)
    log_interval: int = 10
    # --- evaluation (held-out tail of every source; see data.Sampler) ---
    holdout: int = 4_194_304                # tokens per source never used for training (0 disables eval)
    eval_seed: int = 12345                  # fixed PRNG seed for stratified held-out eval masks
    outer_t: float = -1.0                   # >= 0: pin diffusion time t for evaluation / needle probing
    outer_shift: bool = False               # slot i predicts token i+1 (Dream-style objective shift)
    log_dir: str = ""                       # directory for eval_<step>.npz persistence
    eval_batches: int = 1                   # step batches (batch_size x grad_accum sequences) per source per evaluation
    eval_interval: int = 0                  # steps between evaluations (0 -> every checkpoint)
    eval_inner_lr_scales: str = "1"         # extra evaluations at k x inner_lr, e.g. "0.5,1,2,4"
    eval_only: bool = False                 # evaluate the resumed / initialised weights and exit
    eval_step0: bool = True                 # run the step-0 baseline evaluation before the training loop
    eval_probes: bool = False               # per-token probes: recurrence (P1) and rarity (P2) classes
    eval_sources: str = ""                  # evaluation-only sources "name=dir:w,..." (reported, never trained on)
    unigram_path: str = "bucket/data/tokens/unigram_llama3.i64"   # int64 counts for the rarity probe
    eval_fixed_t: str = ""                  # extra evaluations at pinned noise levels, e.g. "0.5,1.0" (unweighted CE,
                                            # write on / off; "[eval N t=1]" lines, "eval t=1/..." keys, eval_N_t1.npz)
    eval_reread_chunks: int = 0             # evaluation: re-read the first n chunks (W written twice) and score
                                            # the tail, paired against the single pass (needs segment_chunks | n)
    eval_score_from_chunk: int = -1         # with eval_reread_chunks: first scored chunk (-1: n; 0: also score the
                                            # re-read chunks themselves in the second pass = verbatim recall)
    eval_planted_recall: bool = False       # planted-recall probe (probes.planted_probe_batch): a span of chunk 1
                                            # copied beyond the window (and, as the ceiling, also inside it); its
                                            # tail after the cue masked; write on / off; eval_<step>_planted.npz
    eval_planted_batches: int = 1           # step batches per source for the probe
    eval_planted_len: int = 64              # planted span length (tokens)
    eval_planted_cue: int = 16              # visible leading tokens of the planted copy
    eval_write_plan: str = ""               # "c:k,..." evaluation-time whole-chunk write steps of chunk c (default 1,
                                            # 0 = no write): the storage / read-out / interference oracle
    prefetch: int = 4
    wandb_project: str = ""
    probe_memory: bool = False              # compile, print XLA's memory analysis, exit
    profile_dir: str = ""                   # jax.profiler traces: 2 steps after start-up, and on `touch <dir>/TRIGGER`
    fsdp_size: int = 0                      # intra-node FSDP size for 2D mesh (0 = pure 1D over all devices)

    def __post_init__(self):
        if self.fsdp_size < 0:
            raise ValueError(f"fsdp_size must be non-negative, got {self.fsdp_size}")
        if self.seq_len % 1 != 0 or self.seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if self.time_reweighting not in ("cart", "linear", "original", "none"):
            raise ValueError(f"time_reweighting {self.time_reweighting!r}")
        lo, hi = self.inner_mask_range
        if not 0.0 <= lo <= hi <= 1.0:
            raise ValueError("inner_mask_range must satisfy 0 <= lo <= hi <= 1")
        if self.inner_view_mode not in ("complement", "strata"):
            raise ValueError(f"inner_view_mode {self.inner_view_mode!r} (complement | strata)")
        if self.inner_view_mode == "complement" and self.inner_views not in (1, 2):
            raise ValueError("inner_views must be 1 or 2 with complementary views (use inner_view_mode strata for more)")
        if self.inner_view_mode == "strata" and self.inner_views < 2:
            raise ValueError("inner_view_mode strata needs inner_views >= 2")
        if self.inner_update not in ("sgd", "sign", "muon2"):
            raise ValueError(f"inner_update {self.inner_update!r} (sgd | sign | muon2)")
        if self.inner_mask_mode not in ("random", "rare", "docaware"):
            raise ValueError(f"inner_mask_mode {self.inner_mask_mode!r} (random | rare | docaware)")
        if not 0.0 < self.inner_update_frac <= 1.0:
            raise ValueError("inner_update_frac in (0, 1]")
        if self.inner_feat not in ("off", "add", "only") or self.inner_feat_weight < 0:
            raise ValueError(f"inner_feat {self.inner_feat!r} (off | add | only), weight >= 0")
        if self.inner_task not in ("mask", "causal", "split"):
            raise ValueError(f"inner_task {self.inner_task!r} (mask | causal | split)")
        dlo, dhi = self.distill_t_range
        if not 0.0 <= dlo <= dhi <= 1.0 or self.distill_topk < 1 or self.distill_weight < 0:
            raise ValueError("distill_t_range must satisfy 0 <= lo <= hi <= 1, distill_topk >= 1, distill_weight >= 0")
        if self.distill and (self.inner_task == "split" or self.inner_views != 1 or self.no_write
                             or self.inner_from_outer or self.eval_reread_chunks):
            raise ValueError("distill needs inner_task mask | causal, one inner view, a write (not no_write / "
                             "inner_from_outer) and no re-read evaluation")
        if self.aux_causal_weight < 0:
            raise ValueError("aux_causal_weight must be >= 0")
        if not 0.0 <= self.plant_frac < 1.0 or not 2 <= self.plant_len_min <= self.plant_len_max:
            raise ValueError("plant_frac in [0, 1) and 2 <= plant_len_min <= plant_len_max")
        if self.train_recall and (self.plant_frac > 0 or self.grad_accum != 1
                                  or not 0 < self.train_recall_cue < self.plant_len_min):
            raise ValueError("train_recall plants itself (plant_frac 0), needs grad_accum 1 and 0 < train_recall_cue < plant_len_min")
        if self.eval_write_plan:
            try:
                plan = [tuple(int(v) for v in x.split(":")) for x in self.eval_write_plan.split(",") if x.strip()]
                ok = all(len(e) == 2 and e[0] >= 0 and e[1] >= 0 for e in plan)
            except ValueError:
                ok = False
            if not ok or self.inner_task != "causal":
                raise ValueError("eval_write_plan: 'chunk:steps,...' with chunk, steps >= 0 (causal write)")
        if self.eval_planted_recall and not (0 < self.eval_planted_cue < self.eval_planted_len
                                             and self.eval_planted_batches >= 1):
            raise ValueError("eval_planted_recall needs 0 < eval_planted_cue < eval_planted_len and eval_planted_batches >= 1")
        if not (0.0 <= self.window_drop <= 1.0 and 0.0 <= self.window_drop_frac <= 1.0 and 0.0 <= self.self_prior <= 1.0):
            raise ValueError("window_drop, window_drop_frac and self_prior are in [0, 1]")
        if self.self_prior > 0 and (self.no_write or self.write_off or self.inner_from_outer):
            raise ValueError("self_prior reads a chunk after its own write: it needs a write")
        if self.fast_weight_lr_scale < 0:
            raise ValueError("fast_weight_lr_scale >= 0")
        if self.outer_views < 1 or (self.outer_views > 1 and (self.distill or self.mtp or self.inner_from_outer)):
            raise ValueError("outer_views >= 1, and > 1 not with distill / mtp / inner_from_outer")
        write_schemes = (self.inner_subchunks != 1 or self.inner_subchunk_scale != 1.0 or self.inner_surprise_gamma > 0
                         or self.inner_kvb_weight > 0 or self.inner_fast_frac > 0 or self.inner_write_steps != 1
                         or self.inner_norm_alpha > 0)
        if write_schemes and (self.inner_task != "causal" or self.inner_views != 1 or self.no_write
                              or self.inner_from_outer or self.write_off or self.distill or self.mtp
                              or self.inner_feat != "off" or self.inner_update_frac < 1.0):
            raise ValueError("the write schemes (inner_subchunks / surprise / kvb / fast / write_steps) need the plain causal write "
                             "(inner_task causal, one view, no no_write / inner_from_outer / write_off / distill / mtp / "
                             "inner_feat / inner_update_frac)")
        if self.write_off and (self.no_write or self.inner_from_outer or self.distill or self.mtp or self.inner_views != 1
                               or self.inner_feat != "off"):
            raise ValueError("write_off keeps one plain inner stream (not no_write / inner_from_outer / distill / mtp / views / feat)")
        if self.aux_causal_weight > 0 and (self.no_write or self.inner_from_outer):
            raise ValueError("aux_causal_weight needs an inner stream (not no_write / inner_from_outer)")
        if self.mtp_weight < 0 or self.inner_mtp_weight < 0:
            raise ValueError("mtp_weight and inner_mtp_weight must be >= 0")
        if self.mtp and (not self.outer_shift or self.inner_task != "causal" or self.no_write or self.inner_from_outer):
            raise ValueError("mtp needs outer_shift, inner_task causal and a causal write (not no_write / inner_from_outer)")
        # `--resume auto --init_from X` is the restartable stage-2 recipe: resume THIS
        # run's newest checkpoint if one exists, otherwise start from X's weights.
        # An explicit --resume PATH with --init_from is contradictory and refused.
        if self.resume and self.resume != "auto" and self.init_from:
            raise ValueError("--resume <path> and --init_from are mutually exclusive (use --resume auto)")
        if self.continue_from and (self.continue_from == "auto" or self.init_from or self.resume not in ("", "auto")):
            raise ValueError("--continue_from <step dir> resumes this run's own newest checkpoint first (resume auto): "
                             "not with --init_from or --resume <path>")
        if self.continue_from and (os.path.dirname(os.path.abspath(self.continue_from.rstrip("/")))
                                   == os.path.abspath(self.ckpt_dir)):
            raise ValueError(f"--continue_from {self.continue_from} is in this run's own ckpt_dir: a branch or stage "
                             "needs a ckpt_dir of its own (to restart a run, use --resume auto)")
        if (self.decay_start >= 0 or self.decay_end >= 0) and not max(1, self.warmup_steps) <= self.decay_start < self.decay_end:
            raise ValueError(f"decay_start {self.decay_start}, decay_end {self.decay_end}: "
                             "warmup_steps <= decay_start < decay_end, or both -1 (no decay)")
        if self.eval_reread_chunks < 0 or self.eval_score_from_chunk < -1:
            raise ValueError("eval_reread_chunks >= 0 and eval_score_from_chunk >= -1")

    @classmethod
    def load(cls, path: str) -> "TrainConfig":
        d = _load_json(path)
        for name in ("inner_mask_range", "distill_t_range"):
            if name in d:
                d[name] = tuple(d[name])
        return _from_dict(cls, d)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(dataclasses.asdict(self), f, indent=2)

    def data_path(self, path: str) -> str:
        """``path`` under ``data_root`` when it is relative (an absolute path is kept)."""
        return os.path.join(self.data_root, path) if path and self.data_root else path

    def replace(self, **kw) -> "TrainConfig":
        return dataclasses.replace(self, **kw)
