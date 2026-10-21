# coding=utf-8
"""Shared pieces of the ``tools.j_lens`` command lines: the platform guard, model and data loading."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_T = "bucket/data/tokens"
# The long-document mix of the 2026-09-28 runs and the PG-19 test set; the sequences are the
# held-out tails the evaluation uses (Sampler.eval_batch), never training data.
DEFAULT_SOURCES = ",".join(f"{n}={_T}/{d}:1.0" for n, d in (
    ("books3", "books3_b01d544bc3b822cb"), ("fineweb_edu_long", "fineweb_edu_long_4fc62ac88e715cab"),
    ("dclm_long", "dclm_long_bffad369a272d9d3"), ("starcoder_repo", "starcoder_repo_55e5d779e13dbf47"),
    ("open_web_math_long", "open_web_math_long_d8aa2b8eeec16cf7"),
    ("finemath_long", "finemath_long_07c6fc8339d65114"), ("wikipedia_long", "wikipedia_long_da5c08b48d22e6d0"),
    ("pg19_test", "pg19_test_a4dfccb3b3ac65f9")))


def require_platform() -> None:
    """Refuse to start without an explicit platform: on a pod host an unpinned JAX takes the TPU the
    running job holds (or waits for the other hosts forever).  Also default the attention to the XLA
    path: ``direct_effect`` is a forward-mode derivative, and the splash kernel's custom VJP has none."""
    if not os.environ.get("JAX_PLATFORMS"):
        sys.exit("set JAX_PLATFORMS=cpu, or pin one idle TPU host: "
                 "TPU_HOST_BOUNDS=1,1,1 TPU_CHIPS_PER_HOST_BOUNDS=2,2,1 JAX_PLATFORMS=tpu")
    os.environ.setdefault("TTT_ATTENTION", "xla")
    # f32 matmuls at full precision: JAX's default is one bf16 pass on TPU and TF32 on GPU - too coarse for the
    # small logit differences the probes decode, even with --dtype float32.
    import jax
    jax.config.update("jax_default_matmul_precision", "highest")


def add_common(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--ckpt", default=None, help="checkpoint step directory (reads model.eqx, train_config.json, state.json)")
    ap.add_argument("--model_config", default=None, help="default: the checkpoint's own")
    ap.add_argument("--random_init", action="store_true", help="tests: an untrained model of --model_config")
    ap.add_argument("--inner_lr", type=float, default=None, help="default: the checkpoint's ramp at its step")
    ap.add_argument("--inner_task", default=None, help="default: the checkpoint's")
    ap.add_argument("--sources", default=DEFAULT_SOURCES)
    ap.add_argument("--n_per_source", type=int, default=2)
    ap.add_argument("--seq_len", type=int, default=8192)
    ap.add_argument("--holdout", type=int, default=4_194_304)
    ap.add_argument("--synthetic", type=int, default=0, help="tests: random sequences over this many token ids")
    ap.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_index", type=int, default=0,
                    help="which held-out batch of every source (0: the probe set; 1, 2, ...: other sequences)")


def load(args):
    """``(model, write settings)`` from ``--ckpt`` or ``--random_init`` + ``--model_config``."""
    import jax
    from src.ttt.config import ModelConfig
    from src.ttt.generate import load_model
    from src.ttt.model import Model
    from . import core
    if args.random_init:
        if args.model_config is None or args.inner_lr is None:
            sys.exit("--random_init needs --model_config and --inner_lr")
        model = Model(ModelConfig.load(args.model_config), jax.random.PRNGKey(args.seed))
        return model, core.WriteSettings(args.inner_lr, inner_task=args.inner_task or "causal", fused_ce_block=0)
    if args.ckpt is None:
        sys.exit("--ckpt (or --random_init) is required")
    model = load_model(args.ckpt, args.model_config)
    return model, core.write_settings(args.ckpt, inner_lr=args.inner_lr, inner_task=args.inner_task)


def sequences(args, cfg):
    """``[(source name, clean tokens [seq_len])]``: held-out tails of every source, or synthetic ones."""
    if args.synthetic:
        rng = np.random.default_rng(args.seed)
        hi = min(args.synthetic, cfg.mask_token_id)
        return [("synthetic", rng.integers(1, hi, args.seq_len).astype(np.int32)) for _ in range(args.n_per_source)]
    from src.ttt.data import Sampler, parse_sources
    srcs = parse_sources(args.sources)
    smp = Sampler(srcs, args.seq_len, args.n_per_source, 42, holdout_tokens=args.holdout, eos=cfg.eos_token_id)
    return [(src.name, np.asarray(row, np.int32)) for si, src in enumerate(srcs)
            for row in smp.eval_batch(si, getattr(args, "batch_index", 0))]


def tokenizer_for(model_config: str):
    try:
        from transformers import PreTrainedTokenizerFast      # not AutoTokenizer: it parses config.json too
        return PreTrainedTokenizerFast.from_pretrained(model_config)
    except Exception:
        return None
