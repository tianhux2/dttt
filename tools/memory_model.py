# coding=utf-8
"""
Per-chip memory and compute model for the 1B parallel-MLP TTT step.

Every number in the plan comes from here. Run it after changing the config so
the prose stays honest:

    .venv/bin/python tools/memory_model.py --config configs/1b_e2e/config.json
"""
import argparse
import json
import math
import os

GB = 1024 ** 3

CHIP_GB = 30.75
N_CHIPS = 32
# f32 master + f32 grad + 2 f32 AdamW moments + the bf16 compute copy, all FSDP-sharded.
BYTES_PER_PARAM = 4 + 4 + 8 + 2
# Working set of one suffix chunk step's backward (grad-of-grad through the inner
# loop, fused CE, flash A/B + XLA C attention) plus one prefix layer's backward,
# per sequence per chip.  MEASURED, not derived: 1B/8K on a 4-chip node compiles
# and runs at 2 seq/chip (B=8) with remat_levels=1 (33.98 GB before the cond-free
# scan and bias-mask attention, fits after); treat as ~6 GiB per sequence.
TRANSIENT_GIB_PER_SEQ = 6.0


def load(path):
    with open(path) as f:
        c = json.load(f)
    c.setdefault("ttt_intermediate_size", None)
    if "sliding_window_chunks" in c:                      # pre-rebuild key
        c["window_chunks"] = c.pop("sliding_window_chunks")
    return c


def params(c):
    H, I, L = c["hidden_size"], c["intermediate_size"], c["num_hidden_layers"]
    hd, nq, nkv = c["head_dim"], c["num_attention_heads"], c["num_key_value_heads"]
    S = c["suffix_len"]
    bias = 3 * nkv * hd if c.get("attention_bias", True) else 0
    attn = 2 * H * nq * hd if nq == nkv else (H * nq * hd + 2 * H * nkv * hd + nq * hd * H)
    attn = H * nq * hd + 2 * H * nkv * hd + nq * hd * H + bias
    qkn = 2 * hd if c.get("qk_norm") else 0
    mlp = 3 * H * I
    prime_I = c["ttt_intermediate_size"] or I
    prime = 3 * H * prime_I
    emb = c["vocab_size"] * H
    return {
        "attn_per_layer": attn + qkn, "mlp_per_layer": mlp,
        "base": L * (attn + qkn + mlp + 2 * H),
        "prime": S * (prime + 2 * H),
        "embed": emb, "tied": bool(c.get("tie_word_embeddings")),
    }


def carry(c, kv_dtype=2, w_dtype=4, suffix_only=True):
    H, S = c["hidden_size"], c["suffix_len"]
    hd, nkv = c["head_dim"], c["num_key_value_heads"]
    cap = c.get("window_chunks", 1) * c["chunk_size"]      # carried history only; own chunk is not cached
    prime_I = c["ttt_intermediate_size"] or c["intermediate_size"]
    n_kv_layers = S if suffix_only else c["num_hidden_layers"]
    return {
        "fast_weight": S * 3 * H * prime_I * w_dtype,
        "kv": n_kv_layers * cap * 2 * nkv * hd * kv_dtype,
    }


def flops_per_token(c, seq_len, remat_levels=1):
    """Model FLOPs per token of one training step (matmuls only; what MFU counts) and
    the hardware FLOPs actually executed (adds rematerialisation).

    Per chunk the suffix runs forwards A (outer), B (clean), C (inner-masked), the
    inner backward of C (w.r.t. W and activations), and the outer backward through
    all of it including the inner backward (second order).  Attention is windowed,
    so its cost per token is independent of seq_len; the splash prefix skips the
    invalid window blocks (44% at 8K), the suffix pays the full (W+1)*C keys.
    """
    H, I, L, S = c["hidden_size"], c["intermediate_size"], c["num_hidden_layers"], c["suffix_len"]
    nh, nkv, d = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
    C, W, V = c["chunk_size"], c["window_chunks"], c["vocab_size"]
    prime_I = c["ttt_intermediate_size"] or I
    nc = seq_len // C
    p_layer = H * nh * d + 2 * H * nkv * d + nh * d * H + 3 * H * I          # per layer, params
    p_prime = 3 * H * prime_I
    # attention matmuls per token per layer: 4 * keys * nh * d (QK^T and PV)
    keys_valid = sum(min(i, W) + 1 for i in range(nc)) / nc * C              # prefix (splash skips invalid)
    attn_prefix = 4 * keys_valid * nh * d
    attn_suffix = 4 * (W + 1) * C * nh * d
    fwd_prefix = (L - S) * (2 * p_layer + attn_prefix)                       # one stream
    fwd_suffix = S * (2 * (p_layer + p_prime) + attn_suffix)                 # one forward
    ce = 2 * V * H                                                           # logits for one forward
    model = (3 * fwd_prefix * 3                                              # 3 streams, fwd + bwd
             + 3 * fwd_suffix + 2 * ce                                       # A, B, C forwards + 2 CEs
             + 2 * (fwd_suffix + ce)                                         # inner backward of C
             + 2 * (2 * fwd_suffix + ce + 3 * (fwd_suffix + ce)))            # outer bwd: A+CE, B, C+CE+inner bwd
    R = remat_levels
    hardware = model + (3 * fwd_suffix + 2 * ce + 2 * (fwd_suffix + ce)) * R    # chunk-step recompute per level
    hardware += 3 * fwd_prefix + fwd_suffix * 3                              # per-layer checkpoints (one extra fwd)
    return model, hardware


def balanced(n, levels):
    from src.ttt.remat_scan import balanced_factors
    return balanced_factors(n, levels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/1b_e2e/config.json")
    ap.add_argument("--chips", type=int, default=N_CHIPS)
    ap.add_argument("--step_time", type=float, default=0.0, help="measured s/step -> MFU")
    ap.add_argument("--seq_len", type=int, default=8192)
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    c = load(a.config)
    p = params(c)

    nonemb = p["base"] + p["prime"]
    total = nonemb + p["embed"] * (1 if p["tied"] else 2)
    print(f"=== {a.config} ===")
    print(f"  hidden {c['hidden_size']}  layers {c['num_hidden_layers']}  "
          f"heads {c['num_attention_heads']}/{c['num_key_value_heads']}  head_dim {c['head_dim']}")
    print(f"  intermediate {c['intermediate_size']}  suffix_len {c['suffix_len']}  "
          f"vocab {c['vocab_size']}  tied {p['tied']}")
    print(f"\nparameters")
    print(f"  attention / layer   {p['attn_per_layer']/1e6:8.2f} M")
    print(f"  MLP / layer         {p['mlp_per_layer']/1e6:8.2f} M")
    print(f"  base ({c['num_hidden_layers']} layers)     {p['base']/1e6:8.2f} M")
    print(f"  prime ({c['suffix_len']} layers)     {p['prime']/1e6:8.2f} M   <- the fast weight")
    print(f"  NON-EMBEDDING       {nonemb/1e6:8.2f} M")
    print(f"  embedding           {p['embed']/1e6:8.2f} M")
    print(f"  TOTAL               {total/1e9:8.3f} B")

    static = BYTES_PER_PARAM * total / a.chips / GB
    print(f"\nstatic per chip on {a.chips} chips: {static:.2f} GiB "
          f"({BYTES_PER_PARAM} B/param: f32 master+grad+2 moments, bf16 copy)")
    print(f"\ncarry, per sequence per live scan carry (chunk {c['chunk_size']}, "
          f"kv window {c.get('window_chunks',1) * c['chunk_size']} tokens)")
    for label, so in (("suffix-only KV (prefix split)", True), ("all-layer KV", False)):
        k = carry(c, suffix_only=so)
        print(f"  {label:<30} fast weight {k['fast_weight']/GB:6.3f} GiB + "
              f"KV {k['kv']/GB:6.3f} GiB = {(k['fast_weight']+k['kv'])/GB:6.3f} GiB")

    k = carry(c, suffix_only=True)
    per_seq = (k["fast_weight"] + k["kv"]) / GB
    H, L, S = c["hidden_size"], c["num_hidden_layers"], c["suffix_len"]
    print(f"\nper-chip total = static + prefix residuals + live carries + transient "
          f"({TRANSIENT_GIB_PER_SEQ:.0f} GiB/seq measured); HBM {CHIP_GB} GiB")
    print(f"  {'stage':<14} {'seq/chip':>8} {'chunks':>6} {'levels':>6} {'live':>5} {'prefix':>7} "
          f"{'carry':>6} {'trans':>6} {'total':>7}  fits?")
    for name, seq_len, spc, rp in (("8K pretrain", 8192, 2, False), ("8K 1/chip", 8192, 1, False),
                                   ("32K extension", 32768, 1, True), ("32K no rp", 32768, 1, False)):
        nch = seq_len // c["chunk_size"]
        # per-layer checkpoint keeps each prefix layer's [3, T, H] bf16 input; remat_prefix
        # keeps only the embedding output and recomputes the stack in the backward.
        prefix = (1 if rp else (L - S)) * 3 * seq_len * H * 2 * spc / GB
        for lv in (1, 2, 3):
            live = sum(balanced(nch, lv))
            cr = live * spc * per_seq
            tr = TRANSIENT_GIB_PER_SEQ * spc
            tot_gib = static + prefix + cr + tr
            print(f"  {name:<14} {spc:>8} {nch:>6} {lv:>6} {live:>5} {prefix:>7.1f} {cr:>6.1f} {tr:>6.1f} "
                  f"{tot_gib:>7.1f}  {'YES' if tot_gib < CHIP_GB else 'no'}")

    # Segmented scan (segment_chunks = 8): the outer scan keeps one carry per
    # segment boundary (every prefix layer's KV window + suffix window + f32 W);
    # inside the segment being differentiated, the 8K program's residuals are live.
    nkv, d, Wn = c["num_key_value_heads"], c["head_dim"], c.get("window_chunks", 1)
    seg_carry = per_seq + (L - S) * Wn * c["chunk_size"] * nkv * d * 2 * 2 / GB
    print(f"\n  segmented (8 chunks/segment, remat levels 1/1, 1 seq/chip): carry per boundary "
          f"{seg_carry:.2f} GiB")
    for seq_len in (32768, 65536, 131072):
        nseg = seq_len // (8 * c["chunk_size"])
        prefix = (L - S) * 3 * 8 * c["chunk_size"] * H * 2 / GB
        tot_gib = static + nseg * seg_carry + prefix + 8 * per_seq + TRANSIENT_GIB_PER_SEQ
        print(f"  {seq_len // 1024:>5}K  {nseg:>3} segments  boundaries {nseg * seg_carry:5.1f} + segment "
              f"{prefix + 8 * per_seq:4.1f} + transient {TRANSIENT_GIB_PER_SEQ:.0f} + static {static:.1f} = "
              f"{tot_gib:5.1f} GiB  {'YES' if tot_gib < CHIP_GB else 'no'}")

    # Predicted fast-weight drift. `--diagnose_ttt` reports ||W_C - W_0|| / ||W_0||,
    # and the inner update is clipped to a GLOBAL norm of `clip`, so each chunk moves
    # W by at most clip*lr in absolute norm:
    #
    #     drift <= n_chunks * clip * lr / ||W_0||,   ||W_0|| = std * sqrt(3*S*H*I)
    #
    # ||W_0|| scales as sqrt(params), so the SAME lr/clip gives wildly different
    # drift at different model sizes: a tiny debug config over-reports it by ~40x.
    # Never read a drift number from a small config as if it transferred.
    H, S = c["hidden_size"], c["suffix_len"]
    prime_I = c["ttt_intermediate_size"] or c["intermediate_size"]
    w0 = c.get("initializer_range", 0.02) * math.sqrt(3 * S * H * prime_I)
    print(f"\npredicted TTT drift at inner_lr=1.0, inner_clip=1.0  (||W_0|| = {w0:.1f})")
    for name, seq_len in (("8K pretrain", 8192), ("32K extension", 32768)):
        nch = seq_len // c["chunk_size"]
        d = nch * 1.0 / w0
        flag = "ok" if 1e-3 < d < 1e-1 else ("HOT" if d >= 1e-1 else "cold - fast weight idle")
        print(f"  {name:<16} {nch:>4} chunks   drift <= {d:6.3f}   {flag}")

    m, hw = flops_per_token(c, a.seq_len)
    tokens = a.batch * a.seq_len
    print(f"\nFLOPs per token (seq {a.seq_len}): model {m/1e9:.1f} GFLOP, hardware incl. remat {hw/1e9:.1f} GFLOP"
          f"  ({6 * total / 1e9:.1f} for a dense LM of the same size)")
    if a.step_time:
        peak = 275e12                                                        # v4 bf16 per chip
        print(f"  step: {tokens/1e3:.0f}k tokens on {a.chips} chips in {a.step_time:.2f} s -> "
              f"{m * tokens / a.step_time / a.chips / 1e12:.1f} TFLOP/s/chip model = {100 * m * tokens / a.step_time / a.chips / peak:.1f}% MFU; "
              f"hardware {100 * hw * tokens / a.step_time / a.chips / peak:.1f}% of peak")

    print(f"\ncompute, layer-forward-equivalents per chunk")
    L, S = c["num_hidden_layers"], c["suffix_len"]
    for lv in (1, 2, 4):
        f = balanced(8192 // c["chunk_size"], lv)
        R = len(f)
        fused = L * (5 * R + 2)
        split = (L - S) * (2 * 2 + 2) + S * (5 * R + 2)
        print(f"  remat levels {R}: fused-scan {fused:>5}   prefix-split {split:>5}   "
              f"({fused/split:.2f}x cheaper)")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
