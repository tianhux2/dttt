# coding=utf-8
"""
Pallas flash attention == XLA reference on the model's exact geometry (key-only
validity mask, GQA folded into the query axis, chunk of C against W*C+C keys with
leading invalid slots), unsharded and under the FSDP mesh, and end to end through
``loss_fn`` (prefix + suffix A/B on the kernel, inner forward C on XLA).

TPU only - skips elsewhere.  Run on one pinned node:
    TPU_HOST_BOUNDS=1,1,1 TPU_CHIPS_PER_HOST_BOUNDS=2,2,1 .venv/bin/python tests/test_flash_attention.py
"""
from __future__ import annotations

import dataclasses
import os
import sys

# NOT "highest": the Pallas kernel is bf16-native, and forcing fp32 contraction over
# bf16 operands makes Mosaic reject the matmul ("Bad lhs type") on v4.
os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "default")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

if jax.default_backend() != "tpu":
    print("[SKIP] flash attention is a Pallas TPU kernel; run on a TPU node")
    sys.exit(0)

from src.ttt import attention as A
from src.ttt import model as M
from src.ttt.chunk_scan import Batch, LossConfig, loss_fn
from src.ttt.config import ModelConfig
from src.ttt.model import Model, key_validity, key_validity_all
from src.ttt.sharding import AXIS, make_mesh

# Production geometry: 1B heads, C=1024, window 3 (K=4096).
BIG = ModelConfig(vocab_size=128, hidden_size=2048, intermediate_size=64, num_hidden_layers=2,
                  num_attention_heads=16, num_key_value_heads=4, head_dim=128, suffix_len=1,
                  chunk_size=1024, window_chunks=3, mask_token_id=1, pad_token_id=0, eos_token_id=0)
# Smallest shapes the kernel accepts (128-multiples): the smoke-config family.
TINY = ModelConfig(vocab_size=256, hidden_size=128, intermediate_size=256, num_hidden_layers=4,
                   num_attention_heads=4, num_key_value_heads=2, head_dim=32, suffix_len=2,
                   ttt_intermediate_size=256, chunk_size=64, window_chunks=1,
                   mask_token_id=250, pad_token_id=0, eos_token_id=0)

failures = []


def report(name, value, tol, fmt="max rel = {:.3e}"):
    ok = value < tol
    failures.append(name) if not ok else None
    print(f"[{'PASS' if ok else 'FAIL'}] {name:<44} " + fmt.format(value))


def rel(a, b):
    return float(jnp.max(jnp.abs(a - b))) / (float(jnp.max(jnp.abs(a))) + 1e-12)


def kernel_checks(cfg, tag):
    attn = M.Attention(cfg, jax.random.PRNGKey(0))
    C, W, nh, nkv, d = cfg.chunk_size, cfg.window_chunks, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    K = (W + 1) * C
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(1), 3)
    q = jax.random.normal(k1, (2, C, nh, d), jnp.bfloat16)
    k = jax.random.normal(k2, (2, K, nkv, d), jnp.bfloat16)
    v = jax.random.normal(k3, (2, K, nkv, d), jnp.bfloat16)
    # chunk 0 (leading invalid window) and chunk W (all valid)
    kv = jnp.stack([key_validity(jnp.int32(0), C, W), key_validity(jnp.int32(W), C, W)])

    def run(impl, q, k, v):
        return attn.attend(q, k, v, kv, impl=impl).astype(jnp.float32)

    o_x, o_f = jax.jit(lambda *a: run("xla", *a))(q, k, v), jax.jit(lambda *a: run("flash", *a))(q, k, v)
    report(f"{tag}: flash == xla forward (bf16)", rel(o_x, o_f), 2e-2)

    # invalid keys must contribute nothing: perturbing them changes nothing
    k_p = k.at[:, :C].add(3.0)
    report(f"{tag}: invalid keys masked out", float(jnp.max(jnp.abs(run("flash", q, k_p, v)[0] - o_f[0]))),
           1e-12, "max abs = {:.1e}")

    def grads(impl):
        f = lambda q, k, v: jnp.sum(run(impl, q, k, v) ** 2)
        return jax.jit(jax.grad(f, argnums=(0, 1, 2)))(q, k, v)
    gx, gf = grads("xla"), grads("flash")
    report(f"{tag}: flash == xla backward (dq,dk,dv)", max(rel(a, b) for a, b in zip(gx, gf)), 2e-2)


def prefix_checks(cfg, tag, n_chunks):
    """Block.prefix_call: splash kernel with the static chunk-window mask == the
    gathered-window XLA reference, all three streams, forward and backward.  The
    TINY geometry has chunks smaller than the kernel blocks, so partial blocks
    exercise the in-kernel mask function too."""
    blk = M.Block(cfg, jax.random.PRNGKey(0))
    C, W, T = cfg.chunk_size, cfg.window_chunks, n_chunks * cfg.chunk_size
    h3 = jax.random.normal(jax.random.PRNGKey(4), (3, T, cfg.hidden_size), jnp.bfloat16)
    pos = jnp.arange(T)
    kva = key_validity_all(n_chunks, C, W)

    def run(impl, kva, kv_win):
        A.ATTENTION_IMPL = impl
        f = lambda h: jnp.sum(blk.prefix_call(h, pos, C, W, kva, 2, kv_win=kv_win)[0].astype(jnp.float32) ** 2)
        return jax.jit(jax.value_and_grad(f))(h3)
    (lx, gx), (lf, gf) = run("xla", kva, None), run("flash", kva, None)
    A.ATTENTION_IMPL = "auto"
    report(f"{tag}: prefix splash == xla value", abs(float(lx) - float(lf)) / abs(float(lx)), 1e-2)
    report(f"{tag}: prefix splash == xla grad", rel(gx.astype(jnp.float32), gf.astype(jnp.float32)), 5e-2)
    # With an incoming window (a later segment: all slots valid, and the first
    # segment: window present but no slot valid -> segment ids mask it).
    shape = (W * C, cfg.num_key_value_heads, cfg.head_dim)
    kv_win = (jax.random.normal(jax.random.PRNGKey(5), shape, jnp.bfloat16),
              jax.random.normal(jax.random.PRNGKey(6), shape, jnp.bfloat16))
    for first, name in ((n_chunks, "window valid"), (0, "window masked")):
        kva_s = key_validity_all(n_chunks, C, W, first=first)
        (lx, gx), (lf, gf) = run("xla", kva_s, kv_win), run("flash", kva_s, kv_win)
        A.ATTENTION_IMPL = "auto"
        report(f"{tag}: prefix splash == xla value ({name})", abs(float(lx) - float(lf)) / abs(float(lx)), 1e-2)
        report(f"{tag}: prefix splash == xla grad ({name})", rel(gx.astype(jnp.float32), gf.astype(jnp.float32)), 5e-2)
    # The inner stream's own-chunk patterns (causal / split) in the static splash mask.
    for own in ("causal", f"split:{C // 2}"):
        def run_own(impl):
            A.ATTENTION_IMPL = impl
            f = lambda h: jnp.sum(blk.prefix_call(h, pos, C, W, kva, 2, inner_own=own)[0][2].astype(jnp.float32) ** 2)
            return jax.jit(jax.value_and_grad(f))(h3)
        (lx, gx), (lf, gf) = run_own("xla"), run_own("flash")
        A.ATTENTION_IMPL = "auto"
        report(f"{tag}: prefix splash == xla value (inner {own})", abs(float(lx) - float(lf)) / abs(float(lx)), 1e-2)
        report(f"{tag}: prefix splash == xla grad (inner {own})", rel(gx.astype(jnp.float32), gf.astype(jnp.float32)), 5e-2)
    # Four streams: a causal inner stream followed by a bidirectional teacher (extra_full).
    h4 = jnp.concatenate([h3, h3[:1] * 0.7], axis=0)
    def run4(impl):
        A.ATTENTION_IMPL = impl
        f = lambda h: jnp.sum(blk.prefix_call(h, pos, C, W, kva, 2, inner_own="causal", extra_full=1)[0].astype(jnp.float32) ** 2)
        return jax.jit(jax.value_and_grad(f))(h4)
    (lx, gx), (lf, gf) = run4("xla"), run4("flash")
    A.ATTENTION_IMPL = "auto"
    report(f"{tag}: prefix splash == xla value (causal inner + teacher stream)", abs(float(lx) - float(lf)) / abs(float(lx)), 1e-2)
    report(f"{tag}: prefix splash == xla grad (causal inner + teacher stream)", rel(gx.astype(jnp.float32), gf.astype(jnp.float32)), 5e-2)


def sharded_checks(cfg, mesh):
    """vmap(spmd_axis_name) over a sharded batch + shard_map around the kernel:
    same numbers as the unsharded XLA path, and the kernel really was partitioned
    (the lowering would fail otherwise)."""
    attn = M.Attention(cfg, jax.random.PRNGKey(0))
    C, W, nh, nkv, d = cfg.chunk_size, cfg.window_chunks, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    K, nb = (W + 1) * C, 2 * mesh.size
    ks = jax.random.split(jax.random.PRNGKey(3), 3)
    sh = NamedSharding(mesh, P(AXIS))
    q = jax.device_put(jax.random.normal(ks[0], (nb, 2, C, nh, d), jnp.bfloat16), sh)
    k = jax.device_put(jax.random.normal(ks[1], (nb, 2, K, nkv, d), jnp.bfloat16), sh)
    v = jax.device_put(jax.random.normal(ks[2], (nb, 2, K, nkv, d), jnp.bfloat16), sh)
    kv = jnp.stack([key_validity(jnp.int32(0), C, W), key_validity(jnp.int32(W), C, W)])

    def loss(impl, q, k, v):
        per_seq = lambda q, k, v: jax.lax.map(lambda t: attn.attend(t[0][None], t[1][None], t[2][None], t[3][None], impl=impl)[0],
                                              (q, k, v, kv))
        out = jax.vmap(per_seq, spmd_axis_name=AXIS)(q, k, v).astype(jnp.float32)
        return jnp.sum(out ** 2), out

    def run(impl):
        (l, out), g = jax.jit(jax.value_and_grad(lambda *a: loss(impl, *a), argnums=(0, 1, 2), has_aux=True))(q, k, v)
        return l, out, g

    lx, ox, gx = run("xla")
    lf, of, gf = run("flash")
    report("sharded: flash == xla forward", rel(ox, of), 2e-2)
    report("sharded: flash == xla backward", max(rel(a, b) for a, b in zip(gx, gf)), 2e-2)
    report("sharded: output stays batch-sharded", 0.0 if of.sharding.spec == P(AXIS) else 1.0, 0.5, "spec ok = {}")


def loss_fn_checks(mesh):
    """End to end: TINY model, B = 2/chip, flash (prefix + suffix A/B) vs xla."""
    model = Model(TINY, jax.random.PRNGKey(0))
    B, T = 2 * mesh.size, 8 * TINY.chunk_size
    batch = Batch(input_ids=jax.device_put(jax.random.randint(jax.random.PRNGKey(1), (B, T), 1, 200),
                                           NamedSharding(mesh, P(AXIS, None))))
    cfg = LossConfig(cart_window=32, inner_lr=0.05, inner_mask_ratio=0.3, fused_ce_block=0,
                     compute_dtype="bfloat16", attn_chunk_batch=2, remat_levels=2)

    def run(impl, **kw):
        A.ATTENTION_IMPL = impl
        c = dataclasses.replace(cfg, **kw)
        f = lambda m: loss_fn(m, batch, jax.random.PRNGKey(2), jnp.int32(5), c, mesh)[0]
        return eqx.filter_jit(eqx.filter_value_and_grad(f))(model)

    def worst(g1, g2):
        return max(rel(a, b) for a, b in zip(jax.tree_util.tree_leaves(eqx.filter(g1, eqx.is_array)),
                                             jax.tree_util.tree_leaves(eqx.filter(g2, eqx.is_array))))

    lx, gx = run("xla")
    lf, gf = run("flash")
    report("loss_fn: flash == xla loss (bf16 model)", abs(float(lx) - float(lf)) / abs(float(lx)), 1e-2)
    report("loss_fn: flash == xla grads (bf16 model)", worst(gx, gf), 5e-2)
    # Segmented (2 x 4 chunks, windowed splash kernel in the scan) == whole.
    ls, gs = run("flash", segment_chunks=4, remat_levels=1)
    report("loss_fn: segmented flash == whole loss", abs(float(lx) - float(ls)) / abs(float(lx)), 1e-2)
    report("loss_fn: segmented flash == whole grads", worst(gx, gs), 5e-2)
    # Re-read layout on the splash path: with the write off, the re-read (first 4 chunks
    # read twice, window restarted at the second pass) scores the tail exactly like the
    # single pass - the kernel's per-segment validity implements the reset.
    def fwd(impl, **kw):
        A.ATTENTION_IMPL = impl
        c = dataclasses.replace(cfg, segment_chunks=4, remat_levels=1, score_from_chunk=4, inner_lr=0.0, **kw)
        return eqx.filter_jit(lambda m: loss_fn(m, batch, jax.random.PRNGKey(2), jnp.int32(5), c, mesh)[0])(model)
    l_ctl, l_rr = fwd("flash"), fwd("flash", reread_chunks=4)
    l_ctl_x, l_rr_x = fwd("xla"), fwd("xla", reread_chunks=4)
    A.ATTENTION_IMPL = "auto"
    report("loss_fn: reread (write off) == single pass, flash", abs(float(l_ctl) - float(l_rr)) / abs(float(l_ctl)), 1e-3)
    report("loss_fn: reread flash == reread xla", abs(float(l_rr) - float(l_rr_x)) / abs(float(l_rr)), 1e-2)
    report("loss_fn: reread (write off) == single pass, xla", abs(float(l_ctl_x) - float(l_rr_x)) / abs(float(l_ctl_x)), 1e-5)
    # Distillation (teacher stream on the splash kernel, approximate top-k) and MTP, flash vs xla.
    model_m = Model(dataclasses.replace(TINY, mtp_head=True), jax.random.PRNGKey(0))
    def run_dm(impl, **kw):
        A.ATTENTION_IMPL = impl
        c = dataclasses.replace(cfg, inner_task="causal", outer_shift=True, distill=True, distill_topk=8, mtp=True, **kw)
        f = lambda m: loss_fn(m, batch, jax.random.PRNGKey(2), jnp.int32(5), c, mesh)[0]
        return eqx.filter_jit(eqx.filter_value_and_grad(f))(model_m)
    lx, gx = run_dm("xla")
    lf, gf = run_dm("flash")
    A.ATTENTION_IMPL = "auto"
    report("loss_fn: distill + mtp flash == xla loss", abs(float(lx) - float(lf)) / abs(float(lx)), 1e-2)
    report("loss_fn: distill + mtp flash == xla grads", worst(gx, gf), 5e-2)


def topk_checks():
    """The streaming teacher top-k (approx_max_k per block) against the exact top-k at
    the production vocabulary width: recall of the true top-64 and exact log-probs."""
    from src.ttt.fused_ce import topk_logprobs
    k1, k2 = jax.random.split(jax.random.PRNGKey(3))
    hidden = jax.random.normal(k1, (512, 256), jnp.bfloat16)
    lm = (jax.random.normal(k2, (128256, 256), jnp.float32) * 0.05).astype(jnp.bfloat16)
    ids, lp = jax.jit(lambda h, w: topk_logprobs(h, w, 64))(hidden, lm)
    logits = jnp.dot(hidden, lm.T, preferred_element_type=jnp.float32)
    full = jax.nn.log_softmax(logits, axis=-1)
    ref_v, ref_i = jax.lax.top_k(full, 64)
    hit = jnp.mean(jnp.any(ids[:, :, None] == ref_i[:, None, :], axis=-1))
    report("topk_logprobs: recall of the exact top-64 >= 0.98", 1.0 - float(hit), 2e-2, "miss = {:.3e}")
    lp_ref = jnp.take_along_axis(full, ids, axis=-1)
    report("topk_logprobs: log-probs of the kept tokens exact", float(jnp.max(jnp.abs(lp - lp_ref))), 1e-3, "max abs = {:.3e}")


def main():
    kernel_checks(BIG, "big")
    kernel_checks(TINY, "tiny")
    prefix_checks(BIG, "big", 4)
    prefix_checks(TINY, "tiny", 8)
    mesh = make_mesh(jax.devices())
    jax.set_mesh(mesh)
    sharded_checks(BIG, mesh)
    loss_fn_checks(mesh)
    topk_checks()
    print("ALL PASS" if not failures else "FAILURES: " + "; ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
