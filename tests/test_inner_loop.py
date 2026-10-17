# coding=utf-8
"""Inner loop: clip scale, LR ramp, and the inner gradient against an unrolled reference."""
from __future__ import annotations

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")
os.environ.setdefault("TTT_ATTENTION", "xla")   # exact reference path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import equinox as eqx
import jax
import jax.numpy as jnp

from src.ttt.config import ModelConfig
from src.ttt.inner_loop import clip_global, drift_stats, inner_loss, inner_lr, inner_step, inner_update
from src.ttt.model import Model, key_validity

CFG = ModelConfig(
    vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
    num_attention_heads=4, num_key_value_heads=2, head_dim=8, suffix_len=2,
    chunk_size=8, window_chunks=1, mask_token_id=120, pad_token_id=0, eos_token_id=0,
)
C, S, B = CFG.chunk_size, CFG.suffix_len, 3


def main():
    failures = []

    # 1. inner_lr ramp: {0.1, 0.55, 1.0, 1.0} at {0, 2500, 5000, 6000}; 0 disables.
    got = [round(float(inner_lr(jnp.int32(s), 1.0, 5000)), 4) for s in (0, 2500, 5000, 6000)]
    flat = [float(inner_lr(jnp.int32(s), 1.0, 0)) for s in (0, 2500, 9999)]
    ratio = round(float(inner_lr(jnp.int32(250), 1.0, -1, total_steps=5000, warmup_ratio=0.1)), 4)
    ok = got == [0.1, 0.55, 1.0, 1.0] and flat == [1.0, 1.0, 1.0] and ratio == 0.55
    failures += [] if ok else [f"inner_lr ramp {got} flat {flat} ratio {ratio}"]
    print(f"[{'PASS' if ok else 'FAIL'}] inner_lr ramp / 0 disables / ratio   {got}")

    # 2. clip_global: one norm over all leaves; scale = min(1, clip/norm); disabled by clip<=0.
    m = Model(CFG, jax.random.PRNGKey(0))
    g = jax.tree_util.tree_map(lambda x: jnp.ones_like(x) * 0.1, m.fast_weight)
    n_el = sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(g, eqx.is_array)))
    norm_ref = (n_el * 0.01) ** 0.5
    gc, n = clip_global(g, 1.0)
    got_norm = float(jnp.sqrt(sum(jnp.sum(x**2) for x in jax.tree_util.tree_leaves(eqx.filter(gc, eqx.is_array)))))
    gu, _ = clip_global(g, 0.0)
    ok = (abs(float(n) - norm_ref) < 1e-4 and abs(got_norm - 1.0) < 1e-5
          and float(jnp.max(jnp.abs(gu.gate.weight - g.gate.weight))) == 0.0)
    failures += [] if ok else [f"clip_global: norm {float(n):.4f} vs {norm_ref:.4f}, after {got_norm:.4f}"]
    print(f"[{'PASS' if ok else 'FAIL'}] clip_global one norm, min(1,clip/n)   {float(n):.3f} -> {got_norm:.3f}")

    # 3. THE ONE THAT MATTERS: inner_loss gradient equals jax.grad of an explicit per-layer
    #    Python forward with the same W - i.e. the scan + fused CE differentiate correctly.
    key = jax.random.PRNGKey(1)
    h_in = jax.random.normal(key, (C, CFG.hidden_size))
    tok = jax.random.randint(key, (C,), 1, 100)
    imask = jax.random.uniform(jax.random.PRNGKey(2), (C,)) < 0.6
    kv = (jax.random.normal(jax.random.PRNGKey(3), (S, C, 2, 8)) * 0.1,
          jax.random.normal(jax.random.PRNGKey(4), (S, C, 2, 8)) * 0.1)
    kvv = key_validity(jnp.int32(1), C, 1)
    pos = jnp.arange(C, 2 * C)
    g_scan = eqx.filter_grad(inner_loss)(m.fast_weight, m, h_in, tok, imask, kv, pos, kvv, 32, jnp.float32)

    def ref_loss(W):
        h = h_in
        for li in range(S):
            one = lambda t: jax.tree_util.tree_map(lambda x: x[li], t)
            blk, pn, ppn, w = one(m.suffix), one(m.prime_norm), one(m.prime_post_norm), one(W)
            h, _ = blk(h, pos, (kv[0][li], kv[1][li]), kvv, prime=(pn, ppn, w))
        lg = m.logits(m.final_norm(h))
        ce = -jax.nn.log_softmax(lg, -1)[jnp.arange(C), tok]
        return jnp.sum(ce * imask) / (jnp.sum(imask) + 1e-6)
    g_ref = eqx.filter_grad(ref_loss)(m.fast_weight)
    r = max(float(jnp.max(jnp.abs(a - b))) / (float(jnp.max(jnp.abs(b))) + 1e-12)
            for a, b in zip(jax.tree_util.tree_leaves(eqx.filter(g_scan, eqx.is_array)),
                            jax.tree_util.tree_leaves(eqx.filter(g_ref, eqx.is_array))))
    ok = r < 1e-4
    failures += [] if ok else [f"inner gradient != unrolled reference ({r:.2e})"]
    print(f"[{'PASS' if ok else 'FAIL'}] inner grad == unrolled reference     max rel = {r:.3e}")

    # 4. inner_step moves W by <= lr*clip in norm and stays f32; inner_update batches it.
    W1, st = inner_step(m.fast_weight, m, h_in, tok, imask, kv, pos, kvv, jnp.float32(1.0), 1.0, 32, jnp.float32)
    d = drift_stats(W1, m.fast_weight)
    ok = (float(d["ttt_delta_norm"]) <= 1.0 + 1e-5 and W1.gate.weight.dtype == jnp.float32
          and float(st.loss) > 0 and float(st.grad_norm) > 0)
    failures += [] if ok else [f"inner_step: delta {float(d['ttt_delta_norm']):.3f}, dtype {W1.gate.weight.dtype}"]
    print(f"[{'PASS' if ok else 'FAIL'}] inner_step |dW| <= lr*clip, f32       |dW| = {float(d['ttt_delta_norm']):.3f}")
    Wb = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (B,) + x.shape), m.fast_weight)
    bc = lambda x: jnp.broadcast_to(x, (B,) + x.shape)
    Wb2, stb = inner_update(Wb, m, bc(h_in), bc(tok), bc(imask), (bc(kv[0]), bc(kv[1])), bc(pos), kvv,
                            jnp.float32(1.0), 1.0, 32, jnp.float32)
    r = float(jnp.max(jnp.abs(Wb2.gate.weight[1] - W1.gate.weight)))
    ok = r < 1e-6 and abs(float(stb.loss) - B * float(st.loss)) < 1e-4 and Wb2.gate.weight.shape[0] == B
    failures += [] if ok else [f"inner_update batch != single ({r:.2e})"]
    print(f"[{'PASS' if ok else 'FAIL'}] inner_update == vmap(inner_step)     max abs = {r:.3e}")

    # 5. Write options.  (a) two views: the gradient is the mean of the per-view
    #    gradients; the complementary pair covers every non-pad token exactly once.
    from src.ttt.inner_loop import _global_norm, direction
    from src.ttt.objective import make_inner_mask, span_mask
    f2 = make_inner_mask(-1.0, (0.5, 1.0), CFG.mask_token_id, CFG.pad_token_id, views=2)
    x2, m2 = f2(tok, jax.random.PRNGKey(5))
    cover = jnp.sum(m2.astype(jnp.int32), axis=0)
    ok = bool(jnp.all(cover == (tok != CFG.pad_token_id).astype(jnp.int32))) and x2.shape == (2, C)
    failures += [] if ok else ["complementary views do not cover each token once"]
    print(f"[{'PASS' if ok else 'FAIL'}] views=2 complementary, full coverage")
    h2 = jnp.stack([h_in, h_in * 0.5 + 0.1])
    g2 = eqx.filter_grad(inner_loss)(m.fast_weight, m, h2, tok, m2, kv, pos, kvv, 32, jnp.float32)
    ga = eqx.filter_grad(inner_loss)(m.fast_weight, m, h2[0], tok, m2[0], kv, pos, kvv, 32, jnp.float32)
    gb = eqx.filter_grad(inner_loss)(m.fast_weight, m, h2[1], tok, m2[1], kv, pos, kvv, 32, jnp.float32)
    r = max(float(jnp.max(jnp.abs(a - 0.5 * (b + c)))) / (float(jnp.max(jnp.abs(a))) + 1e-12)
            for a, b, c in zip(*(jax.tree_util.tree_leaves(eqx.filter(t, eqx.is_array)) for t in (g2, ga, gb))))
    ok = r < 1e-5
    failures += [] if ok else [f"two-view gradient != mean of views ({r:.2e})"]
    print(f"[{'PASS' if ok else 'FAIL'}] views=2 grad == mean of view grads    max rel = {r:.3e}")
    #    (b) views=1 is the old function bit for bit.
    f1 = make_inner_mask(-1.0, (0.5, 1.0), CFG.mask_token_id, CFG.pad_token_id)
    x1, m1 = f1(tok, jax.random.PRNGKey(5))
    ok = bool(jnp.all(m1 == m2[0])) and bool(jnp.all(x1 == x2[0]))
    failures += [] if ok else ["views=1 mask changed"]
    print(f"[{'PASS' if ok else 'FAIL'}] views=1 mask unchanged")
    #    (c) span masks: contiguous runs, coverage near the ratio, pad never masked.
    big = jnp.arange(4096) % 97 + 1
    sm = span_mask(jax.random.PRNGKey(6), 4096, jnp.float32(0.4), 16.0)
    runs = int(jnp.sum(sm[1:] & ~sm[:-1])) + int(sm[0])
    cov = float(jnp.mean(sm))
    ok = 0.25 < cov < 0.5 and runs < 0.5 * cov * 4096 / 4
    failures += [] if ok else [f"span mask: coverage {cov:.2f}, runs {runs}"]
    print(f"[{'PASS' if ok else 'FAIL'}] span mask (ratio .4, span 16)         coverage {cov:.2f}, {runs} runs, mean len {cov*4096/max(runs,1):.1f}")
    #    (d) sign / muon2 directions keep the gradient's global norm (so the clip
    #    acts alike) and muon2 is closer to orthogonal than the gradient.
    for rule in ("sign", "muon2"):
        u = direction(g_scan, rule)
        rn = float(_global_norm(u) / _global_norm(g_scan))
        ok = abs(rn - 1.0) < 1e-3
        if rule == "muon2":
            def orth(x):
                x = x if x.shape[0] <= x.shape[1] else x.T
                sv = jnp.linalg.svd(x, compute_uv=False)
                return float(sv.min() / sv.max())
            c_g, c_u = orth(g_scan.gate.weight), orth(u.gate.weight)
            ok = ok and c_u > c_g
            print(f"       muon2: singular-value spread min/max  grad {c_g:.3f} -> orthogonalised {c_u:.3f}")
        failures += [] if ok else [f"direction {rule}: norm ratio {rn:.4f}"]
        print(f"[{'PASS' if ok else 'FAIL'}] direction {rule:<5} keeps the global norm  ratio {rn:.4f}")
    #    (e) gates at init (zero weight/bias) reproduce the plain step bit for bit; both
    #    gates receive gradient; decay pulls toward W0 on a second step.
    from src.ttt.config import ModelConfig as MC
    import dataclasses
    h_clean = jax.random.normal(jax.random.PRNGKey(7), (C, CFG.hidden_size))
    mg = Model(dataclasses.replace(CFG, inner_lr_gate=True, inner_token_gate=True, inner_decay=0.05), jax.random.PRNGKey(0))
    Wg, _ = inner_step(mg.fast_weight, mg, h_in, tok, imask, kv, pos, kvv, jnp.float32(1.0), 1.0, 32, jnp.float32,
                       h_clean=h_clean)
    r = float(jnp.max(jnp.abs(Wg.gate.weight - W1.gate.weight)))
    m_nog = Model(dataclasses.replace(CFG, inner_lr_gate=False, inner_decay=0.0), jax.random.PRNGKey(0))
    W2_plain, _ = inner_step(W1, m_nog, h_in, tok, imask, kv, pos, kvv, jnp.float32(1.0), 1.0, 32, jnp.float32)
    W2_decay, _ = inner_step(W1, mg, h_in, tok, imask, kv, pos, kvv, jnp.float32(1.0), 1.0, 32, jnp.float32,
                             h_clean=h_clean)
    expect = W2_plain.gate.weight - 0.05 * (W1.gate.weight - m.fast_weight.gate.weight)
    r2 = float(jnp.max(jnp.abs(W2_decay.gate.weight - expect)))
    # gradient of a downstream loss w.r.t. the gate parameters must be non-zero
    def down(model_):
        Wn, _ = inner_step(model_.fast_weight, model_, h_in, tok, imask, kv, pos, kvv, jnp.float32(1.0), 1.0, 32,
                           jnp.float32, h_clean=h_clean)
        return jnp.sum(jnp.square(Wn.gate.weight))
    gm = eqx.filter_grad(down)(mg)
    g_chunk = float(jnp.abs(gm.inner_gate.weight).max()) + float(jnp.abs(gm.inner_gate.bias).max())
    g_tok = float(jnp.abs(gm.inner_token_gate.weight).max())
    ok = r == 0.0 and r2 < 1e-5 and g_chunk > 0 and g_tok > 0
    failures += [] if ok else [f"gates: first-step diff {r:.2e}, decay diff {r2:.2e}, dgate {g_chunk:.1e} dtok {g_tok:.1e}"]
    print(f"[{'PASS' if ok else 'FAIL'}] gates at init == plain (bitwise); decay -> W0; gates get gradient  {r:.1e} / {r2:.1e} / {g_chunk:.1e} / {g_tok:.1e}")
    # a model with gates refuses to step without h_clean
    try:
        inner_step(mg.fast_weight, mg, h_in, tok, imask, kv, pos, kvv, jnp.float32(1.0), 1.0, 32, jnp.float32)
        ok = False
    except ValueError:
        ok = True
    failures += [] if ok else ["gated model accepted a step without h_clean"]
    print(f"[{'PASS' if ok else 'FAIL'}] gated model requires h_clean")

    #    (f) learned inner optimiser at the identity == plain step; its parameters get gradient.
    mo = Model(dataclasses.replace(CFG, inner_optimizer=True), jax.random.PRNGKey(0))
    Wo, _ = inner_step(mo.fast_weight, mo, h_in, tok, imask, kv, pos, kvv, jnp.float32(1.0), 1.0, 32, jnp.float32)
    r = float(jnp.max(jnp.abs(Wo.gate.weight - W1.gate.weight)))
    def down_o(model_):
        Wn, _ = inner_step(model_.fast_weight, model_, h_in, tok, imask, kv, pos, kvv, jnp.float32(1.0), 1.0, 32, jnp.float32)
        return jnp.sum(jnp.square(Wn.gate.weight)) + jnp.sum(jnp.square(Wn.down.weight))
    go = eqx.filter_grad(down_o)(mo)
    gp = float(jnp.abs(go.inner_precond[0].gate.weight).max()) + float(jnp.abs(go.inner_precond[1].down.weight).max())
    ok = r == 0.0 and gp > 0 and float(jnp.abs(go.inner_step_log).max()) > 0 and float(jnp.abs(go.inner_temp_log)) > 0
    failures += [] if ok else [f"inner optimiser: diff {r:.2e}, grads {gp:.1e}"]
    print(f"[{'PASS' if ok else 'FAIL'}] learned inner optimiser: identity == plain, gets gradient  {r:.1e}")

    print("ALL PASS" if not failures else "FAILURES: " + "; ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
