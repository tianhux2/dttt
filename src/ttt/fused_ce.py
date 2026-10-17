# coding=utf-8
"""
Blockwise fused cross-entropy.

`jax.nn.log_softmax` over a 152064-way vocabulary materialises a
`[chunk, vocab]` tensor in the forward pass and keeps another copy for the
backward. At chunk=512 and a 32-way sharded vocab that is the 4.64 GB
`bf16[16,64,512,4752]` buffer in the XLA reports.

This computes the same value and the same gradients while never holding more
than `[chunk, block]` at once: the forward streams the vocabulary in blocks
carrying a running (max, sumexp), and the custom VJP recomputes each block's
logits on the way back.

Peak drops from O(chunk x vocab) to O(chunk x block).
"""
from functools import partial

import jax
import jax.numpy as jnp


def _blocks(vocab_size: int, block_size: int):
    n = (vocab_size + block_size - 1) // block_size
    return n, n * block_size - vocab_size


@partial(jax.custom_vjp, nondiff_argnums=(4,))
def fused_weighted_ce(hidden, lm_weight, target_ids, weights, block_size=8192):
    """
    sum_t weights[t] * ( logsumexp_v(hidden[t] @ lm_weight[v]) - hidden[t] @ lm_weight[target[t]] )

    hidden      [T, H]  float
    lm_weight   [V, H]  float   (eqx.nn.Linear stores [out_features, in_features])
    target_ids  [T]     int
    weights     [T]     float   (zero entries contribute nothing, as in the outer loss)
    """
    return _fwd(hidden, lm_weight, target_ids, weights, block_size)[0]


def _logsumexp_stream(hidden, lm_weight, block_size):
    """``(logZ [T], max [T])`` of ``hidden @ lm_weight.T`` over the vocabulary, one block at a time."""
    V, _ = lm_weight.shape
    n_blocks, pad = _blocks(V, block_size)
    w_pad = jnp.pad(lm_weight, ((0, pad), (0, 0)))
    w_blocks = w_pad.reshape(n_blocks, block_size, -1)

    def body(carry, w_b):
        run_max, run_sum = carry
        # Inputs stay in their own dtype (bf16 in training): the MXU accumulates in
        # f32 either way, and casting a [block, H] weight block to f32 first only
        # writes and re-reads a 64 MB copy per block.
        logits = jnp.dot(hidden, w_b.T, preferred_element_type=jnp.float32)   # [T, block]
        blk_max = jnp.max(logits, axis=-1)
        new_max = jnp.maximum(run_max, blk_max)
        run_sum = run_sum * jnp.exp(run_max - new_max) + jnp.sum(jnp.exp(logits - new_max[:, None]), axis=-1)
        return (new_max, run_sum), None

    T = hidden.shape[0]
    init = (jnp.full((T,), -jnp.inf, jnp.float32), jnp.zeros((T,), jnp.float32))
    (fin_max, fin_sum), _ = jax.lax.scan(body, init, w_blocks)

    # Padding rows are exactly zero, so exp(0 - max) leaks into the sum; subtract it.
    if pad:
        fin_sum = fin_sum - pad * jnp.exp(-fin_max)
    return fin_max + jnp.log(fin_sum), fin_max


def _fwd(hidden, lm_weight, target_ids, weights, block_size):
    logZ, fin_max = _logsumexp_stream(hidden, lm_weight, block_size)
    tgt = jnp.einsum("th,th->t", hidden, lm_weight[target_ids], preferred_element_type=jnp.float32)
    loss = jnp.sum(weights * (logZ - tgt))
    return loss, (logZ, fin_max)


def per_token_ce(hidden, lm_weight, target_ids, block_size=8192):
    """``[N]`` cross-entropy of every row, streaming the vocabulary like the fused
    loss (forward only; evaluation probes)."""
    _, (logZ, _) = _fwd(hidden, lm_weight, target_ids, jnp.ones((hidden.shape[0],), jnp.float32), block_size)
    tgt = jnp.einsum("th,th->t", hidden, lm_weight[target_ids], preferred_element_type=jnp.float32)
    return jax.lax.stop_gradient(logZ - tgt)


def _fused_fwd(hidden, lm_weight, target_ids, weights, block_size):
    loss, (logZ, _) = _fwd(hidden, lm_weight, target_ids, weights, block_size)
    return loss, (hidden, lm_weight, target_ids, weights, logZ)


def _fused_bwd(block_size, res, g):
    hidden, lm_weight, target_ids, weights, logZ = res
    V, H = lm_weight.shape
    n_blocks, pad = _blocks(V, block_size)
    w_pad = jnp.pad(lm_weight, ((0, pad), (0, 0)))
    w_blocks = w_pad.reshape(n_blocks, block_size, -1)
    gw = (g * weights).astype(jnp.float32)                                  # [T]

    def body(carry, xs):
        g_hidden, g_w_acc = carry
        w_b, blk_idx = xs
        logits = jnp.dot(hidden, w_b.T, preferred_element_type=jnp.float32)
        p = jnp.exp(logits - logZ[:, None])                                 # softmax block
        base = blk_idx * block_size
        onehot = (target_ids[:, None] - base) == jnp.arange(block_size)[None, :]
        d = gw[:, None] * (p - onehot.astype(jnp.float32))                  # [T, block]
        # d is f32; its products with bf16 operands are f32 matmuls (default TPU
        # precision rounds d to bf16 for the MXU, as the f32 cast version did).
        g_h = jnp.dot(d, w_b, preferred_element_type=jnp.float32)
        g_wb = jnp.dot(d.T, hidden, preferred_element_type=jnp.float32)
        return (g_hidden + g_h, g_w_acc), g_wb

    # ``checkpoint``: when this backward is itself differentiated (the meta-gradient
    # through the inner step's CE), the scan's transpose would otherwise keep every
    # block's [T, block] f32 logits / softmax as residuals (16 blocks x 32 MB per
    # sequence-chunk at 1B; ~2-3 GB per chip per CE term).  Recomputing them costs one
    # more [T, H] x [H, block] matmul per block in that pass, ~1 % of a step.
    (g_hidden, _), g_w_blocks = jax.lax.scan(
        jax.checkpoint(body), (jnp.zeros(hidden.shape, jnp.float32), jnp.zeros(())),
        (w_blocks, jnp.arange(n_blocks)),
    )
    g_w = g_w_blocks.reshape(n_blocks * block_size, H)[:V]
    return (g_hidden.astype(hidden.dtype), g_w.astype(lm_weight.dtype), None, None)


fused_weighted_ce.defvjp(_fused_fwd, _fused_bwd)


def naive_weighted_ce(hidden, lm_weight, target_ids, weights):
    """Reference implementation, for tests only."""
    logits = hidden.astype(jnp.float32) @ lm_weight.astype(jnp.float32).T
    lp = jax.nn.log_softmax(logits, axis=-1)
    tgt = jnp.take_along_axis(lp, target_ids[:, None], axis=-1).squeeze(-1)
    return jnp.sum(weights * (-tgt))


# ----------------------------------------------------------------------------
# soft (sparse, multi-token) targets: distillation from a teacher's top-k
# ----------------------------------------------------------------------------
def _soft_block(target_ids, target_probs, base, block_size):
    """``[T, block]`` dense slice of the sparse target distribution: probability mass of
    the K target tokens that fall in vocabulary block ``[base, base + block)``.  Built
    by an in-bounds scatter-add (never ``mode="drop"``: negative local indices would
    wrap around to the block's end)."""
    T, K = target_ids.shape
    local = target_ids - base
    ok = (local >= 0) & (local < block_size)
    rows = jnp.broadcast_to(jnp.arange(T)[:, None], (T, K))
    return jnp.zeros((T, block_size), jnp.float32).at[rows, jnp.where(ok, local, 0)].add(
        jnp.where(ok, target_probs, 0.0).astype(jnp.float32))


@partial(jax.custom_vjp, nondiff_argnums=(5,))
def fused_soft_ce(hidden, lm_weight, target_ids, target_probs, weights, block_size=8192):
    """
    sum_t weights[t] * ( sum_k q[t, k] * logsumexp_v(hidden[t] @ lm_weight[v]) - sum_k q[t, k] * hidden[t] @ lm_weight[target[t, k]] )

    The weighted soft cross-entropy against a sparse target distribution: ``target_ids
    [T, K]`` tokens with probabilities ``target_probs [T, K]`` (rows normally sum to 1: a
    teacher's renormalised top-k).  With ``K = 1`` and probability 1 it is
    ``fused_weighted_ce``.  Streams the vocabulary like the one-hot loss; the sparse
    target enters the backward as ``sum_k q_k onehot_k`` per block.  No gradient
    reaches the targets (a teacher's output is held constant).
    """
    return _soft_fwd(hidden, lm_weight, target_ids, target_probs, weights, block_size)[0]


def _soft_fwd(hidden, lm_weight, target_ids, target_probs, weights, block_size):
    logZ, fin_max = _logsumexp_stream(hidden, lm_weight, block_size)
    q = target_probs.astype(jnp.float32)
    # sum_k q_k * hidden . W[target_k], one target column at a time ([T, H] gathers,
    # never a [T, K, H] tensor).
    def body(acc, k):
        row = jnp.einsum("th,th->t", hidden, lm_weight[target_ids[:, k]], preferred_element_type=jnp.float32)
        return acc + q[:, k] * row, None
    tgt, _ = jax.lax.scan(body, jnp.zeros((hidden.shape[0],), jnp.float32), jnp.arange(target_ids.shape[1]))
    qsum = jnp.sum(q, axis=-1)
    loss = jnp.sum(weights * (qsum * logZ - tgt))
    return loss, (logZ, fin_max)


def _soft_fused_fwd(hidden, lm_weight, target_ids, target_probs, weights, block_size):
    loss, (logZ, _) = _soft_fwd(hidden, lm_weight, target_ids, target_probs, weights, block_size)
    return loss, (hidden, lm_weight, target_ids, target_probs, weights, logZ)


def _soft_fused_bwd(block_size, res, g):
    hidden, lm_weight, target_ids, target_probs, weights, logZ = res
    V, H = lm_weight.shape
    n_blocks, pad = _blocks(V, block_size)
    w_pad = jnp.pad(lm_weight, ((0, pad), (0, 0)))
    w_blocks = w_pad.reshape(n_blocks, block_size, -1)
    gw = (g * weights).astype(jnp.float32)                                  # [T]
    qsum = jnp.sum(target_probs.astype(jnp.float32), axis=-1)               # [T]

    def body(carry, xs):
        g_hidden, g_w_acc = carry
        w_b, blk_idx = xs
        logits = jnp.dot(hidden, w_b.T, preferred_element_type=jnp.float32)
        p = jnp.exp(logits - logZ[:, None])                                 # softmax block
        soft = _soft_block(target_ids, target_probs, blk_idx * block_size, block_size)
        d = gw[:, None] * (qsum[:, None] * p - soft)                        # [T, block]
        g_h = jnp.dot(d, w_b, preferred_element_type=jnp.float32)
        g_wb = jnp.dot(d.T, hidden, preferred_element_type=jnp.float32)
        return (g_hidden + g_h, g_w_acc), g_wb

    (g_hidden, _), g_w_blocks = jax.lax.scan(                              # checkpoint: see _fused_bwd
        jax.checkpoint(body), (jnp.zeros(hidden.shape, jnp.float32), jnp.zeros(())),
        (w_blocks, jnp.arange(n_blocks)),
    )
    g_w = g_w_blocks.reshape(n_blocks * block_size, H)[:V]
    return (g_hidden.astype(hidden.dtype), g_w.astype(lm_weight.dtype), None, None, None)


fused_soft_ce.defvjp(_soft_fused_fwd, _soft_fused_bwd)


def naive_soft_ce(hidden, lm_weight, target_ids, target_probs, weights):
    """Reference implementation of ``fused_soft_ce``, for tests only."""
    logits = hidden.astype(jnp.float32) @ lm_weight.astype(jnp.float32).T
    lp = jax.nn.log_softmax(logits, axis=-1)
    tgt = jnp.take_along_axis(lp, target_ids, axis=-1)                      # [T, K]
    return jnp.sum(weights * (-jnp.sum(target_probs.astype(jnp.float32) * tgt, axis=-1)))


def topk_logprobs(hidden, lm_weight, k: int, block_size=8192, recall: float = 0.99):
    """``(ids [T, k] int32, logp [T, k] f32)``: the ``k`` most likely tokens of every row
    and their exact log-probabilities, streaming the vocabulary (a running top-k merged
    block by block next to the running logsumexp).  The per-block selection is
    ``lax.approx_max_k`` at ``recall`` (the TPU's partial reduction; exact on other
    backends), so a token of the true top-k is very occasionally replaced by the next
    one - nothing a soft target notices.  Forward only, held constant
    (``stop_gradient``): the teacher side of a distillation."""
    hidden = jax.lax.stop_gradient(hidden)
    lm_weight = jax.lax.stop_gradient(lm_weight)
    V, _ = lm_weight.shape
    n_blocks, pad = _blocks(V, block_size)
    w_pad = jnp.pad(lm_weight, ((0, pad), (0, 0)))
    w_blocks = w_pad.reshape(n_blocks, block_size, -1)
    T = hidden.shape[0]
    kk = min(k, block_size)

    def body(carry, xs):
        run_max, run_sum, top_v, top_i = carry
        w_b, blk_idx = xs
        logits = jnp.dot(hidden, w_b.T, preferred_element_type=jnp.float32)   # [T, block]
        if pad:                                                                # padding rows never win
            logits = jnp.where((blk_idx * block_size + jnp.arange(block_size))[None, :] < V, logits, -jnp.inf)
        blk_max = jnp.max(logits, axis=-1)
        new_max = jnp.maximum(run_max, blk_max)
        run_sum = run_sum * jnp.exp(run_max - new_max) + jnp.sum(jnp.exp(logits - new_max[:, None]), axis=-1)
        v_b, i_b = jax.lax.approx_max_k(logits, kk, recall_target=recall)
        v_m, order = jax.lax.top_k(jnp.concatenate([top_v, v_b], axis=1), kk)
        i_m = jnp.take_along_axis(jnp.concatenate([top_i, i_b + blk_idx * block_size], axis=1), order, axis=1)
        return (new_max, run_sum, v_m, i_m), None

    init = (jnp.full((T,), -jnp.inf, jnp.float32), jnp.zeros((T,), jnp.float32),
            jnp.full((T, kk), -jnp.inf, jnp.float32), jnp.zeros((T, kk), jnp.int32))
    (fin_max, fin_sum, top_v, top_i), _ = jax.lax.scan(body, init, (w_blocks, jnp.arange(n_blocks)))
    logZ = fin_max + jnp.log(fin_sum)
    return jax.lax.stop_gradient(top_i), jax.lax.stop_gradient(top_v - logZ[:, None])
