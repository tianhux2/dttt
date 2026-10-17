# coding=utf-8
"""
Gated DeltaNet memory as a drop-in replacement for the fast weight (the linear-attention
hybrid baseline).

One ``GDNMemory`` per suffix layer holds a per-head state ``S [nh, dv, dk]`` that plays the
role of ``W``: it is carried across chunks, WRITTEN from the clean stream with the gated
delta rule over the chunk's tokens (causal inside the chunk)

    S_t = alpha_t * S_{t-1} + beta_t * (v_t - alpha_t * S_{t-1} k_t) k_t^T          (Yang et al. 2024)

and READ by every stream of the NEXT chunk as ``o = S q`` (gated, normalised, projected),
added to the residual exactly where ``post_norm(W(norm(h)))`` sits.  Inside a chunk the
streams rely on the bidirectional attention, as with the fast weight; the state is a
constant for the chunk.  This is the paper's Gated DeltaNet baseline transplanted into the
same reader / window / objective, so the only variable against the TTT model is the kind
of cross-chunk memory: a 160M-parameter MLP written by one gradient step, or an
``nh x dv x dk`` linear state written by the delta rule.

The token recurrence is a rematerialised scan (``remat_scan.nested_remat_scan``): a chunk
of 1024 tokens keeps 16 + 64 states live instead of 1024.  q and k are SiLU'd and
L2-normalised (as in the paper); decay ``alpha = exp(-softplus(dt) * exp(A_log))`` initialised
LONG (per-token decay 0.96-0.9999: the state is read only by later chunks); ``beta = sigmoid``.
"""
from __future__ import annotations

from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from .remat_scan import balanced_factors, nested_remat_scan


class Linear(eqx.Module):
    """``[in, out]`` weight, ``x @ w (+ b)``; ``normal(std)`` init, zero bias (model.Linear's twin,
    repeated here to keep this module free of the model import)."""
    weight: Array
    bias: Array | None

    def __init__(self, in_dim: int, out_dim: int, std: float, key: Array, bias: bool = False):
        self.weight = jax.random.normal(key, (in_dim, out_dim), jnp.float32) * std
        self.bias = jnp.zeros((out_dim,), jnp.float32) if bias else None

    def __call__(self, x: Array) -> Array:
        y = x @ self.weight.astype(x.dtype)
        return y if self.bias is None else y + self.bias.astype(x.dtype)


def _l2norm(x: Array, eps: float = 1e-6) -> Array:
    x32 = x.astype(jnp.float32)
    return (x32 * jax.lax.rsqrt(jnp.sum(jnp.square(x32), axis=-1, keepdims=True) + eps)).astype(x.dtype)


class GDNMemory(eqx.Module):
    q: Linear            # H -> nh*dk
    k: Linear            # H -> nh*dk
    v: Linear            # H -> nh*dv
    beta: Linear         # H -> nh     write strength (sigmoid)
    dt: Linear           # H -> nh     decay rate (softplus(dt + dt_bias) * exp(A_log))
    dt_bias: Array       # [nh]
    A_log: Array         # [nh]
    gate: Linear         # H -> nh*dv  output gate (silu)
    o_norm_w: Array      # [dv]        per-head RMSNorm weight on the read
    o: Linear            # nh*dv -> H
    nh: int = eqx.field(static=True)
    dk: int = eqx.field(static=True)
    dv: int = eqx.field(static=True)
    eps: float = eqx.field(static=True)

    def __init__(self, hidden: int, nh: int, dk: int, dv: int, std: float, key: Array, eps: float = 1e-6):
        k1, k2, k3, k4, k5, k6, k7, k8 = jax.random.split(key, 8)
        self.q = Linear(hidden, nh * dk, std, k1)
        self.k = Linear(hidden, nh * dk, std, k2)
        self.v = Linear(hidden, nh * dv, std, k3)
        self.beta = Linear(hidden, nh, std, k4, bias=True)
        self.dt = Linear(hidden, nh, std, k5)
        # Decay initialisation.  Mamba-2 / GDN use softplus(dt_bias) ~ logU(1e-3, 1e-1) and
        # A ~ U(1, 16), i.e. per-token decays down to e^-1.6: memory spans of a few tokens,
        # which the within-chunk attention already covers here.  The state is only ever READ
        # by later chunks, so it starts long: softplus(dt_bias) ~ logU(1e-4, 1e-2), A ~ U(1, 4)
        # -> per-token decay in [0.96, 0.9999] (retention over 10^2-10^4 tokens); the
        # input-dependent dt can shorten it.
        dt0 = jnp.exp(jax.random.uniform(k6, (nh,), minval=jnp.log(1e-4), maxval=jnp.log(1e-2)))
        self.dt_bias = dt0 + jnp.log(-jnp.expm1(-dt0))                     # inverse softplus
        self.A_log = jnp.log(jax.random.uniform(k7, (nh,), minval=1.0, maxval=4.0))
        self.gate = Linear(hidden, nh * dv, std, k8)
        self.o_norm_w = jnp.ones((dv,), jnp.float32)
        self.o = Linear(nh * dv, hidden, std, jax.random.fold_in(key, 9))
        self.nh, self.dk, self.dv, self.eps = nh, dk, dv, eps

    def state0(self, dtype=jnp.float32) -> Array:
        return jnp.zeros((self.nh, self.dv, self.dk), dtype)

    def read(self, x: Array, S: Array) -> Array:
        """``x [C, H]`` (normed residual), ``S [nh, dv, dk]`` -> ``[C, H]`` memory output."""
        C = x.shape[0]
        q = _l2norm(jax.nn.silu(self.q(x)).reshape(C, self.nh, self.dk))
        o = jnp.einsum("chk,hvk->chv", q.astype(jnp.float32), S.astype(jnp.float32))          # [C, nh, dv]
        o = o * jax.lax.rsqrt(jnp.mean(jnp.square(o), axis=-1, keepdims=True) + self.eps) * self.o_norm_w
        o = o * jax.nn.silu(self.gate(x).reshape(C, self.nh, self.dv).astype(jnp.float32))
        return self.o(o.reshape(C, self.nh * self.dv).astype(x.dtype))

    def update(self, x: Array, S: Array, remat_levels: int = 2) -> Array:
        """``x [C, H]`` (the CLEAN chunk's normed residual), ``S [nh, dv, dk]`` -> the state after
        the chunk's tokens, f32.  Gated delta rule, causal over the tokens."""
        C = x.shape[0]
        k = _l2norm(jax.nn.silu(self.k(x)).reshape(C, self.nh, self.dk)).astype(jnp.float32)
        v = jax.nn.silu(self.v(x)).reshape(C, self.nh, self.dv).astype(jnp.float32)
        beta = jax.nn.sigmoid(self.beta(x).astype(jnp.float32))                                # [C, nh]
        alpha = jnp.exp(-jax.nn.softplus(self.dt(x).astype(jnp.float32) + self.dt_bias) * jnp.exp(self.A_log))

        def step(S, t):
            k_t, v_t, b_t, a_t = t                                                              # [nh, dk], [nh, dv], [nh], [nh]
            Sd = a_t[:, None, None] * S
            err = v_t - jnp.einsum("hvk,hk->hv", Sd, k_t)                                       # v - S k
            return Sd + b_t[:, None, None] * jnp.einsum("hv,hk->hvk", err, k_t), ()

        factors = balanced_factors(C, max(1, remat_levels))
        S_next, _ = nested_remat_scan(step, S.astype(jnp.float32), (k, v, beta, alpha), factors)
        return S_next

    def update_reference(self, x: Array, S: Array) -> Array:
        """Plain Python loop over the tokens (tests)."""
        C = x.shape[0]
        k = _l2norm(jax.nn.silu(self.k(x)).reshape(C, self.nh, self.dk)).astype(jnp.float32)
        v = jax.nn.silu(self.v(x)).reshape(C, self.nh, self.dv).astype(jnp.float32)
        beta = jax.nn.sigmoid(self.beta(x).astype(jnp.float32))
        alpha = jnp.exp(-jax.nn.softplus(self.dt(x).astype(jnp.float32) + self.dt_bias) * jnp.exp(self.A_log))
        S = S.astype(jnp.float32)
        for t in range(C):
            Sd = alpha[t][:, None, None] * S
            S = Sd + beta[t][:, None, None] * jnp.einsum("hv,hk->hvk", v[t] - jnp.einsum("hvk,hk->hv", Sd, k[t]), k[t])
        return S
