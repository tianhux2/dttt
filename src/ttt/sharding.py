# coding=utf-8
"""
1-D FSDP over a single ``'fsdp'`` mesh axis.

Rule for parameters: shard the LAST axis divisible by the device count, never
the stacked layer axis (``L-S = 18`` and ``S = 6`` do not divide 32).  Under pure
FSDP the choice of axis only changes the all-gather layout.  Carries and batches
are sharded on their batch axis 0 when it divides.
"""
from __future__ import annotations

from typing import Optional

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

AXIS = "fsdp"


def batch_axis_name(mesh: Optional[Mesh]):
    """Batch axis name: ('dp', 'fsdp') for 2-D meshes, 'fsdp' for 1-D meshes."""
    if mesh is None:
        return AXIS
    return ("dp", AXIS) if "dp" in mesh.axis_names else AXIS


def make_mesh(devices=None, fsdp_size: int = 0) -> Mesh:
    devices = jax.devices() if devices is None else devices
    ndev = len(devices)
    if fsdp_size > 0 and fsdp_size < ndev:
        if ndev % fsdp_size != 0:
            raise ValueError(f"device count {ndev} not divisible by fsdp_size {fsdp_size}")
        dp_size = ndev // fsdp_size
        return Mesh(np.array(devices).reshape(dp_size, fsdp_size), axis_names=("dp", AXIS))
    return Mesh(np.array(devices).reshape(ndev), axis_names=(AXIS,))


def param_spec(shape, fsdp_size: int) -> P:
    """Last divisible axis, skipping axis 0 for 3-D+ (stacked) leaves; else replicate."""
    nd = len(shape)
    first = 1 if nd >= 3 else 0
    for ax in range(nd - 1, first - 1, -1):
        if shape[ax] % fsdp_size == 0 and shape[ax] >= fsdp_size:
            return P(*([None] * ax + [AXIS] + [None] * (nd - ax - 1)))
    return P()


def param_shardings(tree, mesh: Mesh):
    fsdp_size = mesh.shape[AXIS]
    return jax.tree_util.tree_map(
        lambda x: NamedSharding(mesh, param_spec(x.shape, fsdp_size)) if hasattr(x, "shape") else None, tree)


def constrain_replicated(tree, mesh: Optional[Mesh]):
    """Pin every array leaf to be fully replicated (one all-gather at this point
    instead of one per use).  A no-op without a mesh."""
    if mesh is None:
        return tree
    return jax.tree_util.tree_map(
        lambda x: jax.lax.with_sharding_constraint(x, NamedSharding(mesh, P())) if hasattr(x, "shape") else x,
        tree)


def replicate_in_context(tree):
    """``constrain_replicated`` against the ambient ``jax.set_mesh`` context; a no-op
    without one.  For code that cannot be handed a mesh (the model), e.g. the
    per-layer weights inside a layer scan: XLA must gather the 74 MB layer, not
    turn the matmul tensor-parallel and gather the 6 GB batch activations instead
    (which it did on the pod: 6.9 s of all-gathers per step)."""
    mesh = jax.sharding.get_abstract_mesh()
    if mesh.empty:
        return tree
    return jax.tree_util.tree_map(
        lambda x: jax.lax.with_sharding_constraint(x, P()) if hasattr(x, "shape") else x, tree)


def constrain_local(tree):
    """Inside the per-sequence model under the batch ``vmap(spmd_axis_name=AXIS)``:
    pin array leaves to "unsharded here" - the vmap turns that into "sharded on the
    batch axis", so XLA cannot re-layout an activation into a global-batch,
    feature-sharded buffer (the embedding lookup did: ``[B_global, 3, T, H/ndev]``,
    1.5 GB per chip at 128K).  A no-op without a mesh context."""
    mesh = jax.sharding.get_abstract_mesh()
    if mesh.empty:
        return tree
    return jax.tree_util.tree_map(
        lambda x: jax.lax.with_sharding_constraint(x, P(*([None] * x.ndim))) if hasattr(x, "shape") else x,
        tree)


def constrain_batch_axis(tree, mesh: Optional[Mesh], axis: int = 0):
    """Pin every array leaf to be sharded on ``axis`` (the batch axis) when it divides.

    A no-op without a mesh, so the loss is mesh-agnostic and tests run unsharded.
    """
    if mesh is None:
        return tree
    b_axis = batch_axis_name(mesh)
    ndev = mesh.size

    def one(x):
        if not hasattr(x, "shape") or x.ndim <= axis or x.shape[axis] % ndev:
            return x
        spec = [None] * x.ndim
        spec[axis] = b_axis
        return jax.lax.with_sharding_constraint(x, NamedSharding(mesh, P(*spec)))

    return jax.tree_util.tree_map(one, tree)
