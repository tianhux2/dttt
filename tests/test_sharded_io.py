# coding=utf-8
"""Sharded IO: file-like split writer and deserialisation reader."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import equinox as eqx
import jax.numpy as jnp

from src.ttt.sharded_io import (
    ShardedFileWriter,
    ShardedFileReader,
    get_checkpoint_files,
    checkpoint_exists,
)


def test_sharded_io_equinox(tmp_path: Path):
    base_path = str(tmp_path / "model_ckpt.eqx")
    
    # Create a dummy PyTree with ~12 MB of float32 data
    tree = {
        "weight_a": jnp.ones((1000, 1500), dtype=jnp.float32),  # 6 MB
        "weight_b": jnp.zeros((1000, 1500), dtype=jnp.float32),  # 6 MB
        "step": 42,
    }
    
    # Test 1: Write with a small shard size (5 MB limit)
    max_bytes = 5 * 1024 * 1024
    with ShardedFileWriter(base_path, max_bytes=max_bytes) as writer:
        eqx.tree_serialise_leaves(writer, tree)

    # Check generated shard files
    files = get_checkpoint_files(base_path)
    assert len(files) >= 2, f"Expected at least 2 shard files, got {files}"
    assert checkpoint_exists(base_path)
    for f in files:
        assert os.path.getsize(f) <= max_bytes + 1024

    # Test 2: Read back sharded checkpoint
    with ShardedFileReader(base_path) as reader:
        loaded_tree = eqx.tree_deserialise_leaves(reader, tree)

    assert loaded_tree["step"] == 42
    assert loaded_tree["weight_a"].shape == (1000, 1500)
    assert jnp.all(loaded_tree["weight_a"] == tree["weight_a"])
    assert jnp.all(loaded_tree["weight_b"] == tree["weight_b"])
    print("[PASS] sharded write and read round-trip")


def test_sharded_io_backward_compatibility(tmp_path: Path):
    single_path = str(tmp_path / "single_ckpt.eqx")
    tree = {"a": jnp.ones((100, 100), dtype=jnp.float32)}

    # Save as normal single file
    eqx.tree_serialise_leaves(single_path, tree)
    assert os.path.exists(single_path)
    assert checkpoint_exists(single_path)

    # Read back using ShardedFileReader
    with ShardedFileReader(single_path) as reader:
        loaded_tree = eqx.tree_deserialise_leaves(reader, tree)

    assert jnp.all(loaded_tree["a"] == tree["a"])
    print("[PASS] single-file backward compatibility")


def main():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        test_sharded_io_equinox(p)
        test_sharded_io_backward_compatibility(p)
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
