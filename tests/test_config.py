# coding=utf-8
"""Config: JSON round-trip, unknown keys rejected, both shipped configs load, validation."""
from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ttt.config import ModelConfig, TrainConfig

ROOT = os.path.join(os.path.dirname(__file__), "..")


def main():
    failures = []
    # 1. both shipped configs load and derive sensibly
    for name, want in (("1b_e2e", (24, 18, 6, 3072, 128002)), ("tiny_smoke", (4, 3, 1, 64, 128002))):
        c = ModelConfig.load(os.path.join(ROOT, "configs", name))
        got = (c.num_hidden_layers, c.prefix_len, c.suffix_len, c.window_len, c.mask_token_id)
        ok = got == want
        failures += [] if ok else [f"{name}: {got} != {want}"]
        print(f"[{'PASS' if ok else 'FAIL'}] configs/{name:<11} L/prefix/suffix/window/mask = {got}")

    # 2. round-trip
    tmp = tempfile.mkdtemp()
    c = ModelConfig.load(os.path.join(ROOT, "configs", "1b_e2e"))
    c.save(os.path.join(tmp, "config.json"))
    ok = ModelConfig.load(tmp) == c
    t = TrainConfig(inner_mask_range=(0.3, 0.9), steps=5)
    t.save(os.path.join(tmp, "train.json"))
    ok = ok and TrainConfig.load(os.path.join(tmp, "train.json")) == t
    failures += [] if ok else ["round-trip"]
    print(f"[{'PASS' if ok else 'FAIL'}] JSON round-trip (Model + Train)")

    # 3. unknown key is an error, not a warning
    json.dump({**json.load(open(os.path.join(tmp, "config.json"))), "sliding_window_chunks": 1},
              open(os.path.join(tmp, "bad.json"), "w"))
    try:
        ModelConfig.load(os.path.join(tmp, "bad.json")); ok = False
    except ValueError as e:
        ok = "sliding_window_chunks" in str(e)
    failures += [] if ok else ["unknown key accepted"]
    print(f"[{'PASS' if ok else 'FAIL'}] unknown key rejected with its name")

    # 4. validation: mask id must be inside the vocab (the bug the old stack shipped with)
    try:
        ModelConfig(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                    num_attention_heads=4, num_key_value_heads=2, head_dim=8, suffix_len=1,
                    mask_token_id=151666, pad_token_id=0, eos_token_id=0); ok = False
    except ValueError:
        ok = True
    failures += [] if ok else ["out-of-vocab mask_token_id accepted"]
    print(f"[{'PASS' if ok else 'FAIL'}] mask_token_id >= vocab_size rejected")
    try:
        TrainConfig(resume="/some/path", init_from="x"); ok = False
    except ValueError:
        ok = True
    ok = ok and TrainConfig(resume="auto", init_from="x").init_from == "x"   # restartable stage 2
    failures += [] if ok else ["resume/init_from exclusivity wrong"]
    print(f"[{'PASS' if ok else 'FAIL'}] --resume PATH + --init_from refused; --resume auto + --init_from allowed")

    print("ALL PASS" if not failures else "FAILURES: " + "; ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
