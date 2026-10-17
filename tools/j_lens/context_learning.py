#!/usr/bin/env python3
# coding=utf-8
"""
Learning from context beyond the window: does the memory carry what a distant part of the document TEACHES -
a mapping applied to new sentences, a function applied to new inputs - or only what it shows verbatim?

Synthetic material in natural held-out sequences.  A teaching block ends with chunk 3 ("beyond": every target
chunk lies past its window) or right before the target chunk ("within": inside the window, the ceiling); "none"
has no teaching block.  The test block sits in the target chunk, the same tokens in all three, its scored tokens
masked (nothing else of the chunk is).  Tasks (per sequence a fresh draw: the weights cannot know the answers):

    span       retrieval: a 64-token span of another document taught once, read again (cue 16, then every other
               token scored: the planted probe's "alt" mode)
    lex1_same  a fictional lexicon of 10 English nouns -> nonce words, taught in parallel sentences
               ("English: The river was cold." / "Veltan: The zorvik was cold."), one sentence per noun; test: the
               same sentences again, the nonce word scored (retrieval of a sentence seen once)
    lex1_new   ... test: a new sentence with the same noun (the mapping in a new context, one exposure)
    lex8_same  eight different sentences per noun; test: one of them again
    lex8_new   eight sentences per noun; test: a new sentence (the mapping learnt from repetition)
    cipher     a letter substitution shown on 40 words ("horse: h = q, o = k, ..."); test: 8 new words whose
               letters all appeared at least twice, the cipher letters scored (a function of the letters)
    key1       a passkey: "The special magic number for the lantern is 482913." taught once; test: the sentence's
               cue again, the six digits (two tokens) scored (one-off retrieval of a short answer)
    key4       four such facts with different keys taught together; test: one of them (retrieval among distractors)

Every sequence is written chunk by chunk with the checkpoint's own write (TTT) or update (gdn): the natural pass
once, then per task the teaching pass from its first chunk and the in-window variant from the target's window;
the target chunk is read before it would be written, with the memory on and reset (the memory-off read keeps the
windows, whose keys and values were computed with the memory on).  Per task and target, mean CE per scored token:

    recall   = (off - on | beyond) - (off - on | none)     the memory's gain from the teaching
    ceiling  = (off | beyond) - (off | within)             what attention gets from seeing the teaching
    share    = recall / ceiling

``--read l2r``: the read of generation (left-to-right decoding with the true tokens before): one read per scored
token, everything before it clean and it and everything after it in the chunk masked.  The CE numbers above are then
those of this read, and per answer (a run of scored tokens: a nonce word, a passkey) the exact-match rates - all of
its tokens the read's top prediction - under the memory (beyond, on), the memory reset (beyond, off), attention with
the teaching in the window (within, off) and the prior (none).  Greedy left-to-right decoding gets an answer exactly
right with this probability: its prefix is the true one until its first error.

    JAX_PLATFORMS=cuda python -m tools.j_lens.context_learning --ckpt <step dir> --sources <name=dir:1.0,...> \\
        --seq_len 32768 --n_per_source 12 --targets 8,20 [--tasks key1,key4,lex8_new --read l2r] --out <npz>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np

from tools.j_lens import _cli

TASKS = ("span", "lex1_same", "lex1_new", "lex8_same", "lex8_new", "cipher", "key1", "key4")
L2R_MAX = 64                        # scored tokens per test block under the l2r read (lex8_new: 10 nonce words)
PLACEMENTS = ("beyond", "within", "none")
TEACH_END = 4                       # the beyond teaching block ends with chunk 3; chunk 0 stays natural
SIZES = dict(lex_pairs=10, lex_reps=(1, 8), cipher_teach=40, cipher_test=8, span_len=64, span_cue=16)

# single tokens with a leading space (Llama 3); the fillers share no word with the nouns
NOUNS = ("river horse table window garden bottle candle basket mirror ladder pillow kettle hammer jacket bucket carpet "
         "blanket wagon tower bridge castle forest valley harbor temple market village kitchen cellar barrel lantern "
         "rope needle pencil wallet button saddle compass helmet shield sword crown ribbon feather shell rabbit turtle "
         "tiger wolf goat sheep camel pigeon spider beetle mouse salmon lemon pepper cherry carrot potato cabbage "
         "butter cheese honey coffee spoon plate bowl knife fork towel soap brush comb mattress curtain chimney fence "
         "gate road tunnel canal pond lake desert canyon cliff cave hill mountain glacier volcano box chair clock drum "
         "flag lamp map ring rock ship tent tree truck violin whistle").split()
NAMES = "Anna Ben Clara David Elena Frank Grace Henry Iris Jack Laura Mark Nora Oscar Paul Rosa Sam Tina Victor Zoe".split()
ADJS = "old small large red blue green heavy quiet strange new broken clean dark bright wooden tiny famous empty warm cold".split()
PLACES = "station library school square hall barn park shop office church beach museum hospital airport stadium".split()
TEMPLATES = (
    "{name} found the {N} near the {place}.", "The {N} was {adj} and nobody wanted it.",
    "We left the {N} at the {place} yesterday.", "{name} said the {N} looked {adj} in the morning.",
    "Nobody knew why the {N} was in the {place}.", "After lunch {name} carried the {N} to the {place}.",
    "There was one {adj} {N} behind the {place}.", "I asked {name} where the {N} had gone.",
    "The {N} in the {place} was {adj}.", "{name} bought the {adj} {N} for a friend.",
    "Everyone at the {place} talked about the {N}.", "{name} painted the {N} before the rain started.",
    "The children played with the {N} all afternoon.", "{name} put the {N} on the shelf in the {place}.",
    "Without the {N}, the {place} felt {adj}.", "The {adj} {N} belonged to {name}.",
    "{name} could not lift the {N} alone.", "Someone had moved the {N} from the {place}.",
    "The {N} was the first thing {name} noticed.", "{name} cleaned the {N} with great care.",
    "They waited for the {N} outside the {place}.", "The old man repaired the {N} for {name}.",
    "{name} drew a picture of the {N}.", "The {N} near the {place} was very {adj}.")
CIPHER_WORDS = ("apple bread chair dance eagle flame grape house juice knife money night ocean paper queen snake uncle "
                "voice water youth zebra above after again alarm album alive alone angry ankle arena badge baker basic "
                "beach begin bench black blade blank blind block blood board brain brave brick bride brief broad brown "
                "cabin cable candy cargo catch chain chalk charm chart chase cheap check chess chest chief child cloud "
                "coach coast color couch count court cover craft crane cream crowd curve daily delay dirty dozen draft "
                "drama dream dress drink drive early earth eight elbow empty enjoy equal event exact faith fancy fever "
                "field fifty final flash fleet float flock floor flour focus frame fresh front frost fruit funny ghost "
                "giant glass globe glove grace grain grand grass grave green guard guess guest guide habit happy heart "
                "heavy hotel human ideal image index jewel joint judge label large laser laugh layer lemon level light "
                "linen liver lobby local lucky lunch magic major march match medal metal minor model month motor mouse "
                "mouth movie music noble noise north novel nurse olive opera orbit order other owner paint panel party "
                "patch peace pearl penny phone photo piano piece pilot pitch pizza plain plane plant plaza point porch "
                "power price pride print prize proud queen quick quiet quilt radio ranch rapid ready relax rider ridge "
                "right river robin robot rough round royal ruler salad sauce scale scarf scene score scout shade shape "
                "shark sharp sheep sheet shelf shirt shock shore short sight skill skirt sleep slice slope small smile "
                "smoke snack solid sound south space spark speed spice spoon sport stack staff stage stamp steam steel "
                "stick stone storm story stove straw sugar sweet swing sword table taste teeth thick thief thumb tiger "
                "toast tooth torch total touch tower track trade trail train treat trick truck trunk truth tulip twist "
                "uncle union upper urban usual value video visit vivid wagon waist watch whale wheat wheel white world "
                "worth wrist yacht young zebra").split()
CONS, VOWELS = "bdfgklmnprstvz", "aeiou"


class Tok:
    """Text -> token ids: the Llama 3 tokenizer, or - for a model whose vocabulary is smaller (tests) - one id per
    space-led word, hashed into the vocabulary (ids 1 .. vocab - 3: never EOS 0 or a [MASK] at the top)."""

    def __init__(self, path: str, vocab: int):
        self.real = False
        if os.path.exists(path):
            from tokenizers import Tokenizer
            t = Tokenizer.from_file(path)
            if t.get_vocab_size() <= vocab:
                self.t, self.real = t, True
        self.vocab = vocab

    def encode(self, s: str):
        if self.real:
            return self.t.encode(s, add_special_tokens=False).ids
        words, cur = [], ""
        for ch in s:
            if ch == " " and cur.strip():
                words.append(cur)
                cur = ""
            cur += ch
        words += [cur] if cur else []
        return [1 + zlib.crc32(w.encode()) % (self.vocab - 3) for w in words]


def render(tok: Tok, segments):
    """``[(text, scored)]`` -> ``(ids, scored)``.  Adjacent segments of the same kind are joined first; a scored
    run starts with a space-led word and is followed by a space or punctuation, where the pre-tokenizer splits
    anyway, so the ids equal those of the whole text."""
    runs = []
    for text, scored in segments:
        if runs and runs[-1][1] == bool(scored):
            runs[-1][0] += text
        else:
            runs.append([text, bool(scored)])
    ids, sc = [], []
    for text, scored in runs:
        e = tok.encode(text)
        ids += e
        sc += [scored] * len(e)
    return np.asarray(ids, np.int32), np.asarray(sc, bool)


def nonce_words(rng, k: int, tok: Tok):
    out = []
    while len(out) < k:
        w = "".join(rng.choice(list(CONS)) + rng.choice(list(VOWELS)) for _ in range(int(rng.integers(2, 4))))
        w += rng.choice(list(CONS)) if rng.random() < 0.5 else ""
        if w in NOUNS or w in out or (tok.real and not 2 <= len(tok.encode(" " + w)) <= 4):
            continue
        out.append(w)
    return out


def _sentence(rng, template: int):
    t = TEMPLATES[template]
    f = dict(name=str(rng.choice(NAMES)), adj=str(rng.choice(ADJS)), place=str(rng.choice(PLACES)))
    pre, post = t.split("{N}")
    return pre.format(**f), post.format(**f)                          # pre ends with the space before the noun


def _pair_lines(pre, post, noun, nonce, scored: bool):
    """The English line and its Veltan twin; the nonce word scored in the test."""
    return [(f"English: {pre}{noun}{post}\nVeltan: {pre[:-1]}", False), (f" {nonce}", scored), (f"{post}\n", False)]


def lexicon_task(rng, tok: Tok, reps: int, pairs: int):
    """The teaching block and both tests: ``same`` (one taught sentence per noun again) and ``new`` (a sentence
    from a template the noun was not taught with)."""
    nouns = [str(x) for x in rng.choice(NOUNS, pairs, replace=False)]
    nonces = nonce_words(rng, pairs, tok)
    teach, same, new = [], [], []
    for noun, nonce in zip(nouns, nonces):
        tpl = rng.choice(len(TEMPLATES), reps + 1, replace=False)
        sents = [_sentence(rng, int(t)) for t in tpl[:reps]]
        teach += [(noun, nonce, s) for s in sents]
        same.append((noun, nonce, sents[int(rng.integers(reps))]))
        new.append((noun, nonce, _sentence(rng, int(tpl[reps]))))
    teach = [teach[i] for i in rng.permutation(len(teach))]
    order = rng.permutation(pairs)
    block = lambda items, sc: render(tok, [("\n\n", False)] + [x for n, z, (a, b) in items
                                                               for x in _pair_lines(a, b, n, z, sc)] + [("\n", False)])
    return block(teach, False)[0], {"same": block([same[i] for i in order], True),
                                    "new": block([new[i] for i in order], True)}


def cipher_task(rng, tok: Tok, n_teach: int, n_test: int):
    letters = "abcdefghijklmnopqrstuvwxyz"
    while True:                                                       # a derangement of the alphabet
        perm = rng.permutation(26)
        if not (perm == np.arange(26)).any():
            break
    code = {a: letters[p] for a, p in zip(letters, perm)}
    words = sorted(set(CIPHER_WORDS))
    order = rng.permutation(len(words))
    teach = [words[i] for i in order[:n_teach]]
    seen = {a: sum(w.count(a) for w in teach) for a in letters}
    test = [words[i] for i in order[n_teach:] if all(seen[a] >= 2 for a in words[i])][:n_test]
    if len(test) < n_test:
        return None                                                   # the caller draws again

    def line(w, scored):
        seg = [(f"{w}:", False)]
        for i, a in enumerate(w):
            seg += [(("," if i else "") + f" {a} =", False), (f" {code[a]}", scored)]
        return seg + [("\n", False)]
    tb = [("\n\n", False)] + [x for w in teach for x in line(w, False)] + [("\n", False)]
    xb = [("\n\n", False)] + [x for w in test for x in line(w, True)] + [("\n", False)]
    return render(tok, tb)[0], render(tok, xb)


def key_task(rng, tok: Tok, n_facts: int):
    """Passkeys: ``n_facts`` facts "The special magic number for the <noun> is <6 digits>." taught together; the
    test repeats one fact's cue, the digits scored (two tokens with the Llama 3 tokenizer)."""
    keys = [str(x) for x in rng.choice(NOUNS, n_facts, replace=False)]
    vals = [int(v) for v in rng.choice(900000, n_facts, replace=False) + 100000]
    teach = [("\n\n", False)] + [(f"The special magic number for the {k} is {v}.\n", False) for k, v in zip(keys, vals)]
    q = int(rng.integers(n_facts))
    test = [(f"\n\nThe special magic number for the {keys[q]} is ", False), (f"{vals[q]}", True), (".\n", False)]
    return render(tok, teach + [("\n", False)])[0], render(tok, test)


def span_task(rng, foreign: np.ndarray, length: int, cue: int, eos: int):
    for _ in range(16):
        a = int(rng.integers(0, len(foreign) - length + 1))
        span = foreign[a:a + length]
        if not (span == eos).any():
            break
    sc = np.zeros(length, bool)
    sc[np.arange(cue, length)[1::2]] = True
    return np.asarray(span, np.int32), (np.asarray(span, np.int32), sc)


FAMILIES = {"span": ("span",), "lex1": ("lex1_same", "lex1_new"), "lex8": ("lex8_same", "lex8_new"),
            "cipher": ("cipher",), "key1": ("key1",), "key4": ("key4",)}   # appended: the earlier draws stay


def build(family: str, rng, tok: Tok, foreign: np.ndarray, eos: int, sizes: dict):
    """``(teach ids, {task: (test ids, test scored)})`` of one task family for one sequence: the tasks of a family
    share the teaching block (the lexicon's same / new tests are paired)."""
    if family == "span":
        teach, test = span_task(rng, foreign, sizes["span_len"], sizes["span_cue"], eos)
        return teach, {"span": test}
    if family in ("lex1", "lex8"):
        teach, tests = lexicon_task(rng, tok, sizes["lex_reps"][family == "lex8"], sizes["lex_pairs"])
        return teach, {f"{family}_{k}": v for k, v in tests.items()}
    if family == "cipher":
        for _ in range(50):
            r = cipher_task(rng, tok, sizes["cipher_teach"], sizes["cipher_test"])
            if r is not None:
                return r[0], {"cipher": r[1]}
        raise RuntimeError("cipher: too few test words with every letter taught twice")
    if family in ("key1", "key4"):
        teach, test = key_task(rng, tok, int(family[3:]))
        return teach, {family: test}
    raise ValueError(family)


def written(seq: np.ndarray, teach: np.ndarray, C: int, Wn: int, t: int):
    """The sequences the memory is written from (no test block: a target chunk is read before it would be
    written, and the pass writes its natural text): ``{placement: ids}`` - the teaching block ending with chunk
    TEACH_END - 1 (beyond) or with chunk t - 1 (within, inside the window of target t)."""
    Lt = len(teach)
    if Lt > min((TEACH_END - 1) * C, Wn * C) or t < TEACH_END + Wn:
        raise ValueError(f"layout: teaching {Lt} tokens, target {t} (chunk {C}, window {Wn})")
    out = {p: np.array(seq, np.int32, copy=True) for p in PLACEMENTS}
    out["beyond"][TEACH_END * C - Lt:TEACH_END * C] = teach
    out["within"][t * C - Lt:t * C] = teach
    return out


def target_chunk(seq: np.ndarray, test: np.ndarray, test_sc: np.ndarray, off: int, C: int, t: int):
    """The read of target t: chunk t of the natural text with the test block at ``off`` - ``(ids [C], scored [C])``
    (the window holds the teaching block or not; the target chunk itself is the same in every placement)."""
    if not 1 <= off <= C - len(test):
        raise ValueError(f"test block of {len(test)} tokens at offset {off} (chunk {C})")
    x, s = np.array(seq[t * C:(t + 1) * C], np.int32, copy=True), np.zeros(C, bool)
    x[off:off + len(test)], s[off:off + len(test)] = test, test_sc
    return x, s


def run(model, seqs, ws, targets, seed: int, compute_dtype, tok: Tok, tasks=TASKS, sizes=None, log=print,
        read_mode: str = "masked"):
    """``seqs``: ``[(source, tokens [T])]``.  Returns ``{"ce": [n, tasks, targets, placements, (on, off)]`` CE sums
    over the scored tokens, ``"count": [n, tasks]``}``; with ``read_mode`` "l2r" also, per scored token,
    ``"l2r_ce"`` / ``"l2r_ok"`` ``[n, tasks, targets, placements, (on, off), L2R_MAX]`` (CE and top-1 correct, NaN /
    -1 padded; "within, on" is not read) and ``"l2r_item" [n, tasks, L2R_MAX]`` (the answer each token belongs to)."""
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    from tools.j_lens import core
    sizes = dict(SIZES, **(sizes or {}))
    cfg = model.cfg
    C, Wn, mask_id, eos = cfg.chunk_size, cfg.window_chunks, cfg.mask_token_id, cfg.eos_token_id
    targets = sorted(int(t) for t in targets)
    T = len(seqs[0][1])
    if targets[0] < TEACH_END + Wn or targets[-1] >= T // C:
        raise ValueError(f"targets {targets}: from {TEACH_END + Wn} (window {Wn}) to {T // C - 1}")
    if read_mode not in ("masked", "l2r") or (read_mode == "l2r" and cfg.read_block):
        raise ValueError(f"read_mode {read_mode!r}: masked | l2r (l2r: not for a read_block model)")
    f32 = jnp.float32
    key0 = jax.random.PRNGKey(0)                                      # the causal write draws nothing
    commit = eqx.filter_jit(lambda m, ids, pos, st: core.commit(m, ids, pos, st, key0, ws, compute_dtype))

    def _tok(m, x, pos, st, slot, tgt):
        """One l2r read: CE of the target at ``slot`` and whether it is the top prediction (never the mask token)."""
        h = core.read_forward(m, x, pos, st, None, ws.outer_shift, compute_dtype).h_L[slot]
        lg = m.logits(m.final_norm(h[None])).astype(f32)[0]
        return -jax.nn.log_softmax(lg)[tgt], jnp.argmax(lg.at[mask_id].set(-jnp.inf)) == tgt
    tok_reads = eqx.filter_jit(lambda m, xs, pos, st, slots, tgts: jax.lax.map(
        lambda a: _tok(m, a[0], pos, st, a[1], a[2]), (xs, slots, tgts), batch_size=8))

    def _ce(m, ids, pos, st, tgt):
        x, clean = ids                                                # the masked chunk and its clean tokens
        own = core.clean_own(m, clean, pos, st, compute_dtype) if m.cfg.read_block else None
        lg = core.read_forward(m, x, pos, st, None, ws.outer_shift, compute_dtype, own_kv=own).logits.astype(f32)
        return -jnp.take_along_axis(jax.nn.log_softmax(lg, axis=-1), tgt[:, None], axis=-1)[:, 0]
    ce = eqx.filter_jit(_ce)
    reset = lambda s: core.reset_memory(model, s)
    chunk = lambda ids, c: jnp.asarray(ids[c * C:(c + 1) * C])
    pos_of = lambda c: jnp.arange(c * C, (c + 1) * C)

    def read(x, s, t, st):
        """CE sums (memory on, off) over the scored tokens of the target chunk ``x [C]`` (``s``: scored, masked)."""
        inp = (jnp.asarray(np.where(s, mask_id, x)), jnp.asarray(x))  # with the clean tokens (read_block models)
        if ws.outer_shift:                                            # slot j holds token j, predicts token j + 1
            tgt, sel = np.concatenate([x[1:], x[-1:]]), np.concatenate([s[1:], [False]])
        else:
            tgt, sel = x, s
        tgt = jnp.asarray(tgt)
        on = np.asarray(ce(model, inp, pos_of(t), st, tgt), np.float64)[sel].sum()
        off = np.asarray(ce(model, inp, pos_of(t), reset(st), tgt), np.float64)[sel].sum()
        return (on, off), int(sel.sum())

    def read_l2r(x, s, t, st, mems=(0, 1)):
        """Per scored token, in order: ``[(ce [K], correct [K]) for memory on, reset]`` under the l2r read (``mems``:
        which of the two to read; the other is NaN / False).  The reads are padded to a multiple of 8 (few shapes)."""
        P = np.flatnonzero(s)
        K = len(P)
        Kp = -(-K // 8) * 8
        xs = np.repeat(x[None], Kp, 0)
        for i, p in enumerate(P):
            xs[i, p:] = mask_id                                       # it and everything after it in the chunk
        slots = np.zeros(Kp, np.int32)
        slots[:K] = P - 1 if ws.outer_shift else P                    # slot j holds token j, predicts token j + 1
        tgts = np.zeros(Kp, np.int32)
        tgts[:K] = x[P]
        res = []
        for mi, state in enumerate((st, reset(st))):
            if mi not in mems:
                res.append((np.full(K, np.nan), np.zeros(K, bool)))
                continue
            c_, k_ = tok_reads(model, jnp.asarray(xs), pos_of(t), state, jnp.asarray(slots), jnp.asarray(tgts))
            res.append((np.asarray(c_, np.float64)[:K], np.asarray(k_)[:K]))
        return res

    n, nt, ng = len(seqs), len(tasks), len(targets)
    out = {"ce": np.zeros((n, nt, ng, len(PLACEMENTS), 2)), "count": np.zeros((n, nt)),
           "teach_len": np.zeros((n, nt)), "test_len": np.zeros((n, nt))}
    if read_mode == "l2r":
        out.update(l2r_ce=np.full((n, nt, ng, len(PLACEMENTS), 2, L2R_MAX), np.nan),
                   l2r_ok=np.full((n, nt, ng, len(PLACEMENTS), 2, L2R_MAX), -1, np.int8),
                   l2r_item=np.full((n, nt, L2R_MAX), -1, np.int32))
    families = [f for f, members in FAMILIES.items() if set(members) & set(tasks)]
    need_before = set(range(1, TEACH_END)) | {t - Wn for t in targets}
    t0 = time.time()
    for si, (_, seq) in enumerate(seqs):
        seq = np.asarray(seq, np.int32)
        foreign = np.asarray(seqs[(si + 1) % n][1], np.int32)        # the span task's other document
        # the natural pass: the states the teaching passes start from, and the none reads' states
        before, at = {}, {}
        st = core.init_memory(model, compute_dtype)
        for c in range(targets[-1]):
            if c in need_before:
                before[c] = st
            st = commit(model, chunk(seq, c), pos_of(c), st)
            if c + 1 in targets:
                at[c + 1] = st
        for family in families:
            rng = np.random.default_rng([seed, list(FAMILIES).index(family), si])
            teach, tests = build(family, rng, tok, foreign, eos, sizes)
            tests = {k: v for k, v in tests.items() if k in tasks}
            off = int(rng.integers(1, C - max(len(x) for x, _ in tests.values()) + 1))
            wr = {t: written(seq, teach, C, Wn, t) for t in targets}
            reads = {t: {k: target_chunk(seq, x, sc, off, C, t) for k, (x, sc) in tests.items()} for t in targets}
            for k, (x, _) in tests.items():
                out["teach_len"][si, tasks.index(k)], out["test_len"][si, tasks.index(k)] = len(teach), len(x)
            # the states each placement reads target t with: none - the natural pass; within - the window
            # rewritten with the teaching block; beyond - one pass from the teaching block's first chunk on
            states = {t: {"none": at[t]} for t in targets}
            for t in targets:
                st_w = before[t - Wn]
                for c in range(t - Wn, t):
                    st_w = commit(model, chunk(wr[t]["within"], c), pos_of(c), st_w)
                states[t]["within"] = st_w
            first = (TEACH_END * C - len(teach)) // C
            ids_b = wr[targets[0]]["beyond"]                          # the same for every target
            st = before[first]
            for c in range(first, targets[-1]):
                st = commit(model, chunk(ids_b, c), pos_of(c), st)
                if c + 1 in targets:
                    states[c + 1]["beyond"] = st
            for gi, t in enumerate(targets):
                for k, (x, sc) in reads[t].items():
                    ti = tasks.index(k)
                    if read_mode == "masked":
                        for pi, p in enumerate(PLACEMENTS):
                            out["ce"][si, ti, gi, pi], out["count"][si, ti] = read(x, sc, t, states[t][p])
                        continue
                    K = int(sc.sum())
                    if K > L2R_MAX:
                        raise ValueError(f"{k}: {K} scored tokens > {L2R_MAX}")
                    out["count"][si, ti] = K
                    P = np.flatnonzero(sc)
                    out["l2r_item"][si, ti, :K] = np.cumsum(np.concatenate([[1], np.diff(P) > 1])) - 1
                    for pi, p in enumerate(PLACEMENTS):
                        r = read_l2r(x, sc, t, states[t][p], mems=(1,) if p == "within" else (0, 1))
                        for mi, (c_, k_) in enumerate(r):
                            out["l2r_ce"][si, ti, gi, pi, mi, :K] = c_
                            out["l2r_ok"][si, ti, gi, pi, mi, :K] = k_
                            out["ce"][si, ti, gi, pi, mi] = c_.sum()
            del states
        log(f"[cl] sequence {si + 1}/{n}: {time.time() - t0:.0f}s")
    return out, targets


L2R_CONDS = {"mem": (0, 0), "reset": (0, 1), "window": (1, 1), "prior": (2, 0), "prior_off": (2, 1)}   # (placement, memory)


def exact_rates(o: dict, ti: int, gi: int, pi: int, mi: int):
    """Per sequence ``(share of its answers exactly right, share of its scored tokens right)`` under the l2r read."""
    ex, ta = [], []
    for si in range(o["l2r_ok"].shape[0]):
        ok, item = o["l2r_ok"][si, ti, gi, pi, mi], o["l2r_item"][si, ti]
        v = item >= 0
        if not v.any():
            ex.append(np.nan)
            ta.append(np.nan)
            continue
        ex.append(np.mean([np.all(ok[v & (item == i)] == 1) for i in np.unique(item[v])]))
        ta.append(np.mean(ok[v] == 1))
    return np.asarray(ex), np.asarray(ta)


def summarise(o: dict, tasks, targets, chunk: int) -> dict:
    ce, cnt = o["ce"], o["count"]
    S = ce.shape[0]
    ms = lambda x: [float(np.nanmean(x)), float(np.nanstd(x, ddof=1) / np.sqrt(np.sum(~np.isnan(x))))]

    def wm(x, w):
        w = w / w.sum()
        m = float((w * x).sum())
        return m, float(np.sqrt((w ** 2 * (x - m) ** 2).sum() * S / max(S - 1, 1)))
    rows = []
    for ti, task in enumerate(tasks):
        w = cnt[:, ti]
        for gi, t in enumerate(targets):
            per = ce[:, ti, gi] / np.maximum(w, 1)[:, None, None]     # [S, placement, (on, off)] per token
            b, wi, nn = per[:, 0], per[:, 1], per[:, 2]
            recall = (b[:, 1] - b[:, 0]) - (nn[:, 1] - nn[:, 0])
            ceil = b[:, 1] - wi[:, 1]
            r, rse = wm(recall, w)
            c_, cse = wm(ceil, w)
            rows.append({"task": task, "target": t, "distance": t - (TEACH_END - 1), "tokens": int(w.sum()),
                         "recall": [r, rse], "ceiling": [c_, cse], "share": r / c_ if c_ > 0.05 else None,
                         "gain_beyond": wm(b[:, 1] - b[:, 0], w), "gain_none": wm(nn[:, 1] - nn[:, 0], w),
                         "gain_within": wm(wi[:, 1] - wi[:, 0], w),
                         "ce": {p: {"on": wm(per[:, k, 0], w)[0], "off": wm(per[:, k, 1], w)[0]}
                                for k, p in enumerate(PLACEMENTS)}})
            if "l2r_ok" in o:
                r = {c: exact_rates(o, ti, gi, pi, mi) for c, (pi, mi) in L2R_CONDS.items()}
                rows[-1].update(exact={c: ms(v[0]) for c, v in r.items()}, tok_acc={c: ms(v[1]) for c, v in r.items()},
                                exact_gain=ms(r["mem"][0] - r["reset"][0]))
    return {"sequences": S, "chunk": chunk, "rows": rows}


def report(s: dict, tag: str):
    print(f"[cl] {tag}: {s['sequences']} sequences; per scored token, nats (share = recall / ceiling)")
    print(f"[cl] {'task':>10s} {'d':>3s} {'tokens':>7s} {'off|none':>8s} {'ceiling':>8s} {'recall':>16s} {'share':>7s} "
          f"{'gain beyond':>11s} {'gain none':>9s} {'gain within':>11s}")
    for r in s["rows"]:
        sh = f"{100 * r['share']:6.1f}%" if r["share"] is not None else "    n/a"
        print(f"[cl] {r['task']:>10s} {r['distance']:3d} {r['tokens']:7d} {r['ce']['none']['off']:8.3f} "
              f"{r['ceiling'][0]:8.3f} {r['recall'][0]:+8.4f} ± {r['recall'][1]:.4f} {sh} "
              f"{r['gain_beyond'][0]:+11.4f} {r['gain_none'][0]:+9.4f} {r['gain_within'][0]:+11.4f}", flush=True)
    if s["rows"] and "exact" in s["rows"][0]:
        print(f"[cl] l2r read, answers exactly right (tokens right): memory | reset | window (memory off) | prior")
        for r in s["rows"]:
            e, a = r["exact"], r["tok_acc"]
            print(f"[cl] {r['task']:>10s} {r['distance']:3d}  " + " | ".join(
                f"{e[c][0]:.3f} ({a[c][0]:.3f})" for c in ("mem", "reset", "window", "prior")) +
                f"   memory - reset {r['exact_gain'][0]:+.3f} ± {r['exact_gain'][1]:.3f}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    _cli.add_common(ap)
    ap.add_argument("--targets", default="8,20", help="target chunks (the teaching block ends with chunk 3)")
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--tokenizer", default="configs/1b_e2e/tokenizer.json")
    ap.add_argument("--task_seed", type=int, default=20261009)
    ap.add_argument("--read", default="masked", choices=("masked", "l2r"),
                    help="l2r: the read of left-to-right generation, per scored token (exact-match rates too)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    _cli.require_platform()
    import jax.numpy as jnp
    model, ws = _cli.load(a)
    seqs = _cli.sequences(a, model.cfg)
    tasks = [x for x in a.tasks.split(",") if x]
    if set(tasks) - set(TASKS):
        sys.exit(f"--tasks: unknown {sorted(set(tasks) - set(TASKS))} (from {TASKS})")
    tok = Tok(a.tokenizer, model.cfg.vocab_size)
    if not (tok.real or a.random_init):
        sys.exit(f"--tokenizer {a.tokenizer}: missing, or larger than the model's vocabulary")
    o, targets = run(model, seqs, ws, [int(x) for x in a.targets.split(",") if x], a.task_seed, jnp.dtype(a.dtype),
                     tok, tasks, read_mode=a.read)
    s = summarise(o, tasks, targets, model.cfg.chunk_size)
    meta = {"ckpt": a.ckpt, "memory": model.cfg.memory, "window_reach": model.cfg.window_reach,
            "inner_lr": ws.inner_lr, "seq_len": a.seq_len, "targets": targets, "tasks": tasks,
            "task_seed": a.task_seed, "batch_index": a.batch_index, "dtype": a.dtype, "tokenizer": tok.real,
            "read": a.read}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, sources=np.array([nm for nm, _ in seqs]), meta=np.array(json.dumps(meta)),
             summary=np.array(json.dumps(s)), **o)
    report(s, a.tag or (a.ckpt or ""))
    print(f"[cl] saved {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
