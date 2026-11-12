"""Gate-level checks of the B1 sweep inside the Bezout ticks and of the swap wires.

On each prime in PRIMES whose multiplier sweeps inline, over every x in 1 .. q - 1:

  raw      the Euclid pass run without its sweep leaves the semantic tape of `tick_tape`, cell by
           cell, and each swap wire holds g0 AND b of its cell;
  old      `_b1_sweep` takes that tape to the swept tape of `swept`, a model that shares no code
           with the circuit, and leaves clean scratch;
  chain    `_sweep_sw`, the sweep as the ticks apply it, reaches the same tape, leaves every swap
           wire as it found it, and leaves clean scratch;
  product  the multiplier and its inverse are right on every x and the sampled y.

Each mutant in MUTANTS must then fail `product` at some prime in MUTANT_PRIMES:

  no-ccx       `_sweep_in` without its CCX, so the toggle never moves;
  no-toggle    `_sweep_in` without its last CX, so a cell keeps its unswept bit;
  no-restore   the mark on a swap wire turned back without its X;
  no-unsweep   the tape left swept for the inverse Euclid pass.

    python3 -I scripts/sweep_check.py
"""
import io
import json
import random
import sys
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from check import expand, structural
from qec import flatten, sim
from qec.arith import Mod, load_const
from qec.builder import Circuit
from qec.emit import write_circuit
from qec.model import tick_tape
from qec.point_add import DialogMultiplier
from qec.two_bit import TwoBitMod

PRIMES = (7, 13, 31, 61, 127, 251, 509, 1021, 2039, 4093)
MUTANT_PRIMES = (31, 61, 127, 251)


def swept(cells):
    """Each maximal run of no-subtract cells, in tape order, with its bits rotated so that a
    single strip, which Euclid makes last, comes first."""
    out, i = list(cells), 0
    while i < len(out):
        j = i
        while j < len(out) and not out[j][0]:
            j += 1
        if j > i:
            bits = [b for _, b in out[i:j]]
            out[i:j] = [(0, b) for b in bits[-1:] + bits[:-1]]
        i = j + 1
    return out


def tape_value(cells):
    return sum((add + 2 * bit) << (2 * i) for i, (add, bit) in enumerate(cells))


def swap_value(cells, count):
    return sum((add & bit) << (i - 1) for i, (add, bit) in enumerate(cells) if 1 <= i <= count)


def pass_bad(q, stage):
    """Mismatched bits after the unswept Euclid pass and then `stage`: None, 'old' or 'chain'."""
    C = Circuit()
    mod = Mod(q)
    m = DialogMultiplier(C, mod)
    U, V = C.alloc(mod.n), C.alloc(mod.n)
    inputs = list(V)
    pool = V + C.alloc(m.tape_external)
    load_const(C, U, q)
    junk = C.alloc(m.junk_width)
    swr = C.alloc(m.swap_width)
    m._euclid_pass(U, pool, junk, sweep=False, swr=swr or None)
    G = m.coded_tape(pool)
    if stage == "old":
        m._b1_sweep(G)
    elif stage == "chain":
        m._sweep_sw(G, swr)
    structural(C.prims, C.num_wires)
    xs = list(range(1, q))
    state = sim.run(C.prims, C.num_wires, sim.pack(xs, inputs), len(xs))
    cells = [tick_tape(q, x, m.iters)[2] for x in xs]
    want = sim.pack([tape_value(swept(c) if stage else c) for c in cells], G)
    want |= sim.pack([swap_value(c, len(swr)) for c in cells], swr)
    # The junk is the pass's own and model_check.py checks it; every other wire must be as modelled.
    skip = set(junk) | set(U) | set(pool) - set(G)
    bad = 0
    for w, actual in enumerate(state):
        if w not in skip:
            bad |= actual ^ want.get(w, 0)
    return bad.bit_count()


def product_bad(q, inverse):
    ys = range(q) if q < 300 else random.Random(q).sample(range(q), 8)
    pairs = [(x, y) for x in range(1, q) for y in ys]
    C = Circuit(hierarchical=True)
    mod = Mod(q)
    X, Y = C.alloc(mod.n), C.alloc(mod.n)
    inputs = (list(X), list(Y))
    DialogMultiplier(C, mod).apply(X, Y, inverse=inverse)
    buf = io.StringIO()
    write_circuit(buf, C, {"control": [], "x": X, "y": Y})
    ops = expand(flatten.load(io.StringIO(buf.getvalue())))
    structural(ops, C.num_wires)
    xs, values = zip(*pairs)
    state = sim.run(ops, C.num_wires, sim.pack(xs, inputs[0]) | sim.pack(values, inputs[1]), len(pairs))
    expected = [y * (pow(x, -1, q) if inverse else x) % q for x, y in pairs]
    want = sim.pack(xs, X) | sim.pack(expected, Y)
    bad = 0
    for w, actual in enumerate(state):
        bad |= actual ^ want.get(w, 0)
    return bad.bit_count()


def _no_ccx(self, C, g0, b, sw, up, nxt, tog):
    C.cx(tog, b)


def _no_toggle(self, C, g0, b, sw, up, nxt, tog):
    C.x(sw)
    C.cx(g0, sw)
    C.cx(b, sw)
    if up is None:
        C.x(nxt)
    else:
        C.cx(up, nxt)
    C.ccx(sw, nxt, tog)
    if up is None:
        C.x(nxt)
    else:
        C.cx(up, nxt)
    C.cx(b, sw)
    C.cx(g0, sw)
    C.x(sw)


def _no_restore(self, C, prev, p, tape=False):
    if prev is None:
        return
    if tape:
        C.cx(prev[0], p)
        C.cx(prev[1], p)
    else:
        self._mark_b1(C, prev, p)
        C.free([p])


def _no_unsweep(self, G, swr):
    pass


MUTANTS = {
    "no-ccx": (TwoBitMod, "_sweep_in", _no_ccx),
    "no-toggle": (TwoBitMod, "_sweep_in", _no_toggle),
    "no-restore": (TwoBitMod, "_drop_after_b1", _no_restore),
    "no-unsweep": (DialogMultiplier, "_sweep_sw", _no_unsweep),
}


@contextmanager
def mutant(name):
    owner, attr, fn = MUTANTS[name]
    saved = getattr(owner, attr)
    setattr(owner, attr, fn)
    try:
        yield
    finally:
        setattr(owner, attr, saved)


def inline(q):
    m = DialogMultiplier(Circuit(), Mod(q))
    return m.inline_sweep and m.swap_width > 0


def main():
    ok, covered = True, 0
    for q in PRIMES:
        if not inline(q):
            continue
        covered += 1
        m = DialogMultiplier(Circuit(), Mod(q))
        row = {"q": q, "ticks": m.iters, "rank_cells": m.rank_j, "swap_wires": m.swap_width,
               "raw_bad": pass_bad(q, None), "old_bad": pass_bad(q, "old"), "chain_bad": pass_bad(q, "chain"),
               "forward_bad": product_bad(q, False), "inverse_bad": product_bad(q, True)}
        ok &= not any(v for k, v in row.items() if k.endswith("_bad"))
        print(json.dumps(row), flush=True)
    for name in MUTANTS:
        with mutant(name):
            bad = {q: product_bad(q, False) + product_bad(q, True) for q in MUTANT_PRIMES if inline(q)}
        caught = any(bad.values())
        ok &= caught
        print(json.dumps({"mutant": name, "caught": caught, "bad_lanes": bad}), flush=True)
    # Both start shapes must be covered: a pass with a rank-one start and one without.
    shapes = {bool(DialogMultiplier(Circuit(), Mod(q)).rank_j) for q in PRIMES if inline(q)}
    ok &= shapes == {False, True} and covered >= 8
    print(json.dumps({"primes": covered, "start_shapes": sorted(shapes), "ok": ok}))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
