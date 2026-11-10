"""Checks of qec/wide.py against direct arithmetic: every input at small widths, drawn inputs at
the tick's widths, and a mutation control.

    python3.12 -I -B scripts/wide_check.py <tree>

gt_tree     t ^= ctrl * [A > B], with A, B and ctrl restored and the scratch clean.
gt_sweep    the same map over arith.carry_trees' blocks.
PrefixTree  .top = AND(bits). write(t, mid) leaves bits + t mod 2^h and t, cleans the scratch, and
            hands mid a wire z with t * z = t * AND(old bits).
PrefixSplit the same .top (when asked for) and write(t), the high half's tree on the low half's
            own wires (PrefixLent).
LowAdd      .top = the carry out of L + K, K the XOR of the selected masks. undo() restores every
            wire. write() leaves L[0] alone and adds K and its carries into L[1:], the quiet run
            through a PrefixTree's write. The aside write also with its low carries cleared by a
            tree sweep (wide.LOW_FALL_TREE).
Control     one circuit per routine, each of its CCX turned into a CX in turn: the same check
            must fail.

Exits 1 on any failing lane, dirty wire or unseen mutant.
"""
import random
import sys
from array import array
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv[1]).resolve()))
from qec import sim, wide  # noqa: E402
from qec.builder import CCX, CX, Circuit  # noqa: E402


def gt_case(width, block, cases, sweep=None):
    """cases: (A, B, ctrl, t). sweep: gt_sweep over that partition instead of gt_tree."""
    C = Circuit()
    A, B = C.alloc(width), C.alloc(width)
    z, t = C.alloc(2)
    if sweep:
        wide.gt_sweep(C, A, B, t, z, sweep)
    else:
        wide.gt_tree(C, A, B, t, z, block)
    init = {}
    for k, wires in enumerate((A, B, [z], [t])):
        init.update(sim.pack([c[k] for c in cases], wires))
    keep = A + B + [z]

    def failing(ops):
        st = sim.run(ops, C.num_wires, init, len(cases))
        got = sim.unpack(st, [t], len(cases))
        bad = sum(got[i] != c[3] ^ (c[2] & (c[0] > c[1])) for i, c in enumerate(cases))
        bad += sum(st[w] != init[w] for w in keep)
        return bad + sum(1 for w in range(C.num_wires) if w not in keep and w != t and st[w])

    return C, failing


def prefix_case(h, block, cases, hooked=True):
    """cases: (bits, t). `out` takes AND(bits) before the write, `hook` takes t * z inside it, or,
    with no hook in the write, the same product from the bits before it."""
    C = Circuit()
    bits = C.alloc(h)
    t, out, hook = C.alloc(3)
    P = wide.PrefixTree(C, bits, block)
    C.cx(P.top, out)
    if hooked:
        P.write(t, lambda z: C.ccx(t, z, hook))
    else:
        C.ccx(t, P.top, hook)
        P.write(t)
    init = sim.pack([c[0] for c in cases], bits)
    init.update(sim.pack([c[1] for c in cases], [t]))
    full = (1 << h) - 1

    def failing(ops):
        n = len(cases)
        st = sim.run(ops, C.num_wires, init, n)
        got, top, carry = sim.unpack(st, bits, n), sim.unpack(st, [out], n), sim.unpack(st, [hook], n)
        bad = sum(got[i] != (v + c) & full or top[i] != (v == full) or carry[i] != (c and v == full)
                  for i, (v, c) in enumerate(cases))
        bad += st[t] != init[t]
        return bad + sum(1 for w in range(C.num_wires) if w not in (*bits, t, out, hook) and st[w])

    return C, failing


def low_case(w, quiet, masks, block, cases, write, aside=False, fall=None):
    """cases: (L, selector bits). aside: the write that falls against copies, its hook reading
    L[1:a] as sums, and release() at the end. fall: wide.LOW_FALL_TREE for this circuit."""
    a, b = quiet
    saved, wide.LOW_FALL_TREE = wide.LOW_FALL_TREE, fall
    try:
        return _low_case(w, a, b, masks, block, cases, write, aside)
    finally:
        wide.LOW_FALL_TREE = saved


def _low_case(w, a, b, masks, block, cases, write, aside):
    quiet = (a, b)
    C = Circuit()
    L, sel = C.alloc(w), C.alloc(len(masks))
    (out,) = C.alloc(1)
    M = wide.PrefixTree(C, L[a:b], 3) if b > a else None
    add = wide.LowAdd(C, L, sel, masks, quiet, M.top if M else None, block)
    C.cx(add.top, out)
    (seen,) = C.alloc(1)
    if write and aside:
        def hook():
            if a > 1:
                C.cx(L[1], seen)
        add.write((lambda c, mid: M.write(c, mid, carries_first=True)) if M else None, aside=True,
                  before_falls=hook)
        if a > 1:
            C.cx(add.copies[1], seen)                   # the copy holds L[1]'s sum after the falls
        add.release()
    elif write:
        add.write(M.write if M else None)
    else:
        add.undo()
        if M:
            M.tree.play(C, reverse=True)
            M.chains.play(C, reverse=True)
    init = sim.pack([c[0] for c in cases], L)
    init.update(sim.pack([c[1] for c in cases], sel))
    keep = set(L) | set(sel) | {out}

    def failing(ops):
        n = len(cases)
        st = sim.run(ops, C.num_wires, init, n)
        gotL, gotS, gotT = sim.unpack(st, L, n), sim.unpack(st, sel, n), sim.unpack(st, [out], n)
        bad = 0
        for i, (Lv, sv) in enumerate(cases):
            K = 0
            for j, m in enumerate(masks):
                if sv >> j & 1:
                    K ^= m
            total = Lv + (K & (1 << w) - 1)
            want = ((total & ~1) | (Lv & 1)) & (1 << w) - 1 if write else Lv
            bad += gotL[i] != want or gotS[i] != sv or gotT[i] != total >> w & 1
        return bad + sum(1 for x in range(C.num_wires) if x not in keep and st[x])

    return C, failing


def split_case(h, block, cases, top=True):
    """cases: (bits, t). With top, `out` takes AND(bits) and `hook` t * AND(bits) before the write."""
    C = Circuit()
    bits = C.alloc(h)
    t, out, hook = C.alloc(3)
    P = wide.PrefixSplit(C, bits, block, top)
    if top:
        C.cx(P.top, out)
        C.ccx(t, P.top, hook)
    P.write(t)
    init = sim.pack([c[0] for c in cases], bits)
    init.update(sim.pack([c[1] for c in cases], [t]))
    full = (1 << h) - 1

    def failing(ops):
        n = len(cases)
        st = sim.run(ops, C.num_wires, init, n)
        got, seen, carry = sim.unpack(st, bits, n), sim.unpack(st, [out], n), sim.unpack(st, [hook], n)
        bad = sum(got[i] != (v + c) & full or seen[i] != (top and v == full) or carry[i] != (top and c and v == full)
                  for i, (v, c) in enumerate(cases))
        bad += st[t] != init[t]
        return bad + sum(1 for w in range(C.num_wires) if w not in (*bits, t, out, hook) and st[w])

    return C, failing


def mutants(C, failing):
    """(seen, total) over the circuit's CCX-to-CX mutants."""
    ops = C.ops
    seen = total = 0
    for i in range(0, len(ops), 4):
        if ops[i] != CCX:
            continue
        broken = array("i", ops)
        broken[i], broken[i + 2] = CX, broken[i + 3]
        total += 1
        seen += failing(broken) > 0
    return seen, total


fails = 0
rng = random.Random(7)
BLOCKS = (1, 2, 3, 4, 5, 8, (1, 2, 3), (2, 3, 4, 5))

count = 0
for width in range(1, 8):
    cases = [(x, y, z, t) for x in range(1 << width) for y in range(1 << width) for z in (0, 1) for t in (0, 1)]
    for block in BLOCKS:
        C, failing = gt_case(width, block, cases)
        fails += failing(C.ops)
        count += 1
print(f"gt_tree: {count} circuits on every input, widths 1..7: failing {fails}")
for width in (33, 64, 221, 255, 256):
    cases = []
    for _ in range(400):
        x = rng.getrandbits(width)
        k = rng.randrange(width + 1)
        y = x ^ (rng.getrandbits(k) if k else 0)        # equal above bit k, so low blocks decide
        cases += [(x, y, rng.getrandbits(1), rng.getrandbits(1)), (y, x, 1, 0)]
    for block in BLOCKS:
        C, failing = gt_case(width, block, cases)
        fails += failing(C.ops)
print(f"gt_tree: widths 33..256 on 800 drawn inputs each: failing {fails}")

SWEEPS = ((2, None, 128, 2, 2), (1.6, 2, 4, 2, 2), (2, None, 3, 3, 2), (None, None, 2, 2, 1), (1.5, None, 300, 2, 3))
count = 0
for width in range(1, 8):
    cases = [(x, y, z, t) for x in range(1 << width) for y in range(1 << width) for z in (0, 1) for t in (0, 1)]
    for spec in SWEEPS:
        C, failing = gt_case(width, None, cases, spec)
        fails += failing(C.ops)
        count += 1
print(f"gt_sweep: {count} circuits on every input, widths 1..7: failing {fails}")
for width in (33, 64, 221, 255, 256):
    cases = []
    for _ in range(400):
        x = rng.getrandbits(width)
        k = rng.randrange(width + 1)
        y = x ^ (rng.getrandbits(k) if k else 0)
        cases += [(x, y, rng.getrandbits(1), rng.getrandbits(1)), (y, x, 1, 0)]
    for spec in SWEEPS:
        C, failing = gt_case(width, None, cases, spec)
        fails += failing(C.ops)
print(f"gt_sweep: widths 33..256 on 800 drawn inputs each: failing {fails}")

count = 0
for h in range(1, 11):
    cases = [(v, c) for v in range(1 << h) for c in (0, 1)]
    for block in range(1, 7):
        for hooked in (True, False):
            C, failing = prefix_case(h, block, cases, hooked)
            fails += failing(C.ops)
            count += 1
print(f"PrefixTree: {count} circuits on every input, h 1..10: failing {fails}")
for h in (20, 33, 221, 222):
    full = (1 << h) - 1
    cases = [(full, 1), (full, 0), (0, 1), (0, 0)]
    for k in list(range(h + 1)) * 2:                    # the lowest 0 at every position
        v = (rng.getrandbits(h) >> k << k | (1 << k) - 1) & full
        cases += [(v, 0), (v, 1)]
    for block in (2, 3, 4, 5, 8, 16):
        for hooked in (True, False):
            C, failing = prefix_case(h, block, cases, hooked)
            fails += failing(C.ops)
print(f"PrefixTree: h 20..222, the lowest 0 at every position: failing {fails}")

count = 0
for h in range(1, 12):
    cases = [(v, c) for v in range(1 << h) for c in (0, 1)]
    for block in range(1, 7):
        for top in (True, False):
            C, failing = split_case(h, block, cases, top)
            fails += failing(C.ops)
            count += 1
print(f"PrefixSplit: {count} circuits on every input, h 1..11: failing {fails}")
for h in (20, 33, 191, 223):
    full = (1 << h) - 1
    cases = [(full, 1), (full, 0), (0, 1), (0, 0)]
    for k in list(range(h + 1)) * 2:                    # the lowest 0 at every position
        v = (rng.getrandbits(h) >> k << k | (1 << k) - 1) & full
        cases += [(v, 0), (v, 1)]
    for block in (2, 4, 5, 8):
        for top in (True, False):
            C, failing = split_case(h, block, cases, top)
            fails += failing(C.ops)
print(f"PrefixSplit: h 20..223, the lowest 0 at every position: failing {fails}")

FALLS = ((2, 2, 2, 2, 2), (None, 2, 8, 2, 2), (2, None, 16, 3, 2), (None, None, 16, 2, 1))
MODES = [(False, False, None), (True, False, None), (True, True, None)] + [(True, True, f) for f in FALLS]
count = 0
for w in range(2, 9):
    for a in range(1, w + 1):
        for b in range(a, w + 1):
            run_bits = sum(1 << i for i in range(a, b))
            for _ in range(3):
                masks = [rng.getrandbits(w) & ~run_bits | 1 for _ in range(rng.randrange(1, 4))]
                cases = [(Lv, sv) for Lv in range(1 << w) for sv in range(1 << len(masks))]
                for block in (1, 2, 3, 4, 8):
                    for write, aside, fall in MODES:
                        C, failing = low_case(w, (a, b), masks, block, cases, write, aside, fall)
                        fails += failing(C.ops)
                        count += 1
print(f"LowAdd: {count} circuits on every input, w 2..8, every quiet run: failing {fails}")
f = (1 << 256) - 0xfffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f
corner = f ^ 2 * f ^ 3 * f
w = (4 * f).bit_length()
for masks in ([f, f ^ 2 * f, f ^ 3 * f, corner ^ 4 * f], [corner, f, 2 * f, 4 * f]):
    cases = []
    for _ in range(600):
        Lv = rng.getrandbits(w)
        if rng.random() < 0.5:
            Lv |= (1 << 32) - (1 << 12)                 # the quiet run all ones
        cases.append((Lv, rng.getrandbits(4)))
    for block in (1, 2, 3, 4, 6, 12):
        for write, aside, fall in MODES:
            C, failing = low_case(w, (12, 32), masks, block, cases, write, aside, fall)
            fails += failing(C.ops)
print(f"LowAdd: secp256k1's masks, w = {w}, quiet run 12..32, 600 drawn inputs: failing {fails}")

missed = 0
width = 6
# Every bit outside the quiet run is in some mask, so no gate of the control circuit is idle.
LOW_MASKS = [0b11100101, 0b01100011, 0b10100111]
LOW_CASES = [(Lv, sv) for Lv in range(1 << 8) for sv in range(8)]
cases = [(x, y, z, t) for x in range(1 << width) for y in range(1 << width) for z in (0, 1) for t in (0, 1)]
for name, (C, failing) in (
        ("gt_tree", gt_case(width, (2, 3, 4, 5), cases)),
        ("gt_sweep", gt_case(width, None, cases, (2, None, 128, 2, 2))),
        ("PrefixTree", prefix_case(9, 3, [(v, c) for v in range(1 << 9) for c in (0, 1)])),
        ("PrefixTree, no hook", prefix_case(9, 4, [(v, c) for v in range(1 << 9) for c in (0, 1)], False)),
        ("PrefixSplit", split_case(9, 2, [(v, c) for v in range(1 << 9) for c in (0, 1)])),
        ("PrefixSplit, no top", split_case(10, 3, [(v, c) for v in range(1 << 10) for c in (0, 1)], False)),
        ("LowAdd write", low_case(8, (3, 5), LOW_MASKS, 3, LOW_CASES, True)),
        ("LowAdd write aside", low_case(8, (3, 5), LOW_MASKS, 3, LOW_CASES, True, True)),
        ("LowAdd write aside, tree falls", low_case(8, (3, 5), LOW_MASKS, 3, LOW_CASES, True, True, (2, 2, 2, 2, 2))),
        ("LowAdd undo", low_case(8, (3, 5), LOW_MASKS, 3, LOW_CASES, False))):
    seen, total = mutants(C, failing)
    missed += total - seen
    print(f"control, {name}: {seen} of {total} CCX-to-CX mutants seen")
sys.exit(1 if fails or missed else 0)
