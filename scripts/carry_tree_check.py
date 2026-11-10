"""Checks for arith.CarrySweep, the carry sweeps over blocks that are trees, and for its two users.

    python3.12 -I -B scripts/carry_tree_check.py <tree> [mmax] [nmax] [wmax]

1. The map, on every input. For m = 1 .. mmax, every way to cut m steps into blocks, and five tree
   shapes inside the blocks: forward over the blocks must leave U[j + 1] = x ^ P[j] U[j] at every
   step, P unchanged and every temp 0, on all 2^(2m + 1) inputs; backward over a different cut and
   shape must then restore U.
2. The same at the generator's own widths and partitions (the Bezout add's two, and every row of
   arith.SWEEP_GEO at several widths), on lanes whose P bits are mostly 1 so that carries cross
   whole blocks, against the ripple it replaces.
3. The Bezout tick's add-or-shift with tree sweeps, on every input for n = 4 .. nmax and every cut
   of both sweeps: word += R >> 1 off J, word's low n + 1 bits rotated left on J, J and R kept,
   scratch 0.
4. The Euclid step with tree sweeps at w = 9 .. wmax (the ripple fallback switched off), on every
   input of the step's domain, against the same step with the Brent-Kung sweep.
5. The Euclid step with its strip bit made before the difference, the select bits' copies on wires
   outside the ring and a partition of its own, against the step with none of these and against
   the ripple step, on every input of the step's domain.
Each part has a mutation control: every CCX of a representative circuit turned into a CX, in turn,
must make the part fail. Exits 1 on any failure, or on a mutant that passes.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv[1]).resolve()))
from qec import arith, point_add, sim, two_bit  # noqa: E402
from qec.arith import CarrySweep, Mod, carry_trees, geo_sizes, tree_node, tree_shape  # noqa: E402
from qec.builder import CCX, CX, Circuit  # noqa: E402

MMAX = int(sys.argv[2]) if len(sys.argv) > 2 else 8
NMAX = int(sys.argv[3]) if len(sys.argv) > 3 else 7
WMAX = int(sys.argv[4]) if len(sys.argv) > 4 else 10
SHAPES = [(2, 1), (2, 2), (3, 2), (2, 3), (3, 3)]
ok = True


def lane_mask(i, N):
    L = 1 << N
    period = 1 << (i + 1)
    block = ((1 << (1 << i)) - 1) << (1 << i)
    return ((1 << L) - 1) // ((1 << period) - 1) * block


def compositions(m):
    if m == 0:
        yield []
        return
    for r in range(1, m + 1):
        for rest in compositions(m - r):
            yield [r] + rest


def roots(sizes, fan, leaf):
    out, s = [], 0
    for r in sizes:
        out.append(tree_node(s, tree_shape(r, fan, leaf)))
        s += r
    return out


def always_on(ops, num_wires, init, lanes):
    """The CCXs whose second control is 1 on every lane where the first is, when they run: on
    these inputs each is the CX its mutant is, so that mutant is no fault."""
    s = [0] * num_wires
    for x, v in init.items():
        s[x] = v
    full, out = (1 << lanes) - 1, set()
    for i in range(0, len(ops), 4):
        op, a, b, c = ops[i:i + 4]
        if op == CCX:
            if not s[a] & ~s[b] & full:
                out.add(i)
            s[c] ^= s[a] & s[b]
        elif op == CX:
            s[b] ^= s[a]
        else:
            s[a] ^= full
    return out


def mutants(ops, skip=()):
    """Each CCX of ops as a CX from its first control, one at a time."""
    for i in range(0, len(ops), 4):
        if ops[i] == CCX and i not in skip:
            m = list(ops)
            m[i:i + 4] = [CX, ops[i + 1], ops[i + 3], 0]
            yield m


def control(tag, ops, passes, sample=None, seed=7, skip=()):
    """passes(ops) -> bool. The true ops must pass and every mutant must fail."""
    global ok
    cand = list(mutants(ops, skip))
    if sample and len(cand) > sample:
        cand = random.Random(seed).sample(cand, sample)
    seen = sum(1 for m in cand if not passes(m))
    good = passes(ops) and seen == len(cand)
    ok &= good
    print(f"control, {tag}: {seen} of {len(cand)} CCX-to-CX mutants seen{'' if good else '  FAIL'}", flush=True)


# 1. The map on every input.
def sweep_case(m, fsizes, fshape, bsizes, bshape):
    C = Circuit()
    U, P, T = C.alloc(m + 1), C.alloc(m), C.alloc(max(m, 1))
    pool = list(T)
    sweep = CarrySweep(C, U, P, lambda: pool.pop(0), pool.append)
    for root in roots(fsizes, *fshape):
        sweep.forward(root)
    mid = len(C.ops)
    for root in reversed(roots(bsizes, *bshape)):
        sweep.backward(root)
    ins = U + P
    N = len(ins)
    init = {x: lane_mask(i, N) for i, x in enumerate(ins)}

    def passes(ops):
        st = sim.run(ops[:mid], C.num_wires, init, 1 << N)
        f = init[U[0]]
        good = st[U[0]] == f and not any(st[t] for t in T)
        for j in range(m):
            f = init[U[j + 1]] ^ (init[P[j]] & f)
            good &= st[U[j + 1]] == f and st[P[j]] == init[P[j]]
        st = sim.run(ops, C.num_wires, init, 1 << N)
        return good and all(st[x] == init.get(x, 0) for x in range(C.num_wires))

    return list(C.ops), passes


cases = 0
for m in range(1, MMAX + 1):
    comps = list(compositions(m))
    for i, sizes in enumerate(comps):
        for k, shape in enumerate(SHAPES):
            back = comps[(i * 7 + k * 3 + 1) % len(comps)]
            ops, passes = sweep_case(m, sizes, shape, back, SHAPES[(k + 2) % len(SHAPES)])
            good = passes(ops)
            ok &= good
            cases += 1
            if not good:
                print(f"sweep m={m} forward {sizes} {shape} backward {back}: FAIL", flush=True)
print(f"1. CarrySweep on every input, m = 1 .. {MMAX}: {cases} cut-and-shape pairs {'ok' if ok else 'FAIL'}", flush=True)
ops, passes = sweep_case(MMAX, [MMAX - 3, 3], (2, 2), [2, MMAX - 2], (3, 2))
control(f"CarrySweep m={MMAX}", ops, passes)


# 2. The generator's widths and partitions on lanes with long carry runs.
def wide_case(m, fspec, bspec, lanes=1024, seed=5):
    rng = random.Random(seed * 1000 + m)
    C = Circuit()
    U, P = C.alloc(m + 1), C.alloc(m)
    T = C.alloc(m)
    pool = list(T)
    sweep = CarrySweep(C, U, P, lambda: pool.pop(0), pool.append)
    for root in carry_trees(m, fspec):
        sweep.forward(root)
    mid = len(C.ops)
    for root in reversed(carry_trees(m, bspec)):
        sweep.backward(root)
    full = (1 << lanes) - 1
    init = {x: rng.getrandbits(lanes) for x in U}
    # A quarter of the lanes each: P uniform, 15/16 ones, 255/256 ones, all ones.
    quarter = lanes // 4
    for x in P:
        v = rng.getrandbits(quarter)
        v |= (rng.getrandbits(quarter) | rng.getrandbits(quarter) | rng.getrandbits(quarter) | rng.getrandbits(quarter)) << quarter
        dense = 0
        for _ in range(8):
            dense |= rng.getrandbits(quarter)
        v |= dense << 2 * quarter
        v |= ((1 << quarter) - 1) << 3 * quarter
        init[x] = v & full

    def passes(ops):
        st = sim.run(ops[:mid], C.num_wires, init, lanes)
        f = init[U[0]]
        good = st[U[0]] == f and not any(st[t] for t in T)
        for j in range(m):
            f = init[U[j + 1]] ^ (init[P[j]] & f)
            good &= st[U[j + 1]] == f
        st = sim.run(ops, C.num_wires, init, lanes)
        return good and all(st[x] == init.get(x, 0) for x in range(C.num_wires))

    return list(C.ops), passes


wide = []
if two_bit.WIDE_ADD_TREE_F and two_bit.WIDE_ADD_TREE_B:
    wide.append(("the tick's add", 255, two_bit.WIDE_ADD_TREE_F, two_bit.WIDE_ADD_TREE_B))
for least, spec in arith.SWEEP_GEO or ():
    for m in sorted({least - 1, least, least + 7, min(257, least * 2), 257 if least >= 190 else least + 30}):
        if m >= 2:
            wide.append((f"sweep row {least}", m, spec, spec))
for tag, m, fspec, bspec in wide:
    ops, passes = wide_case(m, fspec, bspec)
    good = passes(ops)
    ok &= good
    print(f"2. {tag}: m={m} forward {fspec} backward {bspec}, {len(geo_sizes(m, *fspec[:3]))} and "
          f"{len(geo_sizes(m, *bspec[:3]))} blocks, {ops[0::4].count(CCX)} CCX: {'ok' if good else 'FAIL'}", flush=True)
if wide:
    tag, m, fspec, bspec = wide[0]
    ops, passes = wide_case(m, fspec, bspec)
    control(f"{tag} at m={m}, a sample", ops, passes, sample=120)


# 3. The tick's add-or-shift with tree sweeps, on every input.
def add_case(n, fsizes, bsizes, shape):
    C = Circuit()
    (J,) = C.alloc(1)
    R, word = C.alloc(n), C.alloc(n + 2)
    ins = [J] + R + word
    two_bit.add_or_shift_la(C, J, R, word, n, ntemps=max(n - 1, 1), ncopies=3,
                            ftrees=roots(fsizes, *shape), btrees=roots(bsizes, *shape))
    N = len(ins)
    full = (1 << (1 << N)) - 1
    init = {x: lane_mask(i, N) for i, x in enumerate(ins)}
    j, r, w = init[J], [init[x] for x in R], [init[x] for x in word]
    total, c = [], 0
    for x, y in zip(w, r[1:] + [0, 0, 0]):
        total.append(x ^ y ^ c)
        c = (x & y) | (c & (x ^ y))
    rot = [w[n]] + w[:n] + [w[n + 1]]
    want = [(j & u) | (~j & full & v) for u, v in zip(rot, total)]
    scratch = [x for x in range(C.num_wires) if x not in set(ins)]

    def passes(ops):
        st = sim.run(ops, C.num_wires, init, 1 << N)
        return (st[J] == j and [st[x] for x in R] == r and [st[x] for x in word] == want
                and all(st[x] == 0 for x in scratch))

    return list(C.ops), passes


cases, part = 0, True
saved = two_bit.WIDE_WAVE, two_bit.WIDE_WAVE_BACK
for waves in ((True, True), (False, False)):
    two_bit.WIDE_WAVE, two_bit.WIDE_WAVE_BACK = waves
    for n in range(4, NMAX + 1):
        comps = list(compositions(n - 1))
        for i, fsizes in enumerate(comps):
            for k, shape in enumerate(SHAPES[:3]):
                bsizes = comps[(i * 5 + k + 1) % len(comps)]
                ops, passes = add_case(n, fsizes, bsizes, shape)
                good = passes(ops)
                part &= good
                cases += 1
                if not good:
                    print(f"add n={n} F {fsizes} B {bsizes} {shape} waves {waves}: FAIL", flush=True)
two_bit.WIDE_WAVE, two_bit.WIDE_WAVE_BACK = True, True
ok &= part
print(f"3. add-or-shift with tree sweeps on every input, n = 4 .. {NMAX}: {cases} layouts {'ok' if part else 'FAIL'}", flush=True)
ops, passes = add_case(NMAX, [NMAX - 3, 2], [2, NMAX - 3], (2, 2))
control(f"add-or-shift n={NMAX}", ops, passes)
two_bit.WIDE_WAVE, two_bit.WIDE_WAVE_BACK = saved


# 4. The Euclid step with tree sweeps against the Brent-Kung sweep, on the step's whole domain.
def step_case(w, geo, ring=None, **flags):
    """flags: point_add settings for this build (the early strip bit, the select copies on free
    wires, the per-width partitions); the defaults turn all three off."""
    flags = {"TWO_EARLY": 0, "STEP_FREE": 0, "EUCLID_GEO": {}, "RIPPLE_MAX_W": 0, **flags}
    saved = arith.SWEEP_GEO, {name: getattr(point_add, name) for name in flags}
    arith.SWEEP_GEO = geo
    for name, value in flags.items():
        setattr(point_add, name, value)
    q = next(p for p in (251, 509, 1021, 2039, 4093, 8191) if p.bit_length() >= w)
    C = Circuit()
    m = point_add.DialogMultiplier(C, Mod(q))
    k = 2 * w if ring is None else ring
    U, V, E, cell, ring = C.alloc(w), C.alloc(w), C.alloc(1), C.alloc(2), C.alloc(k)
    C.apply(m._euclid_step(w, k), [U, V, E, cell, ring])
    arith.SWEEP_GEO = saved[0]
    for name, value in saved[1].items():
        setattr(point_add, name, value)
    free = U[1:] + V + E
    N = len(free)
    init = {x: lane_mask(i, N) for i, x in enumerate(free)}
    init[U[0]] = (1 << (1 << N)) - 1
    io = U + V + E + cell
    return C, init, N, io


part = True
for w in range(9, WMAX + 1):
    C0, init, N, io = step_case(w, None)
    want = sim.run(C0.ops, C0.num_wires, init, 1 << N)
    for spec in ((2, 2, 4, 2, 2), (1.5, 2, 3, 2, 1), (2, 3, 8, 3, 2)):
        C1, init1, _, io1 = step_case(w, ((3, spec),))
        assert io1 == io and init1 == init

        def passes(ops, C1=C1):
            st = sim.run(ops, C1.num_wires, init, 1 << N)
            return ([st[x] for x in io] == [want[x] for x in io]
                    and all(st[x] == 0 for x in range(C1.num_wires) if x not in set(io)))

        good = passes(C1.ops) and C1.ops != C0.ops
        part &= good
        print(f"4. Euclid step w={w} with sweep {spec}: lanes 2^{N}, CCX {C0.ops[0::4].count(CCX)} -> "
              f"{C1.ops[0::4].count(CCX)}: {'ok' if good else 'FAIL'}", flush=True)
ok &= part
# The domain fixes U[0] = 1 and ties the cell's wires, so a few CCXs are CXs on every lane of it.
control(f"Euclid step w={WMAX} with tree sweeps, a sample", list(C1.ops), passes, sample=60,
        skip=always_on(C1.ops, C1.num_wires, init, 1 << N))



# 5. The step with its strip bit made before the difference, a copy of each select bit a bit on
# wires outside the ring, and a partition of its own, against the step with none of them and
# against the ripple step, on the step's whole domain. The ring is the pass's size, w - 2.
part = True
for w in range(9, WMAX + 1):
    C0, init, N, io = step_case(w, None, ring=w - 2)
    want = sim.run(C0.ops, C0.num_wires, init, 1 << N)
    Cr, initr, _, ior = step_case(w, None, ring=w - 2, RIPPLE_MAX_W=12)
    wantr = sim.run(Cr.ops, Cr.num_wires, initr, 1 << N)
    assert ior == io and initr == init and Cr.ops != C0.ops
    part &= [wantr[x] for x in io] == [want[x] for x in io]
    for flags in ({"TWO_EARLY": 128, "STEP_FREE": 12}, {"TWO_EARLY": 128, "STEP_FREE": 6, "TWO_ALTERNATE": False, "TWO_SPLIT": True},
                  {"STEP_FREE": 270},
                  {"EUCLID_GEO": {w: (2, 3, 4, 2, 2)}}, {"TWO_EARLY": 2, "STEP_FREE": 270, "EUCLID_GEO": {w: (2, 4, 2, 2, 2)}},
                  {"TWO_EARLY": 128, "STEP_FREE": 270, "EUCLID_GEO": {w: (2, 3, 8, 2, 2)}}):
        C1, init1, _, io1 = step_case(w, arith.SWEEP_GEO, ring=w - 2, **flags)
        assert io1 == io and init1 == init

        def passes(ops, C1=C1):
            st = sim.run(ops, C1.num_wires, init, 1 << N)
            return ([st[x] for x in io] == [want[x] for x in io]
                    and all(st[x] == 0 for x in range(C1.num_wires) if x not in set(io)))

        good = passes(C1.ops) and C1.ops != C0.ops
        part &= good
        print(f"5. Euclid step w={w} with {flags}: lanes 2^{N}, wires {C0.num_wires} -> {C1.num_wires}: "
              f"{'ok' if good else 'FAIL'}", flush=True)
ok &= part
control(f"Euclid step w={WMAX} with the early strip bit and the select copies, a sample", list(C1.ops), passes,
        sample=80, skip=always_on(C1.ops, C1.num_wires, init, 1 << N))

print("carry_tree_check:", "ok" if ok else "FAIL")
sys.exit(0 if ok else 1)
