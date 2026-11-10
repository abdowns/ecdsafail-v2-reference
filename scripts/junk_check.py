"""The junk wires: Euclid carries kept across the Bezout pass, and square row carries kept across
the modadd.

M runs on every x in [1, q) with a random y at every odd prime below the bound, forward from a
clean start and backward from the clean expected output, once per junk budget. The budgets cover
none, a few wires (one step partly kept), about half, all a pass can use, and more than that.
The square's compute runs on every n-bit Y at every prime below the second bound where the fold
applies, once per keep budget, and its inverse must return every junk wire to zero.
A control turns CCX gates into CX at random and counts how many of those mutants a check sees.

Usage: junk_check.py [M prime bound, default 600] [square prime bound, default 4096]
"""
import json
import random
import sys
from array import array
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qec import sim
from qec.arith import Mod
from qec.builder import CCX, CX, Circuit
from qec.point_add import DialogMultiplier
from qec.square import SquareFold


def inverse(ops):
    rev = ops[::-1]
    out = array("i", ops)
    out[0::4], out[1::4], out[2::4], out[3::4] = rev[3::4], rev[2::4], rev[1::4], rev[0::4]
    return out


def pack(values, wires):
    return {w: int("".join("1" if v >> i & 1 else "0" for v in reversed(values)) or "0", 2) for i, w in enumerate(wires)}


def bad_lanes(state, want, num_wires):
    bad = 0
    for w in range(num_wires):
        bad |= state[w] ^ want.get(w, 0)
    return bad.bit_count()


def primes(lo, hi):
    return [p for p in range(lo, hi) if p % 2 and all(p % d for d in range(3, int(p**.5) + 1, 2))]


def build_m(q, junk):
    mod = Mod(q)
    C = Circuit()
    X, Y = C.alloc(mod.n), C.alloc(mod.n)
    X0, Y0 = list(X), list(Y)
    m = DialogMultiplier(C, mod, junk)
    m.apply(X, Y)
    return C, m, X0, Y0, X, Y


def run_m(q, junk, cases, ops=None):
    C, m, X0, Y0, X, Y = build_m(q, junk)
    ops = C.ops if ops is None else ops
    xs, ys = [c[0] for c in cases], [c[1] for c in cases]
    init = pack(xs, X0) | pack(ys, Y0)
    out = pack(xs, X) | pack([a * b % q for a, b in cases], Y)
    forward = bad_lanes(sim.run(ops, C.num_wires, init, len(cases)), out, C.num_wires)
    backward = bad_lanes(sim.run(inverse(ops), C.num_wires, out, len(cases)), init, C.num_wires)
    return forward, backward, min(junk, m.junk_capacity), ops[0::4].count(CCX)


def budgets(capacity):
    return sorted({0, 1, 2, 3, 5, 8, capacity // 3, capacity // 2, capacity - 1, capacity, capacity + 7} - {-1})


def check_m(bound, r):
    rows = fails = 0
    saved = []
    for q in primes(5, bound):
        cases = [(x, r.randrange(q)) for x in range(1, q)]
        capacity = DialogMultiplier(Circuit(), Mod(q)).junk_capacity
        base = None
        for junk in budgets(capacity):
            forward, backward, used, ccx = run_m(q, junk, cases)
            base = ccx if base is None else base
            rows += 1
            fails += forward + backward
            # Each wire a pass uses saves one CCX in the pass and one in its inverse.
            if base - ccx != 2 * used:
                fails += 1
                print(json.dumps({"q": q, "junk": junk, "used": used, "ccx": ccx, "base": base, "error": "saving is not 2 per wire"}))
            if forward or backward:
                print(json.dumps({"q": q, "junk": junk, "forward failures": forward, "inverse failures": backward}))
        saved.append(capacity)
    return {"check": "M", "primes": len(saved), "rows": rows, "failures": fails, "capacity range": [min(saved), max(saved)]}


def run_square(q, keep):
    mod = Mod(q)
    fold = SquareFold(mod, keep)
    n = mod.n
    C = Circuit()
    Y, J = C.alloc(n), C.alloc(fold.junk)
    fold.compute(C, Y, J)
    ys = list(range(1 << n))
    init = pack(ys, Y)
    state = sim.run(C.ops, C.num_wires, init, len(ys))
    got = sim.unpack(state, J[:n], len(ys))
    wrong = sum(g != y * y % q for g, y in zip(got, ys))
    scratch = 0
    live = set(Y) | set(J)
    for w in range(C.num_wires):
        if w not in live:
            scratch |= state[w]
    kept_y = sum(a != b for a, b in zip(sim.unpack(state, Y, len(ys)), ys))
    back = sim.run(inverse(C.ops), C.num_wires, {w: state[w] for w in range(C.num_wires)}, len(ys))
    unclean = bad_lanes(back, init, C.num_wires)
    return wrong + scratch.bit_count() + kept_y + unclean, fold.kept, C.ops[0::4].count(CCX)


def check_square(bound):
    rows = fails = seen = 0
    for q in primes(5, bound):
        if not SquareFold.applies(q):
            continue
        seen += 1
        _, room, base = run_square(q, 1 << 20)
        _, _, plain = run_square(q, 0)
        for keep in sorted({0, 1, 2, 3, room // 2, room - 1, room, room + 5} - {-1}):
            bad, kept, ccx = run_square(q, keep)
            rows += 1
            fails += bad
            if plain - ccx != kept or kept != min(keep, room):
                fails += 1
                print(json.dumps({"q": q, "keep": keep, "kept": kept, "ccx": ccx, "plain": plain, "error": "saving is not 1 per wire"}))
            if bad:
                print(json.dumps({"q": q, "keep": keep, "failures": bad}))
    return {"check": "square compute", "primes": seen, "rows": rows, "failures": fails}


def control(r, q=251, mutants=60):
    """CCX-to-CX mutants of M with every junk wire in use. A mutant counts as seen when the
    forward or the backward run fails on some lane."""
    capacity = DialogMultiplier(Circuit(), Mod(q)).junk_capacity
    C = build_m(q, capacity)[0]
    cases = [(x, r.randrange(q)) for x in range(1, q)]
    sites = [i for i in range(len(C.ops) // 4) if C.ops[4 * i] == CCX]
    seen = 0
    for i in r.sample(sites, mutants):
        ops = array("i", C.ops)
        a, b, c = ops[4 * i + 1:4 * i + 4]
        ops[4 * i:4 * i + 4] = array("i", (CX, a, c, c))
        forward, backward, _, _ = run_m(q, capacity, cases, ops)
        seen += bool(forward or backward)
    return {"check": "control", "q": q, "mutants": mutants, "seen": seen}


def main():
    m_bound = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    sq_bound = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
    r = random.Random(20261109)
    out = [check_m(m_bound, r), check_square(sq_bound), control(r)]
    for row in out:
        print(json.dumps(row))
    ok = out[0]["failures"] == 0 and out[1]["failures"] == 0 and out[2]["seen"] >= out[2]["mutants"] * 9 // 10
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
