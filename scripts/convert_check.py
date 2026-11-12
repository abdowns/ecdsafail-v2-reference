"""convert_check.py J [lanes] [seed] [mutant] : RankOne.convert alone at secp256k1.

Random secp256k1 inputs never reach the conversion with real cells: their Euclid passes end about sixty
ticks before the bound, so the last J cells are padding. This check feeds the conversion directly. Each lane
is a terminal bit and J - 1 cells whose cofactors stay inside the width bound at every level, a y, and the
kept copies the Euclid pass would have left for those cells. It expects (U*y, V*y) -> (U*y mod q, V*y mod q),
the cells and the kept copies as found, and every other wire 0.

The cell sequences come in four styles (uniform, largest cofactors, mostly adds, padded). The y are edge
values, quotient boundaries j*q/d, quotients next to a multiple of a power of two (long carries in the
descent), quotients k with 977*k just above a multiple of a power of two (long borrows in the fold), and
uniform values.

mutant dropK removes the K-th CCX of the flat conversion. callK removes the K-th CCX the builder emits, so a
gate inside the recorded descent is lost in both of its directions. A mutant must fail on some lane."""
import sys, time, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qec import sim, rank_one, model
from qec.builder import Circuit

J = int(sys.argv[1]); LANES = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1
mutant = sys.argv[4] if len(sys.argv) > 4 else ""
q = 2**256 - 2**32 - 977
n = q.bit_length()
T = model.tick_bound(q)
widths = model.tick_widths(q, T)
steps = [1] + [widths[T - k] for k in range(1, J + 1)]
w = widths[T - J]
rng = random.Random(seed)
TYPES = [(1, 0), (1, 1), (0, 1), (0, 0)]
CALLS = [0]
if mutant.startswith("call"):
    _k = int(mutant[4:]); _ccx = Circuit.ccx
    def _ccx2(self, a, b, c):
        CALLS[0] += 1
        if CALLS[0] - 1 != _k: _ccx(self, a, b, c)
        else:
            import traceback
            print("DROPPED", [(f.name, f.lineno) for f in traceback.extract_stack()[-7:-1]], flush=True)
    Circuit.ccx = _ccx2

def step(U, V, c):
    g0, b = c
    if g0:
        V = 2 * V + U
        return (V, U) if b else (U, V)
    return (U, 4 * V) if b else (U, 2 * V)

def seq(style):
    """term and J - 1 cells with every level inside its bound; style biases the choice."""
    while True:
        term = rng.randint(0, 1) if style != "pad" else 0
        U, V = 1, term
        cells = []
        for k in range(2, J + 1):
            opts = []
            for c in TYPES:
                u, v = step(U, V, c)
                if u < 1 << min(w, steps[k]) and v < 1 << min(w, steps[k]):
                    opts.append((c, u, v))
            if not opts:
                break
            if style == "big":
                opts.sort(key=lambda o: -max(o[1], o[2])); c, U, V = opts[0] if rng.random() < .8 else rng.choice(opts)
            elif style == "add":
                pick = [o for o in opts if o[0][0]] or opts; c, U, V = rng.choice(pick)
            elif style == "pad":
                c, U, V = ((0, 1), U, 4 * V) if (V == 0 and rng.random() < .7) else rng.choice(opts)
            else:
                c, U, V = rng.choice(opts)
            cells.append(c)
        else:
            return term, cells, U, V

def ys(U, V):
    r = rng.random()
    if r < .1:
        return rng.choice([0, 1, 2, q - 1, q - 2, q // 2, q // 2 + 1])
    if r < .5:
        d = rng.choice([U, V, U + V]) or 1
        return max(0, min(q - 1, rng.randrange(d) * q // d + rng.choice([-1, 0, 1, 2])))
    if r < .75:
        # a quotient next to a multiple of a power of two, so the carry chains run long
        d = rng.choice([x for x in (U, V) if x] or [1])
        i = rng.randint(1, d.bit_length())
        c = rng.randint(1, max(1, d >> i))
        hi0 = max(0, (c << i) - 1 + rng.choice([-1, 0, 0, 1]))
        return max(0, min(q - 1, ((hi0 << 256) + rng.randrange(1 << 256)) // d))
    if r < .9:
        # a quotient k with 977*k or f*k just above a multiple of a power of two: the fold's borrow runs long
        d = rng.choice([x for x in (U, V) if x] or [1])
        flo = rng.choice([977, (1 << 32) + 977])
        i = rng.randint(10, max(10, (flo * d).bit_length() - 1))
        j = rng.randint(1, max(1, flo * d >> i))
        k = -(-(j << i) // flo) + rng.choice([-1, 0, 0, 0, 1])
        hi0 = max(0, min(d - 1, k - 1))
        return max(0, min(q - 1, ((hi0 << 256) + rng.randrange(1 << 256)) // d))
    return rng.randrange(q)

ro = rank_one.RankOne(q, J, w, steps)
KEPT = True
C = Circuit()
term = C.alloc(1)
cw = [C.alloc(2) for _ in range(J - 1)]
R, S = C.alloc(ro.L), C.alloc(ro.L)
Ri, Si = list(R), list(S)
keep, glow = (C.alloc(ro.kept_size()), C.alloc(1)) if KEPT else ([], [])
if KEPT:
    ro.convert(C, term[0], cw, R, S, keep, glow[0])
else:
    ro.convert(C, term[0], cw, R, S)
ops, nw = C.ops, C.num_wires

def levels(tm, cells):
    U, V = 1, tm
    out = [None, (U, V)]
    for c in cells:
        U, V = step(U, V, c)
        out.append((U, V))
    return out

def kept_value(tm, cells, U, V):
    """(keep wires' value, glow): stash, U_J >> 1, stored V and offset consistent with the swept V."""
    lv = levels(tm, cells)
    val, at = 0, 0
    for k in range(2, J):
        val |= (lv[k][0] >> 1) << at
        at += ro.m(k) - 1
    val |= (U >> 1) << at
    at += w - 1
    tz = (V & -V).bit_length() - 1 if V else 0
    opts = [(1, V)]
    if V == 0 or tz % 2 == 0:
        opts.append((0, V))
        if tz >= 2:
            opts.append((0, V >> 1))
    g, v = rng.choice(opts)
    E = rng.randint(0, 1) if v else 0
    val |= (v - E) << at
    at += w
    val |= E << at
    return val, g
if mutant.startswith("drop"):
    # drop the k-th CCX
    k = int(mutant[4:]); idx = [i for i in range(0, len(ops), 4) if ops[i] == 2][k]
    ops = ops[:idx] + ops[idx + 4:]
keep_w = set(term) | {x for c in cw for x in c} | set(R) | set(S) | set(keep) | set(glow)
others = [x for x in range(nw) if x not in keep_w]
print(f"J={J} w={w} steps={steps} ccx={ops[0::4].count(2)} wires={nw}", flush=True)
fails = 0; first = None; t = time.time(); stats = {}
styles = ["rand", "big", "add", "pad"]
lanes_in = []
for i in range(LANES):
    st = styles[i % 4]
    tm, cells, U, V = seq(st)
    y = ys(U, V)
    lanes_in.append((tm, cells, U, V, y) + (kept_value(tm, cells, U, V) if KEPT else (0, 0)))
init = {}
init.update(sim.pack([l[0] for l in lanes_in], term))
for k in range(J - 1):
    init.update(sim.pack([l[1][k][0] | l[1][k][1] << 1 for l in lanes_in], cw[k]))
init.update(sim.pack([l[2] * l[4] for l in lanes_in], Ri))
init.update(sim.pack([l[3] * l[4] for l in lanes_in], Si))
if KEPT:
    init.update(sim.pack([l[5] for l in lanes_in], keep))
    init.update(sim.pack([l[6] for l in lanes_in], glow))
state = sim.run(ops, nw, init, LANES)
gk = sim.unpack(state, keep, LANES) if KEPT else None
gg = sim.unpack(state, glow, LANES) if KEPT else None
gr, gs = sim.unpack(state, R, LANES), sim.unpack(state, S, LANES)
gt = sim.unpack(state, term, LANES)
gc = [sim.unpack(state, cw[k], LANES) for k in range(J - 1)]
dirty = 0
for x in others:
    dirty |= state[x]
for j, (tm, cells, U, V, y, kv, gl) in enumerate(lanes_in):
    ok = gr[j] == U * y % q and gs[j] == V * y % q and gt[j] == tm and not dirty >> j & 1
    ok = ok and (not KEPT or (gk[j] == kv and gg[j] == gl))
    ok = ok and all(gc[k][j] == cells[k][0] | cells[k][1] << 1 for k in range(J - 1))
    if not ok:
        fails += 1
        first = first or (tm, cells, U, V, hex(y))
print(f"calls={CALLS[0]} lanes={LANES} failures={fails} first={first} maxU={max(l[2] for l in lanes_in)} maxV={max(l[3] for l in lanes_in)} {time.time()-t:.0f}s")
sys.exit(1 if fails else 0)
