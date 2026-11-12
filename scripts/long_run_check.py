"""long_run_check.py [mmax] [lanes] [dropK] : M(x, y) at secp256k1 on x whose Euclid pass runs into the rank-one
window (the last RANK_J cells), so that the window holds real cells. Random x never do: the longest run in 100,000 is
about 325 ticks of 384. The x are m*2^k and q - m*2^k for odd m below mmax. dropK removes the K-th CCX emitted
inside RankOne.convert, as a control that these inputs reach the conversion."""
import sys, time, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qec import sim, rank_one
from qec.arith import Mod
from qec.builder import Circuit
from qec.point_add import DialogMultiplier
MM = int(sys.argv[1]) if len(sys.argv) > 1 else 600
LANES = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
drop = int(sys.argv[3]) if len(sys.argv) > 3 else -1
q = 2**256 - 2**32 - 977
n = 256
rng = random.Random(5)
# the first tick count that reaches the window
LOW = 384 - DialogMultiplier.RANK_J + 1
t = time.time()

def ticks(x):
    u, v, k = q, x, 0
    while v:
        if v & 1:
            u, v = (v, (u - v) >> 1) if u > v else (u, (v - u) >> 1)
        else:
            v >>= 2 if v & 3 == 0 else 1
        k += 1
    return k

found = {}
for m in range(1, MM, 2):
    k = 0
    while m << k < q:
        for x in (m << k, q - (m << k)):
            if 0 < x < q and x not in found:
                c = ticks(x)
                if c >= LOW:
                    found[x] = c
        k += 1
pool = sorted(((c, x) for x, c in found.items()), reverse=True)
rng.shuffle(pool)
pool = pool[:LANES]
xs = [x for _, x in pool]
LANES = len(xs)
ys = [rng.choice([0, 1, q - 1, q // 2]) if i % 9 == 0 else rng.randrange(q) for i in range(LANES)]
state = {"in": 0, "k": 0}
if drop >= 0:
    conv, ccx = rank_one.RankOne.convert, Circuit.ccx
    def convert(self, *a, **kw):
        state["in"] += 1
        try: return conv(self, *a, **kw)
        finally: state["in"] -= 1
    def ccx2(self, a, b, c):
        if state["in"]:
            state["k"] += 1
            if state["k"] - 1 == drop: return
        ccx(self, a, b, c)
    rank_one.RankOne.convert, Circuit.ccx = convert, ccx2
C = Circuit()
x, y = C.alloc(n), C.alloc(n)
xi, yi = list(x), list(y)
M = DialogMultiplier(C, Mod(q))
M.apply(x, y)
ops, nw = C.ops, C.num_wires
init = {}
init.update(sim.pack(xs, xi)); init.update(sim.pack(ys, yi))
st = sim.run(ops, nw, init, LANES)
gx, gy = sim.unpack(st, x, LANES), sim.unpack(st, y, LANES)
live = set(x) | set(y)
dirty = 0
for w_ in range(nw):
    if w_ not in live: dirty |= st[w_]
fails = [i for i in range(LANES) if gx[i] != xs[i] or gy[i] != xs[i] * ys[i] % q or dirty >> i & 1]
tk = sorted(k for k, _ in pool)
print(f"J={M.rank_j} ccx={ops[0::4].count(2)} lanes={LANES} ticks {tk[0]}..{tk[-1]} drop={drop} convert_ccx_calls={state['k']} failures={len(fails)} {time.time()-t:.0f}s")
sys.exit(1 if fails else 0)
