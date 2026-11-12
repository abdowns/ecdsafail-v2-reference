"""m_exhaustive.py q [q ...] : M(x, y) = (x, x*y mod q) on every 1 <= x < q and every y, every other wire 0."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qec import sim, rank_one
from qec.arith import Mod
from qec.builder import Circuit
from qec.point_add import DialogMultiplier

seen = {}
orig = DialogMultiplier._rank_start
def rs(self, G, Yr, S, J, *a):
    seen["J"] = J
    return orig(self, G, Yr, S, J, *a)
DialogMultiplier._rank_start = rs

def check(q, batch=1 << 14):
    mod = Mod(q)
    n = q.bit_length()
    C = Circuit()
    x, y = C.alloc(n), C.alloc(n)
    xi, yi = list(x), list(y)
    seen.clear()
    DialogMultiplier(C, mod).apply(x, y)
    ops, nw = C.ops, C.num_wires
    live = set(x) | set(y)
    others = [w for w in range(nw) if w not in live]
    fails = lanes = 0
    first = None
    t = time.time()
    pairs = [(a, b) for a in range(1, q) for b in range(q)]
    for i in range(0, len(pairs), batch):
        chunk = pairs[i:i + batch]
        init = {}
        init.update(sim.pack([p[0] for p in chunk], xi))
        init.update(sim.pack([p[1] for p in chunk], yi))
        st = sim.run(ops, nw, init, len(chunk))
        gx, gy = sim.unpack(st, x, len(chunk)), sim.unpack(st, y, len(chunk))
        dirty = 0
        for w in others:
            dirty |= st[w]
        for j, (a, b) in enumerate(chunk):
            if gx[j] != a or gy[j] != a * b % q or dirty >> j & 1:
                fails += 1
                first = first or (a, b, gx[j], gy[j], dirty >> j & 1)
        lanes += len(chunk)
    print(f"q={q} J={seen.get('J')} ccx={ops[0::4].count(2)} wires={nw} lanes={lanes} failures={fails} first={first} {time.time()-t:.0f}s", flush=True)
    return fails

bad = sum(check(int(a)) for a in sys.argv[1:])
sys.exit(1 if bad else 0)
