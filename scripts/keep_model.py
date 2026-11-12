"""Model check of the kept copies: the Euclid pass's (U, V, E) before tick T - k against the cofactors
after k cells of the swept tape, at every level of every x for eight small primes and of 3,000 x at secp256k1.
U_k = U, and V_k = (V + E) << d with d = [the cell below is a shift] AND [V + E has an odd number of trailing zeros]."""
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qec import model

def swept(tape):
    """B1 to the lowest index of its shift run: step k swaps b_k and b_(k-1) when both are shifts, k from the top."""
    t = [list(c) for c in tape]
    for k in range(len(t) - 1, 0, -1):
        if t[k][0] == 0 and t[k - 1][0] == 0:
            t[k][1], t[k - 1][1] = t[k - 1][1], t[k][1]
    return t

def cof(cells_top_down):
    U, V = 1, 0
    out = [(U, V)]
    for i, (g0, b) in enumerate(cells_top_down):
        if i == 0:
            V = g0
        elif g0:
            V = 2 * V + U
            if b: U, V = V, U
        else:
            V = 4 * V if b else 2 * V
        out.append((U, V))
    return out

rel = {}
def check(q, x):
    T = model.tick_bound(q)
    states, _ = model.tick_pass(q, x, T)
    u, v, tape = model.tick_tape(q, x, T)
    assert (u, v) == (1, 0)
    sw = swept(tape)
    levels = cof(sw[::-1])           # levels[k] after cells T-1 .. T-k
    for k in range(1, T):
        i = T - k
        U, V, E = states[i]
        Uk, Vk = levels[k]
        # delta: swept cell i-1 is B1 and swept cell i is a shift
        v = V + E
        tz = (v & -v).bit_length() - 1 if v else 0
        d = int(i >= 1 and sw[i - 1][0] == 0 and tz % 2 == 1)
        ok = (U == Uk, (V + E) << d == Vk)
        rel[ok] = rel.get(ok, 0) + 1
        if ok != (True, True) and len(bad) < 5:
            bad.append((q, x, k, (U, V, E), (Uk, Vk), d, sw[i - 1], sw[i]))
bad = []
for q in (61, 127, 251, 509, 1021, 2039, 4093, 8191):
    for x in range(1, q):
        check(q, x)
q = 2**256 - 2**32 - 977
rng = random.Random(3)
for _ in range(3000):
    check(q, rng.randrange(1, q))
print(rel); print(bad[:5])
sys.exit(1 if bad or set(rel) != {(True, True)} else 0)
