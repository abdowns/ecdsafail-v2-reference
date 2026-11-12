"""Integer model of the K-descent: clear (a, b) = (floor(U*y/q), floor(V*y/q)) from D = V*a - U*b by
walking the cells backward. Random cell sequences, random and edge y."""
import random, sys
random.seed(1)

def ascend(term, cells):
    U, V = 1, term
    levels = [(U, V)]
    for g0, bb in cells:
        if g0:
            V = 2 * V + U
            if bb:
                U, V = V, U
        elif bb:
            V = 4 * V
        else:
            V = 2 * V
        levels.append((U, V))
    return levels

def descend(levels, cells, a, b):
    U, V = levels[-1]
    D = V * a - U * b
    digits = []
    stats = [0, 0]
    for k in range(len(cells) - 1, -1, -1):
        g0, bb = cells[k]
        Up, Vp = levels[k]
        sw = g0 & bb
        if sw:
            a, b = b, a
            D = -D
        if g0:
            d0 = D & 1
            D1 = (D + Up * d0) >> 1
            assert (D + Up * d0) % 2 == 0
            d1 = (1 - d0) & (D1 < 0)
            D = D1 + Up * d1
            t = b - a - d0 - 2 * d1
            assert t % 2 == 0 and t >= 0, ("add", t)
            b = t // 2
            stats[d1] += 1
        elif bb:
            d0 = D & 1
            D1 = (D + Up * d0) >> 1
            d1 = D1 & 1
            D = (D1 + Up * d1) >> 1
            assert b % 4 == d0 + 2 * d1, "J digits"
            b >>= 2
        else:
            d0 = D & 1
            d1 = 0
            D = (D + Up * d0) >> 1
            assert b % 2 == d0, "B1 digit"
            b >>= 1
        digits.append((d0, d1))
        assert D == Vp * a - Up * b, "D tracks"
        assert 0 <= a < Up and b >= 0, "box"
        assert (-Vp < D < Up) or (Vp == 0 and D == 0), ("range", D, Up, Vp)
    assert a == 0 and b == 0 and D == 0, ("end", a, b, D)
    return stats

def trial(q, J):
    term = random.randint(0, 1)
    cells = []
    for _ in range(J - 1):
        r = random.random()
        cells.append((1, 0) if r < .3 else (1, 1) if r < .6 else (0, 1) if r < .8 else (0, 0))
    levels = ascend(term, cells)
    U, V = levels[-1]
    ys = [0, 1, q - 1, q // 2, q // 3] + [random.randrange(q) for _ in range(40)]
    # y on both sides of every breakpoint region: multiples of q/U and q/V
    for d in (U, V):
        if d:
            for _ in range(10):
                i = random.randrange(d)
                ys += [max(0, min(q - 1, i * q // d + e)) for e in (-1, 0, 1)]
    tot = [0, 0]
    for y in ys:
        s = descend(levels, cells, U * y // q, V * y // q)
        tot[0] += s[0]; tot[1] += s[1]
    return tot

tot = [0, 0]
for q in (61, 251, 4093, 65521, 2**256 - 2**32 - 977):
    for J in (2, 3, 5, 9, 14, 22, 40):
        for _ in range(300):
            s = trial(q, J); tot[0] += s[0]; tot[1] += s[1]
print("ok; add cells with d1=0:", tot[0], "d1=1:", tot[1])
# exhaustive in y at small q
for q in (61, 251):
    for _ in range(400):
        J = random.randint(2, 8)
        term = random.randint(0, 1)
        cells = [random.choice([(1, 0), (1, 1), (0, 1), (0, 0)]) for _ in range(J - 1)]
        levels = ascend(term, cells)
        U, V = levels[-1]
        for y in range(q):
            descend(levels, cells, U * y // q, V * y // q)
print("exhaustive small q ok")
