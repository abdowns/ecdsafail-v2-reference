"""The rank-one start of a Bezout pass: its first J cells on unreduced integers.

The pass starts at (R, S) = (y, 0), so after its first j cells (R, S) = (U*y, V*y) for the integer
cofactor pair (U, V) of those cells, which is the Euclid pair before Euclid's last j ticks, so
U, V < 2^w with w = tick_widths(q, T)[T - j]. Unreduced, a tick is its add-or-shift and its swap:
no Prefix, low chains, compare or k machinery. One conversion after cell J reduces both registers:

  fold      X = hi*2^n + lo -> (X mod q, m = floor(X/q)), for q = 2^n - f with (2^w + 2)f <= 2^n:
            m = hi + e, e = [lo + (hi + 1)f >= 2^n], X mod q = lo + m*f mod 2^n.
            The quotients are a = floor(U*y/q) from R and b = floor(V*y/q) from S.
  copies    the cofactors are not built here. The Euclid pass copies U before each of its last J
            ticks, and V and its offset before the first of them (keep_state), and the inverse
            Euclid pass clears the copies.
  D         V*a - U*b = D = (U*s - V*r)/q, in (-V, U) when V > 0 and 0 when V = 0, so |D| < 2^w:
            on w + 1 fresh wires, mod 2^(w + 1)
  descent   level k is the state after k cells. D's cells walk from level J to level 1 and leave two
            digits a cell (d_descent); the quotients' cells read the digits and end at (0, 0)
            (k_descent); D's cells undone give D back.
  clear     D*q = U*s - V*r, so -q*D + U*s - V*r = 0 mod 2^(w + 1) clears D from r's and s's low bits.

The descent's invariant, at every level: 0 <= a < U, and D = V*a - U*b in (-V, U), or 0 when V = 0.
Level 1 has (U, V) = (1, g0) with g0 the terminal cell's bit, so there (a, b, D) = (0, 0, 0).
A cell undone takes level k + 1 to level k, with U = U_k odd:

  swap      on an add with b: (a, b) trade places and D -> -D
  d0        D mod 2, then D -> (D + d0*U)/2
  d1        add:  [D < 0] when d0 = 0, then D -> D + d1*U
            J:    D mod 2, then D -> (D + d1*U)/2
            B1:   0
  b         (b - g0*a - d0 - 2*d1)/2, and half of that again on J

Each line keeps D = V*a - U*b and lands in the level's range. On an add the cofactor above is
V' = 2V + U, so D' lies in (-2V - U, U): an odd D' takes d0 = 1 and lands in (-V, U); an even D'
halves into (-V - U/2, U/2), and adding U when that is negative lands in (-V, U). On a shift the
cofactor above is 2V or 4V and each halving lands in the range one level down. The digits are a
choice, not the true floors one level down: any choice that keeps the invariant ends at (0, 0, 0).

The middle cells use the tree's add-or-shift and held offset on wider words. The last cell is
semantic (a B1 doubles, adding nothing), so the reduced tick after it starts with no held lane."""
from itertools import permutations

from .arith import _maj, _uma, add_const, cswap
from .two_bit import Prefix, TwoBitMod


def plan(q, T, widths, J_max):
    """The largest J <= J_max the fold allows, or 0."""
    n = q.bit_length()
    f = (1 << n) - q
    for J in range(min(J_max, T - 2), 2, -1):
        w = widths[T - J]
        if (2 ** w + 2) * f <= 2 ** n and ((2 ** w) * f).bit_length() <= n:
            return J
    return 0


def _uma_ctl(C, ctl, c, b, a):
    C.ccx(c, b, a)
    C.cx(a, c)
    C.cx(a, b)
    C.cx(c, a)
    C.ccx(ctl, a, b)
    C.cx(c, a)


def add(C, a, b, ctl=None, cin=None):
    """b += a (or ctl*a) mod 2^len(b), a zero-extended. 2 CCX a bit, 3 under a control. cin is a
    carry-in wire, kept: b += a + cin, or ctl*(a + cin)."""
    n = len(b)
    a = list(a)
    ext = C.alloc(n - len(a)) if len(a) < n else []
    a = a + ext
    c0 = C.alloc(1)[0] if cin is None else cin
    carry = [c0] + a[:-1]
    for i in range(n - 1):
        _maj(C, carry[i], b[i], a[i])
    i = n - 1
    C.cx(carry[i], a[i])
    if ctl is None:
        C.cx(a[i], b[i])
    else:
        C.ccx(ctl, a[i], b[i])
    C.cx(carry[i], a[i])
    for i in reversed(range(n - 1)):
        if ctl is None:
            _uma(C, carry[i], b[i], a[i])
        else:
            _uma_ctl(C, ctl, carry[i], b[i], a[i])
    C.free(([c0] if cin is None else []) + ext)


def sub(C, a, b, ctl=None, cin=None):
    for w in b:
        C.x(w)
    add(C, a, b, ctl, cin)
    for w in b:
        C.x(w)


def _shift_down(C, ctl, x):
    """Under ctl, x >> 1 with x[0] (0 then) moved to the top."""
    for i in range(1, len(x)):
        C.cx(x[i - 1], x[i])
        C.ccx(ctl, x[i], x[i - 1])
        C.cx(x[i - 1], x[i])


def carry_into(C, a, b, flag, ctl=None):
    """flag ^= [a + b >= 2^len(b)] (times ctl), a and b as found."""
    n = len(b)
    a = list(a)
    ext = C.alloc(n - len(a)) if len(a) < n else []
    a = a + ext
    (c0,) = C.alloc(1)
    carry = [c0] + a[:-1]
    for i in range(n):
        _maj(C, carry[i], b[i], a[i])
    if ctl is None:
        C.cx(a[n - 1], flag)
    else:
        C.ccx(ctl, a[n - 1], flag)
    for i in reversed(range(n)):
        C.ccx(carry[i], b[i], a[i])
        C.cx(a[i], carry[i])
        C.cx(a[i], b[i])
    C.free([c0] + ext)


def inc(C, x, ctl):
    (one,) = C.alloc(1)
    C.cx(ctl, one)
    add(C, [one], x)
    C.cx(ctl, one)
    C.free([one])


def dec(C, x, ctl):
    (one,) = C.alloc(1)
    C.cx(ctl, one)
    sub(C, [one], x)
    C.cx(ctl, one)
    C.free([one])


def mul_acc(C, a, b, acc, sign=1, odd=False):
    """acc += a*b (sign -1: acc -= a*b) mod 2^len(acc), row i on acc[i:]. With odd, a holds the bits
    above the low 1 of an odd factor, and row 0 adds b with no control."""
    if sign < 0:
        for w in acc:
            C.x(w)
    if odd:
        add(C, b[:len(acc)], acc)
    for i, ai in enumerate(a[:len(acc) - odd], int(odd)):
        add(C, b[:len(acc) - i], acc[i:], ctl=ai)
    if sign < 0:
        for w in acc:
            C.x(w)


def mul_odd_const(C, x, c):
    """x = c*x mod 2^len(x) in place, c an odd constant: x[i + 1:] += x_i*(c >> 1), i from the top down."""
    k = len(x)
    for i in range(k - 2, -1, -1):
        part = c >> 1 & ((1 << (k - i - 1)) - 1)
        if part:
            add_const(C, x[i + 1:], part, ctrl=x[i])


def _csd(c):
    out, i = [], 0
    while c:
        if c & 1:
            d = 2 - (c & 3)
            out.append((i, d))
            c -= d
        c >>= 1
        i += 1
    return out


def _times_plan(c, W, top, nx):
    """The order of c's signed digits and the top bit of each row's add, for x <= top on nx wires.
    The first digit is a copy. While every partial sum is a positive multiple of x under 2^W, a row
    stops at the top bit of the larger of the sums around it. The order with the fewest add bits
    wins, then the one that zero-extends x the least."""
    terms = [t for t in _csd(c) if t[0] < W]

    def rows(order):
        coef, exact, ends = 1 << order[0][0], True, []
        for s, d in order[1:]:
            new = coef + d * (1 << s)
            exact = exact and new > 0 and max(coef, new) * top < 1 << W
            ends.append((max(coef, new) * top).bit_length() if exact else W)
            coef = new
        return ends

    def cost(order):
        ends = rows(order)
        bits = [e - s for (s, _), e in zip(order[1:], ends)]
        return sum(bits), max([b - nx for b in bits] + [0])

    orders = [terms]
    if len(terms) <= 7:
        orders = [list(o) for o in permutations(terms) if o[0][1] > 0]
    order = min(orders, key=cost)
    if order[0][1] < 0:
        order = sorted(order, key=lambda t: t[1] < 0)
    return order, rows(order)


def _times(C, x, out, c, inverse, top):
    """out += c*x mod 2^len(out), out 0 before and x <= top: a copy at one +1 digit of c, an add or
    subtract at each other, as _times_plan orders and bounds them."""
    W = len(out)
    order, ends = _times_plan(c, W, top, len(x))
    s0 = order[0][0]
    rest = list(zip(order[1:], ends))
    if not inverse:
        for i, w in enumerate(x[:W - s0]):
            C.cx(w, out[s0 + i])
        for (s, d), e in rest:
            (add if d > 0 else sub)(C, x[:e - s], out[s:e])
    else:
        for (s, d), e in reversed(rest):
            (sub if d > 0 else add)(C, x[:e - s], out[s:e])
        for i, w in enumerate(x[:W - s0]):
            C.cx(w, out[s0 + i])


def split(c, top):
    """(c_lo, t) when c = 2^t + c_lo with c_lo*x < 2^t for every x <= top, else None. Then c*x is
    c_lo*x on its own wires and x itself at t."""
    t = c.bit_length() - 1
    lo_c = c - (1 << t)
    return (lo_c, t) if lo_c * top < 1 << t else None


class RankOne:
    def __init__(self, q, J, w, steps=None):
        """steps[k] bounds the cofactors' bits after k cells, steps[J] = w."""
        self.q, self.J, self.w = q, J, w
        self.steps = steps or list(range(w - J, w + 1))
        # each level's cofactors are one bit wider than the level's below
        assert all(self.steps[k + 1] == self.steps[k] + 1 for k in range(1, J)) and self.steps[J] == w
        self.n = n = q.bit_length()
        self.f = (1 << n) - q
        self.W = ((2 ** w) * self.f).bit_length()
        # R and S through the unreduced cells: below 2^(w + 1)*q on a held lane, and two top wires
        # that the add-or-shift's word needs
        self.L = n + w + 3
        self.nar = TwoBitMod(q)

    # -- the unreduced cells -----------------------------------------------------------------------

    def tick(self, C, g, prev, p, R, S):
        """A middle cell on the held offset, as two_bit's tick without its reduction: the word takes
        R >> 1 off J (under a complement on a held lane) or doubles on J, then S = 2*word + g0*R0.
        Returns [this cell is B1] on the sw wire."""
        g0, b = g
        (sw,) = C.alloc(1)
        C.ccx(g0, b, sw)
        C.cx(sw, b)
        if p is not None:
            for x in S:
                C.cx(p, x)
        shim = _Width(len(R) - 2)
        TwoBitMod._add_or_shift(shim, C, b, R[:len(R) - 2], S, held_lane=p is not None)
        if p is not None:
            for x in S:
                C.cx(p, x)
            self.nar._drop_after_b1(C, prev, p)
        (l0,) = C.alloc(1)
        C.ccx(g0, R[0], l0)
        top = S[-1]
        S[:] = [l0] + S[:-1]
        C.free([top])
        C.cx(sw, b)
        cswap(C, sw, R, S)
        C.x(sw)
        C.cx(g0, sw)
        C.cx(b, sw)
        return sw

    def last_tick(self, C, g, prev, p, R, S):
        """The segment's last cell, semantic: 2S + R, 4S on J, and 2S on B1 with nothing added. On a
        held lane (g0 = 1) the word is H - R0 for H = S + R, and the complement makes 2H - R."""
        g0, b = g
        if p is not None:
            for x in S:
                C.cx(p, x)
        add(C, R[1:], S, ctl=g0)
        if p is not None:
            for x in S:
                C.cx(p, x)
            self.nar._drop_after_b1(C, prev, p)
        (j,) = C.alloc(1)
        C.x(g0)
        C.ccx(g0, b, j)
        C.x(g0)
        for i in range(len(S) - 1, 0, -1):
            C.cx(S[i - 1], S[i])
            C.ccx(j, S[i], S[i - 1])
            C.cx(S[i - 1], S[i])
        C.x(g0)
        C.ccx(g0, b, j)
        C.x(g0)
        C.free([j])
        (l0,) = C.alloc(1)
        C.ccx(g0, R[0], l0)
        top = S[-1]
        S[:] = [l0] + S[:-1]
        C.free([top])
        (sw,) = C.alloc(1)
        C.ccx(g0, b, sw)
        cswap(C, sw, R, S)
        C.ccx(g0, b, sw)
        C.free([sw])

    # -- the conversion ------------------------------------------------------------------------------

    def fold(self, C, X):
        """X (n + w wires) -> X mod q on X[:n] and floor(X/q) on X[n:]. The probe finds
        e = [lo + (hi + 1)f >= 2^n] against Prefix's AND of the high bits; the write adds m*f and its
        window carry clears e and moves the high bits."""
        n, f, W = self.n, self.f, self.W
        lo = X[:n]
        spare = C.alloc(1)
        hi = X[n:] + spare
        H = lo[W:]
        pre = Prefix(C, H) if H else None
        (one,) = C.alloc(1)
        C.x(one)
        inc(C, hi, one)
        C.x(one)
        C.free([one])
        # hi + 1 <= 2^w, on w + 1 wires
        top = 1 << len(hi) - 1
        cut = split(f, top)
        if cut:
            lo_c, t = cut
            low = C.alloc((lo_c * top).bit_length())
            own = low + C.alloc(t - len(low))
            prod = own + hi
            assert len(prod) == W
        else:
            lo_c, low = f, C.alloc(W)
            own = prod = low

        def times(inverse=False):
            if low:
                _times(C, hi, low, lo_c, inverse, top)

        times()
        (e,) = C.alloc(1)
        (c1,) = C.alloc(1)
        carry_into(C, prod, lo[:W], c1)
        if H:
            C.ccx(pre.top, c1, e)
        else:
            C.cx(c1, e)
        carry_into(C, prod, lo[:W], c1)
        C.free([c1])
        # off e, m = hi - 1 and the product loses lo_c on its own wires (it was lo_c*(m + 1) there)
        C.x(e)
        dec(C, hi, e)
        if low:
            for x in low:
                C.x(x)
            add_const(C, low, lo_c, ctrl=e)
            for x in low:
                C.x(x)
        C.x(e)
        a, b = list(prod), lo[:W]
        (c0,) = C.alloc(1)
        carry = [c0] + a[:-1]
        for i in range(W):
            _maj(C, carry[i], b[i], a[i])
        cw = a[W - 1]
        if H:
            C.ccx(pre.top, cw, e)
            pre.write(cw)
        else:
            C.cx(cw, e)
        for i in reversed(range(W)):
            _uma(C, carry[i], b[i], a[i])
        C.free([c0, e])
        times(inverse=True)
        C.free(own + spare)

    # -- the descent ---------------------------------------------------------------------------------
    #
    # Level k is the state after k cells: cofactors (U_k, V_k) below 2^m(k), quotients (a, b) with
    # 0 <= a < U_k, and D = V_k*a - U_k*b in (-V_k, U_k), or 0 when V_k = 0. Cell k + 1 undone:
    #   swap     (a, b) and D -> -D under sw
    #   d0       D mod 2, D -> (D + d0*U_k)/2
    #   d1       add: [D < 0] off d0, D -> D + d1*U_k.  J: D mod 2, D -> (D + d1*U_k)/2.  B1: 0
    #   b        (b - g0*a - d0 - 2*d1)/2, and /4 on J
    # which keeps D = V*a - U*b and the ranges, so level 1 is (a, b, D) = (0, 0, 0). D's cells make
    # the digits, the quotients' cells read them, and D's cells undone give D back.

    def m(self, k):
        return min(self.w, self.steps[k])

    def stash_at(self, stash, k):
        """U_k's bits above its low 1, for 2 <= k < J."""
        at = sum(self.m(i) - 1 for i in range(2, k))
        return stash[at:at + self.m(k) - 1]

    def stash_size(self):
        return sum(self.m(i) - 1 for i in range(2, self.J))

    def pool_size(self):
        """Zero wires the D cells borrow beside D's own: each takes two and gives one back."""
        return self.J

    def _d_cell(self, C, g0, b, Uk, D, pool, m):
        """D (level k + 1, m + 2 wires, two's complement) -> (D at level k on m + 1 wires, d0, d1).
        Uk is U_k."""
        assert len(D) == m + 2
        (sw,) = C.alloc(1)
        C.ccx(g0, b, sw)
        # -D = (D ^ sw) + sw; d0 = D mod 2 leaves D on its own low wire
        for x in D:
            C.cx(sw, x)
        C.cx(sw, D[0])
        d0, X = D[0], D[1:]
        # (D + d0*U)/2 = (D >> 1) + d0*(U >> 1) + [d0 or sw], U odd. It lies in (-V - U/2, U), inside
        # X's m + 1 wires: the cofactors above are below 2^(m + 1) and U is below 2^m.
        # d0*(U >> 1) rests on the pool's idle wires, and on fresh ones where the pool is short
        idle = pool[:len(Uk) - 1]
        fresh = C.alloc(len(Uk) - 1 - len(idle))
        mask = idle + fresh
        for u, t in zip(Uk[1:], mask):
            C.ccx(d0, u, t)
        (cin,) = C.alloc(1)
        C.cx(d0, cin)
        C.cx(sw, cin)
        C.ccx(d0, sw, cin)
        add(C, mask, X, cin=cin)
        C.ccx(d0, sw, cin)
        C.cx(sw, cin)
        C.cx(d0, cin)
        C.free([cin])
        for u, t in zip(Uk[1:], mask):
            C.ccx(d0, u, t)
        C.free(fresh)
        (jj,) = C.alloc(1)
        C.cx(b, jj)
        C.cx(sw, jj)
        d1 = pool.pop()
        (t,) = C.alloc(1)
        C.x(d0)
        C.ccx(g0, d0, t)
        C.ccx(t, X[-1], d1)
        C.ccx(g0, d0, t)
        C.x(d0)
        C.free([t])
        C.ccx(jj, X[0], d1)
        ext = pool.pop()
        C.cx(X[-1], ext)
        X = X + [ext]
        add(C, Uk, X, ctl=d1)
        _shift_down(C, jj, X)
        # the top wire holds the sign again, or 0 after a shift
        C.x(jj)
        C.ccx(jj, X[m], X[-1])
        C.x(jj)
        pool.append(X.pop())
        C.cx(sw, jj)
        C.cx(b, jj)
        C.free([jj])
        C.ccx(g0, b, sw)
        C.free([sw])
        return X, d0, d1

    def d_descent(self, C, flat, stash, DG):
        """DG = D at level J (w + 1 wires) and pool_size() zero wires -> the digits, cell J first,
        then zero wires."""
        J, w = self.J, self.w
        cells = [flat[k:k + 2] for k in range(1, len(flat), 2)]
        D, pool = DG[:w + 1], DG[w + 1:]
        (one,) = C.alloc(1)
        C.x(one)
        digits = []
        for k in range(J - 1, 0, -1):
            g0, b = cells[k - 1]
            Uk = [one] + (self.stash_at(stash, k) if k > 1 else [])
            D, d0, d1 = self._d_cell(C, g0, b, Uk, D, pool, self.m(k))
            digits += [d0, d1]
        C.x(one)
        C.free([one])
        DG[:] = digits + pool + D

    def k_descent(self, C, cells, digits, a, b):
        """(a, b) at level J -> (0, 0) at level 1 by the digits; frees both."""
        J = self.J
        a, b = list(a), list(b)
        for i, k in enumerate(range(J - 1, 0, -1)):
            g0, bc = cells[k - 1]
            d0, d1 = digits[2 * i:2 * i + 2]
            m = self.m(k)
            (sw,) = C.alloc(1)
            C.ccx(g0, bc, sw)
            cswap(C, sw, a, b)
            C.free(a[m:])
            del a[m:]
            C.cx(d0, b[0])
            C.ccx(g0, a[0], b[0])
            C.free([b[0]])
            b = b[1:]
            if b:
                # the low bit's borrow and d1 never both hold
                (t,) = C.alloc(1)
                C.ccx(a[0], d0, t)
                C.cx(d1, t)
                sub(C, a[1:], b, ctl=g0, cin=t)
                C.cx(d1, t)
                C.ccx(a[0], d0, t)
                C.free([t])
                C.x(g0)
                C.ccx(g0, d1, b[0])
                C.x(g0)
                (jj,) = C.alloc(1)
                C.cx(bc, jj)
                C.cx(sw, jj)
                _shift_down(C, jj, b)
                C.cx(sw, jj)
                C.cx(bc, jj)
                C.free([jj])
            C.ccx(g0, bc, sw)
            C.free([sw])
            if len(b) > m:
                C.free(b[m:])
                del b[m:]
            else:
                b += C.alloc(m - len(b))
        C.free(a + b)

    # -- the Euclid pass's copies ----------------------------------------------------------------------
    #
    # Level k's cofactors are the Euclid pair before tick T - k, but for V inside a shift run that
    # the B1 sweep crossed, where the swept V is twice the pair's: there the tick below is a shift
    # and v has an odd count of trailing zeros. The pass holds v as V + E on its offset.

    def kept(self, keep):
        """(stash, U_J above its low 1, V, E) of the kept wires."""
        a = self.stash_size()
        w = self.w
        return keep[:a], keep[a:a + w - 1], keep[a + w - 1:a + 2 * w - 1], keep[a + 2 * w - 1]

    def kept_size(self):
        return self.stash_size() + 2 * self.w

    def keep_state(self, C, k, U, V, E, keep):
        """Before Euclid tick T - k: copy U, and at k = J also V and the offset."""
        stash, UJ, VJ, EJ = self.kept(keep)
        assert len(U) == self.m(k) and len(V) == self.m(k)
        if k == self.J:
            for x, y in zip(U[1:], UJ):
                C.cx(x, y)
            for x, y in zip(V, VJ):
                C.cx(x, y)
            C.cx(E[0], EJ)
        else:
            for x, y in zip(U[1:], self.stash_at(stash, k)):
                C.cx(x, y)

    def _odd_zeros(self, C, v, out):
        """out ^= [v's count of trailing zeros is odd], the XOR of [v's low j bits are 0], j >= 1."""
        for x in v:
            C.x(x)
        chain, z = [], v[0]
        C.cx(z, out)
        for x in v[1:]:
            (a,) = C.alloc(1)
            C.ccx(z, x, a)
            C.cx(a, out)
            chain.append(a)
            z = a
        prev = [v[0]] + chain[:-1]
        for i in reversed(range(len(chain))):
            C.ccx(prev[i], v[i + 1], chain[i])
        C.free(chain)
        for x in v:
            C.x(x)

    def _swept_v(self, C, VJ, EJ, below, inverse=False):
        """A fresh register with the swept V_J: (V + E), doubled inside a crossed run, and the two
        flag wires it was made with. below is the g0 of the cell under the start. inverse takes
        what the forward call returned back to zero wires."""
        def twice(V, d):
            for i in range(len(V) - 1, 0, -1):
                C.cx(V[i - 1], V[i])
                C.ccx(d, V[i], V[i - 1])
                C.cx(V[i - 1], V[i])

        def flag(V, par, d):
            C.x(below)
            C.ccx(below, par, d)
            C.x(below)

        if not inverse:
            V = C.alloc(self.w)
            par, d = C.alloc(2)
            for x, y in zip(VJ, V):
                C.cx(x, y)
            inc(C, V, EJ)
            self._odd_zeros(C, V, par)
            flag(V, par, d)
            twice(V, d)
            return V, par, d
        V, par, d = inverse
        _shift_down(C, d, V)
        flag(V, par, d)
        self._odd_zeros(C, V, par)
        dec(C, V, EJ)
        for x, y in zip(VJ, V):
            C.cx(x, y)
        C.free(V + [par, d])

    def convert(self, C, term, cells, R, S, keep, below):
        """R, S (L wires each) from (U*y, V*y) to (U*y mod q, V*y mod q) on their low n wires; the
        wires above are 0 after. term is the first cell's g0, cells the others' (g0, b), keep the
        Euclid pass's copies, and below the g0 of the cell under the start."""
        n, w, J = self.n, self.w, self.J
        C.free(R[n + w:] + S[n + w:])
        del R[n + w:], S[n + w:]
        self.fold(C, R)
        self.fold(C, S)
        r, a, s, b = R[:n], R[n:], S[:n], S[n:]
        flat = [term] + [x for c in cells for x in c]
        stash, UJ, VJ, EJ = self.kept(keep)
        made = self._swept_v(C, VJ, EJ, below)
        V = made[0]
        D = C.alloc(w + 1)
        mul_acc(C, V, a, D)
        mul_acc(C, UJ, b, D, sign=-1, odd=True)
        DG = D + C.alloc(self.pool_size())
        regs2 = [flat, stash, DG]
        b2 = C.run(lambda flat, stash, DG: self.d_descent(C, flat, stash, DG), regs2)
        nd = 2 * (J - 1)
        spare = len(DG) - nd
        C.free(DG[nd:])
        del DG[nd:]
        self.k_descent(C, cells, DG, a, b)
        DG += C.alloc(spare)
        C.apply(b2, regs2, inverse=True)
        D = DG[:w + 1]
        C.free(DG[w + 1:])
        # D*q = U*s - V*r, so -q*D + U*s - V*r is 0 mod 2^(w + 1); -q is f there, a short constant
        k = w + 1
        mul_odd_const(C, D, -self.q % (1 << k))
        mul_acc(C, UJ, s[:k], D, odd=True)
        mul_acc(C, V, r[:k], D, sign=-1)
        self._swept_v(C, VJ, EJ, below, inverse=made)
        C.free(D)
        del R[n:], S[n:]


class _Width:
    """two_bit's add-or-shift reads only its word width from self."""

    def __init__(self, n):
        self.n = n
