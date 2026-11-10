"""Carry trees that spend clean scratch wires for depth.

Every routine here is a permutation built from X, CX and CCX that restores each wire it does not
name as an output, and leaves every scratch wire it allocates clean.
"""


def target(C, *inputs):
    """A clean wire for a gate that reads `inputs`: the circuit's alloc_after when it has one, which
    picks a wire that is free by the time the inputs are, else its next wire."""
    pick = getattr(C, "alloc_after", None)
    return pick(*inputs) if pick else C.alloc(1)[0]


class Tape:
    """Records gates so a compute can be replayed in reverse as its uncompute."""

    def __init__(self):
        self.ops = []

    def x(self, a):
        self.ops.append(("x", a))

    def cx(self, a, b):
        self.ops.append(("cx", a, b))

    def ccx(self, a, b, c):
        self.ops.append(("ccx", a, b, c))

    def play(self, C, reverse=False):
        for name, *wires in reversed(self.ops) if reverse else self.ops:
            getattr(C, name)(*wires)


def and_tree(T, bits, alloc):
    """AND of bits on a fresh wire, by a balanced tree of len(bits) - 1 CCX. One bit is returned as
    it is."""
    level = list(bits)
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level) - 1, 2):
            t = alloc()
            T.ccx(level[i], level[i + 1], t)
            nxt.append(t)
        if len(level) % 2:
            nxt.append(level[-1])
        level = nxt
    return level[0]


def _spans(n, block):
    """[lo, hi) blocks covering n bits: `block` bits each, or, when it is a tuple, its sizes from
    the bottom with the last one repeated."""
    sizes = block if isinstance(block, tuple) else (block,)
    out, lo, k = [], 0, 0
    while lo < n:
        hi = min(lo + sizes[min(k, len(sizes) - 1)], n)
        out.append((lo, hi))
        lo, k = hi, k + 1
    return out


def _carry_out(T, A, B, alloc, block):
    """The carry out of A + B, no carry in, onto A's top wire. Blocks of `block` bits ripple in
    place as CDKM majorities, each from its own clean carry-in, which leaves A[i] ^ B[i] on B[i].
    A binary tree then joins the blocks: G_hi ^= P_hi G_lo, with P the AND of a block's B wires,
    made only for a node that is some join's upper half."""
    n = len(A)
    nodes = []
    for lo, hi in _spans(n, block):
        cin = alloc()
        T.cx(A[lo], B[lo])
        T.cx(A[lo], cin)
        T.ccx(cin, B[lo], A[lo])
        for i in range(lo + 1, hi):
            T.cx(A[i], B[i])
            T.cx(A[i], A[i - 1])
            T.ccx(A[i - 1], B[i], A[i])
        nodes.append({"g": A[hi - 1], "bits": B[lo:hi], "kids": None})
    level = nodes
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level) - 1, 2):
            nxt.append({"g": level[i + 1]["g"], "kids": (level[i], level[i + 1])})
        if len(level) % 2:
            nxt.append(level[-1])
        level = nxt
    root = level[0]

    def mark(node, need):
        node["need"] = need
        if node["kids"]:
            mark(node["kids"][0], need)
            mark(node["kids"][1], True)

    mark(root, False)

    def build(node):
        if node["kids"] is None:
            node["p"] = and_tree(T, node["bits"], alloc) if node["need"] else None
            return
        lo, hi = node["kids"]
        build(lo)
        build(hi)
        # The product first: it is the next join's P, and both gates hold hi's P.
        node["p"] = None
        if node["need"]:
            node["p"] = alloc()
            T.ccx(hi["p"], lo["p"], node["p"])
        T.ccx(hi["p"], lo["g"], hi["g"])

    build(root)
    return root["g"]


def gt_tree(C, A, B, t, ctrl, block=4):
    """t ^= ctrl * [A > B], with A, B and ctrl restored: the carry out of A + ~B."""
    assert len(A) == len(B) >= 1 and ctrl not in (*A, *B, t)
    taken = []

    def alloc():
        (w,) = C.alloc(1)
        taken.append(w)
        return w

    T = Tape()
    for x in B:
        T.x(x)
    top = _carry_out(T, list(A), list(B), alloc, block)
    T.play(C)
    C.ccx(ctrl, top, t)
    T.play(C, reverse=True)
    C.free(taken[::-1])


def gt_sweep(C, A, B, t, ctrl, spec):
    """gt_tree's map, t ^= ctrl * [A > B], as one arith.CarrySweep that stops short of its fixes.

    With B complemented, B ^= A and A's adjacent XORs put the carries of A + ~B in the sweep's
    form, U[j + 1] ^= U[j] B[j] from a clean carry-in below A. Only the carry out is read, so
    each block gives its local values and its product, the chain crosses the blocks once on the
    way up, and the same gates backwards put everything back: no fix, and each product is made
    and cleared once. spec is arith.carry_trees' (lo, hi, cap, fan, leaf)."""
    from .arith import CarrySweep, _Gates, _emit, adjacent_xor, carry_trees
    assert len(A) == len(B) >= 1 and ctrl not in (*A, *B, t)
    A, B = list(A), list(B)
    (cin,) = C.alloc(1)
    taken = []

    def take():
        (w,) = C.alloc(1)
        taken.append(w)
        return w

    wave = _Gates()
    for a, b in zip(A, B):
        wave.cx(a, b)
    wave.cx(A[0], cin)
    adjacent_xor(wave, A)
    for x in B:
        C.x(x)
    _emit(C, wave.ops)
    U = [cin] + A
    sweep = CarrySweep(C, U, B, take, lambda w: None)
    trees = carry_trees(len(A), spec)
    for root in trees:
        if root["kids"] is not None:
            sweep.up(root)
        C.ccx(U[root["s"]], sweep.beta(root), U[root["e"]])
    C.ccx(ctrl, U[-1], t)
    for root in reversed(trees):
        C.ccx(U[root["s"]], sweep.beta(root), U[root["e"]])
        if root["kids"] is not None:
            sweep.unlocal(root)
    _emit(C, reversed(wave.ops))
    for x in B:
        C.x(x)
    C.free(taken[::-1])
    C.free([cin])


# The products write() rebuilds on the complemented bits serve only its carries and its hook: each
# block's as a balanced tree, since no inner prefix is read, and none that neither reads.
LEAN_REBUILD = True


class PrefixTree:
    """AND(bits) on .top, and write(t): bits += t mod 2^len(bits), on clean scratch.

    Blocks of `block` bits each hold the chain q_k = AND(bits[lo .. lo+k]), and a balanced tree
    over the chain tops holds each subtree's AND. write sends the carry t down the tree
    (c_hi = c * AND(lo)), adds within each block (bit_k ^= cin * q_(k-1)), and clears the scratch
    in two halves. The old chains and tree clear against each block's new bits XOR its cin: a block
    whose carry in is 1 held all ones up to its first zero and now holds zeros there, so that XOR
    has the old chain, and a one-bit block's old bit. The carries clear against the same chains and tree rebuilt on the
    complemented new bits, because c * AND(old lo) = c * AND(~new lo): with c = 1 the lower half
    was all ones exactly when it is all zeros now.
    """

    def __init__(self, C, bits, block=4):
        self.C, self.bits = C, list(bits)
        self.taken, self.blocks, self.top = [], [], None
        h = len(self.bits)
        if h == 0:
            return
        self.chains, self.tree = Tape(), Tape()
        for lo in range(0, h, block):
            seg = self.bits[lo:lo + block]
            q = [seg[0]]
            for x in seg[1:]:
                t = self._alloc()
                self.chains.ccx(q[-1], x, t)
                q.append(t)
            self.blocks.append({"seg": seg, "q": q, "a": q[-1], "kids": None})
        level = self.blocks
        while len(level) > 1:
            nxt = []
            for i in range(0, len(level) - 1, 2):
                a = self._alloc()
                self.tree.ccx(level[i]["a"], level[i + 1]["a"], a)
                nxt.append({"a": a, "kids": (level[i], level[i + 1])})
            if len(level) % 2:
                nxt.append(level[-1])
            level = nxt
        self.root = level[0]
        self.top = self.root["a"]
        self.chains.play(C)
        self.tree.play(C)

    def _alloc(self):
        (w,) = self.C.alloc(1)
        self.taken.append(w)
        return w

    def release(self):
        pass

    def restore(self):
        pass

    def _rebuild(self, top):
        """The tape that puts AND(~bits) of every span a carry reads on that span's wire, the
        root's too when top is set. A carry reads its node's lower half, so the upper halves down
        the right edge are left out."""
        T = Tape()

        def walk(node, need):
            if node["kids"] is None:
                if need and len(node["seg"]) > 1:
                    pool = list(node["q"][1:])
                    got = and_tree(T, node["seg"], lambda: pool.pop(0))
                    assert got == node["a"] and not pool
                return
            lo, hi = node["kids"]
            walk(lo, True)
            walk(hi, need)
            if need:
                T.ccx(lo["a"], hi["a"], node["a"])

        walk(self.root, top)
        return T

    def write(self, t, mid=None, carries_first=False):
        """mid(z), when given, runs while z holds AND(~new bits): t * z is the carry out of the
        bits, t * AND(old bits). It runs before t's last use, or with carries_first after the
        write's own last use of t, so that a hook that comes late does not hold the carries."""
        C = self.C
        if not self.bits:
            return
        carries = Tape()
        made = []

        def down(node, c):
            if node["kids"] is None:
                node["cin"] = c
                return
            lo, hi = node["kids"]
            ch = target(C, c, lo["a"])
            made.append(ch)
            carries.ccx(c, lo["a"], ch)
            C.ccx(c, lo["a"], ch)
            down(lo, c)
            down(hi, ch)

        down(self.root, t)
        for blk in self.blocks:
            seg, q, cin = blk["seg"], blk["q"], blk["cin"]
            for k in range(len(seg) - 1, 0, -1):
                C.ccx(cin, q[k - 1], seg[k])
            C.cx(cin, seg[0])
        for blk in self.blocks:
            for x in blk["seg"]:
                C.cx(blk["cin"], x)
        self.tree.play(C, reverse=True)
        self.chains.play(C, reverse=True)
        for blk in self.blocks:
            for x in blk["seg"]:
                C.cx(blk["cin"], x)
        for x in self.bits:
            C.x(x)
        if LEAN_REBUILD:
            again = self._rebuild(mid is not None)
        else:
            again = Tape()
            again.ops = self.chains.ops + self.tree.ops
        again.play(C)
        if carries_first:
            carries.play(C, reverse=True)
        if mid is not None:
            mid(self.top)
        if not carries_first:
            carries.play(C, reverse=True)
        again.play(C, reverse=True)
        for x in self.bits:
            C.x(x)
        C.free(made[::-1])
        C.free(self.taken[::-1])


# write(aside=True) clears the carries below the quiet run by arith.CarrySweep over carry_trees'
# (lo, hi, cap, fan, leaf), or None for one fall a bit, 11 ticks each in series.
LOW_FALL_TREE = (2, 2, 2, 2, 2)


class LowAdd:
    """The carries of L + K, where K is the XOR of the masks whose selectors are set, and then
    either undo() or write(): L[1:] += K's carries and bits (bit 0 is left alone, as the tick's
    chain leaves it), the quiet run through quiet_write.

    L's bits a .. b-1 are the quiet run: no mask has a bit there, so the carry crosses it only when
    it is all ones, which mtop holds. K's other bits go on clean wires, and each stretch of them
    ripples in place as CDKM majorities in blocks of `block` bits, every block from its own clean
    carry-in. A block's products pi_k = p_lo ... p_(lo+k-1), p = L ^ K, turn its carry out into the
    true one: G ^= P * C. .top holds the carry out of L's top bit.
    """

    def __init__(self, C, L, selectors, masks, run, mtop, block):
        self.C, self.L = C, list(L)
        self.selectors, self.masks = list(selectors), list(masks)
        a, b = run
        w = len(self.L)
        self.a, self.b, self.mtop = a, b, mtop
        self.fan, self.up = Tape(), Tape()
        self.taken = []
        active = [i for i in range(w) if not a <= i < b]
        self.K = {i: self._alloc() for i in active}
        for i in active:
            for s in self._on(i):
                self.fan.cx(s, self.K[i])
        self.blocks = []
        spans = [(lo, min(lo + block, a)) for lo in range(0, a, block)]
        spans += [(lo, min(lo + block, w)) for lo in range(b, w, block)]
        L_, K = self.L, self.K
        for lo, hi in spans:
            z = self._alloc()
            if self._on(lo):                            # else K[lo] is 0, and so is its carry out
                self.up.cx(K[lo], L_[lo])
                self.up.cx(K[lo], z)
                self.up.ccx(z, L_[lo], K[lo])
            for i in range(lo + 1, hi):
                self.up.cx(K[i], L_[i])
                self.up.cx(K[i], K[i - 1])
                self.up.ccx(K[i - 1], L_[i], K[i])
            self.blocks.append({"lo": lo, "hi": hi, "z": z, "pi": None, "cin": None})
        self.pis, self.join = Tape(), Tape()
        self.products, self.copies = [], {}
        self.skip = None
        carry = None
        for blk in self.blocks:
            lo, hi = blk["lo"], blk["hi"]
            if lo == b and b > a:
                carry = self._skip(carry)
            if carry is not None:
                pi = [L_[lo]]
                for i in range(lo + 1, hi):
                    t = self._alloc()
                    self.products.append(t)
                    self.pis.ccx(pi[-1], L_[i], t)
                    pi.append(t)
                blk["pi"], blk["cin"] = pi, carry
                self.join.ccx(carry, pi[-1], K[hi - 1])
            carry = K[hi - 1]
        if b == w and b > a:
            carry = self._skip(carry)
        self.top = carry
        self.fan.play(C)
        self.up.play(C)
        self.pis.play(C)
        self.join.play(C)

    def _alloc(self):
        (w,) = self.C.alloc(1)
        self.taken.append(w)
        return w

    def _skip(self, carry):
        self.skip = self._alloc()
        self.join.ccx(carry, self.mtop, self.skip)
        return self.skip

    def _on(self, i):
        return [s for s, mask in zip(self.selectors, self.masks) if mask >> i & 1]

    def _free(self):
        self.C.free(self.taken[::-1])

    def undo(self):
        C = self.C
        self.join.play(C, reverse=True)
        self.pis.play(C, reverse=True)
        self.up.play(C, reverse=True)
        self.fan.play(C, reverse=True)
        self._free()

    def write(self, quiet_write, aside=False, before_falls=None):
        """quiet_write(c, mid) adds the carry wire c into the quiet run, and calls mid(z) once,
        before c's last use, with a wire z for which c * z = c * [the run was all ones]: the old
        mtop before the run changes, or AND(~run) after. The quiet run's write does not wait for
        the top stretch, whose first fall waits for whoever still reads .top.

        aside: the bits below the quiet run take their sums at once, and their carries fall
        against copies, so whoever reads L next does not wait for that chain. The caller ends
        with release(), once L's readers are done. before_falls() runs when L[1:a] hold their sums,
        ahead of the copies and of everything after them."""
        C, L, K = self.C, self.L, self.K
        # Make each later block's state the single chain's: true inner carries, its products
        # cleared, and its carry-in wire traded for the wire below it.
        for blk in self.blocks:
            cin, pi, lo, hi = blk["cin"], blk["pi"], blk["lo"], blk["hi"]
            if cin is None:
                continue
            for k in range(1, hi - lo):
                C.ccx(cin, pi[k - 1], K[lo + k - 1])
        self.pis.play(C, reverse=True)
        if aside:
            C.free(self.products[::-1])
            self.taken = [x for x in self.taken if x not in self.products]
        for blk in self.blocks:
            if blk["cin"] is None:
                continue
            if self._on(blk["lo"]):
                C.cx(blk["z"], blk["cin"])
            for s in self._on(blk["lo"]):
                C.cx(s, blk["z"])

        def below(i):
            for blk in self.blocks:
                if blk["lo"] == i:
                    return blk["z"] if blk["cin"] is None else blk["cin"]
            return K[i - 1]

        def fall(i):
            c = below(i)
            C.ccx(c, L[i], K[i])
            C.cx(K[i], c)
            if i:
                C.cx(c, L[i])
            else:
                C.cx(K[i], L[i])

        def top(z):
            for i in reversed(range(self.b, len(L))):
                fall(i)
            C.ccx(K[self.a - 1], z, self.skip)

        low = range(1, self.a) if aside else ()
        for i in low:
            # The rail below bit i holds c_i ^ k_i, so L[i] takes its sum now.
            C.cx(below(i), L[i])
            for s in self._on(i):
                C.cx(s, L[i])
        if self.skip is not None:
            quiet_write(K[self.a - 1], top)
        else:
            for i in reversed(range(self.b, len(L))):
                fall(i)
        if before_falls is not None:
            before_falls()
        for i in low:
            # ~(sum ^ k_i) = ~(l_i ^ c_i): with the rail c_i ^ k_i it gives the same product as
            # the fall's own pair, (c_i ^ k_i)(l_i ^ k_i).
            (t,) = C.alloc(1)
            self.copies[i] = t
            C.cx(L[i], t)
            for s in self._on(i):
                C.cx(s, t)
            C.x(t)
        if aside and LOW_FALL_TREE and self.a > 2:
            # Each fall is K[i] ^= below(i) * P_i and then CXs out of K[i]; the rail below is read
            # before those CXs reach it, and K[i] is only XORed into in between. So the CCXs are
            # the carry sweep's inverse on the rails, which runs over trees, and the CXs follow in
            # the falls' order.
            from .arith import CarrySweep, carry_trees
            rails = [below(0)] + [K[i] for i in range(self.a)]
            P = [L[0]] + [self.copies[i] for i in range(1, self.a)]
            sweep = CarrySweep(C, rails, P, lambda: C.alloc(1)[0], lambda t: C.free([t]))
            for root in reversed(carry_trees(self.a, LOW_FALL_TREE)):
                sweep.backward(root)
            for i in reversed(range(self.a)):
                C.cx(K[i], below(i))
                if i in self.copies:
                    C.x(self.copies[i])
                    C.cx(K[i], self.copies[i])
                else:
                    C.cx(K[i], L[i])
        else:
            for i in reversed(range(self.a)):
                if i in self.copies:
                    c, t = below(i), self.copies[i]
                    C.ccx(c, t, K[i])
                    C.cx(K[i], c)
                    C.x(t)
                    C.cx(K[i], t)
                else:
                    fall(i)
        self.fan.play(C, reverse=True)
        self._free()

    def release(self):
        """Clears the copies write(aside=True) left, each holding its bit of L."""
        for i, t in self.copies.items():
            self.C.cx(self.L[i], t)
        self.C.free(list(self.copies.values())[::-1])
        self.copies = {}
