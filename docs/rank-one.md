# The rank-one start of a Bezout pass

The construction is in `qec/rank_one.py`, and `DialogMultiplier._rank_start` in `qec/point_add.py` runs it.
Each statement is numbered so that it can be tested as worded.

## What it does

The Bezout pass starts at (R, S) = (y, 0) and runs the swept tape's cells from the last to the first.
After its first j cells, (R, S) ≡ (U·y, V·y) mod q for the integer cofactor pair (U, V) of those cells, which starts at (1, 0).
`_rank_start` runs the first J cells on integers with no reduction.
Cell T − 1 is the terminal S ^= g0·R, as before.
Cells T − 2 to T − J + 1 are two_bit's add-or-shift and swap on words of n + w + 3 wires, with the held offset after a B1 as in `docs/mux-proof.md`.
Cell T − J is semantic: a B1 doubles S and adds nothing, so no lane is held after it.
One conversion then leaves (U·y mod q, V·y mod q) and every other wire 0.
The tick after it is a first tick (no previous cell), so it reads no held flag.

## Premises

**P1. The width.** After j reverse cells, U, V < 2^w with w = `tick_widths(q, T)[T − j]`, the bound the Euclid pass already uses for its own registers.
On the unswept tape the pair after j reverse cells is the Euclid pair (u, v) before Euclid tick T − j, because each Bézout cell inverts the Euclid tick that wrote it (`out/bezglue-w3/z3/rank_one_z3.py`, "Bezout cell inverts the Euclid tick").
The cells run on the swept tape, where that identity is false inside a shift run: the B1 sweep moves a run's single short tick, so at an interior boundary the swept pair is (u, 2v) where the Euclid pair is (u, v).
The two pairs agree at every run boundary.
At an interior boundary the forward tick before it was a double shift from (u, 4v), whose potential bounds max(u, 2v)² by the same B_t as the Euclid pair (`out/review-rankone/REVIEW.md`, "Exact premises and the width proof").
So the swept cofactors meet the same width; nothing beyond the stack's width premise is assumed.
At J = 21 the generator's widths give w = 22, each level's bound is one bit above the level below, and the fold's premises hold (`rank_one.plan`, and the assert in `RankOne.__init__`).
On a held lane the stored word is H − R0 with H = S + R, so the words hold n + w + 1 bits, and two more wires are the add-or-shift's top.

**P2. The fold.** For X = hi·2ⁿ + lo < 2^w·q with (2^w + 2)·f ≤ 2ⁿ, m = hi + e with e = [lo + (hi + 1)f ≥ 2ⁿ] is ⌊X/q⌋, and lo + m·f mod 2ⁿ is X − m·q.
`rank_one.plan` turns the start off for a prime where this fails.

**P3. The last cell.** A cell after a B1 is a subtract (D1 of `docs/mux-proof.md`), so on a held lane of the last cell g0 = 1, and 2(X − (R >> 1)) + R0 = 2H − R for X = H − R0.

## The conversion

1. Fold R and S (P2): (r, a) and (s, b), with a = ⌊U·y/q⌋ and b = ⌊V·y/q⌋.
2. The cofactors are not rebuilt. The Euclid pass copies U before each of its last J ticks, and V and its offset E before the first of them, onto `keep` wires (`keep_state`). The inverse Euclid pass clears the copies.
   Level k's U is the Euclid U before tick T − k. Level J's swept V is (V + E) << d, with d = [the cell under the start is a shift] AND [V + E has an odd number of trailing zeros] (`_swept_v`).
3. D = V·a − U·b = (U·s − V·r)/q is an integer, in (−V, U) when V > 0 since r, s < q, and 0 when V = 0. It goes on w + 1 fresh wires, mod 2^(w+1).
4. The descent. Level k is the state after k cells, with invariant 0 ≤ a < U_k and D = V_k·a − U_k·b in (−V_k, U_k), or 0 when V_k = 0. A cell undone takes level k + 1 to level k:
   - on an add with a swap, (a, b) trade places and D → −D;
   - d0 = D mod 2, then D → (D + d0·U)/2;
   - on an add, d1 = [D < 0] when d0 = 0, then D → D + d1·U; on J, d1 = D mod 2, then D → (D + d1·U)/2; on B1, d1 = 0;
   - b → (b − g0·a − d0 − 2·d1)/2, and half of that again on J.

   Each line keeps D = V·a − U·b and lands in the level's range, so level 1 is (a, b, D) = (0, 0, 0).
   `d_descent` walks D down and leaves two digits a cell. `k_descent` walks (a, b) down by the digits to (0, 0). `d_descent` undone gives D back.
5. D·q = U·s − V·r, so −q·D + U·s − V·r = 0 mod 2^(w+1) clears D from the low w + 1 bits of r and s. The swept V is unmade.

## Checks

- `scripts/descent_model.py`: the descent on integers, over random cell sequences and every input at small primes.
- `scripts/keep_model.py`: step 2's relation at every level of every x for eight primes up to 8,191 and of 3,000 x at secp256k1, against a sweep model that shares no code with the circuit.
- `scripts/convert_check.py J [lanes] [seed] [mutant]`: the conversion alone at secp256k1, on cell sequences inside the width bound with the kept copies the Euclid pass would leave. Its mutants each drop one CCX.
- `scripts/m_exhaustive.py q ...`: M(x, y) = (x, x·y mod q) on every x in [1, q) and every y with every other wire 0, at q = 61, 127, 251, 509, 1021, 2039, 4093 and 8191 (J from 3 to 8).
- `scripts/long_run_check.py`: M at secp256k1 on x = m·2^k and q − m·2^k whose Euclid pass runs 364 to 382 ticks, so the window holds real cells. Random x never reach it.
- `scripts/accept.py` runs the whole point addition at its toy primes; q = 61 takes the start with J = 3.
