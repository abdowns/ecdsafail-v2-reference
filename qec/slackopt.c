// Slack-aware reorder of one call-free definition body. Same unitary.
//
//   slackopt <circuit.bin> <body> <out.rec> [evals=100000] [K=6] [seed=1] [D=24] [S=0] [W=0]
//
// evals is the number of orders the search may score, so a run is reproducible. With W > 0 a
// body longer than W gates is searched one window of W gates at a time, twice over, the second
// time with the windows shifted by W / 2; each window gets evals scores. A window's arrival
// times are the body's schedule up to the window and its tails are the body's tails after it,
// so a window's P is still the circuit's ticks through that execution.
//
// <circuit.bin> is the int32 stream qec/slack_pass.py writes. The tool schedules the
// whole expanded circuit under the board's tick model, forward and backward, and
// records at every execution of <body> the ready time of each of its wires on entry
// (rin) and the longest path from each of its wires on exit to the end of the circuit
// (tail). The tick model is max-plus linear, so the backward pass is the transposed
// gate: for CCX with outputs a = b = x, t = x - 1 where x = max(ra + 8, rb + 10, rt + 11),
// the tails are M + 8, M + 10, M + 11 with M = max(Ta, Tb, Tt - 1).
//
// For K executions spread over the run the search then minimises
//   P = sum over executions of max over wires (ready on exit + tail)      (the circuit's ticks
//       through that execution, the others held fixed), then
//   Q = sum over executions and gates of the excess of (gate time + tail after the gate)
//       over (the execution's starting P - D)                             (total negative slack)
// over orders of the body's gates and control orders of its CCX. Moves:
//   - a gate g and the gate p before it on a wire, when the two commute: g to just before p,
//     or p to just after g, legal only if every gate in between that shares a wire with the
//     moved gate commutes with it (X-type gates commute when neither's target is among the
//     other's controls);
//   - control1 <-> control2 of a CCX (the same gate).
// Candidates are the binding edges of gates within S ticks of critical. Descent to a local
// optimum, then kicks (random legal moves) and descent again, keeping the best.
//
// Before writing, the new body is run against the old one on 4,096 random basis states.
// <out.rec> gets the body's records (opcode, a, b, c) in the new order.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

typedef struct { int arity, nrec, npool; int32_t *rec, *pool; } Body;
static Body *bodies; static int nb, nw, target;
static int32_t *ready, *Tl;
typedef struct { int inv; int32_t *rin, *tail; } Inst;
static Inst *inst; static int ninst, capinst, rev_at;

static inline int32_t max2(int32_t a, int32_t b) { return a > b ? a : b; }
static inline int32_t max3(int32_t a, int32_t b, int32_t c) { return max2(a, max2(b, c)); }

static void fwd(int id, const int32_t *map, int inv) {
    Body *B = &bodies[id];
    for (int k = 0; k < B->nrec; k++) {
        const int32_t *r = B->rec + 4 * (inv ? B->nrec - 1 - k : k);
        int op = r[0];
        if (op >= 100) {
            int n = r[3]; int32_t *sub = malloc(4 * (n ? n : 1));
            const int32_t *p = B->pool + r[2];
            for (int i = 0; i < n; i++) sub[i] = map ? map[p[i]] : p[i];
            int cinv = inv ^ (op - 100);
            if (r[1] == target) {
                if (ninst == capinst) { capinst = capinst ? 2 * capinst : 64; inst = realloc(inst, capinst * sizeof(Inst)); }
                Inst *I = &inst[ninst++]; I->inv = cinv; I->rin = malloc(4 * n); I->tail = malloc(4 * n);
                for (int i = 0; i < n; i++) I->rin[i] = ready[sub[i]];
            }
            fwd(r[1], sub, cinv); free(sub);
            continue;
        }
        int a = map ? map[r[1]] : r[1], b = map ? map[r[2]] : r[2], c = map ? map[r[3]] : r[3];
        if (op == 0 || op == 6) ready[a] += 1;
        else if (op == 1 || op == 4) ready[a] = ready[b] = max2(ready[a], ready[b]) + 1;
        else if (op == 3) ready[a] = ready[b] = max2(ready[a], ready[b]) + 3;
        else { int32_t x = max3(ready[a] + 8, ready[b] + 10, ready[c] + 11); ready[a] = ready[b] = x; ready[c] = x - 1; }
    }
}
static void bwd(int id, const int32_t *map, int inv) {
    Body *B = &bodies[id];
    for (int k = B->nrec - 1; k >= 0; k--) {
        const int32_t *r = B->rec + 4 * (inv ? B->nrec - 1 - k : k);
        int op = r[0];
        if (op >= 100) {
            int n = r[3]; int32_t *sub = malloc(4 * (n ? n : 1));
            const int32_t *p = B->pool + r[2];
            for (int i = 0; i < n; i++) sub[i] = map ? map[p[i]] : p[i];
            if (r[1] == target) { Inst *I = &inst[--rev_at]; for (int i = 0; i < n; i++) I->tail[i] = Tl[sub[i]]; }
            bwd(r[1], sub, inv ^ (op - 100)); free(sub);
            continue;
        }
        int a = map ? map[r[1]] : r[1], b = map ? map[r[2]] : r[2], c = map ? map[r[3]] : r[3];
        if (op == 0 || op == 6) Tl[a] += 1;
        else if (op == 1 || op == 4) Tl[a] = Tl[b] = max2(Tl[a], Tl[b]) + 1;
        else if (op == 3) Tl[a] = Tl[b] = max2(Tl[a], Tl[b]) + 3;
        else { int32_t M = max3(Tl[a], Tl[b], Tl[c] - 1); Tl[a] = M + 8; Tl[b] = M + 10; Tl[c] = M + 11; }
    }
}

// ---------------------------------------------------------------- body search
typedef struct { int kind, a, b, c; } Gate;   // kind 0 X, 1 CX, 2 CCX, 3 SWAP
static Gate *G; static int n, arity, K, D, S;
static int lo, hi;                       // the window of ord being searched
static Inst **sc; static int64_t *V0;
static int32_t **rinW, **tailW;          // per execution: ready on entry to the window, tail on exit from it
static int32_t *R, *T, *xs;
static int *ord, *pos;

static inline int commute(const Gate *p, const Gate *q) {
    if (p->kind == 3 || q->kind == 3) return 0;
    int pt = p->kind == 0 ? p->a : p->kind == 1 ? p->b : p->c;
    int qt = q->kind == 0 ? q->a : q->kind == 1 ? q->b : q->c;
    if (q->kind == 1 && q->a == pt) return 0;
    if (q->kind == 2 && (q->a == pt || q->b == pt)) return 0;
    if (p->kind == 1 && p->a == qt) return 0;
    if (p->kind == 2 && (p->a == qt || p->b == qt)) return 0;
    return 1;
}
static inline int shares(const Gate *p, const Gate *q) {
    int pw[3] = { p->a, p->kind == 0 ? p->a : p->b, p->kind == 2 ? p->c : p->a };
    int qw[3] = { q->a, q->kind == 0 ? q->a : q->b, q->kind == 2 ? q->c : q->a };
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) if (pw[i] == qw[j]) return 1;
    return 0;
}

static inline int32_t step(int32_t *R, const Gate *g) {
    int32_t x;
    switch (g->kind) {
    case 0: x = R[g->a] += 1; break;
    case 1: x = max2(R[g->a], R[g->b]) + 1; R[g->a] = R[g->b] = x; break;
    case 3: x = max2(R[g->a], R[g->b]) + 3; R[g->a] = R[g->b] = x; break;
    default: x = max3(R[g->a] + 8, R[g->b] + 10, R[g->c] + 11); R[g->a] = R[g->b] = x; R[g->c] = x - 1;
    }
    return x;
}
static inline int32_t unstep(int32_t *T, const Gate *g) {
    int32_t M;
    switch (g->kind) {
    case 0: M = T[g->a]; T[g->a] = M + 1; break;
    case 1: M = max2(T[g->a], T[g->b]); T[g->a] = T[g->b] = M + 1; break;
    case 3: M = max2(T[g->a], T[g->b]); T[g->a] = T[g->b] = M + 3; break;
    default: M = max3(T[g->a], T[g->b], T[g->c] - 1); T[g->a] = M + 8; T[g->b] = M + 10; T[g->c] = M + 11;
    }
    return M;
}
// The gate at step t of the window, in the order execution s runs it.
#define AT(s, t) (sc[s]->inv ? hi - 1 - (t) : lo + (t))
// One execution's forward schedule of the window; fills xs (time per step), returns P.
static int64_t forward(int s) {
    memcpy(R, rinW[s], 4 * arity);
    for (int t = 0; t < hi - lo; t++) xs[t] = step(R, &G[ord[AT(s, t)]]);
    int64_t v = 0; const int32_t *tl = tailW[s];
    for (int w = 0; w < arity; w++) if ((int64_t)R[w] + tl[w] > v) v = (int64_t)R[w] + tl[w];
    return v;
}
// Backward over the same steps: Q against threshold thr. crit, when given, gets the path length through each step.
static int64_t backward(int s, int64_t thr, int32_t *crit) {
    memcpy(T, tailW[s], 4 * arity); int64_t q = 0;
    for (int t = hi - lo - 1; t >= 0; t--) {
        int64_t c = (int64_t)xs[t] + unstep(T, &G[ord[AT(s, t)]]);
        if (crit) crit[t] = (int32_t)c;
        if (c > thr) q += c - thr;
    }
    return q;
}
// Set the window and each execution's arrival times and tails at its edges.
static void window(int l, int h) {
    lo = l; hi = h;
    for (int s = 0; s < K; s++) {
        const Inst *I = sc[s];
        memcpy(rinW[s], I->rin, 4 * arity); memcpy(tailW[s], I->tail, 4 * arity);
        if (!I->inv) {
            for (int k = 0; k < lo; k++) step(rinW[s], &G[ord[k]]);
            for (int k = n - 1; k >= hi; k--) unstep(tailW[s], &G[ord[k]]);
        } else {
            for (int k = n - 1; k >= hi; k--) step(rinW[s], &G[ord[k]]);
            for (int k = 0; k < lo; k++) unstep(tailW[s], &G[ord[k]]);
        }
    }
}
static void objective(int64_t *P, int64_t *Q) {
    int64_t p = 0, q = 0;
    for (int s = 0; s < K; s++) { p += forward(s); q += backward(s, V0[s] - D, NULL); }
    *P = p; *Q = q;
}

typedef struct { int g1, g2; } Cand;   // g2 < 0: flip the controls of g1
static Cand *cand; static int ncand, capcand;
static void push(int g1, int g2) {
    if (ncand == capcand) { capcand = capcand ? 2 * capcand : 4096; cand = realloc(cand, capcand * sizeof(Cand)); }
    cand[ncand].g1 = g1; cand[ncand].g2 = g2; ncand++;
}
static int32_t *crit, *lastpos;
static void candidates(int slack) {
    ncand = 0;
    for (int s = 0; s < K; s++) {
        int64_t v = forward(s); backward(s, v, crit);
        memcpy(R, rinW[s], 4 * arity);
        for (int w = 0; w < arity; w++) lastpos[w] = -1;
        for (int t = 0; t < hi - lo; t++) {
            int gi = ord[AT(s, t)]; const Gate *g = &G[gi];
            int ws[3], lat[3], nws; int32_t x = xs[t];
            if (g->kind == 0) { ws[0] = g->a; lat[0] = 1; nws = 1; }
            else if (g->kind == 1) { ws[0] = g->a; ws[1] = g->b; lat[0] = lat[1] = 1; nws = 2; }
            else if (g->kind == 3) { ws[0] = g->a; ws[1] = g->b; lat[0] = lat[1] = 3; nws = 2; }
            else { ws[0] = g->a; ws[1] = g->b; ws[2] = g->c; lat[0] = 8; lat[1] = 10; lat[2] = 11; nws = 3; }
            if (crit[t] >= v - slack) {
                for (int i = 0; i < nws; i++) {
                    int q = lastpos[ws[i]];
                    if (q >= 0 && R[ws[i]] + lat[i] >= x - (g->kind == 2 ? 2 : 0)) {
                        int pg = ord[AT(s, q)];
                        if (commute(&G[pg], g)) push(pg, gi);
                    }
                }
                if (g->kind == 2 && R[g->b] > R[g->a]) push(gi, -1);
            }
            for (int i = 0; i < nws; i++) lastpos[ws[i]] = t;
            step(R, g);
        }
    }
}
// Move the gate at position j to just before position i (which = 0) or the gate at i to just after j (which = 1).
static int legal(int i, int j, int which) {
    const Gate *m = &G[ord[which ? i : j]];
    for (int q = which ? i + 1 : i; q <= (which ? j : j - 1); q++) {
        const Gate *o = &G[ord[q]];
        if (shares(m, o) && !commute(m, o)) return 0;
    }
    return 1;
}
static void domove(int i, int j, int which) {
    if (!which) { int g = ord[j]; memmove(ord + i + 1, ord + i, sizeof(int) * (j - i)); ord[i] = g; }
    else { int g = ord[i]; memmove(ord + i, ord + i + 1, sizeof(int) * (j - i)); ord[j] = g; }
    for (int q = i; q <= j; q++) pos[ord[q]] = q;
}
static void undomove(int i, int j, int which) {
    if (!which) { int g = ord[i]; memmove(ord + i, ord + i + 1, sizeof(int) * (j - i)); ord[j] = g; }
    else { int g = ord[j]; memmove(ord + i + 1, ord + i, sizeof(int) * (j - i)); ord[i] = g; }
    for (int q = i; q <= j; q++) pos[ord[q]] = q;
}
static inline void flip(int g) { int t = G[g].a; G[g].a = G[g].b; G[g].b = t; }

static uint64_t rs = 88172645463325252ULL;
static inline uint64_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

static int64_t curP, curQ; static long budget, tried, kept;
static void rebase(void) {
    for (int s = 0; s < K; s++) V0[s] = forward(s);
    objective(&curP, &curQ);
}
static void descend(void) {
    for (;;) {
        int got = 0;
        rebase();
        candidates(S);
        for (int c = ncand - 1; c > 0; c--) { int o = rnd() % (c + 1); Cand t = cand[c]; cand[c] = cand[o]; cand[o] = t; }
        for (int c = 0; c < ncand; c++) {
            if (tried >= budget) return;
            int64_t P, Q;
            if (cand[c].g2 < 0) {
                flip(cand[c].g1); objective(&P, &Q); tried++;
                if (P < curP || (P == curP && Q < curQ)) { curP = P; curQ = Q; got++; kept++; } else flip(cand[c].g1);
                continue;
            }
            int i = pos[cand[c].g1], j = pos[cand[c].g2];
            if (i > j) { int t = i; i = j; j = t; }
            for (int which = 0; which < 2; which++) {
                if (!legal(i, j, which)) continue;
                domove(i, j, which); objective(&P, &Q); tried++;
                if (P < curP || (P == curP && Q < curQ)) { curP = P; curQ = Q; got++; kept++; break; }
                undomove(i, j, which);
            }
        }
        if (!got) return;
    }
}

static Gate *bestG; static int *bestord; static int64_t *bestV;
// Descent, then kicks and descent again until the budget is spent; leaves the best order found. Returns the kicks made.
static int search(void) {
    int rounds = 0, idle = 0; long before = tried;
    rebase();
    memcpy(bestG, G, sizeof(Gate) * n); memcpy(bestord, ord, sizeof(int) * n); memcpy(bestV, V0, 8 * K);
    int64_t bestP = curP, bestQ = curQ;
    for (;;) {
        descend();
        // Compare with the best under the best's thresholds.
        memcpy(V0, bestV, 8 * K); objective(&curP, &curQ);
        if (curP < bestP || (curP == bestP && curQ < bestQ)) {
            rebase();
            bestP = curP; bestQ = curQ; memcpy(bestG, G, sizeof(Gate) * n); memcpy(bestord, ord, sizeof(int) * n); memcpy(bestV, V0, 8 * K);
        }
        // A round that scored nothing cannot be told from the next one by the budget; stop after 64 in a row.
        idle = tried == before ? idle + 1 : 0;
        if (tried >= budget || idle >= 64) break;
        before = tried;
        memcpy(G + 0, bestG, sizeof(Gate) * n); memcpy(ord + lo, bestord + lo, sizeof(int) * (hi - lo));
        for (int k = lo; k < hi; k++) pos[ord[k]] = k;
        candidates(S + 2);
        if (!ncand) break;
        for (int kk = 0; kk < 4; kk++) {
            Cand c = cand[rnd() % ncand];
            if (c.g2 < 0) { flip(c.g1); continue; }
            int i = pos[c.g1], j = pos[c.g2]; if (i > j) { int t = i; i = j; j = t; }
            int which = rnd() & 1;
            if (legal(i, j, which)) domove(i, j, which);
        }
        rounds++;
    }
    memcpy(G, bestG, sizeof(Gate) * n); memcpy(ord + lo, bestord + lo, sizeof(int) * (hi - lo));
    for (int k = lo; k < hi; k++) pos[ord[k]] = k;
    return rounds;
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: slackopt <circuit.bin> <body> <out.rec> [evals] [K] [seed] [D] [S] [W]\n"); return 2; }
    FILE *f = fopen(argv[1], "rb"); if (!f) { perror("open"); return 1; }
    target = atoi(argv[2]);
    budget = argc > 4 ? atol(argv[4]) : 100000; K = argc > 5 ? atoi(argv[5]) : 6;
    if (argc > 6) rs ^= (uint64_t)atoll(argv[6]) * 0x9E3779B97F4A7C15ULL;
    D = argc > 7 ? atoi(argv[7]) : 24; S = argc > 8 ? atoi(argv[8]) : 0; int W = argc > 9 ? atoi(argv[9]) : 0;
    int32_t h[3]; fread(h, 4, 2, f); nw = h[0]; nb = h[1];
    bodies = calloc(nb, sizeof(Body));
    for (int i = 0; i < nb; i++) {
        fread(h, 4, 3, f); Body *B = &bodies[i]; B->arity = h[0]; B->nrec = h[1]; B->npool = h[2];
        B->rec = malloc(16 * (size_t)(B->nrec ? B->nrec : 1)); fread(B->rec, 16, B->nrec, f);
        B->pool = malloc(4 * (size_t)(B->npool ? B->npool : 1)); fread(B->pool, 4, B->npool, f);
    }
    fclose(f);
    ready = calloc(nw, 4); Tl = calloc(nw, 4);
    fwd(nb - 1, NULL, 0);
    int32_t ticks = 0; for (int i = 0; i < nw; i++) if (ready[i] > ticks) ticks = ready[i];
    rev_at = ninst; bwd(nb - 1, NULL, 0);
    int32_t back = 0; for (int i = 0; i < nw; i++) if (Tl[i] > back) back = Tl[i];
    if (back != ticks || rev_at != 0) { fprintf(stderr, "backward pass disagrees: %d vs %d\n", back, ticks); return 1; }
    Body *B = &bodies[target]; n = B->nrec; arity = B->arity;
    if (!ninst) { fprintf(stderr, "body %d never runs\n", target); return 1; }
    G = malloc(sizeof(Gate) * n);
    for (int k = 0; k < n; k++) {
        const int32_t *r = B->rec + 4 * k;
        if (r[0] >= 100) { fprintf(stderr, "body %d holds a call\n", target); return 1; }
        if (r[0] > 3) { fprintf(stderr, "op %d not handled\n", r[0]); return 1; }
        G[k].kind = r[0] == 0 ? 0 : r[0] == 1 ? 1 : r[0] == 2 ? 2 : 3; G[k].a = r[1]; G[k].b = r[2]; G[k].c = r[3];
    }
    ord = malloc(sizeof(int) * n); pos = malloc(sizeof(int) * n);
    for (int k = 0; k < n; k++) ord[k] = pos[k] = k;
    R = malloc(4 * arity); T = malloc(4 * arity); xs = malloc(4 * n); crit = malloc(4 * n); lastpos = malloc(4 * arity);
    if (K > ninst) K = ninst;
    sc = malloc(sizeof(Inst *) * K); V0 = malloc(8 * K);
    for (int s = 0; s < K; s++) sc[s] = &inst[(int)(((2 * s + 1) * (int64_t)ninst) / (2 * K))];
    rinW = malloc(sizeof(int32_t *) * K); tailW = malloc(sizeof(int32_t *) * K);
    for (int s = 0; s < K; s++) { rinW[s] = malloc(4 * arity); tailW[s] = malloc(4 * arity); }
    bestG = malloc(sizeof(Gate) * n); bestord = malloc(sizeof(int) * n); bestV = malloc(8 * K);
    window(0, n);
    int onpath = 0; int64_t P0 = 0;
    for (int s = 0; s < K; s++) { V0[s] = forward(s); P0 += V0[s]; if (V0[s] == ticks) onpath++; }
    double t0 = now(); long evals = budget, alltried = 0; int rounds = 0, windows = 0;
    if (W <= 0 || n <= W) { rounds = search(); alltried = tried; windows = 1; }
    else for (int pass = 0; pass < 2; pass++)
        for (int l = pass ? W / 2 : 0; l < n; l += W) {
            window(l, l + W < n ? l + W : n);
            tried = 0; budget = evals; rounds += search(); alltried += tried; windows++;
        }
    window(0, n);
    int64_t P1 = 0; for (int s = 0; s < K; s++) P1 += forward(s);

    // The new body against the old one on random basis states, 64 lanes a word.
    uint64_t *wa = malloc(8 * arity), *wb = malloc(8 * arity);
    for (int rep = 0; rep < 64; rep++) {
        for (int w = 0; w < arity; w++) wa[w] = wb[w] = rnd();
        for (int k = 0; k < n; k++) {
            const int32_t *r = B->rec + 4 * k;
            if (r[0] == 0) wa[r[1]] = ~wa[r[1]]; else if (r[0] == 1) wa[r[2]] ^= wa[r[1]];
            else if (r[0] == 2) wa[r[3]] ^= wa[r[1]] & wa[r[2]]; else { uint64_t t = wa[r[1]]; wa[r[1]] = wa[r[2]]; wa[r[2]] = t; }
            const Gate *g = &G[ord[k]];
            if (g->kind == 0) wb[g->a] = ~wb[g->a]; else if (g->kind == 1) wb[g->b] ^= wb[g->a];
            else if (g->kind == 2) wb[g->c] ^= wb[g->a] & wb[g->b]; else { uint64_t t = wb[g->a]; wb[g->a] = wb[g->b]; wb[g->b] = t; }
        }
        if (memcmp(wa, wb, 8 * arity)) { fprintf(stderr, "body %d: reordered body differs from the original\n", target); return 1; }
    }
    FILE *o = fopen(argv[3], "wb"); if (!o) { perror("out"); return 1; }
    for (int k = 0; k < n; k++) {
        const Gate *g = &G[ord[k]];
        int32_t r[4] = { g->kind == 0 ? 0 : g->kind == 1 ? 1 : g->kind == 2 ? 2 : 3, g->a, g->b, g->c };
        fwrite(r, 4, 4, o);
    }
    fclose(o);
    printf("{\"body\": %d, \"gates\": %d, \"runs\": %d, \"scenarios\": %d, \"on_path\": %d, \"ticks\": %d, \"P\": [%lld, %lld], \"per_run\": %.2f, \"tried\": %ld, \"kept\": %ld, \"rounds\": %d, \"windows\": %d, \"seconds\": %.0f}\n",
           target, n, ninst, K, onpath, ticks, (long long)P0, (long long)P1, (double)(P0 - P1) / K, alltried, kept, rounds, windows, now() - t0);
    return 0;
}
