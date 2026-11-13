// Board tick model with critical-path attribution.
//   ticks <circuit.bin> [--path | --calls <lo> <hi>]
// --calls prints, after each call of a definition numbered lo..hi, the latest ready time of its wires.
// Prints wires, ticks, ccx, qubit_time, ops. With --path, charges the critical path's ticks
// to top-level records and to innermost / level-1 / level-2 definitions.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

typedef struct { int arity, nrec, npool; int32_t *rec, *pool; } Body;
static Body *bodies; static int nb, nw;
static int64_t *ready; static int32_t *lastop;
static int64_t nops, nccx, cap; static int want_path;
static int32_t *pred, *o_top, *o_in, *o_l1, *o_l2; static int32_t *o_dt;
static int cur_top, stk[64], sp; static int call_lo = -1, call_hi = -1;

static inline void note(int64_t fin_in, int64_t fin_out, int critw) {
    if (!want_path) return;
    if (nops >= cap) { fprintf(stderr, "op cap\n"); exit(1); }
    pred[nops] = lastop[critw]; o_dt[nops] = (int32_t)fin_in;
    o_top[nops] = cur_top; o_in[nops] = sp ? stk[sp-1] : -1;
    o_l1[nops] = sp > 0 ? stk[0] : -1; o_l2[nops] = sp > 1 ? stk[1] : -1;
}
static void run(int id, const int32_t *map, int inv) {
    Body *B = &bodies[id];
    for (int k = 0; k < B->nrec; k++) {
        const int32_t *r = B->rec + 4 * (inv ? B->nrec - 1 - k : k);
        int op = r[0];
        if (id == nb - 1) cur_top = inv ? B->nrec - 1 - k : k;
        if (op >= 100) {
            int n = r[3]; int32_t *sub = malloc(sizeof(int32_t) * (n ? n : 1));
            const int32_t *p = B->pool + r[2];
            for (int i = 0; i < n; i++) sub[i] = map ? map[p[i]] : p[i];
            stk[sp++] = r[1]; run(r[1], sub, inv ^ (op - 100)); sp--;
            if (r[1] >= call_lo && r[1] <= call_hi) { int64_t m = 0; for (int i = 0; i < n; i++) if (ready[sub[i]] > m) m = ready[sub[i]]; printf("call %d inv %d end %lld\n", r[1], inv ^ (op - 100), (long long)m); }
            free(sub);
            continue;
        }
        int a = map ? map[r[1]] : r[1], b = map ? map[r[2]] : r[2], c = map ? map[r[3]] : r[3];
        if (op == 0 || op == 6) {
            note(ready[a], ready[a] + 1, a); ready[a] += 1; if (want_path) lastop[a] = nops;
        } else if (op == 1 || op == 4) {
            int cw = ready[a] >= ready[b] ? a : b; int64_t x = ready[cw] + 1;
            note(ready[cw], x, cw); ready[a] = ready[b] = x; if (want_path) lastop[a] = lastop[b] = nops;
        } else if (op == 3) {
            int cw = ready[a] >= ready[b] ? a : b; int64_t x = ready[cw] + 3;
            note(ready[cw], x, cw); ready[a] = ready[b] = x; if (want_path) lastop[a] = lastop[b] = nops;
        } else {
            int64_t ea = ready[a] + 8, eb = ready[b] + 10, et = ready[c] + 11, x = ea; int cw = a;
            if (eb > x) { x = eb; cw = b; } if (et > x) { x = et; cw = c; }
            note(ready[cw], x, cw); ready[a] = ready[b] = x; ready[c] = x - 1; nccx++;
            if (want_path) lastop[a] = lastop[b] = lastop[c] = nops;
        }
        nops++;
    }
}
typedef struct { int key; int64_t ticks, n; } Row;
static int cmp(const void *a, const void *b) { int64_t d = ((Row*)b)->ticks - ((Row*)a)->ticks; return d > 0 ? 1 : d < 0 ? -1 : 0; }
static void table(const char *name, int32_t *lab, int32_t *onpath, int64_t npath, int nkeys, int top) {
    Row *rows = calloc(nkeys + 1, sizeof(Row));
    for (int i = 0; i <= nkeys; i++) rows[i].key = i - 1;
    for (int64_t i = 0; i < npath; i++) { int64_t o = onpath[i]; rows[lab[o] + 1].ticks += o_dt[o]; rows[lab[o] + 1].n++; }
    qsort(rows, nkeys + 1, sizeof(Row), cmp);
    printf("== %s\n", name);
    for (int i = 0; i < top && i <= nkeys && rows[i].ticks; i++) printf("%6d %10lld ticks %8lld ops\n", rows[i].key, (long long)rows[i].ticks, (long long)rows[i].n);
    free(rows);
}
int main(int argc, char **argv) {
    FILE *f = fopen(argv[1], "rb"); if (!f) { perror("open"); return 1; }
    want_path = argc > 2 && !strcmp(argv[2], "--path");
    if (argc > 4 && !strcmp(argv[2], "--calls")) { call_lo = atoi(argv[3]); call_hi = atoi(argv[4]); }
    int32_t h[3]; fread(h, 4, 2, f); nw = h[0]; nb = h[1];
    bodies = calloc(nb, sizeof(Body));
    for (int i = 0; i < nb; i++) {
        fread(h, 4, 3, f); Body *B = &bodies[i]; B->arity = h[0]; B->nrec = h[1]; B->npool = h[2];
        B->rec = malloc(16 * (size_t)(B->nrec ? B->nrec : 1)); fread(B->rec, 16, B->nrec, f);
        B->pool = malloc(4 * (size_t)(B->npool ? B->npool : 1)); fread(B->pool, 4, B->npool, f);
    }
    ready = calloc(nw, sizeof(int64_t)); lastop = malloc(nw * sizeof(int32_t)); memset(lastop, -1, nw * sizeof(int32_t));
    if (want_path) { cap = 40000000; pred = malloc(cap * 4); o_dt = malloc(cap * 4); o_top = malloc(cap * 4); o_in = malloc(cap * 4); o_l1 = malloc(cap * 4); o_l2 = malloc(cap * 4); }
    run(nb - 1, NULL, 0);
    int64_t ticks = 0; int wmax = 0;
    for (int i = 0; i < nw; i++) if (ready[i] > ticks) { ticks = ready[i]; wmax = i; }
    printf("wires %d ticks %lld ccx %lld T %lld qubit_time %lld ops %lld\n", nw, (long long)ticks, (long long)nccx, (long long)(7 * nccx), (long long)(nw * ticks), (long long)nops);
    if (!want_path) return 0;
    int32_t *onpath = malloc(cap * 4); int64_t np = 0;
    { int64_t next = ticks; for (int32_t o = lastop[wmax]; o >= 0; o = pred[o]) { int64_t in = o_dt[o]; o_dt[o] = (int32_t)(next - in); next = in; onpath[np++] = o; } }
    printf("path ops %lld\n", (long long)np);
    table("top-level record", o_top, onpath, np, bodies[nb - 1].nrec, 8);
    table("innermost def", o_in, onpath, np, nb, 40);
    table("level-1 def", o_l1, onpath, np, nb, 40);
    table("level-2 def", o_l2, onpath, np, nb, 40);
    return 0;
}
