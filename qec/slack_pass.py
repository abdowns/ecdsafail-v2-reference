"""Slack-aware reorder of every call-free body, by rounds of qec/slackopt.c. Same unitary.

    python3 -I -B qec/slack_pass.py <in.json> <out.json> [--jobs N] [--rounds N]
        [--hot-evals N] [--cold-evals N] [--hot-runs N] [--window N] [--window-evals N] [--work DIR]

<in.json> is a circuit, or an upload with the circuit under "circuit"; the output keeps
that shape. It compiles qec/slackopt.c and qec/ticks.c with cc into the work directory.
Each round runs slackopt on every call-free body of the current circuit, in
parallel, then puts the new bodies in and keeps them only if the whole circuit's ticks
fall; if they do not, the bodies go in one at a time, largest predicted gain first, and
each stays only if the ticks do not rise. A body that runs at least --hot-runs times may score
--hot-evals orders and the others --cold-evals; a body longer than --window gates is
searched in windows of that size with --window-evals each. The budgets count scored
orders, not seconds, so the result does not depend on the machine or on --jobs. The next round measures arrival times
and tails on the new circuit and searches again from the new orders.

slackopt checks each new body against the old one on random basis states before it
writes it, and it only makes moves that reorder commuting gates or swap the controls of
a CCX, so the circuit's function does not change.
"""
import argparse
import json
import subprocess
import sys
import time
from array import array
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = {"X": 0, "CX": 1, "CCX": 2, "SWAP": 3}
NAME = {v: k for k, v in CODE.items()}


def to_records(ops):
    rec, pool = array("i"), array("i")
    for r in ops:
        if "call" in r:
            w = r["wires"]
            rec.extend((100 + bool(r.get("inverse", False)), r["call"], len(pool), len(w)))
            pool.extend(w)
            continue
        o = r["op"]
        if o == "CCX":
            rec.extend((2, r["control1"], r["control2"], r["target"]))
        elif o == "CX":
            rec.extend((1, r["control"], r["target"], r["target"]))
        elif o == "SWAP":
            rec.extend((3, r["wire1"], r["wire2"], r["wire2"]))
        elif o == "X":
            rec.extend((0, r["target"], r["target"], r["target"]))
        else:
            sys.exit(f"unsupported op {o}")
    return rec, pool


def to_ops(rec):
    ops = []
    for k in range(0, len(rec), 4):
        o, a, b, c = rec[k:k + 4]
        if o == 2:
            ops.append({"op": "CCX", "control1": a, "control2": b, "target": c})
        elif o == 1:
            ops.append({"op": "CX", "control": a, "target": b})
        elif o == 3:
            ops.append({"op": "SWAP", "wire1": a, "wire2": b})
        elif o == 0:
            ops.append({"op": "X", "target": a})
        else:
            sys.exit(f"record {o} is not a gate")
    return ops


def write_bin(path, num_wires, bodies):
    out = array("i", (num_wires, len(bodies)))
    for arity, rec, pool in bodies:
        out.extend((arity, len(rec) // 4, len(pool)))
        out.extend(rec)
        out.extend(pool)
    with open(path, "wb") as fh:
        out.tofile(fh)


def ticks(binary, path):
    out = subprocess.run([binary, str(path)], capture_output=True, text=True, check=True).stdout.split()
    return int(out[out.index("ticks") + 1])


def run_counts(bodies):
    runs = [0] * len(bodies)
    runs[-1] = 1
    for d in range(len(bodies) - 1, -1, -1):
        rec = bodies[d][1]
        for k in range(0, len(rec), 4):
            if rec[k] >= 100:
                runs[rec[k + 1]] += runs[d]
    return runs


def build(work):
    """Compile slackopt.c and ticks.c into work; returns their paths."""
    out = []
    for name in ("slackopt", "ticks"):
        exe = work / name
        subprocess.run(["cc", "-O2", "-o", str(exe), str(HERE / f"{name}.c")], check=True)
        out.append(str(exe))
    return out


def apply(circuit, work, jobs=6, rounds=3, hot_evals=300000, cold_evals=40000, hot_runs=50, scenarios=6,
          margin=24, slack=0, window=8192, window_evals=20000, min_ops=30, only=None, log=True):
    """Run the pass on circuit in place. Returns {"ticks_in", "ticks_out", "bodies_changed"}."""
    say = print if log else (lambda *a, **k: None)
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    slackopt, ticks_bin = build(work)
    t0 = time.time()
    defs = circuit["definitions"]
    nw = circuit["num_wires"]
    bodies = [(d["arity"],) + to_records(d["operations"]) for d in defs]
    bodies.append((nw,) + to_records(circuit["operations"]))
    runs = run_counts(bodies)
    targets = [d for d in range(len(defs))
               if runs[d] and len(bodies[d][1]) // 4 >= min_ops
               and all(bodies[d][1][k] < 100 for k in range(0, len(bodies[d][1]), 4))
               and (not only or d in only)]
    targets.sort(key=lambda d: -runs[d] * len(bodies[d][1]))
    cur = work / "cur.bin"
    write_bin(cur, nw, bodies)
    first = best = ticks(ticks_bin, cur)
    say(json.dumps({"ticks_in": best, "bodies": len(targets)}), flush=True)
    changed = set()

    def search(job):
        d, rnd = job
        gates = len(bodies[d][1]) // 4
        evals = window_evals if gates > window else hot_evals if runs[d] >= hot_runs else cold_evals
        out = work / f"r{rnd}.d{d}.rec"
        p = subprocess.run([slackopt, str(cur), str(d), str(out), str(evals), str(scenarios),
                            str(1000 * rnd + d), str(margin), str(slack), str(window)], capture_output=True, text=True)
        if p.returncode:
            return d, None, p.stderr.strip()
        rec = array("i")
        with open(out, "rb") as fh:
            rec.frombytes(fh.read())
        out.unlink()
        return d, rec, json.loads(p.stdout)

    for rnd in range(rounds):
        found = {}
        with ThreadPoolExecutor(jobs) as pool:
            for d, rec, info in pool.map(search, [(d, rnd) for d in targets]):
                if rec is None:
                    raise SystemExit(f"definition {d}: slackopt failed: {info}")
                if rec != bodies[d][1]:
                    found[d] = (rec, (info["P"][0] - info["P"][1]) * runs[d] / info["scenarios"])
                if runs[d] >= hot_runs or len(rec) // 4 > window:
                    say(f"  def {d}: {json.dumps(info)}", flush=True)
        old = {d: bodies[d] for d in found}
        for d, (rec, _) in found.items():
            bodies[d] = (old[d][0], rec, old[d][2])
        write_bin(cur, nw, bodies)
        t = ticks(ticks_bin, cur)
        predicted = round(sum(g for _, g in found.values()))
        if t < best:
            say(json.dumps({"round": rnd, "bodies": len(found), "predicted": predicted, "ticks": t, "kept": "all",
                            "seconds": round(time.time() - t0)}), flush=True)
            best = t
            changed.update(found)
            continue
        for d in found:
            bodies[d] = old[d]
        kept = 0
        for d in sorted(found, key=lambda d: (-found[d][1], d)):
            bodies[d] = (old[d][0], found[d][0], old[d][2])
            write_bin(cur, nw, bodies)
            t = ticks(ticks_bin, cur)
            if t <= best:
                kept += t < best
                best = t
                changed.add(d)
            else:
                bodies[d] = old[d]
        write_bin(cur, nw, bodies)
        say(json.dumps({"round": rnd, "bodies": len(found), "predicted": predicted, "ticks": best, "kept": kept,
                        "seconds": round(time.time() - t0)}), flush=True)
        if not kept:
            break
    for d in sorted(changed):
        defs[d]["operations"] = to_ops(bodies[d][1])
    return {"ticks_in": first, "ticks_out": best, "bodies_changed": len(changed)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--hot-evals", type=int, default=300000)
    ap.add_argument("--cold-evals", type=int, default=40000)
    ap.add_argument("--hot-runs", type=int, default=50)
    ap.add_argument("--window", type=int, default=8192, help="search longer bodies in windows of this many gates")
    ap.add_argument("--window-evals", type=int, default=20000)
    ap.add_argument("--only", default="", help="comma list of definitions")
    ap.add_argument("--work", default="")
    a = ap.parse_args()
    with open(a.src) as fh:
        doc = json.load(fh)
    res = apply(doc.get("circuit", doc), a.work or (a.dst + ".work"), jobs=a.jobs, rounds=a.rounds,
                hot_evals=a.hot_evals, cold_evals=a.cold_evals, hot_runs=a.hot_runs, window=a.window,
                window_evals=a.window_evals, only={int(x) for x in a.only.split(",") if x})
    with open(a.dst, "w") as fh:
        json.dump(doc, fh, separators=(",", ":"))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
