#!/usr/bin/env python3
"""Release 6.1.0. Prerequisite recommendation with a cross-fitted crisp policy, and the p-values of the
primary-results table with an explicit +1 correction (Sections 8.4 and 8.5, Supplementary Section S13).

A. As scripts/v19_kbs.py part A (same 351 targets, same tie-breaking draws, same 200 halvings, seed 5403),
   with one more policy:
     crisp, cross-fitted   hop distance at the pruning threshold chosen by nDCG@3 on one half of the
                           targets, evaluated on the other half.
   It separates the gain of the guide over the crisp default into a threshold part (crisp cross-fitted
   minus crisp default) and a grades part (guide minus crisp cross-fitted).
C. Two-way cluster bootstrap over exercises (2,000 resamples, seed 5404, as in release 5.4.0); the
   two-sided p-value is now p = min(1, 2 (1 + min(#{d <= 0}, #{d >= 0})) / (B + 1)), which cannot fall
   below 2 / 2001 = 0.0010. Release 5.4.0 used max(2 min(share <= 0, share >= 0), 1/B).
P. Row-permutation tests of the two partition rows with 999 draws instead of 99 (best of seeds 0-2,
   statistic ARI with the grouping, p = (1 + #{null >= observed}) / (1 + draws)), seed 5405.
Writes results/v20/kbs.json.
"""
from __future__ import annotations
import json, os, sys
from collections import Counter
from multiprocessing import Pool
import numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.dirname(__file__))
from data.loaders import load_junyi_expert  # noqa: E402
from composition_robustness import mats  # noqa: E402
from v18_reach import load_tau, TAGS  # noqa: E402
from v19_kbs import ndcg_parts  # noqa: E402

OUT = "results/v20/kbs.json"


def part_a(res):
    _, ann = load_junyi_expert()
    A = ann.Exercise_A.astype(str).to_numpy(); Bn = ann.Exercise_B.astype(str).to_numpy(); y = ann.Prerequisite_avg.to_numpy()
    X, E = {}, {}
    for t in TAGS:
        G, lab = load_tau(t); idx = {l: i for i, l in enumerate(lab)}; E[t] = int(G.m)
        ok = np.array([a in idx and b in idx for a, b in zip(A, Bn)])
        ia = np.array([idx.get(a, 0) for a in A]); ib = np.array([idx.get(b, 0) for b in Bn])
        M, _ = mats(G)
        for k in ("hop_baseline", "product"):
            X[(k, t)] = np.where(ok, M[k][ib, ia], 0.0)
    tg = pd.Series(range(len(Bn))).groupby(Bn).apply(list)
    groups = {g: np.array(v) for g, v in tg.items() if len(v) >= 2}
    keys = sorted(groups)
    rng = np.random.default_rng(5403)
    per = {kt: ndcg_parts(X[kt], y, groups, rng) for kt in X}
    mean = lambda kt, sub, j=0: float(np.mean([per[kt][g][j] for g in sub]))
    rs = np.random.default_rng(5403)
    rows, choices, crisp_choices = [], Counter(), Counter()
    pols = ["crisp-default", "graded-fixed", "crisp-crossfit", "guide"]
    for _ in range(200):
        h = rs.random(len(keys)) < 0.5; hA = [g for g, b in zip(keys, h) if b]; hB = [g for g, b in zip(keys, h) if not b]
        tc = max(TAGS, key=lambda t: mean(("hop_baseline", t), hA)); tgd = max(TAGS, key=lambda t: mean(("product", t), hA))
        pick = ("product", tgd) if mean(("product", tgd), hA) > mean(("hop_baseline", tc), hA) else ("hop_baseline", tc)
        choices[f"{pick[0]}|{pick[1]}"] += 1; crisp_choices[tc] += 1
        rows.append([[mean(kt, hB, j) for j in range(3)] for kt in (("hop_baseline", "0.2"), ("product", "0"), ("hop_baseline", tc), pick)])
    arr = np.array(rows)  # halvings x policy x metric
    pct = lambda v: [float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), float(np.mean(v > 0))]
    names = ["nDCG@3", "P@1", "unnecessary@3"]
    res["A"] = {"heldout": {p: {names[j]: float(arr[:, i, j].mean()) for j in range(3)} for i, p in enumerate(pols)},
                "guide_minus_crisp_default": {names[j]: pct(arr[:, 3, j] - arr[:, 0, j]) for j in range(3)},
                "threshold_part_crisp_crossfit_minus_crisp_default": {names[j]: pct(arr[:, 2, j] - arr[:, 0, j]) for j in range(3)},
                "grades_part_guide_minus_crisp_crossfit": {names[j]: pct(arr[:, 3, j] - arr[:, 2, j]) for j in range(3)},
                "guide_minus_graded_fixed": {names[j]: pct(arr[:, 3, j] - arr[:, 1, j]) for j in range(3)},
                "guide_choices": dict(choices), "crisp_crossfit_thresholds": dict(crisp_choices), "edges": E,
                "crisp_crossfit_edges_stored_mean": float(sum(E[t] * c for t, c in crisp_choices.items()) / 200)}
    print(json.dumps(res["A"], indent=1), flush=True)
    return A, Bn, y, X


def part_c(res, A, Bn, y, X):
    rb = np.random.default_rng(5404)
    exs = np.unique(np.r_[A, Bn]); pos = {e: i for i, e in enumerate(exs)}
    pa = np.array([pos[a] for a in A]); pb = np.array([pos[b] for b in Bn])
    comps = {"threshold_gain_hop_0.05_minus_0.2": (("hop_baseline", "0.05"), ("hop_baseline", "0.2")),
             "best_graded_minus_best_crisp": (("product", "0"), ("hop_baseline", "0.05")),
             "product_minus_hop_at_0.05": (("product", "0.05"), ("hop_baseline", "0.05"))}
    D = {k: [] for k in comps}
    for _ in range(2000):
        m = rb.multinomial(exs.size, np.full(exs.size, 1 / exs.size)); w = m[pa] * m[pb]; kk = w > 0; yy = np.repeat(y[kk], w[kk])
        cache = {}
        for name, (a, b) in comps.items():
            for kt in (a, b):
                if kt not in cache:
                    cache[kt] = spearmanr(np.repeat(X[kt][kk], w[kk]), yy).statistic
            D[name].append(cache[a] - cache[b])
    C = {}
    for name, (a, b) in comps.items():
        d = np.array(D[name]); B = d.size
        lo, hi = int((d <= 0).sum()), int((d >= 0).sum())
        C[name] = {"estimate": float(spearmanr(X[a], y).statistic - spearmanr(X[b], y).statistic),
                   "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                   "n_le_0": lo, "n_ge_0": hi, "B": B, "p_boot": float(min(1.0, 2 * (1 + min(lo, hi)) / (B + 1)))}
    res["C"] = C; print(C, flush=True)


def _perm_draw(args):
    name, k = args
    from v19_robust import best3, grouping
    from v15_value_paths import load
    from fuzzykg.nulltest import permute_rows
    from sklearn.metrics import adjusted_rand_score as ari
    G, _ = load(name); has, gl = grouping(name)
    rng = np.random.default_rng([5405, k])
    return float(ari(gl, best3(permute_rows(G, rng))[has]))


def part_p(res, draws=999):
    from v19_robust import best3, grouping
    from v15_value_paths import load
    from sklearn.metrics import adjusted_rand_score as ari
    P = res.setdefault("P", {})
    for name in ("junyi_transition", "ednet_copractice"):
        if name in P:
            continue
        G, _ = load(name); has, gl = grouping(name)
        o = float(ari(gl, best3(G)[has]))
        with Pool(os.cpu_count()) as pool:
            nn = pool.map(_perm_draw, [(name, k) for k in range(draws)], chunksize=8)
        nn = np.array(nn)
        P[name] = {"observed": o, "null_mean": float(nn.mean()), "null_max": float(nn.max()),
                   "n_ge_observed": int((nn >= o).sum()), "draws": draws, "p": float((1 + (nn >= o).sum()) / (1 + draws))}
        print(name, P[name], flush=True)
        json.dump(res, open(OUT, "w"), indent=1)


def main(parts):
    os.makedirs("results/v20", exist_ok=True)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    if "A" in parts or "C" in parts:
        A, Bn, y, X = part_a(res)
        part_c(res, A, Bn, y, X)
        json.dump(res, open(OUT, "w"), indent=1)
    if "P" in parts:
        part_p(res)
    json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:] or ["A", "C", "P"])
