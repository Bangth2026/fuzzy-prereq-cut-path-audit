#!/usr/bin/env python3
"""Release 6.1.0. Next-concept recommendation with the full candidate set of Algorithm 2 (Section 8.4,
Supplementary Section S13).

Release 5.4.0 (scripts/v19_kbs.py, part B) let the "guide" pick, on validation learners, among hop
distance and the nine t-norm scores. That is selection on validation data, not Algorithm 2: it never
used the usable share u of step 1, and the candidate set lacked the direct edge and the hop-penalised
score that step 6 and row D6 prescribe when u < 0.1. This script rebuilds the eleven held-out graphs
of results/heldout_*.json (same split by learner hash, same construction and parameters) and adds:

  direct edge        mu_E(c_prev, c) as the score (0 where there is no edge);
  hop-penalised      best path under edge length lam + (1 - lam)(-ln mu_E), lam in LAMBDAS, with lam
                     chosen on the validation learners (cross-fitted lam; lam = 0 is the product,
                     lam = 1 hop count); fuzzykg.pathscore.hop_penalised;
  u                  usable share of Proposition 3 on the training graph (connectivity level at 95% of
                     the connected ordered pairs; fuzzykg.pathscore.connectivity_level).

Policies, all evaluated by test MRR (mean over test learners of the expected reciprocal rank):
  crisp              hop distance;
  graded             product best-path score;
  selection          best validation MRR among all 13 candidates (hop, direct edge, hop-penalised at
                     the cross-fitted lam, ten t-norm scores), no constraint;
  Algorithm 2        step 1 and step 6: if u < 0.1, the better on validation of the direct edge and
                     the hop-penalised score with lam > 0 chosen on validation (never max-min, never
                     hop count alone); otherwise the better on validation of hop distance, the product
                     and the hop-penalised score at the cross-fitted lam.
Regret of a policy = best test MRR among the 13 candidates minus the policy's test MRR. The t-norm
test and validation MRRs are recomputed here; they reproduce the stored release-4 values (checked
and reported as max_abs_diff_vs_release4).
Writes results/v20/nextconcept.json.
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.dirname(__file__))
from data.loaders import load_assistments, load_junyi  # noqa: E402
from fuzzykg.construction import (ConstructionConfig, TransitionConfig, build_fuzzy_graph,  # noqa: E402
                                  build_transition_graph, preprocess)
from fuzzykg.pathscore import hop_penalised, score_from_length, connectivity_level  # noqa: E402
from heldout_transitions import OPERATORS, split_id, strength_matrix, evaluate, transitions_of  # noqa: E402
from scipy.sparse import csr_matrix  # noqa: E402
from scipy.sparse.csgraph import shortest_path  # noqa: E402

LAMBDAS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
GRAPHS = [("2009", "item", "copractice"), ("2009", "kc", "copractice"), ("2009", "kc", "transition"),
          ("2015", "kc", "copractice"), ("2015", "kc", "transition"), ("2017", "item", "copractice"),
          ("2017", "item", "transition"), ("2017", "kc", "copractice"), ("2017", "kc", "transition"),
          ("junyi", "kc", "copractice"), ("junyi", "kc", "transition")]
OUT = "results/v20/nextconcept.json"


def tag_of(rel, lev, con):
    t = f"{rel}_{lev}" if rel != "junyi" else "junyi"
    return t + ("_transition" if con == "transition" else "")


def derive(r):
    """Policies from the stored candidates and lambda curve. lam > 0 under step 6 means 0 < lam < 1:
    lam = 1 is hop count itself, which step 6 excludes when u < 0.1."""
    cand = {k: v for k, v in r["candidates"].items() if k != "hop_penalised"}
    curve = {float(l): v for l, v in r["hop_penalised_curve"].items()}
    lam_all = max(curve, key=lambda l: curve[l]["val"])
    lam_pos = max([l for l in curve if 0 < l < 1], key=lambda l: curve[l]["val"])
    cand["hop_penalised"] = dict(curve[lam_all], lam=lam_all)
    hp_pos = dict(curve[lam_pos], lam=lam_pos)
    best = max(cand, key=lambda k: cand[k]["test"]); sel = max(cand, key=lambda k: cand[k]["val"])
    if r["u"] < 0.1:
        opts = {"direct_edge": cand["direct_edge"], "hop_penalised": hp_pos}
    else:
        opts = {"hop": cand["hop"], "product": cand["product"], "hop_penalised": cand["hop_penalised"]}
    alg = max(opts, key=lambda k: opts[k]["val"]); bt = cand[best]["test"]
    r.update({"candidates": cand, "lam_crossfit": lam_all, "lam_crossfit_in_(0,1)": lam_pos, "best_on_test": best, "best": bt,
              "policies": {"crisp": {"choice": "hop", "test": cand["hop"]["test"]},
                           "graded": {"choice": "product", "test": cand["product"]["test"]},
                           "selection": {"choice": sel + (f" (lam={cand[sel]['lam']})" if "lam" in cand[sel] else ""), "test": cand[sel]["test"]},
                           "algorithm2": {"choice": alg + (f" (lam={opts[alg]['lam']})" if "lam" in opts[alg] else ""), "test": opts[alg]["test"]}}})
    for p in r["policies"].values():
        p["regret"] = bt - p["test"]
    return r


def run(rel, lev, con, data_cache):
    tag = tag_of(rel, lev, con)
    old = json.load(open(f"results/heldout_{tag}.json"))
    cfg = ConstructionConfig(); cfg.horizon = old["horizon"]; cfg.min_support = old["min_support"]
    key = (rel, lev)
    if key not in data_cache:
        data_cache.clear()
        data_cache[key] = load_junyi() if rel == "junyi" else load_assistments(rel, lev)
    d = preprocess(data_cache[key], cfg)
    d["split"] = d["user_id"].map(split_id)
    train = d[d.split == "train"].drop(columns=["split"])
    if con == "copractice":
        G, _ = build_fuzzy_graph(train, cfg, preprocessed=True)
    else:
        G, _ = build_transition_graph(train, TransitionConfig(), preprocessed=True)
    node_index = {lab: i for i, lab in enumerate(G.labels)}
    val_uids = sorted(d[d.split == "val"].user_id.unique())
    test_uids = sorted(d[d.split == "test"].user_id.unique())[:2000]
    val_tr, _ = transitions_of(d, node_index, val_uids)
    test_tr, _ = transitions_of(d, node_index, test_uids)
    cand, diffs = {}, []

    def add(name, S):
        cand[name] = {"val": evaluate(S, val_tr, B=10)[0]["mrr"], "test": evaluate(S, test_tr, B=10)[0]["mrr_learner_mean"]}

    for k, (nm, p) in OPERATORS.items():
        S, _ = strength_matrix(G, nm, p)
        add(k, S); diffs.append(abs(cand[k]["test"] - old["operators"][k]["mrr_learner_mean"]))
    A = G.adjacency().toarray(); np.fill_diagonal(A, -1.0)
    add("direct_edge", A); diffs.append(abs(cand["direct_edge"]["test"] - old["baselines"]["direct_edge"]["mrr_learner_mean"]))
    Ah = csr_matrix((np.ones(G.m), (G.src, G.dst)), shape=(G.n, G.n))
    Dh = shortest_path(Ah, directed=True, unweighted=True)
    Hs = np.where(np.isfinite(Dh), 1.0 / (1.0 + Dh), 0.0); np.fill_diagonal(Hs, -1.0)
    add("hop", Hs); diffs.append(abs(cand["hop"]["test"] - old["baselines"]["hop_baseline"]["mrr_learner_mean"]))
    curve = {}
    for lam in LAMBDAS:
        S = score_from_length(hop_penalised(G, lam)); np.fill_diagonal(S, S.min() - 1.0)
        curve[lam] = {"val": evaluate(S, val_tr, B=10)[0]["mrr"], "test": evaluate(S, test_tr, B=10)[0]["mrr_learner_mean"]}
    u = connectivity_level(G)["usable_share"]
    r = {"construction": con, "n_nodes": int(G.n), "n_edges": int(G.m), "u": float(u),
         "max_abs_diff_vs_release4": float(max(diffs)), "candidates": cand,
         "hop_penalised_curve": {str(l): v for l, v in curve.items()}}
    derive(r)
    return tag, r


def main():
    os.makedirs("results/v20", exist_ok=True)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    cache = {}
    for g in GRAPHS:
        tag = tag_of(*g)
        if tag in res.get("graphs", {}):
            derive(res["graphs"][tag])
            continue
        t0 = time.time(); tag, r = run(*g, cache)
        res.setdefault("graphs", {})[tag] = r
        print(tag, f"{time.time()-t0:.0f}s u={r['u']:.3f} diff={r['max_abs_diff_vs_release4']:.2e} best={r['best_on_test']}",
              {k: (v["choice"], round(v["regret"], 4)) for k, v in r["policies"].items()}, flush=True)
        json.dump(res, open(OUT, "w"), indent=1)
    G = res["graphs"]
    res["summary"] = {pol: {"mean_regret": float(np.mean([g["policies"][pol]["regret"] for g in G.values()])),
                            "max_regret": float(np.max([g["policies"][pol]["regret"] for g in G.values()])),
                            "within_0.001": int(sum(g["policies"][pol]["regret"] <= 0.001 for g in G.values()))}
                      for pol in ("crisp", "graded", "selection", "algorithm2")}
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(res["summary"], indent=1))


if __name__ == "__main__":
    main()
