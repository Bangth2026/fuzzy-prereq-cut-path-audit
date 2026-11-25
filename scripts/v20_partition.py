#!/usr/bin/env python3
"""Release 6.1.0. Partition residual V_res with the row null drawn per seed triple on all ten graphs with
an informative grouping (Section 8.1, Supplementary Table S4e).

Release 5.2.0 (scripts/v18_partition.py) gave V_res over the six seed triples with one row null
partitioned with seeds 0-2, so its range mixed the seed variation of the graded side only. Here, as in
scripts/v19_interior.py at s = 1: for each of the six disjoint seed triples, V_res = ARI(best of the
triple on mu_E) - mean ARI of ten row permutations of mu_E, each partitioned with the best of the same
triple (s = 1 on the segments of the graph's quantile grid; random generator seeded 54020, as for the point s = 1 of v19_interior, so the values for Junyi
transition and EdNet co-practice reproduce results/v19/interior.json).
Writes results/v20/partition_triples.json.
"""
from __future__ import annotations
import json, os, sys
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.dirname(__file__))
from v19_interior import _point  # noqa: E402
from v19_robust import INFORMATIVE  # noqa: E402
from v16_units import segment_basis  # noqa: E402
from v15_value_paths import load  # noqa: E402

OUT = "results/v20/partition_triples.json"


def main():
    os.makedirs("results/v20", exist_ok=True)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    todo = [n for n in INFORMATIVE if n not in res]
    with Pool(os.cpu_count()) as pool:
        for name, (k, obs, nul) in zip(todo, pool.imap(_point, [(n, [1.0] * segment_basis(load(n)[0].mu)[1].shape[1], 0) for n in todo])):
            R = np.array(obs) - np.array(nul)
            res[name] = {"graded_triples": obs, "row_null_triples": nul, "V_res_triples": R.tolist(),
                         "V_res_median": float(np.median(R)), "V_res_range": [float(R.min()), float(R.max())]}
            print(name, res[name]["V_res_median"], res[name]["V_res_range"], flush=True)
            json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
