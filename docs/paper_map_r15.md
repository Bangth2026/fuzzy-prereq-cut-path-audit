# Paper map, review round 15 (release 6.1.0)

| Review item | Manuscript place | Script | Result |
|---|---|---|---|
| M1 next-concept guide is not Algorithm 2 | Section 8.4, Table 7b, Table 8 (last row), Algorithm 2 step 6, Table 9 D6, Section 9.3, Conclusion; Supplementary S13, Table S13b | `scripts/v20_nextconcept.py` | `results/v20/nextconcept.json` |
| M1 crisp with cross-fitted threshold | Section 8.4, Table 7a, Table 8, Table 9 D7; Supplementary S13 | `scripts/v20_kbs.py A` | `results/v20/kbs.json` (`A`) |
| M2 regret 0.028 location | Section 8.4 (rewritten from the new candidate set) | `scripts/v20_nextconcept.py` | `results/v20/nextconcept.json` |
| M3 partition V_res with per-triple null | Section 8.1, Table 3 caption, Table S4e | `scripts/v20_partition.py` | `results/v20/partition_triples.json` |
| M3 bootstrap resamples | Table 8 caption, Table 3 caption, Supplementary S13 | — | — |
| m6 999 permutation draws (partitions) | Table 8 | `scripts/v20_kbs.py P` | `results/v20/kbs.json` (`P`) |
| m7 bootstrap p with +1 correction | Table 8, Supplementary S13 | `scripts/v20_kbs.py C` | `results/v20/kbs.json` (`C`) |
