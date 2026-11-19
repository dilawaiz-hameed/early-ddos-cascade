"""Experiment E: run NSGA-II (proposed) and the two search baselines with equal evaluation budgets.

Usage: python src/run_search.py [seeds...]
Writes results/search_<method>_s<seed>.json with every evaluated configuration.
"""
import json
import sys
import time
import warnings
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
import os
from data import FlowSet, main_split, MAIN, RESULTS
from search import Evaluator, nsga2, random_search, reference_cost

POP, GENS = int(os.environ.get("CN_POP", 32)), int(os.environ.get("CN_GENS", 24))                      # budget = POP * (GENS + 1) = 800 evaluations per run


def main(seeds):
    os.makedirs(RESULTS, exist_ok=True)
    cic = FlowSet(MAIN)
    split, _ = main_split(cic, nblocks=10)
    ref = reference_cost(cic, split)
    json.dump(ref, open(f"{RESULTS}/reference_cost.json", "w"), indent=1)
    budget = POP * (GENS + 1)
    for seed in seeds:
        for method in ("nsga2", "random", "ga_f1"):
            t = time.time()
            ev = Evaluator(cic, split, ref)
            if method == "nsga2":
                pop, fit = nsga2(ev, POP, GENS, seed=seed)
            elif method == "ga_f1":
                pop, fit = nsga2(ev, POP, GENS, seed=seed, objectives=(0,))
            else:
                random_search(ev, budget, seed=seed)
                pop = []
            final = [{k: v for k, v in c.items() if k != "cols"} for c in pop]
            json.dump(dict(method=method, seed=seed, budget=budget, n_unique=len(ev.log),
                           final_population=final, log=ev.log, minutes=(time.time() - t) / 60),
                      open(f"{RESULTS}/search_{method}_s{seed}.json", "w"))
            print(method, seed, "unique evals", len(ev.log), "%.1f min" % ((time.time() - t) / 60), flush=True)


if __name__ == "__main__":
    main([int(s) for s in sys.argv[1:]] or [0, 1, 2])
