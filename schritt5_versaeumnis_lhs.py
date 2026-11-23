# -*- coding: utf-8 -*-
"""
schritt5_versaeumnis_lhs.py -- Step 5: single lapse test and Latin hypercube (200 combinations)
(1) Single lapse: one partner skips one period of investment; survival of the
    relationship is compared with an untouched copy of the same network.
(2) Latin hypercube over ten parameters; predicted (rule) vs simulated collapse point.
Results: results_S5/S5_versaeumnis.csv, results_S5/S5_lhs200.csv
"""
import os
import sys
import copy
import time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import zukunft as zk            # noqa: E402
import schritt4_regel as s4     # noqa: E402

OUT = os.path.join(HERE, "results_S5")
QUICK = False
SEEDS = [1, 2, 3, 4, 5]
N_LHS = 200


def main():
    os.makedirs(OUT, exist_ok=True)
    seeds = SEEDS[:2] if QUICK else SEEDS
    n_lhs = 20 if QUICK else N_LHS
    ALPHAS = s4.ALPHAS
    t0 = time.time()
    nets, D = s4.networks_and_dependence("Wissenschaft", [1, 2, 3, 4, 5])
    # (1) single lapse
    rows = []
    for s in seeds:
        m = zk.Future(nets[s], s)
        for a in ALPHAS:
            for _ in range(60):
                m.step(D[a])
            iu, ju = np.nonzero(np.triu(m.A, 1))
            if len(iu) == 0:
                continue
            rng = np.random.default_rng(1000 * s + a)
            pick = rng.choice(len(iu), size=min(30, len(iu)), replace=False)
            pairs = [(int(iu[k]), int(ju[k])) if rng.random() < 0.5 else (int(ju[k]), int(iu[k])) for k in pick]
            for arm in ("Kontrolle", "Versaeumnis"):
                for rep in range(4):
                    mm = copy.deepcopy(m)
                    mm.rng = np.random.default_rng(77 + rep + 10 * s + a)
                    if arm == "Versaeumnis":
                        mm.force_lapse = pairs
                    alive_after = {}
                    for t in range(1, 25):
                        mm.step(D[a])
                        if t in (1, 6, 12, 24):
                            alive_after[t] = np.mean([mm.A[i, j] for i, j in pairs])
                    for t, v in alive_after.items():
                        rows.append(dict(seed=s, alpha=a, arm=arm, rep=rep, perioden=t, ueberlebt=v))
        print("  lapse test seed", s, "%.0f s" % (time.time() - t0), flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "S5_versaeumnis.csv"), index=False)
    # (2) Latin hypercube
    rows = []
    for k, par in enumerate(s4.lhs(n_lhs, np.random.default_rng(2026))):
        R = pd.DataFrame(zk.sweep(nets[1], D.get, ALPHAS, 60, 100 + k, back=False, **par))
        L = dict(zip(R["alpha"], R["lcc"]))
        Dg = dict(zip(R["alpha"], R["beziehungen"]))
        pp = dict(zk.DEFAULTS)
        pp.update(par)
        jumps = [((L[a0] - L[a1]) / max(L[a0], 1e-9), (Dg[a0] - Dg[a1]) / max(Dg[a0], 1e-9))
                 for a0, a1 in zip(ALPHAS[:-1], ALPHAS[1:])]
        dl, dk = max(jumps, key=lambda z: z[0])
        row = dict(par)
        row.update(lcc_alpha0=L[0], lcc_alpha8=L[8], alpha_kollaps_sim=s4.collapse_alpha_sim(L),
                   alpha_kollaps_regel=s4.collapse_alpha_theory(par["mu"], D, pp),
                   groesster_sprung_zusammenhalt=dl, beziehungsverlust_dabei=dk)
        rows.append(row)
        if (k + 1) % 20 == 0:
            print("  LHS", k + 1, "%.0f s" % (time.time() - t0), flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "S5_lhs200.csv"), index=False)


if __name__ == "__main__":
    main()
