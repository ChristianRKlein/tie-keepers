# -*- coding: utf-8 -*-
"""
schritt7_review2.py -- Step 7: further checks of the collapse
E  Degree-preserving rewiring: is cohesion explained by the degree distribution alone?
F  Mapping between collaboration share and relationship investment:
   D_eff = D(0) * (D(alpha)/D(0))**theta, theta = 0.25, 0.5, 1, 2
G  More replicates (20 seeds, N = 300) and network size (N = 300, 1000, 2000)
   on a finer grid of AI capability around the transition.
Results: results_S7/*.csv
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import networkx as nx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import netzmodell as nm                     # noqa: E402
import zukunft as zk                        # noqa: E402
from schritt6_review import (nets_and_D, up_sweep, lcc, mr_ratio,   # noqa: E402
                             collapse_alpha, ALPHAS)

OUT = os.path.join(HERE, "results_S7")
QUICK = False
SEEDS = [1, 2, 3, 4, 5]
FINE = [0, 1, 2, 2.5, 3, 3.25, 3.5, 3.75, 4, 5, 6, 8]
SIZES = {300: list(range(1, 21)), 1000: [1, 2, 3, 4, 5], 2000: [1, 2]}


def part_E(nets, D):
    rows = []
    for s in SEEDS:
        _, G = up_sweep(nets[s], D, s, keep=True)
        for a in ALPHAS:
            A = G[a]
            g = nx.from_numpy_array(A.astype(int))
            m = g.number_of_edges()
            vals = []
            for k in range(20):
                h = g.copy()
                if m > 2:
                    try:
                        nx.double_edge_swap(h, nswap=10 * m, max_tries=200 * m, seed=100 * s + k)
                    except nx.NetworkXException:
                        pass
                vals.append(len(max(nx.connected_components(h), key=len)) / A.shape[0])
            rows.append(dict(seed=s, alpha=a, lcc_modell=lcc(A), lcc_umverdrahtet=np.mean(vals),
                             lcc_umverdrahtet_sd=np.std(vals), mr=mr_ratio(A)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "S7E_umverdrahtung.csv"), index=False)
    return df


def part_F(nets, D):
    rows = []
    D0 = D[0]
    for th in [0.25, 0.5, 1.0, 2.0]:
        fn = (lambda a, th=th: D0 * (D[a] / D0) ** th)
        for s in SEEDS:
            R, _ = up_sweep(nets[s], D, s, D_fn=fn)
            for r in R:
                r.update(theta=th, seed=s, D_eff=fn(r["alpha"]))
                rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "S7F_zuordnung.csv"), index=False)
    return df


def part_G():
    rows = []
    sizes = {300: [1, 2], 1000: [1]} if QUICK else SIZES
    for N, seeds in sizes.items():
        for s in seeds:
            t0 = time.time()
            res = nm.simulate_alpha_grid("Wissenschaft", s, FINE, N=N)
            A0 = res[0].A.copy()
            Dg = {}
            for a in FINE:
                m = res[a]
                d = m.A.sum(1)
                Dg[a] = np.mean(np.where(d > 0, 1.0 - m.Y_alone() / m.Y_all(), 0.0))
            f = zk.Future(A0, s)
            for a in FINE:
                for _ in range(60):
                    f.step(Dg[a])
                st = f.stats(Dg[a])
                rows.append(dict(N=N, seed=s, alpha=a, D=Dg[a], lcc=st["lcc"],
                                 beziehungen=st["beziehungen"], mr=mr_ratio(f.A),
                                 deg3=float((f.A.sum(1) >= 3).mean())))
            print("  G N=%d seed %d (%.0f s)" % (N, s, time.time() - t0), flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(OUT, "S7G_groesse.csv"), index=False)
    return pd.DataFrame(rows)


def main(parts="EFG"):
    os.makedirs(OUT, exist_ok=True)
    nets, D = nets_and_D(SEEDS)
    if "E" in parts:
        part_E(nets, D)
        print("  E done", flush=True)
    if "F" in parts:
        part_F(nets, D)
        print("  F done", flush=True)
    if "G" in parts:
        part_G()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "EFG")
