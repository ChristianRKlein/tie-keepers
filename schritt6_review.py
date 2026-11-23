# -*- coding: utf-8 -*-
"""
schritt6_review.py -- Robustheitspruefungen nach externer Begutachtung
=====================================================================
A  Welche Kanten gehen beim Einbruch verloren?  Vergleich mit zufaelligem
   Entfernen derselben Zahl von Kanten; Bruecken (Kantenbetweenness,
   Kanten zwischen Gemeinschaften); Molloy-Reed auf den echten Gradverteilungen.
B  Heterogene Startnetze (Erdos-Renyi, Barabasi-Albert, bimodal) und
   unterschiedliche Pflegekosten zwischen Personen.
C  Gegenmechanismus: KI schafft neue gemeinsame Arbeit (Anteil gamma der
   verlorenen Nebenprodukt-Investition kehrt zurueck) bzw. Zeitgewinn
   fuehrt zu neuen Kontakten ausserhalb des eigenen Umfelds.
D  Hysterese mit seltenen Zufallskontakten und sehr langen Laeufen bei festem alpha.
Ergebnisse: results_S6/*.csv
"""
import os
import sys
import numpy as np
import pandas as pd
import networkx as nx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import netzmodell as nm   # noqa: E402
import zukunft as zk      # noqa: E402
from struktur import largest_component  # noqa: E402

OUT = os.path.join(HERE, "results_S6")
os.makedirs(OUT, exist_ok=True)
ALPHAS = [0, 1, 2, 3, 4, 6, 8, 12, 16]
T_PER = 60
SEEDS = [1, 2, 3, 4, 5]
N_RAND = 300
QUICK = False


def nets_and_D(seeds):
    nets, Ds = {}, []
    for s in seeds:
        res = nm.simulate_alpha_grid("Wissenschaft", s, ALPHAS)
        nets[s] = res[0].A.copy()
        row = []
        for a in ALPHAS:
            m = res[a]
            d = m.A.sum(1)
            row.append(np.mean(np.where(d > 0, 1.0 - m.Y_alone() / m.Y_all(), 0.0)))
        Ds.append(row)
    return nets, dict(zip(ALPHAS, np.mean(Ds, axis=0)))


def lcc(A):
    return len(largest_component(A)) / A.shape[0]


def mr_ratio(A):
    d = A.sum(1).astype(float)
    return (d ** 2).mean() / max(d.mean(), 1e-12)


def up_sweep(A0, D, seed, alphas=ALPHAS, T=T_PER, D_fn=None, per_alpha=None, keep=False, **kw):
    m = zk.Future(A0, seed, **kw)
    rows, G = [], {}
    for a in alphas:
        if per_alpha:
            per_alpha(m, a)
        Da = D_fn(a) if D_fn else D[a]
        for _ in range(T):
            m.step(Da)
        st = m.stats(Da)
        st.update(alpha=a, mr=mr_ratio(m.A), deg3=float((m.A.sum(1) >= 3).mean()))
        rows.append(st)
        if keep:
            G[a] = m.A.copy()
    return rows, G


def collapse_alpha(rows):
    L = {r["alpha"]: r["lcc"] for r in rows}
    base = L[0]
    for a in ALPHAS:
        if a in L and L[a] < 0.5 * base:
            return a
    return np.inf


# ---------------------------------------------------------------------------
def part_A(nets, D):
    rows, summ = [], []
    for s in SEEDS:
        R, G = up_sweep(nets[s], D, s, keep=True)
        L = [r["lcc"] for r in R]
        k = int(np.argmax([L[i] - L[i + 1] for i in range(len(L) - 1)]))
        a0, a1 = ALPHAS[k], ALPHAS[k + 1]
        G0, G1 = G[a0], G[a1]
        lost = np.argwhere(np.triu(G0 & ~G1, 1))
        gained = np.argwhere(np.triu(G1 & ~G0, 1))
        E0 = np.argwhere(np.triu(G0, 1))
        rng = np.random.default_rng(1000 + s)
        rand_l = []
        for _ in range(N_RAND):
            A = G0.copy()
            pick = E0[rng.choice(len(E0), size=len(lost), replace=False)]
            A[pick[:, 0], pick[:, 1]] = A[pick[:, 1], pick[:, 0]] = False
            A[gained[:, 0], gained[:, 1]] = A[gained[:, 1], gained[:, 0]] = True
            rand_l.append(lcc(A))
        # Bruecken
        g = nx.from_numpy_array(G0.astype(int))
        eb = nx.edge_betweenness_centrality(g)
        vals = np.array([eb[tuple(sorted(e))] for e in map(tuple, E0)])
        rank = pd.Series(vals).rank(pct=True).values
        idx = {tuple(e): i for i, e in enumerate(map(tuple, E0))}
        lost_rank = np.array([rank[idx[tuple(e)]] for e in map(tuple, lost)])
        comms = nx.algorithms.community.greedy_modularity_communities(g)
        cid = np.zeros(G0.shape[0], int)
        for c, nodes in enumerate(comms):
            cid[list(nodes)] = c
        inter_all = np.mean(cid[E0[:, 0]] != cid[E0[:, 1]])
        inter_lost = np.mean(cid[lost[:, 0]] != cid[lost[:, 1]]) if len(lost) else np.nan
        bridges = set(map(lambda e: tuple(sorted(e)), nx.bridges(g)))
        br_all = np.mean([tuple(e) in bridges for e in map(tuple, E0)])
        br_lost = np.mean([tuple(e) in bridges for e in map(tuple, lost)]) if len(lost) else np.nan
        deg0 = G0.sum(1)
        low_end = np.mean(np.minimum(deg0[lost[:, 0]], deg0[lost[:, 1]]) <= 2) if len(lost) else np.nan
        low_all = np.mean(np.minimum(deg0[E0[:, 0]], deg0[E0[:, 1]]) <= 2)
        summ.append(dict(seed=s, alpha_vor=a0, alpha_nach=a1, kanten_vor=len(E0), verloren=len(lost),
                         neu=len(gained), lcc_vor=L[k], lcc_modell=L[k + 1],
                         lcc_zufall_mittel=np.mean(rand_l), lcc_zufall_p05=np.percentile(rand_l, 5),
                         lcc_zufall_p95=np.percentile(rand_l, 95),
                         betweenness_perzentil_verloren=lost_rank.mean(),
                         anteil_zwischen_gruppen_verloren=inter_lost, anteil_zwischen_gruppen_alle=inter_all,
                         anteil_bruecken_verloren=br_lost, anteil_bruecken_alle=br_all,
                         anteil_schwacher_endpunkt_verloren=low_end, anteil_schwacher_endpunkt_alle=low_all,
                         mr_vor=mr_ratio(G0), mr_nach=mr_ratio(G1),
                         grad_vor=deg0.mean(), grad_nach=G1.sum(1).mean(),
                         deg3_vor=float((deg0 >= 3).mean()), deg3_nach=float((G1.sum(1) >= 3).mean())))
        for r in R:
            r.update(seed=s)
            rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "S6A_verlauf.csv"), index=False)
    pd.DataFrame(summ).to_csv(os.path.join(OUT, "S6A_kantenverlust.csv"), index=False)
    return pd.DataFrame(summ)


# ---------------------------------------------------------------------------
def start_net(kind, s, N=300):
    rng = np.random.default_rng(5000 + s)
    if kind == "Erdos-Renyi":
        g = nx.gnp_random_graph(N, 4.8 / (N - 1), seed=int(rng.integers(1e9)))
    elif kind == "Barabasi-Albert":
        g = nx.barabasi_albert_graph(N, 2, seed=int(rng.integers(1e9)))
        extra = int(0.4 * N)        # mittlerer Grad ~4.8
        while extra > 0:
            u, v = rng.integers(N, size=2)
            if u != v and not g.has_edge(u, v):
                p = (g.degree(u) + 1) * (g.degree(v) + 1)
                if rng.random() < p / (p + 25.0):
                    g.add_edge(u, v)
                    extra -= 1
    elif kind == "bimodal 2/8":
        deg = np.array([2] * (N // 2) + [8] * (N - N // 2))
        rng.shuffle(deg)
        g = nx.Graph(nx.configuration_model(deg.tolist(), seed=int(rng.integers(1e9))))
        g.remove_edges_from(nx.selfloop_edges(g))
    return nx.to_numpy_array(g, nodelist=range(N)).astype(bool)


def part_B(nets, D):
    rows = []
    variants = [("Modellnetz", None, {})]
    variants += [(k, k, {}) for k in ["Erdos-Renyi", "Barabasi-Albert", "bimodal 2/8"]]
    variants += [("Pflegekosten streuen (sd 0.5)", None, {"mu_sd": 0.5}),
                 ("Pflegekosten streuen (sd 1.0)", None, {"mu_sd": 1.0})]
    for lab, kind, kw in variants:
        for s in SEEDS:
            A0 = nets[s] if kind is None else start_net(kind, s)
            d0 = A0.sum(1)
            R, _ = up_sweep(A0, D, s, **kw)
            for r in R:
                r.update(variante=lab, seed=s, start_grad=d0.mean(), start_grad_var=d0.var(),
                         start_mr=mr_ratio(A0))
                rows.append(r)
        print("  B", lab, "fertig", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "S6B_heterogen.csv"), index=False)
    return df


# ---------------------------------------------------------------------------
def part_C(nets, D):
    rows = []
    D0 = D[0]
    for g in [0.0, 0.25, 0.5, 0.75]:
        fn = (lambda a, g=g: D[a] + g * (D0 - D[a]))
        for s in SEEDS:
            R, _ = up_sweep(nets[s], D, s, D_fn=fn)
            for r in R:
                r.update(variante="neue gemeinsame Arbeit", staerke=g, seed=s, D_eff=fn(r["alpha"]))
                rows.append(r)
    for rho in [0.05, 0.2, 0.5]:
        def pa(m, a, rho=rho):
            m.p["r_rand"] = rho * (1.0 - D[a] / D0)
        for s in SEEDS:
            R, _ = up_sweep(nets[s], D, s, per_alpha=pa)
            for r in R:
                r.update(variante="Zeitgewinn -> neue Kontakte", staerke=rho, seed=s, D_eff=D[r["alpha"]])
                rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "S6C_gegenmechanismus.csv"), index=False)
    return df


# ---------------------------------------------------------------------------
def part_D(nets, D):
    rows, longrows = [], []
    for K in [0.3, 0.6]:
        for rr in [0.0, 0.01, 0.05]:
            for s in SEEDS[:3]:
                for r in zk.sweep(nets[s], D.get, ALPHAS, T_PER, s, back=True, K=K, r_rand=rr):
                    r.update(K=K, r_rand=rr, seed=s)
                    rows.append(r)
                # langer Lauf: nach Kollaps bei alpha = 16 zurueck auf alpha = 1, 1200 Perioden (100 Jahre)
                m = zk.Future(nets[s], s, K=K, r_rand=rr)
                for a in ALPHAS:
                    for _ in range(T_PER):
                        m.step(D[a])
                for t in range(1, 1201):
                    m.step(D[1])
                    if t in (60, 120, 240, 480, 720, 1200):
                        st = m.stats(D[1])
                        longrows.append(dict(K=K, r_rand=rr, seed=s, perioden=t, lcc=st["lcc"],
                                             beziehungen=st["beziehungen"]))
            print("  D K=%g r_rand=%g fertig" % (K, rr), flush=True)
    a = pd.DataFrame(rows)
    a.to_csv(os.path.join(OUT, "S6D_hysterese.csv"), index=False)
    b = pd.DataFrame(longrows)
    b.to_csv(os.path.join(OUT, "S6D_lange_laeufe.csv"), index=False)
    return a, b


# ---------------------------------------------------------------------------
def part_A2(nets, D):
    """Model loss vs random loss of the same number of links (Supplementary Fig. 2b)."""
    rows = []
    for s in SEEDS:
        R, G = up_sweep(nets[s], D, s, keep=True)
        G0, G1 = G[3], G[4]
        lost = np.argwhere(np.triu(G0 & ~G1, 1))
        gained = np.argwhere(np.triu(G1 & ~G0, 1))
        E0 = np.argwhere(np.triu(G0, 1))
        rng = np.random.default_rng(1000 + s)
        for k in range(N_RAND):
            A = G0.copy()
            pick = E0[rng.choice(len(E0), len(lost), replace=False)]
            A[pick[:, 0], pick[:, 1]] = A[pick[:, 1], pick[:, 0]] = False
            A[gained[:, 0], gained[:, 1]] = A[gained[:, 1], gained[:, 0]] = True
            rows.append(dict(seed=s, typ="zufall", rep=k, lcc=lcc(A), deg3=(A.sum(1) >= 3).mean(), mr=mr_ratio(A)))
        rows.append(dict(seed=s, typ="modell", rep=0, lcc=lcc(G1), deg3=(G1.sum(1) >= 3).mean(), mr=mr_ratio(G1)))
        rows.append(dict(seed=s, typ="vorher", rep=0, lcc=lcc(G0), deg3=(G0.sum(1) >= 3).mean(), mr=mr_ratio(G0)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "S6A2_zufallsvergleich.csv"), index=False)
    return df


def part_B2(nets, D):
    """Heterogeneous maintenance costs, mean- or median-preserving (Supplementary Fig. 2d)."""
    rows = []
    for sd in [0.25, 0.5, 1.0]:
        for med in [False, True]:
            for s in SEEDS[:3]:
                m = zk.Future(nets[s], s, mu_sd=sd)
                if med:
                    m.mu_i = m.mu_i * np.exp(0.5 * sd * sd)      # median = 1
                for a in ALPHAS:
                    for _ in range(T_PER):
                        m.step(D[a])
                    st = m.stats(D[a])
                    hub = m.A.sum(1) >= 3
                    rows.append(dict(sd=sd, median1=med, seed=s, alpha=a, lcc=st["lcc"], mr=mr_ratio(m.A),
                                     deg3=hub.mean(), mu_hub=np.median(m.mu_i[hub]) if hub.any() else np.nan,
                                     mu_all=np.median(m.mu_i)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "S6B2_kostenstreuung.csv"), index=False)
    return df


def main():
    nets, D = nets_and_D(SEEDS)
    print("D:", {a: round(v, 3) for a, v in D.items()})
    A = part_A(nets, D)
    print(A.T.to_string())
    part_A2(nets, D)
    part_B2(nets, D)
    B = part_B(nets, D)
    C = part_C(nets, D)
    Dh, Dl = part_D(nets, D)
    T = ["Schritt 6 -- Robustheit nach Begutachtung", ""]
    T.append("B  Zusammenhalt je alpha (Mittel ueber Seeds) und Kollaps-alpha:")
    for lab, g in B.groupby("variante", sort=False):
        H = g.groupby("alpha")["lcc"].mean()
        ca = [collapse_alpha(gg.to_dict("records")) for _, gg in g.groupby("seed")]
        T.append("   %-32s " % lab + " ".join("%g:%.2f" % (a, H[a]) for a in ALPHAS) + "  Kollaps bei " + str(ca))
    T.append("C  Gegenmechanismus:")
    for (v, st), g in C.groupby(["variante", "staerke"], sort=False):
        H = g.groupby("alpha")["lcc"].mean()
        ca = [collapse_alpha(gg.to_dict("records")) for _, gg in g.groupby("seed")]
        T.append("   %-28s %.2f " % (v, st) + " ".join("%g:%.2f" % (a, H[a]) for a in ALPHAS) + "  Kollaps " + str(ca))
    T.append("D  Hysterese (hoch|runter):")
    for (K, rr), g in Dh.groupby(["K", "r_rand"]):
        up = g[g.phase == "hoch"].groupby("alpha")["lcc"].mean()
        dn = g[g.phase == "runter"].groupby("alpha")["lcc"].mean()
        T.append("   K=%.2f r_rand=%.2f " % (K, rr) + " ".join("%g:%.2f|%.2f" % (a, up[a], dn.get(a, np.nan)) for a in ALPHAS))
    T.append("D  lange Laeufe bei alpha=1 nach Kollaps:")
    T.append(Dl.groupby(["K", "r_rand", "perioden"])["lcc"].mean().unstack().round(2).to_string())
    open(os.path.join(OUT, "S6_zusammenfassung.txt"), "w").write("\n".join(T))
    print("\n".join(T))


if __name__ == "__main__":
    main()
