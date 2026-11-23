# -*- coding: utf-8 -*-
"""
=====================================================================
  KI und das Kooperationsdilemma auf Netzwerken  --  SCHRITT 1
  Evolutionaeres Gefangenendilemma auf den Netzen unseres Modells
=====================================================================

In Spyder oeffnen und mit F5 ausfuehren.
Benoetigt nur numpy, pandas, matplotlib (Python >= 3.7).
Ergebnis: Ordner results_S1/ und results_S1.zip im selben Ordner.
Laufzeit: QUICK = True ca. 2 min, QUICK = False ca. 10-15 min.

Fragestellung
-------------
Unser NatComm-Modell erzeugt fuer jede KI-Staerke alpha ein Kooperations-
netzwerk. Auf jeder Kante liefert das Modell selbst
    b_ji = Outputgewinn, den Partner j fuer i stiftet   (sinkt mit Substitution)
    c_j  = Kosten, die j fuer eine Verbindung traegt    (sinkt mit Koordination)
Wir fragen: Wuerde Kooperation auf diesem Netz gegen Trittbrettfahrer
evolutionaer bestehen (Death-Birth-Update, schwache Selektion)?

  A  Ohtsuki-Regel (2006):            b/c > k
  B  exakte Schwelle (Allen 2017):    b/c > (b/c)*   fuer den konkreten Graphen
  C  heterogene Bedingung:            MARGE = B_term / C_term > 1
     (nutzt jedes einzelne b_ji und c_j aus dem Modell; siehe struktur.py)

Ausgaben
--------
  S1_netze.csv           eine Zeile je Szenario x Seed x alpha
  S1_zusammenfassung.csv Mittelwerte und SD ueber Seeds
  S1_validierung.csv     Loeser gegen exakte Formel / exakte Markov-Kette
  Fig_S1a ... Fig_S1d    Abbildungen (PNG und PDF)
  ERGEBNISSE.txt         Kernzahlen in Worten
"""
import os
import sys
import time
import shutil
import zipfile

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # Abbildungen nur als Dateien (stabil in Spyder)
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import netzmodell as nm   # noqa: E402
import struktur as st     # noqa: E402

# ---------------------------------------------------------------------------
QUICK = False
SEEDS = [1, 2] if QUICK else [1, 2, 3, 4, 5]
ALPHAS = [0, 2, 4, 8, 16] if QUICK else [0, 1, 2, 3, 4, 6, 8, 12, 16]
SCENARIOS = [
    # (Name, Domaene, Abweichungen von der Basiskonfiguration)
    ("Wissenschaft", "Wissenschaft", {}),
    ("Wirtschaft", "Wirtschaft", {}),
    ("Sozial", "Sozial", {}),
    ("Wissenschaft + Koordination (kappa=1)", "Wissenschaft", {"kappa": 1.0}),
]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "^", "D"]
OUT = os.path.join(HERE, "results_S1")
# ---------------------------------------------------------------------------


def analyse_network(m):
    """Alle Kennzahlen eines Endnetzes."""
    N = m.p["N"]
    d = m.A.sum(1)
    Y = m.Y_all()
    dep = np.where(d > 0, 1.0 - m.Y_alone() / Y, 0.0)
    B, c, _ = m.edge_payoffs()
    idx = st.largest_component(m.A)
    W = m.A[np.ix_(idx, idx)].astype(float)
    Bs, cs = B[np.ix_(idx, idx)], c[idx]
    r = st.structure_condition(W, Bs, cs)
    ii, jj = np.nonzero(W)                      # i empfaengt von j
    ratio = Bs[jj, ii] / cs[jj]
    kdeg = W.sum(1)
    b_mean, c_mean = Bs[jj, ii].mean(), cs[jj].mean()
    return dict(
        grad_mittel=d.mean(), isoliert=(d == 0).mean(), lcc_anteil=len(idx) / N,
        abhaengigkeit=dep.mean(), output_mittel=Y.mean(),
        b_mittel=b_mean, c_mittel=c_mean, bc=b_mean / c_mean,
        bc_median_kante=np.median(ratio),
        k_lcc=r["k_mean"], bc_star_allen=r["bc_star_allen"], sigma=r["sigma"],
        marge_ohtsuki=(b_mean / c_mean) / r["k_mean"],
        marge_allen=(b_mean / c_mean) / r["bc_star_allen"],
        marge_het=r["marge_het"],
        anteil_kanten_bc_gt_k=np.mean(ratio > kdeg[ii]),
        phi_mittel=float(m.phi.mean()),                  # Anteil der Luecke, den KI fuellt
        amp_mittel=float(np.mean(m.bfac / m.a)),         # Amplifikationsfaktor
        tau1=r["tau1"], tau2=r["tau2"], tau3=r["tau3"],
        cg_iter=r["cg_iter"])


def run_simulations():
    rows = []
    t0 = time.time()
    for name, dom, ov in SCENARIOS:
        for s in SEEDS:
            res = nm.simulate_alpha_grid(dom, s, ALPHAS, **ov)
            for al in ALPHAS:
                row = dict(szenario=name, domaene=dom, seed=s, alpha=al)
                row.update(analyse_network(res[al]))
                rows.append(row)
            print("  %-40s seed %d fertig  (%.0f s)" % (name, s, time.time() - t0))
    return pd.DataFrame(rows)


def run_validation():
    """(1) CG-Loeser gegen exakte Formel auf regulaeren Graphen,
       (2) heterogene Bedingung gegen exakte Markov-Kette (N = 7)."""
    rng = np.random.default_rng(0)
    rows = []
    for N, k in [(50, 2), (100, 4), (200, 6), (300, 4), (300, 8)]:
        A = st.cycle(N) if k == 2 else st.random_regular(N, k, rng)
        r = st.structure_condition(A.astype(float))
        rows.append(dict(test="regulaer N=%d k=%d" % (N, k), numerisch=r["bc_star_allen"],
                         exakt=st.allen_regular_formula(N, k)))
    # exakte Markov-Kette
    N = 7
    A = np.zeros((N, N), dtype=bool)
    for i, j in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 0), (0, 3), (1, 5), (2, 6), (0, 4)]:
        A[i, j] = A[j, i] = True
    W = A.astype(float)
    P = W / W.sum(1)[:, None]
    B = np.where(A, rng.uniform(1, 4, (N, N)), 0.0)
    c = rng.uniform(0.5, 1.5, N)
    S = 2 ** N
    X = ((np.arange(S)[:, None] >> np.arange(N)) & 1).astype(float)

    def rhoC_minus_rhoD(scale, delta=1e-4):
        T = np.zeros((S, S))
        for s in range(S):
            x = X[s]
            F = 1.0 + delta * (-c * x + (P * (scale * B).T) @ x)
            for i in range(N):
                pr = W[i] * F
                pC = (pr / pr.sum()) @ x
                T[s, s | (1 << i)] += pC / N
                T[s, s & ~(1 << i)] += (1 - pC) / N
        tr = np.arange(1, S - 1)
        h = np.linalg.solve(np.eye(len(tr)) - T[np.ix_(tr, tr)], T[tr, S - 1])
        H = np.zeros(S)
        H[tr] = h
        H[S - 1] = 1
        rC = np.mean([H[1 << i] for i in range(N)])
        rD = np.mean([1 - H[(S - 1) & ~(1 << i)] for i in range(N)])
        return rC - rD

    lo, hi = 0.5, 50.0
    for _ in range(30):
        mid = np.sqrt(lo * hi)
        if rhoC_minus_rhoD(mid) > 0:
            hi = mid
        else:
            lo = mid
    r = st.structure_condition(W, B, c)
    rows.append(dict(test="heterogen, exakte Markov-Kette N=7 (kritische Skalierung)",
                     numerisch=1.0 / r["marge_het"], exakt=np.sqrt(lo * hi)))
    V = pd.DataFrame(rows)
    V["rel_abweichung"] = (V["numerisch"] - V["exakt"]).abs() / V["exakt"]
    return V


# ---------------------------------------------------------------------------
# Abbildungen
# ---------------------------------------------------------------------------
def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e6e5e0", lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#55554f", labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("#b7b6ae")


def xaxis(ax):
    try:
        ax.set_xscale("symlog", linthresh=4)
    except TypeError:                     # aeltere matplotlib-Versionen (< 3.3)
        ax.set_xscale("symlog", linthreshx=4)
    ax.set_xticks([0, 1, 2, 4, 8, 16])
    ax.set_xticklabels(["0", "1", "2", "4", "8", "16"])
    ax.set_xlabel("KI-Fähigkeit α", fontsize=9)


def line(ax, G, col, k, label=None, ls="-", marker=True):
    x = G.index.values
    ax.plot(x, G[(col, "mean")], ls=ls, lw=2, color=COLORS[k],
            marker=MARKERS[k] if marker else None, ms=5, label=label,
            markeredgecolor="white", markeredgewidth=0.8)
    ax.fill_between(x, G[(col, "mean")] - G[(col, "std")], G[(col, "mean")] + G[(col, "std")],
                    color=COLORS[k], alpha=0.12, lw=0)


def figures(D):
    agg = D.drop(columns=["domaene", "seed"]).groupby(["szenario", "alpha"]).agg(["mean", "std"])
    names = [s[0] for s in SCENARIOS]

    # --- S1a: Das Netz und die Spielparameter -------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.4))
    for k, nmn in enumerate(names):
        G = agg.loc[nmn]
        line(axs[0], G, "grad_mittel", k, nmn)
        line(axs[1], G, "b_mittel", k)
        line(axs[2], G, "bc", k)
    axs[0].set_ylabel("Mittlerer Grad k", fontsize=9)
    axs[1].set_ylabel("Nutzen je Kante b (Output)", fontsize=9)
    axs[2].set_ylabel("Nutzen-Kosten-Verhältnis b/c", fontsize=9)
    for a, t in zip(axs, "abc"):
        style(a)
        xaxis(a)
        a.set_title(t, loc="left", fontweight="bold", fontsize=11)
    axs[0].legend(fontsize=7, frameon=False, loc="upper right")
    fig.tight_layout()
    save(fig, "Fig_S1a_netz_und_spielparameter")

    # --- S1b: b/c gegen die Schwellen, je Szenario --------------------------
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.4), sharey=True)
    for k, (a, nmn) in enumerate(zip(axs, names)):
        G = agg.loc[nmn]
        line(a, G, "bc", k, "b/c aus dem Modell")
        x = G.index.values
        a.plot(x, G[("bc_star_allen", "mean")], color="#3a3a36", lw=1.6, ls="--",
               label="(b/c)* exakt (Allen 2017)")
        a.plot(x, G[("k_lcc", "mean")], color="#8c8b84", lw=1.4, ls=":",
               label="k (Ohtsuki 2006)")
        a.set_title(nmn if len(nmn) < 20 else "Wiss. + Koordination", fontsize=9, loc="left")
        style(a)
        xaxis(a)
        a.set_ylim(0, 8.6)
    axs[0].set_ylabel("Verhältnis bzw. Schwelle", fontsize=9)
    axs[0].legend(fontsize=7, frameon=False, loc="upper right")
    fig.text(0.005, 0.97, "b", fontweight="bold", fontsize=12, va="top")
    fig.tight_layout(rect=(0.01, 0, 1, 1))
    save(fig, "Fig_S1b_bc_gegen_schwellen")

    # --- S1c: Kooperationsmarge ---------------------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for k, nmn in enumerate(names):
        G = agg.loc[nmn]
        line(axs[0], G, "marge_het", k, nmn)
        line(axs[1], G, "marge_ohtsuki", k, nmn)
    axs[0].set_ylabel("Strukturelle Marge (exakt, heterogen)", fontsize=9)
    axs[1].set_ylabel("Marge nach Ohtsuki  (b/c)/k", fontsize=9)
    for a, t in zip(axs, "ab"):
        style(a)
        xaxis(a)
        a.axhline(1, color="#3a3a36", lw=1, ls="--")
        a.text(0.02, 1.0, " Kooperation begünstigt ↑", fontsize=7, color="#3a3a36",
               transform=a.get_yaxis_transform(), va="bottom")
        a.set_ylim(0, 1.15)
        a.set_title(t, loc="left", fontweight="bold", fontsize=11)
    axs[0].legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    save(fig, "Fig_S1c_kooperationsmarge")

    # --- S1d: Zerlegung: was treibt die Marge? -------------------------------
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for k, nmn in enumerate(names):
        G = agg.loc[nmn]
        x0 = G.index.values[0]
        bc0, st0 = G.loc[x0, ("bc", "mean")], G.loc[x0, ("bc_star_allen", "mean")]
        ax.plot(np.log(G[("bc", "mean")] / bc0), -np.log(G[("bc_star_allen", "mean")] / st0),
                color=COLORS[k], marker=MARKERS[k], lw=2, ms=5, label=nmn,
                markeredgecolor="white", markeredgewidth=0.8)
    lim = 0.6
    ax.plot([-lim, lim], [lim, -lim], color="#8c8b84", lw=1, ls=":")
    ax.text(-0.25, 0.85, "Marge steigt ↗", fontsize=7, color="#55554f")
    ax.text(0.2, -0.3, "Marge sinkt ↙", fontsize=7, color="#55554f")
    ax.axhline(0, color="#b7b6ae", lw=0.8)
    ax.axvline(0, color="#b7b6ae", lw=0.8)
    ax.set_xlabel("Änderung log(b/c) gegenüber α = 0", fontsize=9)
    ax.set_ylabel("Strukturgewinn  −Δ log (b/c)*", fontsize=9)
    style(ax)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout()
    save(fig, "Fig_S1d_zerlegung")
    return agg


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=200)
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    plt.close(fig)


def write_summary(agg, V, runtime):
    L = []
    L.append("SCHRITT 1 - Evolutionaeres Gefangenendilemma auf KI-geformten Netzen")
    L.append("=" * 70)
    L.append("Seeds: %s | alpha: %s | Laufzeit %.1f min" % (SEEDS, ALPHAS, runtime / 60))
    L.append("")
    L.append("Validierung (relative Abweichung, Ziel < 1e-3):")
    for _, r in V.iterrows():
        L.append("  %-62s %.2e" % (r["test"], r["rel_abweichung"]))
    L.append("")
    amax = max(ALPHAS)
    for name, _, _ in SCENARIOS:
        G = agg.loc[name]
        g0, g1 = G.loc[0], G.loc[amax]
        L.append(name)
        L.append("  alpha 0 -> %g:  k %.2f -> %.2f | b/c %.2f -> %.2f | (b/c)* %.2f -> %.2f"
                 % (amax, g0[("k_lcc", "mean")], g1[("k_lcc", "mean")], g0[("bc", "mean")],
                    g1[("bc", "mean")], g0[("bc_star_allen", "mean")], g1[("bc_star_allen", "mean")]))
        L.append("  Marge (exakt, heterogen): %.2f -> %.2f   |  Ohtsuki (b/c)/k: %.2f -> %.2f"
                 % (g0[("marge_het", "mean")], g1[("marge_het", "mean")],
                    g0[("marge_ohtsuki", "mean")], g1[("marge_ohtsuki", "mean")]))
        L.append("  Kanten mit b_ji/c_j > k_i: %.0f %% -> %.0f %%"
                 % (100 * g0[("anteil_kanten_bc_gt_k", "mean")], 100 * g1[("anteil_kanten_bc_gt_k", "mean")]))
        L.append("")
    L.append("Lesart: Marge > 1 heisst, Kooperation setzt sich auf diesem Netz allein durch")
    L.append("Netzwerk-Reziprozitaet gegen Trittbrettfahrer durch (schwache Selektion, DB).")
    txt = "\n".join(L)
    with open(os.path.join(OUT, "ERGEBNISSE.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print("\n" + txt)


def main():
    t0 = time.time()
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    print("=== Schritt 1: Ohtsuki/Allen auf KI-geformten Netzen | QUICK=%s ===" % QUICK)
    print("Validierung ...")
    V = run_validation()
    V.to_csv(os.path.join(OUT, "S1_validierung.csv"), index=False)
    print(V.to_string(index=False))
    print("Simulationen ...")
    D = run_simulations()
    D.to_csv(os.path.join(OUT, "S1_netze.csv"), index=False)
    agg = figures(D)
    agg.round(5).to_csv(os.path.join(OUT, "S1_zusammenfassung.csv"))
    runtime = time.time() - t0
    write_summary(agg, V, runtime)
    with open(os.path.join(OUT, "INFO.txt"), "w", encoding="utf-8") as f:
        f.write("Schritt 1 | QUICK=%s | Laufzeit %.1f min | Python %s | numpy %s | pandas %s\n"
                % (QUICK, runtime / 60, sys.version.split()[0], np.__version__, pd.__version__))
    z = os.path.join(HERE, "results_S1.zip")
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for fn in sorted(os.listdir(OUT)):
            zf.write(os.path.join(OUT, fn), arcname="results_S1/" + fn)
    print("\nFERTIG nach %.1f min: %s" % (runtime / 60, z))


if __name__ == "__main__":
    main()
