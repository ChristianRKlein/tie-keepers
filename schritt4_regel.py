# -*- coding: utf-8 -*-
"""
=====================================================================
  KI und das Kooperationsdilemma auf Netzwerken  --  SCHRITT 4 (v4)
  Die einfache Regel, Abstimmung als Hebel, Robustheit
=====================================================================

Wird von run_all.py aufgerufen. Modell und Regel: zukunft.py.

Die Regel (exakt im Modell, je Beziehung):
    b / c  >  (1 - D) * (1 + max(1/w, K/c) / T)
  b = erwarteter Ernstfallnutzen einer Beziehung je Periode, c = Pflegekosten
  ohne Nebenprodukt, D = Anteil, den gemeinsame Arbeit nebenbei traegt
  (aus dem NatComm-Netzwerkmodell, sinkt mit der KI-Faehigkeit), T = Zeithorizont,
  w = Schaden durch einseitige Vernachlaessigung, K = Wiederaufbaukosten.
Netz (Molloy-Reed): das Netz bleibt nur zusammen, solange sich die DRITTE
Beziehung lohnt.

Analysen
  A  Regel gegen Simulation im Phasendiagramm Pflegekosten x KI-Faehigkeit
  B  Regel gegen Simulation ueber 80 zufaellige Parameterkombinationen (Latin Hypercube)
     + partielle Rangkorrelationen fuer Kollaps-Zeitpunkt und Abruptheit
  C  Abstimmung als Hebel (Hauser et al. 2014): verbindliche Vereinbarung,
     kurzsichtige Minderheit
  D  Robustheit: Kontakte in drei Schritten zaehlen mit
"""
import os
import sys
import time
import shutil

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import netzmodell as nm   # noqa: E402
import zukunft as zk      # noqa: E402

QUICK = False
OUT = os.path.join(HERE, "results_S4")
ALPHAS = [0, 1, 2, 3, 4, 6, 8, 12, 16]
T_PER = 60
SEEDS_MAIN = [1, 2, 3, 4, 5]
SEEDS_SIDE = [1, 2]
MU_FINE = [0.005, 0.0075, 0.01, 0.0115, 0.0125, 0.0135, 0.015, 0.0175, 0.02, 0.025, 0.03, 0.035]
N_LHS = 80
LHS_RANGES = dict(mu=(0.008, 0.03), p=(0.05, 0.2), k0=(2.0, 8.0), eta=(0.0, 0.5),
                  beta=(0.90, 0.98), d0=(0.15, 0.5), d1_ratio=(0.5, 0.95), K=(0.0, 0.5),
                  eps=(0.002, 0.03), memory=(0.5, 0.95))
HAUSER = [
    ("Basis (jeder entscheidet allein)", {}),
    ("Verbindliche Vereinbarung", {"binding": True}),
    ("Kurzsichtige Minderheit (30 %)", {"frac_myopic": 0.3}),
    ("Minderheit + Vereinbarung", {"frac_myopic": 0.3, "binding": True}),
]
H_COLORS = ["#3a3a36", "#2a78d6", "#eb6834", "#1baf7a"]
H_MARK = ["o", "s", "^", "D"]


def setup_quick():
    global SEEDS_MAIN, SEEDS_SIDE, T_PER, N_LHS, MU_FINE
    if QUICK:
        SEEDS_MAIN, SEEDS_SIDE, T_PER, N_LHS = [1, 2], [1], 40, 20
        MU_FINE = MU_FINE[::2]


def networks_and_dependence(domain, seeds):
    nets, Ds = {}, []
    for s in seeds:
        res = nm.simulate_alpha_grid(domain, s, ALPHAS)
        nets[s] = res[0].A.copy()
        row = []
        for a in ALPHAS:
            m = res[a]
            d = m.A.sum(1)
            row.append(np.mean(np.where(d > 0, 1.0 - m.Y_alone() / m.Y_all(), 0.0)))
        Ds.append(row)
    return nets, dict(zip(ALPHAS, np.mean(Ds, axis=0)))


def run(label, nets, D, seeds, back=True, **kw):
    rows = []
    for s in seeds:
        for r in zk.sweep(nets[s], D.get, ALPHAS, T_PER, s, back=back, **kw):
            r.update(variante=label, seed=s)
            rows.append(r)
    return rows


def collapse_alpha_sim(lcc_by_alpha):
    """erstes alpha, bei dem der Zusammenhalt unter die Haelfte des Werts bei alpha = 0 faellt."""
    base = lcc_by_alpha[0]
    for a in ALPHAS:
        if lcc_by_alpha[a] < 0.5 * base:
            return a
    return np.inf


def collapse_alpha_theory(mu, D, p):
    for a in ALPHAS:
        if zk.collapse_predicted(mu, D[a], p):
            return a
    return np.inf


def lhs(n, rng):
    keys = list(LHS_RANGES)
    U = np.zeros((n, len(keys)))
    for k in range(len(keys)):
        U[:, k] = (rng.permutation(n) + rng.random(n)) / n
    rows = []
    for u in U:
        r = {k: lo + x * (hi - lo) for (k, (lo, hi)), x in zip(LHS_RANGES.items(), u)}
        r["d1"] = r["d0"] * r.pop("d1_ratio")
        rows.append(r)
    return rows


def prcc(X, y):
    """Partielle Rangkorrelationen (nur NumPy)."""
    def rank(v):
        r = np.empty(len(v))
        r[np.argsort(v, kind="mergesort")] = np.arange(len(v))
        return r
    Xr = np.column_stack([rank(X[:, j]) for j in range(X.shape[1])])
    yr = rank(y)
    out = []
    for j in range(X.shape[1]):
        Z = np.column_stack([np.ones(len(yr)), np.delete(Xr, j, axis=1)])
        rx = Xr[:, j] - Z @ np.linalg.lstsq(Z, Xr[:, j], rcond=None)[0]
        ry = yr - Z @ np.linalg.lstsq(Z, yr, rcond=None)[0]
        out.append(np.corrcoef(rx, ry)[0, 1])
    return np.array(out)


# ---------------------------------------------------------------------------
# Abbildungen
# ---------------------------------------------------------------------------
def style(ax, grid=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis="y", color="#e6e5e0", lw=0.8)
        ax.set_axisbelow(True)
    ax.tick_params(colors="#55554f", labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("#b7b6ae")


def xaxis(ax):
    try:
        ax.set_xscale("symlog", linthresh=4)
    except TypeError:
        ax.set_xscale("symlog", linthreshx=4)
    ax.set_xticks([0, 1, 2, 4, 8, 16])
    ax.set_xticklabels(["0", "1", "2", "4", "8", "16"])
    ax.set_xlabel("KI-Fähigkeit α", fontsize=9)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=200)
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    plt.close(fig)


def curve(ax, R, col, phase, color, marker, label=None, ls="-"):
    G = R[R["phase"] == phase].groupby("alpha")[col]
    x = G.mean().index.values
    m, sd = G.mean().values, G.std().fillna(0).values
    ax.plot(x, m, ls=ls, color=color, lw=2, marker=marker, ms=5, label=label,
            markeredgecolor="white", markeredgewidth=0.8)
    ax.fill_between(x, m - sd, m + sd, color=color, alpha=0.12, lw=0)


def fig_rule(D):
    """Die Regel: Schwelle fuer b/c in Abhaengigkeit von alpha und Zeithorizont."""
    p = zk.DEFAULTS
    a = np.array(ALPHAS, dtype=float)
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.8))
    ax = axs[0]
    for T, c, mk in [(10, "#eb6834", "^"), (20, "#2a78d6", "o"), (50, "#1baf7a", "s")]:
        beta = 1 - 1.0 / T
        thr = [zk.tie_rule_threshold(D[x], beta, p["d1"], p["K"], p["mu"]) for x in ALPHAS]
        ax.plot(a, thr, color=c, lw=2, marker=mk, ms=5, label="Zeithorizont T = %d Perioden" % T,
                markeredgecolor="white", markeredgewidth=0.8)
    ax.set_ylabel("Mindestwert für Nutzen/Kosten  b/c", fontsize=9)
    ax.text(0.15, 0.93, "Beziehung überlebt oberhalb der Kurve", transform=ax.transAxes, fontsize=7.5,
            color="#55554f")
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    ax.set_ylim(0, None)
    ax = axs[1]
    mus = np.linspace(0.005, 0.035, 61)
    for k_show, c, ls in [(1, "#b7b6ae", ":"), (2, "#8c8b84", "--"), (3, "#3a3a36", "-")]:
        line = []
        for mu in mus:
            hit = [x for x in ALPHAS if zk.k_star_mf(mu, D[x], p) < k_show]
            line.append(hit[0] if hit else np.nan)
        ax.plot(mus / p["mu"], line, color=c, ls=ls, lw=2,
                label="ab hier lohnt die %d. Beziehung nicht mehr" % k_show)
    ax.set_xlabel("Pflegekosten c (relativ zum Standard)", fontsize=9)
    ax.set_ylabel("KI-Fähigkeit α", fontsize=9)
    try:
        ax.set_yscale("symlog", linthresh=4)
    except TypeError:
        ax.set_yscale("symlog", linthreshy=4)
    ax.set_yticks([0, 1, 2, 4, 8, 16])
    ax.set_yticklabels(["0", "1", "2", "4", "8", "16"])
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    for x, t in zip(axs, "ab"):
        style(x)
        x.set_title(t, loc="left", fontweight="bold", fontsize=11)
    xaxis(axs[0])
    fig.tight_layout()
    save(fig, "Fig_S4a_die_regel")


def fig_phase_theory(RM, D):
    H = RM.groupby(["mu", "alpha"])["lcc"].mean().unstack("alpha")
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "blau", ["#f4f8fd", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])
    im = ax.pcolormesh(np.arange(len(ALPHAS) + 1) - 0.5, np.arange(len(H.index) + 1) - 0.5,
                       H.values, cmap=cmap, vmin=0, vmax=1, shading="flat")
    # Regel: erste alpha-Spalte, in der die 3. Beziehung nicht mehr lohnt
    xs, ys = [], []
    for k, mu in enumerate(H.index):
        hit = [i for i, a in enumerate(ALPHAS) if zk.collapse_predicted(mu, D[a])]
        xs.append(hit[0] - 0.5 if hit else len(ALPHAS) - 0.5)
        ys.append(k)
    ax.step(xs, ys, where="mid", color="#eb6834", lw=2.4, label="Regel: 3. Beziehung lohnt nicht mehr")
    ax.set_xticks(np.arange(len(ALPHAS)))
    ax.set_xticklabels([str(a) for a in ALPHAS], fontsize=8)
    ax.set_yticks(np.arange(len(H.index)))
    ax.set_yticklabels(["%.2f" % (m / zk.DEFAULTS["mu"]) for m in H.index], fontsize=8)
    ax.set_xlabel("KI-Fähigkeit α", fontsize=9)
    ax.set_ylabel("Pflegekosten c (relativ zum Standard)", fontsize=9)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Zusammenhalt in der Simulation", fontsize=8)
    ax.legend(fontsize=7.5, loc="lower left", frameon=True)
    fig.tight_layout()
    save(fig, "Fig_S4b_regel_vs_simulation")


def fig_lhs(L, P):
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.9))
    ax = axs[0]
    idx = {a: i for i, a in enumerate(ALPHAS)}
    idx[np.inf] = len(ALPHAS)
    xs = np.array([idx[v] for v in L["alpha_kollaps_regel"]], dtype=float)
    ys = np.array([idx[v] for v in L["alpha_kollaps_sim"]], dtype=float)
    rng = np.random.default_rng(0)
    ax.scatter(xs + rng.uniform(-0.18, 0.18, len(xs)), ys + rng.uniform(-0.18, 0.18, len(ys)), s=26,
               color="#2a78d6", edgecolor="white", lw=0.6, alpha=0.85)
    ax.plot([-0.5, len(ALPHAS) + 0.5], [-0.5, len(ALPHAS) + 0.5], color="#8c8b84", lw=1, ls=":")
    lab = [str(a) for a in ALPHAS] + ["kein"]
    ax.set_xticks(range(len(lab)))
    ax.set_xticklabels(lab, fontsize=7.5)
    ax.set_yticks(range(len(lab)))
    ax.set_yticklabels(lab, fontsize=7.5)
    ax.set_xlabel("Kollaps vorhergesagt bei α (Regel)", fontsize=9)
    ax.set_ylabel("Kollaps simuliert bei α", fontsize=9)
    exact = np.mean(xs == ys)
    near = np.mean(np.abs(xs - ys) <= 1)
    ax.set_title("a   %d Parameterkombinationen: %.0f %% exakt, %.0f %% ±1 Schritt"
                 % (len(L), 100 * exact, 100 * near), loc="left", fontsize=9, fontweight="bold")
    style(ax, grid=False)
    ax = axs[1]
    order = np.argsort(np.abs(P["prcc_kollaps"].values))[::-1]
    names = P["parameter"].values[order]
    vals = P["prcc_kollaps"].values[order]
    ax.barh(np.arange(len(vals)), vals, color=["#2a78d6" if v > 0 else "#eb6834" for v in vals],
            height=0.6)
    ax.set_yticks(np.arange(len(vals)))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="#3a3a36", lw=0.8)
    ax.set_xlabel("Partielle Rangkorrelation mit dem Kollaps-α\n(positiv = Kollaps erst bei stärkerer KI)",
                  fontsize=8.5)
    ax.set_title("b   Was den Kollaps verschiebt", loc="left", fontsize=9, fontweight="bold")
    style(ax, grid=False)
    fig.tight_layout()
    save(fig, "Fig_S4c_regel_robust_lhs")


def fig_hauser(RH):
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.7))
    for k, (lab, _) in enumerate(HAUSER):
        H = RH[RH["variante"] == lab]
        curve(axs[0], H, "lcc", "hoch", H_COLORS[k], H_MARK[k], lab)
        curve(axs[1], H, "ernstfallwert", "hoch", H_COLORS[k], H_MARK[k], lab)
        curve(axs[2], H, "lcc", "runter", H_COLORS[k], H_MARK[k], lab, ls="--")
    axs[0].set_ylabel("Zusammenhalt (größte Komponente)", fontsize=9)
    axs[1].set_ylabel("Wert des Netzes im Ernstfall", fontsize=9)
    axs[2].set_ylabel("Zusammenhalt, wenn KI wieder schwächer wird", fontsize=9)
    for x, t in zip(axs, "abc"):
        style(x)
        xaxis(x)
        x.set_ylim(0, None)
        x.set_title(t, loc="left", fontweight="bold", fontsize=11)
    axs[0].legend(fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout()
    save(fig, "Fig_S4d_abstimmung_hauser")


def fig_3hop(R3):
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    for k, (lab, c, mk) in enumerate([("2 Schritte (Standard)", "#3a3a36", "o"),
                                      ("3 Schritte (η₃ = 0,1)", "#2a78d6", "s")]):
        H = R3[R3["variante"] == lab]
        curve(axs[0], H, "lcc", "hoch", c, mk, lab)
        curve(axs[1], H, "beziehungen", "hoch", c, mk, lab)
    axs[0].set_ylabel("Zusammenhalt (größte Komponente)", fontsize=9)
    axs[1].set_ylabel("Beziehungen je Person", fontsize=9)
    for x, t in zip(axs, "ab"):
        style(x)
        xaxis(x)
        x.set_ylim(0, None)
        x.set_title(t, loc="left", fontweight="bold", fontsize=11)
    axs[0].legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    save(fig, "Fig_S4e_robustheit_3_schritte")


# ---------------------------------------------------------------------------
def main():
    setup_quick()
    t0 = time.time()
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    print("=== Schritt 4 (v4): Regel, Abstimmung, Robustheit | QUICK=%s ===" % QUICK)
    nets, D = networks_and_dependence("Wissenschaft", SEEDS_MAIN)
    print("  Netze fertig (%.0f s)" % (time.time() - t0))

    # A: Phasendiagramm mit Regel
    rows = []
    for mu in MU_FINE:
        for r in run("mu", nets, D, SEEDS_SIDE, back=False, mu=mu):
            r["mu"] = mu
            rows.append(r)
    RM = pd.DataFrame(rows)
    print("  A Phasendiagramm fertig (%.0f s)" % (time.time() - t0))

    # B: Latin Hypercube
    rows = []
    s0 = SEEDS_MAIN[0]
    for k, par in enumerate(lhs(N_LHS, np.random.default_rng(2026))):
        R = pd.DataFrame(zk.sweep(nets[s0], D.get, ALPHAS, T_PER, 100 + k, back=False, **par))
        L = dict(zip(R["alpha"], R["lcc"]))
        Dg = dict(zip(R["alpha"], R["beziehungen"]))
        pp = dict(zk.DEFAULTS)
        pp.update(par)
        a_sim = collapse_alpha_sim(L)
        a_th = collapse_alpha_theory(par["mu"], D, pp)
        # Abruptheit: groesster Sprung im Zusammenhalt relativ zum Rueckgang der Beziehungen
        jumps = []
        for a0, a1 in zip(ALPHAS[:-1], ALPHAS[1:]):
            dl = (L[a0] - L[a1]) / max(L[a0], 1e-9)
            dk = (Dg[a0] - Dg[a1]) / max(Dg[a0], 1e-9)
            jumps.append((dl, dk))
        dl, dk = max(jumps, key=lambda z: z[0])
        row = dict(par)
        row.update(lcc_alpha0=L[0], lcc_alpha8=L[8], alpha_kollaps_sim=a_sim, alpha_kollaps_regel=a_th,
                   groesster_sprung_zusammenhalt=dl, beziehungsverlust_dabei=dk)
        rows.append(row)
        if (k + 1) % 10 == 0:
            print("  B LHS %d/%d (%.0f s)" % (k + 1, N_LHS, time.time() - t0))
    LH = pd.DataFrame(rows)
    keys = [k for k in LHS_RANGES if k != "d1_ratio"] + ["d1"]
    ok = LH["lcc_alpha0"] > 0.5
    X = LH.loc[ok, keys].values
    ycol = LH.loc[ok, "alpha_kollaps_sim"].replace(np.inf, 32).values
    P = pd.DataFrame({"parameter": keys, "prcc_kollaps": prcc(X, ycol)})
    names = dict(mu="Pflegekosten c", p="Häufigkeit Ernstfall", k0="Sättigung k0", eta="Gewicht 2 Schritte η",
                 beta="Zeithorizont (β)", d0="Zerfall ohne Pflege d0", d1="Zerfall einseitig d1 (w)",
                 K="Wiederaufbaukosten K", eps="Versehen ε", memory="Gedächtnis Vertrauen")
    P["parameter"] = P["parameter"].map(names)

    # C: Abstimmung (Hauser)
    rows = []
    for lab, kw in HAUSER:
        rows += run(lab, nets, D, SEEDS_MAIN, **kw)
        print("  C %-35s fertig (%.0f s)" % (lab, time.time() - t0))
    RH = pd.DataFrame(rows)

    # D: drei Schritte
    rows = run("2 Schritte (Standard)", nets, D, SEEDS_SIDE, back=False)
    rows += run("3 Schritte (η₃ = 0,1)", nets, D, SEEDS_SIDE, back=False, eta3=0.1)
    R3 = pd.DataFrame(rows)
    print("  D Robustheit 3 Schritte fertig (%.0f s)" % (time.time() - t0))

    for name, X_ in [("S4_phasendiagramm", RM), ("S4_lhs", LH), ("S4_prcc", P), ("S4_abstimmung", RH),
                     ("S4_drei_schritte", R3)]:
        X_.to_csv(os.path.join(OUT, name + ".csv"), index=False, float_format="%.6g")
    fig_rule(D)
    fig_phase_theory(RM, D)
    fig_lhs(LH[ok], P)
    fig_hauser(RH)
    fig_3hop(R3)

    # Zusammenfassung
    idx = {a: i for i, a in enumerate(ALPHAS)}
    idx[np.inf] = len(ALPHAS)
    Lk = LH[ok]
    xs = np.array([idx[v] for v in Lk["alpha_kollaps_regel"]])
    ys = np.array([idx[v] for v in Lk["alpha_kollaps_sim"]])
    coll = Lk[np.isfinite(Lk["alpha_kollaps_sim"])]
    T = ["SCHRITT 4 (v4) - Regel, Abstimmung, Robustheit", "=" * 70, ""]
    T.append("Regel je Beziehung:  b/c > (1 - D) * (1 + max(1/w, K/c) / T)")
    T.append("Netz: Kollaps, sobald sich die 3. Beziehung nicht mehr lohnt (Molloy-Reed)")
    T.append("")
    T.append("B  Latin Hypercube: %d Kombinationen, davon %d mit anfaenglich zusammenhaengendem Netz"
             % (len(LH), ok.sum()))
    T.append("   Kollaps-alpha Regel = Simulation: %.0f %% exakt, %.0f %% innerhalb eines Gitterschritts"
             % (100 * np.mean(xs == ys), 100 * np.mean(np.abs(xs - ys) <= 1)))
    T.append("   Rangkorrelation Regel ~ Simulation (Gitterindex): %.2f" % np.corrcoef(
        np.argsort(np.argsort(xs)), np.argsort(np.argsort(ys)))[0, 1])
    if len(coll):
        ratio = coll["groesster_sprung_zusammenhalt"] / coll["beziehungsverlust_dabei"].clip(lower=0.01)
        T.append("   Beim groessten Sprung: Zusammenhalt -%.0f %% bei Beziehungen -%.0f %% (Median); "
                 "Zusammenhalt faellt mind. doppelt so stark wie die Beziehungen in %.0f %% der Kollapsfaelle"
                 % (100 * coll["groesster_sprung_zusammenhalt"].median(),
                    100 * coll["beziehungsverlust_dabei"].median(), 100 * np.mean(ratio >= 2)))
    T.append("   PRCC mit Kollaps-alpha:")
    for _, r in P.sort_values("prcc_kollaps", key=np.abs, ascending=False).iterrows():
        T.append("     %-28s %+.2f" % (r["parameter"], r["prcc_kollaps"]))
    T.append("")
    T.append("C  Abstimmung (Zusammenhalt, KI wird staerker | wieder schwaecher):")
    for lab, _ in HAUSER:
        H = RH[RH["variante"] == lab]
        up = H[H["phase"] == "hoch"].groupby("alpha")["lcc"].mean()
        dn = H[H["phase"] == "runter"].groupby("alpha")["lcc"].mean()
        T.append("   %-35s " % lab + " ".join("a%g:%.2f|%.2f" % (a, up[a], dn.get(a, np.nan)) for a in ALPHAS))
    T.append("")
    T.append("D  3 Schritte (Zusammenhalt):")
    for lab in R3["variante"].unique():
        H = R3[R3["variante"] == lab].groupby("alpha")["lcc"].mean()
        T.append("   %-25s " % lab + " ".join("a%g:%.2f" % (a, H[a]) for a in ALPHAS))
    txt = "\n".join(T)
    with open(os.path.join(OUT, "ERGEBNISSE.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt)
    runtime = time.time() - t0
    with open(os.path.join(OUT, "INFO.txt"), "w", encoding="utf-8") as f:
        f.write("Schritt 4 (v4) | QUICK=%s | Laufzeit %.1f min | Python %s | numpy %s\n"
                % (QUICK, runtime / 60, sys.version.split()[0], np.__version__))
    print("Schritt 4 fertig nach %.1f min" % (runtime / 60))


if __name__ == "__main__":
    main()
