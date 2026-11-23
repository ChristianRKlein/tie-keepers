# -*- coding: utf-8 -*-
"""
=====================================================================
  KI und das Kooperationsdilemma auf Netzwerken  --  SCHRITT 3 (v3)
  Kooperation mit der Zukunft: Lohnt es sich noch, menschliche
  Netzwerke zu pflegen, wenn KI die taegliche Arbeit erledigt?
=====================================================================

Wird von run_all.py aufgerufen. Modell: siehe zukunft.py.

Drei Ebenen
  1  Einzelner:  optimale Zahl gepflegter Beziehungen k*(alpha), analytisch
  2  Paar:       Pflege braucht beide -> Hirschjagd; Anteil der Beziehungen je Spieltyp
  3  Netzwerk:   Zusammenhalt (groesste Komponente) als Gemeingut; Kollaps,
                 Hysterese (Wiederaufbaukosten K), Politikhebel

Analysen
  A  Hauptlauf Wissenschaft: alpha steigt 0 -> 16 und faellt zurueck (5 Seeds)
     Varianten: Basis | Gemeinwohl | Pflegesubvention 30 % | Wiederaufbauhilfe
     (K / 4) | Kombination
  B  Wirtschaft (Robustheit, Basis)
  C  Hysterese in Abhaengigkeit von den Wiederaufbaukosten K
  D  Subventionsstufen: Zusammenhalt und Wohlfahrt (volle Kosten) bei alpha = 8
  E  Phasendiagramm Pflegekosten mu x alpha
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
OUT = os.path.join(HERE, "results_S3")
ALPHAS = [0, 1, 2, 3, 4, 6, 8, 12, 16]
T_PER = 60
SEEDS_MAIN = [1, 2, 3, 4, 5]
SEEDS_SIDE = [1, 2, 3]

VARIANTS = [
    ("Basis (eigennützig)", {}),
    ("Gemeinwohl (Nutzen für Dritte mitgedacht)", {"social": 1.0}),
    ("Pflegesubvention 30 %", {"subsidy": 0.3}),
    ("Wiederaufbauhilfe (K / 4)", {"K": zk.DEFAULTS["K"] / 4}),
    ("Kombination", {"subsidy": 0.3, "K": zk.DEFAULTS["K"] / 4}),
]
VAR_COLORS = ["#3a3a36", "#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
VAR_MARK = ["o", "s", "^", "D", "v"]
K_LEVELS = [0.0, 0.15, 0.3, 0.6, 1.0]
SUBSIDIES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
MU_LEVELS = [0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02, 0.025, 0.03]


def setup_quick():
    global SEEDS_MAIN, SEEDS_SIDE, T_PER
    if QUICK:
        SEEDS_MAIN, SEEDS_SIDE, T_PER = [1, 2], [1], 40


# ---------------------------------------------------------------------------
def networks_and_dependence(domain, seeds):
    """Ausgangsnetz (alpha = 0) und Abhaengigkeit D(alpha) aus dem Netzwerkmodell."""
    nets, Ds = {}, []
    for s in seeds:
        res = nm.simulate_alpha_grid(domain, s, ALPHAS)
        nets[s] = res[0].A.copy()
        row = []
        for a in ALPHAS:
            m = res[a]
            d = m.A.sum(1)
            Y = m.Y_all()
            row.append(np.mean(np.where(d > 0, 1.0 - m.Y_alone() / Y, 0.0)))
        Ds.append(row)
    D = dict(zip(ALPHAS, np.mean(Ds, axis=0)))
    return nets, D


def run(label, nets, D, seeds, back=True, **kw):
    rows = []
    for s in seeds:
        for r in zk.sweep(nets[s], D.get, ALPHAS, T_PER, s, back=back, **kw):
            r.update(variante=label, seed=s)
            r.update({k: v for k, v in kw.items()})
            rows.append(r)
    return rows


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


def fig_mechanism(R, D, p):
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.7))
    a = np.array(ALPHAS, dtype=float)
    Dv = np.array([D[x] for x in ALPHAS])
    ax = axs[0]
    ax.plot(a, 1 - Dv, color="#2a78d6", lw=2, marker="o", ms=5, markeredgecolor="white")
    ax.set_ylabel("Pflegekosten relativ zu ohne Arbeitskontakt\n1 − D(α)", fontsize=8.5)
    ax.set_ylim(0, 1)
    ax.text(0.1, 0.12, "Pflege geschieht nebenbei", fontsize=7.5, color="#55554f")
    ax.text(5, 0.82, "Pflege ist bewusste Investition", fontsize=7.5, color="#55554f")
    ax = axs[1]
    B = R[(R["variante"] == VARIANTS[0][0])]
    curve(ax, B, "beziehungen", "hoch", "#3a3a36", "o", "Simulation (Netz)")
    for trust, ls, lab in [(1.0, "--", "Analytik: Partner verlässlich"),
                           (0.5, ":", "Analytik: Partner unsicher")]:
        k = zk.k_star(p["mu"] * (1 - Dv), p["p"], p["B"], p["k0"], p["beta"], p["d1"], p["d0"], trust)
        ax.plot(a, k, ls=ls, color="#eb6834", lw=1.8, label=lab)
    ax.set_ylabel("Gepflegte Beziehungen je Person", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax = axs[2]
    for col, c, lab in [("typ_pflege_dominant", "#1baf7a", "Pflegen lohnt immer"),
                        ("typ_hirschjagd", "#2a78d6", "Hirschjagd: nur wenn der Partner pflegt"),
                        ("typ_aufgeben", "#e87ba4", "Aufgeben lohnt")]:
        curve(ax, B, col, "hoch", c, "o", lab)
    ax.set_ylabel("Anteil bestehender Beziehungen", fontsize=9)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, frameon=False, loc="center right")
    for x, t in zip(axs, "abc"):
        style(x)
        xaxis(x)
        x.set_title(t, loc="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    save(fig, "Fig_S3a_mechanismus")


def fig_main(R):
    B = R[R["variante"] == VARIANTS[0][0]]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.7))
    for ax, col, lab in zip(axs, ["beziehungen", "lcc", "ernstfallwert"],
                            ["Beziehungen je Person", "Zusammenhalt: Anteil im größten\nzusammenhängenden Netz",
                             "Wert des Netzes im Ernstfall"]):
        curve(ax, B, col, "hoch", "#2a78d6", "o", "KI wird stärker")
        curve(ax, B, col, "runter", "#eb6834", "s", "KI wird wieder schwächer", ls="--")
        ax.set_ylabel(lab, fontsize=9)
        ax.set_ylim(0, None)
        style(ax)
        xaxis(ax)
    axs[0].legend(fontsize=7.5, frameon=False)
    for x, t in zip(axs, "abc"):
        x.set_title(t, loc="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    save(fig, "Fig_S3b_kollaps_und_hysterese")


def fig_K(RK):
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.5), sharey=True)
    show = [0.0, 0.3, 1.0]
    for ax, K in zip(axs[:3], show):
        H = RK[np.isclose(RK["K"], K)]
        curve(ax, H, "lcc", "hoch", "#2a78d6", "o", "KI wird stärker")
        curve(ax, H, "lcc", "runter", "#eb6834", "s", "KI wird wieder schwächer", ls="--")
        ax.set_title("Wiederaufbaukosten K = %g  (= %.0f × Pflegekosten)" % (K, K / zk.DEFAULTS["mu"]),
                     fontsize=8, loc="left")
        style(ax)
        xaxis(ax)
    axs[0].set_ylabel("Zusammenhalt (größte Komponente)", fontsize=9)
    axs[0].legend(fontsize=7, frameon=False)
    ax = axs[3]
    area = []
    for K in K_LEVELS:
        H = RK[np.isclose(RK["K"], K)]
        up = H[H["phase"] == "hoch"].groupby(["seed", "alpha"])["lcc"].mean().unstack()
        dn = H[H["phase"] == "runter"].groupby(["seed", "alpha"])["lcc"].mean().unstack()
        cols = [a for a in ALPHAS if a in dn.columns]
        area.append((up[cols] - dn[cols]).clip(lower=0).mean(axis=1).values)
    m = np.array([x.mean() for x in area])
    sd = np.array([x.std() for x in area])
    xk = np.array(K_LEVELS) / zk.DEFAULTS["mu"]
    ax.plot(xk, m, color="#3a3a36", lw=2, marker="o", ms=5, markeredgecolor="white")
    ax.fill_between(xk, m - sd, m + sd, color="#3a3a36", alpha=0.12, lw=0)
    ax.set_xlabel("Wiederaufbaukosten K / Pflegekosten μ", fontsize=9)
    ax.set_title("Hysterese: verlorener Zusammenhalt\n(mittlere Lücke hoch − runter)", fontsize=8, loc="left")
    style(ax)
    fig.tight_layout()
    save(fig, "Fig_S3c_hysterese_wiederaufbau")


def fig_policies(R):
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for k, (lab, _) in enumerate(VARIANTS):
        H = R[R["variante"] == lab]
        curve(axs[0], H, "lcc", "hoch", VAR_COLORS[k], VAR_MARK[k], lab)
        curve(axs[1], H, "lcc", "runter", VAR_COLORS[k], VAR_MARK[k], lab, ls="--")
    axs[0].set_title("a   KI wird stärker", loc="left", fontsize=10, fontweight="bold")
    axs[1].set_title("b   KI wird danach wieder schwächer", loc="left", fontsize=10, fontweight="bold")
    for ax in axs:
        ax.set_ylabel("Zusammenhalt (größte Komponente)", fontsize=9)
        ax.set_ylim(0, 1.05)
        style(ax)
        xaxis(ax)
    axs[0].legend(fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout()
    save(fig, "Fig_S3d_politikhebel")


def fig_subsidy(RS):
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    for k, (a, c) in enumerate([(4, "#2a78d6"), (8, "#eb6834"), (16, "#1baf7a")]):
        H = RS[(RS["alpha"] == a) & (RS["phase"] == "hoch")]
        for ax, col in zip(axs, ["lcc", "wohlfahrt"]):
            G = H.groupby("subsidy")[col]
            x = G.mean().index.values * 100
            ax.plot(x, G.mean().values, color=c, lw=2, marker="os^"[k], ms=5, label="α = %d" % a,
                    markeredgecolor="white", markeredgewidth=0.8)
            ax.fill_between(x, G.mean() - G.std().fillna(0), G.mean() + G.std().fillna(0), color=c,
                            alpha=0.12, lw=0)
    axs[0].set_ylabel("Zusammenhalt (größte Komponente)", fontsize=9)
    axs[1].set_ylabel("Wohlfahrt je Person und Periode\n(Ernstfallwert − volle Pflegekosten)", fontsize=8.5)
    for x, t in zip(axs, "ab"):
        x.set_xlabel("Pflegesubvention (% der Kosten)", fontsize=9)
        style(x)
        x.set_title(t, loc="left", fontweight="bold", fontsize=11)
    axs[0].legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    save(fig, "Fig_S3e_subvention_wohlfahrt")


def fig_phase(RM):
    H = RM[RM["phase"] == "hoch"].groupby(["mu", "alpha"])["lcc"].mean().unstack("alpha")
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "blau", ["#f4f8fd", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])
    im = ax.pcolormesh(np.arange(len(ALPHAS) + 1) - 0.5, np.arange(len(MU_LEVELS) + 1) - 0.5,
                       H.values, cmap=cmap, vmin=0, vmax=1, shading="flat")
    cs = ax.contour(np.arange(len(ALPHAS)), np.arange(len(MU_LEVELS)), H.values, levels=[0.5],
                    colors="#eb6834", linewidths=2)
    ax.set_xticks(np.arange(len(ALPHAS)))
    ax.set_xticklabels([str(a) for a in ALPHAS], fontsize=8)
    ax.set_yticks(np.arange(len(MU_LEVELS)))
    ax.set_yticklabels(["%.2f" % (m / zk.DEFAULTS["mu"]) for m in MU_LEVELS], fontsize=8)
    ax.set_xlabel("KI-Fähigkeit α", fontsize=9)
    ax.set_ylabel("Pflegekosten μ (relativ zum Standard)", fontsize=9)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Zusammenhalt (größte Komponente)", fontsize=8)
    ax.text(0.02, 1.02, "orange: 50 %-Grenze", transform=ax.transAxes, fontsize=7.5, color="#55554f")
    fig.tight_layout()
    save(fig, "Fig_S3f_phasendiagramm")


# ---------------------------------------------------------------------------
def summarize(R, RW, RK, RS, RM, D):
    L = ["SCHRITT 3 (v3) - Kooperation mit der Zukunft", "=" * 70,
         "Parameter: %s" % {k: v for k, v in zk.DEFAULTS.items()},
         "Abhaengigkeit D(alpha) Wissenschaft: %s" % {a: round(D[a], 3) for a in ALPHAS}, ""]

    def tab(H, cols):
        up = H[H["phase"] == "hoch"].groupby("alpha")[cols].mean()
        dn = H[H["phase"] == "runter"].groupby("alpha")[cols].mean()
        out = up.add_suffix("_hoch").join(dn.add_suffix("_runter"))
        return out.round(3).to_string()

    for lab, _ in VARIANTS:
        L.append(lab)
        L.append(tab(R[R["variante"] == lab], ["beziehungen", "lcc", "ernstfallwert", "wohlfahrt"]))
        L.append("")
    L.append("Spieltypen der Beziehungen (Basis, KI wird staerker):")
    L.append(R[(R["variante"] == VARIANTS[0][0]) & (R["phase"] == "hoch")].groupby("alpha")[
        ["typ_pflege_dominant", "typ_hirschjagd", "typ_aufgeben"]].mean().round(3).to_string())
    L.append("")
    L.append("Wirtschaft (Basis):")
    L.append(tab(RW, ["beziehungen", "lcc", "ernstfallwert"]))
    L.append("")
    L.append("Hysterese nach Wiederaufbaukosten K (Zusammenhalt hoch | runter):")
    for K in K_LEVELS:
        H = RK[np.isclose(RK["K"], K)]
        up = H[H["phase"] == "hoch"].groupby("alpha")["lcc"].mean()
        dn = H[H["phase"] == "runter"].groupby("alpha")["lcc"].mean()
        L.append("  K=%-5g " % K + " ".join("a%g:%.2f|%.2f" % (a, up[a], dn.get(a, np.nan)) for a in ALPHAS))
    L.append("")
    L.append("Subvention (alpha = 8): Zusammenhalt und Wohlfahrt mit vollen Kosten")
    L.append(RS[(RS["alpha"] == 8) & (RS["phase"] == "hoch")].groupby("subsidy")[
        ["lcc", "beziehungen", "wohlfahrt"]].mean().round(4).to_string())
    txt = "\n".join(L)
    with open(os.path.join(OUT, "ERGEBNISSE.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt)


def main():
    setup_quick()
    t0 = time.time()
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    print("=== Schritt 3 (v3): Kooperation mit der Zukunft | QUICK=%s ===" % QUICK)
    nets, D = networks_and_dependence("Wissenschaft", SEEDS_MAIN)
    netsW, DW = networks_and_dependence("Wirtschaft", SEEDS_SIDE)
    print("  Netze und D(alpha) fertig (%.0f s)" % (time.time() - t0))

    rows = []
    for lab, kw in VARIANTS:
        rows += run(lab, nets, D, SEEDS_MAIN, **kw)
        print("  A %-45s fertig (%.0f s)" % (lab, time.time() - t0))
    R = pd.DataFrame(rows)
    RW = pd.DataFrame(run("Wirtschaft Basis", netsW, DW, SEEDS_SIDE))
    print("  B Wirtschaft fertig (%.0f s)" % (time.time() - t0))
    rows = []
    for K in K_LEVELS:
        rows += run("K", nets, D, SEEDS_SIDE, K=K)
    RK = pd.DataFrame(rows)
    print("  C Hysterese fertig (%.0f s)" % (time.time() - t0))
    rows = []
    for sb in SUBSIDIES:
        rows += run("Subvention", nets, D, SEEDS_SIDE, back=False, subsidy=sb)
    RS = pd.DataFrame(rows)
    print("  D Subvention fertig (%.0f s)" % (time.time() - t0))
    rows = []
    for mu in MU_LEVELS:
        rows += run("mu", nets, D, SEEDS_SIDE[:2], back=False, mu=mu)
    RM = pd.DataFrame(rows)
    print("  E Phasendiagramm fertig (%.0f s)" % (time.time() - t0))

    for name, X in [("S3_varianten", R), ("S3_wirtschaft", RW), ("S3_hysterese_K", RK),
                    ("S3_subvention", RS), ("S3_phasendiagramm", RM)]:
        X.to_csv(os.path.join(OUT, name + ".csv"), index=False, float_format="%.6g")
    pd.DataFrame({"alpha": ALPHAS, "D_wissenschaft": [D[a] for a in ALPHAS],
                  "D_wirtschaft": [DW[a] for a in ALPHAS]}).to_csv(os.path.join(OUT, "S3_abhaengigkeit.csv"),
                                                                   index=False)
    fig_mechanism(R, D, zk.DEFAULTS)
    fig_main(R)
    fig_K(RK)
    fig_policies(R)
    fig_subsidy(RS)
    fig_phase(RM)
    summarize(R, RW, RK, RS, RM, D)
    runtime = time.time() - t0
    with open(os.path.join(OUT, "INFO.txt"), "w", encoding="utf-8") as f:
        f.write("Schritt 3 (v3) | QUICK=%s | Laufzeit %.1f min | Python %s | numpy %s\n"
                % (QUICK, runtime / 60, sys.version.split()[0], np.__version__))
    print("Schritt 3 fertig nach %.1f min" % (runtime / 60))


if __name__ == "__main__":
    main()
