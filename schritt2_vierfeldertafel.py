# -*- coding: utf-8 -*-
"""
=====================================================================
  KI und das Kooperationsdilemma auf Netzwerken  --  SCHRITT 2 (v2)
  Die Vierfeldertafel: Kooperation mit Menschen oder mit KI?
=====================================================================

Wird von run_all.py nach Schritt 1 aufgerufen (nutzt results_S1/S1_netze.csv).

Das Spiel (Auszahlung fuer den Zeilenspieler)
---------------------------------------------
                         Partner: Mensch (H)    Partner: KI
    Ich: Mensch (H)      R = b - c              S = -c
    Ich: KI              T = s + theta*b        P = s

  b, c   Nutzen und Kosten menschlicher Kooperation, direkt aus dem
         Netzwerkmodell (Mittel ueber alle Kanten des Netzes bei KI-Staerke alpha)
  s      Nettowert der KI fuer dieselbe Aufgabe:
             s = phi(alpha) * b0 * amp(alpha)  -  c_A
         phi   Anteil der Faehigkeitsluecke, den KI fuellt (Modell)
         b0    Wert eines menschlichen Partners vor KI (Modell, alpha = 0)
         amp   KI-Amplifikation 1 + lambda*alpha*u (Modell)
         c_A   Kosten der KI-Nutzung fuer diese Aufgabe (Pruefen, Fehler,
               Daten, fehlendes Vertrauen) -- der KONTEXT der Aufgabe
  theta  wie viel ein KI-Nutzer noch von einem kooperierenden Menschen hat

Vier Regime (je nach s):
  s < -c                         Mensch dominiert (keine Dilemma)
  -c < s < (1-theta) b - c       Hirschjagd (Koordination/Vertrauen)
  (1-theta) b - c < s < b - c    Gefangenendilemma (KI individuell besser,
                                 menschliche Kooperation kollektiv besser)
  s > b - c                      KI dominiert (kein Dilemma)

Netzwerk (Nowak/Tarnita sigma-Regel, exakt bei schwacher Selektion, DB):
  Mensch setzt sich durch, wenn  sigma*R + S > T + sigma*P
  <=>  s < s* = [sigma (b - c) - c - theta b] / (1 + sigma)
  sigma aus den Koaleszenzzeiten des Modellnetzes (Allen et al. 2017):
  sigma = (-tau1 + tau2 + tau3) / (tau1 + tau2 - tau3);  gut gemischt: sigma = 1
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
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import struktur as st    # noqa: E402

QUICK = False
OUT = os.path.join(HERE, "results_S2")
IN_CSV = os.path.join(HERE, "results_S1", "S1_netze.csv")

THETA = 0.5                                   # Standard
THETAS = [0.0, 0.25, 0.5, 0.75, 1.0]
CA_GRID = np.linspace(0.0, 16.0, 321)        # KI-Kosten in Einheiten von c(alpha=0)
CA_SHOW = [0.0, 2.0, 6.0]                     # fuer die T-S-Ebene

SCEN_COLORS = {"Wissenschaft": "#2a78d6", "Wirtschaft": "#eb6834", "Sozial": "#1baf7a",
               "Wissenschaft + Koordination (kappa=1)": "#eda100"}
SCEN_MARK = {"Wissenschaft": "o", "Wirtschaft": "s", "Sozial": "^",
             "Wissenschaft + Koordination (kappa=1)": "D"}
REG_NAMES = ["Mensch dominiert", "Hirschjagd", "Gefangenendilemma", "KI dominiert"]
REG_COLORS = ["#1baf7a", "#2a78d6", "#eb6834", "#e87ba4"]


# ---------------------------------------------------------------------------
# Spieltheorie
# ---------------------------------------------------------------------------
def payoffs(b, c, s, theta):
    return b - c, -c, s + theta * b, s          # R, S, T, P


def regime(b, c, s, theta):
    """0 Mensch dominiert, 1 Hirschjagd, 2 Gefangenendilemma, 3 KI dominiert."""
    s = np.asarray(s, dtype=float)
    out = np.full(s.shape, 3)
    out[s < b - c] = 2
    out[s < (1 - theta) * b - c] = 1
    out[s < -c] = 0
    return out


def s_threshold(b, c, theta, sigma):
    """Mensch setzt sich durch, wenn s < s*."""
    return (sigma * (b - c) - c - theta * b) / (1.0 + sigma)


def sigma_from_tau(t1, t2, t3):
    return (-t1 + t2 + t3) / (t1 + t2 - t3)


# ---------------------------------------------------------------------------
# Daten aus Schritt 1
# ---------------------------------------------------------------------------
def load():
    if not os.path.exists(IN_CSV):
        raise RuntimeError("results_S1/S1_netze.csv fehlt - bitte ueber run_all.py starten.")
    D = pd.read_csv(IN_CSV)
    D["sigma_netz"] = sigma_from_tau(D["tau1"], D["tau2"], D["tau3"])
    base = D[D["alpha"] == 0].set_index(["szenario", "seed"])
    D["b0"] = [base.loc[(z, s), "b_mittel"] for z, s in zip(D["szenario"], D["seed"])]
    D["c0"] = [base.loc[(z, s), "c_mittel"] for z, s in zip(D["szenario"], D["seed"])]
    D["v_ki"] = D["phi_mittel"] * D["b0"] * D["amp_mittel"]      # Bruttowert der KI
    return D


def evaluate(D):
    """Fuer jede Zeile (Szenario x Seed x alpha) und jede KI-Kostenstufe:
    Regime, Schwellen, T-S-Koordinaten, Vorhersage Netz vs. gut gemischt."""
    rows = []
    for _, r in D.iterrows():
        for th in THETAS:
            for g in CA_GRID:
                cA = g * r["c0"]
                s = r["v_ki"] - cA
                b, c = r["b_mittel"], r["c_mittel"]
                R, S, T, P = payoffs(b, c, s, th)
                sn = s_threshold(b, c, th, r["sigma_netz"])
                sw = s_threshold(b, c, th, 1.0)
                rows.append(dict(szenario=r["szenario"], seed=r["seed"], alpha=r["alpha"],
                                 theta=th, ki_kosten=g, b=b, c=c, s=s, sigma=r["sigma_netz"],
                                 regime=int(regime(b, c, s, th)),
                                 mensch_netz=bool(s < sn), mensch_gemischt=bool(s < sw),
                                 s_stern_netz=sn, s_stern_gemischt=sw,
                                 T_norm=(T - P) / (R - P) if R > P else np.nan,
                                 S_norm=(S - P) / (R - P) if R > P else np.nan))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Validierung: sigma-Regel gegen exakte Markov-Kette (N = 7, unregelmaessig)
# ---------------------------------------------------------------------------
def validate(rng):
    N = 7
    A = np.zeros((N, N), dtype=bool)
    for i, j in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 0), (0, 3), (1, 5), (2, 6), (0, 4)]:
        A[i, j] = A[j, i] = True
    W = A.astype(float)
    P = W / W.sum(1)[:, None]
    r = st.structure_condition(W)
    sig = sigma_from_tau(r["tau1"], r["tau2"], r["tau3"])
    nS = 2 ** N
    X = ((np.arange(nS)[:, None] >> np.arange(N)) & 1).astype(float)

    def rho_diff(Rr, Ss, Tt, Pp, delta=1e-4):
        M = np.zeros((nS, nS))
        for s in range(nS):
            x = X[s]
            q = P @ x
            F = 1 + delta * (x * (Rr * q + Ss * (1 - q)) + (1 - x) * (Tt * q + Pp * (1 - q)))
            for i in range(N):
                pr = W[i] * F
                pH = (pr / pr.sum()) @ x
                M[s, s | (1 << i)] += pH / N
                M[s, s & ~(1 << i)] += (1 - pH) / N
        tr = np.arange(1, nS - 1)
        h = np.linalg.solve(np.eye(len(tr)) - M[np.ix_(tr, tr)], M[tr, nS - 1])
        H = np.zeros(nS)
        H[tr] = h
        H[-1] = 1
        return np.mean([H[1 << i] for i in range(N)]) - np.mean([1 - H[(nS - 1) & ~(1 << i)] for i in range(N)])

    rows = []
    for k in range(3):
        Rr, Tt, Pp = rng.uniform(-1, 2, 3)
        lo, hi = -20.0, 20.0
        for _ in range(32):
            mid = 0.5 * (lo + hi)
            if rho_diff(Rr, mid, Tt, Pp) > 0:
                hi = mid
            else:
                lo = mid
        theo = Tt + sig * Pp - sig * Rr
        rows.append(dict(test="sigma-Regel 2x2-Spiel %d (N=7, unregelmaessig)" % (k + 1),
                         theorie=theo, exakt=0.5 * (lo + hi)))
    V = pd.DataFrame(rows)
    V["abs_abweichung"] = (V["theorie"] - V["exakt"]).abs()
    return V


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


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=200)
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    plt.close(fig)


def fig_table():
    """Abb. 1: Die Vierfeldertafel (fuer das Manuskript)."""
    fig = plt.figure(figsize=(8.6, 4.9))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.6)
    x0, y0, w, h = 2.6, 0.35, 3.6, 2.35
    cells = [
        (0, 1, "Mensch – Mensch", "R = b − c", "Beide kooperieren;\njeder gewinnt b und zahlt c", "#e6f6ef"),
        (1, 1, "Mensch – KI", "S = −c", "Ich investiere in den anderen,\nder arbeitet mit seiner KI", "#fbe7e1"),
        (0, 0, "KI – Mensch", "T = s + θb", "Ich nutze KI und profitiere\nnoch teilweise vom anderen", "#fdf1d8"),
        (1, 0, "KI – KI", "P = s", "Beide arbeiten allein\nmit ihrer KI", "#eaf1fb"),
    ]
    for col, row, title, formula, text, fc in cells:
        x, y = x0 + col * w, y0 + row * h
        ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor="white", lw=3))
        ax.text(x + w / 2, y + h - 0.32, title, ha="center", va="center", fontsize=9, color="#55554f")
        ax.text(x + w / 2, y + h / 2 + 0.12, formula, ha="center", va="center", fontsize=17,
                fontweight="bold", color="#1f1f1c")
        ax.text(x + w / 2, y + 0.45, text, ha="center", va="center", fontsize=8, color="#3a3a36")
    ax.text(x0 + w / 2, y0 + 2 * h + 0.3, "anderer kooperiert\nmit Menschen", ha="center", va="bottom",
            fontsize=10, fontweight="bold", color="#1f1f1c")
    ax.text(x0 + 1.5 * w, y0 + 2 * h + 0.3, "anderer arbeitet\nmit KI", ha="center", va="bottom",
            fontsize=10, fontweight="bold", color="#1f1f1c")
    ax.text(x0 - 0.2, y0 + 1.5 * h, "ich kooperiere\nmit Menschen", ha="right", va="center",
            fontsize=10, fontweight="bold", color="#1f1f1c")
    ax.text(x0 - 0.2, y0 + 0.5 * h, "ich arbeite\nmit KI", ha="right", va="center",
            fontsize=10, fontweight="bold", color="#1f1f1c")
    ax.text(x0 - 0.2, y0 + 2 * h + 0.3, "Meine Auszahlung", ha="right", va="bottom", fontsize=8,
            color="#55554f", style="italic")
    save(fig, "Fig_S2a_vierfeldertafel")


def fig_regimes_axis():
    """Abb.: die vier Regime entlang des KI-Werts s (schematisch, mit Formeln)."""
    b, c, th = 1.0, 0.35, 0.5
    bounds = [-1.0, -c, (1 - th) * b - c, b - c, 1.3]
    fig, ax = plt.subplots(figsize=(8.6, 1.9))
    for k in range(4):
        ax.add_patch(Rectangle((bounds[k], 0), bounds[k + 1] - bounds[k], 1, facecolor=REG_COLORS[k],
                               alpha=0.85, edgecolor="white", lw=2))
        ax.text(0.5 * (bounds[k] + bounds[k + 1]), 0.5, REG_NAMES[k].replace(" ", "\n", 1)
                if k in (0, 3) else REG_NAMES[k], ha="center", va="center", fontsize=8.5,
                color="white", fontweight="bold")
    for xv, lab in zip(bounds[1:4], ["−c", "(1−θ)b − c", "b − c"]):
        ax.text(xv, -0.12, lab, ha="center", va="top", fontsize=8.5, color="#1f1f1c")
    ax.annotate("", xy=(1.3, 1.25), xytext=(-1.0, 1.25),
                arrowprops=dict(arrowstyle="->", color="#55554f", lw=1.2))
    ax.text(0.15, 1.33, "Nettowert der KI für die Aufgabe  s = φ·b₀·amp − c_A   (steigt mit KI-Fähigkeit, sinkt mit KI-Kosten)",
            ha="center", va="bottom", fontsize=8, color="#55554f")
    ax.set_xlim(-1.0, 1.3)
    ax.set_ylim(-0.45, 1.6)
    ax.axis("off")
    fig.tight_layout()
    save(fig, "Fig_S2b_vier_regime")


def fig_ts_plane(E):
    """Abb.: T-S-Ebene (Santos, Pacheco & Lenaerts 2006) mit Pfaden ueber alpha."""
    fig, axs = plt.subplots(1, len(CA_SHOW), figsize=(12, 4.0), sharey=True)
    G = E[(E["theta"] == THETA)]
    for ax, g in zip(axs, CA_SHOW):
        ax.add_patch(Rectangle((-0.2, 0), 1.2, 1.6, facecolor=REG_COLORS[0], alpha=0.10, lw=0))
        ax.add_patch(Rectangle((-0.2, -2.6), 1.2, 2.6, facecolor=REG_COLORS[1], alpha=0.10, lw=0))
        ax.add_patch(Rectangle((1.0, -2.6), 3.0, 2.6, facecolor=REG_COLORS[2], alpha=0.10, lw=0))
        ax.text(0.4, 1.4, "Mensch dominiert", ha="center", fontsize=7.5, color="#3a3a36")
        ax.text(0.4, -2.45, "Hirschjagd", ha="center", fontsize=7.5, color="#3a3a36")
        ax.text(2.5, -2.45, "Gefangenendilemma", ha="center", fontsize=7.5, color="#3a3a36")
        tt = np.linspace(-0.2, 4, 10)
        ax.plot(tt, tt - 1, color="#3a3a36", lw=1, ls="--", label="Grenze ohne Netz (σ = 1)")
        sig_w = G[(G["szenario"] == "Wissenschaft") & (G["alpha"] == 8)]["sigma"].mean()
        ax.plot(tt, tt - sig_w, color="#3a3a36", lw=1, ls=":",
                label="Grenze Wissenschaftsnetz α = 8 (σ = %.1f)" % sig_w)
        Gg = G[np.isclose(G["ki_kosten"], g)]
        for name in SCEN_COLORS:
            H = Gg[Gg["szenario"] == name].groupby("alpha")[["T_norm", "S_norm"]].mean().dropna()
            ax.plot(H["T_norm"], H["S_norm"], color=SCEN_COLORS[name], marker=SCEN_MARK[name],
                    ms=5, lw=2, label=name, markeredgecolor="white", markeredgewidth=0.8)
            if len(H):
                ax.plot(H["T_norm"].iloc[0], H["S_norm"].iloc[0], marker=SCEN_MARK[name], ms=8,
                        color="white", markeredgecolor=SCEN_COLORS[name], markeredgewidth=1.5)
        ax.axhline(0, color="#b7b6ae", lw=0.8)
        ax.axvline(1, color="#b7b6ae", lw=0.8)
        ax.set_xlim(-0.2, 4)
        ax.set_ylim(-2.6, 1.6)
        ax.set_xlabel("Versuchung T (normiert, R = 1, P = 0)", fontsize=9)
        ax.set_title("KI-Kosten c_A = %g · c₀" % g, fontsize=9, loc="left")
        style(ax, grid=False)
    axs[0].set_ylabel("Ausnutzungsrisiko S (normiert)", fontsize=9)
    axs[-1].legend(fontsize=7, frameon=True, loc="upper right")
    fig.text(0.01, 0.01, "Offener Marker: α = 0; Pfade laufen mit steigender KI-Fähigkeit. Oberhalb der "
             "Grenzlinien setzt sich Kooperation mit Menschen durch. "
             "Pfade verlassen die Ebene, sobald KI dominiert (R < P).", fontsize=7, color="#55554f")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "Fig_S2c_TS_ebene")


def fig_phase(E):
    """Abb.: Phasendiagramm alpha x KI-Kosten, Regime + Netz- und gut gemischte Grenze."""
    names = list(SCEN_COLORS)
    fig, axs = plt.subplots(1, len(names), figsize=(13, 3.7), sharey=True)
    G = E[E["theta"] == THETA]
    for ax, name in zip(axs, names):
        H = G[G["szenario"] == name]
        alphas = np.sort(H["alpha"].unique())
        reg = H.groupby(["ki_kosten", "alpha"])["regime"].agg(lambda v: int(np.round(np.median(v))))
        Z = reg.unstack("alpha").reindex(columns=alphas).values
        cmap = matplotlib.colors.ListedColormap(REG_COLORS)
        ax.pcolormesh(np.arange(len(alphas) + 1) - 0.5, np.r_[CA_GRID - 0.025, CA_GRID[-1] + 0.025],
                      Z, cmap=cmap, vmin=-0.5, vmax=3.5, alpha=0.55, shading="flat")
        for col, ls, lab in [("mensch_netz", "-", "Grenze mit Netzstruktur"),
                             ("mensch_gemischt", "--", "Grenze ohne Netzstruktur")]:
            frac = H.groupby(["alpha", "ki_kosten"])[col].mean().unstack("ki_kosten")
            line = []
            for a in alphas:
                ok = CA_GRID[frac.loc[a].values >= 0.5]
                line.append(ok.min() if len(ok) else np.nan)
            ax.plot(np.arange(len(alphas)), line, color="#1f1f1c", lw=1.8, ls=ls, label=lab)
        ax.set_xticks(np.arange(len(alphas)))
        ax.set_xticklabels(["%g" % a for a in alphas], fontsize=7)
        ax.set_xlabel("KI-Fähigkeit α", fontsize=9)
        ax.set_title(name if len(name) < 20 else "Wiss. + Koordination", fontsize=9, loc="left")
        ax.set_ylim(0, CA_GRID[-1])
        style(ax, grid=False)
    axs[0].set_ylabel("KI-Kosten der Aufgabe  c_A / c₀", fontsize=9)
    handles = [Rectangle((0, 0), 1, 1, facecolor=REG_COLORS[k], alpha=0.55) for k in range(4)]
    l1 = axs[-1].legend(handles, REG_NAMES, fontsize=7, frameon=True, loc="upper right")
    axs[-1].add_artist(l1)
    axs[0].legend(fontsize=7, frameon=True, loc="upper left")
    fig.text(0.01, 0.01, "Oberhalb der Linien setzt sich Kooperation mit Menschen durch (σ-Regel, schwache "
             "Selektion, θ = %g)." % THETA, fontsize=7, color="#55554f")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "Fig_S2d_phasendiagramm")


def fig_network_rescue(E):
    """Abb.: Wie stark verschiebt die Netzstruktur die Grenze? (in Einheiten von c0)."""
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    G = E[(E["theta"] == THETA) & (E["ki_kosten"] == 0)].copy()
    D0 = G.groupby(["szenario", "seed", "alpha"]).first().reset_index()
    D0["gewinn"] = (D0["s_stern_netz"] - D0["s_stern_gemischt"]) / D0["c"]
    for name in SCEN_COLORS:
        H = D0[D0["szenario"] == name].groupby("alpha")
        x = np.array(sorted(D0["alpha"].unique()))
        m, sd = H["gewinn"].mean().values, H["gewinn"].std().values
        axs[0].plot(x, m, color=SCEN_COLORS[name], marker=SCEN_MARK[name], lw=2, ms=5, label=name,
                    markeredgecolor="white", markeredgewidth=0.8)
        axs[0].fill_between(x, m - sd, m + sd, color=SCEN_COLORS[name], alpha=0.12, lw=0)
        m2 = H["sigma"].mean().values
        axs[1].plot(x, m2, color=SCEN_COLORS[name], marker=SCEN_MARK[name], lw=2, ms=5,
                    markeredgecolor="white", markeredgewidth=0.8)
    axs[0].set_ylabel("Netzwerk-Vorteil für Kooperation\nmit Menschen  (s*_Netz − s*_gemischt)/c",
                      fontsize=8.5)
    axs[1].set_ylabel("Strukturkoeffizient σ", fontsize=9)
    axs[1].axhline(1, color="#3a3a36", lw=1, ls="--")
    for a, t in zip(axs, "ab"):
        style(a)
        try:
            a.set_xscale("symlog", linthresh=4)
        except TypeError:
            a.set_xscale("symlog", linthreshx=4)
        a.set_xticks([0, 1, 2, 4, 8, 16])
        a.set_xticklabels(["0", "1", "2", "4", "8", "16"])
        a.set_xlabel("KI-Fähigkeit α", fontsize=9)
        a.set_title(t, loc="left", fontweight="bold", fontsize=11)
    axs[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    save(fig, "Fig_S2e_netzwerkvorteil")


def summary(E, V, runtime):
    L = ["SCHRITT 2 (v2) - Vierfeldertafel: Kooperation mit Menschen oder mit KI", "=" * 70,
         "theta = %g (Standard) | KI-Kosten c_A in Einheiten von c0 | Laufzeit %.1f min" % (THETA, runtime / 60), ""]
    L.append("Validierung sigma-Regel (exakte Markov-Kette):")
    for _, r in V.iterrows():
        L.append("  %-50s Theorie %.5f | exakt %.5f" % (r["test"], r["theorie"], r["exakt"]))
    L.append("")
    G = E[E["theta"] == THETA]
    for name in SCEN_COLORS:
        H = G[G["szenario"] == name]
        L.append(name)
        for a in sorted(H["alpha"].unique()):
            Ha = H[H["alpha"] == a]
            # kritische KI-Kosten, ab denen Menschen sich durchsetzen (Median ueber Seeds)
            crit_n = Ha.groupby("ki_kosten")["mensch_netz"].mean()
            crit_w = Ha.groupby("ki_kosten")["mensch_gemischt"].mean()
            cn = crit_n[crit_n >= 0.5].index.min()
            cw = crit_w[crit_w >= 0.5].index.min()
            reg0 = REG_NAMES[int(np.median(Ha[Ha["ki_kosten"] == 0]["regime"]))]
            L.append("  alpha=%4g  ohne KI-Kosten: %-18s | Mensch gewinnt ab c_A/c0 = %s (Netz) vs %s (gemischt)"
                     % (a, reg0, "%.2f" % cn if pd.notna(cn) else ">16", "%.2f" % cw if pd.notna(cw) else ">16"))
        L.append("")
    txt = "\n".join(L)
    with open(os.path.join(OUT, "ERGEBNISSE.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt)


def main():
    t0 = time.time()
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    print("=== Schritt 2 (v2): Vierfeldertafel Mensch vs. KI ===")
    V = validate(np.random.default_rng(5))
    V.to_csv(os.path.join(OUT, "S2_validierung.csv"), index=False)
    D = load()
    E = evaluate(D)
    keep = np.isin(np.round(E["ki_kosten"], 6), np.round(CA_GRID[::8], 6))   # kompakte Ausgabe
    E[keep].to_csv(os.path.join(OUT, "S2_spiel.csv"), index=False, float_format="%.6g")
    fig_table()
    fig_regimes_axis()
    fig_ts_plane(E)
    fig_phase(E)
    fig_network_rescue(E)
    runtime = time.time() - t0
    summary(E, V, runtime)
    with open(os.path.join(OUT, "INFO.txt"), "w", encoding="utf-8") as f:
        f.write("Schritt 2 (v2) | Laufzeit %.1f min | Python %s | numpy %s | pandas %s\n"
                % (runtime / 60, sys.version.split()[0], np.__version__, pd.__version__))


if __name__ == "__main__":
    main()
