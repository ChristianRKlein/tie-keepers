# -*- coding: utf-8 -*-
"""Publication figures (English) for the 'cooperating with the future' manuscript.
Reads results_S1..S6 (default: this folder) and writes paper_figs/Fig1..Fig5, FigS1, FigS2."""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
import networkx as nx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import netzmodell as nm   # noqa
import zukunft as zk      # noqa
from struktur import largest_component  # noqa

RES = sys.argv[1] if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]) else HERE
OUT = os.path.join(HERE, "paper_figs")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "Liberation Sans", "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.2, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "pdf.fonttype": 42, "svg.fonttype": "none"})
BLUE, ORANGE, GREEN, YELLOW, PINK, INK = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#3a3a36"
GREY, LIGHT = "#8c8b84", "#b7b6ae"
ALPHAS = [0, 1, 2, 3, 4, 6, 8, 12, 16]
MM = 1 / 25.4
W2 = 180 * MM


def style(ax, grid=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#77766f")
    ax.tick_params(colors="#3a3a36")
    if grid:
        ax.grid(axis="y", color="#ebeae5", lw=0.5)
        ax.set_axisbelow(True)


def xalpha(ax, label=True):
    try:
        ax.set_xscale("symlog", linthresh=4, linscale=1.2)
    except TypeError:
        ax.set_xscale("symlog", linthreshx=4)
    ax.set_xticks([0, 1, 2, 4, 8, 16])
    ax.set_xticklabels(["0", "1", "2", "4", "8", "16"])
    ax.set_xlim(-0.2, 17)
    if label:
        ax.set_xlabel("AI capability, α")


def panel(ax, letter, x=-0.18, y=1.06):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")


def curve(ax, R, col, phase, color, marker, label=None, ls="-", ms=3.2):
    G = R[R["phase"] == phase].groupby("alpha")[col]
    x = G.mean().index.values
    m, sd = G.mean().values, G.std().fillna(0).values
    ax.fill_between(x, m - sd, m + sd, color=color, alpha=0.15, lw=0)
    ax.plot(x, m, ls=ls, color=color, lw=1.3, marker=marker, ms=ms, label=label,
            markeredgecolor="white", markeredgewidth=0.5)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300)
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
def fig1():
    S3 = pd.read_csv(os.path.join(RES, "results_S3", "S3_abhaengigkeit.csv"))
    fig = plt.figure(figsize=(W2, 72 * MM))
    # a: payoff table
    ax = fig.add_axes([0.0, 0.05, 0.36, 0.88])
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    x0, y0, w, h = 2.7, 0.3, 3.5, 2.95
    cells = [(0, 1, "b − c", "both cooperate", "#e3f4ec"), (1, 1, "−c", "I invest, partner\nworks with AI", "#fbe5dc"),
             (0, 0, "s + θb", "I use AI, still gain\nfrom the partner", "#fcefd2"), (1, 0, "s", "both work\nwith AI", "#e4eefb")]
    for c, r, f, t, fc in cells:
        X, Y = x0 + c * w, y0 + r * h
        ax.add_patch(Rectangle((X, Y), w, h, facecolor=fc, edgecolor="white", lw=2))
        ax.text(X + w / 2, Y + h * 0.62, f, ha="center", va="center", fontsize=10, fontweight="bold", color="#1f1f1c")
        ax.text(X + w / 2, Y + h * 0.24, t, ha="center", va="center", fontsize=5.6, color="#3a3a36")
    ax.text(x0 + w / 2, y0 + 2 * h + 0.2, "Partner:\nhuman", ha="center", va="bottom", fontsize=6.5, fontweight="bold")
    ax.text(x0 + 1.5 * w, y0 + 2 * h + 0.2, "Partner:\nAI", ha="center", va="bottom", fontsize=6.5, fontweight="bold")
    ax.text(x0 - 0.15, y0 + 1.5 * h, "Me:\nhuman", ha="right", va="center", fontsize=6.5, fontweight="bold")
    ax.text(x0 - 0.15, y0 + 0.5 * h, "Me:\nAI", ha="right", va="center", fontsize=6.5, fontweight="bold")
    ax.text(0.0, 7.75, "a", fontsize=9, fontweight="bold", va="top")
    ax.text(0.55, 7.75, "One task: my payoff", fontsize=6.5, va="top", color="#55554f")
    # b: regimes along s
    ax = fig.add_axes([0.40, 0.60, 0.27, 0.28])
    b, c, th = 1.0, 0.35, 0.5
    bounds = [-1.0, -c, (1 - th) * b - c, b - c, 1.3]
    names = ["Human\ndominates", "Stag\nhunt", "Prisoner's\ndilemma", "AI\ndominates"]
    cols = [GREEN, BLUE, ORANGE, PINK]
    for k in range(4):
        ax.add_patch(Rectangle((bounds[k], 0), bounds[k + 1] - bounds[k], 1, facecolor=cols[k], edgecolor="white", lw=1.2))
        ax.text(0.5 * (bounds[k] + bounds[k + 1]), 0.5, names[k], ha="center", va="center", fontsize=5.4,
                color="white", fontweight="bold")
    for xv, lab in zip(bounds[1:4], ["−c", "(1−θ)b − c", "b − c"]):
        ax.text(xv, -0.08, lab, ha="center", va="top", fontsize=5.6)
    ax.annotate("", xy=(1.3, 1.22), xytext=(-1.0, 1.22), arrowprops=dict(arrowstyle="->", color=INK, lw=0.7))
    ax.text(0.15, 1.3, "net value of AI for the task, s", ha="center", va="bottom", fontsize=6)
    ax.set_xlim(-1.0, 1.3)
    ax.set_ylim(-0.5, 1.6)
    ax.axis("off")
    ax.text(-1.0, 1.75, "b", fontsize=9, fontweight="bold", va="bottom")
    # c: tie mechanics
    ax = fig.add_axes([0.40, 0.04, 0.27, 0.46])
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    rows = [("both maintain", "tie persists", 1.0, "-"), ("one maintains", "decays with prob. w", 0.6, "--"),
            ("nobody maintains", "decays with prob. d₀ > w", 0.3, ":")]
    for k, (lab, res, alph, ls) in enumerate(rows):
        y = 8.2 - k * 3.0
        for x, filled in ((1.0, True), (3.6, k == 0)):
            ax.add_patch(Circle((x, y), 0.55, facecolor=BLUE if filled else "white", edgecolor=BLUE, lw=0.9))
        ax.plot([1.55, 3.05], [y, y], color=INK, lw=1.4, ls=ls, alpha=alph)
        ax.text(4.6, y + 0.45, lab, fontsize=6.2, va="center", fontweight="bold")
        ax.text(4.6, y - 0.55, res, fontsize=5.8, va="center", color="#55554f")
    ax.text(0.0, 10.4, "c", fontsize=9, fontweight="bold", va="bottom")
    ax.text(0.8, 10.4, "Tie maintenance needs both sides", fontsize=6.5, va="bottom", color="#55554f")
    # d: by-product share
    ax = fig.add_axes([0.76, 0.17, 0.22, 0.70])
    a = S3["alpha"].values
    D = S3["D_wissenschaft"].values
    ax.fill_between(a, 0, D, color=BLUE, alpha=0.18, lw=0)
    ax.fill_between(a, D, 1, color=ORANGE, alpha=0.14, lw=0)
    ax.plot(a, D, color=BLUE, lw=1.4, marker="o", ms=3, markeredgecolor="white", markeredgewidth=0.5)
    ax.text(0.3, 0.12, "by-product\n(carried by joint work)", fontsize=5.6, color=BLUE)
    ax.text(2.6, 0.62, "deliberate\ninvestment", fontsize=5.6, color=ORANGE, ha="center")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of tie maintenance, D(α)")
    style(ax, grid=False)
    xalpha(ax)
    panel(ax, "d", x=-0.32)
    save(fig, "Fig1_concept")


def fig2():
    R = pd.read_csv(os.path.join(RES, "results_S3", "S3_varianten.csv"))
    B = R[R["variante"] == R["variante"].unique()[0]]
    S3 = pd.read_csv(os.path.join(RES, "results_S3", "S3_abhaengigkeit.csv")).set_index("alpha")["D_wissenschaft"]
    p = zk.DEFAULTS
    fig, axs = plt.subplots(1, 4, figsize=(W2, 56 * MM))
    ax = axs[0]
    for T, col, mk in [(10, ORANGE, "^"), (20, BLUE, "o"), (50, GREEN, "s")]:
        thr = [zk.tie_rule_threshold(S3[x], 1 - 1.0 / T, p["d1"], p["K"], p["mu"]) for x in ALPHAS]
        ax.plot(ALPHAS, thr, color=col, lw=1.3, marker=mk, ms=3.2, markeredgecolor="white", markeredgewidth=0.5,
                label="T = %d months" % T)
    ax.set_ylabel("Required benefit-to-cost ratio, b/c")
    ax.text(0.03, 0.97, "b/c > (1 − D)(1 + max(1/w, K/c)/T)", transform=ax.transAxes, fontsize=5.6, va="top",
            bbox=dict(facecolor="white", edgecolor=LIGHT, lw=0.5, boxstyle="round,pad=0.3"))
    ax.legend(frameon=False, loc="lower right")
    ax.set_ylim(0, 1.75)
    ax = axs[1]
    for col, c, lab in [("typ_pflege_dominant", GREEN, "always pays"),
                        ("typ_hirschjagd", BLUE, "stag hunt"),
                        ("typ_aufgeben", PINK, "never pays")]:
        curve(ax, B, col, "hoch", c, "o", lab)
    ax.set_ylabel("Share of existing ties")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="center right", bbox_to_anchor=(1.03, 0.47), fontsize=5.8, handlelength=1.6,
              title="investing in a tie", title_fontsize=5.8)
    ax = axs[2]
    curve(ax, B, "beziehungen", "hoch", INK, "o", "simulation")
    Dv = np.array([S3[x] for x in ALPHAS])
    for trust, ls, lab in [(1.0, "--", "mean field, trusted"), (0.5, ":", "mean field, uncertain")]:
        k = zk.k_star(p["mu"] * (1 - Dv), p["p"], p["B"], p["k0"], p["beta"], p["d1"], p["d0"], trust)
        ax.plot(ALPHAS, k, ls=ls, color=ORANGE, lw=1.2, label=lab)
    ax.set_ylabel("Maintained ties per person")
    ax.legend(frameon=False, loc="upper right", fontsize=5.6)
    ax = axs[3]
    V = pd.read_csv(os.path.join(HERE, "results_S5", "S5_versaeumnis.csv"))
    V = V[V["perioden"] == 24].groupby(["arm", "seed", "alpha"])["ueberlebt"].mean().reset_index()
    for arm, c, mk, lab in [("Kontrolle", INK, "o", "no lapse"), ("Versaeumnis", ORANGE, "s", "one partner skips\none period")]:
        H = V[V["arm"] == arm].groupby("alpha")["ueberlebt"]
        x = H.mean().index.values
        ax.fill_between(x, H.mean() - H.std(), H.mean() + H.std(), color=c, alpha=0.15, lw=0)
        ax.plot(x, H.mean().values, color=c, lw=1.3, marker=mk, ms=3.2, markeredgecolor="white",
                markeredgewidth=0.5, label=lab)
    ax.set_ylabel("Relationship survives 24 months")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="lower left", fontsize=5.6)
    for a, L in zip(axs, "abcd"):
        style(a)
        xalpha(a)
        panel(a, L, x=-0.28)
    fig.tight_layout(w_pad=1.0)
    save(fig, "Fig2_rule_and_stag_hunt")


def snapshots():
    """Netze bei alpha = 0, 3, 4 (Wissenschaft, Seed 1) mit festen Positionen."""
    res = nm.simulate_alpha_grid("Wissenschaft", 1, ALPHAS)
    A0 = res[0].A.copy()
    Dtab = pd.read_csv(os.path.join(RES, "results_S3", "S3_abhaengigkeit.csv")).set_index("alpha")["D_wissenschaft"]
    m = zk.Future(A0, 1)
    snaps = {}
    for a in ALPHAS[:5]:
        for _ in range(60):
            m.step(Dtab[a])
        if a in (0, 3, 4):
            snaps[a] = m.A.copy()
    G0 = nx.from_numpy_array(A0.astype(int))
    core = G0.subgraph(largest_component(A0).tolist())
    pos = nx.kamada_kawai_layout(core)
    P = np.array(list(pos.values()))
    P = (P - P.mean(0)) / np.abs(P - P.mean(0)).max()
    pos = dict(zip(pos.keys(), P))
    rng = np.random.default_rng(0)
    for n in G0.nodes():                     # Isolierte am Rand verteilen
        if n not in pos:
            t = rng.uniform(0, 2 * np.pi)
            pos[n] = np.array([np.cos(t), np.sin(t)]) * 1.04
    return snaps, pos


def fig3():
    R = pd.read_csv(os.path.join(RES, "results_S3", "S3_varianten.csv"))
    B = R[R["variante"] == R["variante"].unique()[0]]
    RM = pd.read_csv(os.path.join(RES, "results_S4", "S4_phasendiagramm.csv"))
    f200 = os.path.join(HERE, "results_S5", "S5_lhs200.csv")
    LH = pd.read_csv(f200) if os.path.exists(f200) else pd.read_csv(os.path.join(RES, "results_S4", "S4_lhs.csv"))
    D = pd.read_csv(os.path.join(RES, "results_S3", "S3_abhaengigkeit.csv")).set_index("alpha")["D_wissenschaft"]
    snaps, pos = snapshots()
    fig = plt.figure(figsize=(W2, 155 * MM))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.85, 0.78, 1.1], hspace=0.42, wspace=0.42,
                          left=0.08, right=0.97, top=0.97, bottom=0.06)
    ax = fig.add_subplot(gs[0, 0])
    curve(ax, B, "beziehungen", "hoch", BLUE, "o")
    ax.set_ylabel("Ties per person")
    ax.set_ylim(0, 4.2)
    style(ax)
    xalpha(ax)
    panel(ax, "a", x=-0.16)
    ax.annotate("−6%", xy=(3.5, 1.5), xytext=(6, 2.6), fontsize=6.2, color=BLUE,
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.5))
    ax = fig.add_subplot(gs[0, 1])
    curve(ax, B, "lcc", "hoch", ORANGE, "o")
    ax.set_ylabel("Cohesion")
    ax.set_ylim(0, 1.02)
    ax.axvspan(3, 4, color=ORANGE, alpha=0.08, lw=0)
    ax.annotate("58% → 16%", xy=(3.5, 0.37), xytext=(6.5, 0.62), fontsize=6.2, color=ORANGE,
                arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.5))
    ax.text(0.98, 0.98, "share of people in the\nlargest connected component", transform=ax.transAxes,
            ha="right", va="top", fontsize=5.8, color="#55554f")
    style(ax)
    xalpha(ax)
    panel(ax, "b", x=-0.16)
    gsc = gs[1, :].subgridspec(1, 3, wspace=0.05)
    for k, a in enumerate((0, 3, 4)):
        axn = fig.add_subplot(gsc[0, k])
        A = snaps[a]
        lcc = set(largest_component(A).tolist())
        G = nx.from_numpy_array(A.astype(int))
        cols = [ORANGE if n in lcc else LIGHT for n in G.nodes()]
        nx.draw_networkx_edges(G, pos, ax=axn, width=0.35, edge_color="#8f8e87", alpha=0.75)
        nx.draw_networkx_nodes(G, pos, ax=axn, node_size=5, node_color=cols, linewidths=0)
        axn.set_xlim(-1.1, 1.1)
        axn.set_ylim(-1.02, 1.02)
        axn.set_axis_off()
        lab = {0: "α = 0: before AI", 3: "α = 3: just before collapse", 4: "α = 4: just after collapse"}[a]
        axn.set_title("%s\n%.1f ties per person, cohesion %d%%" % (lab, A.sum(1).mean(), round(100 * len(lcc) / A.shape[0])),
                      fontsize=6.3)
        if k == 0:
            panel(axn, "c", x=-0.16, y=1.12)
    ax = fig.add_subplot(gs[2, 0])
    H = RM.groupby(["mu", "alpha"])["lcc"].mean().unstack("alpha")
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "b", ["#f4f8fd", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])
    im = ax.pcolormesh(np.arange(len(ALPHAS) + 1) - 0.5, np.arange(len(H.index) + 1) - 0.5, H.values,
                       cmap=cmap, vmin=0, vmax=1, shading="flat")
    xs, ys = [], []
    for k, mu in enumerate(H.index):
        hit = [i for i, a in enumerate(ALPHAS) if zk.collapse_predicted(mu, D[a])]
        xs.append(hit[0] - 0.5 if hit else len(ALPHAS) - 0.5)
        ys.append(k)
    ax.step(xs, ys, where="mid", color=ORANGE, lw=1.8, label="rule: third tie no longer pays")
    ax.set_xticks(range(len(ALPHAS)))
    ax.set_xticklabels([str(a) for a in ALPHAS])
    ax.set_yticks(range(len(H.index)))
    ax.set_yticklabels(["%.2f" % (m / zk.DEFAULTS["mu"]) for m in H.index])
    ax.set_xlabel("AI capability, α")
    ax.set_ylabel("Maintenance cost (relative to default)")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("Cohesion (simulation)")
    cb.outline.set_linewidth(0.4)
    ax.legend(loc="lower left", frameon=True, framealpha=0.9, edgecolor="none")
    panel(ax, "d", x=-0.2)
    ax = fig.add_subplot(gs[2, 1])
    ok = LH["lcc_alpha0"] > 0.5
    L = LH[ok]
    idx = {a: i for i, a in enumerate(ALPHAS)}
    conv = lambda v: len(ALPHAS) if not np.isfinite(v) else idx[int(v)]
    xs = np.array([conv(v) for v in L["alpha_kollaps_regel"]], dtype=float)
    ys = np.array([conv(v) for v in L["alpha_kollaps_sim"]], dtype=float)
    cnt = pd.Series(list(zip(xs, ys))).value_counts()
    for (x, y), n in cnt.items():
        ax.scatter(x, y, s=10 + 9 * n, color=BLUE if x == y else (YELLOW if abs(x - y) <= 1 else GREY),
                   edgecolor="white", lw=0.5, alpha=0.9, zorder=3)
    for c_, t_ in [(BLUE, "exact"), (YELLOW, "one step off"), (GREY, "further off")]:
        ax.scatter([], [], s=18, color=c_, label=t_)
    ax.plot([-0.5, len(ALPHAS) + 0.5], [-0.5, len(ALPHAS) + 0.5], color=LIGHT, lw=0.7, ls=":", zorder=1)
    lab = [str(a) for a in ALPHAS] + ["none"]
    ax.set_xticks(range(len(lab)))
    ax.set_xticklabels(lab)
    ax.set_yticks(range(len(lab)))
    ax.set_yticklabels(lab)
    ax.set_xlabel("Collapse predicted at α (rule)")
    ax.set_ylabel("Collapse simulated at α")
    exact, near = np.mean(xs == ys), np.mean(np.abs(xs - ys) <= 1)
    ax.text(0.03, 0.97, "%d random parameter sets\nexact %.0f%%, within one step %.0f%%\n(marker area: number of sets)"
            % (len(L), 100 * exact, 100 * near), transform=ax.transAxes, va="top", fontsize=6.0)
    ax.legend(frameon=False, loc="lower right")
    style(ax, grid=False)
    panel(ax, "e", x=-0.16)
    save(fig, "Fig3_collapse_third_tie")


def fig4():
    RK = pd.read_csv(os.path.join(RES, "results_S3", "S3_hysterese_K.csv"))
    mu = zk.DEFAULTS["mu"]
    fig, axs = plt.subplots(1, 4, figsize=(W2, 52 * MM), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.05]))
    for ax, K, L in zip(axs[:3], [0.0, 0.3, 1.0], "abc"):
        H = RK[np.isclose(RK["K"], K)]
        curve(ax, H, "lcc", "hoch", BLUE, "o", "AI grows stronger")
        curve(ax, H, "lcc", "runter", ORANGE, "s", "AI weakens again", ls="--")
        up = H[H["phase"] == "hoch"].groupby("alpha")["lcc"].mean()
        dn = H[H["phase"] == "runter"].groupby("alpha")["lcc"].mean()
        x = [a for a in ALPHAS if a in dn.index]
        ax.fill_between(x, dn[x].values, up[x].values, where=up[x].values > dn[x].values, color=GREY, alpha=0.18, lw=0)
        ax.set_title("K = %.0f × maintenance cost" % (K / mu), loc="center", fontsize=6.5)
        ax.set_ylim(0, 1.05)
        style(ax)
        xalpha(ax)
        panel(ax, L, x=-0.24, y=1.1)
        if L != "a":
            ax.set_yticklabels([])
    axs[0].set_ylabel("Cohesion (largest component)")
    axs[1].legend(frameon=False, loc="upper right")
    ax = axs[3]
    Ks = sorted(RK["K"].unique())
    rec = []
    for K in Ks:
        H = RK[np.isclose(RK["K"], K)]
        up = H[H["phase"] == "hoch"].groupby(["seed", "alpha"])["lcc"].mean().unstack()
        dn = H[H["phase"] == "runter"].groupby(["seed", "alpha"])["lcc"].mean().unstack()
        cols = [a for a in ALPHAS if a in dn.columns]
        rec.append((up[cols] - dn[cols]).clip(lower=0).sum(axis=1).values)
    m = np.array([r.mean() for r in rec])
    sd = np.array([r.std() for r in rec])
    xk = np.array(Ks) / mu
    ax.fill_between(xk, m - sd, m + sd, color=INK, alpha=0.12, lw=0)
    ax.plot(xk, m, color=INK, lw=1.3, marker="o", ms=3.2, markeredgecolor="white", markeredgewidth=0.5)
    ax.set_xlabel("Rebuilding cost, K / c (months)")
    ax.set_ylabel("Hysteresis (cohesion lost\non the way back)")
    style(ax)
    panel(ax, "d", x=-0.3, y=1.1)
    fig.tight_layout(w_pad=1.2)
    fig.subplots_adjust(right=0.98)
    save(fig, "Fig4_hysteresis")


def fig5():
    RH = pd.read_csv(os.path.join(RES, "results_S4", "S4_abstimmung.csv"))
    R = pd.read_csv(os.path.join(RES, "results_S3", "S3_varianten.csv"))
    RS = pd.read_csv(os.path.join(RES, "results_S3", "S3_subvention.csv"))
    fig, axs = plt.subplots(1, 3, figsize=(W2, 86 * MM))
    ax = axs[0]
    labs = RH["variante"].unique()
    names = ["everyone decides alone", "binding agreement", "short-sighted minority (30%)", "minority + agreement"]
    for lab, nm_, c, mk in zip(labs, names, [INK, BLUE, ORANGE, GREEN], ["o", "s", "^", "D"]):
        curve(ax, RH[RH["variante"] == lab], "lcc", "hoch", c, mk, nm_)
    ax.set_ylabel("Cohesion (largest component)")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.24))
    ax = axs[1]
    labs = list(R["variante"].unique())
    names = {labs[0]: "everyone decides alone", labs[1]: "value to third parties included",
             labs[2]: "institutions carry 30% of cost", labs[3]: "cheaper rebuilding (K/4)", labs[4]: "30% cost + cheaper rebuilding"}
    for lab, c, mk in zip(labs, [INK, BLUE, ORANGE, GREEN, YELLOW], ["o", "s", "^", "D", "v"]):
        curve(ax, R[R["variante"] == lab], "lcc", "hoch", c, mk, names[lab])
    ax.set_ylabel("Cohesion (largest component)")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.24))
    for a in axs[:2]:
        xalpha(a)
    ax = axs[2]
    for k, (a, c, mk) in enumerate([(4, BLUE, "o"), (8, ORANGE, "s"), (16, GREEN, "^")]):
        H = RS[(RS["alpha"] == a) & (RS["phase"] == "hoch")].groupby("subsidy")["wohlfahrt"]
        x = H.mean().index.values * 100
        ax.fill_between(x, H.mean() - H.std().fillna(0), H.mean() + H.std().fillna(0), color=c, alpha=0.15, lw=0)
        ax.plot(x, H.mean().values, color=c, lw=1.3, marker=mk, ms=3.2, markeredgecolor="white",
                markeredgewidth=0.5, label="α = %d" % a)
    ax.set_xlabel("Cost share carried by institutions (%)")
    ax.set_ylabel("Welfare per person and period\n(full costs included)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3)
    for a, L in zip(axs, "abc"):
        style(a)
        panel(a, L)
    fig.tight_layout(w_pad=2.0)
    fig.subplots_adjust(bottom=0.37)
    save(fig, "Fig5_institutions")


def figS():
    R3 = pd.read_csv(os.path.join(RES, "results_S4", "S4_drei_schritte.csv"))
    import schritt4_regel as s4
    f200 = os.path.join(HERE, "results_S5", "S5_lhs200.csv")
    LHx = pd.read_csv(f200)
    okx = LHx["lcc_alpha0"] > 0.5
    keys = [k for k in s4.LHS_RANGES if k != "d1_ratio"] + ["d1"]
    yv = LHx.loc[okx, "alpha_kollaps_sim"].replace(np.inf, 32).values
    P = pd.DataFrame({"parameter": keys, "prcc_kollaps": s4.prcc(LHx.loc[okx, keys].values, yv)})
    nm2 = dict(mu="Pflegekosten c", p="Häufigkeit Ernstfall", k0="Sättigung k0", eta="Gewicht 2 Schritte η",
               beta="Zeithorizont (β)", d0="Zerfall ohne Pflege d0", d1="Zerfall einseitig d1 (w)",
               K="Wiederaufbaukosten K", eps="Versehen ε", memory="Gedächtnis Vertrauen")
    P["parameter"] = P["parameter"].map(nm2)
    P.to_csv(os.path.join(HERE, "results_S5", "S5_prcc200.csv"), index=False)
    en = {"Pflegekosten c": "maintenance cost c", "Häufigkeit Ernstfall": "frequency of need p",
          "Zeithorizont (β)": "time horizon (β)", "Sättigung k0": "saturation k₀", "Zerfall ohne Pflege d0": "decay, no maintenance d₀",
          "Zerfall einseitig d1 (w)": "decay, one-sided w", "Versehen ε": "lapses ε", "Wiederaufbaukosten K": "rebuilding cost K",
          "Gedächtnis Vertrauen": "trust memory", "Gewicht 2 Schritte η": "weight of 2-step contacts η"}
    fig, axs = plt.subplots(1, 2, figsize=(W2 * 0.8, 55 * MM))
    ax = axs[0]
    for lab, nm_, c, mk in zip(R3["variante"].unique(), ["two steps (default)", "three steps (η₃ = 0.1)"], [INK, BLUE], ["o", "s"]):
        curve(ax, R3[R3["variante"] == lab], "lcc", "hoch", c, mk, nm_)
    ax.set_ylabel("Cohesion (largest component)")
    ax.legend(frameon=False)
    xalpha(ax)
    ax = axs[1]
    P = P.sort_values("prcc_kollaps", ascending=False).reset_index(drop=True)
    ax.barh(range(len(P)), P["prcc_kollaps"], color=[BLUE if v > 0 else ORANGE for v in P["prcc_kollaps"]], height=0.6)
    ax.set_yticks(range(len(P)))
    ax.set_yticklabels([en.get(x, x) for x in P["parameter"]])
    ax.invert_yaxis()
    ax.axvline(0, color=INK, lw=0.6)
    ax.set_xlabel("Partial rank correlation with collapse α\n(positive: collapse only at stronger AI)")
    for k, v in enumerate(P["prcc_kollaps"]):
        ax.text(v + (0.02 if v >= 0 else -0.02), k, "%+.2f" % v, va="center", ha="left" if v >= 0 else "right", fontsize=5.6)
    ax.set_xlim(-1, 1)
    for a, L in zip(axs, "ab"):
        style(a, grid=(L == "a"))
        panel(a, L, x=-0.2 if L == "a" else -0.6)
    fig.tight_layout(w_pad=2.0)
    save(fig, "FigS1_robustness")


def figS2():
    R6 = os.path.join(HERE, "results_S6")
    A = pd.read_csv(os.path.join(R6, "S6A_verlauf.csv"))
    A2 = pd.read_csv(os.path.join(R6, "S6A2_zufallsvergleich.csv"))
    B = pd.read_csv(os.path.join(R6, "S6B_heterogen.csv"))
    B2 = pd.read_csv(os.path.join(R6, "S6B2_kostenstreuung.csv"))
    C = pd.read_csv(os.path.join(R6, "S6C_gegenmechanismus.csv"))
    L = pd.read_csv(os.path.join(R6, "S6D_lange_laeufe.csv"))
    fig, axs = plt.subplots(2, 3, figsize=(W2, 108 * MM))
    x = np.arange(len(ALPHAS))

    def line(ax, g, col, color, label, marker="o", ls="-"):
        H = g.groupby("alpha")[col].agg(["mean", "std"]).reindex(ALPHAS)
        ax.fill_between(x, H["mean"] - H["std"].fillna(0), H["mean"] + H["std"].fillna(0), color=color, alpha=0.15, lw=0)
        ax.plot(x, H["mean"], color=color, marker=marker, ms=3, lw=1.2, ls=ls, label=label,
                markeredgecolor="white", markeredgewidth=0.4)

    def ax_alpha(ax):
        ax.set_xticks(x)
        ax.set_xticklabels([str(a) for a in ALPHAS])
        ax.set_xlabel("AI capability α")

    # a: Molloy-Reed ratio on the simulated networks
    ax = axs[0, 0]
    line(ax, A, "mr", INK, "simulated networks")
    ax.axhline(2, color=ORANGE, lw=0.9, ls="--")
    ax.text(8.4, 2.12, r"$\langle k^2\rangle/\langle k\rangle = 2$", color=ORANGE, fontsize=5.8, ha="right")
    ax.set_ylabel(r"Molloy–Reed ratio $\langle k^2\rangle/\langle k\rangle$")
    ax.set_ylim(1, 5)
    ax_alpha(ax)
    # b: model loss vs random loss of the same edges
    ax = axs[0, 1]
    cats = [("vorher", "before\n(α = 3)"), ("zufall", "random loss,\nsame number"), ("modell", "model\n(α = 4)")]
    for k, (t, lab) in enumerate(cats):
        g = A2[A2.typ == t].groupby("seed")["deg3"].mean()
        ax.bar(k, g.mean(), width=0.6, color=[LIGHT, GREY, ORANGE][k], edgecolor="white")
        ax.scatter(np.full(len(g), k) + np.linspace(-0.12, 0.12, len(g)), g.values, s=7, color=INK, zorder=3, lw=0)
    ax.set_xticks(range(3))
    ax.set_xticklabels([c[1] for c in cats], fontsize=6)
    ax.set_ylabel("People with three or more ties")
    ax.set_yticks([0, 0.05, 0.10, 0.15])
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    # c: start networks
    ax = axs[0, 2]
    for lab, c, mk, nm_ in [("Modellnetz", INK, "o", "formation model"), ("Erdos-Renyi", BLUE, "s", "Erdős–Rényi"),
                            ("Barabasi-Albert", GREEN, "^", "Barabási–Albert"), ("bimodal 2/8", PINK, "D", "bimodal (2 / 8 ties)")]:
        line(ax, B[B.variante == lab], "lcc", c, nm_, marker=mk)
    ax.set_ylabel("Cohesion (largest component)")
    ax.legend(frameon=False, loc="upper right", fontsize=5.6)
    ax_alpha(ax)
    # d: heterogeneous maintenance costs (median-preserving)
    ax = axs[1, 0]
    line(ax, B[B.variante == "Modellnetz"], "lcc", INK, "equal costs")
    for sd, c, mk in [(0.25, YELLOW, "s"), (0.5, ORANGE, "^"), (1.0, PINK, "D")]:
        line(ax, B2[(B2.sd == sd) & (B2.median1)], "lcc", c, "spread σ = %g" % sd, marker=mk)
    ax.set_ylabel("Cohesion (largest component)")
    ax.legend(frameon=False, loc="upper right", fontsize=5.6)
    ax_alpha(ax)
    # e: counter-mechanisms
    ax = axs[1, 1]
    for (v, st), c, mk, ls, nm_ in [(("neue gemeinsame Arbeit", 0.0), INK, "o", "-", "none"),
                                    (("neue gemeinsame Arbeit", 0.25), BLUE, "s", "-", "25% replaced"),
                                    (("neue gemeinsame Arbeit", 0.5), GREEN, "^", "-", "50% replaced"),
                                    (("Zeitgewinn -> neue Kontakte", 0.2), ORANGE, "D", "--", "time saved →\nnew contacts")]:
        line(ax, C[(C.variante == v) & (C.staerke == st)], "lcc", c, nm_, marker=mk, ls=ls)
    ax.set_ylabel("Cohesion (largest component)")
    leg = ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.04), fontsize=5.4,
                    title="new joint work", title_fontsize=5.6)
    leg._legend_box.align = "left"     # works with matplotlib < 3.6
    ax.set_ylim(0, 1)
    ax_alpha(ax)
    # f: long runs after collapse
    ax = axs[1, 2]
    for (K, rr), c, mk, ls, nm_ in [((0.3, 0.0), BLUE, "o", "-", "K = 20 months"),
                                    ((0.3, 0.05), BLUE, "o", "--", "K = 20 months, random contacts"),
                                    ((0.6, 0.0), ORANGE, "s", "-", "K = 40 months"),
                                    ((0.6, 0.05), ORANGE, "s", "--", "K = 40 months, random contacts")]:
        g = L[(L.K == K) & (L.r_rand == rr)].groupby("perioden")["lcc"].mean()
        ax.plot(g.index / 12.0, g.values, color=c, marker=mk, ms=3, ls=ls, lw=1.2, label=nm_,
                markeredgecolor="white", markeredgewidth=0.4)
    ax.set_xscale("log")
    ax.set_xticks([5, 10, 20, 50, 100])
    ax.set_xticklabels(["5", "10", "20", "50", "100"])
    ax.set_xlabel("Years at α = 1 after collapse")
    ax.set_ylabel("Cohesion (largest component)")
    ax.set_ylim(-0.02, 1)
    ax.legend(frameon=False, loc="center right", fontsize=5.4)
    for a, l_ in zip(axs.ravel(), "abcdef"):
        style(a, grid=False)
        panel(a, l_, x=-0.24)
    fig.tight_layout(w_pad=1.6, h_pad=1.8)
    save(fig, "FigS2_review_robustness")


def figS3():
    R7 = os.path.join(HERE, "results_S7")
    E = pd.read_csv(os.path.join(R7, "S7E_umverdrahtung.csv"))
    F = pd.read_csv(os.path.join(R7, "S7F_zuordnung.csv"))
    G = pd.read_csv(os.path.join(R7, "S7G_groesse.csv"))
    fig, axs = plt.subplots(1, 3, figsize=(W2, 58 * MM))
    x = np.arange(len(ALPHAS))
    # a: model vs degree-preserving rewiring
    ax = axs[0]
    for col, c, mk, lab in [("lcc_modell", INK, "o", "simulated networks"),
                            ("lcc_umverdrahtet", ORANGE, "s", "rewired, same degrees")]:
        H = E.groupby("alpha")[col].agg(["mean", "std"]).reindex(ALPHAS)
        ax.fill_between(x, H["mean"] - H["std"], H["mean"] + H["std"], color=c, alpha=0.15, lw=0)
        ax.plot(x, H["mean"], color=c, marker=mk, ms=3, lw=1.2, label=lab,
                markeredgecolor="white", markeredgewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([str(a) for a in ALPHAS])
    ax.set_xlabel("AI capability α")
    ax.set_ylabel("Cohesion (largest component)")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="upper right", fontsize=5.6)
    # b: cohesion against effective by-product share, for different mappings
    ax = axs[1]
    ax.axvspan(0.2, 0.25, color=LIGHT, alpha=0.35, lw=0)
    for th, c, mk in [(0.25, GREEN, "^"), (0.5, BLUE, "s"), (1.0, INK, "o"), (2.0, PINK, "D")]:
        g = F[F.theta == th].groupby("alpha")[["D_eff", "lcc"]].mean().reindex(ALPHAS)
        ax.plot(g["D_eff"], g["lcc"], color=c, marker=mk, ms=3, lw=1.2, label="θ = %g" % th,
                markeredgecolor="white", markeredgewidth=0.4)
    ax.set_xlim(0.9, 0.0)
    ax.set_xlabel("Effective by-product share $D_{\\mathrm{eff}}$")
    ax.set_ylabel("Cohesion (largest component)")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="lower left", fontsize=5.6, title="mapping", title_fontsize=5.8)
    # c: replicates and network size on a fine grid
    ax = axs[2]
    g3 = G[G.N == 300]
    for s, gs in g3.groupby("seed"):
        ax.plot(gs["alpha"], gs["lcc"], color=LIGHT, lw=0.5, alpha=0.8, zorder=1)
    for N, c, mk in [(300, INK, "o"), (1000, BLUE, "s"), (2000, ORANGE, "D")]:
        gN = G[G.N == N]
        if len(gN) == 0:
            continue
        H = gN.groupby("alpha")["lcc"].mean()
        ax.plot(H.index, H.values, color=c, marker=mk, ms=3, lw=1.2, zorder=3,
                label="N = %d (%d runs)" % (N, gN.seed.nunique()),
                markeredgecolor="white", markeredgewidth=0.4)
    ax.set_xlabel("AI capability α")
    ax.set_ylabel("Cohesion (largest component)")
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.2, 8.2)
    ax.legend(frameon=False, loc="upper right", fontsize=5.6)
    for a, l_ in zip(axs, "abc"):
        style(a, grid=False)
        panel(a, l_, x=-0.24)
    fig.tight_layout(w_pad=1.6)
    save(fig, "FigS3_degree_mapping_size")


def main():
    os.makedirs(OUT, exist_ok=True)
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    figS()
    figS2()
    figS3()
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
