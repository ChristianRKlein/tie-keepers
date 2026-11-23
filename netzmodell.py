# -*- coding: utf-8 -*-
"""
netzmodell.py  --  Strategische Netzwerkbildung mit KI (kompakter Nachbau)
=========================================================================

Eigenstaendiger Nachbau des Netzwerkmodells aus dem Nature-Communications-
Manuskript ("AI can substitute for collaborators and reshape networks in
science and the economy"), Abschnitt Methoden:

  phi_i      = phi_max * (1 - exp(-alpha * u_i))           Substitution
  o_ik       = 1 - (1 - s_ik)(1 - phi_i)
  c_ik       = 1 - (1 - o_ik) * prod_{j in N(i)} (1 - q s_jk)
  Y_i        = a_i (1 + lam*alpha*u_i*a_i^eta) * exp( mean_k log c_ik )
  C_i(d)     = c0/(1 + kappa*alpha*u_i) * b_i^omega * (d + gamma d(d-1)/2)

Dynamik: paarweise stabil, gegenseitige Zustimmung (oder Ausgleichszahlungen),
Pruefwahrscheinlichkeit 0.5, Freunde-von-Freunden 0.7, Zerfall 0.003,
Watts-Strogatz-Start (mittlerer Grad 6), 60 Perioden ohne KI + 80 mit KI.

Laeuft mit Python >= 3.7, NumPy >= 1.17 (keine weiteren Abhaengigkeiten).
"""
import numpy as np

# ---------------------------------------------------------------------------
# Parameter (Tabelle im Methodenteil des Manuskripts)
# ---------------------------------------------------------------------------
DOMAINS = {
    "Wissenschaft": dict(K=6, n=2, s_hi=0.90, s_lo=0.05, q=0.60, c0=0.020,
                         gamma=0.10, phi_max=0.95, lam=0.5, sigma_a=0.6),
    "Wirtschaft":   dict(K=8, n=2, s_hi=0.90, s_lo=0.05, q=0.60, c0=0.025,
                         gamma=0.08, phi_max=0.90, lam=0.8, sigma_a=0.8),
    "Sozial":       dict(K=4, n=1, s_hi=0.60, s_lo=0.02, q=0.50, c0=0.015,
                         gamma=0.15, phi_max=0.70, lam=0.0, sigma_a=0.3),
}

COMMON = dict(
    N=300,            # Akteure
    eta=0.0,          # Komplementaritaet KI x Ideen (Basis: 0)
    kappa=0.0,        # Koordinationsgewinn (Basis: 0)
    omega=0.0,        # Kosten ~ Produktivitaet^omega (Basis: 0)
    rho=0.3,          # Rangkorrelation Ideen ~ KI-Nutzung
    u_mean=0.5, u_sigma=1.0, u_cap=10.0,
    transfers=False,  # False = gegenseitige Zustimmung (Basiskonfiguration)
    p_review=0.5, p_fof=0.7, decay=0.003,
    k0=6, ws_beta=0.1,
    T_burn=60, T_ai=80,
)


def params(domain, **overrides):
    p = dict(COMMON)
    p.update(DOMAINS[domain])
    p.update(overrides)
    return p


# ---------------------------------------------------------------------------
# Akteure und Faehigkeiten
# ---------------------------------------------------------------------------
def draw_agents(p, rng):
    N = p["N"]
    z1 = rng.standard_normal(N)
    z2 = p["rho"] * z1 + np.sqrt(1.0 - p["rho"] ** 2) * rng.standard_normal(N)
    a = np.exp(p["sigma_a"] * z1)
    a = a / a.mean()
    u = np.exp(p["u_sigma"] * z2)
    u = u * (p["u_mean"] / u.mean())
    u = np.minimum(u, p["u_cap"])
    K, n = p["K"], p["n"]
    S = np.full((N, K), p["s_lo"])
    for i in range(N):
        S[i, rng.choice(K, n, replace=False)] = p["s_hi"]
    return a, u, S


def watts_strogatz(N, k, beta, rng):
    A = np.zeros((N, N), dtype=bool)
    h = k // 2
    for i in range(N):
        for s in range(1, h + 1):
            j = (i + s) % N
            A[i, j] = A[j, i] = True
    for i in range(N):
        for s in range(1, h + 1):
            j = (i + s) % N
            if A[i, j] and rng.random() < beta:
                cand = np.flatnonzero(~A[i])
                cand = cand[cand != i]
                if len(cand) == 0:
                    continue
                m = rng.choice(cand)
                A[i, j] = A[j, i] = False
                A[i, m] = A[m, i] = True
    return A


# ---------------------------------------------------------------------------
# Das Modell als Objekt: Zustand = Netzwerk A und M = A @ log(1 - qS)
# ---------------------------------------------------------------------------
class Model(object):

    def __init__(self, domain, seed, **overrides):
        self.domain = domain
        self.p = params(domain, **overrides)
        self.seed = seed
        rng = np.random.default_rng(seed)
        self.a, self.u, self.S = draw_agents(self.p, rng)
        self.Lp = np.log(1.0 - self.p["q"] * self.S)     # Beitrag eines Partners (log)
        self.A = watts_strogatz(self.p["N"], self.p["k0"], self.p["ws_beta"], rng)
        self.M = self.A.astype(float) @ self.Lp
        self.set_alpha(0.0)

    # --- KI-Staerke setzen --------------------------------------------------
    def set_alpha(self, alpha):
        p = self.p
        self.alpha = float(alpha)
        phi = p["phi_max"] * (1.0 - np.exp(-alpha * self.u))
        self.phi = phi
        self.U0 = (1.0 - self.S) * (1.0 - phi)[:, None]   # Restluecke nach eigener Faehigkeit + KI
        self.bfac = self.a * (1.0 + p["lam"] * alpha * self.u * self.a ** p["eta"])
        self.cc = p["c0"] / (1.0 + p["kappa"] * alpha * self.u) * self.bfac ** p["omega"]

    # --- Output ----------------------------------------------------------------
    def Y_of(self, idx, Mrows):
        """Output der Akteure idx bei (hypothetischen) Partnersummen Mrows."""
        g = np.log(1.0 - self.U0[idx] * np.exp(Mrows))
        return self.bfac[idx] * np.exp(g.mean(axis=-1))

    def Y_all(self):
        return self.Y_of(np.arange(self.p["N"]), self.M)

    def Y_alone(self):
        N = self.p["N"]
        return self.Y_of(np.arange(N), np.zeros((N, self.p["K"])))

    def mc(self, i, d):
        """Grenzkosten der d-ten Verbindung von i."""
        return self.cc[i] * (1.0 + self.p["gamma"] * (d - 1))

    # --- Verbindungen aendern -------------------------------------------------
    def _add(self, i, j):
        self.A[i, j] = self.A[j, i] = True
        self.M[i] += self.Lp[j]
        self.M[j] += self.Lp[i]

    def _remove(self, i, j):
        self.A[i, j] = self.A[j, i] = False
        self.M[i] -= self.Lp[j]
        self.M[j] -= self.Lp[i]

    # --- eine Periode -----------------------------------------------------------
    def step(self, rng):
        p = self.p
        N = p["N"]
        deg = self.A.sum(1)
        order = rng.permutation(N)
        reviewers = order[rng.random(N) < p["p_review"]]
        for i in reviewers:
            nb = np.flatnonzero(self.A[i])
            # 1) schwaechste Verbindung loesen?
            if len(nb) > 0:
                Yi = self.Y_of(i, self.M[i])
                loss = Yi - self.Y_of(i, self.M[i][None, :] - self.Lp[nb])
                k = int(np.argmin(loss))
                j = nb[k]
                di, dj = deg[i], deg[j]
                if p["transfers"]:
                    Yj = self.Y_of(j, self.M[j])
                    loss_j = Yj - self.Y_of(j, self.M[j] - self.Lp[i])
                    cut = loss[k] + loss_j < self.mc(i, di) + self.mc(j, dj)
                else:
                    cut = loss[k] < self.mc(i, di)
                if cut:
                    self._remove(i, j)
                    deg[i] -= 1
                    deg[j] -= 1
                    nb = np.delete(nb, k)
            # 2) neue Verbindung vorschlagen
            j = -1
            if len(nb) > 0 and rng.random() < p["p_fof"]:
                m = nb[rng.integers(len(nb))]
                nb2 = np.flatnonzero(self.A[m])
                if len(nb2) > 0:
                    j = nb2[rng.integers(len(nb2))]
            if j < 0:
                j = rng.integers(N)
            if j == i or self.A[i, j]:
                continue
            gi = self.Y_of(i, self.M[i] + self.Lp[j]) - self.Y_of(i, self.M[i])
            gj = self.Y_of(j, self.M[j] + self.Lp[i]) - self.Y_of(j, self.M[j])
            ci, cj = self.mc(i, deg[i] + 1), self.mc(j, deg[j] + 1)
            ok = (gi + gj > ci + cj) if p["transfers"] else (gi > ci and gj > cj)
            if ok:
                self._add(i, j)
                deg[i] += 1
                deg[j] += 1
        # 3) exogener Zerfall
        if p["decay"] > 0:
            iu, ju = np.nonzero(np.triu(self.A, 1))
            kill = rng.random(len(iu)) < p["decay"]
            for i, j in zip(iu[kill], ju[kill]):
                self._remove(i, j)

    def run(self, T, rng):
        for _ in range(T):
            self.step(rng)

    def copy(self):
        new = Model.__new__(Model)
        new.__dict__.update(self.__dict__)
        new.A = self.A.copy()
        new.M = self.M.copy()
        return new

    # --- Spieltheoretische Groessen auf dem aktuellen Netz ----------------------
    def edge_payoffs(self):
        """Fuer jede gerichtete Kante j -> i:
             b[j, i] = Y_i - Y_i(ohne j)  (Nutzen, den j fuer i stiftet)
           und je Akteur
             c[i]    = C_i(d_i) / d_i     (durchschnittliche Kosten je Verbindung)
        """
        N = self.p["N"]
        ii, jj = np.nonzero(self.A)            # i hat Partner j
        Y = self.Y_all()
        B = np.zeros((N, N))
        if len(ii):
            Yrem = self.Y_of(ii, self.M[ii] - self.Lp[jj])
            B[jj, ii] = Y[ii] - Yrem
        d = self.A.sum(1).astype(float)
        c = self.cc * (1.0 + self.p["gamma"] * (d - 1.0) / 2.0)
        c[d == 0] = np.nan
        return B, c, Y


def simulate_alpha_grid(domain, seed, alphas, **overrides):
    """Gemeinsame Einschwingphase ohne KI, danach fuer jedes alpha eigener Lauf
    mit identischem Zufallsstrom (gleiche Akteure und Seeds ueber alle alpha)."""
    base = Model(domain, seed, **overrides)
    base.run(base.p["T_burn"], np.random.default_rng(seed + 10_000))
    out = {}
    for al in alphas:
        m = base.copy()
        m.set_alpha(al)
        m.run(m.p["T_ai"], np.random.default_rng(seed + 20_000))
        out[al] = m
    return out
