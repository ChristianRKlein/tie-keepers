# -*- coding: utf-8 -*-
"""
zukunft.py  --  Kooperation mit der Zukunft: Pflege menschlicher Netzwerke im KI-Zeitalter
==========================================================================================

Grundidee
---------
Vor der KI war die Pflege einer Beziehung ein kostenloses Nebenprodukt gemeinsamer
Arbeit. Mit KI erledigt die KI die taegliche Arbeit; eine Beziehung bleibt nur
erhalten, wenn man bewusst Zeit investiert -- fuer einen Ernstfall in der Zukunft,
in dem man Menschen wieder braucht (KI faellt aus, wird teuer, scheitert an einem
neuen Problem; Vertrauen, Haftung, Koerperliches).

Bausteine (eine Periode = z.B. ein Monat)
-----------------------------------------
Pflegekosten je Beziehung      mu_eff(alpha) = mu * (1 - D(alpha)) * (1 - Subvention)
                               D(alpha) = Abhaengigkeit von Partnern aus dem
                               NatComm-Netzwerkmodell (84 % ohne KI, 5 % bei alpha = 16):
                               der Teil der Pflege, der "nebenbei" durch Arbeit geschieht.
Ernstfall                      tritt fuer jeden Akteur mit Wahrscheinlichkeit p ein.
Wert des Netzes im Ernstfall   V_i = B * (1 - exp(-S_i / k0)),
                               S_i = (direkte Kontakte) + eta * (Kontakte der Kontakte)
Pflege braucht beide           beide pflegen  -> Beziehung bleibt
                               nur einer      -> zerfaellt mit Wahrscheinlichkeit d1
                               keiner         -> zerfaellt mit Wahrscheinlichkeit d0
Wiederaufbau                   eine verlorene Beziehung neu aufzubauen kostet
                               K_eff = K * (1 - D(alpha)); wo Arbeit Partner erfordert,
                               entstehen neue Kooperationen ebenfalls nebenbei.

Entscheidung (kurzsichtig-rational, je Beziehung und Periode):
   Wert der Beziehung (Barwert bei beidseitiger Pflege)
        Pi_ij = max(0, p * dV_i(ij) - mu_eff) / (1 - beta)
   pflegen, wenn  Pi_ij * [ q*d1 + (1-q)*(d0-d1) ] > mu_eff,
   q = erwartete Pflege durch den Partner (gleitender Mittelwert seiner Pflege,
       Gedaechtnis 'memory').
   Mit kleiner Zitterwahrscheinlichkeit eps wird die Entscheidung umgedreht.
   'Gemeinwohl'-Variante: dV_i enthaelt zusaetzlich die Haelfte des Werts, den
   die Beziehung fuer Dritte stiftet (Erreichbarkeit ueber zwei Schritte).

Wiederaufbau: jeder Akteur prueft je Periode mit Wahrscheinlichkeit r eine
Beziehung zu einem frueheren Partner oder Freund eines Freundes; sie entsteht,
wenn fuer beide  Pi - K > 0.
"""
import numpy as np


DEFAULTS = dict(
    mu=0.015,       # Pflegekosten einer Beziehung ohne Nebenprodukt-Pflege
    p=0.10,         # Wahrscheinlichkeit eines Ernstfalls je Periode
    B=1.0,          # Wert eines voll tragfaehigen Netzes im Ernstfall
    k0=4.0,         # Saettigung: ab wie vielen Kontakten das Netz 'reicht'
    eta=0.25,       # Gewicht der Kontakte von Kontakten
    beta=0.95,      # Diskontfaktor je Periode
    d0=0.30,        # Zerfall ohne Pflege
    d1=0.25,        # Zerfall bei einseitiger Pflege
    K=0.10,         # Kosten des Wiederaufbaus einer Beziehung
    r=0.10,         # Wahrscheinlichkeit, je Periode einen Wiederaufbau zu pruefen
    eps=0.01,       # Zittern in der Entscheidung
    subsidy=0.0,    # Anteil der Pflegekosten, den Institutionen tragen
    social=0.0,     # 0 = eigennuetzig, 1 = Wert fuer Dritte voll mitgedacht
    memory=0.8,     # Gedaechtnis fuer die Verlaesslichkeit des Partners
    binding=False,  # verbindliche Vereinbarung je Beziehung (Abstimmung, vgl. Hauser et al. 2014)
    frac_myopic=0.0,  # Anteil kurzsichtiger Akteure
    beta_myopic=0.8,  # deren Diskontfaktor
    eta3=0.0,       # Gewicht der Kontakte in drei Schritten (Robustheit)
    r_rand=0.0,     # Wahrscheinlichkeit je Periode, einen voellig neuen Kontakt zu pruefen (Robustheit)
    mu_sd=0.0,      # Streuung der Pflegekosten zwischen Personen (lognormal, Mittel 1)
)


def f(S, k0):
    return 1.0 - np.exp(-S / k0)


class Future(object):

    def __init__(self, A0, seed, **kw):
        self.p = dict(DEFAULTS)
        self.p.update(kw)
        self.N = A0.shape[0]
        self.A = A0.copy().astype(bool)
        self.A0 = A0.copy().astype(bool)          # fruehere Partner
        self.act = self.A.copy()                  # act[i, j]: i pflegt Beziehung zu j
        self.trust = self.A.astype(float)         # trust[i, j]: erwartete Pflege durch j
        self.rng = np.random.default_rng(seed)
        self.eye = np.eye(self.N, dtype=bool)
        self.beta_i = np.full(self.N, self.p["beta"])
        if self.p["frac_myopic"] > 0:
            m = np.random.default_rng(seed + 777).random(self.N) < self.p["frac_myopic"]
            self.beta_i[m] = self.p["beta_myopic"]
        self.mu_i = np.ones(self.N)
        if self.p["mu_sd"] > 0:
            sd = self.p["mu_sd"]
            z = np.random.default_rng(seed + 999).normal(-0.5 * sd * sd, sd, self.N)
            self.mu_i = np.exp(z)

    # --- Wert des Netzes ------------------------------------------------------
    def _nbrs(self):
        return [np.flatnonzero(self.A[i]) for i in range(self.N)]

    def _S_bfs(self, nb, i, skip=-1, add=-1):
        """S_i = n1 + eta n2 + eta3 n3 per Breitensuche (Kante i-skip entfernt / i-add ergaenzt)."""
        first = [j for j in nb[i] if j != skip]
        if add >= 0:
            first.append(add)
        seen = {i}
        seen.update(first)
        counts = [len(first), 0, 0]
        frontier = first
        for depth in (1, 2):
            nxt = []
            for u in frontier:
                for v in nb[u]:
                    if v not in seen:
                        seen.add(v)
                        nxt.append(v)
            counts[depth] = len(nxt)
            frontier = nxt
        return counts[0] + self.p["eta"] * counts[1] + self.p["eta3"] * counts[2]

    def reach(self):
        A = self.A
        Af = A.astype(float)
        C = np.rint(Af @ Af)                      # Zahl der 2-Schritt-Pfade
        two = (C > 0) & ~A & ~self.eye
        S = A.sum(1) + self.p["eta"] * two.sum(1)
        return S, C, two

    def marginal_values(self):
        """dV[i, j]: Verlust an Ernstfallwert fuer i, wenn Beziehung ij wegfiele.
           ext[i, j]: Verlust fuer alle Dritten (symmetrisch)."""
        p = self.p
        if p["eta3"] > 0:
            nb = self._nbrs()
            S = np.array([self._S_bfs(nb, i) for i in range(self.N)])
            dV = np.zeros((self.N, self.N))
            for i in range(self.N):
                for j in nb[i]:
                    dV[i, j] = p["B"] * (f(S[i], p["k0"]) - f(self._S_bfs(nb, i, skip=j), p["k0"]))
            return dV, np.zeros_like(dV), S
        A = self.A
        Af = A.astype(float)
        S, C, two = self.reach()
        eta, k0 = p["eta"], p["k0"]
        only_via = ((C == 1) & ~A & ~self.eye).astype(float)   # k nur ueber einen Pfad erreichbar
        L2 = only_via @ Af.T                                   # L2[i,j]: 2-Schritt-Knoten nur via j
        back = (C > 0).astype(float)                           # j bleibt 2-Schritt-Kontakt
        S_minus = S[:, None] - 1.0 - eta * L2 + eta * back
        dV = p["B"] * (f(S[:, None], k0) - f(S_minus, k0)) * Af
        # Wert fuer Dritte: Nachbarn k von i verlieren j als 2-Schritt-Kontakt
        g = p["B"] * (f(S, k0) - f(S - eta, k0))
        G = only_via * g[:, None]
        E1 = Af @ G                                            # E1[i,j] = sum_k A_ik G_kj
        ext = (E1 + E1.T) * Af
        return dV, ext, S

    # --- eine Periode -----------------------------------------------------------
    def step(self, D_alpha):
        p = self.p
        rng = self.rng
        mu_eff = p["mu"] * (1.0 - D_alpha) * (1.0 - p["subsidy"])
        if p["mu_sd"] > 0:
            mu_eff = mu_eff * self.mu_i[:, None]
        dV, ext, S = self.marginal_values()
        val = dV + 0.5 * p["social"] * ext
        Pi = np.maximum(0.0, p["p"] * val - mu_eff) / (1.0 - self.beta_i[:, None])
        if p["binding"]:
            # verbindliche Vereinbarung: beide pflegen, wenn der mittlere Gewinn bei
            # gesicherter Gegenseitigkeit positiv ist (Median zweier Stimmen = Mittel)
            g1 = Pi * p["d1"] - mu_eff
            act = ((g1 + g1.T) > 0) & self.A
        else:
            q = self.trust                               # erwartete Pflege durch den Partner
            gain = Pi * (q * p["d1"] + (1 - q) * (p["d0"] - p["d1"])) - mu_eff
            act = (gain > 0) & self.A
        flip = (rng.random(act.shape) < p["eps"]) & self.A
        act = act ^ flip
        if getattr(self, "force_lapse", None) is not None:   # erzwungenes einmaliges Versaeumnis
            for i, j in self.force_lapse:
                act[i, j] = False
            self.force_lapse = None
        # Zerfall
        both = act & act.T
        one = (act ^ act.T)
        U = rng.random(act.shape)
        U = np.triu(U, 1)
        U = U + U.T
        die = self.A & ((one & (U < p["d1"])) | (~both & ~one & (U < p["d0"])))
        self.A = self.A & ~die
        m_ = p["memory"]
        self.trust = np.where(self.A, m_ * self.trust + (1 - m_) * act.T, 0.0)
        self.act = act & self.A
        # Wiederaufbau
        mu_vec = mu_eff[:, 0] if np.ndim(mu_eff) else np.full(self.N, mu_eff)
        self._rebuild(mu_vec, p["K"] * (1.0 - D_alpha))
        self.last = dict(mu_eff=mu_eff, S=S, n_act=int(act.sum()))

    def _rebuild(self, mu_eff, K_eff):
        p = self.p
        rng = self.rng
        if p["r"] <= 0:
            return
        S, C, two = self.reach()
        eta, k0, N = p["eta"], p["k0"], self.N
        nb = self._nbrs() if p["eta3"] > 0 else None
        if nb is not None:
            S = np.array([self._S_bfs(nb, i) for i in range(N)])
        def try_link(i, j):
            if j == i or self.A[i, j]:
                return
            for a, b in ((i, j), (j, i)):
                if nb is not None:
                    dv = p["B"] * (f(self._S_bfs(nb, a, add=b), k0) - f(S[a], k0))
                else:
                    new2 = self.A[b] & ~self.A[a] & ~two[a]
                    new2[a] = False
                    gainS = 1.0 - (eta if two[a, b] else 0.0) + eta * new2.sum()
                    dv = p["B"] * (f(S[a] + gainS, k0) - f(S[a], k0))
                Pi = max(0.0, p["p"] * dv - mu_eff[a]) / (1.0 - self.beta_i[a])
                if Pi - K_eff <= 0:
                    return
            self.A[i, j] = self.A[j, i] = True
            self.act[i, j] = self.act[j, i] = True
            self.trust[i, j] = self.trust[j, i] = 1.0

        for i in np.flatnonzero(rng.random(N) < p["r"]):
            former = np.flatnonzero(self.A0[i] & ~self.A[i])
            if len(former) and rng.random() < 0.5:
                j = former[rng.integers(len(former))]
            else:
                c2 = np.flatnonzero(two[i])
                if len(c2) == 0:
                    continue
                j = c2[rng.integers(len(c2))]
            try_link(i, j)
        if p["r_rand"] > 0:          # seltene Kontakte ausserhalb des eigenen Umfelds
            for i in np.flatnonzero(rng.random(N) < p["r_rand"]):
                try_link(i, int(rng.integers(N)))

    # --- Kennzahlen -------------------------------------------------------------
    def stats(self, D_alpha):
        from struktur import largest_component
        S, C, two = self.reach()
        if self.p["eta3"] > 0:
            nb = self._nbrs()
            S = np.array([self._S_bfs(nb, i) for i in range(self.N)])
        d = self.A.sum(1)
        V = f(S, self.p["k0"]) * self.p["B"]
        mu_full = self.p["mu"] * (1.0 - D_alpha)          # volle Kosten (inkl. Subvention)
        # Spieltyp jeder bestehenden Beziehung (aus Sicht beider Endpunkte)
        p = self.p
        mu_eff = mu_full * (1.0 - p["subsidy"]) * self.mu_i[:, None]
        dV, ext, _ = self.marginal_values()
        Pi = np.maximum(0.0, p["p"] * (dV + 0.5 * p["social"] * ext) - mu_eff) / (1.0 - self.beta_i[:, None])
        g1 = Pi * p["d1"] - mu_eff                  # Partner pflegt
        g0 = Pi * (p["d0"] - p["d1"]) - mu_eff      # Partner pflegt nicht
        E = self.A
        n = max(1, int(E.sum()))
        return dict(
            typ_pflege_dominant=float(((g0 > 0) & E).sum() / n),
            typ_hirschjagd=float(((g1 > 0) & (g0 <= 0) & E).sum() / n),
            typ_aufgeben=float(((g1 <= 0) & E).sum() / n),
            beziehungen=d.mean(),
            anteil_erhalten=float(self.A.sum() / max(1, self.A0.sum())),
            isoliert=float((d == 0).mean()),
            lcc=len(largest_component(self.A)) / self.N,
            ernstfallwert=float(V.mean()),
            wohlfahrt=float(self.p["p"] * V.mean() - mu_full * (self.mu_i * self.act.sum(1)).mean()),
            pflege_anteil=float(self.act.sum() / max(1, self.A.sum())),
            pflege_kosten=mu_full)


def sweep(A0, D_of_alpha, alphas_up, T_per, seed, back=True, **kw):
    """Quasi-statischer Durchlauf: alpha steigt (und faellt danach wieder, wenn back)."""
    m = Future(A0, seed, **kw)
    path = list(alphas_up) + (list(alphas_up[-2::-1]) if back else [])
    rows = []
    for k, a in enumerate(path):
        D = D_of_alpha(a)
        for _ in range(T_per):
            m.step(D)
        st = m.stats(D)
        st.update(alpha=a, phase="hoch" if k < len(alphas_up) else "runter", schritt=k)
        rows.append(st)
    return rows


# ---------------------------------------------------------------------------
# Analytik fuer einen einzelnen Akteur (Ebene 1 und 2, ohne Netzeffekte)
# ---------------------------------------------------------------------------
def k_star(mu_eff, p_, B, k0, beta, d1, d0, trust=1.0):
    """Groesste Kontaktzahl k, bei der die k-te Beziehung noch gepflegt wird.
    Marginalwert der k-ten Beziehung: dV(k) = B e^{-k/k0} (e^{1/k0} - 1).
    Pflege lohnt, wenn  [trust*d1 + (1-trust)(d0-d1)] (p dV - mu)/(1-beta) > mu.
    trust = 1: Partner pflegt sicher (kooperatives Gleichgewicht der Hirschjagd);
    trust = 1/2: Unsicherheit ueber den Partner."""
    w = trust * d1 + (1 - trust) * (d0 - d1)
    need = mu_eff * (1.0 + (1.0 - beta) / w) / p_       # erforderlicher dV
    a = B * (np.exp(1.0 / k0) - 1.0)
    with np.errstate(divide="ignore"):
        k = k0 * np.log(a / need)
    return np.maximum(0.0, np.floor(k))


# ---------------------------------------------------------------------------
# Die einfache Regel
# ---------------------------------------------------------------------------
def tie_rule_threshold(D, beta, d1, K, mu):
    """Eine Beziehung bleibt dauerhaft erhalten, wenn
           b / c  >  (1 - D) * (1 + max(1/w, K/c) / T)
       b = p * dV (erwarteter Ernstfallnutzen je Periode), c = mu (Pflegekosten
       ohne Nebenprodukt), w = d1, T = 1/(1-beta), D = Nebenprodukt-Anteil.
       Der erste Term (1/w) ist die Pflegebedingung im kooperativen Gleichgewicht
       der Hirschjagd, der zweite (K/c) die Bedingung, verlorene Beziehungen zu ersetzen."""
    T = 1.0 / (1.0 - beta)
    return (1.0 - D) * (1.0 + max(1.0 / d1, K / mu) / T)


def k_star_mf(mu, D, p=None):
    """Mean-Field: groesste Kontaktzahl k, bei der in einem Netz, in dem alle k
    Kontakte haben, die k-te Beziehung die Regel erfuellt (2-Schritt-Reichweite,
    baumartige Umgebung: S_k = k (1 + eta (k-1)))."""
    p = dict(DEFAULTS) if p is None else p
    thr = tie_rule_threshold(D, p["beta"], p["d1"], p["K"], mu)
    best = 0
    for k in range(1, 60):
        S = k * (1.0 + p["eta"] * (k - 1))
        dV = p["B"] * (f(S, p["k0"]) - f(S - 1.0 - p["eta"] * (k - 1), p["k0"]))
        if p["p"] * dV / mu > thr:
            best = k
    return best


def collapse_predicted(mu, D, p=None, k_crit=3):
    """Netz zerfaellt, wenn sich die dritte Beziehung nicht mehr lohnt.
    Begruendung (Molloy-Reed): bleibt von k* gewuenschten Beziehungen jede mit
    Wahrscheinlichkeit rho < 1 erhalten (natuerlicher Verlust), existiert eine
    Riesenkomponente nur, wenn (k* - 1) rho > 1, also nur fuer k* >= 3."""
    return k_star_mf(mu, D, p) < k_crit
