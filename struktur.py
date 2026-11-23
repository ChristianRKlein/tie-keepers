# -*- coding: utf-8 -*-
"""
struktur.py  --  Wann beguenstigt eine Netzwerkstruktur Kooperation?
====================================================================

Evolutionaeres Gefangenendilemma (Spenderspiel) auf einem gegebenen Graphen,
Death-Birth-Update, schwache Selektion.

1) Ohtsuki et al. (2006, Nature 441:502):   b/c > k
2) Allen et al.  (2017, Nature 544:227):    exakte Schwelle fuer JEDEN Graphen
       (b/c)* = tau2 / (tau3 - tau1),
   tau_ij = Koaleszenzzeiten:  tau_ii = 0,
       tau_ij = 1 + 1/2 * sum_k (p_ik tau_kj + p_jk tau_ik)   (i != j)
   tau_n  = sum_ij pi_i p^(n)_ij tau_ij
3) Heterogene Verallgemeinerung (hier hergeleitet wie in Allen et al. 2017,
   Gl. fuer erste Ordnung in delta; vgl. McAvoy, Allen & Nowak 2020,
   Nat. Hum. Behav. 4:819): Kante j->i stiftet Nutzen b_ji, Spender j zahlt c_j.
   Kooperation wird beguenstigt, wenn  B_term > C_term  mit
       C_term = sum_{i,k} pi_i p2_ik c_k tau_ik
       B_term = sum_{i,k} pi_i p2_ik sum_j p_kj b_jk tau_ij
                - sum_{i,j} pi_i p_ij b_ji tau_ij
   Fuer einheitliche b, c reduziert sich das exakt auf b/c > (b/c)*.
   Strukturelle Kooperationsmarge:  MARGE = B_term / C_term  (>1: beguenstigt)

Die Koaleszenzzeiten werden mit konjugierten Gradienten im pi(x)pi-gewichteten
Skalarprodukt geloest (dort ist der Operator selbstadjungiert und positiv
definit). Nur NumPy noetig.
"""
import numpy as np


def largest_component(A):
    """Indizes der groessten Zusammenhangskomponente (Breitensuche)."""
    N = A.shape[0]
    seen = np.zeros(N, dtype=bool)
    best = np.array([], dtype=int)
    deg = A.sum(1)
    for s in range(N):
        if seen[s] or deg[s] == 0:
            continue
        comp = [s]
        seen[s] = True
        frontier = np.array([s])
        while len(frontier):
            nb = np.flatnonzero(A[frontier].any(0) & ~seen)
            seen[nb] = True
            comp.extend(nb.tolist())
            frontier = nb
        if len(comp) > len(best):
            best = np.array(sorted(comp))
    return best


def coalescence_times(W, tol=1e-10, maxiter=20000):
    """Koaleszenzzeiten tau (n x n) fuer gewichteten, zusammenhaengenden Graphen W."""
    W = np.asarray(W, dtype=float)
    w = W.sum(1)
    P = W / w[:, None]
    pi = w / w.sum()
    PI = np.outer(pi, pi)
    n = len(w)
    off = ~np.eye(n, dtype=bool)

    def op(X):
        Y = X - 0.5 * (P @ X + X @ P.T)
        Y[~off] = 0.0
        return Y

    def dot(X, Y):
        return float(np.sum(PI * X * Y))

    b = off.astype(float)
    x = np.zeros((n, n))
    r = b.copy()
    pdir = r.copy()
    rr = dot(r, r)
    bnorm = np.sqrt(rr)
    it = 0
    for it in range(1, maxiter + 1):
        Ap = op(pdir)
        a = rr / dot(pdir, Ap)
        x += a * pdir
        r -= a * Ap
        rr_new = dot(r, r)
        if np.sqrt(rr_new) < tol * bnorm:
            break
        pdir = r + (rr_new / rr) * pdir
        rr = rr_new
    x = 0.5 * (x + x.T)
    return x, P, pi, it


def structure_condition(W, B=None, c=None):
    """Kennzahlen fuer einen zusammenhaengenden Graphen W.
    B[j, i] = Nutzen von j fuer i (nur auf Kanten), c[i] = Kosten des Spenders i.
    """
    tau, P, pi, it = coalescence_times(W)
    P2 = P @ P
    P3 = P2 @ P
    t1 = float(np.sum(pi[:, None] * P * tau))
    t2 = float(np.sum(pi[:, None] * P2 * tau))
    t3 = float(np.sum(pi[:, None] * P3 * tau))
    den = t3 - t1
    bc_star = t2 / den if den > 0 else np.inf
    deg = (np.asarray(W) > 0).sum(1)
    out = dict(n=len(pi), cg_iter=it, tau1=t1, tau2=t2, tau3=t3,
               bc_star_allen=bc_star, k_mean=float(deg.mean()),
               k_harm=float(1.0 / np.mean(1.0 / deg)),
               k2_over_k=float((deg ** 2).mean() / deg.mean()),
               sigma=(bc_star + 1) / (bc_star - 1) if np.isfinite(bc_star) else 1.0)
    if B is not None and c is not None:
        E = P * B.T                         # E[k, j] = p_kj * b_jk
        B_term = float(np.sum(pi[:, None] * P2 * (tau @ E.T)) - np.sum(pi[:, None] * E * tau))
        C_term = float(np.sum(pi[:, None] * P2 * tau * c[None, :]))
        out.update(B_term=B_term, C_term=C_term,
                   marge_het=B_term / C_term if C_term > 0 else np.nan)
    return out


# ---------------------------------------------------------------------------
# Testgraphen fuer die Validierung
# ---------------------------------------------------------------------------
def random_regular(N, k, rng, swaps_per_edge=10):
    """Zufaelliger k-regulaerer Graph (k gerade): Ringgitter + gradtreue Kantentausche."""
    A = np.zeros((N, N), dtype=bool)
    for i in range(N):
        for s in range(1, k // 2 + 1):
            A[i, (i + s) % N] = A[(i + s) % N, i] = True
    E = int(A.sum() // 2)
    for _ in range(swaps_per_edge * E):
        iu, ju = np.nonzero(np.triu(A, 1)) if _ % E == 0 else (iu, ju)
        e1, e2 = rng.integers(len(iu), size=2)
        a, b, c, d = iu[e1], ju[e1], iu[e2], ju[e2]
        if rng.random() < 0.5:
            c, d = d, c
        if len({a, b, c, d}) < 4 or A[a, c] or A[b, d] or not (A[a, b] and A[c, d]):
            continue
        A[a, b] = A[b, a] = A[c, d] = A[d, c] = False
        A[a, c] = A[c, a] = A[b, d] = A[d, b] = True
    if len(largest_component(A)) < N:
        return random_regular(N, k, rng, swaps_per_edge)
    return A


def cycle(N):
    A = np.zeros((N, N), dtype=bool)
    for i in range(N):
        A[i, (i + 1) % N] = A[(i + 1) % N, i] = True
    return A


def allen_regular_formula(N, k):
    """Exakte Schwelle fuer transitive k-regulaere Graphen: (N-2)/(N/k-2)."""
    den = N / k - 2.0
    return (N - 2.0) / den if den > 0 else np.inf
