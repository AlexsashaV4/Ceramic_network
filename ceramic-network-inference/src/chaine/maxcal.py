"""Maximum Caliber (MaxCal) on a finite space of chains.

MaxCal (Jaynes 1980; Pressé et al. 2013, Rev. Mod. Phys. 85:1115) picks the
path distribution of maximum entropy S = -sum_G P(G) ln P(G) that reproduces
chosen averages <A_k(G)> = a_k. The solution is an exponential family

    P(G) = exp(sum_k lambda_k A_k(G)) / Z(lambda),

and lambda solves the convex dual  min_lambda  ln Z(lambda) - lambda . a.
Here the path space is finite (all chains the grammar allows, up to a
probability cut-off), so Z is an explicit sum.
"""
from collections import Counter

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from .data import END, START


def transition_features(steps):
    padded = [START] + list(steps) + [END]
    return Counter(zip(padded[:-1], padded[1:]))


def feature_matrix(paths, feature_fn):
    """Rows = paths, columns = features (e.g. transition counts)."""
    rows = [feature_fn(p) for p in paths]
    names = sorted({k for r in rows for k in r}, key=str)
    col = {k: j for j, k in enumerate(names)}
    F = np.zeros((len(paths), len(names)))
    for i, r in enumerate(rows):
        for k, v in r.items():
            F[i, col[k]] = v
    return F, names


def fit_maxcal(F, target, log_prior=None, l2=1e-6):
    """Lagrange multipliers lambda and path probabilities P matching F^T P = target.

    With log_prior (log Q(G)) this is minimum relative entropy (MaxCal relative
    to a reference process Q): P(G) ∝ Q(G) exp(lambda . A(G)).
    """
    base = np.zeros(F.shape[0]) if log_prior is None else np.asarray(log_prior, float)

    def dual(lam):
        s = base + F @ lam
        logz = logsumexp(s)
        p = np.exp(s - logz)
        return logz - lam @ target + 0.5 * l2 * lam @ lam, F.T @ p - target + l2 * lam

    res = minimize(dual, np.zeros(F.shape[1]), jac=True, method="L-BFGS-B",
                   options={"maxiter": 5000, "gtol": 1e-10})
    s = base + F @ res.x
    return res.x, np.exp(s - logsumexp(s)), res
