"""Biterm topic model (Yan, Guo, Lan & Cheng 2013) for short documents.

A *biterm* is an unordered pair of tokens occurring in the same document; a
document of N tokens has N(N-1)/2 biterms. BTM models the biterms of the whole
corpus rather than the documents:

    theta ~ Dir(alpha)                    one topic mixture for the corpus
    phi_k ~ Dir(beta)                     topics, as in LDA
    z_b ~ Cat(theta),  w_b1, w_b2 ~ Cat(phi_{z_b})   for every biterm b

Both tokens of a biterm share one topic, so the topics are learned from
co-occurrence pooled over all documents. This is BTM's answer to sparse
documents - the role the legacy pseudo-documents played, but without using
the labels. A document's topic mixture is inferred afterwards from its
biterms: p(z | d) = sum_b p(z | b) p(b | d), with p(z | b) proportional to
theta_z phi_z(w_b1) phi_z(w_b2) and p(b | d) the share of b among d's biterms.
"""
from itertools import combinations

import numpy as np

try:                                   # optional: compiles the sampler (same draws, same result)
    from numba import njit
except ImportError:                    # pragma: no cover
    def njit(f):
        return f


def biterms(tokens):
    """All unordered pairs of token positions of one document (repeats kept)."""
    return list(combinations(tokens, 2))


@njit
def _gibbs(B, z, n_z, n_wz, u, alpha, beta):
    """Collapsed Gibbs sweeps over all biterms; u holds the uniform draws (n_iter x n_biterms)."""
    K, V = n_z.shape[0], n_wz.shape[0]
    p = np.empty(K)
    for it in range(u.shape[0]):
        for i in range(B.shape[0]):
            w1, w2, k = B[i, 0], B[i, 1], z[i]
            n_z[k] -= 1
            n_wz[w1, k] -= 1
            n_wz[w2, k] -= 1
            same = 1.0 if w1 == w2 else 0.0
            total = 0.0
            for j in range(K):
                total += ((n_z[j] + alpha) * (n_wz[w1, j] + beta) * (n_wz[w2, j] + beta + same)
                          / ((2 * n_z[j] + V * beta) * (2 * n_z[j] + 1 + V * beta)))
                p[j] = total
            r, k = u[it, i] * total, 0
            while p[k] < r and k < K - 1:
                k += 1
            z[i] = k
            n_z[k] += 1
            n_wz[w1, k] += 1
            n_wz[w2, k] += 1


class BTM:
    """Collapsed Gibbs sampler for the biterm topic model.

    Defaults follow Yan et al. (2013): alpha = 50 / K, beta = 0.01.
    """

    def __init__(self, n_topics, alpha=None, beta=0.01, n_iter=300, seed=0):
        self.K, self.beta, self.n_iter, self.seed = n_topics, beta, n_iter, seed
        self.alpha = 50.0 / n_topics if alpha is None else alpha

    def fit(self, docs, vocab=None):
        """docs: list of token lists (strings). vocab: optional fixed vocabulary."""
        self.vocab = sorted({t for d in docs for t in d}) if vocab is None else list(vocab)
        self.index = {w: i for i, w in enumerate(self.vocab)}
        V, K, a, b = len(self.vocab), self.K, self.alpha, self.beta
        B = np.array([pair for d in docs for pair in biterms([self.index[t] for t in d if t in self.index])],
                     dtype=int).reshape(-1, 2)
        rng = np.random.default_rng(self.seed)
        z = rng.integers(K, size=len(B))
        n_z = np.bincount(z, minlength=K).astype(float)
        n_wz = np.zeros((V, K))
        np.add.at(n_wz, (B[:, 0], z), 1)
        np.add.at(n_wz, (B[:, 1], z), 1)
        u = rng.random((self.n_iter, len(B)))
        _gibbs(B, z, n_z, n_wz, u, float(a), float(b))
        self.n_biterms = len(B)
        self.theta = (n_z + a) / (len(B) + K * a)
        self.phi = ((n_wz + b) / (2 * n_z + V * b)).T            # K x V, rows sum to 1
        return self

    def transform(self, docs):
        """p(z | d) for each document, from its biterms (from its single token if it has one)."""
        out = np.empty((len(docs), self.K))
        for j, d in enumerate(docs):
            ids = [self.index[t] for t in d if t in self.index]
            if len(ids) >= 2:
                bt = np.array(biterms(ids))
                pzb = self.theta * self.phi[:, bt[:, 0]].T * self.phi[:, bt[:, 1]].T
                pzb /= pzb.sum(1, keepdims=True)
                out[j] = pzb.mean(0)
            elif len(ids) == 1:
                p = self.theta * self.phi[:, ids[0]]
                out[j] = p / p.sum()
            else:
                out[j] = self.theta
        return out
