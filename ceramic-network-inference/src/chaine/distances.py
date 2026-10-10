"""Comparing two bodies of knowledge (sets of chains).

kroon_wasserstein reproduces Pathfinder's compare_boks (Kroon 2024): for every
link of the technique network, the percentage of chains that use it, then
scipy's 1-D Wasserstein distance between the two lists of percentages. scipy
treats those lists as *samples of values*, so the distance compares the
histograms of edge usage and ignores which edge is which. The edge-aware
alternatives below keep the edge identity.
"""
from collections import Counter

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

from .data import END


def transitions(steps):
    return list(zip(steps[:-1], steps[1:]))


def edge_usage(chains, edges):
    """Percentage of chains using each edge, in the order of `edges` (Kroon's weights)."""
    counts = Counter(e for s in chains for e in transitions(s))
    return np.array([counts[e] for e in edges], float) / len(chains) * 100


def kroon_wasserstein(a, b, edges):
    return wasserstein_distance(edge_usage(a, edges), edge_usage(b, edges))


def edge_js(a, b, edges):
    """Jensen-Shannon divergence (bits) between the two edge-count distributions."""
    ca, cb = edge_usage(a, edges), edge_usage(b, edges)
    return jensenshannon(ca / ca.sum(), cb / cb.sum(), base=2) ** 2


def markov_divergence(a, b, make_model):
    """Symmetrised mean log-likelihood ratio between two Markov models (nats/chain).

    0.5 * ( E_a[log p_a - log p_b] + E_b[log p_b - log p_a] ): a plug-in
    estimate of the symmetrised KL divergence between the two chain
    distributions. `make_model(chains)` returns a fitted MarkovChain.
    """
    ma, mb = make_model(a), make_model(b)
    kl_ab = np.mean([ma.logprob(s) - mb.logprob(s) for s in a])
    kl_ba = np.mean([mb.logprob(s) - ma.logprob(s) for s in b])
    return 0.5 * (kl_ab + kl_ba)


def bootstrap(stat, a, b, n_boot=200, seed=0):
    """Bootstrap distribution of stat(a, b), resampling chains within each group."""
    rng = np.random.default_rng(seed)
    a, b = list(a), list(b)
    out = []
    for _ in range(n_boot):
        ra = [a[i] for i in rng.integers(len(a), size=len(a))]
        rb = [b[i] for i in rng.integers(len(b), size=len(b))]
        out.append(stat(ra, rb))
    return np.array(out)


def permutation_test(stat, a, b, n_perm=200, seed=0):
    """Observed stat, null distribution under shuffled labels, and one-sided p-value."""
    rng = np.random.default_rng(seed)
    pooled, na = list(a) + list(b), len(a)
    observed = stat(list(a), list(b))
    null = []
    for _ in range(n_perm):
        idx = rng.permutation(len(pooled))
        null.append(stat([pooled[i] for i in idx[:na]], [pooled[i] for i in idx[na:]]))
    null = np.array(null)
    return observed, null, (1 + np.sum(null >= observed)) / (1 + n_perm)


def random_chains(grammar, vocab, n, seed=0, max_len=25):
    """Kroon's control group 'BoK C': uniformly random valid chains from the grammar."""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        steps = []
        while len(steps) < max_len:
            cand = grammar.candidates(steps, vocab)
            if not cand:
                break
            x = cand[rng.integers(len(cand))]
            if x == END:
                out.append(steps)
                break
            steps.append(x)
    return out
