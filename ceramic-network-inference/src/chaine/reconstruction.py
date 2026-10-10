"""Reconstructing incomplete chaînes opératoires.

Masking experiment: for each held-out vessel hide a contiguous block of m
steps (never the first step, and at least one step stays after the gap),
then ask each model for its top-k gap fillers given what is left.
"""
from collections import Counter

import numpy as np
import pandas as pd

from .data import BOKS
from .markov import MarkovChain


def lookup_fill(table, prefix, suffix, k=3):
    """Baseline: the k most frequent training chains that match prefix and suffix."""
    n0, n1 = len(prefix), len(suffix)
    hits = Counter()
    for chain, n in table.items():
        if len(chain) > n0 + n1 and list(chain[:n0]) == prefix and list(chain[len(chain) - n1:]) == suffix:
            hits[tuple(chain[n0:len(chain) - n1])] += n
    return [list(g) for g, _ in hits.most_common(k)]


def gap_experiment(df, grammar, folds, gap_sizes=(1, 2, 3), k=3, betas=None, seed=0):
    """Top-1 / top-k exact recovery of hidden gaps on held-out vessels.

    Models: pooled vs BoK-specific (the BoK assumed known) Markov chains of
    order 1 and 2, and the lookup baseline. Returns one row per (vessel, m).
    """
    betas = betas or {1: 2.0, 2: 8.0}
    rng = np.random.default_rng(seed)
    rows = []
    for train_idx, test_idx in folds:
        train = df.iloc[train_idx]
        vocab = sorted({x for s in train["steps"] for x in s})
        fit = lambda o, chains: MarkovChain(o, betas[o], grammar=grammar, vocab=vocab).fit(chains)
        models = {f"pooled, order {o}": {b: fit(o, train["steps"]) for b in [None]} for o in (1, 2)}
        for o in (1, 2):
            models[f"per BoK, order {o}"] = {b: fit(o, train.loc[train["bok"] == b, "steps"]) for b in BOKS}
        table = Counter(tuple(s) for s in train["steps"])
        for r in df.iloc[test_idx].itertuples():
            s = list(r.steps)
            for m in gap_sizes:
                i = int(rng.integers(1, len(s) - m))
                prefix, gap, suffix = s[:i], s[i:i + m], s[i + m:]
                row = {"vessel_id": r.vessel_id, "bok": r.bok, "gap": m}
                for name, per in models.items():
                    model = per.get(r.bok, per.get(None))
                    fills = [f for f, _ in model.complete(prefix, suffix, k=k)]
                    row[(name, "top-1")] = bool(fills) and fills[0] == gap
                    row[(name, f"top-{k}")] = gap in fills
                fills = lookup_fill(table, prefix, suffix, k)
                row[("lookup", "top-1")] = bool(fills) and fills[0] == gap
                row[("lookup", f"top-{k}")] = gap in fills
                rows.append(row)
    return pd.DataFrame(rows)


def summarise_gaps(rows):
    cols = [c for c in rows.columns if isinstance(c, tuple)]
    out = rows.groupby("gap")[cols].mean().T
    out.index = pd.MultiIndex.from_tuples(out.index, names=["model", "metric"])
    return out.unstack("metric")
