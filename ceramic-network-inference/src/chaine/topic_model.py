"""The LDA pipeline of the legacy notebook `Presentation_topic_modelling.ipynb`,
function by function, with the same names and call signatures.

Each docstring names the legacy cell and what was checked or corrected. The
pipeline is: load documents -> vocabulary -> remove frequent triplets ->
bootstrap pseudo-documents -> document-term matrix -> LDA (collapsed Gibbs) ->
match topics to cultures -> classify new vessels.
"""
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.spatial.distance import jensenshannon

from .data import documents


# ------------------------------------------------------------------ documents
def load_pottery_data(df):
    """Documents as in legacy cell 2: one dict per vessel.

    Legacy: parsed data/clustered_paths_<culture>.csv with a regex, one file per
    culture, vessel IDs = line numbers. Here: built from the clean chains, with
    Kroon's vessel IDs and site/type metadata; the triplets are identical
    (checked in tests/test_core.py).
    """
    out = []
    for r, trip in zip(df.itertuples(), documents(df)):
        out.append({
            "vessel_id": r.vessel_id, "culture": str(r.bok), "site_id": r.site_id,
            "vessel_type": r.vessel_type, "triplets": trip,
            "triplets_text": " ".join(trip), "num_triplets": len(trip),
        })
    return out


def extract_vocabulary(docs):
    """Legacy cell 2: vocabulary in order of first appearance, and term frequencies."""
    vocab_counter = Counter()
    for doc in docs:
        vocab_counter.update(doc["triplets"])
    return list(vocab_counter), dict(vocab_counter)


def remove_common_triplets(docs, k=5, fit_docs=None):
    """Legacy cell 3: drop the k most frequent triplets ("stop words").

    Corrections: the legacy cell (i) counted frequencies on all vessels
    including the ones later used for testing, and (ii) edited the documents in
    place, which also changed `cw_docs` etc. except their last two vessels (the
    held-out ones kept the stop triplets). Here the top k are counted on
    `fit_docs` (the training vessels) and new document dicts are returned.
    """
    counter = Counter(t for d in (fit_docs or docs) for t in d["triplets"])
    removed = [t for t, _ in counter.most_common(k)]
    drop = set(removed)
    new = []
    for d in docs:
        trip = [t for t in d["triplets"] if t not in drop]
        new.append({**d, "triplets": trip, "triplets_text": " ".join(trip), "num_triplets": len(trip)})
    return new, removed


def stop_triplet_test(docs, triplets, cultures):
    """Chi-square test of independence between presence of each triplet and culture,
    with the share of vessels of each culture that contain it."""
    from scipy.stats import chi2_contingency

    rows = []
    for t in triplets:
        table = np.zeros((2, len(cultures)))
        for d in docs:
            table[int(t in d["triplets"]), cultures.index(d["culture"])] += 1
        chi2, p, _, _ = chi2_contingency(table)
        rows.append({"triplet": t, "chi2": chi2, "p_value": p,
                     **{f"share {c}": s for c, s in zip(cultures, table[1] / table.sum(0))}})
    return pd.DataFrame(rows).set_index("triplet")


def create_resampled_documents(docs, group_size=3, num_new_docs=100, seed=0):
    """Legacy cell 5: bootstrap pseudo-documents.

    For each culture, `num_new_docs` times: draw `group_size` vessels of that
    culture with replacement and concatenate their triplets.
    Correction: a seeded generator (the legacy used the global np.random state,
    so every run gave different pseudo-documents).
    """
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for d in docs:
        groups[d["culture"]].append(d)
    out = []
    for culture, pool in groups.items():
        for _ in range(num_new_docs):
            pick = [pool[i] for i in rng.integers(len(pool), size=group_size)]
            trip = [t for v in pick for t in v["triplets"]]
            out.append({
                "vessel_id": f"RESAMPLED_{culture}_{len(out) + 1}",
                "original_vessels": [v["vessel_id"] for v in pick],
                "culture": culture, "triplets": trip, "triplets_text": " ".join(trip),
                "num_triplets": len(trip), "group_size": group_size,
            })
    return out


def normalize_triplet(t):
    """Legacy cell 6: canonical '(A, B, C)' form of a triplet string."""
    t = re.sub(r"\s+", " ", t.strip().strip('"').strip("'").replace("',", ",").replace("'", ""))
    parts = [p.strip() for p in t.strip("()").split(",")]
    return f"({parts[0]}, {parts[1]}, {parts[2]})" if len(parts) == 3 else t


def build_dtm_with_external_vocab(docs, vocab_index):
    """Legacy cell 6: document-term count matrix over a fixed vocabulary.

    Correction: the legacy version also drew a spy plot on every call.
    """
    rows, cols, vals, lengths = [], [], [], []
    for d, doc in enumerate(docs):
        counts = Counter(normalize_triplet(t) for t in doc["triplets"])
        lengths.append(sum(counts.values()))
        for tok, c in counts.items():
            j = vocab_index.get(tok)
            if j is not None:
                rows.append(d); cols.append(j); vals.append(c)
    X = csr_matrix((vals, (rows, cols)), shape=(len(docs), len(vocab_index)), dtype=np.int32)
    return X, lengths


def dataset_summary(datasets, vocab_index):
    """The per-dataset printout of legacy cell 6, as a table."""
    rows = []
    for name, docs in datasets.items():
        X, lengths = build_dtm_with_external_vocab(docs, vocab_index)
        rows.append({"dataset": name, "documents": X.shape[0], "vocabulary": X.shape[1],
                     "non-zero entries": X.nnz, "density": X.nnz / np.prod(X.shape),
                     "mean triplets / document": np.mean(lengths),
                     "documents per culture": dict(Counter(d["culture"] for d in docs))})
    return pd.DataFrame(rows).set_index("dataset")


def get_empirical_word_distribution(docs, vocab, cultures):
    """p(triplet | culture) from pooled counts (legacy cells 11, 18, 61, 80).

    Correction: the legacy notebook recomputed MI_dist, CW_dist, ... four times,
    over three different vocabulary orders, from `cw_docs` etc., which included
    the held-out vessels and, for them, the removed stop triplets. Here: one
    function, one vocabulary order, only the documents passed in.
    """
    index = {w: i for i, w in enumerate(vocab)}
    out = np.zeros((len(cultures), len(vocab)))
    for d in docs:
        for t in d["triplets"]:
            if t in index:
                out[cultures.index(d["culture"]), index[t]] += 1
    return out / out.sum(1, keepdims=True)


# ------------------------------------------------------------------------ LDA
class LDA_manual:
    """Legacy cell 14: LDA fitted by collapsed Gibbs sampling, same interface.

        p(z = k | rest) ∝ (n_dk + alpha) (n_kw + beta) / (n_k + V beta)

    Checked: the sampler of the legacy class is correct. Changes: a local random
    generator instead of np.random.seed; a token-array implementation of the
    same sampler (much faster than looping over the dense matrix); fold-in
    starts from a uniform theta (the legacy code started from a random
    Dirichlet draw, so predictions changed between calls); column alignment in
    transform fixed for missing features.
    """

    def __init__(self, n_topics=4, alpha=1, beta=1, n_iter=100, random_state=42):
        self.n_topics, self.alpha, self.beta = n_topics, alpha, beta
        self.n_iter, self.random_state = n_iter, random_state

    def fit(self, X, feature_names=None):
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        self.n_docs, self.n_words = X.shape
        self.feature_names_ = np.array(feature_names if feature_names is not None
                                       else np.arange(self.n_words), dtype=object)
        rng = np.random.default_rng(self.random_state)
        K, V, a, b = self.n_topics, self.n_words, self.alpha, self.beta
        words = [np.repeat(np.arange(V), X[d]).tolist() for d in range(self.n_docs)]
        ndk = [[0] * K for _ in words]
        nkw = [[0] * V for _ in range(K)]
        nk = [0] * K
        z = []
        for d, ws in enumerate(words):
            zd = rng.integers(K, size=len(ws)).tolist()
            for w, k in zip(ws, zd):
                ndk[d][k] += 1; nkw[k][w] += 1; nk[k] += 1
            z.append(zd)
        u, vb = rng.random, V * b
        for _ in range(self.n_iter):
            for d, ws in enumerate(words):
                nd, zd = ndk[d], z[d]
                for i, w in enumerate(ws):
                    k = zd[i]
                    nd[k] -= 1; nkw[k][w] -= 1; nk[k] -= 1
                    p = [(nd[j] + a) * (nkw[j][w] + b) / (nk[j] + vb) for j in range(K)]
                    r, k = u() * sum(p), 0
                    while r > p[k] and k < K - 1:
                        r -= p[k]; k += 1
                    zd[i] = k
                    nd[k] += 1; nkw[k][w] += 1; nk[k] += 1
        self.doc_topic_counts = np.array(ndk, float)
        self.topic_word_counts = np.array(nkw, float)
        self.doc_topic_dist = (self.doc_topic_counts + a) / (self.doc_topic_counts.sum(1, keepdims=True) + K * a)
        self.topic_word_dist = (self.topic_word_counts + b) / (self.topic_word_counts.sum(1, keepdims=True) + vb)
        return self

    def get_top_words(self, feature_names, n_words=10):
        return [[(feature_names[i], self.topic_word_dist[k, i])
                 for i in np.argsort(-self.topic_word_dist[k])[:n_words]] for k in range(self.n_topics)]

    def transform(self, X, n_iter=50, feature_names=None):
        """Fold-in: theta of new documents with the topics phi fixed (EM iterations)."""
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X, float)
        if feature_names is not None:  # align columns to the training vocabulary
            pos = {f: i for i, f in enumerate(feature_names)}
            aligned = np.zeros((X.shape[0], len(self.feature_names_)))
            for j, f in enumerate(self.feature_names_):
                if f in pos:
                    aligned[:, j] = X[:, pos[f]]
            X = aligned
        phi, K = self.topic_word_dist, self.n_topics
        theta = np.full((X.shape[0], K), 1.0 / K)
        for d in range(X.shape[0]):
            x = X[d]
            if x.sum() == 0:
                continue
            for _ in range(n_iter):
                r = phi * theta[d][:, None]
                r /= r.sum(0, keepdims=True) + 1e-12
                n_k = (r * x[None, :]).sum(1)
                theta[d] = (n_k + self.alpha) / (n_k + self.alpha).sum()
        return theta

    def predict_proba(self, X, **kw):
        return self.transform(X, **kw)

    def predict(self, X, **kw):
        return self.transform(X, **kw).argmax(1)


# --------------------------------------------------------------- evaluation
def js_divergence(p, q):
    """Legacy cell 18: Jensen-Shannon divergence (here in bits, base 2)."""
    return jensenshannon(p, q, base=2) ** 2


def matching_empirical(empirical, topic_dists, cultures, one_to_one=True):
    """Legacy cell 24: which topic corresponds to which culture.

    Legacy: each culture took its closest topic (argmin JS) independently, so two
    cultures could claim the same topic (in the legacy run MI and EFG -> topic 3)
    and a topic could stay unassigned. one_to_one=True uses the Hungarian
    algorithm to find the one-to-one assignment with minimal total JS.
    Returns ({culture: topic}, JS matrix cultures x topics).
    """
    J = np.array([[js_divergence(e, t) for t in topic_dists] for e in empirical])
    if one_to_one:
        rows, cols = linear_sum_assignment(J)
        topics = cols[np.argsort(rows)]
    else:
        topics = J.argmin(1)
    return {c: int(k) for c, k in zip(cultures, topics)}, J


def classify_new_vessel(new_triplets, trained_lda, vocab, culture_mapping=None):
    """Legacy cell 29: classify one vessel from its triplets.

    Correction: `culture_mapping` is {culture: topic} (the output of
    matching_empirical), but the legacy function looked it up as
    {topic: culture}, so it always returned 'Topic_k' instead of a culture. The
    mapping is inverted here.
    """
    vocab_index = {w: i for i, w in enumerate(vocab)}
    X, _ = build_dtm_with_external_vocab([{"triplets": list(new_triplets)}], vocab_index)
    topic_probs = trained_lda.transform(X)[0]
    dominant = int(topic_probs.argmax())
    topic_to_culture = {t: c for c, t in (culture_mapping or {}).items()}
    max_prob = float(topic_probs.max())
    return {
        "predicted_culture": topic_to_culture.get(dominant, f"Topic_{dominant}"),
        "topic_probabilities": topic_probs, "max_probability": max_prob,
        "confidence": "High" if max_prob >= 0.6 else ("Medium" if max_prob >= 0.4 else "Low"),
        "dominant_topic": dominant,
        "in_vocabulary": sum(t in vocab_index for t in new_triplets) / max(len(new_triplets), 1),
    }


def lda_culture_proba(trained_lda, docs, vocab, culture_mapping, cultures):
    """Class probabilities for many vessels: theta of each culture's topic, renormalised."""
    X, _ = build_dtm_with_external_vocab(docs, {w: i for i, w in enumerate(vocab)})
    theta = trained_lda.transform(X)
    p = theta[:, [culture_mapping[c] for c in cultures]]
    return p / p.sum(1, keepdims=True)
