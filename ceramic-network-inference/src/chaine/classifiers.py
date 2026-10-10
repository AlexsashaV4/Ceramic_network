"""Vessel -> body-of-knowledge classifiers with a common interface.

Every classifier has fit(train_df) and predict_proba(test_df) -> array
(n_test, n_classes) in the order of `classes`. All learning - stop-triplet
removal, bootstrap pseudo-documents, topics, block partitions - happens inside
fit, on the training vessels only.

The two topic-model pipelines call exactly the functions of the replicated
legacy notebook (chaine.topic_model, chaine.sbm), so the cross-validated
numbers refer to the same procedure shown step by step in the notebook.
"""
import numpy as np

from . import sbm
from . import topic_model as tm
from .data import BOKS, documents
from .markov import MarkovChain


class Majority:
    def __init__(self, classes=BOKS, label="bok"):
        self.classes, self.label = list(classes), label

    def fit(self, df):
        counts = df[self.label].astype(str).value_counts()
        self.p = np.array([counts.get(c, 0) for c in self.classes], float) / len(df)
        return self

    def predict_proba(self, df):
        return np.tile(self.p, (len(df), 1))


def _conditional_prior(df, column, classes, label, alpha=1.0):
    """p(class | column value), Dirichlet-smoothed, plus the marginal as fallback."""
    ys = df[label].astype(str)
    marginal = np.array([(ys == c).sum() for c in classes], float) + alpha
    table = {}
    for value, part in df.groupby(column, observed=True):
        py = part[label].astype(str)
        table[value] = np.array([(py == c).sum() for c in classes], float) + alpha
    norm = lambda v: v / v.sum()
    return {k: norm(v) for k, v in table.items()}, norm(marginal)


class ColumnPrior:
    """Metadata-only baseline: predict the class from one column (e.g. vessel type)."""

    def __init__(self, column, classes=BOKS, label="bok"):
        self.column, self.classes, self.label = column, list(classes), label

    def fit(self, df):
        self.table, self.marginal = _conditional_prior(df, self.column, self.classes, self.label)
        return self

    def predict_proba(self, df):
        return np.array([self.table.get(v, self.marginal) for v in df[self.column]])


class NaiveBayesNgram:
    """Multinomial naive Bayes on n-gram tokens with Dirichlet(alpha) smoothing.

    Equivalent to 'labelled LDA' with one topic per class: the supervised
    version of the topic model, no bootstrap needed.
    """

    def __init__(self, n=3, alpha=1.0, classes=BOKS, label="bok"):
        self.n, self.alpha, self.classes, self.label = n, alpha, list(classes), label

    def fit(self, df):
        docs, ys = documents(df, self.n), df[self.label].astype(str).tolist()
        self.vocab = sorted({w for d in docs for w in d})
        index = {w: i for i, w in enumerate(self.vocab)}
        counts = np.zeros((len(self.classes), len(self.vocab)))
        for d, y in zip(docs, ys):
            for w in d:
                counts[self.classes.index(y), index[w]] += 1
        self.log_phi = np.log((counts + self.alpha) / (counts + self.alpha).sum(1, keepdims=True))
        prior = np.array([ys.count(c) for c in self.classes], float)
        self.log_prior = np.log(prior / prior.sum())
        self.index = index
        return self

    def predict_proba(self, df):
        out = []
        for d in documents(df, self.n):
            ll = self.log_prior.copy()
            for w in d:
                if w in self.index:
                    ll += self.log_phi[:, self.index[w]]
            p = np.exp(ll - ll.max())
            out.append(p / p.sum())
        return np.array(out)


class MarkovClassifier:
    """Generative sequence classifier: p(BoK | chain) ∝ p(BoK) p(chain | BoK).

    prior_column (e.g. 'vessel_type') replaces p(BoK) by p(BoK | vessel type),
    to ask whether the chain adds information beyond that metadata.
    """

    def __init__(self, order=1, beta=1.0, alpha=0.5, grammar=None, classes=BOKS, label="bok",
                 prior_column=None):
        self.kw = dict(order=order, beta=beta, alpha=alpha, grammar=grammar)
        self.classes, self.label, self.prior_column = list(classes), label, prior_column

    def fit(self, df):
        vocab = sorted({x for s in df["steps"] for x in s})
        ys = df[self.label].astype(str)
        self.models = {c: MarkovChain(vocab=vocab, **self.kw).fit(df.loc[ys == c, "steps"])
                       for c in self.classes}
        prior = np.array([(ys == c).sum() for c in self.classes], float)
        self.log_prior = np.log(prior / prior.sum())
        if self.prior_column:
            self.table, _ = _conditional_prior(df, self.prior_column, self.classes, self.label)
        return self

    def _log_prior(self, df):
        if not self.prior_column:
            return self.log_prior
        return np.log([self.table.get(v, np.exp(self.log_prior)) for v in df[self.prior_column]])

    def loglik(self, df, complete=True):
        return np.array([[self.models[c].logprob(s, complete) for c in self.classes]
                         for s in df["steps"]])

    def predict_proba(self, df, complete=True):
        """complete=False scores the steps as an unfinished chain (e.g. a fragment)."""
        ll = self.loglik(df, complete) + self._log_prior(df)
        ll = np.where(np.isfinite(ll), ll, -1e9)
        p = np.exp(ll - ll.max(1, keepdims=True))
        return p / p.sum(1, keepdims=True)


# --------------------------------------------------- legacy topic-model pipelines
def _prepare(df, remove_top):
    """Legacy preprocessing on the training vessels: documents and stop-triplet removal."""
    docs = tm.load_pottery_data(df)
    docs, removed = tm.remove_common_triplets(docs, remove_top) if remove_top else (docs, [])
    return docs, removed


class LDAPipeline:
    """Legacy LDA pipeline (notebook Section 4), fold-safe:
    remove_common_triplets -> create_resampled_documents -> build_dtm ->
    LDA_manual (K = number of classes) -> matching_empirical (one-to-one) ->
    fold-in theta of each test vessel (lda_culture_proba)."""

    def __init__(self, g=7, n_per_class=100, alpha=1.0, beta=1.0, n_iter=200, remove_top=5,
                 seed=0, classes=BOKS, label="bok"):
        self.g, self.n_per_class, self.alpha, self.beta = g, n_per_class, alpha, beta
        self.n_iter, self.remove_top, self.seed = n_iter, remove_top, seed
        self.classes, self.label = list(classes), label

    def fit(self, df):
        docs, self.removed = _prepare(df.assign(bok=df[self.label]), self.remove_top)
        self.vocabulary, _ = tm.extract_vocabulary(docs)
        pdocs = tm.create_resampled_documents(docs, self.g, self.n_per_class, self.seed)
        X, _ = tm.build_dtm_with_external_vocab(pdocs, {w: i for i, w in enumerate(self.vocabulary)})
        self.lda = tm.LDA_manual(len(self.classes), self.alpha, self.beta, self.n_iter, self.seed)
        self.lda.fit(X, self.vocabulary)
        emp = tm.get_empirical_word_distribution(docs, self.vocabulary, self.classes)
        self.mapping, _ = tm.matching_empirical(emp, self.lda.topic_word_dist, self.classes)
        return self

    def predict_proba(self, df):
        docs = tm.load_pottery_data(df.assign(bok=df[self.label]))
        drop = set(self.removed)
        docs = [{**d, "triplets": [t for t in d["triplets"] if t not in drop]} for d in docs]
        return tm.lda_culture_proba(self.lda, docs, self.vocabulary, self.mapping, self.classes)


class SBMPipeline:
    """Legacy hSBM pipeline (notebook Section 5), fold-safe:
    remove_common_triplets -> create_resampled_documents -> hSBM (sbmtm fit +
    merge-split refinement) -> classify each test vessel with

    method='clusters' : the legacy cluster classifier (classify_vessel_simple), or
    method='fold-in'  : description-length fold-in (infer_new_doc_membership).

    Both methods on the same training data share one fitted hSBM (cached), so
    they differ only in how a new vessel is placed.
    """

    _cache = {}

    def __init__(self, method="fold-in", g=7, n_per_class=100, remove_top=5, seed=0,
                 refine_sweeps=50, classes=BOKS, label="bok"):
        self.method, self.g, self.n_per_class, self.remove_top = method, g, n_per_class, remove_top
        self.seed, self.refine_sweeps = seed, refine_sweeps
        self.classes, self.label = list(classes), label

    def fit(self, df):
        docs, self.removed = _prepare(df.assign(bok=df[self.label]), self.remove_top)
        if self.g:
            pdocs = tm.create_resampled_documents(docs, self.g, self.n_per_class, self.seed)
        else:  # no bootstrap: one document per vessel
            pdocs = docs
        tokens = [d["triplets"] for d in pdocs]
        labels = [d["culture"] for d in pdocs]
        key = (tuple(df["vessel_id"]), self.g, self.n_per_class, self.remove_top, self.seed,
               self.refine_sweeps, self.label)
        if key not in SBMPipeline._cache:
            if len(SBMPipeline._cache) >= 50:  # keep memory bounded
                SBMPipeline._cache.pop(next(iter(SBMPipeline._cache)))
            SBMPipeline._cache[key] = sbm.fit_hsbm(tokens, seed=self.seed, refine_sweeps=self.refine_sweeps)
        model = SBMPipeline._cache[key]
        cls = sbm.SBMClassifier if self.method == "clusters" else sbm.SBMFoldInClassifier
        self.clf = cls(self.seed, self.refine_sweeps).fit(tokens, labels, self.classes, model=model)
        return self

    def predict_proba(self, df):
        drop = set(self.removed)
        return self.clf.predict_proba([[t for t in d if t not in drop] for d in documents(df)])
