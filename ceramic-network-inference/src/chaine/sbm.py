"""Hierarchical stochastic block model (hSBM) topic model with sbmtm and graph-tool.

Gerlach, Peixoto & Altmann (2018): documents and words are the two node types
of a bipartite multigraph (one edge per token). A nested, degree-corrected SBM
is fitted by minimising the description length (MDL)

    S = -log P(A, b) = -log P(A | b) - log P(b),

which sets the number of document groups and word groups ("topics") at every
level. Beyond `sbmtm.fit`, this module follows graph-tool's recommended
workflow (Peixoto 2014, 2020): refine the partition with merge-split MCMC at
zero temperature, sample the posterior over partitions for uncertainty, and
place a new document by the change of description length it causes.

The legacy SBM functions of `Presentation_topic_modelling.ipynb` are kept under
their names where they are used (compute_word_cluster_matrix,
classify_vessel_simple, infer_new_doc_membership), with the corrections noted.
Requires graph-tool (conda-forge only).
"""
import contextlib
import io
from collections import Counter

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def graph_tool_available():
    try:
        import graph_tool.all  # noqa: F401
        return True
    except ImportError:
        return False


# ------------------------------------------------------------------ fitting
def fit_hsbm(docs, names=None, seed=32, overlap=False, n_init=1, refine_sweeps=0):
    """Fit sbmtm on token lists; single-threaded and seeded for reproducibility.

    n_init > 1 keeps the fit with the lowest description length. refine_sweeps
    > 0 then runs zero-temperature merge-split sweeps (see `refine`).
    """
    import graph_tool.all as gt
    from .sbmtm import sbmtm

    gt.openmp_set_num_threads(1)
    gt.seed_rng(seed + 1)  # graph-tool treats seed 0 as "use system entropy"
    np.random.seed(seed)
    model = sbmtm()
    model.make_graph(docs, documents=names or [str(i) for i in range(len(docs))], counts=not overlap)
    with contextlib.redirect_stdout(io.StringIO()):  # sbmtm prints the state
        model.fit(overlap=overlap, n_init=n_init)
    model.mdl_initial = model.mdl
    if refine_sweeps and not overlap:
        refine(model, refine_sweeps)
    return model


def refine(model, sweeps=50, niter=10):
    """Zero-temperature merge-split MCMC on the fitted nested state (in place).

    minimize_nested_blockmodel_dl returns a good but not optimal partition, and
    sbmtm then truncates the hierarchy; a few sweeps at beta = inf usually lower
    the description length further. Returns (S before, S after) in nats.
    """
    before = model.state.entropy()
    for _ in range(sweeps):
        model.state.multiflip_mcmc_sweep(beta=np.inf, niter=niter)
    model.mdl = model.state.entropy()
    return before, model.mdl


def sample_partitions(model, n_samples=200, niter=10):
    """Sample the posterior P(b | A) over hierarchical partitions (MCMC at beta = 1).

    Returns a graph-tool PartitionModeState, which aligns group labels across
    samples, and the description length of each sample.
    """
    import graph_tool.all as gt

    state = model.state.copy(sampling=True)
    bs, dl = [], []

    def collect(s):
        bs.append(s.get_bs())
        dl.append(s.entropy())

    gt.mcmc_equilibrate(state, force_niter=n_samples, mcmc_args=dict(niter=niter), callback=collect)
    return gt.PartitionModeState(bs, nested=True, converge=True), np.array(dl)


def posterior_marginals(model, pmode):
    """Posterior probability of each level-0 group for every document and word node.

    Returns (doc_marginals D x Bd, word_marginals V x Bw); rows sum to one.
    These are the "soft memberships" of a non-overlapping SBM: uncertainty about
    the group of a node, not mixed membership.
    """
    pv = pmode.get_marginal(model.g)
    D, V = model.get_D(), model.get_V()
    width = max(len(pv[v]) for v in model.g.vertices())
    M = np.zeros((D + V, width))
    for i, v in enumerate(model.g.vertices()):
        a = np.array(pv[v], float)
        M[i, :len(a)] = a
    M /= M.sum(1, keepdims=True)
    docs, words = M[:D], M[D:]
    return docs[:, docs.sum(0) > 0], words[:, words.sum(0) > 0]


# --------------------------------------------------------------- summaries
def hierarchy(model):
    """Number of document and word groups at each level (level 0 = finest)."""
    out = []
    for level in range(len(model.state.levels)):
        g = model.get_groups(level)
        out.append({"level": level, "doc_groups": g["Bd"], "word_groups": g["Bw"]})
        if g["Bd"] == 1:
            break
    return out


def doc_partition(model, level=0):
    return model.get_groups(level)["p_td_d"].argmax(0)


def word_partition(model, level=0):
    return model.get_groups(level)["p_tw_w"].argmax(0)


def agreement(groups, labels):
    return {"NMI": normalized_mutual_info_score(labels, groups),
            "ARI": adjusted_rand_score(labels, groups)}


def culture_topic_mixture(model, doc_labels, cultures, level=0):
    """Topic weights of each culture and the word distribution they imply.

    p(t | c) = mean over documents d of culture c of p(t | d)   (sbmtm p_tw_d)
    p(w | c) = sum_t p(w | t) p(t | c)                          (sbmtm p_w_tw)

    This is the model's own answer to legacy cells 64-65 ("combination of
    topics"). Those cells averaged rows of p(t | w) - group *memberships*, which
    for a non-overlapping SBM are 0/1 indicators - and compared them with a
    binarised empirical matrix, i.e. sets of words rather than distributions,
    chosen by a greedy search.
    """
    G = model.get_groups(level)
    labels = np.asarray(doc_labels)
    p_t_c = np.array([G["p_tw_d"][:, labels == c].mean(1) for c in cultures])
    return p_t_c, p_t_c @ G["p_w_tw"].T


def compute_word_cluster_matrix(doc_groups, docs, vocab, doc_labels=None):
    """Legacy cell 77: word counts per document cluster, and cluster names such as
    'MI (C2, 58/61)' (majority culture, cluster id, majority / size)."""
    index = {w: i for i, w in enumerate(vocab)}
    n_clusters = int(doc_groups.max()) + 1
    counts = np.zeros((len(vocab), n_clusters))
    for g, doc in zip(doc_groups, docs):
        for w, c in Counter(doc).items():
            if w in index:
                counts[index[w], g] += c
    names = []
    for k in range(n_clusters):
        members = Counter(y for y, g in zip(doc_labels, doc_groups) if g == k) if doc_labels is not None else Counter()
        if members:
            top, n = members.most_common(1)[0]
            names.append(f"{top} (C{k}, {n}/{sum(members.values())})")
        else:
            names.append(f"C{k}")
    return counts, names


def classify_vessel_simple(vessel_triplets, counts_matrix, vocab, smoothing=1e-6):
    """Legacy cell 84: posterior over document clusters of one vessel,
    multinomial likelihood of its triplets under each cluster's word counts
    (uniform prior over clusters). Triplets outside the vocabulary are ignored.

    Corrections: in the legacy notebook the counts came from pseudo-documents
    that contained the evaluated vessels, and the evaluation ran on all vessels
    (in-sample); here it is only ever applied to held-out vessels. The printed
    'true culture: EFG' of the legacy test vessel was a CW vessel.
    """
    index = {w: i for i, w in enumerate(vocab)}
    p = counts_matrix + smoothing
    log_p = np.log(p / p.sum(0, keepdims=True))
    ll = np.zeros(counts_matrix.shape[1])
    for w, c in Counter(vessel_triplets).items():
        if w in index:
            ll += c * log_p[index[w]]
    post = np.exp(ll - ll.max())
    return post / post.sum()


def infer_new_doc_membership(model, new_docs):
    """Legacy cell 86, corrected: place new documents in the fitted hSBM.

    For each new document, add its node and edges to a copy of the graph and
    compute the description length S_r of the whole hierarchy with the new node
    in each level-0 document group r, all other nodes fixed. Then

        P(r | new document, A, b) ∝ exp(-(S_r - min_r S_r)),

    the conditional posterior that graph-tool's MCMC samples from.

    The differences S_r - S_r0 come from `virtual_vertex_move` on level 0 of the
    nested state, which accounts for the whole hierarchy (tests check that it
    equals rebuilding the state for every r).

    The legacy version added the vertex to model.g in place (corrupting the
    fitted model), copied a projected state of the wrong size, and read
    collect_vertex_marginals incorrectly; it was never run.

    Returns (P: n_docs x n_groups, raw group labels).
    """
    P, groups = [], None
    for dS, groups in _fold_in_delta_s(model, new_docs):
        p = np.exp(-(dS - dS.min()))
        P.append(p / p.sum())
    return np.array(P), groups


def _fold_in_delta_s(model, new_docs, exact=False):
    """Yield (S_r - S_r0 for every level-0 document group r, group labels) per document."""
    import graph_tool.all as gt

    g, D = model.g, model.get_D()
    word_vertex = {w: D + i for i, w in enumerate(model.words)}
    bs = [np.asarray(b) for b in model.state.get_bs()]
    groups = np.unique(bs[0][:D])
    for doc in new_docs:
        known = Counter(w for w in doc if w in word_vertex)
        if not known:
            yield np.zeros(len(groups)), groups
            continue
        g2 = g.copy()
        kind, count = g2.vp["kind"], g2.ep["count"]
        v = g2.add_vertex()
        kind[v] = 0
        for w, c in known.items():
            count[g2.add_edge(v, g2.vertex(word_vertex[w]))] = c
        make = lambda r: gt.NestedBlockState(g2, bs=[np.append(bs[0], r)] + bs[1:],
                                             state_args=dict(clabel=kind, pclabel=kind, eweight=count))
        if exact:
            S = np.array([make(r).entropy() for r in groups])
            yield S - S[0], groups
        else:
            state = make(groups[0])
            yield np.array([0.0] + [state.levels[0].virtual_vertex_move(int(v), int(r))
                                    for r in groups[1:]]), groups


# ------------------------------------------------------------ classifiers
class SBMClassifier:
    """Legacy SBM classifier (cells 77, 84): name each level-0 document cluster
    after its majority culture; score a new vessel with classify_vessel_simple;
    sum the cluster posteriors by cluster name."""

    def __init__(self, seed=32, refine_sweeps=50):
        self.seed, self.refine_sweeps = seed, refine_sweeps

    def fit(self, docs, labels, classes, model=None):
        self.classes = list(classes)
        self.model = model or fit_hsbm(docs, seed=self.seed, refine_sweeps=self.refine_sweeps)
        order = {name: i for i, name in enumerate(self.model.documents)}
        groups = doc_partition(self.model)[[order[str(i)] for i in range(len(docs))]]
        self.vocab = list(self.model.words)
        self.counts, self.names = compute_word_cluster_matrix(groups, docs, self.vocab, labels)
        self.cluster_label = [n.split()[0] for n in self.names]
        self.groups = groups
        return self

    def predict_proba(self, docs):
        out = np.zeros((len(docs), len(self.classes)))
        for d, doc in enumerate(docs):
            for k, p in enumerate(classify_vessel_simple(doc, self.counts, self.vocab)):
                out[d, self.classes.index(self.cluster_label[k])] += p
        return out


class SBMFoldInClassifier:
    """Attribution by description length (Peixoto): fold each new vessel into the
    fitted hSBM (infer_new_doc_membership), then
    p(culture | vessel) = sum_r P(r | vessel) p(culture | r),
    with p(culture | r) the share of each culture among the training documents
    in group r."""

    def __init__(self, seed=32, refine_sweeps=50):
        self.seed, self.refine_sweeps = seed, refine_sweeps

    def fit(self, docs, labels, classes, model=None):
        self.classes = list(classes)
        self.model = model or fit_hsbm(docs, seed=self.seed, refine_sweeps=self.refine_sweeps)
        D = self.model.get_D()
        order = {name: i for i, name in enumerate(self.model.documents)}
        b0 = np.asarray(self.model.state.get_bs()[0])[:D]
        b0 = b0[[order[str(i)] for i in range(len(docs))]]
        self.groups = np.unique(b0)
        M = np.zeros((len(self.groups), len(self.classes)))
        for r, y in zip(b0, labels):
            M[np.searchsorted(self.groups, r), self.classes.index(y)] += 1
        self.p_class_given_group = M / M.sum(1, keepdims=True)
        return self

    def predict_proba(self, docs):
        P, groups = infer_new_doc_membership(self.model, docs)
        assert (groups == self.groups).all()
        return P @ self.p_class_given_group
