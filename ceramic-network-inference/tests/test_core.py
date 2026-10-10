"""Core checks. Run with `pytest tests/` or, without pytest, `python tests/test_core.py`."""
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chaine import data, evaluation as ev, markov, topic_model as tm  # noqa: E402

DF = data.load_vessels()
GRAMMAR = data.load_grammar(extra_chains=DF["steps"])
# legacy clustered_paths_*.csv of the original project, if present next to this repository
LEGACY_DATA = next((p for p in (ROOT.parent / "Old_code" / "data", ROOT.parent / "data") if p.is_dir()), ROOT.parent / "data")


def test_dataset_shape():
    assert len(DF) == 276
    assert DF["bok"].value_counts().to_dict() == {"MI": 95, "CW": 89, "EFG": 62, "LFG": 30}
    assert DF["chain"].nunique() == 101
    assert DF["vessel_id"].is_unique


def test_every_chain_is_grammatical():
    assert all(GRAMMAR.is_valid(s) for s in DF["steps"])


def test_triplets_match_legacy_clustered_paths():
    """The triplet 'words' equal the legacy clustered_paths_*.csv files (if available)."""
    for bok in data.BOKS:
        path = LEGACY_DATA / f"clustered_paths_{bok}.csv"
        if not path.exists():
            continue
        legacy = sorted(
            tuple(re.findall(r"\('([^']+)', '([^']+)', '([^']+)'\)", line))
            for line in path.read_text().splitlines() if line.strip(";").strip()
        )
        ours = sorted(tuple(data.ngrams(s, 3)) for s in DF.loc[DF["bok"] == bok, "steps"])
        assert ours == legacy, bok


def test_markov_is_normalised():
    for order in (1, 2):
        m = markov.MarkovChain(order, beta=2.0, grammar=GRAMMAR).fit(DF["steps"])
        _, mass = markov.enumerate_paths(m, min_prob=1e-8)
        assert 0.999 < mass <= 1.0 + 1e-9, (order, mass)


def test_completion_returns_valid_chains_in_order():
    m = markov.MarkovChain(1, beta=2.0, grammar=GRAMMAR).fit(DF["steps"])
    s = DF["steps"].iloc[0]
    fills = m.complete(s[:5], s[7:], k=5)
    scores = [lp for _, lp in fills]
    assert scores == sorted(scores, reverse=True)
    assert all(GRAMMAR.is_valid(s[:5] + f + s[7:]) for f, _ in fills)


def test_folds_do_not_leak():
    for scheme in ("stratified", "site"):
        for tr, te in ev.splits(DF, scheme, 5, seed=0):
            assert not set(DF["vessel_id"].iloc[tr]) & set(DF["vessel_id"].iloc[te])
            if scheme == "site":
                assert not set(DF["site_id"].iloc[tr]) & set(DF["site_id"].iloc[te])


def test_pseudo_documents_use_training_vessels_only():
    tr, te = ev.splits(DF, "stratified", 5, seed=0)[0]
    pdocs = tm.create_resampled_documents(tm.load_pottery_data(DF.iloc[tr]), 7, 20, seed=0)
    used = {v for d in pdocs for v in d["original_vessels"]}
    assert used <= set(DF["vessel_id"].iloc[tr])
    assert not used & set(DF["vessel_id"].iloc[te])
    assert sorted({d["culture"] for d in pdocs}) == sorted(data.BOKS)


def test_stop_triplets_are_learned_on_training_only():
    tr, te = ev.splits(DF, "stratified", 5, seed=0)[0]
    docs = tm.load_pottery_data(DF)
    train = [docs[i] for i in tr]
    _, removed = tm.remove_common_triplets(train, 5)
    test_c, removed_again = tm.remove_common_triplets([docs[i] for i in te], 5, fit_docs=train)
    assert removed == removed_again
    assert all(t not in d["triplets"] for d in test_c for t in removed)
    assert all(docs[i]["triplets"] for i in te)  # originals untouched


def test_hungarian_matching_is_one_to_one():
    rng = np.random.default_rng(0)
    emp, phi = rng.dirichlet(np.ones(10), 4), rng.dirichlet(np.ones(10), 4)
    mapping, _ = tm.matching_empirical(emp, phi, data.BOKS)
    assert sorted(mapping.values()) == [0, 1, 2, 3]


def test_classify_new_vessel_returns_a_culture():
    docs = tm.load_pottery_data(DF)
    vocab, _ = tm.extract_vocabulary(docs)
    pdocs = tm.create_resampled_documents(docs, 3, 10, seed=0)
    X, _ = tm.build_dtm_with_external_vocab(pdocs, {w: i for i, w in enumerate(vocab)})
    model = tm.LDA_manual(4, 1, 1, n_iter=5, random_state=0).fit(X, vocab)
    emp = tm.get_empirical_word_distribution(docs, vocab, data.BOKS)
    mapping, _ = tm.matching_empirical(emp, model.topic_word_dist, data.BOKS)
    assert tm.classify_new_vessel(docs[0]["triplets"], model, vocab, mapping)["predicted_culture"] in data.BOKS


def test_hsbm_is_reproducible():
    from chaine import sbm
    if not sbm.graph_tool_available():
        return
    docs = data.documents(DF)
    fits = [sbm.fit_hsbm(docs, seed=0) for _ in range(2)]  # seed 0 must not mean "random"
    assert fits[0].mdl == fits[1].mdl
    assert (sbm.doc_partition(fits[0]) == sbm.doc_partition(fits[1])).all()


def test_hsbm_fold_in_matches_full_description_length():
    """virtual_vertex_move on level 0 must equal rebuilding the nested state for each group."""
    from chaine import sbm
    if not sbm.graph_tool_available():
        return
    pdocs = tm.create_resampled_documents(tm.load_pottery_data(DF), 3, 20, seed=0)
    model = sbm.fit_hsbm([d["triplets"] for d in pdocs], seed=1, refine_sweeps=5)
    new = data.documents(DF.iloc[:3])
    for (fast, _), (exact, _) in zip(sbm._fold_in_delta_s(model, new),
                                    sbm._fold_in_delta_s(model, new, exact=True)):
        assert np.allclose(fast, exact, atol=1e-6)


def test_bayes_ceiling():
    assert abs(data.bayes_ceiling(DF, "bok") - 0.732) < 1e-3
    assert abs(data.bayes_ceiling(DF, "culture") - 0.888) < 1e-3


if __name__ == "__main__":
    tests = [(k, v) for k, v in dict(globals()).items() if k.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} tests passed")
