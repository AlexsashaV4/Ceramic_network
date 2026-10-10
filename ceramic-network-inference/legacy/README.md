# Legacy notebooks (provenance only)

The two exploratory notebooks this project grew from, copied with outputs stripped (the originals, with outputs, are 32 MB and 65 MB). They are kept for reference and are **not maintained or runnable as-is**: paths assume the original project layout, cells were executed out of order, and some depend on `gravis` and files not included here. In the local working folder the originals, with their `data/`, `functions.py` and `sbmtm.py`, are in `Old_code/` next to this repository; their paths resolve when they are run from there.

| file | content | replicated in the main notebook |
|---|---|---|
| `Presentation_topic_modelling.ipynb` | LDA from scratch and with gensim on bootstrap pseudo-documents; hierarchical SBM topic model (`sbmtm`); SBM classifiers | Sections 4–7, step by step, with the same function names in `src/chaine/topic_model.py` and `src/chaine/sbm.py` (each docstring cites the legacy cell and lists corrections) |
| `presentation_notebook.ipynb` | technique networks, triplet ("clustered") graphs, path probabilities, random walks, centralities, word2vec embeddings of walks | Section 10 |
| `functions.py` | helper functions used by `presentation_notebook.ipynb` | `src/chaine/` |

Section 11 of the main notebook lists every issue found in these notebooks and how it is addressed.
