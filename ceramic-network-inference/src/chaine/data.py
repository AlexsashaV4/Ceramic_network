"""Loading chaînes opératoires, vessel metadata and Kroon's technique grammar.

A chaîne opératoire (chain) is the ordered list of techniques used to make one
vessel, e.g. ``Wet clay > Modelling > Coiling > ... > Open firing``.
"""
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "clean"

# Bodies of knowledge (BoK): Corded Ware and three Funnel Beaker West phases.
BOKS = ["CW", "EFG", "MI", "LFG"]
BOK_NAMES = {
    "CW": "Corded Ware",
    "EFG": "FBW early",
    "MI": "FBW middle",
    "LFG": "FBW late",
}
CULTURE = {"CW": "CW", "EFG": "FBW", "MI": "FBW", "LFG": "FBW"}

START, END = "<s>", "</s>"
HYDRIC_ORDER = "WLDF"  # wet, leather-hard, dry, fired
# techniques that move the vessel to the next hydric state
STATE_CHANGE = {"Leather-hard": "L", "Dry": "D", "Open firing": "F", "Enclosed firing": "F"}


def load_vessels(data_dir=DATA_DIR):
    """One row per vessel: chain (list of steps), BoK, culture, site and vessel type."""
    data_dir = Path(data_dir)
    chains = pd.read_csv(data_dir / "chains.csv", dtype={"vessel_id": str})
    meta = pd.read_csv(data_dir / "vessels.csv", dtype={"vessel_id": str, "site_id": str})
    df = chains.merge(meta, on="vessel_id", validate="one_to_one")
    df["steps"] = df["chain"].str.split(" > ")
    df["culture"] = df["bok"].map(CULTURE)
    df["bok"] = pd.Categorical(df["bok"], categories=BOKS, ordered=True)
    return df.sort_values(["bok", "vessel_id"], key=_natural).reset_index(drop=True)


def _natural(col):
    return col.astype(int) if col.name == "vessel_id" else col


def ngrams(steps, n=3):
    """Sliding-window n-grams, e.g. triplets (n=3) as used in the legacy notebooks."""
    return [tuple(steps[i:i + n]) for i in range(len(steps) - n + 1)]


def token(gram):
    """String form of an n-gram, identical to the legacy '(A, B, C)' tokens."""
    return "(" + ", ".join(gram) + ")"


def documents(df, n=3):
    """Bag-of-n-gram documents: one list of string tokens per vessel."""
    return [[token(g) for g in ngrams(s, n)] for s in df["steps"]]


def bayes_ceiling(df, label="bok"):
    """Best accuracy any chain-only classifier can reach on these data (in-sample).

    Vessels with identical chains are indistinguishable, so the best possible
    rule predicts the majority label of each distinct chain.
    """
    by_chain = defaultdict(Counter)
    for steps, y in zip(df["steps"], df[label]):
        by_chain[tuple(steps)][y] += 1
    return sum(max(c.values()) for c in by_chain.values()) / len(df)


class Grammar:
    """Kroon's syntax for valid chains (Kroon 2024, section 4.2; Pathfinder README).

    A step is allowed if (1) the link previous -> next exists in the technique
    network, (2) the technique has not been used yet (no loops), and (3) its
    hydric tolerance (W/L/D/F) contains the vessel's current state. A chain may
    only end once the vessel is fired.
    """

    def __init__(self, techniques, links):
        self.tolerance = dict(zip(techniques["technique"], techniques["tolerance"]))
        self.successors = defaultdict(set)
        for a, b in links:
            self.successors[a].add(b)

    def state(self, steps):
        s = "W"
        for x in steps:
            s = STATE_CHANGE.get(x, s)
        return s

    def allowed(self, steps, nxt, state=None):
        state = state or self.state(steps)
        if nxt == END:
            return bool(steps) and state == "F"
        if not steps:
            return nxt == "Wet clay"
        if nxt in steps or nxt not in self.successors[steps[-1]]:
            return False
        new_state = STATE_CHANGE.get(nxt)
        if new_state is not None:  # state-changing steps must move forward
            return HYDRIC_ORDER.index(new_state) > HYDRIC_ORDER.index(state)
        return state in self.tolerance.get(nxt, "").split("|")

    def is_valid(self, steps):
        return all(self.allowed(steps[:i], steps[i]) for i in range(len(steps))) and \
            self.allowed(steps, END)

    def candidates(self, steps, vocab):
        state = self.state(steps)
        return [x for x in list(vocab) + [END] if self.allowed(steps, x, state)]


def load_grammar(data_dir=DATA_DIR, extra_chains=()):
    """Grammar from Kroon's link list, extended with transitions seen in the data.

    One observed transition (Application separate element -> Scraping) is absent
    from Kroon's link list; passing the chains here adds it.
    """
    data_dir = Path(data_dir)
    techniques = pd.read_csv(data_dir / "techniques.csv")
    links = pd.read_csv(data_dir / "technique_links.csv")
    pairs = set(zip(links["source"], links["target"]))
    for steps in extra_chains:
        pairs |= set(zip(steps[:-1], steps[1:]))
    return Grammar(techniques, pairs)
