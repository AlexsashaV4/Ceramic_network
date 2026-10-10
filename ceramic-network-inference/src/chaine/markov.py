"""k-th order Markov models of chaînes opératoires.

p(x_1..x_T, END) = prod_t p(x_t | x_{t-k}..x_{t-1}), with START padding.

Order 1 is the model behind Kroon's edge weights, p(x_{t+1} | x_t); order 2 is
the "triplet graph" of the legacy notebooks, p(x_{t+1} | x_{t-1}, x_t).

Small data: probabilities are smoothed with a hierarchical Dirichlet prior,
each order shrinking toward the next lower one (MacKay & Peto 1995):

    p_k(x | h) = (c(h, x) + beta * p_{k-1}(x | h')) / (c(h) + beta),

where h' drops the oldest step of h, and p_0 is add-alpha smoothed. Unseen
contexts therefore fall back to shorter ones instead of to uniform.
"""
import heapq
import math
from collections import Counter, defaultdict

import numpy as np

from .data import END, START


class MarkovChain:
    def __init__(self, order=1, beta=1.0, alpha=0.5, grammar=None, vocab=None):
        self.order = order
        self.beta = beta
        self.alpha = alpha
        self.grammar = grammar
        self.vocab = sorted(vocab) if vocab is not None else None

    # ------------------------------------------------------------------ fit
    def fit(self, chains):
        chains = [list(c) for c in chains]
        if self.vocab is None:
            self.vocab = sorted({x for c in chains for x in c})
        self.symbols = self.vocab + [END]
        # counts[j][context of length j][next symbol]
        self.counts = [defaultdict(Counter) for _ in range(self.order + 1)]
        for c in chains:
            padded = [START] * self.order + c + [END]
            for t in range(self.order, len(padded)):
                for j in range(self.order + 1):
                    ctx = tuple(padded[t - j:t])
                    self.counts[j][ctx][padded[t]] += 1
        self._cache = {}
        return self

    def _p(self, ctx, x):
        """Smoothed p(x | ctx) before grammar restriction."""
        j = len(ctx)
        c = self.counts[j].get(ctx)
        if j == 0:
            n = sum(c.values()) if c else 0
            return ((c[x] if c else 0) + self.alpha) / (n + self.alpha * len(self.symbols))
        lower = self._p(ctx[1:], x)
        if not c:
            return lower
        return (c[x] + self.beta * lower) / (sum(c.values()) + self.beta)

    def _context(self, steps):
        padded = [START] * self.order + list(steps)
        return tuple(padded[len(padded) - self.order:]) if self.order else ()

    def dist(self, steps):
        """Distribution of the next symbol (technique or END) given the steps so far."""
        key = tuple(steps) if self.grammar is not None else self._context(steps)
        if key in self._cache:
            return self._cache[key]
        ctx = self._context(steps)
        if self.grammar is not None:
            cand = self.grammar.candidates(list(steps), self.vocab)
        else:
            cand = self.symbols
        w = {x: self._p(ctx, x) for x in cand}
        z = sum(w.values())
        d = {x: v / z for x, v in w.items()} if z > 0 else {}
        self._cache[key] = d
        return d

    # ------------------------------------------------------------ scoring
    def logprob(self, steps, complete=True):
        """log p(chain) (natural log). complete=True includes the END symbol;
        complete=False gives the probability of an unfinished chain (a prefix),
        i.e. marginalised over all its possible continuations."""
        lp = 0.0
        steps = list(steps)
        for t in range(len(steps) + int(complete)):
            x = steps[t] if t < len(steps) else END
            p = self.dist(steps[:t]).get(x, 0.0)
            if p <= 0:
                return -math.inf
            lp += math.log(p)
        return lp

    # ----------------------------------------------------------- sampling
    def sample(self, rng, max_len=25, max_tries=100):
        """Draw one chain; dead ends (no allowed continuation) are resampled."""
        for _ in range(max_tries):
            steps = []
            while len(steps) < max_len:
                d = self.dist(steps)
                if not d:
                    break
                keys = list(d)
                x = keys[rng.choice(len(keys), p=np.fromiter(d.values(), float))]
                if x == END:
                    return steps
                steps.append(x)
        raise RuntimeError("could not sample a complete chain")

    # --------------------------------------------------------- completion
    def complete(self, prefix, suffix=(), k=3, max_len=25, max_expansions=200_000):
        """Top-k most probable completions prefix + gap + suffix + END.

        Uniform-cost search over the missing steps ('gap'), exact because every
        added step can only increase the cost -log p. With an empty suffix this
        completes an unfinished chain; with a suffix it fills a gap. Returns
        [(gap, log p(gap, suffix, END | prefix)), ...].
        """
        prefix, suffix = list(prefix), list(suffix)
        heap = [(0.0, 0, (), False)]
        tie, found, expanded = 1, [], 0
        while heap and len(found) < k and expanded < max_expansions:
            cost, _, gap, terminal = heapq.heappop(heap)
            if terminal:
                found.append((list(gap), -cost))
                continue
            expanded += 1
            seq = prefix + list(gap)
            closing = self._closing_cost(seq, suffix)
            if closing is not None:
                heapq.heappush(heap, (cost + closing, tie, gap, True))
                tie += 1
            if len(seq) + len(suffix) >= max_len:
                continue
            for x, p in self.dist(seq).items():
                if x != END and p > 0:
                    heapq.heappush(heap, (cost - math.log(p), tie, gap + (x,), False))
                    tie += 1
        return found

    def _closing_cost(self, seq, suffix):
        cost, cur = 0.0, list(seq)
        for x in list(suffix) + [END]:
            p = self.dist(cur).get(x, 0.0)
            if p <= 0:
                return None
            cost -= math.log(p)
            cur.append(x)
        return cost


def enumerate_paths(model, max_len=25, min_prob=0.0):
    """All complete chains with probability > min_prob (depth-first).

    Returns (paths, total probability). With min_prob=0 and a grammar without
    dead ends the total is 1 - a check that the model is normalised.
    """
    out, stack = [], [([], 1.0)]
    while stack:
        steps, p = stack.pop()
        if len(steps) > max_len:
            continue
        for x, q in model.dist(steps).items():
            pq = p * q
            if pq <= min_prob:
                continue
            if x == END:
                out.append((steps, pq))
            else:
                stack.append((steps + [x], pq))
    return out, sum(p for _, p in out)


def fit_per_label(df, labels, order=1, beta=1.0, alpha=0.5, grammar=None, vocab=None, label="bok"):
    """One Markov model per label (BoK), sharing vocabulary and grammar."""
    vocab = vocab or sorted({x for s in df["steps"] for x in s})
    return {
        y: MarkovChain(order, beta, alpha, grammar, vocab).fit(df.loc[df[label] == y, "steps"])
        for y in labels
    }
