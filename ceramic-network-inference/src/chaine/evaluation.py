"""Honest evaluation: every vessel is predicted by a model that never saw it.

Two splitting schemes:
- 'stratified': stratified K-fold over vessels;
- 'site': stratified group K-fold with sites as groups, so all vessels of a
  site are held out together (vessels from one grave/cemetery are not
  independent draws).
"""
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from .data import BOKS


def splits(df, scheme="stratified", n_splits=5, seed=0, label="bok"):
    y = df[label].astype(str).to_numpy()
    if scheme == "stratified":
        return list(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(df, y))
    if scheme == "site":
        cv = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed)
        return list(cv.split(df, y, groups=df["site_id"].to_numpy()))
    raise ValueError(scheme)


def cross_val_proba(df, make_model, scheme="stratified", n_splits=5, seed=0, classes=BOKS,
                    label="bok"):
    """Out-of-fold class probabilities for every vessel."""
    proba = np.zeros((len(df), len(classes)))
    for train, test in splits(df, scheme, n_splits, seed, label):
        model = make_model().fit(df.iloc[train])
        proba[test] = model.predict_proba(df.iloc[test])
    return proba


def scores(y_true, proba, classes=BOKS):
    """Accuracy, balanced accuracy, macro-F1, log-loss, Brier score and ECE."""
    y = np.array([classes.index(c) for c in y_true])
    pred = proba.argmax(1)
    p_true = np.clip(proba[np.arange(len(y)), y], 1e-6, 1)
    onehot = np.eye(len(classes))[y]
    conf = proba.max(1)
    bins = np.minimum((conf * 10).astype(int), 9)
    ece = sum(abs((pred[bins == b] == y[bins == b]).mean() - conf[bins == b].mean()) * (bins == b).mean()
              for b in range(10) if (bins == b).any())
    return {
        "accuracy": accuracy_score(y, pred),
        "balanced_acc": balanced_accuracy_score(y, pred),
        "macro_F1": f1_score(y, pred, average="macro"),
        "log_loss": -np.log(p_true).mean(),
        "brier": ((proba - onehot) ** 2).sum(1).mean(),
        "ECE": ece,
    }


def confusion(y_true, proba, classes=BOKS, normalize=True):
    y = [classes.index(c) for c in y_true]
    cm = confusion_matrix(y, proba.argmax(1), labels=range(len(classes)))
    return cm / cm.sum(1, keepdims=True) if normalize else cm


class Results:
    """Registry of every evaluation in the notebook, so that all models are
    scored with the same metrics and shown in the same format."""

    def __init__(self, classes=BOKS):
        self.classes, self.rows, self.proba = list(classes), [], {}

    def add(self, model, protocol, y_true, proba):
        y_true = np.asarray(y_true)
        row = {"model": model, "protocol": protocol, "n": len(y_true), **scores(y_true, proba, self.classes)}
        self.rows = [r for r in self.rows if (r["model"], r["protocol"]) != (model, protocol)] + [row]
        self.proba[(model, protocol)] = (y_true, proba)
        return pd.DataFrame([row]).set_index(["model", "protocol"])

    def table(self, protocol=None, models=None):
        t = pd.DataFrame(self.rows)
        if protocol is not None:
            t = t[t["protocol"] == protocol]
        if models is not None:
            t = t.set_index("model").loc[list(models)].reset_index()
        return t.set_index(["model", "protocol"])

    def confusion(self, model, protocol):
        y, p = self.proba[(model, protocol)]
        return confusion(y, p, self.classes)


def compare(df, models, schemes=("stratified", "site"), n_splits=5, seed=0, classes=BOKS,
            label="bok", verbose=True):
    """Run every model under every scheme. Returns (table, {(name, scheme): proba})."""
    rows, probas = [], {}
    y = df[label].astype(str).to_numpy()
    for scheme in schemes:
        for name, make_model in models.items():
            t0 = time.time()
            proba = cross_val_proba(df, make_model, scheme, n_splits, seed, classes, label)
            probas[(name, scheme)] = proba
            rows.append({"model": name, "split": scheme, **scores(y, proba, classes)})
            if verbose:
                print(f"{scheme:>10} | {name:<28} acc={rows[-1]['accuracy']:.3f} "
                      f"({time.time() - t0:.0f}s)")
    return pd.DataFrame(rows), probas
