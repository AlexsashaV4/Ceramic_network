"""Radiocarbon dates of Kroon (2024, Appendix A) and their calibration with IntCal20.

A radiocarbon age a ± e (years BP) is calibrated on a grid of calendar years t:

    p(t | a) ∝ N(a; mu(t), e² + s(t)²),

with mu(t) ± s(t) the IntCal20 curve (Reimer et al. 2020), linearly interpolated.
Calendar years are given as positive numbers BCE; cal BP = BCE + 1949.
This is single-date calibration, as in OxCal's R_Date; it is not a Bayesian
phase model. Phase boundaries are quoted from Kroon's OxCal models.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE = Path(__file__).resolve().parents[2] / "data" / "source"
INTCAL = SOURCE / "intcal20" / "intcal20.14c"
DATES = SOURCE / "dans" / "Appendices" / "Appendix_A_Radiocarbon_dates" / "Appendix_A_Radiocarbon_Dates.csv"

# Group of every date in Kroon's Appendix A, Bayesian models 3 (CW) and 4 (early and late flat graves)
KROON_GROUPS = {
    "EFG": ["GrN-5103", "GrA-39456", "GrN-16492", "GrN-16494", "Poz-71757", "GrN-2221", "GrN-1824C", "GrN-18785",
            "GrN-12262", "GrN-12263", "GrN-11763", "GrN-11764", "GrN-11765", "GrN-11766", "GrN-9209", "Beta-421914"],
    "LFG": ["R-combine", "GrA-13599", "GrA-13705", "GrA-13600", "GrA-13706", "GrA-13958", "GrA-14086", "GrA-14088",
            "GrA-14089", "GrA-14090", "GrA-14095", "GrA-14168", "GrA-14973", "GrA-15642", "GrA-15459", "GrA-16021",
            "GrM-26730", "GrN-16040", "Poz-68248", "Poz-68249", "Poz-88040", "GrA-41638", "GrA-41639", "GrN-4201",
            "GrN-5767", "GrN-6156", "KlA-23204", "KlA-23207"],
    "CW": ["GrN-851", "GrN-1855", "GrN-11918", "GrA-14965", "GrN-7518", "GrN-6129", "GrN-6368", "GrN-6349",
           "GrA-41646", "SUERC-26360", "GrN-6126", "GrN-8247", "GrN-6724", "UtC-1945", "UtC-1949", "GrA-56014",
           "GrN-56013", "GrA-12299"],
}
# Modelled boundaries (95 %, cal BCE) stated in the text of Appendix A (model 3, used as priors in model 4)
KROON_BOUNDARIES = {"start FBW": (3802, 3645), "end FBW": (2761, 2581)}


def lab_key(lab):
    """'Poz 71757', 'Poz-71757' -> 'POZ71757'."""
    return re.sub(r"[^A-Za-z0-9]", "", str(lab)).upper()


def load_curve(path=INTCAL):
    """IntCal20 as a DataFrame (cal_bp, c14, sigma), sorted by cal BP."""
    curve = pd.read_csv(path, comment="#", header=None, usecols=[0, 1, 2], names=["cal_bp", "c14", "sigma"])
    return curve.sort_values("cal_bp").reset_index(drop=True)


def calibrate(age, error, years, curve=None):
    """Calibrated probability of each calendar year in `years` (BCE), summing to 1."""
    curve = load_curve() if curve is None else curve
    bp = np.asarray(years, float) + 1949
    mu = np.interp(bp, curve["cal_bp"], curve["c14"])
    s2 = np.interp(bp, curve["cal_bp"], curve["sigma"]) ** 2 + error ** 2
    p = np.exp(-0.5 * (age - mu) ** 2 / s2) / np.sqrt(s2)
    return p / p.sum()


def interval(years, p, mass=0.95):
    """Central interval of a probability vector on `years` (BCE): (older, younger)."""
    order = np.argsort(-np.asarray(years))                      # oldest first
    c = np.cumsum(np.asarray(p)[order])
    y = np.asarray(years)[order]
    return y[np.searchsorted(c, (1 - mass) / 2)], y[np.searchsorted(c, (1 + mass) / 2)]


def load_dates(path=DATES):
    """The 62 dates of Appendix A with Kroon's group (EFG, LFG or CW) from his models 3 and 4."""
    raw = pd.read_csv(path, sep=";", encoding="latin-1")
    group = {lab_key(lab): g for g, labs in KROON_GROUPS.items() for lab in labs}
    out = pd.DataFrame({"lab": raw["Laboratory_number"], "age": raw["Radiocarbon age (years BP)"].astype(float),
                        "error": raw["Error"].astype(float), "site": raw["Site"].str.strip(),
                        "culture": raw["Group"], "typochronology": raw["Typochronology"].astype(str),
                        "material": raw["Dated material"], "context": raw["Context"]})
    out["kroon_group"] = out["lab"].map(lambda lab: group.get(lab_key(lab)))
    return out
