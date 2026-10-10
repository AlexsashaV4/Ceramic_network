"""Build the clean CSVs in data/clean/ from Kroon's (2024) DANS deposit.

The source files (CC-BY-4.0, doi:10.17026/AR/FAXIOW) are kept in
data/source/. Run with --download to fetch them again from DANS.

    python scripts/prepare_data.py [--download]

Only the Python standard library is used (the .xlsx is read as zipped XML),
so the script runs before the conda environment exists.
"""
import argparse
import csv
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "source"
CLEAN = ROOT / "data" / "clean"

DANS_FILES = {  # DANS Data Station Archaeology file ids
    "Appendix_C_Macroscopy_Paths.csv": 2887574,
    "Appendix_C_Macroscopic_Data.xlsx": 2671227,
    "Appendix_B_Sites_Table.csv": 2671234,
    "nodes_list.csv": 2671228,
    "link_list.csv": 2671220,
}
DANS_URL = "https://archaeology.datastations.nl/api/access/datafile/{}?format=original"
BOKS = ("CW", "EFG", "MI", "LFG")


def download():
    SOURCE.mkdir(parents=True, exist_ok=True)
    for name, file_id in DANS_FILES.items():
        print(f"downloading {name}")
        urllib.request.urlretrieve(DANS_URL.format(file_id), SOURCE / name)


def read_lines(path, encoding="utf-8"):
    return path.read_text(encoding=encoding).replace("\r\n", "\n").strip("\n").split("\n")


def read_xlsx(path):
    """Return the first sheet of an .xlsx as a list of dicts (header = first row)."""
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    tag = lambda t: "{%s}%s" % (ns["m"], t)
    with zipfile.ZipFile(path) as z:
        shared = [
            "".join(t.text or "" for t in si.iter(tag("t")))
            for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", ns)
        ]
        sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
    rows = []
    for row in sheet.iter(tag("row")):
        cells = {}
        for c in row.findall("m:c", ns):
            col = "".join(ch for ch in c.attrib["r"] if ch.isalpha())
            v = c.find("m:v", ns)
            if v is not None:
                cells[col] = shared[int(v.text)] if c.attrib.get("t") == "s" else v.text
        rows.append(cells)
    header = rows[0]
    return [{header[k]: r.get(k, "") for k in header} for r in rows[1:]]


def as_int(x):
    return str(int(float(x)))


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path.relative_to(ROOT)} ({len(rows)} rows)")


def main():
    CLEAN.mkdir(parents=True, exist_ok=True)

    # 1. techniques (nodes) and allowed transitions (links) of Kroon's network
    techniques = []
    for line in read_lines(SOURCE / "nodes_list.csv")[1:]:
        parts = line.split(";")
        techniques.append({
            "node_id": parts[0],
            "technique": parts[1],
            "tolerance": "|".join(p for p in parts[2:] if p),  # hydric states W/L/D/F
        })
    id2name = {t["node_id"]: t["technique"] for t in techniques}
    links = [
        {"source": id2name[s], "target": id2name[t]}
        for s, t, _ in (l.split(";") for l in read_lines(SOURCE / "link_list.csv")[1:])
    ]
    write_csv(CLEAN / "techniques.csv", techniques, ["node_id", "technique", "tolerance"])
    write_csv(CLEAN / "technique_links.csv", links, ["source", "target"])

    # 2. one chaîne opératoire per vessel
    chains = []
    for line in read_lines(SOURCE / "Appendix_C_Macroscopy_Paths.csv")[1:]:
        parts = line.split(";")
        steps = [p for p in parts[2:] if p]
        chains.append({"vessel_id": as_int(parts[0]), "bok": parts[1],
                       "n_steps": len(steps), "chain": " > ".join(steps)})
    write_csv(CLEAN / "chains.csv", chains, ["vessel_id", "bok", "n_steps", "chain"])

    # 3. vessel metadata joined with site metadata
    sites = {}
    with open(SOURCE / "Appendix_B_Sites_Table.csv", encoding="latin-1", newline="") as f:
        for r in csv.DictReader(f, delimiter=";"):
            sites[as_int(r["Site ID"])] = r
    vessels = []
    for r in read_xlsx(SOURCE / "Appendix_C_Macroscopic_Data.xlsx"):
        site = sites.get(as_int(r["Site ID"]), {})
        num = lambda k: site.get(k, "").replace(",", ".")
        vessels.append({
            "vessel_id": as_int(r["Vessel ID"]),
            "context_id": r["Context_ID"],
            "site_id": as_int(r["Site ID"]),
            "phase": r["Phase"],
            "vessel_type": r["Type"],
            "petrographic_sample": r["Petrographic sample"],
            "site_name": site.get("Site name", ""),
            "region": site.get("Region", ""),
            "site_type": site.get("Site type", ""),
            "longitude": num("Longitude"),
            "latitude": num("Latitude"),
            "approximate_position": site.get("Approximate position", ""),
        })
    write_csv(CLEAN / "vessels.csv", vessels, list(vessels[0]))

    # 4. consistency checks
    by_id = {v["vessel_id"]: v for v in vessels}
    assert len(chains) == len(by_id) == 276, "expected 276 vessels"
    assert all(by_id[c["vessel_id"]]["phase"] == c["bok"] for c in chains), "BoK/phase mismatch"
    no_site = [(v["vessel_id"], v["site_id"]) for v in vessels if not v["site_name"]]
    print(f"vessels whose site is missing from Appendix B (vessel_id, site_id): {no_site}")
    names = set(id2name.values())
    used = Counter(s for c in chains for s in c["chain"].split(" > "))
    assert set(used) <= names, f"unknown techniques: {set(used) - names}"
    allowed = {(l["source"], l["target"]) for l in links}
    observed = {(a, b) for c in chains for a, b in zip(c["chain"].split(" > "), c["chain"].split(" > ")[1:])}
    missing = observed - allowed
    print(f"{len(used)} techniques used; {len(observed)} observed transitions, "
          f"{len(missing)} not in Kroon's link list: {sorted(missing)}")
    print("vessels per BoK:", Counter(c["bok"] for c in chains))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="re-download source files from DANS")
    if parser.parse_args().download:
        download()
    main()
