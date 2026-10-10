"""Download Kroon's (2024) DANS deposit into data/source/dans/, keeping its folder structure.

    python scripts/download_dans.py [--all]

Deposit: doi:10.17026/AR/FAXIOW (DANS Data Station Archaeology), CC-BY-4.0.
By default the photographs (Appendix D, 13 tar files, ~36 GB) and the
high-resolution figures of the book (~0.6 GB) are skipped; everything else
(data tables, Kroon's code, bibliographies, reports) is about 37 MB.
Every file is checked against the checksum published by DANS and listed in
data/source/dans/MANIFEST.csv. Files already present with the right
checksum are not downloaded again. Standard library only.
"""
import argparse
import csv
import hashlib
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "source" / "dans"
API = "https://archaeology.datastations.nl/api"
DOI = "doi:10.17026/AR/FAXIOW"


def skipped(path):
    return path.endswith(".tar") or "High_Resolution_Figures/" in path


def checksum(path, kind):
    h = hashlib.new(kind.lower().replace("-", ""))
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url, target, tries=3):
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(target, "wb") as f:
                while block := r.read(1 << 20):
                    f.write(block)
            return
        except OSError:
            if attempt == tries - 1:
                raise
            time.sleep(5 * (attempt + 1))


def main(all_files=False):
    meta = json.load(urllib.request.urlopen(f"{API}/datasets/:persistentId/?persistentId={DOI}", timeout=60))
    version = meta["data"]["latestVersion"]
    print(f"{DOI}, version {version['versionNumber']}.{version['versionMinorNumber']} "
          f"({version['lastUpdateTime'][:10]}), {len(version['files'])} files")
    rows = []
    for f in version["files"]:
        d = f["dataFile"]
        name = d.get("originalFileName", d["filename"])
        rel = f"{f['directoryLabel']}/{name}" if f.get("directoryLabel") else name
        if not all_files and skipped(rel):
            continue
        target = OUT / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        kind, value = d["checksum"]["type"], d["checksum"]["value"]
        if not (target.exists() and checksum(target, kind) == value):
            print(f"downloading {rel} ({d.get('filesize', 0) / 1e6:.2f} MB)")
            fetch(f"{API}/access/datafile/{d['id']}?format=original", target)
        ok = checksum(target, kind) == value
        if not ok:
            print(f"  checksum mismatch for {rel}")
        rows.append({"path": rel, "dans_file_id": d["id"], "bytes": target.stat().st_size,
                     "checksum_type": kind, "checksum": value, "verified": ok})
    with open(OUT / "MANIFEST.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["path"]))
    print(f"{len(rows)} files in {OUT.relative_to(ROOT)}, {sum(r['bytes'] for r in rows) / 1e6:.1f} MB; "
          f"{sum(not r['verified'] for r in rows)} checksum mismatches")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true", help="also download the photographs and high-resolution figures")
    main(parser.parse_args().all)
