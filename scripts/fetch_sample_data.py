# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""
Download a CCS sample from PubChem (CCSbase source).

For each hard-coded PubChem CID the script fetches CCS values from the
CCSbase source and the corresponding SMILES.  Each (CID, adduct) pair
becomes one row; replicate measurements are averaged.

Supported adducts: [M+H]+, [M-H]-, [M+Na]+

The output CSV contains one row per (molecule, adduct) combination
(185 rows for the default CID list).
Output CSV columns: index, smiles, adducts, label

Usage
-----
    python scripts/fetch_sample_data.py
    python scripts/fetch_sample_data.py -o my_out.csv
"""

import argparse
import csv
import json
import re
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

# 100 PubChem CIDs with CCSbase measurements (fixed sample, seed 42)
SAMPLE_CIDS = [
    243, 834, 863, 2343, 2378, 2381, 2520, 2581, 2896, 3052, 3410, 3759,
    3779, 3783, 3827, 4034, 4169, 4192, 4497, 4624, 4680, 4977, 5011, 5029,
    5387, 5419, 5504, 5509, 5533, 5879, 5944, 5960, 6291, 6293, 6613, 6723,
    6912, 7027, 7091, 8567, 9687, 9700, 9903, 10250, 10685, 11128, 18381,
    38853, 39965, 65063, 65188, 66461, 67069, 68741, 69867, 74706, 92180,
    92803, 102175, 182362, 439377, 440049, 440707, 440735, 443943, 444679,
    629853, 2761491, 4778266, 5280581, 5281542, 5281807, 5390108, 5460662,
    5464170, 5935070, 5946498, 6708739, 6708809, 21596360, 23728435,
    40490664, 52925136, 54675839, 75302455, 90657965, 135398619, 135398646,
    135398737, 135409400, 136212488, 171114866, 171114940, 171114985,
    171115065, 171115068, 171115083, 171115092, 171115131, 171115212,
]

# PubChem API endpoints
PUG_VIEW_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON"
    "?source=CCSbase"
)
PROPS_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cids}/"
    "property/SMILES/JSON"
)

TARGET_ADDUCTS  = {"[M+H]+", "[M-H]-", "[M+Na]+"}
PROPS_BATCH_SIZE = 100
REQUEST_DELAY    = 0.34   # ~3 requests/s — PubChem's recommended rate limit

# Matches lines like "154.23 Å² [M+H]+"  (Å = U+212B, ² = U+00B2)
CCS_RE = re.compile(r"^([\d.]+)\s*\u212b\u00b2\s*(\[M[^\]]+\][\+\-])")


def fetch_json(url: str) -> dict:
    """GET url, wait for the polite delay, and return parsed JSON."""
    time.sleep(REQUEST_DELAY)
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode())


def find_section(obj, heading: str) -> dict | None:
    """Recursively find the first dict with TOCHeading == heading."""
    if isinstance(obj, dict):
        if obj.get("TOCHeading") == heading:
            return obj
        for v in obj.values():
            result = find_section(v, heading)
            if result is not None:
                return result
    elif isinstance(obj, list):
        for item in obj:
            result = find_section(item, heading)
            if result is not None:
                return result
    return None


def fetch_properties(cids: list[int]) -> dict[int, str]:
    """Return {cid: isomeric_smiles} for all CIDs, fetched in one batch."""
    result: dict[int, str] = {}
    for i in range(0, len(cids), PROPS_BATCH_SIZE):
        batch = cids[i: i + PROPS_BATCH_SIZE]
        url = PROPS_URL.format(cids=",".join(str(c) for c in batch))
        print(f"  Fetching SMILES for {len(batch)} CIDs …", flush=True)
        try:
            data = fetch_json(url)
            for prop in data.get("PropertyTable", {}).get("Properties", []):
                smiles = prop.get("SMILES", "")
                if smiles:
                    result[prop["CID"]] = smiles
        except Exception as exc:
            print(f"  Warning: SMILES batch failed: {exc}", file=sys.stderr)
    return result


def fetch_ccs_for_cid(cid: int) -> dict[str, float]:
    """Return {adduct: mean_ccs} from the CCSbase record for cid."""
    url = PUG_VIEW_URL.format(cid=cid)
    try:
        data = fetch_json(url)
    except Exception as exc:
        print(f"  Warning: CCS fetch failed for CID {cid}: {exc}", file=sys.stderr)
        return {}

    sec = find_section(data, "Collision Cross Section")
    if sec is None:
        return {}

    # Collect all replicate measurements per adduct, then average
    raw: dict[str, list[float]] = defaultdict(list)
    for info in sec.get("Information", []):
        for sv in info.get("Value", {}).get("StringWithMarkup", []):
            m = CCS_RE.match(sv.get("String", ""))
            if m and m.group(2) in TARGET_ADDUCTS:
                raw[m.group(2)].append(float(m.group(1)))

    return {adduct: sum(vals) / len(vals) for adduct, vals in raw.items()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-o", "--output", default="data/data.csv",
                        help="Output CSV path (default: data/data.csv)")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print(f"Fetching data for {len(SAMPLE_CIDS)} CIDs from PubChem …")

    smiles_map = fetch_properties(SAMPLE_CIDS)
    missing = set(SAMPLE_CIDS) - smiles_map.keys()
    if missing:
        print(f"  Warning: no SMILES found for CIDs: {sorted(missing)}", file=sys.stderr)

    print("Fetching CCSbase values per CID …")
    rows: list[tuple[int, str, str, float]] = []
    for i, cid in enumerate(sorted(SAMPLE_CIDS), 1):
        if cid not in smiles_map:
            continue
        print(f"  [{i}/{len(SAMPLE_CIDS)}] CID {cid} …", flush=True)
        ccs_map = fetch_ccs_for_cid(cid)
        for adduct, ccs in sorted(ccs_map.items()):
            rows.append((cid, smiles_map[cid], adduct, ccs))

    print(f"Writing {len(rows)} rows to {args.output} …")
    with open(args.output, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["index", "smiles", "adducts", "label"])
        for idx, (cid, smiles, adduct, ccs) in enumerate(rows):
            writer.writerow([idx, smiles, adduct, ccs])

    print(f"Done. {len(rows)} rows written to {args.output}.")


if __name__ == "__main__":
    main()
