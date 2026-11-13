#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import requests
import argparse
from tqdm import tqdm
import json
import os
import shutil

from config import ROOT_DIR, DATA_DIR, GRAPHS_PATH


SETTINGS_PATH = ROOT_DIR / "download_settings.json"
config = json.load(open(SETTINGS_PATH)) if SETTINGS_PATH.exists() else {"download": {}}
resolution = config["download"].get("resolution")
metal = config["download"].get("metal")
max_weight = config["download"].get("max_weight")
total = config["download"].get("total_pdbs")
RAW_DIR = DATA_DIR / "raw_cu"
CLUSTERED_DIR = DATA_DIR / "clustered"
RAW_DIR.mkdir(parents=True, exist_ok=True)
CLUSTERED_DIR.mkdir(parents=True, exist_ok=True)

def fetch_pdb(metal: str = "CU", resolution: float = 2.5, max_weight: float = 300):
    all_ids = []
    start = 0
    rows = 1000  # število na stran

    while True:
        query = {
            "query": {
                "type": "group",
                "logical_operator": "and",
                "nodes": [
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "rcsb_entry_info.molecular_weight",
                            "operator": "less_or_equal",
                            "value": max_weight,
                        }
                    },
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "rcsb_nonpolymer_instance_annotation.comp_id",
                            "operator": "exact_match",
                            "value": metal,
                        }
                    },
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "rcsb_entry_info.resolution_combined",
                            "operator": "less_or_equal",
                            "value": resolution,
                        }
                    },
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "rcsb_entry_info.experimental_method",
                            "operator": "exact_match",
                            "value": "X-ray"
                        }
                    }
                ]
            },
            "return_type": "entry",
            "request_options": {
                "paginate": {
                    "start": start,
                    "rows": rows
                }
            }
        }

        response = requests.post("https://search.rcsb.org/rcsbsearch/v2/query", json=query)
        response.raise_for_status()
        results = response.json()

        batch = [r["identifier"] for r in results["result_set"]]
        all_ids.extend(batch)

        total_count = results["total_count"]
        print(f"Naloženo {len(all_ids)}/{total_count}")

        if len(all_ids) >= total_count or len(batch) < rows:
            break

        start += rows

    print(f"Skupaj najdenih {len(all_ids)} PDB")
    return all_ids


def download_pdb(pdb_id: str):
    pdb_id = pdb_id.upper()
    out_path = RAW_DIR / f"{pdb_id}.pdb"

    if out_path.exists():
        return False

    url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    out_path = RAW_DIR / f"{pdb_id}.cif"
    response = requests.get(url, timeout=30)

    if response.status_code == 200:
        out_path.write_bytes(response.content)
        return True
    else:
        print(f"Napaka pri nalaganju {pdb_id}")
        return False


def down_all(pdb_ids: list[str]):
    downloaded = 0
    failed = 0
    for pdb_id in tqdm(pdb_ids, desc="Nalaganje PDB"):
        try:
            result = download_pdb(pdb_id)
            if result:
                downloaded += 1

        except Exception as e:
            failed += 1
            print(f"Napaka {pdb_id}: {e}")


# cd src
# py download.py --metal CU --resolution 2.5 --max_weight 300
if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("--metal", type=str, default="CU")
    ap.add_argument("--resolution", type=float, default=2.5)
    ap.add_argument("--max_weight", type=float, default=300.0)
    args = ap.parse_args()

    confirm = input(f"Delete ALL content in {RAW_DIR} and {CLUSTERED_DIR}? y/n: ")

    if confirm.lower() == "y":
        for directory in [RAW_DIR, CLUSTERED_DIR]:
            for item in os.listdir(directory):
                path = os.path.join(directory, item)
                if os.path.isfile(path) or os.path.islink(path):
                    os.remove(path)
                elif os.path.isdir(path):
                    shutil.rmtree(path)

        print("Deleted.")

    pdb_ids = fetch_pdb(metal=str(metal), resolution=float(resolution), max_weight=float(max_weight))

    down_all(pdb_ids)