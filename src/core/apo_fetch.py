import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import requests
from tqdm import tqdm

from config import ROOT_DIR, DATA_DIR, RAW_APO_DIR, APO_MAPPING_PATH as MAPPING_CACHE_PATH, APO_MANIFEST_PATH as MANIFEST_PATH
RAW_APO_DIR.mkdir(exist_ok=True)

SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
ENTRY_URL = "https://data.rcsb.org/rest/v1/core/entry/{entry_id}"
AF_URL = "https://alphafold.ebi.ac.uk/api/prediction/{accession}"
MAX_CANDIDATES = 5
DELAY = 0.2
SAVE_EVERY = 25


def get_accessions(mapping_cache):
    accs = set()
    for chains in mapping_cache.values():
        for v in chains.values():
            accs.add(v["accession"])
    return sorted(accs)


def search_rcsb(accession):
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {"type": "terminal", "service": "text", "parameters": {
                    "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
                    "operator": "exact_match", "value": accession}},
                {"type": "terminal", "service": "text", "parameters": {
                    "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_name",
                    "operator": "exact_match", "value": "UniProt"}},
            ],
        },
        "return_type": "entry",
        "request_options": {"paginate": {"start": 0, "rows": MAX_CANDIDATES}},
    }
    r = requests.post(SEARCH_URL, json=query, timeout=30)
    if r.status_code == 404:
        return []
    r.raise_for_status()
    ids = []
    for res in r.json().get("result_set", []):
        ids.append(res["identifier"])
    return ids


def has_zn(entry_id):
    r = requests.get(ENTRY_URL.format(entry_id=entry_id), timeout=30)
    r.raise_for_status()
    ligands = r.json().get("rcsb_entry_info", {}).get("nonpolymer_bound_components")
    if ligands is None:
        ligands = []
    return "ZN" in ligands


def download(url, path):
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        return False
    path.write_bytes(r.content)
    return True


def get_real_apo(accession):
    for entry_id in search_rcsb(accession):
        time.sleep(DELAY)
        try:
            if has_zn(entry_id):
                continue
        except requests.HTTPError:
            continue
        path = RAW_APO_DIR / (accession + "_real_" + entry_id + ".cif")
        if download("https://files.rcsb.org/download/" + entry_id + ".cif", path):
            return entry_id, path
    return None, None


def get_alphafold(accession):
    r = requests.get(AF_URL.format(accession=accession), timeout=30)
    if r.status_code != 200 or not r.json():
        return None
    url = r.json()[0]["cifUrl"]
    path = RAW_APO_DIR / (accession + "_af.cif")
    if download(url, path):
        return path
    return None


def load_manifest():
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    return {}


def save_manifest(manifest):
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)


def build_manifest(accessions):
    manifest = load_manifest()
    todo = []
    for a in accessions:
        if a not in manifest:
            todo.append(a)
    print(len(manifest), "in manifest,", len(todo), "to fetch")

    count = 0
    for acc in tqdm(todo):
        try:
            entry_id, path = get_real_apo(acc)
            if path is not None:
                manifest[acc] = {"source": "real_apo", "structure_id": entry_id,
                                 "path": str(path.relative_to(ROOT_DIR))}
            else:
                time.sleep(DELAY)
                path = get_alphafold(acc)
                if path is not None:
                    manifest[acc] = {"source": "alphafold", "structure_id": "AF-" + acc + "-F1",
                                     "path": str(path.relative_to(ROOT_DIR))}
                else:
                    manifest[acc] = {"source": None, "structure_id": None, "path": None}
        except Exception as e:
            manifest[acc] = {"source": None, "structure_id": None, "path": None}
            print("error", acc, e)
        time.sleep(DELAY)

        count += 1
        if count >= SAVE_EVERY:
            save_manifest(manifest)
            count = 0

    save_manifest(manifest)
    return manifest


if __name__ == "__main__":
    with open(MAPPING_CACHE_PATH) as f:
        mapping_cache = json.load(f)
    accessions = get_accessions(mapping_cache)
    print(len(accessions), "accessions")

    manifest = build_manifest(accessions)
    n_real = 0
    n_af = 0
    n_fail = 0
    for v in manifest.values():
        if v["source"] == "real_apo":
            n_real += 1
        elif v["source"] == "alphafold":
            n_af += 1
        else:
            n_fail += 1
    print("real_apo:", n_real, "alphafold:", n_af, "failed:", n_fail, "/", len(manifest))
    print("saved", MANIFEST_PATH)