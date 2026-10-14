

import requests
from tqdm import tqdm
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))







RCSB_URL = "https://files.rcsb.org/download/{}.cif"
AF_API_URL = "https://alphafold.ebi.ac.uk/api/prediction/{}"
SIFTS_URL = "https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{}"

UNIPROT_RE = re.compile(r"^([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})$")
AF_RE = re.compile(r"^AF-(.+)-F\d+$")


def uniprot_for(struct_id: str):
    m = AF_RE.match(struct_id)
    if m:
        return m.group(1)
    if UNIPROT_RE.match(struct_id):
        return struct_id

    response = requests.get(SIFTS_URL.format(struct_id.lower()), timeout=30)
    if response.status_code != 200:
        return None
    mappings = response.json().get(struct_id.lower(), {}).get("UniProt", {})
    return next(iter(mappings), None)


def download_rcsb(pdb_id: str, out_path: Path):
    response = requests.get(RCSB_URL.format(pdb_id), timeout=30)
    if response.status_code != 200:
        return False
    out_path.write_bytes(response.content)
    return True


def download_alphafold(accession: str, destination: Path):
    response = requests.get(AF_API_URL.format(accession), timeout=30)
    if response.status_code != 200:
        return None
    entries = response.json()
    if not entries:
        return None

    entry = entries[0]
    out_path = destination / f"{entry['entryId']}.cif"
    if out_path.exists():
        return out_path

    cif = requests.get(entry["cifUrl"], timeout=60)
    if cif.status_code != 200:
        return None
    out_path.write_bytes(cif.content)
    return out_path


def download_pdb(pdb_id: str, destination: Path):
    pdb_id = pdb_id.strip().upper()
    out_path = destination / f"{pdb_id}.cif"

    if out_path.exists():
        return "exists"

    if not AF_RE.match(pdb_id) and download_rcsb(pdb_id, out_path):
        return "rcsb"

    accession = uniprot_for(pdb_id)
    if accession is None:
        print(f"Error while loading {pdb_id}")
        return None

    af_path = download_alphafold(accession, destination)
    if af_path is None:
        print(f"Error while loading {pdb_id}")
        return None

    return "alphafold"


def down_all(pdb_ids: list[str], destination: Path):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)

    counts = {"rcsb": 0, "alphafold": 0, "exists": 0, None: 0}
    for pdb_id in tqdm(pdb_ids, desc="Loading CIF"):
        try:
            counts[download_pdb(pdb_id, destination)] += 1
        except Exception as e:
            counts[None] += 1
            print(f"Error {pdb_id}: {e}")

    return counts
