import json
import random
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import requests
import torch
from tqdm import tqdm
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

from config import ROOT_DIR, DATA_DIR, RAW_ZN_DIR as RAW_ZN, GRAPHS_PATH, APO_MAPPING_PATH as MAPPING_CACHE_PATH

URL = "https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id}"
DELAY = 0.35
SEED = 42
SIZE = 40
SAVE_EVERY = 50


def pilot_sample(seed=SEED, size=SIZE):
    graphs = torch.load(GRAPHS_PATH, weights_only=False)
    splits = {"train": [], "val": [], "test": []}
    for g in graphs:
        splits[g.split].append(g.pdb_id)
    del graphs

    total = 0
    for v in splits.values():
        total += len(v)
    rng = random.Random(seed)
    sample = []
    for split, ids in splits.items():
        ids = sorted(ids)
        rng.shuffle(ids)
        n = round(size * len(ids) / total)
        for pid in ids[:n]:
            sample.append((pid, split))
    return sample


def full_sample():
    graphs = torch.load(GRAPHS_PATH, weights_only=False)
    sample = []
    for g in graphs:
        sample.append((g.pdb_id, g.split))
    del graphs
    return sorted(sample)


def load_cache():
    if MAPPING_CACHE_PATH.exists():
        with open(MAPPING_CACHE_PATH) as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(MAPPING_CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def fetch_mapping(pdb_id):
    r = requests.get(URL.format(pdb_id=pdb_id), timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    data = r.json().get(pdb_id, {}).get("UniProt", {})
    if not data:
        return None

    chains = {}
    for acc, entry in data.items():
        for m in entry["mappings"]:
            cid = m["chain_id"]
            u0 = m["unp_start"]
            u1 = m["unp_end"]
            a_start = m["start"]["author_residue_number"]
            a_end = m["end"]["author_residue_number"]
            s_start = m["start"]["residue_number"]
            s_end = m["end"]["residue_number"]
            if a_start is not None:
                a0 = a_start
            elif a_end is not None:
                a0 = a_end - (u1 - u0)
            else:
                a0 = None
            seg = {"author_start": a0, "unp_start": u0, "unp_end": u1,
                   "seq_start": s_start, "seq_end": s_end}
            score = m.get("identity", 0.0) * m.get("coverage", 0.0)
            if cid not in chains:
                chains[cid] = {"accession": acc, "score": score, "segments": []}
            c = chains[cid]
            if acc != c["accession"] and score > c["score"]:
                c = {"accession": acc, "score": score, "segments": []}
                chains[cid] = c
            if acc == c["accession"]:
                c["segments"].append(seg)

    result = {}
    for cid, v in chains.items():
        result[cid] = {"accession": v["accession"], "segments": v["segments"]}
    return result


def local_offset(cif_path, chain_id, seq_start, seq_end):
    if cif_path is None or not Path(cif_path).exists():
        return None
    try:
        d = MMCIF2Dict(str(cif_path))
    except Exception:
        return None
    seq_ids = d.get("_pdbx_poly_seq_scheme.seq_id")
    auth = d.get("_pdbx_poly_seq_scheme.auth_seq_num")
    strand = d.get("_pdbx_poly_seq_scheme.pdb_strand_id")
    if not seq_ids or not auth or not strand:
        return None
    first = None
    for sid, au, cid in zip(seq_ids, auth, strand):
        if cid != chain_id or au in ("?", "."):
            continue
        try:
            sid = int(sid)
            au = int(au)
        except ValueError:
            continue
        off = au - sid
        if seq_start is not None and seq_end is not None and seq_start <= sid <= seq_end:
            return off
        if first is None:
            first = off
    return first


def fix_missing(mapping, cif_path):
    if not mapping:
        return
    for cid, info in mapping.items():
        good = []
        for seg in info["segments"]:
            if seg["author_start"] is None:
                off = local_offset(cif_path, cid, seg["seq_start"], seg["seq_end"])
                if off is not None:
                    seg["author_start"] = seg["seq_start"] + off
            if seg["author_start"] is not None:
                good.append(seg)
        info["segments"] = good


def map_residue(mapping, resnum):
    for seg in mapping["segments"]:
        a0 = seg["author_start"]
        u0 = seg["unp_start"]
        u1 = seg["unp_end"]
        a1 = a0 + (u1 - u0)
        if a0 <= resnum <= a1:
            return u0 + (resnum - a0)
    return None


def find_cif(pdb_id):
    path = RAW_ZN / (pdb_id.upper() + ".cif")
    if path.exists():
        return path
    return None


def build_cache(pdb_ids=None, cif_lookup=find_cif):
    cache = load_cache()
    todo = []
    for pid in (pdb_ids or []):
        if pid not in cache:
            todo.append(pid)
    print(len(cache), "in cache,", len(todo), "to fetch")

    failed = []
    count = 0
    for pdb_id in tqdm(todo):
        try:
            mapping = fetch_mapping(pdb_id)
            if mapping is not None and cif_lookup is not None:
                fix_missing(mapping, cif_lookup(pdb_id))
            if mapping is None:
                cache[pdb_id] = {}
                failed.append(pdb_id)
            else:
                cache[pdb_id] = mapping
        except Exception as e:
            failed.append(pdb_id)
            cache[pdb_id] = {}
            print("error", pdb_id, e)
        time.sleep(DELAY)

        count += 1
        if count >= SAVE_EVERY:
            save_cache(cache)
            count = 0

    save_cache(cache)
    return cache, failed


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()

    if args.full:
        sample = full_sample()
        out = DATA_DIR / "apo_full_sample.json"
    else:
        sample = pilot_sample()
        out = DATA_DIR / "apo_pilot_sample.json"

    n_train = len([s for _, s in sample if s == "train"])
    n_val = len([s for _, s in sample if s == "val"])
    n_test = len([s for _, s in sample if s == "test"])
    print(len(sample), "pdb_ids, train", n_train, "val", n_val, "test", n_test)

    with open(out, "w") as f:
        json.dump(sample, f, indent=2)

    ids = [pid for pid, _ in sample]
    cache, failed = build_cache(ids)
    ok = 0
    for pid, _ in sample:
        if cache.get(pid):
            ok += 1
    print("mapped:", ok, "/", len(sample))
    if failed:
        print("no mapping:", len(failed), failed[:20])
    print("saved", MAPPING_CACHE_PATH)