import json
import sys
import warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from Bio.PDB import FastMMCIFParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from tqdm import tqdm

warnings.filterwarnings("ignore", category=PDBConstructionWarning)

from src.data_pipeline.graphing import gen_graph, node_res_ids, AA
from src.data_pipeline import apo_mapping
from config import ROOT_DIR, DATA_DIR, RAW_ZN_DIR as RAW_ZN, RAW_APO_DIR as RAW_APO, MAX_CA_ATOMS

OUT_PATH = DATA_DIR / "graphs_apo_pilot.pt"
MAX_BYTES = 3_000_000
MIN_RATIO = 0.5

PARSER = FastMMCIFParser(QUIET=True)


def holo_positives(pdb_id, holo_graph):
    structure = PARSER.get_structure(pdb_id, str(RAW_ZN / (pdb_id.upper() + ".cif")))
    res_ids = node_res_ids(structure)
    if len(res_ids) != holo_graph.x.size(0):
        raise ValueError(pdb_id + ": " + str(len(res_ids)) + " vs " + str(holo_graph.x.size(0)))
    mask = (holo_graph.y[:, 0] == 1).tolist()
    pos = []
    for i in range(len(res_ids)):
        if mask[i]:
            pos.append((res_ids[i][0], res_ids[i][1]))
    return pos


def to_uniprot(positives, holo_mapping):
    result = {}
    unmapped = 0
    for cid, resnum in positives:
        cmap = holo_mapping.get(cid)
        if cmap is None:
            unmapped += 1
            continue
        u = apo_mapping.map_residue(cmap, resnum)
        if u is None:
            unmapped += 1
            continue
        acc = cmap["accession"]
        if acc not in result:
            result[acc] = set()
        result[acc].add(u)
    return result, unmapped


def af_positions(uniprot_pos):
    out = set()
    for u in uniprot_pos:
        out.add(("A", u))
    return out


def real_apo_positions(accession, uniprot_pos, apo_id, mapping_cache):
    chains = mapping_cache.get(apo_id, {})
    out = set()
    for cid, v in chains.items():
        if v["accession"] != accession:
            continue
        for seg in v["segments"]:
            a0 = seg["author_start"]
            u0 = seg["unp_start"]
            u1 = seg["unp_end"]
            for u in uniprot_pos:
                if u0 <= u <= u1:
                    out.add((cid, a0 + (u - u0)))
    return out


def build_one(pdb_id, holo_graph, holo_mapping, manifest, mapping_cache):
    split = holo_graph.split
    positives = holo_positives(pdb_id, holo_graph)
    if not positives:
        return None, {"pdb_id": pdb_id, "status": "holo_no_positives"}

    by_acc, unmapped = to_uniprot(positives, holo_mapping)
    if not by_acc:
        return None, {"pdb_id": pdb_id, "status": "mapping_miss",
                      "n_holo_positive": len(positives), "n_unmapped": unmapped}

    accession = None
    uniprot_pos = None
    for acc, pos in by_acc.items():
        if uniprot_pos is None or len(pos) > len(uniprot_pos):
            accession = acc
            uniprot_pos = pos
    n_dominant = len(uniprot_pos)
    n_total = 0
    for v in by_acc.values():
        n_total += len(v)

    info = manifest.get(accession)
    if not info or info["source"] is None:
        return None, {"pdb_id": pdb_id, "status": "no_apo_structure",
                      "n_holo_positive": len(positives), "accession": accession}

    apo_path = ROOT_DIR / info["path"]
    size = apo_path.stat().st_size
    if size > MAX_BYTES:
        return None, {"pdb_id": pdb_id, "status": "apo_file_too_large",
                      "n_bytes": size, "accession": accession, "source": info["source"]}
    apo = PARSER.get_structure(accession, str(apo_path))
    for m in list(apo)[1:]:
        apo.detach_child(m.id)
    n_ca = 0
    for res in apo[0].get_residues():
        if res.get_resname().strip() in AA and "CA" in res:
            n_ca += 1
    if n_ca > MAX_CA_ATOMS:
        return None, {"pdb_id": pdb_id, "status": "apo_too_large", "n_ca": n_ca,
                      "accession": accession, "source": info["source"]}

    if info["source"] == "alphafold":
        pos_res = af_positions(uniprot_pos)
    else:
        pos_res = real_apo_positions(accession, uniprot_pos, info["structure_id"].lower(), mapping_cache)

    if not pos_res:
        return None, {"pdb_id": pdb_id, "status": "apo_mapping_miss",
                      "n_holo_positive": len(positives), "accession": accession,
                      "source": info["source"]}

    g = gen_graph(apo, positive_residues=pos_res)
    if g is None or g.num_nodes == 0:
        return None, {"pdb_id": pdb_id, "status": "empty_apo_graph", "accession": accession}

    n_trans = int(g.y[:, 0].sum().item())
    ratio = n_trans / n_dominant
    report = {
        "pdb_id": pdb_id, "source": info["source"], "accession": accession,
        "n_holo_positive": len(positives), "n_unmapped_to_uniprot": unmapped,
        "n_uniprot_positions_dominant_chain": n_dominant,
        "n_uniprot_positions_total": n_total,
        "n_transferred_to_apo_graph": n_trans,
        "transfer_ratio": round(ratio, 3),
    }

    if ratio < MIN_RATIO:
        report["status"] = "low_transfer_ratio_excluded"
        return None, report

    g.pdb_id = pdb_id + "_apo"
    g.split = split
    g.is_apo = True
    g.apo_source = info["source"]
    g.apo_accession = accession
    report["status"] = "ok"
    return g, report


def find_real_apo_cif(entry_id):
    matches = list(RAW_APO.glob("*_real_" + entry_id.upper() + ".cif"))
    if matches:
        return matches[0]
    return None


def main(sample_path, out_path, report_path):
    with open(sample_path) as f:
        sample = json.load(f)
    with open(DATA_DIR / "apo_manifest.json") as f:
        manifest = json.load(f)

    print("loading graphs.pt")
    holo_graphs = {}
    for g in torch.load(apo_mapping.GRAPHS_PATH, weights_only=False):
        holo_graphs[g.pdb_id] = g

    ids = set()
    for v in manifest.values():
        if v["source"] == "real_apo":
            ids.add(v["structure_id"].lower())
    ids = sorted(ids)
    print("mapping", len(ids), "real apo entries")

    apo_mapping.build_cache(ids, cif_lookup=find_real_apo_cif)
    with open(apo_mapping.MAPPING_CACHE_PATH) as f:
        mapping_cache = json.load(f)

    graphs = []
    reports = []
    count = 0
    for pdb_id, _ in tqdm(sample):
        holo_mapping = mapping_cache.get(pdb_id)
        holo_graph = holo_graphs.get(pdb_id)
        if not holo_mapping or holo_graph is None:
            reports.append({"pdb_id": pdb_id, "status": "no_holo_uniprot_mapping"})
            continue
        try:
            g, report = build_one(pdb_id, holo_graph, holo_mapping, manifest, mapping_cache)
        except Exception as e:
            reports.append({"pdb_id": pdb_id, "status": "exception", "error": repr(e)})
            continue
        reports.append(report)
        if g is not None:
            graphs.append(g)

        count += 1
        if count >= 50:
            torch.save(graphs, out_path)
            with open(report_path, "w") as f:
                json.dump(reports, f, indent=2)
            count = 0

    torch.save(graphs, out_path)

    counts = {}
    for r in reports:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    print(len(sample), "pdb_ids ->", len(graphs), "apo graphs")
    for status, n in counts.items():
        print(status, n)

    ok = [r for r in reports if r["status"] == "ok"]
    if ok:
        total_holo = 0
        total_trans = 0
        sources = {}
        for r in ok:
            total_holo += r["n_holo_positive"]
            total_trans += r["n_transferred_to_apo_graph"]
            sources[r["source"]] = sources.get(r["source"], 0) + 1
        print("sources:", sources)
        print("transferred:", total_trans, "/", total_holo)

    splits = {}
    for g in graphs:
        splits[g.split] = splits.get(g.split, 0) + 1
    print("splits:", splits)

    with open(report_path, "w") as f:
        json.dump(reports, f, indent=2)
    print("saved", out_path)
    print("report", report_path)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()

    if args.full:
        main(DATA_DIR / "apo_full_sample.json", DATA_DIR / "graphs_apo_full.pt",
             DATA_DIR / "apo_full_report.json")
    else:
        main(DATA_DIR / "apo_pilot_sample.json", OUT_PATH,
             DATA_DIR / "apo_pilot_report.json")