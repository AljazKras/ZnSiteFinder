import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Bio.PDB import FastMMCIFParser
from torch_geometric.data import Data
from Bio.PDB.NeighborSearch import NeighborSearch
import torch
import numpy as np
import json
import traceback
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
from src.data_pipeline.feature_generation import node_feature, structure_context
from src.data_pipeline.feature_generation import local_frame
from src.data_pipeline.feature_generation import sidechain_donors
from config import ROOT_DIR, DATA_DIR, RAW_ZN_DIR as RAW_DIR, SCORED_MARKED_PATH

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")




def con_density(ns, atomi, atomj, thresh = 3):
    coordi = np.array(atomi.get_vector().get_array())
    coordj = np.array(atomj.get_vector().get_array())
    diff = coordi - coordj
    xyz = np.linalg.norm(diff)
    if xyz == 0:
        return 0
    mid = (coordi + coordj) / 2
    search = xyz / 2 + thresh
    nej = ns.search(mid, search, level="R")
    den = 0
    for res in nej:
        if res == atomi.get_parent() or res == atomj.get_parent():
            continue
        min_perp = float("inf")
        for a in res:
            if a.element == "H":
                continue
            p = np.array(a.get_vector().get_array())
            v = p - coordi
            t = np.dot(v, diff) / (xyz ** 2)
            if 0.0 < t < 1.0:
                closest_point = coordi + t * diff
                perp = np.linalg.norm(p - closest_point)
                min_perp = min(min_perp, perp)
        if min_perp < thresh:
            den += 1

    return den/xyz

AA = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D",
    "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
    "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
    "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"
}

def node_res_ids(protein):

    res_ids = []
    for atom in protein.get_atoms():
        if atom.get_name() != "CA":
            continue
        residue = atom.get_parent()
        resname = residue.get_resname().strip()
        if resname not in AA:
            continue
        if not all(a in residue for a in ("N", "CA", "C")):
            continue
        chain_id = residue.get_parent().get_id()
        res_ids.append((chain_id, residue.get_id()[1], resname))
    return res_ids


def gen_graph(protein, sites=None, radii: float = 11.5, return_res_ids: bool = False,
              include_cb_dist: bool = True, include_cb_edge: bool = True,
              positive_residues=None):

    ctx = structure_context(protein)
    atoms = list(protein.get_atoms())
    ns = NeighborSearch(atoms)
    ca_atoms = [a for a in atoms if a.get_name() == "CA" and a.get_parent().get_resname().strip() in AA]

    metal_atoms = []
    if sites is not None:
        metal_lookup = {}
        for atom in atoms:
            if atom.get_id() != "ZN":
                continue
            chain_id = atom.get_parent().get_parent().id
            resnum = str(atom.get_parent().id[1])
            metal_lookup[(chain_id, resnum)] = atom
        for id_str in sites["ID"]:
            if pd.isna(id_str):
                continue
            chain, resnum = id_str.split(":")
            metal = metal_lookup.get((chain, resnum))
            if metal is not None:
                metal_atoms.append(metal)

    y, x, edge_index, edge_attr, pos, frames = [], [], [], [], [], []
    node_idx = {}
    valid_atoms = []
    cb_pos = []
    res_ids = []
    for atom in ca_atoms:
        residue = atom.get_parent()
        chain_id = residue.get_parent().get_id()
        f = node_feature(residue, ctx, chain_id, include_cb_dist=include_cb_dist)
        if f is None:
            continue
        node_idx[id(atom)] = len(x)
        x.append(f)
        pos.append(atom.get_coord())

        cb_pos.append(residue["CB"].get_coord() if "CB" in residue else atom.get_coord())
        frames.append(local_frame(residue))
        valid_atoms.append(atom)
        if return_res_ids:
            res_ids.append((chain_id, residue.get_id()[1], residue.get_resname().strip()))

    if not x:
        return None

    donor_pos = []
    for atom in valid_atoms:
        coord = np.array(atom.get_vector().get_array())

        nej = ns.search(coord, radii, level="A")
        for atomx in nej:
            if atomx.get_name() != "CA" or atomx is atom or id(atomx) not in node_idx:
                continue

            diff = coord - np.array(atomx.get_vector().get_array())
            xyz = np.linalg.norm(diff)

            i = node_idx[id(atom)]
            j = node_idx[id(atomx)]
            framei = frames[i]
            framej = frames[j]
            veci = framei@(diff / (xyz + 1e-8))
            vecj = framej@(-diff / (xyz + 1e-8))
            if i >= j:
                continue

            l = con_density(ns,atom, atomx)
            dists = [xyz, np.linalg.norm(cb_pos[i] - cb_pos[j])] if include_cb_edge else [xyz]
            edge_index.extend([[i, j], [j, i]])
            edge_attr.extend([[*veci, *dists, l], [*vecj, *dists, l]])

        if metal_atoms:
            residue = atom.get_parent()
            atom_dists = [
                np.linalg.norm(a.get_coord() - m.get_coord())
                for a in residue if a.element != "H"
                for m in metal_atoms
            ]
            m_dist = min(atom_dists)
            y_cls = 1 if m_dist < 4 else 0

            resname = residue.get_resname().strip()
            donor_atoms = [residue[a] for a in sidechain_donors.get(resname, []) if a in residue]
            reg_atoms = donor_atoms if donor_atoms else [residue["CA"]]
            reg_pairs = [(a, m) for a in reg_atoms for m in metal_atoms]
            reg_dists = [np.linalg.norm(a.get_coord() - m.get_coord()) for a, m in reg_pairs]
            best = int(np.argmin(reg_dists))
            y_reg = min(reg_dists[best], 8)

            donor_coord = reg_pairs[best][0].get_coord()

        else:
            residue = atom.get_parent()
            key = (residue.get_parent().get_id(), residue.get_id()[1])
            y_cls = 1 if positive_residues is not None and key in positive_residues else 0
            y_reg = 8.0
            donor_coord = coord  # no metal in this structure

        y.append([y_cls, y_reg])
        donor_pos.append(donor_coord)
    mask = [True] * len(x)

    metal_pos = np.array([m.get_coord() for m in metal_atoms]) if metal_atoms else np.zeros((0, 3))

    data = Data(
        x=torch.tensor(x, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float),
        y=torch.tensor(y, dtype=torch.float),
        mask=torch.tensor(mask, dtype=torch.bool),
        pos=torch.tensor(np.array(pos), dtype=torch.float),

        frame=torch.tensor(np.array(frames), dtype=torch.float),

        donor_pos=torch.tensor(np.array(donor_pos), dtype=torch.float),
        metal_pos=torch.tensor(metal_pos, dtype=torch.float),

        reg_valid=positive_residues is None,
        pdb=protein.id,
    )
    if return_res_ids:
        data.res_id = res_ids
    return data

def build_all_graphs(include_cb_dist: bool = True, include_cb_edge: bool = True):

    parser = FastMMCIFParser(QUIET=True)
    df = pd.read_csv(SCORED_MARKED_PATH)
    sites_by_pdb = {pid: g for pid, g in df.groupby("pdb_id")}
    split_by_pdb = df.drop_duplicates("pdb_id").set_index("pdb_id")["SPLIT"].to_dict()

    all_graphs, missing, failed = [], [], []
    for pdb_id in tqdm(df["pdb_id"].unique()):

        path = RAW_DIR / (pdb_id.upper() + ".cif")
        if not path.exists():
            missing.append(pdb_id)
            continue

        try:
            structure = parser.get_structure(pdb_id, path)
            g = gen_graph(structure, sites_by_pdb[pdb_id],include_cb_dist=include_cb_dist, include_cb_edge=include_cb_edge)
        except Exception as e:
            failed.append((pdb_id, repr(e)))
            continue

        if g is None or g.num_nodes == 0:
            failed.append((pdb_id, "No"))
            continue

        g.pdb_id = pdb_id
        g.split = split_by_pdb[pdb_id]
        all_graphs.append(g)

    return all_graphs, missing, failed


if __name__ == "__main__":
    all_graphs, missing, failed = build_all_graphs()
    torch.save(all_graphs, DATA_DIR / "graphs.pt")

    print(f"{len(all_graphs)} graph, {len(missing)} missing, {len(failed)} mistakes")
    if failed:
        for pid, err in failed[:10]:
            print(" ", pid, err)
