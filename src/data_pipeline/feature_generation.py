from pathlib import Path
from Bio.PDB import FastMMCIFParser
from torch_geometric.data import Data
from Bio.PDB.NeighborSearch import NeighborSearch
import torch
import numpy as np
import json
import traceback
from tqdm import tqdm
import pandas as pd
from Bio.PDB.SASA import ShrakeRupley
import copy


AA = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D",
    "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
    "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
    "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"
}
AA_IDX = {aa: i for i, aa in enumerate(AA)}


#selected donors in the sidechain, most likely to coordinate with the metal ion
sidechain_donors = {
    "ALA": [],
    "ARG": ["NH1", "NH2"],
    "ASN": ["OD1", "ND2"],
    "ASP": ["OD1", "OD2"],
    "CYS": ["SG"],
    "GLN": ["OE1", "NE2"],
    "GLU": ["OE1", "OE2"],
    "GLY": [],
    "HIS": ["ND1", "NE2"],
    "ILE": [],
    "LEU": [],
    "LYS": ["NZ"],
    "MET": ["SD"],
    "PHE": [],
    "PRO": [],
    "SER": ["OG"],
    "THR": ["OG1"],
    "TRP": ["NE1"],
    "TYR": ["OH"],
    "VAL": []
}

#https://pmc.ncbi.nlm.nih.gov/articles/PMC3836772/
#maximum surface available to the solvent (water)
MAX_ASA = {
    "ALA": 129, "ARG": 274, "ASN": 195, "ASP": 193, "CYS": 167,
    "GLN": 225, "GLU": 223, "GLY": 104, "HIS": 224, "ILE": 197,
    "LEU": 201, "LYS": 236, "MET": 224, "PHE": 240, "PRO": 159,
    "SER": 155, "THR": 172, "TRP": 285, "TYR": 263, "VAL": 174,
}
#approximate charge of each residue
CHARGE = {
    "ASP": -1.0, "GLU": -1.0, "LYS": +1.0, "ARG": +1.0,
    "HIS": +0.1, "CYS": -0.1,
}

#cleaning the structure of all non protein residues
def clean_structure(structure):
    #deepcopy so the original strucutre doesn't get modified
    model = copy.deepcopy(structure[0])
    for chain in list(model):
        for res in list(chain):
            if res.get_id()[0] != " ":
                chain.detach_child(res.get_id())
            elif res.get_resname().strip() not in AA:
                chain.detach_child(res.get_id())
            elif not all(a in res for a in ("N", "CA", "C")):
                chain.detach_child(res.get_id())
        if len(chain) == 0:
            model.detach_child(chain.get_id())
    return model


#local coordinate system for the invariance of unit vectors
def local_frame(residue):
    ca, n, c = (residue[a].get_coord() for a in ("CA", "N", "C"))
    e1 = n - ca
    e1 = e1 / (np.linalg.norm(e1) + 1e-8)
    w = c - ca
    e2 = w - np.dot(w, e1) * e1
    e2 = e2 / (np.linalg.norm(e2) + 1e-8)
    return np.stack([e1, e2, np.cross(e1, e2)])

#local charge
def coulomb(point, charges, cutoff=12.0, eps_r=4.0):
    v = 0.0
    for coord, q in charges:
        d = np.linalg.norm(point - coord)
        if 1e-3 < d < cutoff:
            v += q / (eps_r * d)
    return v

#local density
def neighbor_density(point, ns, radius=8.0):
    return len(ns.search(point, radius)) / 100.0


def structure_context(structure):
    apo = clean_structure(structure)

    ShrakeRupley().compute(apo, level="R")

    sasa = {}
    for chain in apo:
        for res in chain:
            key = (chain.get_id(), res.get_id()[1])
            ref = MAX_ASA.get(res.get_resname().strip())
            if ref:
                sasa[key] = min(res.sasa / ref, 1.0)

    charged_atoms = []
    for chain in apo:
        for res in chain:
            q = CHARGE.get(res.get_resname().strip())
            if q is None:
                continue
            atoms = [res[a] for a in sidechain_donors.get(res.get_resname().strip(), [])
                     if a in res]
            if atoms:
                for a in atoms:
                    charged_atoms.append((a.get_coord(), q / len(atoms)))

    return {"sasa": sasa,
            "charges": charged_atoms,
            "apo": apo,
            "ns": NeighborSearch(list(apo.get_atoms()))}


def node_feature(residue, ctx, chain_id, include_cb_dist=True):
    resname = residue.get_resname().strip()
    if resname not in AA_IDX:
        return None
    if not all(a in residue for a in ("N", "CA", "C")):
        return None

    amino = [0] * len(AA)
    amino[AA_IDX[resname]] = 1

    ca = residue["CA"].get_coord()
    frame = local_frame(residue)

    donors = [residue[a].get_coord() for a in sidechain_donors.get(resname, [])
              if a in residue]
    if donors:
        centre = np.mean(donors, axis=0)
        diff = centre - ca
        dist = np.linalg.norm(diff)
        v_don = frame @ (diff / (dist + 1e-8))
        has_donor = 1.0
        probe = centre
    else:
        dist, v_don, has_donor = 0.0, np.zeros(3), 0.0
        probe = ca

    if "CB" in residue:
        cb = residue["CB"].get_coord() - ca
        cb_dist = np.linalg.norm(cb)
        v_cb = frame @ (cb / (cb_dist + 1e-8))
    else:
        cb_dist = 0.0
        v_cb = np.zeros(3)

    key = (chain_id, residue.get_id()[1])
    rsa = ctx["sasa"].get(key, 0.0)
    dens = neighbor_density(ca, ctx["ns"])
    pot = coulomb(probe, ctx["charges"])
    q_own = CHARGE.get(resname, 0.0)

    scalar_feats = [dist / 4.0, has_donor, rsa, dens, pot, q_own]
    if include_cb_dist:
        scalar_feats.append(cb_dist / 4.0)

    return (list(v_don) + list(v_cb) +scalar_feats +amino)
