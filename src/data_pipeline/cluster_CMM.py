import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from Bio.PDB import FastMMCIFParser
from Bio.PDB.NeighborSearch import NeighborSearch

import shutil
from tqdm import tqdm
import numpy as np
import subprocess
import json
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import matplotlib.pyplot as plt

from config import ROOT_DIR, DATA_DIR, RAW_ZN_DIR as RAW_DIR

warnings.filterwarnings("ignore", category=PDBConstructionWarning)

AA = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D",
    "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
    "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
    "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"
}

def fasta():
    df = pd.read_csv(DATA_DIR / 'CMM_validation_scored.csv')
    parser = FastMMCIFParser(QUIET=True)
    with open(DATA_DIR / "input.fasta", "w") as f:
        dx = [a.upper() for a in df[df["Non protein ligand"] == False]["pdb_id"].tolist()]

        for cif in tqdm(RAW_DIR.glob("*.cif"), desc="generiranje zaporedji"):
            if cif.stem in dx:
                try:
                    structure = parser.get_structure(cif.stem, str(cif))
                    seq_comb =[]
                    for chain in structure[0]:
                        res = [i.get_resname() for i in chain.get_residues() if i.get_resname() in AA]
                        seq = "".join([AA[i] for i in res])
                        seq_comb.append(seq)
                    if len("".join(seq_comb)) < 30:
                        continue
                    f.write(f">{cif.stem}\n{"".join(seq_comb)}\n")


                except Exception as e:

                    print(f"Napaka {cif.stem}")



MMSEQS_PATH_WSL = "/mnt/c/Users/aljaz/mmseqs/bin/mmseqs"



def to_wsl_path(windows_path):

    p = str(windows_path).replace("\\", "/")
    if p[1] == ":":
        drive = p[0].lower()
        p = f"/mnt/{drive}{p[2:]}"
    return p


def mmseqs2():
    fasta_wsl = to_wsl_path(DATA_DIR / "input.fasta")
    db_wsl = to_wsl_path(DATA_DIR / "db")
    clustered_wsl = to_wsl_path(DATA_DIR / "clustered")
    tmp_wsl = to_wsl_path(DATA_DIR / "Tmp")
    tsv_wsl = to_wsl_path(DATA_DIR / "tsv")

    subprocess.run([
        "wsl", MMSEQS_PATH_WSL, "createdb", fasta_wsl, db_wsl
    ], check=True)

    subprocess.run([
        "wsl", MMSEQS_PATH_WSL, "cluster",
        db_wsl, clustered_wsl, tmp_wsl,
        "--min-seq-id", "0.3",
        "-c", "0.8",
        "--cov-mode", "1"
    ], check=True)

    subprocess.run([
        "wsl", MMSEQS_PATH_WSL, "createtsv",
        db_wsl, db_wsl, clustered_wsl, tsv_wsl
    ], check=True)

    return 0


def write_scored():
    df = pd.read_csv(DATA_DIR / 'CMM_validation_scored.csv')
    with open(DATA_DIR/"tsv") as f:
        for line in tqdm(f):
            x = line.strip().split()

            df.loc[(df["pdb_id"] == x[1].lower()) ,"Cluster"] = x[0].lower()

        df.to_csv(DATA_DIR / "CMM_validation_scored.csv", index=False)


def reorder():
    df = pd.read_csv(DATA_DIR / 'CMM_validation_split_clustered.csv')
    averages = df.groupby(["Cluster","pdb_id"], as_index=False)["Total_score"].mean()

    b = averages.loc[averages.groupby("Cluster")["Total_score"].idxmax()]
    b = b[b["Total_score"] > 0.7]

    df = df[df["pdb_id"].isin(b["pdb_id"])]

    df.to_csv(DATA_DIR / "CMM_validation_split_clustered.csv", index=False)

    return 0


MARKED = DATA_DIR / "CMM_validation_split_clustered.csv"


def split(seed=42):
    df = pd.read_csv(MARKED)
    ids = sorted(df["pdb_id"].unique())      # sorted = ponovljivo
    rng = random.Random(seed)
    rng.shuffle(ids)

    n = len(ids)
    n_train, n_val = int(0.8 * n), int(0.1 * n)
    assign = {}
    for pid in ids[:n_train]:               assign[pid] = "train"
    for pid in ids[n_train:n_train+n_val]:  assign[pid] = "val"
    for pid in ids[n_train+n_val:]:         assign[pid] = "test"

    df["SPLIT"] = df["pdb_id"].map(assign)
    df.to_csv(MARKED, index=False)
    print(df["SPLIT"].value_counts())

#fasta()
#mmseqs2()
#write_scored()
reorder()
split()