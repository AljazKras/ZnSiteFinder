import argparse
import sys
from pathlib import Path

# folder where this script is
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "temp"))

import numpy as np
import torch
from Bio.PDB import FastMMCIFParser

from config import ESM_PCA_PATH, MODELS_DIR
from src.core.GNN import ZnSiteFinder
from src.core.localize import predict_sites
from src.data_pipeline.esm_features import attach_esm_live, load_esm
from src.data_pipeline.feature_generation import sidechain_donors
from src.data_pipeline.graphing import AA, gen_graph


THRESHOLD = 0.5

# all trained models (one for each seed)
checkpoints = sorted(MODELS_DIR.glob("ZnSiteFinder_s*.pt"))


def find_cif(name):
    # if user gave a full path just use it
    if Path(name).is_file():
        return Path(name)

    # remove .cif from the end if it is there
    if name.lower().endswith(".cif"):
        name = name[:-4]
    name = Path(name).name

    # look in script folder and in current folder
    folders = [SCRIPT_DIR, Path.cwd()]
    for folder in folders:
        possible_names = [name + ".cif", name.upper() + ".cif", name.lower() + ".cif"]
        for file_name in possible_names:
            file_path = folder / file_name
            if file_path.is_file():
                return file_path

    print("ERROR: " + name + ".cif not found in " + str(SCRIPT_DIR) + " or " + str(Path.cwd()))
    sys.exit(1)


def get_donors(structure):
    # for every residue get the middle point of the donor atoms
    # and the first and last donor atom
    centres = []
    atoms = []
    for residue in structure.get_residues():
        resname = residue.get_resname().strip()

        # skip residues that are not normal amino acids or dont have backbone
        if resname not in AA:
            continue
        if "N" not in residue or "CA" not in residue or "C" not in residue:
            continue

        donors = []
        for atom_name in sidechain_donors[resname]:
            if atom_name in residue:
                donors.append(residue[atom_name].get_coord())

        # if no donor atoms use CA instead
        if len(donors) == 0:
            donors = [residue["CA"].get_coord()]

        centres.append(np.mean(donors, axis=0))
        atoms.append([donors[0], donors[-1]])

    centres = torch.tensor(np.array(centres), dtype=torch.float)
    atoms = torch.tensor(np.array(atoms), dtype=torch.float)
    return centres, atoms


def make_graph(cif_path):
    parser = FastMMCIFParser(QUIET=True)
    structure = parser.get_structure(cif_path.stem, str(cif_path))

    # only keep the first model (NMR files have more)
    models = list(structure)
    for model in models[1:]:
        structure.detach_child(model.id)

    data = gen_graph(structure, return_res_ids=True)
    if data is None:
        print("ERROR: " + cif_path.name + " has no normal protein residues")
        sys.exit(1)

    data.donor_pos, data.donor_atoms = get_donors(structure)

    # check that we have one donor for every node
    if len(data.donor_pos) != data.num_nodes:
        raise RuntimeError("number of donors (" + str(len(data.donor_pos)) +
                           ") is not the same as number of nodes (" + str(data.num_nodes) + ")")

    # add one extra column of zeros (model was trained with it)
    zeros = torch.zeros(data.num_nodes, 1)
    data.x = torch.cat([data.x, zeros], dim=1)

    # add ESM features
    esm_model = load_esm()
    data = attach_esm_live(data, data.res_id, esm_model, cache_path=ESM_PCA_PATH)
    return data


def load_models(in_channels, edge_channels, device):
    models = []
    for path in checkpoints:
        model = ZnSiteFinder(in_c=in_channels, out_c=2, hid=64, edge=edge_channels, heads=8,
                             layers=3, dropout=0.3, crf_iterations=5)
        model = model.to(device)
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        models.append(model)
    return models


# the models are stored here so run_models can use them
all_models = []


def run_models(model, data, device):
    # model is not used, we average all models instead
    probs = []
    for m in all_models:
        out = m(data.x, data.edge_index, data.edge_attr)
        probs.append(torch.softmax(out, dim=-1))
    mean_prob = torch.stack(probs).mean(dim=0)
    return mean_prob.log()


def sort_key(site):
    # reliable sites first, then highest confidence first
    return (not site["reliable"], -site["mean_conf"])


def print_results(name, sites, res_ids):
    if len(sites) == 0:
        print(name + ": no Zn sites predicted")
        return

    sites = sorted(sites, key=sort_key)
    print(name + ": " + str(len(sites)) + " predicted Zn site(s)")

    i = 1
    for site in sites:
        x, y, z = site["coord"]

        residue_names = []
        for n in sorted(site["residues"]):
            chain = res_ids[n][0]
            number = res_ids[n][1]
            aa = res_ids[n][2]
            residue_names.append(f"{aa}{number}:{chain}")
        residues = ", ".join(residue_names)

        if site["reliable"]:
            note = ""
        else:
            note = "  [<3 residues, low reliability]"

        print(f"  site {i}: conf={site['mean_conf']:.3f}  xyz=({x:.2f}, {y:.2f}, {z:.2f})  {residues}{note}")
        i = i + 1


def main():
    parser = argparse.ArgumentParser(description="Predict Zn binding sites from an mmCIF file.")
    parser.add_argument("--pdb", required=True, help="PDB id of a local <id>.cif, or a path")
    parser.add_argument("--threshold", type=float, default=THRESHOLD,
                        help="residue probability cutoff (default 0.5)")
    args = parser.parse_args()

    cif_path = find_cif(args.pdb)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    data = make_graph(cif_path)
    res_ids = data.res_id

    global all_models
    all_models = load_models(data.x.size(1), data.edge_attr.size(1), device)

    sites = predict_sites(None, data, device, cls_threshold=args.threshold,
                          forward_fn=run_models, cluster_mode="donor")
    print_results(cif_path.stem, sites, res_ids)


if __name__ == "__main__":
    main()