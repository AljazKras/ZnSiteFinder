import torch



import argparse
import subprocess
import sys
from pathlib import Path
import pandas as pd
from config import ROOT_DIR, SRC_DIR, CSV_DIR, DATA_MANAGMENT_DIR,DATA_DIR
from src.data_managment.download import down_all







#
#
#
DATASETS = {
    "holo": ("HOLO_sites.csv", "HOLO_SET"),
    "jcim": ("JCIM_benchmark.csv", "JCIM_BENCHMARK"),
    "apo": ("APO_MAPPED.csv", "APO_MAPPED"),
}


def download(names):
    for name in names:
        csv_name, out_name = DATASETS[name]
        df = pd.read_csv(CSV_DIR / csv_name)
        pdb_ids = df["pdb_id"].dropna().unique().tolist()
        down_all(pdb_ids, DATA_DIR / out_name)


def parse_args():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_down = sub.add_parser("download", help="downloads CIF files from a csv")
    p_down.add_argument("--holo", action="store_true", help="HOLO_sites.csv")
    p_down.add_argument("--jcim", action="store_true", help="JCIM_benchmark.csv")
    p_down.add_argument("--apo", action="store_true", help="JCIM_benchmark.csv")

    p_train = sub.add_parser("train", help="treniranje modela")
    p_train.add_argument("--epochs", type=int, default=100)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--model", type=str, default="s1", choices=["s1", "s2", "s3", "s4", "s5"])

    p_predict = sub.add_parser("predict", help="predict Zinc binding sites")

    return ap.parse_args()

def predict(structure: list):
    return 0
if __name__ == "__main__":
    args = parse_args()

    if args.cmd == "download":
        names = [n for n in DATASETS if getattr(args, n)] or list(DATASETS)
        download(names)
