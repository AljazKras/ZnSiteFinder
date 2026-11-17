import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from src.core.GNN import ZnSiteFinder
from config import (ROOT_DIR, DATA_DIR, GRAPHS_PATH, MODELS_DIR,
                    TRILATERATE_MIN_SIZE, SITE_MATCH_RADIUS, CLUSTER_CUTOFF)

MODEL_PATH = MODELS_DIR / "midas_best.pt"
CLS_THRESHOLD = 0.5


def trilaterate(pos, weights):
    w = weights / weights.sum()
    return (pos * w[:, None]).sum(axis=0)


def cluster_nodes(node_ids, edge_index):
    nodes = set(node_ids.tolist())
    adj = {}
    for i in nodes:
        adj[i] = []
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    for k in range(len(src)):
        if src[k] in nodes and dst[k] in nodes:
            adj[src[k]].append(dst[k])

    visited = set()
    clusters = []
    for start in nodes:
        if start in visited:
            continue
        stack = [start]
        comp = []
        visited.add(start)
        while stack:
            n = stack.pop()
            comp.append(n)
            for nb in adj[n]:
                if nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
        clusters.append(comp)
    return clusters


def cluster_positive_nodes(node_ids, data, mode="graph", cutoff=CLUSTER_CUTOFF):
    if mode == "graph":
        return cluster_nodes(node_ids, data.edge_index)
    if mode != "donor":
        raise ValueError("unknown mode " + str(mode))

    ids = node_ids.tolist()
    n = len(ids)
    if n == 0:
        return []
    if n == 1:
        return [[ids[0]]]

    pts = data.donor_pos[node_ids]
    close = (torch.cdist(pts, pts) <= cutoff).cpu().numpy()

    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        for j in range(i + 1, n):
            if close[i, j]:
                a = find(i)
                b = find(j)
                if a != b:
                    parent[a] = b

    groups = {}
    for i in range(n):
        r = find(i)
        if r not in groups:
            groups[r] = []
        groups[r].append(ids[i])
    return list(groups.values())


@torch.no_grad()
def predict_sites(model, data, device, cls_threshold=CLS_THRESHOLD, forward_fn=None,
                  cluster_mode="graph", cutoff=CLUSTER_CUTOFF):
    data = data.to(device)
    if forward_fn is not None:
        out = forward_fn(model, data, device)
    else:
        out = model(data.x, data.edge_index, data.edge_attr)

    probs = torch.softmax(out, dim=-1)[:, 1]
    cand = data.mask & (probs > cls_threshold)
    node_ids = cand.nonzero(as_tuple=True)[0]
    if len(node_ids) == 0:
        return []

    pos = data.donor_pos.cpu().numpy()
    conf = probs.cpu().numpy()

    preds = []
    for comp in cluster_positive_nodes(node_ids, data, cluster_mode, cutoff):
        comp = np.array(comp)
        coord = trilaterate(pos[comp], conf[comp])
        preds.append({
            "coord": coord,
            "residues": comp.tolist(),
            "n_residues": len(comp),
            "mean_conf": float(conf[comp].mean()),
            "reliable": len(comp) >= TRILATERATE_MIN_SIZE,
        })
    return preds


def match_sites_indexed(true_pos, preds, match_radius=SITE_MATCH_RADIUS):
    if len(true_pos) == 0 or len(preds) == 0:
        return set(), []

    coords = np.array([p["coord"] for p in preds])
    dmat = np.linalg.norm(true_pos[:, None, :] - coords[None, :, :], axis=-1)
    rows, cols = linear_sum_assignment(dmat)

    matched = set()
    errors = []
    for r, c in zip(rows, cols):
        if dmat[r, c] <= match_radius:
            matched.add(int(r))
            errors.append(dmat[r, c])
    return matched, errors


def match_sites(true_pos, preds, match_radius=SITE_MATCH_RADIUS):
    matched, errors = match_sites_indexed(true_pos, preds, match_radius)
    return len(matched), errors


def evaluate_sites(model, graphs, device, cls_threshold=CLS_THRESHOLD, match_radius=SITE_MATCH_RADIUS,
                   forward_fn=None, cluster_mode="graph", cutoff=CLUSTER_CUTOFF):
    n_true = 0
    n_pred = 0
    n_matched = 0
    errors = []
    for data in graphs:
        preds = predict_sites(model, data, device, cls_threshold=cls_threshold, forward_fn=forward_fn,
                              cluster_mode=cluster_mode, cutoff=cutoff)
        true_pos = data.metal_pos.cpu().numpy()
        n_true += len(true_pos)
        n_pred += len(preds)
        m, errs = match_sites(true_pos, preds, match_radius)
        n_matched += m
        errors.extend(errs)

    precision = n_matched / n_pred if n_pred else 0.0
    recall = n_matched / n_true if n_true else 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return {
        "n_true": n_true, "n_pred": n_pred, "n_matched": n_matched,
        "precision": precision, "recall": recall, "f1": f1,
        "loc_error_mean": float(np.mean(errors)) if errors else None,
        "loc_error_median": float(np.median(errors)) if errors else None,
    }


@torch.no_grad()
def diagnose_misses(model, graphs, device, cls_threshold=CLS_THRESHOLD, match_radius=SITE_MATCH_RADIUS,
                    forward_fn=None, cluster_mode="graph", cutoff=CLUSTER_CUTOFF):
    buckets = {"0": 0, "1": 0, "2+": 0}
    examples = []
    for data in graphs:
        d = data.to(device)
        if forward_fn is not None:
            out = forward_fn(model, d, device)
        else:
            out = model(d.x, d.edge_index, d.edge_attr)
        probs = torch.softmax(out, dim=-1)[:, 1].cpu().numpy()

        true_pos = d.metal_pos.cpu().numpy()
        if len(true_pos) == 0:
            continue
        preds = predict_sites(model, data, device, cls_threshold=cls_threshold, forward_fn=forward_fn,
                              cluster_mode=cluster_mode, cutoff=cutoff)
        matched, _ = match_sites_indexed(true_pos, preds, match_radius)

        pos = d.pos.cpu().numpy()
        y = d.y[:, 0].cpu().numpy()
        mask = d.mask.cpu().numpy()
        tp_idx = np.where(mask & (y == 1))[0]
        if len(tp_idx) > 0:
            dist = np.linalg.norm(pos[tp_idx][:, None, :] - true_pos[None, :, :], axis=-1)
            owner = dist.argmin(axis=1)
        else:
            owner = np.zeros(0, dtype=int)

        for site in range(len(true_pos)):
            if site in matched:
                continue
            res = tp_idx[owner == site]
            if len(res) > 0:
                n_hit = int((probs[res] > cls_threshold).sum())
            else:
                n_hit = 0
            if n_hit == 0:
                buckets["0"] += 1
            elif n_hit == 1:
                buckets["1"] += 1
            else:
                buckets["2+"] += 1
            examples.append({
                "pdb_id": getattr(data, "pdb_id", None),
                "site_idx": site,
                "n_true_residues": int(len(res)),
                "n_hit": n_hit,
            })

    return buckets, examples


def tune_threshold(model, val_graphs, device, thresholds=None, match_radius=SITE_MATCH_RADIUS,
                   forward_fn=None, cluster_mode="graph", cutoff=CLUSTER_CUTOFF):
    if thresholds is None:
        thresholds = list(np.linspace(0.05, 0.95, 19)) + [0.96, 0.97, 0.98, 0.99]

    best = {"threshold": CLS_THRESHOLD, "f1": -1.0}
    results = []
    for t in thresholds:
        m = evaluate_sites(model, val_graphs, device, cls_threshold=float(t), match_radius=match_radius,
                           forward_fn=forward_fn, cluster_mode=cluster_mode, cutoff=cutoff)
        m["threshold"] = float(t)
        results.append(m)
        if m["f1"] > best["f1"]:
            best = m
    return best, results


def load_model(all_graphs, device):
    model = ZnSiteFinder(
        in_c=all_graphs[0].x.shape[1],
        out_c=2,
        hid=64,
        edge=all_graphs[0].edge_attr.shape[1],
        heads=8,
        layers=3,
        dropout=0.3,
        crf_iterations=5,
    ).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    graphs = torch.load(GRAPHS_PATH, weights_only=False)
    val = [g for g in graphs if g.split == "val"]
    test = [g for g in graphs if g.split == "test"]

    model = load_model(graphs, device)

    print("tuning threshold on val")
    best, sweep = tune_threshold(model, val, device)
    print("best threshold", round(best["threshold"], 2), "f1", round(best["f1"], 3),
          "precision", round(best["precision"], 3), "recall", round(best["recall"], 3))
    for m in sweep:
        print(round(m["threshold"], 2), round(m["f1"], 3), round(m["precision"], 3),
              round(m["recall"], 3), m["n_pred"])

    m = evaluate_sites(model, test, device, cls_threshold=best["threshold"])
    print("TEST true", m["n_true"], "pred", m["n_pred"], "matched", m["n_matched"],
          "precision", round(m["precision"], 3), "recall", round(m["recall"], 3), "f1", round(m["f1"], 3))
    if m["loc_error_mean"] is not None:
        print("loc error mean", round(m["loc_error_mean"], 2), "median", round(m["loc_error_median"], 2))


if __name__ == "__main__":
    main()