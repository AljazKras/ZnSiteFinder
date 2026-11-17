import copy
import json
import random
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef
from tqdm import tqdm

from src.core.GNN import ZnSiteFinder
from src.core.localise import tune_threshold, evaluate_sites
from config import (ROOT_DIR, DATA_DIR, GRAPHS_PATH, APO_GRAPHS_PATH,
                    CLUSTER_MODE, CLUSTER_CUTOFF, GRAD_CLIP, LR)

seed = 1
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def simple_forward(model, data, device):
    return model(data)


def load_raw(graphs_path=None, mix_apo=False, hid=64, layers=3, heads=8, crf_iterations=5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Naprava: {device}")
    all_graphs = torch.load(graphs_path or GRAPHS_PATH, weights_only=False)
    apo_graphs = torch.load(APO_GRAPHS_PATH, weights_only=False) if mix_apo else []

    weight_src = all_graphs + list(apo_graphs)
    primary = sum((d.y[d.mask][:, 0] == 1).sum().item() for d in weight_src)
    nonbonded = sum((d.y[d.mask][:, 0] == 0).sum().item() for d in weight_src)
    counts = torch.clamp(torch.tensor([nonbonded, primary], dtype=torch.float), min=1.0)
    loss_weights = 1.0 / np.sqrt(counts)
    loss_weights = loss_weights / loss_weights.sum() * len(counts)

    train_set = [g.to(device) for g in all_graphs if g.split == "train"]
    val_set = [g.to(device) for g in all_graphs if g.split == "val"]
    test_set = [g.to(device) for g in all_graphs if g.split == "test"]
    apo_test_set = [g.to(device) for g in apo_graphs if g.split == "test"]
    if mix_apo:
        train_set += [g.to(device) for g in apo_graphs if g.split == "train"]
        val_set += [g.to(device) for g in apo_graphs if g.split == "val"]
        print(f"Grafi: holo={len(all_graphs)} apo={len(apo_graphs)} | "
              f"train={len(train_set)} val={len(val_set)} test holo={len(test_set)} apo={len(apo_test_set)}")
    else:
        print(f"Grafi: {len(all_graphs)} (train={len(train_set)} val={len(val_set)} test={len(test_set)})")

    model = ZnSiteFinder(
        in_c=all_graphs[0].x.shape[1], out_c=2, hid=hid,
        edge=all_graphs[0].edge_attr.shape[1], heads=heads, layers=layers, dropout=0.3,
        crf_iterations=crf_iterations,
    ).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"Model: hid={hid} layers={layers} heads={heads} -> {n_par/1e6:.2f}M parametrov")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    cls_criterion = nn.CrossEntropyLoss(weight=loss_weights.to(device))
    return train_set, val_set, test_set, apo_test_set, model, optimizer, cls_criterion, device


def run_epoch(model, graphs, optimizer, cls_criterion, device, training):
    model.train(training)
    total_loss, n = 0.0, 0
    y_true, y_pred = [], []

    order = list(range(len(graphs)))
    if training:
        random.shuffle(order)

    for idx in tqdm(order, desc="train" if training else "eval", leave=False):
        data = graphs[idx]
        if training:
            optimizer.zero_grad()

        out_cls = model(data)
        target_cls = data.y[data.mask][:, 0].long()
        logits = out_cls[data.mask]
        loss = cls_criterion(logits, target_cls)

        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

        total_loss += loss.item() * target_cls.numel()
        n += target_cls.numel()
        y_true.append(target_cls.detach().cpu())
        y_pred.append(logits.argmax(dim=1).detach().cpu())

    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()
    return {
        "loss": total_loss / n,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def main(epochs=80, patience=15, tag="v6", graphs_path=None, select_on="mcc",
         run_seed=None, mix_apo=False, hid=64, layers=3, heads=8, crf_iterations=5):
    if run_seed is not None:
        torch.manual_seed(run_seed); np.random.seed(run_seed); random.seed(run_seed)
        torch.cuda.manual_seed_all(run_seed)
    (train_set, val_set, test_set, apo_test_set, model,
     optimizer, cls_criterion, device) = load_raw(graphs_path, mix_apo, hid, layers, heads,
                                                   crf_iterations)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=LR * 0.02)

    best_val = -1.0
    best_model_state = None
    patience_ctr = 0
    history = []

    model_dir = ROOT_DIR / "models"
    model_dir.mkdir(exist_ok=True)
    ckpt_name = f"midas_{tag}.pt"
    hist_name = f"history_{tag}.json"

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_metrics = run_epoch(model, train_set, optimizer, cls_criterion, device, True)
        val_metrics = run_epoch(model, val_set, optimizer, cls_criterion, device, False)
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": train_metrics["loss"],
                        "lr": optimizer.param_groups[0]["lr"], **val_metrics})

        print(f"[{epoch:03d}] train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
              f"f1={val_metrics['f1']:.4f} P={val_metrics['precision']:.4f} "
              f"R={val_metrics['recall']:.4f} mcc={val_metrics['mcc']:.4f} ({time.time() - t0:.1f}s)")

        with open(DATA_DIR / hist_name, "w") as f:
            json.dump({"history": history}, f, indent=2)

        if val_metrics[select_on] > best_val:
            best_val = val_metrics[select_on]
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save({"model": best_model_state}, model_dir / ckpt_name)
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"Zgodnja ustavitev po {epoch} epohah.")
                break

    model.load_state_dict(best_model_state)
    test_metrics = run_epoch(model, test_set, optimizer, cls_criterion, device, False)
    print(f"TEST (node-level) f1={test_metrics['f1']:.4f} precision={test_metrics['precision']:.4f} "
          f"recall={test_metrics['recall']:.4f} mcc={test_metrics['mcc']:.4f}")

    apo_test_metrics = None
    if apo_test_set:
        apo_test_metrics = run_epoch(model, apo_test_set, optimizer, cls_criterion, device, False)
        print(f"TEST APO (node-level) f1={apo_test_metrics['f1']:.4f} "
              f"precision={apo_test_metrics['precision']:.4f} "
              f"recall={apo_test_metrics['recall']:.4f} mcc={apo_test_metrics['mcc']:.4f}")

    best_thr, _ = tune_threshold(model, val_set, device, forward_fn=simple_forward,
                                 cluster_mode=CLUSTER_MODE, cutoff=CLUSTER_CUTOFF)
    site_test = evaluate_sites(model, test_set, device, cls_threshold=best_thr["threshold"],
                               forward_fn=simple_forward, cluster_mode=CLUSTER_MODE, cutoff=CLUSTER_CUTOFF)
    print(f"TEST (site-level, threshold={best_thr['threshold']:.2f}) "
          f"precision={site_test['precision']:.3f} recall={site_test['recall']:.3f} "
          f"f1={site_test['f1']:.3f} (true={site_test['n_true']} pred={site_test['n_pred']} "
          f"matched={site_test['n_matched']})")

    with open(DATA_DIR / hist_name, "w") as f:
        json.dump({"history": history, "test": test_metrics, "apo_test": apo_test_metrics,
                   "mix_apo": mix_apo, "site_test": site_test,
                   "site_threshold": best_thr["threshold"], "select_on": select_on,
                   "epochs_run": len(history), "seed": run_seed or seed,
                   "hid": hid, "layers": layers, "heads": heads,
                   "crf_iterations": crf_iterations}, f, indent=2)
    return model, test_metrics, site_test


if __name__ == "__main__":
    main()