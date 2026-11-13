import functools
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn.functional as F
from Bio.PDB import FastMMCIFParser
from sklearn.decomposition import PCA
from tqdm import tqdm

import esm as esm_lib
from src.data_pipeline.graphing import node_res_ids, AA
from config import (ROOT_DIR, DATA_DIR, RAW_ZN_DIR as RAW_DIR, GRAPHS_PATH,
                    ESM_CACHE_PATH as CACHE_PATH, ESM_PCA_DIM, ESM_MODEL_NAME)
MAX_CHUNK = 1000


@torch.no_grad()
def embed_seq(model, alphabet, bc, device, seq):
    embs = []
    scs = []
    for i in range(0, len(seq), MAX_CHUNK):
        chunk = seq[i:i + MAX_CHUNK]
        _, _, tokens = bc([("s", chunk)])
        tokens = tokens.to(device)
        out = model(tokens, repr_layers=[model.num_layers], return_contacts=False)
        rep = out["representations"][model.num_layers][0, 1:len(chunk) + 1]
        logp = F.log_softmax(out["logits"][0], dim=-1)
        idx = []
        for a in chunk:
            idx.append(alphabet.get_idx(a))
        idx = torch.tensor(idx, device=device)
        sc = logp[1:len(chunk) + 1].gather(1, idx.unsqueeze(1)).squeeze(1)
        embs.append(rep.cpu())
        scs.append(sc.cpu())
    return torch.cat(embs, dim=0), torch.cat(scs, dim=0)


def get_feats(model, alphabet, bc, device, res_ids):
    embs = []
    scs = []
    start = 0
    n = len(res_ids)
    for end in range(1, n + 1):
        if end == n or res_ids[end][0] != res_ids[start][0]:
            seq = ""
            for r in res_ids[start:end]:
                seq += AA[r[2]]
            e, s = embed_seq(model, alphabet, bc, device, seq)
            embs.append(e)
            scs.append(s)
            start = end
    return torch.cat(embs, dim=0), torch.cat(scs, dim=0)


def build_cache(graphs_path=GRAPHS_PATH, out_path=CACHE_PATH, model_name=ESM_MODEL_NAME,
                pca_dim=ESM_PCA_DIM, limit=None, cif_lookup=None, reuse_pca=None):
    graphs = torch.load(graphs_path, weights_only=False)
    if limit:
        graphs = graphs[:limit]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("loading", model_name, device)
    model, alphabet = getattr(esm_lib.pretrained, model_name)()
    model = model.eval().to(device)
    bc = alphabet.get_batch_converter()

    parser = FastMMCIFParser(QUIET=True)
    cache = {}
    train = []
    failed = []

    for g in tqdm(graphs):
        pdb_id = g.pdb_id
        try:
            if cif_lookup is not None:
                path = cif_lookup(g)
            else:
                path = RAW_DIR / (pdb_id.upper() + ".cif")
            if path is None:
                raise FileNotFoundError(pdb_id)
            structure = parser.get_structure(pdb_id, path)
            for m in list(structure)[1:]:
                structure.detach_child(m.id)
            res_ids = node_res_ids(structure)
            if len(res_ids) != g.x.size(0):
                raise ValueError(str(len(res_ids)) + " vs " + str(g.x.size(0)))
            e, s = get_feats(model, alphabet, bc, device, res_ids)
        except Exception as ex:
            failed.append((pdb_id, repr(ex)))
            continue

        cache[pdb_id] = {"embed": e, "score": s}
        if g.split == "train":
            train.append(e)

    if failed:
        print(len(failed), "failed")
        for pid, err in failed[:10]:
            print(pid, err)

    if reuse_pca is not None:
        d = torch.load(reuse_pca, weights_only=False)
        components = d['pca_components']
        mean = d['pca_mean']
        embed_dim = d['embed_dim']
        pca_dim = d['pca_dim']
        print("using pca from", reuse_pca)
    else:
        X = torch.cat(train, dim=0).numpy()
        print("pca fit", X.shape)
        pca = PCA(n_components=pca_dim, random_state=0).fit(X)
        print(pca.explained_variance_ratio_.sum())
        components = torch.tensor(pca.components_, dtype=torch.float)
        mean = torch.tensor(pca.mean_, dtype=torch.float)
        embed_dim = X.shape[1]

    torch.save({
        "cache": cache,
        "pca_components": components,
        "pca_mean": mean,
        "embed_dim": embed_dim,
        "pca_dim": pca_dim,
        "model_name": model_name,
        "failed": failed,
    }, out_path)
    print("saved", out_path)
    return cache


_loaded = {}


def load_cache(p):
    if p not in _loaded:
        _loaded[p] = torch.load(p, weights_only=False)
    return _loaded[p]


def attach_esm_features(data, cache_path=CACHE_PATH, use_embed=True, use_score=True):
    if not use_embed and not use_score:
        return data
    if isinstance(cache_path, (str, Path)):
        paths = [cache_path]
    else:
        paths = list(cache_path)
    payload = None
    entry = None
    for p in paths:
        payload = load_cache(p)
        entry = payload["cache"].get(data.pdb_id)
        if entry is not None:
            break
    if entry is None:
        raise KeyError(data.pdb_id)

    out = data.clone()
    cols = []
    if use_embed:
        red = (entry["embed"] - payload["pca_mean"]) @ payload["pca_components"].T
        cols.append(red.to(dtype=out.x.dtype, device=out.x.device))
    if use_score:
        cols.append(entry["score"].unsqueeze(1).to(dtype=out.x.dtype, device=out.x.device))
    out.x = torch.cat([out.x[:, :-1]] + cols + [out.x[:, -1:]], dim=1)
    return out


def load_esm(model_name=ESM_MODEL_NAME, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, alphabet = getattr(esm_lib.pretrained, model_name)()
    model = model.eval().to(device)
    return model, alphabet, alphabet.get_batch_converter(), device


@torch.no_grad()
def attach_esm_live(data, res_ids, bundle, cache_path=CACHE_PATH, use_embed=True, use_score=True):
    if not use_embed and not use_score:
        return data
    model, alphabet, bc, device = bundle
    if len(res_ids) != data.x.size(0):
        raise ValueError("res_ids and x dont match")
    e, s = get_feats(model, alphabet, bc, device, res_ids)

    payload = load_cache(cache_path)
    out = data.clone()
    cols = []
    if use_embed:
        red = (e - payload["pca_mean"]) @ payload["pca_components"].T
        cols.append(red.to(dtype=out.x.dtype, device=out.x.device))
    if use_score:
        cols.append(s.unsqueeze(1).to(dtype=out.x.dtype, device=out.x.device))
    out.x = torch.cat([out.x[:, :-1]] + cols + [out.x[:, -1:]], dim=1)
    return out


if __name__ == "__main__":
    build_cache()