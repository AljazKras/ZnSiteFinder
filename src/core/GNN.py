import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv


class MeanFieldCRF(nn.Module):
    # Mean-field CRF over the residues graph.
    # Motivation: a single binding site is defined by multiple residues (typically 3-4), so labels of
    # spatial neighbours are correlated. The edge kernel was meant to learn pair-specific
    # couplings from their edge features.

    def __init__(self, num_classes, edge_dim, iterations=5, hidden=16):
        super().__init__()
        self.iterations = iterations
        self.edge_kernel = nn.Sequential(
            nn.Linear(edge_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.compat = nn.Parameter(1.0 - torch.eye(num_classes)) # (K x K) -> learnable

    def forward(self, unary, edge_index, edge_a):
        src, dst = edge_index
        edge_w = F.softplus(self.edge_kernel(edge_a)).reshape(-1)

        logits = unary
        for _ in range(self.iterations):
            q = F.softmax(logits, dim=-1)
            msg = q[src] * edge_w.unsqueeze(-1)
            agg = torch.zeros_like(q).index_add_(0, dst, msg)
            pairwise = agg @ self.compat
            logits = unary - pairwise

        return logits  # no need for normalisation


class ZnSiteFinder(nn.Module):

    def __init__(self, in_c, out_c, hid, edge, heads, layers, dropout, crf_iterations=5):
        super().__init__()

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        #First
        self.convs.append(TransformerConv(in_channels=in_c, out_channels=hid, heads=heads, concat=True,
                                          edge_dim=edge, dropout=dropout, beta=True))
        self.norms.append(nn.LayerNorm(hid * heads))
        #Between
        for _ in range(layers - 2):
            self.convs.append(TransformerConv(in_channels=hid * heads, out_channels=hid, heads=heads, concat=True,
                                              edge_dim=edge, dropout=dropout, beta=True))
            self.norms.append(nn.LayerNorm(hid * heads))
        #Last
        self.convs.append(TransformerConv(in_channels=hid * heads, out_channels=hid, heads=heads, concat=False,
                                          edge_dim=edge, dropout=dropout, beta=True))
        self.norms.append(nn.LayerNorm(hid))

        self.dropout = dropout
        self.cls_head = nn.Linear(in_features=hid, out_features=out_c)
        self.crf = MeanFieldCRF(num_classes=out_c, edge_dim=edge, iterations=crf_iterations) if crf_iterations > 0 else None

    def encode(self, x, edge_i, edge_a):
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_i, edge_attr=edge_a)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, x, edge_i, edge_a):
        x = self.encode(x, edge_i, edge_a)

        unary = self.cls_head(x)
        return self.crf(unary, edge_i, edge_a) if self.crf is not None else unary
