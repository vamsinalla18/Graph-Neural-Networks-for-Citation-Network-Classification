"""
models.py

Model definitions for the Cora node classification project.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

from config import (
    ANN_HIDDEN1,
    ANN_HIDDEN2,
    GCN2_HIDDEN,
    GCN3_HIDDEN1,
    GCN3_HIDDEN2,
)


# ---------------------------------------------------------
# Base Model
# ---------------------------------------------------------

class BaseModel(nn.Module):
    """
    Base class for all models.
    """

    def get_embeddings(self, x, edge_index=None):
        return self.forward(
            x,
            edge_index=edge_index,
            return_emb=True,
        )


# ---------------------------------------------------------
# ANN
# ---------------------------------------------------------

class ANN(BaseModel):

    def __init__(self, input_dim, num_classes):

        super().__init__()

        self.layer1 = nn.Linear(
            input_dim,
            ANN_HIDDEN1,
        )

        self.layer2 = nn.Linear(
            ANN_HIDDEN1,
            ANN_HIDDEN2,
        )

        self.layer3 = nn.Linear(
            ANN_HIDDEN2,
            num_classes,
        )

    def forward(
        self,
        x,
        edge_index=None,
        return_emb=False,
    ):

        h = torch.tanh(self.layer1(x))

        h = torch.tanh(self.layer2(h))

        if return_emb:
            return h

        return self.layer3(h)


# ---------------------------------------------------------
# Two Layer GCN
# ---------------------------------------------------------

class GCN2(BaseModel):

    def __init__(self, input_dim, num_classes):

        super().__init__()

        self.conv1 = GCNConv(
            input_dim,
            GCN2_HIDDEN,
        )

        self.conv2 = GCNConv(
            GCN2_HIDDEN,
            num_classes,
        )

    def forward(
        self,
        x,
        edge_index,
        return_emb=False,
    ):

        h = torch.tanh(
            self.conv1(
                x,
                edge_index,
            )
        )

        if return_emb:
            return h

        return self.conv2(
            h,
            edge_index,
        )


# ---------------------------------------------------------
# Three Layer GCN
# ---------------------------------------------------------

class GCN3(BaseModel):

    def __init__(self, input_dim, num_classes):

        super().__init__()

        self.conv1 = GCNConv(
            input_dim,
            GCN3_HIDDEN1,
        )

        self.conv2 = GCNConv(
            GCN3_HIDDEN1,
            GCN3_HIDDEN2,
        )

        self.conv3 = GCNConv(
            GCN3_HIDDEN2,
            num_classes,
        )

    def forward(
        self,
        x,
        edge_index,
        return_emb=False,
    ):

        h = torch.tanh(
            self.conv1(
                x,
                edge_index,
            )
        )

        h = torch.tanh(
            self.conv2(
                h,
                edge_index,
            )
        )

        if return_emb:
            return h

        return self.conv3(
            h,
            edge_index,
        )