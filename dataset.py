"""
dataset.py

Loads the Cora dataset and provides utilities for
dataset exploration and visualization.
"""

import os

import matplotlib.pyplot as plt
import networkx as nx
import torch

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.utils import degree, to_networkx

from config import (
    DATASET_NAME,
    DATA_DIR,
    VALIDATION_NODES,
    TEST_NODES,
)

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


class CoraDataset:

    def __init__(self):

        self.dataset = Planetoid(
            root=DATA_DIR,
            name=DATASET_NAME,
            transform=RandomNodeSplit(
                num_val=VALIDATION_NODES,
                num_test=TEST_NODES,
            ),
        )

        self.data = self.dataset[0]

    # --------------------------------------------------
    # Basic Getters
    # --------------------------------------------------

    def get_data(self):
        return self.data

    def num_features(self):
        return self.dataset.num_features

    def num_classes(self):
        return self.dataset.num_classes

    # --------------------------------------------------
    # Dataset Summary
    # --------------------------------------------------

    def summary(self):

        avg_degree = self.data.num_edges / self.data.num_nodes

        print("=" * 70)
        print("CORA DATASET SUMMARY")
        print("=" * 70)

        print(f"Dataset Name          : {DATASET_NAME}")
        print("Task                  : Node Classification")
        print("Node                  : Research Paper")
        print("Edge                  : Citation")
        print("Features              : Bag-of-Words")
        print()

        print(f"Number of Nodes       : {self.data.num_nodes}")
        print(f"Number of Edges       : {self.data.num_edges}")
        print(f"Number of Features    : {self.dataset.num_features}")
        print(f"Number of Classes     : {self.dataset.num_classes}")
        print(f"Average Degree        : {avg_degree:.2f}")
        print()

        print(f"Training Nodes        : {self.data.train_mask.sum().item()}")
        print(f"Validation Nodes      : {self.data.val_mask.sum().item()}")
        print(f"Testing Nodes         : {self.data.test_mask.sum().item()}")

        print("=" * 70)

    # --------------------------------------------------
    # Node Information
    # --------------------------------------------------

    def show_node(self, node_idx):

        print("\n" + "=" * 70)
        print(f"NODE {node_idx}")
        print("=" * 70)

        print("Class Label:")

        print(self.data.y[node_idx].item())

        print()

        print("Feature Vector Shape:")

        print(self.data.x[node_idx].shape)

        print()

        print("First 30 Feature Values:")

        print(self.data.x[node_idx][:30])

    # --------------------------------------------------
    # Neighbors
    # --------------------------------------------------

    def show_neighbors(self, node_idx):

        edge_index = self.data.edge_index

        neighbors = edge_index[1][edge_index[0] == node_idx]

        print("\nNeighbors of node", node_idx)

        print(neighbors.tolist())

        print("Total Neighbors:", len(neighbors))

    # --------------------------------------------------
    # Feature Statistics
    # --------------------------------------------------

    def feature_statistics(self):

        x = self.data.x

        print("\nFeature Statistics")

        print("-" * 40)

        print("Shape :", x.shape)

        print("Mean  :", x.mean().item())

        print("Std   :", x.std().item())

        print("Min   :", x.min().item())

        print("Max   :", x.max().item())

        sparsity = (x == 0).sum().item() / x.numel()

        print(f"Sparsity : {sparsity:.4f}")

    # --------------------------------------------------
    # Class Distribution
    # --------------------------------------------------

    def class_distribution(self, save=True):

        labels = self.data.y.numpy()

        plt.figure(figsize=(7, 5))

        plt.hist(
            labels,
            bins=self.dataset.num_classes,
            edgecolor="black",
        )

        plt.title("Class Distribution")

        plt.xlabel("Class")

        plt.ylabel("Number of Nodes")

        plt.tight_layout()

        if save:

            plt.savefig(
                os.path.join(
                    RESULTS_DIR,
                    "class_distribution.png",
                ),
                dpi=300,
            )

        plt.show()

        plt.close()

    # --------------------------------------------------
    # Degree Distribution
    # --------------------------------------------------

    def degree_distribution(self, save=True):

        deg = degree(
            self.data.edge_index[0],
            self.data.num_nodes,
        )

        plt.figure(figsize=(7, 5))

        plt.hist(
            deg.numpy(),
            bins=30,
            edgecolor="black",
        )

        plt.title("Degree Distribution")

        plt.xlabel("Degree")

        plt.ylabel("Frequency")

        plt.tight_layout()

        if save:

            plt.savefig(
                os.path.join(
                    RESULTS_DIR,
                    "degree_distribution.png",
                ),
                dpi=300,
            )

        plt.show()

        plt.close()

    # --------------------------------------------------
    # Graph Visualization
    # --------------------------------------------------

    def visualize_graph(self, num_nodes=150, save=True):

        subset = torch.arange(num_nodes)

        subgraph = self.data.subgraph(subset)

        G = to_networkx(
            subgraph,
            to_undirected=True,
        )

        plt.figure(figsize=(8, 8))

        pos = nx.spring_layout(
            G,
            seed=42,
        )

        nx.draw_networkx_nodes(
            G,
            pos,
            node_size=20,
            node_color="skyblue",
        )

        nx.draw_networkx_edges(
            G,
            pos,
            alpha=0.35,
        )

        plt.title(f"Cora Subgraph ({num_nodes} Nodes)")

        plt.axis("off")

        plt.tight_layout()

        if save:

            plt.savefig(
                os.path.join(
                    RESULTS_DIR,
                    "cora_graph.png",
                ),
                dpi=300,
            )

        plt.show()

        plt.close()