"""
visualization.py

Visualization utilities.
"""

import os

import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE

from config import (
    TSNE_COMPONENTS,
    TSNE_PERPLEXITY,
    TSNE_ITERATIONS,
    RANDOM_SEED,
)

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


class Visualizer:

    # --------------------------------------------------

    @staticmethod
    def plot_training_curves(history, model_name):

        plt.figure(figsize=(8, 5))

        plt.plot(history["train_loss"], label="Train Loss")
        plt.plot(history["val_loss"], label="Validation Loss")
        plt.plot(history["val_accuracy"], label="Validation Accuracy")

        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title(f"{model_name} Training")

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                RESULTS_DIR,
                f"{model_name}_training.png",
            ),
            dpi=300,
        )

        plt.close()

    # --------------------------------------------------

    @staticmethod
    def plot_confusion_matrix(confusion, model_name):

        plt.figure(figsize=(6, 6))

        plt.imshow(confusion, cmap="Blues")

        plt.title(f"{model_name} Confusion Matrix")

        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")

        plt.colorbar()

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                RESULTS_DIR,
                f"{model_name}_confusion.png",
            ),
            dpi=300,
        )

        plt.close()

    # --------------------------------------------------

    @staticmethod
    def plot_tsne(
        embeddings,
        labels,
        model_name,
    ):

        tsne = TSNE(
            n_components=TSNE_COMPONENTS,
            perplexity=TSNE_PERPLEXITY,
            max_iter=TSNE_ITERATIONS,
            random_state=RANDOM_SEED,
        )

        embedding2d = tsne.fit_transform(
            embeddings.numpy()
        )

        plt.figure(figsize=(8, 6))

        scatter = plt.scatter(
            embedding2d[:, 0],
            embedding2d[:, 1],
            c=labels,
            cmap="tab10",
            s=12,
            alpha=0.8,
        )

        plt.title(
            f"{model_name} Node Embeddings (t-SNE)"
        )

        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")

        plt.colorbar(scatter)

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                RESULTS_DIR,
                f"{model_name}_tsne.png",
            ),
            dpi=300,
        )

        plt.close()

    # --------------------------------------------------

    @staticmethod
    def compare_models(df):

        plt.figure(figsize=(8, 5))

        plt.bar(
            df["Model"],
            df["Accuracy"],
        )

        plt.ylabel("Accuracy")
        plt.title("Model Comparison")

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                RESULTS_DIR,
                "model_comparison.png",
            ),
            dpi=300,
        )

        plt.close()

    # --------------------------------------------------
    # Explainability
    # --------------------------------------------------

    @staticmethod
    def plot_feature_importance(
        feature_indices,
        feature_scores,
        model_name,
        node_idx,
    ):

        plt.figure(figsize=(8, 5))

        labels = [
            f"F{i.item()}"
            for i in feature_indices
        ]

        plt.barh(
            labels,
            feature_scores.numpy(),
        )

        plt.xlabel("Importance")

        plt.title(
            f"{model_name} - Node {node_idx} Feature Importance"
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                RESULTS_DIR,
                f"{model_name}_node{node_idx}_features.png",
            ),
            dpi=300,
        )

        plt.close()

    # --------------------------------------------------

    @staticmethod
    def plot_edge_importance(
        edge_pairs,
        edge_scores,
        model_name,
        node_idx,
    ):

        plt.figure(figsize=(10, 5))

        labels = [
            f"{u}->{v}"
            for u, v in edge_pairs
        ]

        plt.barh(
            labels,
            edge_scores.numpy(),
        )

        plt.xlabel("Importance")

        plt.title(
            f"{model_name} - Node {node_idx} Important Edges"
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                RESULTS_DIR,
                f"{model_name}_node{node_idx}_edges.png",
            ),
            dpi=300,
        )

        plt.close()

    # --------------------------------------------------

    @staticmethod
    def plot_subgraph(
        edge_pairs,
        edge_scores,
        center_node,
        model_name,
    ):

        G = nx.Graph()

        for (u, v), score in zip(
            edge_pairs,
            edge_scores,
        ):

            G.add_edge(
                int(u),
                int(v),
                weight=float(score),
            )

        plt.figure(figsize=(8, 8))

        pos = nx.spring_layout(
            G,
            seed=42,
        )

        widths = [
            G[u][v]["weight"] * 4
            for u, v in G.edges()
        ]

        colors = [
            "red"
            if node == center_node
            else "skyblue"
            for node in G.nodes()
        ]

        nx.draw_networkx(
            G,
            pos,
            node_color=colors,
            width=widths,
            with_labels=True,
            node_size=700,
        )

        plt.title(
            f"{model_name} - Explanation Subgraph"
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                RESULTS_DIR,
                f"{model_name}_node{center_node}_subgraph.png",
            ),
            dpi=300,
        )

        plt.close()