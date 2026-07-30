"""
explainability.py

Model explainability using the modern PyTorch Geometric Explainer API.
"""

import torch

from torch_geometric.explain import (
    Explainer,
    GNNExplainer,
    ModelConfig,
)

from config import (
    EXPLAIN_EPOCHS,
    TOP_K_FEATURES,
    TOP_K_EDGES,
)


class GraphExplainer:

    def __init__(self, model, device):

        self.model = model.to(device)
        self.device = device

        self.model.eval()

        self.explainer = Explainer(

            model=self.model,

            algorithm=GNNExplainer(
                epochs=EXPLAIN_EPOCHS,
            ),

            explanation_type="model",

            node_mask_type="attributes",

            edge_mask_type="object",

            model_config=ModelConfig(

                mode="multiclass_classification",

                task_level="node",

                return_type="raw",

            ),
        )

    # --------------------------------------------------

    def explain_node(
        self,
        data,
        node_idx,
    ):

        data = data.to(self.device)

        explanation = self.explainer(

            x=data.x,

            edge_index=data.edge_index,

            index=node_idx,

        )

        with torch.no_grad():

            logits = self.model(
                data.x,
                data.edge_index,
            )

        prediction = logits.argmax(dim=1)[node_idx].item()

        true_label = data.y[node_idx].item()

        # ---------------------------------------------
        # Top Features
        # ---------------------------------------------

        feature_scores = explanation.node_mask[node_idx]

        feature_values, feature_indices = torch.topk(

            feature_scores,

            k=min(
                TOP_K_FEATURES,
                feature_scores.numel(),
            ),
        )

        # ---------------------------------------------
        # Top Edges
        # ---------------------------------------------

        edge_scores = explanation.edge_mask

        edge_values, edge_indices = torch.topk(

            edge_scores,

            k=min(
                TOP_K_EDGES,
                edge_scores.numel(),
            ),
        )

        edge_pairs = []

        for idx in edge_indices:

            src = data.edge_index[0, idx].item()

            dst = data.edge_index[1, idx].item()

            edge_pairs.append(
                (src, dst)
            )

        # ---------------------------------------------
        # Sparsity
        # ---------------------------------------------

        sparsity = 1.0 - (

            len(feature_indices)

            / feature_scores.numel()

        )

        return {

            "node": node_idx,

            "prediction": prediction,

            "true_label": true_label,

            "feature_indices": feature_indices.cpu(),

            "feature_scores": feature_values.cpu(),

            "edge_pairs": edge_pairs,

            "edge_scores": edge_values.cpu(),

            "sparsity": sparsity,

            "fidelity": None,

            "explanation": explanation,

        }

    # --------------------------------------------------

    def print_summary(
        self,
        result,
    ):

        print()

        print("=" * 60)

        print(f"Explanation for Node {result['node']}")

        print("=" * 60)

        print(
            f"Prediction : {result['prediction']}"
        )

        print(
            f"True Label : {result['true_label']}"
        )

        print(
            f"Sparsity   : {result['sparsity']:.4f}"
        )

        print()

        print("Top Features")

        for idx, score in zip(

            result["feature_indices"],

            result["feature_scores"],

        ):

            print(
                f"Feature {idx.item():4d}"
                f" -> {score:.4f}"
            )

        print()

        print("Top Edges")

        for pair, score in zip(

            result["edge_pairs"],

            result["edge_scores"],

        ):

            print(
                f"{pair[0]} -> {pair[1]}"
                f" : {score:.4f}"
            )

        # --------------------------------------------------

    def save_summary(
        self,
        result,
        filepath,
    ):
        """
        Save the explanation summary to a text file.
        """

        with open(filepath, "w") as f:

            f.write("=" * 60 + "\n")
            f.write(f"Explanation for Node {result['node']}\n")
            f.write("=" * 60 + "\n\n")

            f.write(
                f"Prediction : {result['prediction']}\n"
            )

            f.write(
                f"True Label : {result['true_label']}\n"
            )

            f.write(
                f"Sparsity   : {result['sparsity']:.4f}\n"
            )

            if result["fidelity"] is not None:

                f.write(
                    f"Fidelity   : {result['fidelity']:.4f}\n"
                )

            f.write("\n")

            f.write("Top Features\n")
            f.write("-" * 30 + "\n")

            for idx, score in zip(

                result["feature_indices"],

                result["feature_scores"],

            ):

                f.write(

                    f"Feature {idx.item():4d}"
                    f" -> {score:.4f}\n"

                )

            f.write("\n")

            f.write("Top Edges\n")
            f.write("-" * 30 + "\n")

            for pair, score in zip(

                result["edge_pairs"],

                result["edge_scores"],

            ):

                f.write(

                    f"{pair[0]} -> {pair[1]}"
                    f" : {score:.4f}\n"

                )

        print(f"Explanation summary saved to: {filepath}")