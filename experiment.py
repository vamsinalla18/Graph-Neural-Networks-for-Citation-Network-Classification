"""
experiment.py

Runs multiple experiments, visualizations, and explainability.
"""

import os

import pandas as pd

from trainer import Trainer
from explainability import GraphExplainer
from visualization import Visualizer
from utils import count_parameters

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


class ExperimentManager:

    def __init__(
        self,
        dataset,
        device,
    ):

        self.dataset = dataset
        self.data = dataset.get_data()
        self.device = device

        # Stores complete information for every trained model
        self.results = {}

    # --------------------------------------------------

    def run_model(
        self,
        model_name,
        model,
    ):

        print("\n" + "=" * 70)
        print(f"Training {model_name}")
        print("=" * 70)

        trainer = Trainer(
            model,
            self.device,
        )

        result = trainer.train(self.data)

        self.results[model_name] = {

            "model": trainer.model,

            "trainer": trainer,

            "history": result["history"],

            "metrics": result["metrics"],

            "embeddings": result["embeddings"],

            "training_time": result["training_time"],

            "parameters": count_parameters(model),

        }

    # --------------------------------------------------

    def run_all(
        self,
        models,
    ):

        for name, model in models.items():

            self.run_model(
                name,
                model,
            )

    # --------------------------------------------------

    def comparison_table(self):

        rows = []

        for name, result in self.results.items():

            rows.append({

                "Model": name,

                "Accuracy": result["metrics"]["accuracy"],

                "Parameters": result["parameters"],

                "Training Time (s)": round(
                    result["training_time"],
                    2,
                ),

            })

        return (

            pd.DataFrame(rows)

            .sort_values(
                by="Accuracy",
                ascending=False,
            )

            .reset_index(drop=True)

        )

    # --------------------------------------------------

    def visualize_all(self):

        print("\nGenerating visualizations...")

        for name, result in self.results.items():

            Visualizer.plot_training_curves(

                result["history"],

                name,

            )

            Visualizer.plot_confusion_matrix(

                result["metrics"]["confusion_matrix"],

                name,

            )

            Visualizer.plot_tsne(

                result["embeddings"],

                self.data.y.cpu(),

                name,

            )

    # --------------------------------------------------

    def _get_correct_test_node(
        self,
        metrics,
    ):

        preds = metrics["predictions"]

        labels = metrics["labels"]

        test_nodes = metrics["test_nodes"]

        correct = test_nodes[preds == labels]

        if len(correct) == 0:

            raise RuntimeError(
                "No correctly classified test node found."
            )

        return correct[0].item()

    # --------------------------------------------------

    def explain_best_model(self):

        best_model = max(

            self.results,

            key=lambda name:
            self.results[name]["metrics"]["accuracy"],

        )

        model_result = self.results[best_model]

        node_idx = self._get_correct_test_node(

            model_result["metrics"]

        )

        print(
            f"\nGenerating explanation using {best_model}"
        )

        print(
            f"Explaining correctly classified node {node_idx}"
        )

        explainer = GraphExplainer(

            model_result["model"],

            self.device,

        )

        result = explainer.explain_node(

            self.data,

            node_idx,

        )

        explainer.print_summary(result)

        summary_path = os.path.join(

            RESULTS_DIR,

            f"{best_model}_node{node_idx}_summary.txt",

        )

        explainer.save_summary(

            result,

            summary_path,

        )

        Visualizer.plot_feature_importance(

            result["feature_indices"],

            result["feature_scores"],

            best_model,

            node_idx,

        )

        Visualizer.plot_edge_importance(

            result["edge_pairs"],

            result["edge_scores"],

            best_model,

            node_idx,

        )

        Visualizer.plot_subgraph(

            result["edge_pairs"],

            result["edge_scores"],

            node_idx,

            best_model,

        )

        return result

    # --------------------------------------------------

    def save_results(self):

        df = self.comparison_table()

        csv_path = os.path.join(

            RESULTS_DIR,

            "model_results.csv",

        )

        df.to_csv(

            csv_path,

            index=False,

        )

        Visualizer.compare_models(df)

        print("\n" + "=" * 70)
        print("Model Comparison")
        print("=" * 70)

        print(df)

        print(f"\nResults saved to: {csv_path}")