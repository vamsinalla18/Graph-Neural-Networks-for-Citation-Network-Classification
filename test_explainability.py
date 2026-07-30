from dataset import CoraDataset
from models import GCN2
from trainer import Trainer
from explainability import GraphExplainer
from visualization import Visualizer
from utils import get_device


def main():

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    dataset = CoraDataset()

    data = dataset.get_data()

    # --------------------------------------------------
    # Device
    # --------------------------------------------------

    device = get_device()

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = GCN2(
        dataset.num_features(),
        dataset.num_classes(),
    )

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    trainer = Trainer(
        model,
        device,
    )

    trainer.train(data)

    # --------------------------------------------------
    # Explainability
    # --------------------------------------------------

    explainer = GraphExplainer(
        trainer.model,
        device,
    )

    result = explainer.explain_node(
        data,
        node_idx=10,
    )

    explainer.print_summary(result)

    # --------------------------------------------------
    # Visualizations
    # --------------------------------------------------

    Visualizer.plot_feature_importance(
        result["feature_indices"],
        result["feature_scores"],
        "GCN2",
        result["node"],
    )

    Visualizer.plot_edge_importance(
        result["edge_pairs"],
        result["edge_scores"],
        "GCN2",
        result["node"],
    )

    Visualizer.plot_subgraph(
        result["edge_pairs"],
        result["edge_scores"],
        result["node"],
        "GCN2",
    )

    print("\nSaved explanation visualizations to the 'results/' folder.")


if __name__ == "__main__":
    main()