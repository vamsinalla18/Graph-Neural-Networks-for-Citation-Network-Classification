"""
main.py

Entry point for the GNN Cora Explainability project.
"""

from dataset import CoraDataset
from experiment import ExperimentManager
from models import ANN, GCN2, GCN3
from utils import get_device


def main():
    """Run the complete experiment pipeline."""

    dataset = CoraDataset()
    device = get_device()

    manager = ExperimentManager(
        dataset=dataset,
        device=device,
    )

    models = {
        "ANN": ANN(
            dataset.num_features(),
            dataset.num_classes(),
        ),
        "GCN2": GCN2(
            dataset.num_features(),
            dataset.num_classes(),
        ),
        "GCN3": GCN3(
            dataset.num_features(),
            dataset.num_classes(),
        ),
    }

    manager.run_all(models)
    manager.visualize_all()
    manager.explain_best_model()
    manager.save_results()

    print("\n" + "=" * 70)
    print("Pipeline completed successfully!")
    print("All outputs are available in the 'results/' directory.")
    print("=" * 70)


if __name__ == "__main__":
    main()