from dataset import CoraDataset

from models import ANN

from experiment import ExperimentManager

from utils import get_device

dataset = CoraDataset()

device = get_device()

manager = ExperimentManager(
    dataset,
    device,
)

results = manager.run_model(
    "ANN",
    ANN(
        dataset.num_features(),
        dataset.num_classes(),
    ),
)

print()

print(manager.comparison_table())