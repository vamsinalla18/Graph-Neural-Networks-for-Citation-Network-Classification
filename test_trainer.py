from dataset import CoraDataset
from models import GCN3
from trainer import Trainer
from utils import get_device

dataset = CoraDataset()

data = dataset.get_data()

device = get_device()

model = GCN3(
    dataset.num_features(),
    dataset.num_classes(),
)

trainer = Trainer(
    model,
    device,
)

history, train_time = trainer.train(data)

results = trainer.test(data)

print()

print("Training Time:", train_time)

print("Accuracy:", results["accuracy"])

print("Confusion Matrix Shape")

print(results["confusion_matrix"].shape)