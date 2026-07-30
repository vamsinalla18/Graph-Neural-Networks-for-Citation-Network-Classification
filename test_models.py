from dataset import CoraDataset
from models import ANN, GCN2, GCN3

dataset = CoraDataset()

data = dataset.get_data()

input_dim = dataset.num_features()
num_classes = dataset.num_classes()

models = {
    "ANN": ANN(input_dim, num_classes),
    "GCN2": GCN2(input_dim, num_classes),
    "GCN3": GCN3(input_dim, num_classes),
}

for name, model in models.items():

    print("-" * 40)

    print(name)

    logits = model(
        data.x,
        data.edge_index,
    )

    print("Output shape:", logits.shape)

    embeddings = model.get_embeddings(
        data.x,
        data.edge_index,
    )

    print("Embedding shape:", embeddings.shape)