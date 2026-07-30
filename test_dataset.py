from dataset import CoraDataset

dataset = CoraDataset()

dataset.summary()

dataset.show_node(0)

dataset.show_neighbors(0)

dataset.feature_statistics()

dataset.class_distribution()

dataset.degree_distribution()

dataset.visualize_graph()