"""
Configuration file for the Cora GNN Explainability Project.
Contains all experiment hyperparameters.
"""

# -----------------------
# Dataset
# -----------------------

DATASET_NAME = "Cora"
DATA_DIR = "data"

# -----------------------
# Data Split
# -----------------------

VALIDATION_NODES = 200
TEST_NODES = 500

# -----------------------
# Training
# -----------------------

EPOCHS = 100
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4

EARLY_STOPPING_PATIENCE = 10
MIN_IMPROVEMENT = 1e-4

# -----------------------
# Model Architecture
# -----------------------

GCN3_HIDDEN1 = 500
GCN3_HIDDEN2 = 100

GCN2_HIDDEN = 100

ANN_HIDDEN1 = 500
ANN_HIDDEN2 = 100

# -----------------------
# Visualization
# -----------------------

TSNE_COMPONENTS = 2
TSNE_PERPLEXITY = 30
TSNE_ITERATIONS = 1000
RANDOM_SEED = 42

# -----------------------
# Explainability
# -----------------------

EXPLAIN_NODE_INDEX = 0

TOP_K_FEATURES = 10
TOP_K_EDGES = 10

# -----------------------
# Explainability
# -----------------------

EXPLAIN_EPOCHS = 200

TOP_K_FEATURES = 10

TOP_K_EDGES = 10