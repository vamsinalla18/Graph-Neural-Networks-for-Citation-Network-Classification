# GNN Cora Explainability

A modular Graph Neural Network (GNN) project that performs node classification on the **Cora citation network** while providing **interpretable explanations** using **PyTorch Geometric's Explainer API (GNNExplainer)**.

The project compares multiple neural network architectures, visualizes their performance, and explains why a GCN predicts a particular node class.

---

## Features

* Modular project structure
* Multiple graph learning models

  * Artificial Neural Network (ANN)
  * 2-Layer Graph Convolutional Network (GCN2)
  * 3-Layer Graph Convolutional Network (GCN3)
* Automatic training pipeline
* Early stopping
* Model comparison
* Training and validation curves
* Confusion matrix visualization
* t-SNE embedding visualization
* Explainability using GNNExplainer
* Feature importance visualization
* Edge importance visualization
* Important subgraph visualization
* Automatic generation of experiment reports

---

## Project Structure

```text
gnn-cora-explainability/
│
├── config.py                 # Project configuration
├── dataset.py                # Loads the Cora dataset
├── models.py                 # ANN, GCN2, GCN3 models
├── trainer.py                # Training & evaluation
├── experiment.py             # Experiment manager
├── explainability.py         # GNN explainability
├── visualization.py          # Plotting utilities
├── utils.py                  # Helper functions
├── main.py                   # Entry point
│
├── results/
│   ├── model_results.csv
│   ├── model_comparison.png
│   ├── ANN_training.png
│   ├── GCN2_training.png
│   ├── GCN3_training.png
│   ├── ANN_confusion_matrix.png
│   ├── GCN2_confusion_matrix.png
│   ├── GCN3_confusion_matrix.png
│   ├── ANN_tsne.png
│   ├── GCN2_tsne.png
│   ├── GCN3_tsne.png
│   ├── GCN2_nodeXX_features.png
│   ├── GCN2_nodeXX_edges.png
│   ├── GCN2_nodeXX_subgraph.png
│   └── GCN2_nodeXX_summary.txt
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Dataset

The project uses the **Cora Citation Network** available through **PyTorch Geometric**.

### Dataset Statistics

| Property |  Value |
| -------- | -----: |
| Nodes    |  2,708 |
| Edges    | 10,556 |
| Features |  1,433 |
| Classes  |      7 |

Each node represents a research paper.

Each edge represents a citation relationship.

Each node belongs to one of seven research topics.

---

# Models

## ANN

A multilayer perceptron that ignores graph connectivity and uses only node features.

This model serves as a baseline.

---

## GCN2

A two-layer Graph Convolutional Network using graph neighborhood information.

Architecture:

```
Input
   ↓
GCNConv
   ↓
ReLU
   ↓
Dropout
   ↓
GCNConv
   ↓
Output
```

---

## GCN3

A deeper Graph Convolutional Network with three graph convolution layers.

Architecture:

```
Input
   ↓
GCNConv
   ↓
ReLU
   ↓
GCNConv
   ↓
ReLU
   ↓
GCNConv
   ↓
Output
```

---

# Explainability

The project uses the modern **PyTorch Geometric Explainer API** with **GNNExplainer**.

For a selected node, the explainer identifies:

* Most influential node features
* Most important graph edges
* Important neighborhood subgraph
* Prediction confidence
* Explanation sparsity

This helps understand **why** the model predicts a particular class.

---

# Generated Visualizations

After running the pipeline, the following outputs are automatically generated.

## Training Curves

* Training loss
* Validation loss
* Validation accuracy

---

## Confusion Matrix

Shows per-class classification performance.

---

## t-SNE Embeddings

Visualizes learned node embeddings in two dimensions.

---

## Feature Importance

Ranks the most influential node features for a prediction.

---

## Edge Importance

Displays the most influential graph edges.

---

## Important Subgraph

Visualizes the neighborhood responsible for the prediction.

---

## Model Comparison

Compares

* Accuracy
* Number of parameters
* Training time

---

# Results

A CSV file is automatically generated.

Example:

| Model | Accuracy | Parameters | Training Time (s) |
| ----- | -------: | ---------: | ----------------: |
| GCN2  |     0.82 |     24,903 |              8.13 |
| GCN3  |     0.81 |     37,959 |             11.46 |
| ANN   |     0.73 |     94,215 |              3.82 |

---

# Installation

Clone the repository

```bash
git clone https://github.com/<username>/gnn-cora-explainability.git

cd gnn-cora-explainability
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Usage

Run the complete experiment pipeline

```bash
python main.py
```

The pipeline automatically:

* Trains all models
* Evaluates performance
* Generates visualizations
* Compares models
* Explains the best-performing model
* Saves all outputs inside the `results/` directory

---

# Technologies Used

* Python
* PyTorch
* PyTorch Geometric
* NetworkX
* NumPy
* Pandas
* Matplotlib
* Scikit-learn

---


