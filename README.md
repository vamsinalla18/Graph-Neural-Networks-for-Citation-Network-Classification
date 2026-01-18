# Graph Convolutional Networks for Citation Network Classification

This project explores **node classification on citation networks** using **Graph Convolutional Networks (GCN)** and compares their performance with a feature-only **Artificial Neural Network (ANN)** baseline on the **Cora dataset**.

---

## Problem Statement
Given a citation network where:
- Nodes represent scientific papers
- Edges represent citation relationships
- Each node has a bag-of-words feature vector

the goal is to **classify each paper into its research topic** by leveraging both **node features** and **graph structure**.

---

## Dataset
- **Dataset:** Cora Citation Network  
- **Nodes:** 2,708 papers  
- **Edges:** 5,429 citation links  
- **Features:** 1,433-dimensional binary word vectors  
- **Classes:** 7 research topics  

---

##  Models Implemented

### 1) Artificial Neural Network (ANN)
- Uses only node features
- Treats each paper independently
- Ignores citation relationships

### 2)Graph Convolutional Network (GCN)
- Aggregates information from neighboring nodes
- Exploits citation structure
- Implemented with 2-layer and 3-layer architectures

---

##  Training Details
- Framework: PyTorch & PyTorch Geometric  
- Loss Function: Cross Entropy Loss  
- Optimizer: Adam  
- Activation: Tanh  
- Early stopping based on validation loss  
- Random node splits for train / validation / test  

---

## 📈 Results

| Model              | Test Accuracy |
|--------------------|---------------|
| ANN (feature-only) | ~76%          |
| 2-layer GCN        | ~89%          |
| 3-layer GCN        | ~89%          |

Graph-based learning significantly improves classification performance by incorporating citation relationships.

---

##  Analysis
- GCN models outperform ANN by leveraging neighborhood aggregation.
- 2-layer and 3-layer GCNs achieve comparable performance, indicating most useful information lies within 2–3 hops.
- t-SNE visualization of node embeddings shows improved class-wise clustering for GCN compared to ANN.

---

##  Embedding Visualization
t-SNE is used to visualize learned node embeddings from the last hidden layer, highlighting the difference between feature-only and graph-aware representations.

---

