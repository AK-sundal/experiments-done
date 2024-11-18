# experiments-done
 Practice of implementation of paper
 
 # Graph Representation Learning Algorithms

This repository contains my implementations of several state-of-the-art graph representation learning algorithms. These methods are designed for tasks like node classification, link prediction, and graph classification, and they demonstrate a variety of techniques to learn meaningful representations of graph-structured data.

---

## 📚 Implemented Algorithms

### 1. [Graph Attention Networks (GAT)](./gat)
- **Description**: Introduces attention mechanisms into graph neural networks, allowing for adaptive weighting of neighboring nodes during feature aggregation.
- **Paper**: [Graph Attention Networks](https://arxiv.org/abs/1710.10903)
- **Key Features**:
  - Attention mechanisms for node weighting.
  - Handles graphs with varying neighborhood sizes efficiently.
  - Demonstrates state-of-the-art results on node classification tasks.

### 2. [Non-convolutional GNN (RUM)](./rum)
- **Description**: A non-convolutional GNN architecture using random walks and memory units to process graph data.
- **Paper**: [Non-convolutional Graph Neural Networks](https://arxiv.org/abs/2408.00165)
- **Key Features**: No convolutions, memory-efficient, expressive.

### 3. [DiffPool](./diffpool)
- **Description**: A hierarchical graph pooling method that clusters nodes and generates hierarchical graph representations for graph-level tasks.
- **Paper**: [Hierarchical Graph Representation Learning with Differentiable Pooling](https://arxiv.org/abs/1806.08804)
- **Key Features**: Differentiable pooling for hierarchical representations.

### 4. [Semi-Supervised Classification with GCN](./gcn)
- **Description**: A semi-supervised learning framework using spectral graph convolutions for embedding nodes in a graph.
- **Paper**: [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)
- **Key Features**: Efficient graph convolution operations.

### 5. [node2vec](./node2vec)
- **Description**: Extends DeepWalk with biased random walks, allowing for flexible exploration of graph neighborhoods to capture diverse connectivity patterns.
- **Paper**: [node2vec: Scalable Feature Learning for Networks](https://arxiv.org/abs/1607.00653)
- **Key Features**: Biased random walks, flexible neighborhood definitions.

### 6. [DeepWalk](./deepwalk)
- **Description**: Learns latent representations of nodes by treating random walks on the graph as sentences and applying techniques from natural language processing.
- **Paper**: [DeepWalk: Online Learning of Social Representations](https://arxiv.org/abs/1403.6652)
- **Key Features**: Simple, scalable, unsupervised.

### 7. [GraphSAGE](./graphsage)
- **Description**: GraphSAGE (Graph Sample and Aggregation) is an inductive graph embedding method that generates node embeddings by sampling and aggregating features from a node’s neighbors.
- **Paper**: [GraphSAGE: Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216)
- **Key Features**:
  - Inductive learning: Can generate embeddings for unseen nodes.
  - Scalable: Samples neighbors for efficient learning.
  - Flexible aggregation: Supports multiple aggregation functions (e.g., mean, LSTM, pooling).


---
### some paper implemetation are not completed
#### with some not feasable due to hardware constraints and no idea how to execute
---
## 🛠 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/AK-sundal/experiments-done.git
   cd experiments-done

