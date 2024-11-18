# experiments-done
 Practice of implementation of GraphSage paper
 
 # GraphSAGE: Graph Sample and Aggregation

This repository provides an implementation of **GraphSAGE (Graph Sample and Aggregation)**, a framework for inductive learning on graph-structured data. GraphSAGE learns node representations by sampling and aggregating features from a node’s neighbors, making it suitable for large-scale graphs and out-of-sample nodes.

---

## 📚 Description

**GraphSAGE** is a framework for inductive graph embedding that generates embeddings for unseen nodes by learning a function that aggregates features from a node’s neighbors. Unlike traditional methods that require access to the full graph during training, GraphSAGE learns node representations by sampling neighbors, making it scalable and applicable to dynamic graphs.

- **Paper**: [GraphSAGE: Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216)
- **Key Features**:
  - **Inductive learning**: Can generate embeddings for unseen nodes.
  - **Scalable**: Scales to large graphs by sampling neighbors.
  - **Flexible aggregation functions**: Supports various aggregation functions (e.g., mean, LSTM, pooling, etc.).

---


