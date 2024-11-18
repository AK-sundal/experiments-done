# experiments-done
 Practice of implementation of  GAT-paper
 # Graph Attention Network (GAT)

This repository provides an implementation of the **Graph Attention Network (GAT)** as introduced in the paper:  
[Graph Attention Networks](https://arxiv.org/abs/1710.10903) by Petar Veličković et al. (2018).

---

## 📚 Paper Summary

GAT introduces attention mechanisms to graph-structured data, allowing nodes to attend over their neighbors' features with different weights. This enables better learning of graph representations, especially in cases of noisy or incomplete graphs.

---

## 🚀 Features

- Implementation of GAT using PyTorch.
- Flexible and scalable architecture for graph-based data.
- Support for common graph datasets like Cora, Citeseer, and PubMed.
- Easy-to-follow code for training, validation, and testing.

---

## 📂 Datasets

The following datasets are used in this implementation:
1. **Cora**: A citation network dataset where nodes represent publications, and edges denote citations.
2. **Citeseer**: Another citation network dataset with labeled documents.
3. **PubMed**: A citation dataset from the medical field.

Datasets are automatically downloaded and preprocessed using [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/).

---

## 🛠 Libraries and Tools

The implementation relies on the following libraries and tools:
- [Python 3.8+](https://www.python.org/)
- [PyTorch](https://pytorch.org/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [NetworkX](https://networkx.org/) (optional, for graph visualization)

To install the dependencies, run:

```bash
pip install -r requirements.txt

 
