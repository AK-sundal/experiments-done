# experiments-done
 Practice of implementation of DiffPool paper
 
 # Hierarchical Graph Representation Learning with Differentiable Pooling (DiffPool)

This repository provides an implementation of **DiffPool**, a differentiable graph pooling module introduced in the paper:  
[Hierarchical Graph Representation Learning with Differentiable Pooling](https://doi.org/10.48550/arXiv.1806.08804)  
by Rex Ying, Jiaxuan You, Christopher Morris, Xiang Ren, William L. Hamilton, and Jure Leskovec.

---

## 📚 Paper Overview

DiffPool addresses the limitations of flat graph neural networks (GNNs) by introducing a hierarchical pooling mechanism. It enables GNNs to learn hierarchical graph representations suitable for tasks like **graph classification**. The key features of DiffPool include:

- **Differentiable soft clustering**: Nodes are grouped into clusters in a soft, trainable manner.
- **Hierarchical graph representation**: Captures multi-scale information within graphs.
- **Improved performance**: Achieves state-of-the-art results on multiple graph classification benchmarks.

---

## 🚀 Features

- **Modular design**: DiffPool can be integrated with various GNN architectures in an end-to-end manner.
- **Improved accuracy**: Demonstrates a 5-10% improvement over existing pooling methods.
- **Scalable and efficient**: Suitable for large-scale graph classification tasks.

---

## 📂 Datasets

DiffPool has been evaluated on several benchmark datasets for graph classification, including:
1. **PROTEINS**
2. **D&D**
3. **NCI1**
4. **Mutagenicity**
5. **COLLAB**

Datasets can be downloaded and preprocessed using the [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) library.

---

## 🛠 Libraries and Tools

This implementation uses the following libraries:
- [Python 3.8+](https://www.python.org/)
- [PyTorch](https://pytorch.org/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [NetworkX](https://networkx.org/) (optional for graph analysis)

Install all required dependencies with:

```bash
pip install -r requirements.txt

