# experiments-done
 Practice of implementation of node2-vec paper:
 this similar to example availabe in pyg i took from there for understanding 
 my implemention is not done yet
 # node2vec: Scalable Feature Learning for Networks

This repository provides an implementation of **node2vec**, a framework for learning low-dimensional representations for nodes in a graph, as introduced in the paper:  
[node2vec: Scalable Feature Learning for Networks](https://arxiv.org/abs/1607.00653)  
by Aditya Grover and Jure Leskovec.

---

## 📚 Paper Overview

**node2vec** is a feature learning framework that generates continuous feature representations for nodes in networks. By introducing a flexible biased random walk strategy, node2vec efficiently explores node neighborhoods to capture diverse structural roles and community structures.

### Key Features:
- **Biased random walks**: Flexibly explores neighborhoods with parameters `p` and `q` to control the walk's breadth-first or depth-first search.
- **Task-independent embeddings**: Generates feature representations that can be used for multiple downstream tasks such as node classification and link prediction.
- **Scalable**: Applicable to large networks due to its efficient random walk strategy and parallelizable implementation.

---

## 🚀 Features

- **Flexible neighborhood exploration**: Supports both structural equivalence and community detection by tuning the `p` and `q` parameters.
- **Scalable to large networks**: Handles large real-world networks efficiently.
- **Application versatility**: Works for tasks like node classification, link prediction, and visualization.

---

## 📂 Datasets

node2vec is evaluated on several real-world datasets, including:
1. **BlogCatalog**: A social network of bloggers.
2. **Wikipedia**: A co-editing network of Wikipedia pages.
3. **Protein-Protein Interaction (PPI)**: A biological network of protein interactions.

These datasets can be formatted as edge lists for input to the algorithm.

---

## 🛠 Libraries and Tools

The following tools and libraries are used in this implementation:
- [Python 3.8+](https://www.python.org/)
- [NetworkX](https://networkx.org/) for graph operations.
- [Gensim](https://radimrehurek.com/gensim/) for Word2Vec modeling.
- [NumPy](https://numpy.org/) for numerical computations.
- [Scikit-learn](https://scikit-learn.org/) for evaluation tasks.

To install all dependencies, run:

```bash
pip install -r requirements.txt

