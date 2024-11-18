# experiments-done
 Practice of implementation of DeepWalk-paper
 # DeepWalk: Online Learning of Social Representations

This repository provides an implementation of **DeepWalk**, a novel approach for learning latent representations of nodes in a graph, as introduced in the paper:  
[DeepWalk: Online Learning of Social Representations](https://arxiv.org/abs/1403.6652)  
by Bryan Perozzi, Rami Al-Rfou, and Steven Skiena.

---

## 📚 Paper Overview

DeepWalk generalizes recent advancements in language modeling and deep learning by applying them to graph structures. By using truncated random walks to generate sequences of nodes and treating them as sentences, DeepWalk learns latent representations of nodes that capture graph structure and social relations in a continuous vector space.

### Key Features:
- **Unsupervised learning**: Learns node embeddings without requiring labeled data.
- **Truncated random walks**: Gathers local graph structure information.
- **Scalable**: Suitable for large graphs, with a parallelizable and online learning algorithm.
- **Robust**: Performs well in multi-label classification tasks, even with sparse labeled data.

---

## 🚀 Features

- **Graph-based representation learning**: Maps nodes in a graph to dense vectors in a continuous vector space.
- **Versatile applications**: Useful for network classification, anomaly detection, and other graph-based tasks.
- **Performance improvements**: Outperforms baseline methods with up to 10% higher F1 scores in sparse data settings.
- **Scalable and parallelizable**: Processes large-scale graphs efficiently.

---

## 📂 Datasets

DeepWalk has been evaluated on several social network datasets, including:
1. **BlogCatalog**: A network of bloggers and their interests.
2. **Flickr**: A network of users and shared media.
3. **YouTube**: A network of users connected via subscriptions.

Users can apply DeepWalk to any graph dataset by preparing it in an edge list format.

---

## 🛠 Libraries and Tools

The following libraries are used in this implementation:
- [Python 3.8+](https://www.python.org/)
- [NetworkX](https://networkx.org/) for graph operations.
- [Gensim](https://radimrehurek.com/gensim/) for Word2Vec modeling.
- [NumPy](https://numpy.org/) for numerical operations.
- [Scikit-learn](https://scikit-learn.org/) for evaluation.

To install all dependencies, run:

```bash
pip install -r requirements.txt

