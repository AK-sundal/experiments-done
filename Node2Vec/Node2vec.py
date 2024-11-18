import os.path as osp
import sys

import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import Node2Vec

# Load the Cora dataset
path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data", "Planetoid")
dataset = Planetoid(path, name="Cora")
data = dataset[0]

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define the Node2Vec model
model = Node2Vec(
    data.edge_index,
    embedding_dim=128,
    walk_length=20,
    context_size=10,
    walks_per_node=10,
    num_negative_samples=1,
    p=1.0,
    q=1.0,
    sparse=True,
).to(device)

# Set up the data loader and optimizer
num_workers = 4 if sys.platform == "linux" else 0
loader = model.loader(batch_size=128, shuffle=True, num_workers=num_workers)
optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)


# Training function
def train():
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# Testing function
@torch.no_grad()
def test():
    model.eval()
    z = model()  # Get node embeddings
    acc = model.test(
        train_z=z[data.train_mask],
        train_y=data.y[data.train_mask],
        test_z=z[data.test_mask],
        test_y=data.y[data.test_mask],
        max_iter=150,
    )
    return acc


# Training loop
for epoch in range(1, 101):
    loss = train()
    acc = test()
    print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Acc: {acc:.4f}")


# Function to visualize embeddings using t-SNE
@torch.no_grad()
def plot_points(colors):
    model.eval()
    z = model().cpu().numpy()  # Get the embeddings and move to CPU
    z = TSNE(n_components=2).fit_transform(z)  # Reduce dimensions with t-SNE
    y = data.y.cpu().numpy()  # Get labels

    plt.figure(figsize=(8, 8))
    for i in range(dataset.num_classes):
        plt.scatter(z[y == i, 0], z[y == i, 1], s=20, color=colors[i])
    plt.axis("off")
    plt.show()


# Define colors for visualization
colors = ["#ffc0cb", "#bada55", "#008080", "#420420", "#7fe5f0", "#065535", "#ffd700"]
plot_points(colors)
