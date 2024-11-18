import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric
from torch_geometric.utils import train_test_split_edges
from torch_geometric.datasets import Planetoid


class DeepWalk(nn.Module):
    def __init__(self, num_nodes, embedding_dim):
        super(DeepWalk, self).__init__()
        self.embeddings = nn.Embedding(num_nodes, embedding_dim)

    def forward(self, target, context, negative_samples=None):
        target_emb = self.embeddings(target)  # shape: (batch_size, embedding_dim)
        context_emb = self.embeddings(context)  # shape: (batch_size, embedding_dim)

        # Compute the positive loss (dot product between target and context)
        positive_loss = torch.sum(
            target_emb * context_emb, dim=1
        )  # shape: (batch_size,)

        # If negative samples are provided, compute the negative loss
        if negative_samples is not None:
            # shape of negative_emb: (num_negative_samples, batch_size, embedding_dim)
            negative_emb = self.embeddings(
                negative_samples
            )  # shape: (num_neg_samples, batch_size, embedding_dim)

            # Compute negative loss: sum over negative samples
            # Unsqueeze target_emb to align dimensions for broadcasting: (batch_size, 1, embedding_dim)
            negative_loss = torch.sum(
                target_emb.unsqueeze(1) * negative_emb, dim=2
            )  # shape: (num_neg_samples, batch_size)

            # We compute the mean negative loss over negative samples
            negative_loss = negative_loss.mean(dim=0)  # shape: (batch_size,)

            # Subtract negative loss from the positive loss (negative sampling)
            loss = -(positive_loss - negative_loss).mean()
        else:
            loss = (
                -positive_loss.mean()
            )  # Standard Skip-Gram loss (no negative sampling)

        return loss


# Training loop
def train(model, data, optimizer, batch_size):
    model.train()
    total_loss = 0
    for i in range(0, data.num_nodes, batch_size):
        target = data.edge_index[0][i : i + batch_size]
        context = data.edge_index[1][i : i + batch_size]
        optimizer.zero_grad()

        # Prepare negative samples
        negative_samples = torch.randint(
            0, data.num_nodes, (data.num_edges, batch_size)
        )

        loss = model(target, context, negative_samples)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / (data.num_nodes // batch_size)


# Load Planetoid dataset (Cora example)
dataset = Planetoid(root="/tmp/Cora", name="Cora")
data = dataset[0]

# Define model, optimizer and other parameters
embedding_dim = 128
model = DeepWalk(num_nodes=data.num_nodes, embedding_dim=embedding_dim)

optimizer = optim.Adam(model.parameters(), lr=0.01)

# Hyperparameters
batch_size = 128
epochs = 100

# Train the model
for epoch in range(epochs):
    loss = train(model, data, optimizer, batch_size)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

# The model is now trained
