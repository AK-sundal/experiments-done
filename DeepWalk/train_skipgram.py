import torch
import torch.optim as optim
from skip_gram import SkipGramModel
from random_walk import generate_walks


def train_skipgram(data, num_walks, walk_length, epochs, embedding_dim, learning_rate):
    walks = generate_walks(data, num_walks, walk_length)
    vocab = set(item for walk in walks for item in walk)
    vocab_size = len(vocab)

    # Compute frequencies
    frequencies = {node: 0 for node in vocab}
    for walk in walks:
        for node in walk:
            frequencies[node] += 1

    model = SkipGramModel(vocab_size, embedding_dim, frequencies)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        total_loss = 0
        for walk in walks:
            for i in range(1, len(walk) - 1):
                context = walk[i]
                targets = walk[i - 1 : i + 2]  # Previous and next nodes as targets
                targets.remove(context)  # Remove the context itself
                for target in targets:
                    optimizer.zero_grad()
                    context_tensor = torch.tensor(context, dtype=torch.long)
                    target_tensor = torch.tensor(target, dtype=torch.long)
                    loss = model(context_tensor, target_tensor)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(walks)}")

    return model.get_embeddings().detach().cpu().numpy()
