import torch.nn as nn
from hierarchical_softmax import HierarchicalSoftmax


class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, frequencies):
        super(SkipGramModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.hierarchical_softmax = HierarchicalSoftmax(
            vocab_size, embedding_dim, frequencies
        )

    def forward(self, context, target):
        input_embedding = self.embeddings(context)
        return self.hierarchical_softmax(input_embedding, target)

    def get_embeddings(self):
        return self.embeddings.weight
