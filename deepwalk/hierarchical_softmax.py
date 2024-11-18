import torch
import torch.nn as nn
import heapq
from collections import defaultdict


class HierarchicalSoftmax(nn.Module):
    def __init__(self, vocab_size, embedding_dim, frequencies):
        super(HierarchicalSoftmax, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.build_tree(frequencies)

        self.classifiers = nn.ModuleDict()
        for parent in self.tree:
            self.classifiers[str(parent)] = nn.Linear(embedding_dim, 1)

    def build_tree(self, frequencies):
        heap = [[weight, [node]] for node, weight in frequencies.items()]
        heapq.heapify(heap)
        self.tree = defaultdict(list)

        while len(heap) > 1:
            lo = heapq.heappop(heap)
            hi = heapq.heappop(heap)
            for node in lo[1]:
                self.tree[node].append((len(heap), 0))  # Left child path
            for node in hi[1]:
                self.tree[node].append((len(heap), 1))  # Right child path
            heapq.heappush(heap, [lo[0] + hi[0], lo[1] + hi[1]])

    def forward(self, embeddings, target):
        path = self.tree[target.item()]
        prob = torch.tensor(1.0, device=embeddings.device)
        for parent, direction in path:
            classifier = self.classifiers[str(parent)]
            score = classifier(embeddings).squeeze()
            prob *= (
                torch.sigmoid(score) if direction == 1 else (1 - torch.sigmoid(score))
            )
        return prob
