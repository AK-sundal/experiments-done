from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


def evaluate_embeddings(embeddings, labels, train_nodes, test_nodes):
    classifier = LogisticRegression(solver="liblinear")
    classifier.fit(embeddings[train_nodes], labels[train_nodes])

    predictions = classifier.predict(embeddings[test_nodes])
    macro_f1 = f1_score(labels[test_nodes], predictions, average="macro")
    micro_f1 = f1_score(labels[test_nodes], predictions, average="micro")

    return macro_f1, micro_f1
