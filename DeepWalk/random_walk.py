import random



def random_walk(edge_index, start_node, walk_length):
    walk = [start_node]
    current_node = start_node

    for _ in range(walk_length - 1):
        # Get neighbors from the edge_index
        neighbors = edge_index[1][
            edge_index[0] == current_node
        ].tolist()  # Get neighbors for the current node
        if not neighbors:
            break
        next_node = random.choice(neighbors)
        walk.append(next_node)
        current_node = next_node  # Move to the next node

    return walk


def generate_walks(data, num_walks, walk_length):
    walks = []
    num_nodes = data.x.size(0)  # Get the number of nodes from the node features

    for node in range(num_nodes):
        for _ in range(num_walks):
            walk = random_walk(data.edge_index, node, walk_length)
            walks.append(walk)  # Ensure we append the walk as a list
    return walks
