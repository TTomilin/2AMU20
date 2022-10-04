import argparse
import csv
import itertools

import numpy as np
from numpy import ndarray
from scipy.sparse.csgraph import breadth_first_order
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.special import logsumexp
from tqdm import tqdm


class Node:
    """A basic single node building block of a tree structure"""

    def __init__(self, key):
        self.key = key
        self.parent = None
        self.children = []


class BinaryCLT:

    def __init__(self, data, root: int = None, alpha: float = 0.01):
        n_combinations = 4
        n_data_rows = data.shape[0]
        n_variables = data.shape[1]

        # Initialize joint and marginal counts
        self.marginals = np.zeros(n_variables, dtype=float)
        self.joint_probs = np.zeros((n_combinations, n_variables, n_variables), dtype=float)

        # If root is not specified, pick a random root node
        self.root = root if root is not None else np.random.randint(n_variables)
        self.count_variable_occurrences(data)

        # Verify whether the sum of all occurrences corresponds to the dataset size and number of RVs
        assert (np.sum(self.joint_probs) == n_data_rows * n_variables**2 - n_data_rows * n_variables)

        # Apply Laplace smoothing
        self.laplace_smoothing(n_data_rows, n_combinations, alpha)

        # Compute the mutual information matrix
        mi_matrix = self.compute_mi_matrix(n_variables)

        # Create a maximum spanning tree from the graph of mutual information
        self.tree = minimum_spanning_tree(-mi_matrix)
        self.tree = breadth_first_order(self.tree, i_start=self.root, directed=False)

        # Determine the predecessors of each node in the tree
        self.predecessors = self.tree[1]
        self.predecessors[self.predecessors < 0] = -1  # -9999 --> -1
        print(f'Array of predecessor nodes: {self.predecessors}')

        self.log_params = np.zeros((len(self.predecessors), 2, 2), dtype=float)
        self.calc_log_params(root)

        # Verify that all probabilities add up to 1
        assert (np.round(logsumexp(np.exp(self.log_params), axis=2), decimals=2).all() == 1)

    def calc_log_params(self, root: int):
        for node, parent in enumerate(self.predecessors):
            if parent == -1:  # Root node
                self.log_params[node, 0, 0] = self.log_params[node, 1, 0] = np.log(1 - self.marginals[root])
                self.log_params[node, 0, 1] = self.log_params[node, 1, 1] = np.log(self.marginals[root])
                continue
            self.log_params[node, 0, 0] = np.log(self.joint_probs[0, node, parent] / (1 - self.marginals[parent]))
            self.log_params[node, 0, 1] = np.log(
                self.joint_probs[1 if parent < node else 2, node, parent] / (1 - self.marginals[parent]))
            self.log_params[node, 1, 0] = np.log(
                self.joint_probs[2 if parent < node else 1, node, parent] / self.marginals[parent])
            self.log_params[node, 1, 1] = np.log(self.joint_probs[3, node, parent] / self.marginals[parent])

    def compute_mi_matrix(self, n_variables: int):
        mi_matrix = np.zeros((n_variables, n_variables), dtype=float)
        for var_1 in range(n_variables):
            for var_2 in range(var_1 + 1, n_variables):
                mi_matrix[var_1, var_2] = self.mi_entry(var_1, var_2)
        return mi_matrix

    def mi_entry(self, var_1, var_2):
        return compute_MI(self.joint_probs[0, var_1, var_2], 1 - self.marginals[var_1], 1 - self.marginals[var_2]) + \
               compute_MI(self.joint_probs[1, var_1, var_2], 1 - self.marginals[var_1], self.marginals[var_2]) + \
               compute_MI(self.joint_probs[2, var_1, var_2], self.marginals[var_1], 1 - self.marginals[var_2]) + \
               compute_MI(self.joint_probs[3, var_1, var_2], self.marginals[var_1], self.marginals[var_2])

    def laplace_smoothing(self, n_data_rows: int, n_combinations: int, alpha: float):
        self.joint_probs += alpha
        self.joint_probs /= (n_combinations * alpha + n_data_rows)
        self.marginals += 2 * alpha
        self.marginals /= (n_combinations * alpha + n_data_rows)

    def count_variable_occurrences(self, data: ndarray):
        for node, val in enumerate(tqdm(data)):
            # Increase marginal count if the RV = 1
            self.marginals[np.nonzero(val)] += 1
            # Increase joint count of all pairs of RVs
            self.increase_joint_occurrences(val, data.shape[1])

    def increase_joint_occurrences(self, val, n_variables: int):
        for pair in list(itertools.combinations(np.arange(n_variables), 2)):
            if not val[pair[0]] and not val[pair[1]]:
                self.joint_probs[0, pair[0], pair[1]] += 1
                self.joint_probs[0, pair[1], pair[0]] += 1
            elif not val[pair[0]] and val[pair[1]]:
                self.joint_probs[1, pair[0], pair[1]] += 1
                self.joint_probs[1, pair[1], pair[0]] += 1
            elif val[pair[0]] and not val[pair[1]]:
                self.joint_probs[2, pair[0], pair[1]] += 1
                self.joint_probs[2, pair[1], pair[0]] += 1
            elif val[pair[0]] and val[pair[1]]:
                self.joint_probs[3, pair[0], pair[1]] += 1
                self.joint_probs[3, pair[1], pair[0]] += 1

    def get_tree(self):
        return self.predecessors

    def get_log_params(self):
        return self.log_params

    def log_prob(self, queries: ndarray, exhaustive: bool = False):
        lp = np.zeros(queries.shape[0], dtype=float)
        for i, query in enumerate(queries):
            if exhaustive:
                # TODO implement
                pass
            else:
                # Message passing
                root = create_tree(self.predecessors)
                message = self.get_message(root, query)
                print(f'Log prob of query {query} is {message}\n')
                lp[i] = message
        return lp

    def sample(self, n_samples: int):
        # TODO implement
        pass

    def get_message(self, node: Node, query: ndarray):
        """Recursively compute the log probability of a query by passing messages through the tree"""
        parent = node.parent
        node_observed = query[node.key]
        parent_index = None
        parent_observed = None

        # The node is the root
        if parent is None:
            message = self.log_params[node.key, 0, :] if np.isnan(node_observed) else self.log_params[
                node.key, 0, int(node_observed)]
        else:
            parent_index = parent.key
            parent_observed = query[parent_index]

            # 4 options of retrieving the message according to whether the node and parent are observed
            if np.isnan(parent_observed) and np.isnan(node_observed):
                message = self.log_params[node.key, :, :]
            elif np.isnan(parent_observed):
                message = self.log_params[node.key, :, int(node_observed)]
            elif np.isnan(node_observed):
                message = self.log_params[node.key, int(parent_observed), :]
            else:
                message = self.log_params[node.key, int(parent_observed), int(node_observed)]

        # Retrieve the messages from the children nodes
        child_messages = [self.get_message(child, query) for child in node.children]

        # Avoid the *= operator since it modifies the original array
        message = message * np.prod(child_messages, axis=0)

        # Sum the messages if the node is root or the parent is node_observed
        if parent is None or not np.isnan(parent_observed) or len(message.shape) > 1 and message.shape[1] == 2:
            # Sum per condition if the node and its parent are not node_observed
            partial_sum = (np.isnan(node_observed) and parent_observed and np.isnan(parent_observed))
            message = np.sum(message, axis=1 if partial_sum else 0)
        print(f"Message from {node.key} to {parent_index if parent_index is not None else -1} --> {message}")
        return message


def compute_MI(joint, marginal_1, marginal_2):
    """Compute the Mutual Information between two random variables"""
    return joint * np.log(joint / (marginal_1 * marginal_2))


def create_node(parent, i, created, root):
    """ Creates a node with key as 'i'. If i is root, then it changes root.
    If parent of i is not created, then it creates parent first """
    # If this node is already created
    if created[i] is not None:
        return

    # Create a new node and set created[i]
    created[i] = Node(i)

    # If 'i' is root, change root pointer and return
    if parent[i] == -1:
        root[0] = created[i]  # root[0] denotes root of the tree
        return

    # If parent is not created, then create parent first
    if created[parent[i]] is None:
        create_node(parent, parent[i], created, root)

    # Find parent pointer
    p = created[parent[i]]

    p.children.append(created[i])
    created[i].parent = p


def create_tree(predecessors: ndarray) -> Node:
    n = len(predecessors)

    # Keep track of created nodes, initialize all entries as None
    created = [None] * (n + 1)

    root = [Node(-1)]  # Init with dummy root
    for i in range(n):
        create_node(predecessors, i, created, root)

    return root[0]


# Inorder traversal of tree
def print_tree(root):
    if not root.children:
        print(root.key, end=" ")
    for child in root.children:
        print_tree(child)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset', type=str, default='nltcs', help='Binary dataset to use',
        choices=['accidents', 'ad', 'baudio', 'bbd', 'bin-mnist', 'bnetflix', 'book', 'c20ng', 'cr52', 'cwebkb', 'dna',
                 'jester', 'kosarel', 'msweb', 'nltcs', 'plants', 'pumsb', 'tmovie', 'tretail']
    )
    parser.add_argument('-r', '--root', type=int, default=0, help='Index of the root node')
    parser.add_argument('-q', '--n_queries', type=int, default=5, help='Number of queries to test the model')
    parser.add_argument('-e', '--exhaustive', action='store_true', default='False',
                        help='Whether to use exhaustive inference')
    args = parser.parse_args()
    dataset = args.dataset
    print(f'Using dataset {dataset}')
    train_data = f'data/{dataset}/{dataset}.train.data'

    # Load the training data
    with open(train_data, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        data = np.array(list(reader)).astype(int)

    clt = BinaryCLT(data, root=args.root)
    tree = clt.get_tree()
    log_params = clt.get_log_params()

    test_queries = np.random.choice([0, 1, np.nan], (args.n_queries, data.shape[1]))
    log_probs = clt.log_prob(test_queries)
    print(f'Resulting log probabilities of test queries: {log_probs}')
