import argparse
import csv
import itertools

import numpy as np
from PrettyPrint import PrettyPrintTree
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

    def __init__(self, data, root: int, alpha: float):
        n_combinations = 4
        n_datapoints = data.shape[0]
        n_variables = data.shape[1]

        # Initialize joint and marginal counts
        self.marginals = np.zeros(n_variables, dtype=float)
        self.joint_probs = np.zeros((n_combinations, n_variables, n_variables), dtype=float)

        # If root node index is not specified, pick a random index
        self.root_index = root if root is not None else np.random.randint(n_variables)
        self.root_node = None
        self.count_variable_occurrences(data)

        # Verify whether the sum of all occurrences corresponds to the dataset size and number of RVs
        assert (np.sum(self.joint_probs) == n_datapoints * n_variables**2 - n_datapoints * n_variables)

        # Apply Laplace smoothing
        self.laplace_smoothing(n_datapoints, n_combinations, alpha)

        # Compute the mutual information matrix
        mi_matrix = self.compute_mi_matrix(n_variables)

        # Create a maximum spanning tree from the graph of mutual information
        self.tree = minimum_spanning_tree(-mi_matrix)
        self.tree = breadth_first_order(self.tree, i_start=self.root_index, directed=False)

        # Determine the predecessors of each node in the tree
        self.predecessors = self.tree[1]
        self.predecessors[self.predecessors < 0] = -1  # -9999 --> -1

        # Create a tree structure from the predecessor array
        self.root_node = create_tree(self.predecessors)

        self.log_probs = np.zeros((len(self.predecessors), 2, 2), dtype=float)
        self.calc_log_params(root)
        self.probs = np.exp(self.log_probs)

        # Verify that all probabilities add up to 1
        assert (np.round(logsumexp(self.probs, axis=2), decimals=2).all() == 1)

    def calc_log_params(self, root: int):
        for node, parent in enumerate(self.predecessors):
            if parent == -1:  # Root node
                self.log_probs[node, 0, 0] = self.log_probs[node, 1, 0] = np.log(1 - self.marginals[root])
                self.log_probs[node, 0, 1] = self.log_probs[node, 1, 1] = np.log(self.marginals[root])
                continue
            self.log_probs[node, 0, 0] = np.log(self.joint_probs[0, node, parent] / (1 - self.marginals[parent]))
            self.log_probs[node, 0, 1] = np.log(
                self.joint_probs[1 if parent < node else 2, node, parent] / (1 - self.marginals[parent]))
            self.log_probs[node, 1, 0] = np.log(
                self.joint_probs[2 if parent < node else 1, node, parent] / self.marginals[parent])
            self.log_probs[node, 1, 1] = np.log(self.joint_probs[3, node, parent] / self.marginals[parent])

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

    def count_variable_occurrences(self, data: np.ndarray):
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

    def print_tree(self):
        if self.root_node is None:
            print('No tree structure available.')
            return
        printer = PrettyPrintTree(lambda x: x.children, lambda x: x.key)
        printer(self.root_node)

    def get_log_params(self):
        return self.log_probs

    def log_prob(self, queries: np.ndarray, exhaustive: bool = False):
        lp = np.zeros(queries.shape[0], dtype=float)
        for i, query in enumerate(queries):
            if exhaustive:
                # TODO implement
                pass
            else:
                # Message passing
                message = np.log(self.get_message(self.root_node, query))
                if verbose:
                    print(f'Log prob of query {query} is {message}\n')
                lp[i] = message
        return lp

    def sample_node(self, samples: np.ndarray, index: int, node: Node, parent_sample: int):
        # Sample a binary value with the conditional distribution probability
        sample = np.random.choice([0, 1], p=np.exp(self.log_probs[node.key, parent_sample, :]))
        samples[index, node.key] = sample
        # Generate samples for each of the children
        for child in node.children:
            self.sample_node(samples, index, child, sample)

    def sample(self, n_samples: int):
        samples = np.zeros((n_samples, self.get_log_params().shape[0]), dtype=int)
        for i in range(n_samples):
            self.sample_node(samples, i, self.root_node, 0)
        return samples

    def get_message(self, node: Node, query: np.ndarray):
        """Recursively compute the log probability of a query by passing messages through the tree"""
        parent = node.parent
        node_observed = query[node.key]
        parent_index = None
        parent_observed = None

        # The node is the root
        if parent is None:
            message = self.probs[node.key, 0, :] if np.isnan(node_observed) else self.probs[
                node.key, 0, int(node_observed)]
        else:
            parent_index = parent.key
            parent_observed = query[parent_index]

            # 4 options of retrieving the message according to whether the node and parent are observed
            if np.isnan(parent_observed) and np.isnan(node_observed):
                message = self.probs[node.key, :, :]
            elif np.isnan(parent_observed):
                message = self.probs[node.key, :, int(node_observed)]
            elif np.isnan(node_observed):
                message = self.probs[node.key, int(parent_observed), :]
            else:
                message = self.probs[node.key, int(parent_observed), int(node_observed)]

        # Retrieve the messages from the children nodes
        child_messages = [self.get_message(child, query) for child in node.children]

        # Avoid the *= operator since it modifies the original array
        message = message * np.prod(child_messages, axis=0)

        # Sum the messages if the node is root or the parent is node_observed
        if parent is None or not np.isnan(parent_observed) or len(message.shape) > 1 and message.shape[1] == 2:
            # Sum per condition if the node and its parent are not node_observed
            partial_sum = (np.isnan(node_observed) and parent_observed and np.isnan(parent_observed))
            message = np.sum(message, axis=1 if partial_sum else 0)
        if verbose:
            print(f"Message from {node.key} to {parent_index if parent_index is not None else -1} --> {message}")
        return message


def compute_MI(joint, marginal_1, marginal_2):
    """Compute the Mutual Information between two random variables"""
    return joint * np.log(joint / (marginal_1 * marginal_2))


def create_node(parents, i, created, root):
    """ Creates a node with key as 'i'. If i is root, then it changes root.
    If parent of i is not created, then it creates parent first """
    # If this node is already created
    if created[i] is not None:
        return

    # Create a new node and set created[i]
    created[i] = Node(i)

    # If 'i' is root, change root pointer and return
    if parents[i] == -1:
        root[0] = created[i]  # root[0] denotes root of the tree
        return

    # If parent is not created, then create parent first
    if created[parents[i]] is None:
        create_node(parents, parents[i], created, root)

    # Find parent pointer
    p = created[parents[i]]

    p.children.append(created[i])
    created[i].parent = p


def create_tree(predecessors: np.ndarray) -> Node:
    n = len(predecessors)

    # Keep track of created nodes, initialize all entries as None
    created = [None] * (n + 1)

    root = [Node(-1)]  # Init with dummy root
    for i in range(n):
        create_node(predecessors, i, created, root)

    return root[0]


def load_data(dataset_name: str, data_purpose: str, data_type: type) -> np.ndarray:
    assert data_purpose in ['train', 'test', 'marginals'], 'Unknown data purpose'
    file_path = f'data/{dataset_name}/{dataset_name}.{data_purpose}.data'
    with open(file_path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        data = np.array(list(reader)).astype(data_type)
    print(f'Found {len(data)} data entries in {file_path}')
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset', type=str, default='nltcs', help='Binary dataset to use',
        choices=['accidents', 'ad', 'baudio', 'bbd', 'bin-mnist', 'bnetflix', 'book', 'c20ng', 'cr52', 'cwebkb', 'dna',
                 'jester', 'kosarel', 'msweb', 'nltcs', 'plants', 'pumsb', 'tmovie', 'tretail']
    )
    parser.add_argument('-r', '--root', type=int, default=0, help='Index of the root node')
    parser.add_argument('-a', '--alpha', type=float, default=0.01,
                        help='Smoothing parameter, were 0 corresponds to no smoothing, and 1 to "add-one-smoothing"')
    parser.add_argument('-s', '--n_samples', type=int, default=1000, help='Number of samples to generate')
    parser.add_argument('-q', '--n_queries', type=int, default=100,
                        help='Number of queries to evaluate if no dataset is provided')
    parser.add_argument('-e', '--exhaustive', action='store_true', default=False,
                        help='Whether to use exhaustive inference')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='Whether to print extra information')
    args = parser.parse_args()
    verbose = args.verbose
    dataset = args.dataset
    print(f'Using dataset {dataset}')

    # For printing the results
    np.set_printoptions(linewidth=np.inf)

    # Load the data
    train_data = load_data(dataset, 'train', int)
    test_data = load_data(dataset, 'test', int)
    try:
        query_data = load_data(dataset, 'marginals', float)
    except FileNotFoundError:
        # Generate random queries
        query_data = np.random.choice([0, 1, np.nan], (args.n_queries, train_data.shape[1]))

    # Create the binary Chow-Liu trees
    print('Creating the Chow-Liu trees...')
    clt_train = BinaryCLT(train_data, root=args.root, alpha=args.alpha)
    clt_test = BinaryCLT(test_data, root=args.root, alpha=args.alpha)

    print('Train tree structure:')
    clt_train.print_tree()
    print('Test tree structure:')
    clt_test.print_tree()

    predecessors_train = clt_train.get_tree()
    predecessors_test = clt_test.get_tree()
    print(f'Array of train predecessor nodes: {predecessors_train}')
    print(f'Array of test predecessor nodes: {predecessors_test}')

    log_params_train = clt_train.get_log_params()
    log_params_test = clt_test.get_log_params()
    print(f'Log parameters of train tree: {log_params_train.round(3).tolist()}')
    print(f'Log parameters of test tree: {log_params_test.round(3).tolist()}')

    log_probs_train = clt_train.log_prob(query_data)
    log_probs_test = clt_test.log_prob(query_data)
    print(f'Resulting train log probabilities of evaluation queries: {log_probs_train.round(3).tolist()}')
    print(f'Resulting test log probabilities of evaluation queries: {log_probs_test.round(3).tolist()}')

    binary_samples_train = clt_train.sample(args.n_samples)
    binary_samples_test = clt_test.sample(args.n_samples)
    print(f'Train sample averages: {np.mean(binary_samples_train, axis=0).round(3)}')
    print(f'Test sample averages: {np.mean(binary_samples_test, axis=0).round(3)}')
