import argparse
import csv
import itertools
import re
import time
from typing import *

import numpy as np
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


class DictTree:
    def __init__(self, val, children=None):
        if children is None:
            if isinstance(val, Iterable) and not isinstance(val, str):
                children = val
                val = str(type(val)).removeprefix("<class '").removesuffix("'>").upper()
            else:
                children = []

        self.val = val
        if isinstance(children, dict):
            self.children = [DictTree(v, c) for v, c in children.items()]
        elif isinstance(children, Iterable) and not isinstance(children, str):
            self.children = [DictTree(x) for x in children]
        else:
            self.children = [children]


class PrettyPrintTree:
    VERTICAL = False
    HORIZONTAL = True

    def __init__(self,
                 get_children: Callable[[object], Iterable] = None,
                 get_val: Callable[[object], object] = None,
                 get_label: Callable[[object], object] = None,
                 *,
                 label_color="",
                 show_newline_literal: bool = False,
                 return_instead_of_print: bool = False,
                 trim=False,
                 start_message: Callable[[object], str] = None,
                 # color=Back.LIGHTBLACK_EX,
                 color='\x1b[100m',
                 border: bool = False,
                 max_depth: int = -1,
                 default_orientation: bool = False
                 ):
        # this is a lambda which returns a list of all the children
        # in order to support trees of different kinds eg:
        #   self.child_right, self.child_left... or
        #   self.children = []... or
        #   self.children = {}... or anything else

        self.get_children = (lambda x: x.children) if get_children is None else get_children
        self.get_node_val = (lambda x: x.value) if get_val is None else get_val
        self.get_label = get_label
        self.label_color = label_color
        # only display first x chars
        self.trim = trim
        # if true will display \n as \n and not as new lines
        self.show_newline = show_newline_literal
        self.dont_print = return_instead_of_print
        self.start_message = start_message
        self.color = color
        self.border = border
        self.max_depth = max_depth
        self.default_orientation = default_orientation

    def print_json(self, dic, name='JSON', max_depth: int = 0):
        if max_depth:
            self.max_depth = max_depth
        self.get_children = lambda x: x.children if isinstance(x, DictTree) else []
        self.get_node_val = lambda x: x.val if isinstance(x, DictTree) else x
        self(DictTree(name, dic))

    def __call__(self, node, max_depth: int = 0, orientation=None):
        if orientation is not None:
            self.default_orientation = orientation
        if self.default_orientation == PrettyPrintTree.HORIZONTAL and self.get_label is not None:
            print("""  ^                                                                    ^
 /^\\                                                                  /^\\
 /^\\                                                                  /^\\
//|\\\\     As of now labels are only supported for vertical trees     //|\\\\
//|\\\\                                                                //|\\\\
  |                                                                    |
""")
        if isinstance(node, dict) or isinstance(node, list) or isinstance(node, tuple):
            self.print_json(node, max_depth=max_depth)
            return
        if self.start_message is not None and not self.dont_print:
            print(self.start_message(node))
        if max_depth:
            self.max_depth = max_depth
        if self.default_orientation == PrettyPrintTree.HORIZONTAL:
            res = self.tree_to_str_horizontal(node)
        else:
            res = self.tree_to_str(node)

        is_node = lambda x: (x.startswith('[') or x.startswith('|') or
                             (x.startswith('│') and x.strip() != '│') or
                             len(x) > 1 and x[1:-1] == '─' * (len(x) - 2) and x[0] + x[-1] in ['┌┐', '└┘'])
        lines = ["".join(self.color_txt(x) if is_node(x) else x for x in line) for line in res]
        if self.dont_print:
            if self.start_message:
                return self.start_message(node) + "\n" + "\n".join(lines)
            return "\n".join(lines)
        print("\n".join(lines))

    def get_val(self, node):
        st_val = str(self.get_node_val(node))
        if self.trim and len(st_val) > self.trim:
            st_val = st_val[:self.trim] + "..."
        if self.show_newline:
            escape_newline = lambda match: '\\n' if match.group(0) == '\n' else '\\\\n'
            st_val = re.sub(r'(\n|\\n)', escape_newline, st_val)
        if '\n' not in st_val:
            return [[st_val]]
        lst_val = st_val.split("\n")
        longest = max(len(x) for x in lst_val)
        return [[f'{x}{" " * (longest - len(x))}'] for x in lst_val]

    def tree_to_str(self, node, depth=0):
        val = self.get_val(node)
        children = self.get_children(node)
        if not isinstance(children, list):
            children = list(children)
        if len(children) == 0:
            if len(val) == 1:
                res = [[f'[{val[0][0]}]']]
                if self.get_label:
                    self.put_label(node, res)
                return res
            res = self.format_box("", val)
            if self.get_label:
                self.put_label(node, res)
            return res
        to_print = [[]]
        spacing = 0
        if depth + 1 != self.max_depth:
            for child in children:
                child_print = self.tree_to_str(child, depth=depth + 1)
                for l, line in enumerate(child_print):
                    if l + 1 >= len(to_print):
                        to_print.append([])
                    if l == 0:
                        len_line = len("".join(line))
                        middle_of_child = len_line - sum(divmod(len(line[-1]), 2))
                        len_to_print_0 = len("".join(to_print[0]))
                        to_print[0].append((spacing - len_to_print_0 + middle_of_child) * " " + '┬')
                    to_print[l + 1].append(' ' * (spacing - len("".join(to_print[l + 1]))))
                    to_print[l + 1].extend(line)
                spacing = max(len("".join(x)) for x in to_print) + 1

            if len(to_print[0]) != 1:
                new_lines = "".join(to_print[0])
                space_before = len(new_lines) - len(new_lines.strip())
                new_lines = new_lines.strip()
                ln_of_stripped = len(new_lines)
                new_lines = " " * space_before + '┌' + new_lines[1:-1].replace(' ', '─') + '┐'
                pipe_pos = middle = len(new_lines) - sum(divmod(ln_of_stripped, 2))
                new_ch = {'─': '┴', '┬': '┼', '┌': '├', '┐': '┤'}[new_lines[middle]]
                new_lines = new_lines[:middle] + new_ch + new_lines[middle + 1:]
                to_print[0] = [new_lines]
            else:
                to_print[0][0] = to_print[0][0][:-1] + '│'
                pipe_pos = len(to_print[0][0]) - 1
            spacing = " " * (pipe_pos - sum(divmod(len(val[0][0]), 2)))
        else:
            spacing = ""

        if len(val) == 1:
            val = [[spacing, f'[{val[0][0]}]']]
        else:
            val = self.format_box(spacing, val)
        to_print = val + to_print

        if self.get_label:
            self.put_label(node, to_print)
        return to_print

    def put_label(self, node, res):
        label = self.get_label(node)
        label = ('|' + str(label) + '|') if label else '│'
        if len(label) > len(res[0][-1]):
            diff = len(label) - len(res[0][-1])
            for row in res:
                row[0] = " " * diff + row[0]
        if len(res[0]) == 2:
            d, m = divmod(len(res[0][-1]), 2)
            pos = len(res[0][0]) + d - (1 * (1 - m))
        else:
            pos = len(res[0][0]) // 2
        res.insert(0, [" " * pos, "│"])
        res.insert(0, [" " * (pos - len(label) // 2), label])

    def tree_to_str_horizontal(self, node, depth=0):
        val = self.get_val(node)
        children = self.get_children(node)
        if not isinstance(children, list):
            children = list(children)
        if len(children) == 0:
            if len(val) == 1:
                return [['-', '[' + val[0][0] + ']']]
            box = self.format_box("", val)
            box[0][0] = '-'
            for i in range(1, len(box)):
                box[i][0] = ' '
            return box
        to_print = []
        if depth + 1 != self.max_depth:
            for i, child in enumerate(children):
                if i != 0:
                    to_print.extend([[]])
                child_print = self.tree_to_str_horizontal(child, depth=depth + 1)
                to_print.extend(child_print)

        val_width = max(len("".join(x)) for x in val) + 2
        middle_children = sum(divmod(len(to_print), 2))
        middle_this = sum(divmod(len(val), 2))
        pos = max(0, middle_children - middle_this)
        first = last = -1
        added = False
        for r, row in enumerate(to_print[:pos]):
            if len("".join(row)) > 0 and "".join(row)[0] == '-':
                while row[0] == '':
                    row.pop(0)
                row[0] = ('┌' if first == -1 else '├') + row[0][1:]
                if first == -1:
                    first = r
                last = r
            elif first != -1:
                if row:
                    row[0] = '│' + row[0][1:]
                else:
                    row.append('│')
            row.insert(0, " " * (val_width + 1))  # + "├" "┬" "¬" "┼" "┤" "│" '┌'
        for r, row in enumerate(val):
            if len(to_print) == r + pos:
                to_print.append([])
            if "".join(to_print[r + pos]).startswith('-'):
                while to_print[r + pos][0] == '':
                    to_print[r + pos].pop(0)
                symbol = {'TT': '┌', 'TF': '├', 'FT': '┬', 'FF': '┼'}[str(added)[0] + str(first == -1)[0]]
                to_print[r + pos] = [symbol] + to_print[r + pos][1:]
                if first == -1:
                    first = r + pos
                last = r + pos
            else:
                if to_print[r + pos]:
                    to_print[r + pos][0] = ('┤' if not added else '│') + to_print[r + pos][0][1:]
                elif depth + 1 != self.max_depth:
                    to_print[r + pos].append('┤' if not added else '│')
                else:
                    to_print[r + pos].append('')
                if not added:
                    last = r + pos
            added = True
            to_print[r + pos].insert(0, '[' + "".join(row) + ']')
            if r + 1 == sum(divmod(len(val), 2)):
                to_print[r + pos].insert(0, '-')
            else:
                to_print[r + pos].insert(0, ' ')
            to_print[r + pos].insert(2, "  " * (val_width - len('[' + "".join(row) + ']')))
        for r, row in enumerate(to_print[pos + len(val):]):
            if len("".join(row)) > 0 and "".join(row)[0] == '-':
                while row[0] == '':
                    row.pop(0)
                row[0] = '├' + row[0][1:]
                last = r + pos + len(val)
            else:
                if row:
                    row[0] = '│' + row[0][1:]
                else:
                    row.append('│')
            row.insert(0, " " * (val_width + 1))
        indx = 0
        while to_print[last][indx].strip() != '' or to_print[last][indx + 1].startswith('['):
            indx += 1
        indx += 1
        to_print[last][indx] = {'┬': '─', '├': '└', '┌': '─', '┼': '┴', '┤': '┘', '': ''}[to_print[last][indx]]
        for i in range(last + 1, len(to_print)):
            indx = 0
            while to_print[i][indx].strip() != '' or to_print[i][indx + 1].startswith('['):
                indx += 1
            indx += 1
            to_print[i][indx] = ' ' + to_print[i][indx][1:]
        return to_print

    def color_txt(self, x):
        spaces = " " * (len(x) - len(x.lstrip()))
        x = x.lstrip()
        is_label = x.startswith('|')
        if is_label:
            x = f' {x[1:-1]} '
            x = (self.label_color + x + '0') if self.label_color else x
            return spaces + x
        txt = x if self.border else f' {x[1:-1]} '
        txt = (self.color + txt + '\x1b[0m') if self.color else txt
        return spaces + txt

    def format_box(self, spacing, val):
        for r, row in enumerate(val):
            val[r] = [spacing, f'│{row[0]}│']
        if self.border:
            top = [[spacing, '┌' + '─' * (len(val[0][1]) - 2) + '┐']]
            bottom = [[spacing, '└' + '─' * (len(val[0][1]) - 2) + '┘']]
            return top + val + bottom
        return val


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

        self.probs = self.log_probs = np.zeros((len(self.predecessors), 2, 2), dtype=float)
        self.calc_log_params(root)

        # Verify that all probabilities add up to 1
        assert (np.round(logsumexp(self.probs, axis=2), decimals=2).all() == 1)

    def calc_log_params(self, root: int):
        for node, parent in enumerate(self.predecessors):
            if parent == -1:  # Root node
                self.probs[node, 0, 0] = self.probs[node, 1, 0] = 1 - self.marginals[root]
                self.probs[node, 0, 1] = self.probs[node, 1, 1] = self.marginals[root]
                continue
            self.probs[node, 0, 0] = self.joint_probs[0, node, parent] / (1 - self.marginals[parent])
            self.probs[node, 0, 1] = self.joint_probs[1 if parent < node else 2, node, parent] / (
                    1 - self.marginals[parent])
            self.probs[node, 1, 0] = self.joint_probs[2 if parent < node else 1, node, parent] / self.marginals[parent]
            self.probs[node, 1, 1] = self.joint_probs[3, node, parent] / self.marginals[parent]
        self.log_probs = np.log(self.probs)

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
        jpmf = None
        rvs = None
        n_queries = queries.shape[0]
        lp = np.zeros(n_queries, dtype=float)

        if exhaustive:
            rvs = np.array(list(itertools.product([0, 1], repeat=queries.shape[1])))
            jpmf = self.calc_joint_probs(rvs)

        print(f'Evaluating {n_queries} queries with exhaustive={exhaustive}...')
        for i, query in enumerate(tqdm(queries)):
            start_time = time.time()
            if exhaustive:
                zeros = np.where(query == 0)[0]
                ones = np.where(query == 1)[0]

                filtered = filter(lambda comb: np.all(comb[zeros] == 0) and np.all(comb[ones] == 1), rvs)
                probabilities = jpmf[list(map(lambda comb: bin2dec(comb), filtered))]
                lp[i] = np.log(np.sum(probabilities))
            else:
                # Message passing
                message = np.log(self.get_message(self.root_node, query))
                if verbose:
                    print(f'Log prob of query {query} is {message}\n')
                lp[i] = message

            if verbose:
                print(f'Query {i} took {time.time() - start_time} seconds.')
        return lp

    def calc_joint_probs(self, bin_combinations):
        jpmf = []  # Joint Probability Mass Function
        for comb in bin_combinations:
            probability = 1
            for index, rv in enumerate(comb):
                index_parent = self.predecessors[index]
                val_parent = comb[index_parent]
                probability *= self.probs[index][val_parent][rv]
            jpmf.append(probability)
        jpmf = np.array(jpmf)
        return jpmf

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


def bin2dec(binary: np.ndarray) -> int:
    return sum(val * (2**idx) for idx, val in enumerate(reversed(binary)))


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
    parser.add_argument('-q', '--n_queries', type=int, default=10,
                        help='Number of queries to evaluate if no dataset is provided')
    parser.add_argument('-f', '--full_data', action='store_true', default=False,
                        help='Whether to use the full query dataset')
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
    print('Planting the Chow-Liu trees...')
    clt_train = BinaryCLT(train_data, root=args.root, alpha=args.alpha)
    clt_test = BinaryCLT(test_data, root=args.root, alpha=args.alpha)

    # Use a partial query dataset for faster evaluation
    if not args.full_data:
        query_data = query_data[:args.n_queries]

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
    print(f'Correlation: {np.mean(np.array([np.corrcoef(log_params_train[:, i, j], log_params_test[:, i, j]) for i in range(2) for j in range(2)])[:, 1, 0]).round(3)}\n')

    log_probs_train = clt_train.log_prob(query_data)
    log_probs_test = clt_test.log_prob(query_data)
    print(f'Resulting train log probabilities of evaluation queries: {log_probs_train.round(3).tolist()}')
    print(f'Resulting test log probabilities of evaluation queries: {log_probs_test.round(3).tolist()}')
    print(f'Correlation: {np.corrcoef(log_probs_train, log_probs_test)[0, 1].round(3)}\n')

    log_probs_ex_train = clt_train.log_prob(query_data, exhaustive=True)
    log_probs_ex_test = clt_test.log_prob(query_data, exhaustive=True)
    print(f'Resulting exhaustive train log probabilities of evaluation queries: {log_probs_ex_train.round(3).tolist()}')
    print(f'Resulting exhaustive test log probabilities of evaluation queries: {log_probs_ex_test.round(3).tolist()}')
    print(f'Correlation: {np.corrcoef(log_probs_train, log_probs_test)[0, 1].round(3)}\n')

    assert(log_probs_train.all() == log_probs_ex_train.all())
    assert(log_probs_test.all() == log_probs_ex_test.all())

    avg_samples_train = np.mean(clt_train.sample(args.n_samples), axis=0)
    avg_samples_test = np.mean(clt_test.sample(args.n_samples), axis=0)
    print(f'Train sample averages: {avg_samples_train.round(3)}')
    print(f'Test sample averages: {avg_samples_test.round(3)}')
    print(f'Correlation : {np.corrcoef(avg_samples_train, avg_samples_test)[0, 1].round(3)}\n')
