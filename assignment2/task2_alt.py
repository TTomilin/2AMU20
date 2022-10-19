# DEPENDENCIES
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.csgraph import breadth_first_order
from scipy.special import logsumexp
import numpy as np
import csv
import time
from itertools import product

# Define the data to work with:
DATASET = "nltcs"
PATH = f"data/{DATASET}/"

# GRAPH VIS
# TODO delete graph
import networkx as nx
import matplotlib.pyplot as plt

# TODO delete graph
class GraphVisualization:
    def __init__(self):
        # visual is a list which stores all
        # the set of edges that constitutes a
        # graph
        self.visual = []

    # addEdge function inputs the vertices of an
    # edge and appends it to the visual list
    def addEdge(self, a, b):
        temp = (a, b)
        self.visual.append(temp)

    # In visualize function G is an object of
    # class Graph given by networkx G.add_edges_from(visual)
    # creates a graph with a given list
    # nx.draw_networkx(G) - plots the graph
    # plt.show() - displays the graph
    def visualize(self):
        G = nx.Graph()
        G.add_edges_from(self.visual)
        nx.draw_networkx(G)
        plt.show()

# TODO delete graph
def visualize_graph(adj_mat):
    G = GraphVisualization()
    n_rows, n_cols = np.shape(adj_mat)
    for i in range(n_rows):
        for j in range(n_cols):
            if adj_mat[i][j] != 0:
                G.addEdge(i, j)
    G.visualize()


# HELPER FUNCTIONS
def get_MI(data):
    """
    Get MI from data
    :param data: data matrix of size nr_samples x nr_variables.
    :return: MI-matrix of size nr_variables x nr_variables
    """
    nr_samples, nr_variables = np.shape(data)
    MI_matrix = np.zeros((nr_variables, nr_variables))
    for ind_X in range(nr_variables):
        for ind_Y in range(ind_X):
            joint_prob = get_joint(data, ind_X, ind_Y)  # 2x2 joint probability matrix
            p_X, p_Y = get_marginals(joint_prob)
            MI = 0

            for x, y in product(range(2), repeat=2):
                MI += joint_prob[y, x] * np.log((joint_prob[y, x]) / (p_X[x] * p_Y[y]))
            MI_matrix[ind_X, ind_Y] = MI
            MI_matrix[ind_Y, ind_X] = MI
    return MI_matrix


def get_prob_root(data, ind_x: int, alpha):
    """
    Gets the probability for random binary variables X
    and repeats the row, so it becomes a 2x2 matrix
    :param data: data matrix of size nr_samples x nr_variables.
    :param ind_x: index of parameter
    :param alpha: Laplace's correction parameter
    :return: 2x2 matrix
    [[p(X=0), p(X=1)],
     [p(X=0), p(X=1)]]
    """
    nr_variables = data.shape[0]
    joint_mat = np.zeros((2, 2)) + 2 * alpha
    for sample in data:
        X = int(sample[ind_x])
        joint_mat[0][X] += 1
    joint_mat = joint_mat / (4 * alpha + nr_variables)
    joint_mat[1] = joint_mat[0]
    return joint_mat


def get_joint(data, ind_X: int, ind_Y: int, alpha=0):
    """
    Gets the joint probability for random binary variables var, par
    :param data: data matrix of size nr_samples x nr_variables.
    :param ind_X: index of parameter
    :param ind_Y: index of parameter
    :param alpha: Laplace's correction parameter
    :return: 2x2 matrix
    [[p(X=0, Y=0), p(X=1 Y=0)],
     [p(X=0, Y=1), p(X=1, Y=1)]]
    """
    nr_variables = data.shape[0]
    joint_mat = np.zeros((2, 2)) + alpha

    for sample in data:
        X = int(sample[ind_X])
        Y = int(sample[ind_Y])
        joint_mat[Y][X] += 1
    joint_mat = joint_mat / (4 * alpha + nr_variables)
    return joint_mat


def get_marginals(joint_mat):
    """
    Gets the marginals of binary random variables var and par
    :param joint_mat: Joint probability matrix of var and par
    :return: tuple of two marginal vectors
    ([p(var = 0), p(var = 1)], [p([p(par = 0), p(par = 1)])
    """
    marginal_X = np.sum(joint_mat, axis=0)
    marginal_Y = np.sum(joint_mat, axis=1)
    return marginal_X, marginal_Y


def get_rooted_tree(adj_mat, root):
    """
    Make a directional adjacency matrix NxN
    :param adj_mat: adjecency matrix returned from the max spanning tree NxN
    :param root: index of the root node [0,N>
    :return: directed adjacency matrix where row = from and colums = to, NxN
    """
    nr_variables = adj_mat.shape[0]
    dir_adj_mat = np.minimum(adj_mat, adj_mat.transpose())
    next_nodes = [root]
    while next_nodes:
        for i in range(nr_variables):
            if dir_adj_mat[next_nodes[0]][i] != 0:
                dir_adj_mat[i][next_nodes[0]] = 0
                next_nodes.append(i)
        next_nodes.pop(0)
    return dir_adj_mat


# CLASS DEFINITION
class BinaryCLT:
    """A tree-shaped Bayesian Network, learned through the Chow-Liu Algorithm,
    allowing tractable inference.
    """

    def __init__(self, data, root: int = None, alpha: float = 0.01):
        """Creates and learns a CLT, given a binary dataset.

        :param data: A numpy matrix with rows corresponding to samples, and columns
                corresponding to random variables. Matrix entries are binary (0 or 1)
        :param root: Integer corresponding to the RV the tree must be rooted at. If None,
                The root node will be selected at random
        :param alpha: Laplace's correction
        """
        self.data = data
        self.nr_variables = data.shape[1]
        self.alpha = alpha
        self.full_joint = None
        self.full_log_joint = None
        self.log_params = None
        self.params = None
        self.tree = None
        self.top_order = None

        # Estimate MI
        MI = get_MI(data)

        # Maximum Spanning Tree
        max_spanning_tree = minimum_spanning_tree(-MI).toarray()

        # Directed Tree
        if root is None:
            self.root = np.random.randint(self.nr_variables)
        else:
            self.root = root
        dir_adj_mat = get_rooted_tree(max_spanning_tree, self.root)
        self.CLT = dir_adj_mat

    def get_tree(self):
        """
        Returns the list of predecessors of the learned CLT.
        :return: List of predecessors of length nr_variables (aka nr of nodes). If the node is the root node, its
                predecessor is -1
        """
        if self.tree is None:
            top_order, tree = breadth_first_order(self.CLT, self.root)
            tree[self.root] = -1
            self.tree = tree
            self.top_order = top_order
        return self.tree

    def get_log_params(self):
        """
        Returns the log of the conditional probability tables (CPTs) of the CLT, estimated by maximum likelihood and
        using Laplace's correction. CPT can be indexed by [i, j, k] where:
        i is the index of the variable / node
        j is the value of the parent of node i
        k is the value of node i
        :return: A np array of size nr_variables x 2 x 2
        [[p(A = 0 | D = 0), p(A = 1 | D = 0)],
         [p(A = 0 | D = 1), p(A = 1 | D = 1)]]
        """
        if self.log_params is None:
            if self.tree is None:
                self.get_tree()
            parent_list = self.tree
            CPT_arr_log = np.zeros((self.nr_variables, 2, 2))
            root_prob_log = np.log(get_prob_root(self.data, self.root, self.alpha))
            CPT_arr_log[self.root, :, :] = root_prob_log

            for cur_node in self.top_order[1:]:
                parent_node = parent_list[cur_node]
                joint_prob = get_joint(self.data, cur_node, parent_node)
                _, marginal_parent = get_marginals(joint_prob)
                joint_prob_log = np.log(joint_prob)
                marginal_parent_log = np.log(marginal_parent)
                marginal_arr_log = np.array(
                    [[marginal_parent_log[0], marginal_parent_log[0]], [marginal_parent_log[1], marginal_parent_log[1]]]
                )
                conditional_arr_log = joint_prob_log - marginal_arr_log
                CPT_arr_log[cur_node, :, :] = conditional_arr_log

            self.log_params = CPT_arr_log
            self.params = np.exp(CPT_arr_log)
        return self.log_params

    def log_prob(self, queries, exhaustive: bool = False):
        """
        Computes the log probability for both fully observed samples and marginal queries
        :param queries: array of size nr_samples x nr_variables. If a variable is observed in a sample, its entry takes
            either 0 or 1 as value. If not observed, it takes np.nan as value.
        :param exhaustive: If True, perform exhaustive inference by explicitly summing out the unobserved variables
            Z from the joint. If False, perform efficient inference by applying [INSERT ALGORITHM HERE]
        :return:
        """
        if self.log_params is None:
            self.get_log_params()
        log_probs = self.log_params
        if exhaustive:
            lp = self.exh_get_prob(self.get_full_log_joint(log_probs), queries)
        else:
            lp = self.message_passing(log_probs, queries)
        return lp

    def message_passing(self, log_params, queries):
        """
        Returns the log probabilities for each query in a list of queries. These log probabilities are computed by
        applying the message passing algorithm.
        :param log_params: conditional probabilities
        :param queries: an array of size nr_queries x nr_variables. Observed variables have a value of 0 or 1. Unobserved
            variables have the value np.nan.
        :return: An array of size nr_queries x 1, containing the (marginal) probability of each query.
        """

        # TODO document this function
        def get_msg(var, val_par):
            '''

            :param var:
            :param val_par:
            :return:
            '''
            var = int(var)
            if self.in_query(query, var):
                msg = get_msg_in_query(var, val_par)
            else:
                msg = get_msg_not_query(var, val_par)
            return msg

        # TODO document this function
        def get_msg_in_query(var, val_par):
            '''

            :param var:
            :param val_par:
            :return:
            '''
            var = int(var)
            msg = 0
            msg += log_params[var][val_par][query[var]]
            for child in np.where(self.tree == var)[0]:
                msg += get_msg(int(child), query[var])
            return msg

        # TODO document this function
        def get_msg_not_query(var, val_par):
            '''

            :param var:
            :param val_par:
            :return:
            '''
            msg = np.zeros((2,))
            msg[0] = log_params[var][val_par][0]
            msg[1] = log_params[var][val_par][1]
            for child in np.where(self.tree == var)[0]:
                msg[0] += get_msg(int(child), 0)
            for child in np.where(self.tree == var)[0]:
                msg[1] += get_msg(int(child), 1)
            return logsumexp(msg)

        lp = np.zeros(len(queries))
        for i, query in enumerate(queries):
            lp[i] = get_msg(self.root, query[self.root])
        lp = np.reshape(lp, (len(lp), 1))
        return lp

    def in_query(self, query, node):
        return True if query[node] != -1 else False

    def get_full_log_joint(self, params):
        """
        Returns the joint distribution 2^N x N
        :params log probabilities
        :return: the joint distribution 2^N x N
        """
        if self.full_log_joint is None:
            full_log_joint = np.zeros(tuple(2 for i in range(self.nr_variables)))
            parent_list = self.get_tree()
            for ind in product(range(2), repeat=self.nr_variables):
                combination = tuple(ind)
                probability = 0
                for ind_rv, val_rv in enumerate(combination):
                    ind_parent = parent_list[ind_rv]
                    val_parent = combination[ind_parent]
                    probability += params[ind_rv][val_parent][val_rv]
                full_log_joint[combination] = probability
            self.full_log_joint = full_log_joint
            self.full_joint = np.exp(self.full_log_joint)
        return self.full_log_joint

    def exh_get_prob(self, full_joint, queries):
        """
        Returns the (marginal) probabilities of a list of queries. If a variable is unobserved, The marginal will be
        computed over all possible values of the unobserved variable.
        :param full_joint: The full joint probability table for all possible values of each variable
        :param queries: an array of size nr_queries x nr_variables. Observed variables have a value of 0 or 1. Unobserved
            variables have the value np.nan.
        :return: An array of size nr_queries x 1, containing the (marginal) probability of each query.
        """
        nr_queries = queries.shape[0]
        lp = np.zeros((nr_queries, 1))
        for i, q in enumerate(queries):
            if not any(q < 0):
                query_prob = full_joint[tuple(q)]
            else:
                if len(np.argwhere(q < 0)) > 1:
                    nan_idx = list(np.squeeze(np.argwhere(q < 0)))
                else:
                    nan_idx = [np.squeeze(np.argwhere(q < 0)).tolist()]
                idx_list = list(q.copy())
                for idx_nan in nan_idx:
                    idx_list[idx_nan] = slice(None)
                query_prob = logsumexp(full_joint[tuple(idx_list)])
            lp[i] = query_prob
        return lp

    def sample(self, n_samples: int):
        """
        Produces i.i.d. samples from the CLT distribution using ancestral sampling.
        :param n_samples: The number of samples that should be generated
        :return: An array of size n_samples x nr_variables. Each row corresponds to a
        randomly generated sample containing values for each RV in the CLT (value is either 0 or 1)
        """
        # get topological order:
        if self.tree is None:
            self.get_tree()

        if self.log_params is None:
            self.get_log_params()

        samples = np.random.random_sample((n_samples, self.nr_variables))

        for var in self.top_order: #loop over variables in topological order
            prob_table = np.exp(self.log_params[var][:][1])
            parent = self.tree[var]
            for i_sample in range(n_samples): #generate for each sample the value for that variable
                if samples[i_sample, var] <= prob_table[int(samples[i_sample, parent])]:
                    samples[i_sample, var] = 1
                else:
                    samples[i_sample, var] = 0
        return samples


# GET DATA
# training data
with open(PATH + f"{DATASET}.train.data") as file:
    reader = csv.reader(file, delimiter=",")
    train_data = np.array(list(reader)).astype(np.float64)

# validation data
with open(PATH + f"{DATASET}.valid.data") as file:
    reader = csv.reader(file, delimiter=",")
    val_data = np.array(list(reader)).astype(np.float64)

# testing data
with open(PATH + f"{DATASET}.test.data") as file:
    reader = csv.reader(file, delimiter=",")
    test_data = np.array(list(reader)).astype(np.float64)

# query dummy
# with open(PATH + f"{DATASET}.query_test2") as file:
#     reader = csv.reader(file, delimiter=",")
#     query_dummy = np.array(list(reader)).astype(np.float64)
#     query_dummy = np.nan_to_num(query_dummy, copy=False, nan=-1).astype(np.int64)

# query 2e
with open(PATH + f"{DATASET}.marginals.data") as file:
    reader = csv.reader(file, delimiter=",")
    query_data = np.array(list(reader)).astype(np.float64)
    query_data = np.nan_to_num(query_data, copy=False, nan=-1).astype(np.int64)


CLT = BinaryCLT(train_data, root=0, alpha=0.01)
tree = CLT.get_tree()
visualize_graph(CLT.CLT)
log_params = CLT.get_log_params()
full_log_joint = CLT.get_full_log_joint(log_params)
total_prob = np.testing.assert_allclose(np.sum(np.exp(full_log_joint)), 1)
startt_exh = time.time()
lp = CLT.log_prob(query_data, exhaustive=True)
elapsed_time_exh = time.time() - startt_exh

startt_mp = time.time()
lp_mp = CLT.log_prob(query_data, exhaustive=False)
elapsed_time_mp = time.time() - startt_mp


samples = CLT.sample(n_samples=1000)


print(f"The log probabilities (lp) by exhaustive and message passing inference took {elapsed_time_exh} and {elapsed_time_mp} respectively.")
print(f"lp by exhaustive inference: {np.exp(lp)}")
print(f"lp by message passing inference: {np.exp(lp)}")
print(f"Generated samples: {samples}")
print("finished")

# CLT = BinaryCLT(train_data, root=0, alpha=0.01)
# CLT.CLT = np.zeros((5, 5))
# CLT.CLT[0][1] = -10
# CLT.CLT[0][4] = -10
# CLT.CLT[4][3] = -10
# CLT.CLT[4][2] = -10
# CLT.nr_variables = 5
# tree = CLT.get_tree()
# visualize_graph(CLT.CLT)
# log_params = CLT.get_log_params()
# CLT.log_params = np.array([[[-1.204, -0.357], [-1.204, -0.357]],
# [[-1.609, -0.223], [-0.511, -0.916]],
# [[-0.916, -0.511], [-2.303, -0.105]],
# [[-0.223, -1.609], [-0.693, -0.693]],
# [[-0.105, -2.303], [-0.916, -0.511]]])
# query_data = np.array([[1, 0, 1, 1, 0], [-1, 0, -1, -1, 1], [-1, 0, 1, 1, -1]])
# log_params = CLT.log_params
# full_log_joint = CLT.get_full_log_joint(log_params)
# # total_prob = np.testing.assert_allclose(np.sum(np.exp(full_log_joint)), 1)
# total_prob = np.sum(np.exp(full_log_joint))
# startt_exh = time.time()
# lp = CLT.log_prob(query_data, exhaustive=True)
# elapsed_time_exh = time.time() - startt_exh
#
# startt_mp = time.time()
# lp_mp = CLT.log_prob(query_data, exhaustive=False)
# elapsed_time_mp = time.time() - startt_mp
#
# test_lp = np.array([[-3.904], [-1.354], [-1.946]])
# samples = CLT.sample(n_samples=1000)
