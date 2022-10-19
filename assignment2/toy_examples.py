import numpy as np


def mi(joint, marginal_1, marginal_2):
    return joint * np.log(joint / (marginal_1 * marginal_2))


joint_x1_x2_0 = 0.3
joint_x1_x2_1 = 0.4

marginal_x1_0 = 0.45
marginal_x2_0 = 0.45

marginal_x1_1 = 0.55
marginal_x2_1 = 0.55

print(mi(0.3, marginal_x1_0, marginal_x2_0) +
      mi(0.15, marginal_x1_0, marginal_x2_1) +
      mi(0.15, marginal_x1_1, marginal_x2_0) +
      mi(0.4, marginal_x1_1, marginal_x2_1))

marginal_x2_0 = 0.45
marginal_x3_0 = 0.45

marginal_x2_1 = 0.55
marginal_x3_1 = 0.55

print(mi(0.35, marginal_x1_0, marginal_x2_0) +
      mi(0.1, marginal_x1_0, marginal_x2_1) +
      mi(0.1, marginal_x1_1, marginal_x2_0) +
      mi(0.45, marginal_x1_1, marginal_x2_1))

log_probs = np.array(([[-1.204, -0.357], [-1.204, -0.357]], [[-1.609, -0.223], [-0.511, -0.916]],
                     [[-0.916, -0.511], [-2.303, -0.105]], [[-0.223, -1.609], [-0.693, -0.693]],
                     [[-0.105, -2.303], [-0.916, -0.511]]))

print(np.round(np.exp(log_probs), decimals=3))

clt.log_probs_train = np.array([[[0.3, 0.7], [0.3, 0.7]],
                                [[0.2, 0.8], [0.6, 0.4]],
                                [[0.4, 0.6], [0.1, 0.9]],
                                [[0.8, 0.2], [0.5, 0.5]],
                                [[0.9, 0.1], [0.4, 0.6]]])

clt.predecessors_train = np.array([-1, 0, 4, 4, 0])

test_queries = np.array([[1., 0., 1., 1., 0.]])
log_probs = clt.log_prob(test_queries)
print('Query', test_queries)
print('Log Probs', np.log(log_probs))
print()

test_queries = np.array([[np.nan, 0., np.nan, np.nan, 1.]])
log_probs = clt.log_prob(test_queries)
print('Query', test_queries)
print('Log Probs', np.log(log_probs))
print()

test_queries = np.array([[np.nan, 0., 1., 1., np.nan]])
log_probs = clt.log_prob(test_queries)
print('Query', test_queries)
print('Log Probs', np.log(log_probs))
print()

# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX #
clt_train.probs = clt_test.probs = np.array([[[0.3, 0.7], [0.3, 0.7]],
                                             [[0.2, 0.8], [0.6, 0.4]],
                                             [[0.4, 0.6], [0.1, 0.9]],
                                             [[0.8, 0.2], [0.5, 0.5]],
                                             [[0.9, 0.1], [0.4, 0.6]]])

clt_train.log_probs = clt_test.log_probs = np.log(clt_train.probs)

clt_train.predecessors_train = clt_test.predecessors_train = np.array([-1, 0, 4, 4, 0])
clt_train.root_node = clt_test.root_node = create_tree(clt_train.predecessors_train)

probs = [0.01728, 0.0003, 0.00432, 0.0003, 0.02592, 0.0027, 0.00648, 0.0027, 0.06912, 0.0012, 0.01728, 0.0012,
         0.10368,
         0.0108, 0.02592, 0.0108, 0.05376, 0.0126, 0.01344, 0.0126, 0.08064, 0.1134, 0.02016, 0.1134, 0.03584,
         0.0084,
         0.00896, 0.0084, 0.05376, 0.0756, 0.01344, 0.0756]
clt_train.marginals[0] = np.sum(probs[15:31])
clt_train.marginals[1] = np.sum(probs[7:15]) + np.sum(probs[23:31])
clt_train.marginals[2] = np.sum(probs[3:7]) + np.sum(probs[11:15]) + np.sum(probs[19:23]) + np.sum(probs[27:31])
clt_train.marginals[3] = 1 - (np.sum(probs[::4]) + np.sum(probs[1::4]))
clt_train.marginals[4] = 1 - np.sum(probs[::2])

query_data = np.array([[1., 0., 1., 1., 0.],
                         [np.nan, 0., np.nan, np.nan, 1.],
                         [np.nan, 0., 1., 1., np.nan]])
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX #


