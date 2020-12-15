from utils import *
from scipy.linalg import sqrtm

import matplotlib.pyplot as plt
import numpy as np


def squared_error_loss(data, u, z):
    """ Return the squared-error-loss given the data.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param u: 2D matrix
    :param z: 2D matrix
    :return: float
    """
    loss = 0
    for i, q in enumerate(data["question_id"]):
        loss += (data["is_correct"][i]
                 - np.sum(u[data["user_id"][i]] * z[q])) ** 2.
    return 0.5 * loss


def update_u_z(train_data, lr, u, z):
    """ Return the updated U and Z after applying
    stochastic gradient descent for matrix completion.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param u: 2D matrix
    :param z: 2D matrix
    :return: (u, z)
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # Randomly select a pair (user_id, question_id).
    i = \
        np.random.choice(len(train_data["question_id"]), 1)[0]

    c = train_data["is_correct"][i]
    n = train_data["user_id"][i]
    q = train_data["question_id"][i]

    cur_u = u[ n]
    cur_z = z[ q]
    # print('------------------------')
    # print(c)
    # print(np.dot( cur_u.T ,cur_z))
    # print(c - np.dot( cur_u.T ,cur_z))
    # print(cur_z.shape)
    # print(cur_u.shape)
    u[n] = cur_u + lr * (c - np.dot( cur_u.T ,cur_z)) * cur_z
    z[q] = cur_z + lr * (c - np.dot( cur_u.T ,cur_z)) * cur_u


    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return u, z


def als(train_data, k, lr, num_iteration):
    """ Performs ALS algorithm. Return reconstructed matrix.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :param lr: float
    :param num_iteration: int
    :return: 2D reconstructed Matrix.
    """

    # Initialize u and z
    n_users = len(set(train_data["user_id"]))
    n_quest = len(set(train_data["question_id"]))
    u = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(n_users, k))
    z = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(n_quest, k))
    
    matrix = np.zeros((n_users, n_quest))
    for i in train_data["user_id"]:
        matrix[train_data["user_id"][i], train_data["question_id"][i]] = train_data["is_correct"][i]

    matrix_size = np.prod(matrix.shape)
    interaction = np.flatnonzero(matrix).shape[0]
    sparsity = 100 * (interaction / matrix_size)

    print(matrix)

    print('dimension: ', matrix.shape)
    print('sparsity: {:.1f}%'.format(sparsity))

    


def main():
    train_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    run_als = 1

    #####################################################################
    # TODO:                                                             #
    # (ALS) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################
    if run_als == 1:
        k_values = [5,10,15,20,25]
        #k_values = [35]
        k_performance = [None] * 5
        k_star = 0
        #for i in range(len(k_values)):
        for i in range(len(k_values)):
            factorized_matrix = als(train_data,k_values[i],0.01, 500000)
            return
            cur_score = 0
            for n in range(len(val_data['user_id'])):
                if (int(factorized_matrix.item(val_data['user_id'][n], val_data['question_id'][n]) > 0.5) == val_data['is_correct'][n]):
                    cur_score += 1
            k_performance[i] = cur_score/len(val_data['user_id'])
            print(k_performance[i])
            # if k_performance[k_star] < k_performance[i]:
            #     k_star = i

            # k_performance[i] = cur_score/len(val_data['user_id'])
            # if k_performance[k_star] < k_performance[i]:
            #     k_star = i
            

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
