from utils import *
from scipy.linalg import sqrtm

import matplotlib.pyplot as plt
import numpy as np


def svd_reconstruct(matrix, k):
    """ Given the matrix, perform singular value decomposition
    to reconstruct the matrix.

    :param matrix: 2D sparse matrix
    :param k: int
    :return: 2D matrix
    """
    # First, you need to fill in the missing values (NaN) to perform SVD.
    # Fill in the missing values using the average on the current item.
    # Note that there are many options to do fill in the
    # missing values (e.g. fill with 0).
    new_matrix = matrix.copy()
    mask = np.isnan(new_matrix)
    masked_matrix = np.ma.masked_array(new_matrix, mask)
    item_means = np.mean(masked_matrix, axis=0)
    new_matrix = masked_matrix.filled(item_means)

    # Next, compute the average and subtract it.
    item_means = np.mean(new_matrix, axis=0)
    mu = np.tile(item_means, (new_matrix.shape[0], 1))
    new_matrix = new_matrix - mu

    # Perform SVD.
    Q, s, Ut = np.linalg.svd(new_matrix, full_matrices=False)
    s = np.diag(s)

    # Choose top k eigenvalues.
    s = s[0:k, 0:k]
    Q = Q[:, 0:k]
    Ut = Ut[0:k, :]
    s_root = sqrtm(s)

    # Reconstruct the matrix.
    reconst_matrix = np.dot(np.dot(Q, s_root), np.dot(s_root, Ut))
    reconst_matrix = reconst_matrix + mu
    return np.array(reconst_matrix)


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
    u = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["user_id"])), k))
    z = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["question_id"])), k))
    
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # itterations = [None] * int(num_iteration//250)
    # results = [None] * int(num_iteration//250)
    for i in range(num_iteration):
        print(i)
        u,z = update_u_z(train_data, lr, u,z)
        # print(results[i])
        # if i%250 == 0:
        #     itterations[i//250] = i
        #     results[i//250] = squared_error_loss(train_data,u,z)
    # plt.plot(itterations,results, label = 'square_err')
    # plt.legend()
    # plt.savefig('../figs/matrix_als_k'+str(k))
    # plt.cla()
    print(squared_error_loss(train_data,u,z))
    mat = np.dot(u, z.T)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return mat


def main():
    train_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    
    #####################################################################
    # TODO:                                                             #
    # (SVD) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################
    run_svd = 0
    run_als = 1
    if run_svd == 1:
        k_values = [4,6,8,10,12]
        k_performance = [None] * 5
        k_star = 0
        # print(len(val_data))
        # print(len(val_data['user_id']))
        # print(len(val_data['question_id']))
        # print(len(val_data['is_correct']))

        for i in range(len(k_values)):
            factorized_matrix = svd_reconstruct(train_matrix,k_values[i])
            cur_score = 0
            for n in range(len(val_data['user_id'])):

                if (int(factorized_matrix.item(val_data['user_id'][n], val_data['question_id'][n]) > 0.5) == val_data['is_correct'][n]):
                    cur_score += 1
            k_performance[i] = cur_score/len(val_data['user_id'])
            if k_performance[k_star] < k_performance[i]:
                k_star = i
        print("We find that the best value for k, k* is", k_values[k_star] )
        plt.plot(k_values,k_performance, label = 'correct_accuracy')
        plt.legend()
        plt.savefig('../figs/matrix_fac_svd')
        plt.cla()

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # (ALS) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################
    if run_als == 1:
        k_values = [5,10,15,20,25]
        k_values = [35]
        k_performance = [None] * 5
        k_star = 0
        #for i in range(len(k_values)):
        for i in range(len(k_values)):
            factorized_matrix = als(train_data,k_values[i],0.01, 500000)
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
