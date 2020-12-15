from utils import *

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.
    input_size = len(data['question_id'])

    for i in range(input_size):
        question_id = data['question_id'][i]
        user_id = data['user_id'][i]
        prediction = data['is_correct'][i]
        log_lklihood += prediction * (theta[user_id] - beta[question_id]) - \
                        np.log(1 + np.exp(theta[user_id] - beta[question_id]))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def derivative_theta(data, theta, beta, nonz):
    """Compute derivative of MLE respect to theta

    :param data:
    :param theta:
    :param beta:
    :return:
    """
    theta_derivative = np.zeros((len(theta), 1))
    student_num = data.shape[0]
    question_num = data.shape[1]

    if nonz is None:
        dense = data.toarray()
        nonz = np.nan_to_num(np.where(dense==0., 1., dense)).nonzero()

    for i, j in zip(*nonz):
        theta_derivative[i] += data[i, j] - 1/(1+1/np.exp(theta[i]-beta[j]))
    return theta_derivative


def derivative_beta(data, theta, beta, nonz):
    """Compute derivative of MLE respect to beta

        :param data:
        :param theta:
        :param beta:
        :return:
        """
    beta_derivative = np.zeros((len(beta), 1))
    student_num = data.shape[0]
    question_num = data.shape[1]

    if nonz is None:
        dense = data.toarray()
        nonz = np.nan_to_num(np.where(dense==0., 1., dense)).nonzero()

    beta_derivative = np.zeros((len(beta), 1))
    for i, j in zip(*nonz):
        beta_derivative[j] += -data[i, j] + 1/(1+1/np.exp(theta[i]-beta[j]))
    return beta_derivative


def update_theta_beta(data, lr, theta, beta, nonz=None):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    theta_derivative = derivative_theta(data, theta, beta, nonz)
    theta += lr * theta_derivative
    beta_derivative = derivative_beta(data, theta, beta, nonz)
    beta += lr * beta_derivative
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(train_data, train_sparse, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param train_sparse: a sparse training data
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    theta_len = train_sparse.shape[0]
    beta_len = train_sparse.shape[1]
    theta = np.zeros((theta_len, 1))
    beta = np.zeros((beta_len, 1))

    val_acc_lst = []
    train_loglikes = []
    valid_loglikes = []

    dense = train_sparse.toarray()
    nonz = np.nan_to_num(np.where(dense==0., 1., dense)).nonzero()

    for i in range(iterations):
        train_neg_lld = neg_log_likelihood(train_data, theta=theta, beta=beta)
        train_loglikes.append(train_neg_lld)

        valid_loglike = neg_log_likelihood(val_data, theta, beta)
        valid_loglikes.append(valid_loglike)

        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        # print("NLLK: {} \t Score: {}".format(train_neg_lld, score))
        theta, beta = update_theta_beta(train_sparse, lr, theta, beta, nonz)

    # TODO: You may change the return values to achieve what you want.

    return theta, beta, train_loglikes, valid_loglikes


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    #lr = 0.016
    #theta, beta, train_log, valid_log = irt(train_data, sparse_matrix,
    #                                        val_data, lr, 10)
    ## plot log-likelihood as training process
    #num_iter = [i for i in range(10)]
    #plt.plot(num_iter, valid_log, linestyle='--',
    #         color='b', label="validation")
    #plt.plot(num_iter, train_log, linestyle='--',
    #         color='r', label="train")
    #plt.title("Change for log likelihoods")
    #plt.xlabel('Iterations', fontsize=14)
    #plt.ylabel('loglike', fontsize=14)
    #plt.legend(loc=1)
    ##plt.show()
    #np.save("../irt_output/theta", theta)
    #np.save("../irt_output/beta", beta)
    theta = np.load("../irt_output/theta.npy")
    beta = np.load("../irt_output/beta.npy")

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (c)                                                #
    #####################################################################
    # final train and test accuracies
    valid_accuracy = evaluate(val_data, theta, beta)
    test_accuracy = evaluate(test_data, theta, beta)
    print("Final validation accuracy: {}\n".format(valid_accuracy))
    print("Final test accuracy: {}\n".format(test_accuracy))

    colors = ["blue", "red", "orange", "green", "purple"]
    for i, j in enumerate([4, 8, 16, 32, 64]):
        probs = []
        for th in theta:
            probs.append(1/(1+1/np.exp(th-beta[j])))
        plt.plot(theta, probs, color=colors[i], label=f"question ${j}$")
    plt.xlabel(r"$\theta_i=i$-th student's ability")
    plt.ylabel(r"$p(c_{ij}=1|\theta_i,\beta_j)$")
    plt.legend()
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()


