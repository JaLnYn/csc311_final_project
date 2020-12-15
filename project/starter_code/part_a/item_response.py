from utils import *
import seaborn as sns
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


def derivative_theta(data, theta, beta):
    """Compute derivative of MLE respect to theta

    :param data:
    :param theta:
    :param beta:
    :return:
    """
    theta_derivative = np.zeros((len(theta), 1))
    student_num = data.shape[0]
    question_num = data.shape[1]

    for i in range(student_num):
        derivative = 0
        for j in range(question_num):
            c_ij = data[i, j]
            if c_ij == 0 or c_ij == 1:
                derivative += c_ij - 1 / (1 + 1/np.exp(theta[i] - beta[j]))
        theta_derivative[i] = derivative
    return theta_derivative


def derivative_beta(data, theta, beta):
    """Compute derivative of MLE respect to beta

        :param data:
        :param theta:
        :param beta:
        :return:
        """
    beta_derivative = np.zeros((len(beta), 1))
    student_num = data.shape[0]
    question_num = data.shape[1]

    for j in range(question_num):
        derivative = 0
        for i in range(student_num):
            c_ij = data[i, j]
            if c_ij == 0 or c_ij == 1:
                derivative += (- c_ij) + 1 / (1 + 1/np.exp(theta[i] - beta[j]))
        beta_derivative[j] = derivative
    return beta_derivative


def update_theta_beta(data, lr, theta, beta):
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
    theta_derivative = derivative_theta(data, theta, beta)
    theta += lr * theta_derivative
    beta_derivative = derivative_beta(data, theta, beta)
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

    for i in range(iterations):
        train_neg_lld = neg_log_likelihood(train_data, theta=theta, beta=beta)
        train_loglikes.append(train_neg_lld)

        valid_loglike = neg_log_likelihood(val_data, theta, beta)
        valid_loglikes.append(valid_loglike)

        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        # print("NLLK: {} \t Score: {}".format(train_neg_lld, score))
        theta, beta = update_theta_beta(train_sparse, lr, theta, beta)

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
    lr = 0.016
    theta, beta, train_log, valid_log = irt(train_data, sparse_matrix,
                                            val_data, lr, 10)
    # plot log-likelihood as training process
    num_iter = [i for i in range(10)]
    plt.plot(num_iter, valid_log, linestyle='--',
             color='b', label="validation")
    plt.plot(num_iter, train_log, linestyle='--',
             color='r', label="train")
    plt.title("Change for log likelihoods")
    plt.xlabel('Iterations', fontsize=14)
    plt.ylabel('loglike', fontsize=14)
    plt.legend(loc=1)
    plt.show()

    # (question d)
    question = [1, 2, 3, 4, 5]
    probs = {}
    num_theta = len(theta)
    for j in question:
        probs[j] = []
        beta_j = beta[j]
        for i in range(num_theta):
            probs[j].append(sigmoid(theta[i] - beta_j)[0])

    sns.lineplot(x=theta, y=probs[1], label="question id 1")
    sns.lineplot(x=theta, y=probs[2], label="question id 2")
    sns.lineplot(x=theta, y=probs[3], label="question id 3")
    sns.lineplot(x=theta, y=probs[4], label="question id 4")
    sns.lineplot(x=theta, y=probs[5], label="question id 5")
    plt.title("Change for probability")
    plt.xlabel('theta', fontsize=14)
    plt.ylabel('probability', fontsize=14)
    plt.show()

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
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()


