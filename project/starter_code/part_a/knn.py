from sklearn.impute import KNNImputer
from utils import *
import matplotlib.pyplot as plt


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix.T)
    acc = sparse_matrix_evaluate(valid_data, mat.T)
    print("Validation Accuracy: {}".format(acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    k_values = [1,6,11,16,21,26]
    k_performance_user = [None] * 6
    k_star_user = 0
    print("by users")
    for i in range(len(k_values)):
        k_performance_user[i] = knn_impute_by_user(sparse_matrix, val_data, k_values[i])
        if k_performance_user[k_star_user] < k_performance_user[i]:
            k_star_user = i
    print("We find that the best value for k, k* is", k_values[k_star_user] )
    plt.plot(k_values,k_performance_user, label = 'user')
    plt.legend()
    plt.savefig('../figs/knn_user')
    plt.cla()
    print("by questions")
    k_performance_item = [None] * 6
    k_star_item = 0
    for i in range(len(k_values)):
        k_performance_item[i] = knn_impute_by_item(sparse_matrix, val_data, k_values[i])
        if k_performance_item[k_star_item] < k_performance_item[i]:
            k_star_item = i
    print("We find that the best value for k, k* is", k_values[k_star_item] )
    plt.plot(k_values,k_performance_item, label = 'item')
    plt.legend()
    plt.savefig('../figs/knn_item')
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
