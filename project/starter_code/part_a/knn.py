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
    nbrs2 = KNNImputer(n_neighbors=k)
    mat2 = nbrs2.fit_transform(matrix.T)
    acc = sparse_matrix_evaluate(valid_data, mat)
    #print("Validation Accuracy: {}".format(acc))
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
    # Again, use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix.T) # take transpose to impute on
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
    opt_k, opt_acc = None, float("-inf")
    k_vals = [1, 6, 11, 16, 21, 26]
    for k in k_vals:
        acc = knn_impute_by_user(sparse_matrix, val_data, k)
        if acc > opt_acc:
            opt_acc = acc
            opt_k = k
        plt.plot(k, acc, "ro")
    plt.show()

    # Question 1a
    print(f"Optimal k: {opt_k}, top accuracy for validation set: {opt_acc}")

    # Question 1b
    print(f"Test accuracy for k={opt_k}: {knn_impute_by_user(sparse_matrix, test_data, opt_k)}")
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
