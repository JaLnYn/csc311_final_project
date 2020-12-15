# TODO: complete this file.
import pandas as pd
import numpy as np
from utils import *
from sklearn.tree import DecisionTreeClassifier
import scipy
import item_response as irt
import matrix_factorization as mfac


def train_irt(dense, sparse):
    dense2 = sparse.toarray()
    nonz = np.nan_to_num(np.where(dense2 == 0., 1., dense2)).nonzero()
    theta = np.zeros((sparse.shape[0], 1))
    beta = np.zeros((sparse.shape[1], 1))

    print("Training IRT model...")
    for i in range(10):
        print(i, end=" ")
        theta, beta = irt.update_theta_beta(sparse, 0.016, theta, beta, nonz)
    print("")
    return theta, beta


def pred_irt(data, theta, beta):
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = irt.sigmoid(x)
        pred.append(p_a >= 0.5)
    return pred


def train_mfac(data):
    print("Training ALS matrix factorization model...")
    factorized = mfac.als(data, 35, 0.01, 500000)
    return factorized


def pred_mfac(data, factorized):
    pred = []
    for n in range(len(data["user_id"])):
        pred.append(factorized.item(data["user_id"][n], data["question_id"][n]) > 0.5)
    return pred


def main():
    train_notdf = load_train_csv("../data")
    train_data = pd.DataFrame(train_notdf)
    test_notdf = load_public_test_csv("../data")
    test_data = pd.DataFrame(test_notdf)
    valid_notdf = load_valid_csv("../data")
    valid_data = pd.DataFrame(valid_notdf)

    num_tree = 1
    test_preds = []
    valid_preds = []
    predictors = ['user_id', 'question_id']
    np.random.seed(0)

    irt_bootstrap = train_data.sample(frac=1, replace=True, random_state=1)
    irt_sparse = scipy.sparse.csr_matrix(
        irt_bootstrap.pivot_table(values="is_correct", index="user_id", columns="question_id"))
    irt_model = train_irt(train_data, irt_sparse)
    test_preds.append(pred_irt(test_data, *irt_model))
    valid_preds.append(pred_irt(valid_data, *irt_model))

    mfac_sample = np.random.choice(range(len(train_data["user_id"])), len(train_data["user_id"]))
    mfac_bootstrap = {
        "user_id": np.take(train_notdf["user_id"], mfac_sample),
        "question_id": np.take(train_notdf["question_id"], mfac_sample),
        "is_correct": np.take(train_notdf["is_correct"], mfac_sample),
    }
    mfac_model = train_mfac(mfac_bootstrap)
    testvals = test_data.values
    test_preds.append(pred_mfac({
        "user_id": testvals[:, 0],
        "question_id": testvals[:, 1],
        "is_correct": testvals[:, 2]
    }, mfac_model))
    validvals = valid_data.values
    valid_preds.append(pred_mfac({
        "user_id": validvals[:, 0],
        "question_id": validvals[:, 1],
        "is_correct": validvals[:, 2]
    }, mfac_model))

    #for i in range(num_tree):
    #    boostrapt_sample = train_data.sample(frac=1, replace=True, random_state=i)
    #    tree = DecisionTreeClassifier(random_state=1, min_samples_leaf=2,
    #                                  splitter="random")
    #    tree.fit(boostrapt_sample[predictors], boostrapt_sample['is_correct'])
    #    test_preds.append(tree.predict(test_data[predictors]))
    #    valid_preds.append(tree.predict(valid_data[predictors]))

    comb_test_preds = np.sum(test_preds, axis=0) / 3
    comb_valid_preds = np.sum(valid_preds, axis=0) / 3
    test_acc = (np.sum((comb_test_preds > 0.5) == test_data["is_correct"])
            / float(len(test_data["is_correct"])))
    valid_acc = (np.sum((comb_valid_preds > 0.5) == valid_data["is_correct"])
                / float(len(valid_data["is_correct"])))
    print(f"Validation accuracy: {valid_acc}")
    print(f"Test accuracy: {test_acc}")



if __name__ == "__main__":
    main()
