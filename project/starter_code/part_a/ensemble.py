# TODO: complete this file.
import pandas as pd
import numpy as np
from utils import *
from sklearn.tree import DecisionTreeClassifier

def main():
   train_data = pd.DataFrame(load_train_csv("../data"))
   test_data = pd.DataFrame(load_public_test_csv("../data"))

   num_tree = 3
   predictions = []
   predictors = ['user_id', 'question_id']

   for i in range(num_tree):
      boostrapt_sample = train_data.sample(frac=1, replace=True, random_state=i)
      tree = DecisionTreeClassifier(random_state=1, min_samples_leaf=2,
                                    splitter="random")
      tree.fit(boostrapt_sample[predictors], boostrapt_sample['is_correct'])
      predictions.append(tree.predict(test_data[predictors]))

   combined_prediction = np.sum(predictions, axis=0)/3
   return (np.sum((combined_prediction > 0.5) == test_data["is_correct"])
           / float(len(test_data["is_correct"])))


if __name__ == "__main__":
   main()
