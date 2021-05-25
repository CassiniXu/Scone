""" This file just compares the lines in one CSV to another CSV, matching up
keys (column 'id'). It can be used to compute both interaction-level and instruction-level
accuracy as long as the two files match up with IDs. 

For instruction-level accuracies, the label files (train_y.csv and dev_y.csv) contain
IDs specific to the instruction index. E.g., "train-1234-0" is the 0th instruction
in interaction #1234.
"""

import pandas as pd
from sklearn.metrics import accuracy_score
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-p", "--predicted", dest="pred_path", required=True, help="path to your predicted labels for accuracy")
parser.add_argument("-l", "--labels", dest="labels_path", required=True, help="path to the file containing the labels")

args = parser.parse_args()

pred = pd.read_csv(args.pred_path, index_col="id")
labels = pd.read_csv(args.labels_path, index_col="id")

pred.columns = ["predicted"]
labels.columns = ["actual"]

data = labels.join(pred)

print("Accuracy: ", accuracy_score(data.actual, data.predicted))
