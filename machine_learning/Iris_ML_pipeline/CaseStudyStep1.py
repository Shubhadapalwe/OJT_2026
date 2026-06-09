import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier,plot_tree

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)
Border = "-"*40
##############################################################################
# Step 1 : Load the dataset
##############################################################################
print(Border)
print("Step 1 : Load the DataSet")
print(Border)

DataSetPath = "iris.csv"

df = pd.read_csv(DataSetPath)
print("DataSet gets loaded sucessfully")
print("Initial entries from dataset :")
print(df.head())



