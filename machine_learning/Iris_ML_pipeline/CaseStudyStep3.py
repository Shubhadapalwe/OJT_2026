import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

Border = "-"*40

#########################################################
# Step 1 : Load the dataset
#########################################################

print(Border)
print("Step 1 : Load the dataset")
print(Border)

DatasetPath = "iris.csv"

df = pd.read_csv(DatasetPath)

print("Dataset gets loaded succesfully...")
print("Initial entries from dataset :")
print(df.head())

#########################################################
# Step 2 : Data Analysis (EDA)
#########################################################

print(Border)
print("Step 2 : Data analysis")
print(Border)

print("Shape of dataset : ",df.shape)   #shape is a property (dispaly (150,5))
print("Column Names : ",list(df.columns))   # type casting to get result in row-wise

print("Missing values (Per Column)")
print(df.isnull().sum())    # identify missing values

print("Class Distribution (Species count)")
print(df["species"].value_counts())

print("Statistical Report of dataset")
print(df.describe())

#########################################################
# Step 3 : Decide Independent and Dependednt variables
#########################################################

print(Border)
print("Step 2 : Decide Independent and Dependednt variables")
print(Border)

# X: Independent variables for Features 
# Y: Dependent Variables for Lables
feature_cols = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)"
]
X = df[feature_cols]
Y = df["species"]
print("X shape:",X.shape)
print("Y shape :",Y.shape)