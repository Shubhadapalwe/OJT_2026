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
print("Step 3 : Decide Independent and Dependednt variables")
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

########################################################
# Step 4 : Visualization of Dataset
#########################################################

print(Border)
print("Step 4 : Visualization of Dataset")
print(Border)

# scatter plot
plt.figure(figsize=(7,5))

for sp in df["species"].unique():
    temp = df[df["species"]==sp]
    plt.scatter(temp["petal length (cm)"],temp["petal width (cm)"],label = sp)
    
plt.title("Iris : Petal length vs petal width")
plt.xlabel("petal lenght (cm)")
plt.ylabel("petal width (cm)")

plt.legend()
plt.grid(True)
plt.show()

########################################################
# Step 5 : Split the dataSet for training and testing
#########################################################

print(Border)
print("Step 5 : Split the dataSet for training and testing")
print(Border)

# total Data set : 150,5
#X: 150,4
# Y = 150,1
# test Size = 20% 
# train Size = 80%

X_train , X_test ,Y_train, Y_test = train_test_split(

    X,
    Y,
    test_size= 0.5,  #keyword argument
    random_state= 42
)
print("Data spliting activity done: ")

print("X - Independent :",X.shape)  #(150,4)
print("Y - Dependent:",Y.shape) # (150,)

print("X_train :",X_train.shape)    #(120,4)
print("X_tets :",X_test.shape)      #(30,4)

print("Y_train :", Y_train.shape)   # (120,)
print("Y_test :",Y_test.shape)      # (30,)


#########################################################
# Step 6 : Build the model
#########################################################

print(Border)
print("Step 6 : Build the model")
print(Border)

print("We are going to use decision Tree classifier ")
model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,    # this will treated as hyper-parameter tunning
    random_state= 42
)

print("model Successfully created :",model)

#########################################################
# Step 7 : Train the model
#########################################################

print(Border)
print("Step 7 : Train the model")
print(Border)

model.fit(X_train,Y_train)
print("Model training completed")

#########################################################
# Step 8 : Evaluate the model
#########################################################

print(Border)
print("Step 8 : Evaluate the model")
print(Border)

Y_pred = model.predict(X_test)

print("Model evaluate (testing) complete")

print(Y_pred.shape)
print(Y_pred)

#########################################################
# Step 9 : Evaluate of the model perfrmance
#########################################################

print(Border)
print("Step 9 : Evaluate of the model performance")
print(Border)

accuracy = accuracy_score(Y_test,Y_pred)
print("Accuracy ofmodel is :",accuracy*100)

cm = confusion_matrix(Y_test,Y_pred)
print("Confusion matrix :")
print(cm)

print("Classification report")
print(classification_report(Y_test,Y_pred))

#########################################################
# Step 10 : plot confusion matrix
#########################################################

print(Border)
print("Step 10 : plot confustion matrix")
print(Border)

data = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model.classes_) 
data.plot()
plt.title("Confusion matrix of iris data set")
plt.show()