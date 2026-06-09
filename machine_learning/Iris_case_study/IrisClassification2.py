from sklearn.datasets import load_iris
from sklearn import tree

def main():
    print("Iris classification case study")

    Dataset = load_iris()
    # meta data of data set
    print("Independent variables are :")
    print(Dataset.feature_names)    # prints header's names (colums names)Independent variables are :['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

    print("Dependent variables are :" )
    print(Dataset.target_names) #Dependent variables are :['setosa' 'versicolor' 'virginica']

if __name__ == "__main__":
    main()
