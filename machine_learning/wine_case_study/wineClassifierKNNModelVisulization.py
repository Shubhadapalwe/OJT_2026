
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.preprocessing import StandardScaler
def MarvellousClassifier(Datapath):
    border = "-"*40
#----------------------------------------------------------
#Step 1: 
#------------------------------------------------------------    
    print(border)
    print("step 1: load the data set from csv file")
    print(border)
    
    df = pd.read_csv(Datapath)
    print(border)
    print("Some entries from dataset")
    print(df.head())
    print(border)

#----------------------------------------------------------
#step 2 : clean the dataset by removing empty rows 
#------------------------------------------------------------
    print(border)
    print("step 2 : clean the dataset by removing empty rows")
    print(border)

    df.dropna(inplace=True)
    print("total records :",df.shape[0])
    print("Total colunms :",df.shape[0])
#----------------------------------------------------------
#step 3 : Separate independt and dependent variables 
#------------------------------------------------------------
    print(border)
    print("step 3 : Separate independt and dependent variables")
    print(border)

    X = df.drop(columns=['Class'])
    Y = df['Class']

    print("Shape of X :",X.shape)
    print("Shape of Y :",Y.shape)

    print(border)
    print("Input colunms :",X.columns.to_list())
    print("Output colunms :Class")

#----------------------------------------------------------
#step 4: split the data set for training and testing 
#------------------------------------------------------------
    print(border)
    print("step 4: split the data set for training and testing")
    print(border)

    X_train , X_test , Y_train, Y_test = train_test_split(X,Y,test_size= 0.2,random_state=42,stratify=Y)
    print(border)
    print("Information of training and testing data")
    print("X_train shape :",X_train.shape)
    print("X_test shape  :",X_test.shape)
    print("Y_train shape :",Y_train.shape)
    print("Y_test shape  :",Y_test.shape)
    print(border)
#----------------------------------------------------------
#step 5 : feature Scaling
#------------------------------------------------------------
    print(border)
    print("step 5 : feature Scaling ")
    print(border)

    scalar = StandardScaler()
    #independent variable scaling
    X_train_scaled = scalar.fit_transform(X_train)
    X_test_scaled = scalar.fit_transform(X_test)

    print("Feature scaling is done ")
#----------------------------------------------------------
#step 6: Explore the multiple values of K 
# Hyperparameter tunning (K)
#------------------------------------------------------------
    print(border)
    print("step 6: Explore the multiple values of K ")
    print(border)

    accuracy_scores = []
    K_values = range(1,21)

    for k in K_values:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train_scaled,Y_train)
        Y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(Y_test,Y_pred)
        accuracy_scores.append(accuracy)

    print(border)
    print("Accuracy report of all K values from 1 to 20")
    for value in accuracy_scores:
        print(value)
    print(border)

#----------------------------------------------------------
#step 7 : plot graph of K vs Accuracy
#------------------------------------------------------------
    print(border)
    print("step 6: Explore the multiple values of K ")
    print(border)

    plt.figure(figsize =(8,5))
    plt.plot(K_values,accuracy_scores,marker = 'o')
    plt.title("k value vs Accuracy")
    plt.xlabel("Value of K")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.xticks(list(K_values))
    plt.show()

    
def main():
    border = "-"*40
    print(border)
    print("Wine classifier using KNN algorithm")
    print(border)

    MarvellousClassifier("WinePredictor.csv")

if __name__ == "__main__":
    main()