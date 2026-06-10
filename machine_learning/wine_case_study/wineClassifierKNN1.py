
import pandas as pd
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
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


def main():
    border = "-"*40
    print(border)
    print("Wine classifier using KNN algorithm")
    print(border)

    MarvellousClassifier("WinePredictor.csv")

if __name__ == "__main__":
    main()