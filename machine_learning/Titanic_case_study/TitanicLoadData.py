import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix

#--------------------------------------------------------------------
#   Fucntion name :   DisplayInfo
#   Description :   It displays the formated title 
#   Parameter :     title (str)
#   Return :        None
#   Date :          14/03/2026
#   Author :        Shubhada Anil Palwe
#--------------------------------------------------------------------
def DisplayInfo(title):
    print("\n" + "="*70)
    print(title)
    print("="*70)
#--------------------------------------------------------------------
#   Fucntion name : ShowData
#   Description : It shows basic information about dataset
#   Parameter : df
#               df -> pandas dataframe object
#               Message
#               Message -> Heading text to display
#   Return :    None
#   Date :      14/03/2026
#   Author :    Shubhada Anil Palwe
#--------------------------------------------------------------------
def ShowData(df,message):
    DisplayInfo(message)

    print("First 5 rows of dataset")
    print(df.head())

    print("\n Shape of Data set")
    print(df.shape)

    print("\n Column names")
    print(df.columns.tolist())

    print("\n Missing values in each column")
    print(df.isnull().sum())



#--------------------------------------------------------------------
#   Fucntion name : MarvellousTitanicLogistic
#   Description : This is main pipeline controller. It loads the dataset , 
#                 show raw data ,preposess the data set and train the model
#   Parameter : Datapath of dataset file
#   Return :    None
#   Date :      14/03/2026
#   Author :    Shubhada Anil Palwe
#--------------------------------------------------------------------
def MarvellousTitanicLogistic(Datapath):
    DisplayInfo("Step 1:- Loding the data set")
    df = pd.read_csv(Datapath)
    
    ShowData(df,"Initial Dataset")


#--------------------------------------------------------------------
#   Fucntion name : Main
#   Description : Starting point of application
#   Parameter : None
#   Return :    None
#   Date :  14/03/2026
#   Author :    Shubhada Anil Palwe
#--------------------------------------------------------------------

def main():
    MarvellousTitanicLogistic("MarvellousTitanicDataset.csv")


if __name__ =="__main__":
    main()