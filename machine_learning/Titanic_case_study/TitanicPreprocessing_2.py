import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
#--------------------------------------------------------------------
#   Fucntion name :   CleanTitanicData
#   Description :     It does precessing 
#                     It removes unnecessary columns
#                     It handles missing values
#                     It convert text data to numric format
#                     It does encoding to categorical columns
#   Parameter :     df -> Pandas Data frame
#   Return :        df -> clean Pandas dataframe 
#   Date :          14/03/2026
#   Author :        Shubhada Anil Palwe
#--------------------------------------------------------------------
def CleanTitanicData(df):
    DisplayInfo("Step 2:- Original Data")
    print(df.head())
    # Remove unnecessary columns
    drop_columns = ["Passengerid","zero","Name","Cabin"]
    existing_columns = [col for col in drop_columns if col in df.columns]

    print("\n Columns to be dropped : ")
    print(existing_columns)
    
    #Dropped the unwanted columns
    df = df.drop(columns = existing_columns)

    DisplayInfo("Step 2:- Data After column removal")
    print(df.head())
    # Handle Age column
    if "Age" in df.columns:
        print("Age column before filling missing values")
        print(df["Age"].head(10))

        # invalid value gets converted into Nan
        df["Age"] = pd.to_numeric(df["Age"],errors="coerce")

        age_median = df["Age"].median()

        #Replace missing values with median 
        df["Age"] = df["Age"].fillna(age_median)

        print("\n Age column after preprocessing :")
        print(df["Age"].head(10))

        # Handel fare column
    if "Fare" in df.columns:
        print("\n Fare column before preprocessing")
        print(df["Fare"].head(10))

        df["Fare"] = pd.to_numeric(df["Fare"],errors="coerce")
        fare_median = df["Fare"].median()

        #Replace missing values with median 
        df["Fare"] = df["Fare"].fillna(fare_median)

        fare_median = df["Fare"].median()

        print("\n Fare column after preprocessing :")
        print(df["Fare"].head(10))
    
        # handel embarked column
    if "Embarked" in df.columns:
        print("\n Embarked column before preprocessing")
        print(df["Embarked"].head(10))

        #convert the data into string
        df["Embarked"] = df["Embarked"].astype(str).str.strip()

        # remove missing values
        df["Embarked"] = df["Embarked"].replace(['nan','None',''],np.nan)

        # get most frequent value (Frequency)
        embarked_mode = df["Embarked"].mode()[0]
        print("mode of embarked column :",embarked_mode)

        df["Embarked"] = df["Embarked"].fillna(embarked_mode)
        
        print("\Embarked column after preprocessing")
        print(df["Embarked"].head(10))

            # Handel fare column
    if "Sex" in df.columns:
        print("\n Sex column before preprocessing")
        print(df["Sex"].head(10))

        df["Sex"] = pd.to_numeric(df["Sex"],errors="coerce")
        fare_median = df["Sex"].median()

        print("\Sex column after preprocessing")
        print(df["Sex"].head(10))

    DisplayInfo("Data after preprocessing")
    print(df.head())
    print("\n Missing values after preprocessing")
    print(df.isnull().sum())


    return df

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
    df = CleanTitanicData(df)


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