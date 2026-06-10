import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
#--------------------------------------------------------------------
#   Fucntion name :   LoadPreserveModel
#   Description :    Used to load preserved model
#   Parameter :     df 
#   Return :        None
#   Date :          14/03/2026
#   Author :        Shubhada Anil Palwe
#--------------------------------------------------------------------
def LoadPreserveModel(filename):
    loaded_model = joblib.load(filename)
    print("Model successfully loaded")
    return loaded_model

#--------------------------------------------------------------------
#   Fucntion name :   reserveModel
#   Description :    Used to preseve the model on secondary
#   Parameter :     df 
#   Return :        None
#   Date :          14/03/2026
#   Author :        Shubhada Anil Palwe
#--------------------------------------------------------------------
    
def preserveModel(model,filename):
    joblib.dump(model,filename)

    print("Model preserve successfully with name :",filename)

#--------------------------------------------------------------------
#   Fucntion name :   CTrainTitanicModel
#   Description :    It does split X,Y,traing data , testing data
#   Parameter :     df 
#   Return :        None
#   Date :          14/03/2026
#   Author :        Shubhada Anil Palwe
#--------------------------------------------------------------------
def TrainTitanicModel(df):
    #split feature and labels
    X = df.drop("Survived",axis = 1)
    Y = df["Survived"]

    print("\n Features :")
    print(X.head())

    print("\n lables :")
    print(Y.head())

    print("Shape of X :",X.shape)
    print("Shape of Y:",Y.shape)

    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state= 42)
    print("X_train shape :",X_train.shape)
    print("X_test shape :",X_test.shape)
    print("Y_train shape :",Y_train.shape)
    print("Y_test shape :",X_test.shape)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train,Y_train)

    print("Model trained successfully")

    print("\n Intercept of model :")
    print(model.intercept_)

    print("\n Coeeficient of model")
    for feature,coefocient in zip(X.columns,model.coef_[0]):
        print(feature," : ",coefocient)

    preserveModel(model,"Marvelloustitanic.pkl")

    loaded_model = LoadPreserveModel("Marvelloustitanic.pkl")
    Y_pred = loaded_model.predict(X_test)
    accuracy = accuracy_score(Y_pred,Y_test)
    print("Accuracy is :",accuracy)
    cm = confusion_matrix(Y_pred,Y_test)

    print("Confusion matrix is:")
    print(cm)

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

    #Encode Embarked column
    
    df = pd.get_dummies(df,columns=["Embarked"],drop_first=True)
    print("\n Data after encoding")

    print(df.head())
    print("Shape of Dataset",df.shape)

    #convert boolean columns into integers
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)
    print("\n Data after encoding")
    print(df.head())

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

    TrainTitanicModel(df)


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