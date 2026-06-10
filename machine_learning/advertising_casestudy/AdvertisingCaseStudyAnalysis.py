import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

def MarvellousAdvertise(datapath):
    border=("-"*40)
#---------------------------------------------------------
# step 1: Load data set
#---------------------------------------------------------
    print(border)
    print("Step 1: load data set")
    print(border)

    df = pd.read_csv(datapath)
    print("Few records from the dataset:")
    print(df.head())
#---------------------------------------------------------
# step 2 : Remove unwanted colums
#---------------------------------------------------------
    print(border)
    print("Step 2: remove unwanted colums")
    print(border)
    print("Shape of dataset before removal",df.shape)


    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'],inplace= True)
    print("shape of dataset after removal",df.shape)

    print(border)
    print("Clean dataset is:")
    print(border)


#-------------------------------------------------
# step 3 : check missing values
#--------------------------------------------------

    print(border)
    print("Step 3: check missing value")
    print(border)

    print("missing values count \n :",df.isnull().sum())

#-------------------------------------------------
# step 4 : Display staatistical summary
#--------------------------------------------------

    print(border)
    print("step 4 : Display staatistical summary")
    print(border)

    print(df.describe())

#-------------------------------------------------
# step 5 : Correlation between columns
#--------------------------------------------------

    print(border)
    print("step 5 : Correlation between columns ")
    print(border)

    print("Correlation matrix :")
    print(df.corr())
    

def main():
    MarvellousAdvertise("Advertising.csv")
if __name__ == "__main__":
    main()