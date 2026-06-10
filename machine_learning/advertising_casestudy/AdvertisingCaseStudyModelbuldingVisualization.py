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

    print(df.head())


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
#---------------------------------------------------------------------
# step 6 : Split dataset into independent and dependent varaibles
#----------------------------------------------------------------

    print(border)
    print("step 6 : Split dataset into independent and dependent varaibles ")
    print(border) 

    X = df[['TV','radio','newspaper']]

    Y =  df['sales']
    print("shape of Independet variables :",X.shape)
    print("shape of Dependet variables :",Y.shape)

#---------------------------------------------------------------------
# step 7 : Split dataset for training and testing
#----------------------------------------------------------------
    print(border)
    print("step 7 : Split dataset for training and testing")
    print(border) 

    X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size=0.05,random_state=42)
    print("X_train shape ",X_train.shape)
    print("Y_train shape ",Y_train.shape)
    print("Y_train shape ",Y_train.shape)
    print("Y_test shape ",Y_test.shape)
#---------------------------------------------------------------------
# step 8 : create and train the model
#----------------------------------------------------------------
    print(border)
    print("step 8 : create and train the model")
    print(border) 

    model = LinearRegression()

    model.fit(X_train,Y_train)
#---------------------------------------------------------------------
# step 9 : test the model
#----------------------------------------------------------------
    print(border)
    print("step 9 : test the model")
    print(border)  

    Y_pred = model.predict(X_test)

#---------------------------------------------------------------------
# step 10 : evaluate the model
#----------------------------------------------------------------
    print(border)
    print("step 10 : evaluate the model")
    print(border)  

    MSE = mean_squared_error(Y_test,Y_pred)
    RMSE = np.sqrt(MSE)
    R2 = r2_score(Y_test,Y_pred)

    print("mean squared error :",MSE)
    print("Root mean squared error :",RMSE)
    print("R square value :",R2) 
#---------------------------------------------------------------------
# step 11 : calculate model coefficient
#----------------------------------------------------------------
    print(border)
    print("step 11 : calculate model coefficient")
    print(border) 

    for colunm, value in zip(X.columns,model.coef_):
        print(f"{colunm} : {value}")
    print("Intercept :",model.intercept_)
#---------------------------------------------------------------------
# step 12 : compare the actual and predicted values
#----------------------------------------------------------------
    print(border)
    print("step 12 : compare the actual and predicted values")
    print(border) 

    Result = pd.DataFrame({
        'Actual sale ': Y_test.values, 
        'Predicted sale ': Y_pred
        })
    print(Result.head(10))   
#---------------------------------------------------------------------
# step 13: plot actual vs predicted
#----------------------------------------------------------------
    print(border)
    print("step 13: plot actual vs predicted")
    print(border)

    plt.figure(figsize=(8,5))
    plt.scatter(Y_test,Y_pred)
    plt.xlabel("Actual sales")
    plt.ylabel("Predicted sales")
    plt.title("Actual sales vs predicted sales") 
    plt.grid(True)
    plt.show()


def main():
    MarvellousAdvertise("Advertising.csv")
if __name__ == "__main__":
    main()