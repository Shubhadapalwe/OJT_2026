import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
def main():
    df = pd.read_csv("Advertising.csv")
    print(df.shape)
    # data cleaning (unwanted coloum removed from the csv)
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'],inplace=True)
    print(df.shape)
    #split the data into X and Y
    X = df [['TV','radio','newspaper']]
    Y = df['sales']
    print("Independent variales :",X.shape)
    print("Dependent variables :",Y.shape)

    # split the data for training and testing
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size= 0.1,random_state=42)
    model = LinearRegression()

    model.fit(X_train,Y_train)

    Y_pred = model.predict(X_test)

    print("testing data :")
    print(X_test)

    print("Predicted values :")
    print(Y_pred)

    print("Actual values :")
    print(Y_test)

    print("Coefficient : ",model.coef_)
    print("intercept :",model.intercept_)

if __name__ == "__main__":
    main()