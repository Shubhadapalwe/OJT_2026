from sklearn.metrics import r2_score

def main():
    Y_actual = [3,4,2,4,5]      # Y
    Y_predicted = [3,4,2,4,5] #Yp

    r2 = r2_score(Y_actual,Y_predicted)
    print("Actual value : Y ",Y_actual)
    print("Predicted values : yp ",Y_predicted)
    print("R Square value : ",r2)

    
if __name__ == "__main__":
    main()