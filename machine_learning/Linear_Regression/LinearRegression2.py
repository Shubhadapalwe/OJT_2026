import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def MarvellousPredictor():
    # load the data
    X = [1,2,3,4,5]
    Y = [3,4,2,4,5]

    print("Values of independent variables : X",X)
    print("Values of Dependet variables  : Y",Y)

    mean_x = np.mean(X)
    mean_y = np.mean(Y)

    print("X_MEAN is :",mean_x)     # 3.0
    print("Y_MEAN is :",mean_y)     # 3.6

    n = len(X)  #5

    # Y = mX + C
    # formaula for m = slope 
    # m = (summ (X-X_bar) * (Y-Y_bar))/(summ (X-X_bar) ** 2)
    numerator = 0
    denominator = 0

    for i in range(n):
        numerator = numerator + ((X[i]-mean_x ) * (Y[i] - mean_y))
        denominator = denominator + ((X[i] - mean_x) ** 2)
    m = numerator / denominator
    print("Slope of Line i.e., (m) : ",m)   # 0.4

    C = mean_y - (m * mean_x) 
    print("Y - intercept of line i.e., C : ",C)     #   2.4
    






def main():
    MarvellousPredictor()
if __name__ == "__main__":
    main()