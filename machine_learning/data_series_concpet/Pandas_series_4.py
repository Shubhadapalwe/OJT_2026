import pandas as pd

def main():
   
    #Custum indexing 
    sobj = pd.Series([11.0,21.0,51.0,101.0,11.0],index=[5,6,7,8,9])
    print(sobj)
if __name__ == "__main__":
    main()