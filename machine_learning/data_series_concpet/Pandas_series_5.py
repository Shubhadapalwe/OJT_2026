import pandas as pd
# series should contain homogenous data(data of same data types)
def main():
   
    # we can use custumized indexing as strings
    sobj = pd.Series([25000,27000,29000,30000],index=["PPA","LB","Python","React"])
    print(sobj)
    print(sobj["Python"]) # uses key to display the value

if __name__ == "__main__":
    main()