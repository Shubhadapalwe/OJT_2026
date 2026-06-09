import pandas as pd
# specific accessing in dataframe
def main():
    Data = {
        "Name":["Sagar","Amit","Pooja"],
        "Age":[23,26,25],
        "City":["Pune","Mumbai","Satara"]
    }

    dobj = pd.DataFrame(Data)
    # featch specific row
    print(dobj.loc[0])  # it will featch the row


if __name__ == "__main__":
    main()