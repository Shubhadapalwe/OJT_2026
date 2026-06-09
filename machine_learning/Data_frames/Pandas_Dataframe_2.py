import pandas as pd
# specific accessing in dataframe
def main():
    Data = {
        "Name":["Sagar","Amit","Pooja"],
        "Age":[23,26,25],
        "City":["Pune","Mumbai","Satara"]
    }

    dobj = pd.DataFrame(Data)
    print(dobj)
    print("-------------")
    print(dobj["Age"])
    print("-------------")
    print(dobj["City"])

if __name__ == "__main__":
    main()