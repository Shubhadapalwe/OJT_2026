from sklearn.datasets import load_iris
from sklearn import tree

def main():
    print("Iris classification case study")

    Dataset = load_iris()
    
    Border = "-"*40
    print(Border)

    for i in range(len(Dataset.target)):
        print("ID %d, features %s, label %s"%(i,Dataset.data[i],Dataset.target[i]))
    print(Border)
if __name__ == "__main__":
    main()
