from sklearn.datasets import load_iris
from sklearn import tree

def main():
    print("Iris classification case study")

    Dataset = load_iris()
    print(Dataset)

if __name__ == "__main__":
    main()
