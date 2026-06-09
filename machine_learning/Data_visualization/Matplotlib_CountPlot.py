import matplotlib.pyplot as plt
import seaborn as sns

def main():
    
    sns.countplot(x=["A","B","A","A","B","A","C"])
    # jar countigus variable astil tr histogram
    # if values are categorial use countplot
    plt.show()
if __name__ == "__main__":
    main()