# Rough = 1
# Smooth = 0

# Tennis = 1
# Cricket = 2

from sklearn import tree

def main():
    print("Ball classification case study")

    # Independent variables: [weight, surface]
    X = [
        [35, 1], [47, 1], [90, 0], [48, 1], [90, 0],
        [35, 1], [92, 0], [35, 1], [35, 1], [35, 1],
        [96, 0], [43, 1], [110, 0], [35, 1], [95, 0]
    ]

    # Dependent variables (labels): Tennis=1, Cricket=2
    Y = [1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 2, 1, 2]

    modelobj = tree.DecisionTreeClassifier()

    trainedmodel = modelobj.fit(X, Y)

    result = trainedmodel.predict([[37, 1], [94, 0]])
    print("Model predicts the object as:", result)

if __name__ == "__main__":
    main()