# ========================================================
# BALL CLASSIFICATION CASE STUDY USING DECISION TREE
# ========================================================

# ========================================================
# Import Required Library
# ========================================================

from sklearn import tree

Border = "-" * 50


# ========================================================
# Main Function
# ========================================================

def main():

    print(Border)
    print("BALL CLASSIFICATION CASE STUDY")
    print(Border)

    # ========================================================
    # Step 1 : Define Dataset
    # ========================================================

    print("\nStep 1 : Define Dataset")

    # Independent Variables (Weight, Surface Type)
    X = [
        [35,1],[47,1],[90,0],[48,1],[90,0],
        [35,1],[92,0],[35,1],[35,1],[35,1],
        [96,0],[43,1],[110,0],[35,1],[95,0]
    ]

    # Dependent Variable (Ball Type)
    Y = [1,1,2,1,2,1,2,1,1,1,2,1,2,1,2]

    print("Dataset defined successfully")


    # ========================================================
    # Step 2 : Split Dataset into Training & Testing
    # ========================================================

    print("\nStep 2 : Split Dataset")

    Xtrain = X[:13]
    Ytrain = Y[:13]

    Xtest = X[13:]
    Ytest = Y[13:]

    print("Training Size :", len(Xtrain))
    print("Testing Size :", len(Xtest))


    # ========================================================
    # Step 3 : Build Decision Tree Model
    # ========================================================

    print("\nStep 3 : Build Model")

    modelobj = tree.DecisionTreeClassifier()
    print("Model created successfully")


    # ========================================================
    # Step 4 : Train Model
    # ========================================================

    print("\nStep 4 : Train Model")

    trainedmodel = modelobj.fit(Xtrain, Ytrain)
    print("Model training completed")


    # ========================================================
    # Step 5 : Test Model
    # ========================================================

    print("\nStep 5 : Test Model")

    Result = trainedmodel.predict([[35,1]])

    print("Prediction Result :", Result)


    # ========================================================
    # Step 6 : Display Final Output
    # ========================================================

    print("\nStep 6 : Final Classification")

    if Result == 1:
        print("Object looks like a Tennis Ball")
    elif Result == 2:
        print("Object looks like a Cricket Ball")


# ========================================================
# Entry Point
# ========================================================

if __name__ == "__main__":
    main()