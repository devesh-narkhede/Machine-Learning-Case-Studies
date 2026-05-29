# ========================================================
# WINE CLASSIFICATION CASE STUDY
# USING K-NEAREST NEIGHBORS (KNN)
# ========================================================

# ========================================================
# Import Required Libraries
# ========================================================
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# =========================================================
# Function Name :  WineClassifier
# Description   :  Classifies wine types based on features
#                  using K-Nearest Neighbors algorithm.
#                  Includes data loading, cleaning, scaling,
#                  hyperparameter tuning and model evaluation
# Input         :  Datapath (str) - File path to the Wine CSV dataset
# Output        :  Prints step-by-step results and displays K vs Accuracy plot
# Author        :  Devesh Manoj Narkhede
# =========================================================
def WineClassifier(Datapath):
    Border = "-"*40

    #----------------------------------------------
    # Step 1 : Load the dataset from csv file
    #----------------------------------------------

    print(Border)
    print("Step 1 : Load the dataset from csv file")
    print(Border)

    df = pd.read_csv(Datapath)
    print(Border)
    print("Some Entries from Dataset")
    print(df.head())
    print(Border)

    #----------------------------------------------
    # Step 2 : Clean the Dataset by removing empty rows
    #----------------------------------------------

    print(Border)
    print("Step 2 : Clean the Dataset by removing empty rows")
    print(Border)

    df.dropna(inplace=True)
    print("Total Records : ",df.shape[0])
    print("Total Columns : ",df.shape[1])
    print(Border)

    #----------------------------------------------
    # Step 3 : Separate Independent And Dependent Variables
    #----------------------------------------------

    print(Border)
    print("Step 3 : Separate Independent And Dependent Variables")
    print(Border)

    X = df.drop(columns=['Class'])
    Y = df['Class']

    print("Shape of X : ",X.shape)
    print("Shape of Y : ",Y.shape)

    print(Border)
    print("Input Columns : ",X.columns.tolist())
    print("Output Columns : Class")

    #----------------------------------------------
    # Step 4 : Split the dataset for training and testing
    #----------------------------------------------

    print(Border)
    print("Step 4 : Split the dataset for training and testing")
    print(Border)

    X_train, X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42,stratify=Y)

    print(Border)
    print("Information of training and testing data")
    print("X_train Shape : ",X_train.shape)
    print("X_test Shape : ",X_test.shape)
    print("Y_train Shape : ",Y_train.shape)
    print("Y_test Shape : ",Y_test.shape)

    #----------------------------------------------
    # Step 5 : Feature Scaling
    #----------------------------------------------

    print(Border)
    print("Step 5 : Feature Scaling")
    print(Border)

    scaler = StandardScaler()

    # Independent Variables Scaling
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Feature Scaling is Done...")

    #----------------------------------------------
    # Step 6 : Explore multiple values of K
    # Hyper Parameter Tuning (K)
    #----------------------------------------------

    print(Border)
    print("Step 6 : Explore multiple values of K")
    print(Border)

    accuracy_scores = []
    K_values = range(1,21)

    for k in K_values:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train_scaled,Y_train)
        Y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(Y_test,Y_pred)
        accuracy_scores.append(accuracy)
    
    print(Border)
    print("Accuracy Report of K values of 1 to 20 :")
    for value in accuracy_scores:
        print(value)

    print(Border)

    #----------------------------------------------
    # Step 7 : Plot Graph of K vs Accuracy
    #----------------------------------------------

    print(Border)
    print("Step 7 : Plot Graph of K vs Accuracy")
    print(Border)

    plt.figure(figsize=(8,5))
    plt.plot(K_values,accuracy_scores,marker = 'o')
    plt.title("K Values Vs Accuracy")
    plt.xlabel("Value of K")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.xticks(list(K_values))
    plt.show()

    #----------------------------------------------
    # Step 8 : Find Best Value of K
    #----------------------------------------------

    print(Border)
    print("Step 8 : Find Best Value of K")
    print(Border)

    best_k = list(K_values)[accuracy_scores.index(max(accuracy_scores))]

    print("Best Value of K is : ",best_k)

    #----------------------------------------------
    # Step 9 : Build Final Model using best value of K
    #----------------------------------------------

    print(Border)
    print("Step 9 : Build Final Model using best value of K")
    print(Border)

    final_model = KNeighborsClassifier(n_neighbors=best_k)
    final_model.fit(X_train_scaled,Y_train)
    Y_Pred = final_model.predict(X_test_scaled)

    #----------------------------------------------
    # Step 10 : Calculate Final Accuracy
    #----------------------------------------------

    print(Border)
    print("Step 10 : Calculate Final Accuracy")
    print(Border)

    accuracy = accuracy_score(Y_test,Y_Pred)
    print("Accuracy of Model is : ",accuracy * 100)

    #----------------------------------------------
    # Step 11 : Display Confusion Matrix
    #----------------------------------------------

    print(Border)
    print("Step 11 : Display Confusion Matrix")
    print(Border)

    cm = confusion_matrix(Y_test,Y_Pred)
    print(cm)

    #----------------------------------------------
    # Step 12 : Display Classification Report
    #----------------------------------------------

    print(Border)
    print("Step 12 : Display Classification Report")
    print(Border)

    print(classification_report(Y_test,Y_Pred))

# =========================================================
# Function Name :  main
# Description   :  Entry point of the program
# Input         :  None
# Output        :  None
# Author        :  Devesh Manoj Narkhede
# =========================================================
def main():

    Border = "-"*40
    print(Border)
    print("Wine Classifier Using KNN")
    print(Border)

    WineClassifier("WinePredictor.csv")

if __name__ == "__main__":
    main()