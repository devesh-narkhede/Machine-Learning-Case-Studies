# ========================================================
# TITANIC SURVIVAL PREDICTION CASE STUDY
# USING LOGISTIC REGRESSION
# ========================================================

# ========================================================
# Import Required Libraries
# ========================================================
import pandas as pd
import numpy as np 
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# =========================================================
# Function Name :  LoadPreservedModel
# Description   :  Loads a previously saved model from disk
# Input         :  filename (str) - Path to the saved model file
# Output        :  loaded_model - Loaded ML model object
# Author        :  Devesh Manoj Narkhede
# =========================================================
def LoadPreservedModel(filename):
    loaded_model = joblib.load(filename)
    print("Model Successfully Loaded")

    return loaded_model

# =========================================================
# Function Name :  PreserveModel
# Description   :  Saves the trained model to disk
# Input         :  model - Trained ML model object
#                  filename (str) - Name to save the model file
# Output        :  None
# Author        :  Devesh Manoj Narkhede
# =========================================================
def PreserveModel(model,filename):
    joblib.dump(model,filename)

    print("Model Preserved Successfully with name : ",filename)

# =========================================================
# Function Name :  TrainTitanicModel
# Description   :  Splits data into features and labels,
#                  trains Logistic Regression model,
#                  evaluates accuracy and confusion matrix
# Input         :  df - Cleaned Pandas DataFrame object
# Output        :  None
# Author        :  Devesh Manoj Narkhede
# =========================================================
def TrainTitanicModel(df):
    # Split Features and labels
    X = df.drop("Survived",axis = 1)
    Y = df["Survived"]

    print("Features : ")
    print(X.head())

    print("\nLabels : ")
    print(Y.head())

    print("Shape of X : ", X.shape)
    print("Shape of Y : ", Y.shape)

    X_train, X_test, Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
    
    print("X_train Shape : ",X_train.shape)
    print("X_test Shape : ",X_test.shape)
    print("Y_train Shape : ",Y_train.shape)
    print("Y_test Shape : ",Y_test.shape)

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train,Y_train)

    print("\nModel Trained Successfully...")

    print("\nIntercept of model : ")
    print(model.intercept_)

    print("\nCoefficient of model : ")
    for feature,coefficient in zip(X.columns,model.coef_[0]):
        print(feature, " : ",coefficient)

    PreserveModel(model,"titanic.pkl")

    loaded_model = LoadPreservedModel("titanic.pkl")

    Y_pred = loaded_model.predict(X_test)

    accuracy = accuracy_score(Y_pred,Y_test)
    print("Accuracy is : ",accuracy*100)

    cm = confusion_matrix(Y_pred,Y_test)
    print("Confusion Matrix is : ",cm)

# =========================================================
# Function Name :  DisplayInfo
# Description   :  Displays a formatted section title
# Input         :  title (str) - Title text to display
# Output        :  None
# Author        :  Devesh Manoj Narkhede
# =========================================================
def DisplayInfo(title):
    print("\n" + "="*70)
    print(title)
    print("="*70)

# =========================================================
# Function Name :  ShowData
# Description   :  Shows basic information about the dataset
# Input         :  df - Pandas DataFrame object
#                  message (str) - Heading text to display
# Output        :  None
# Author        :  Devesh Manoj Narkhede
# =========================================================
def ShowData(df,message):
    DisplayInfo(message)
    print("First 5 Rows of dataset : ")
    print(df.head())

    print("\nShape of Dataset : ")
    print(df.shape)

    print("\nColumn Names : ")
    print(df.columns.tolist())

    print("\nMissing Values in Each Columns : ")
    print(df.isnull().sum())

# =========================================================
# Function Name :  CleanTitanicData
# Description   :  Performs data preprocessing -
#                  removes unnecessary columns,
#                  handles missing values,
#                  converts text data to numeric format,
#                  encodes categorical columns
# Input         :  df - Pandas DataFrame object
# Output        :  df - Cleaned Pandas DataFrame
# Author        :  Devesh Manoj Narkhede
# =========================================================
def CleanTitanicData(df):
    DisplayInfo("Step 2 : Original Data")
    print(df.head())

    # Remove unnecessary Columns
    drop_columns = ["Passengerid","zero","Name","Cabin"]
    existing_columns = [col for col in drop_columns if col in df.columns]

    print("\nColumns to be dropped : ")
    print(existing_columns)

    # Drop the unwanted columns
    df = df.drop(columns = existing_columns)

    DisplayInfo("Step 3 : Data after column removal")
    print(df.head())

    # Handle Age Column
    if "Age" in df.columns:
        print("Age column before filling missing values")
        print(df["Age"].head(10))

        # Invalid value gets converted as NaN
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

        age_median = df["Age"].median()

        # Replace Missing Values with median
        df["Age"] = df["Age"].fillna(age_median)

        print("\nAge column after preprocessing : ")
        print(df["Age"].head(10))

    # Handle Fare Column
    if "Fare" in df.columns:
        print("\nFare Column Before Preprocessing")
        print(df["Fare"].head(10))

        df["Fare"] = pd.to_numeric(df["Fare"], errors="coerce")

        fare_median = df["Fare"].median()

        print("\nMedian of Fare column is : ",fare_median)

        # Replace Missing Values with median
        df["Fare"] = df["Fare"].fillna(fare_median)

        print("\nFare column after preprocessing : ")
        print(df["Fare"].head(10))

    # Handle Embarked column
    if "Embarked" in df.columns:
        print("\nEmbarked Column Before Preprocessing")
        print(df["Embarked"].head(10))

        # Convert the data into String
        df["Embarked"] = df["Embarked"].astype(str).str.strip()

        # Remove Missing Values
        df["Embarked"] = df["Embarked"].replace(['nan','None',''],np.nan)

        # Get most frequent value
        embarked_mode = df["Embarked"].mode()[0]
        print("\nMode of Embarked Column : ",embarked_mode)

        df["Embarked"] = df["Embarked"].fillna(embarked_mode)

        print("\nEmbarked column after preprocessing : ")
        print(df["Embarked"].head(10))

    # Handle Sex Column
    if "Sex" in df.columns:
        print("\nSex Column Before Preprocessing")
        print(df["Sex"].head(10))

        df["Sex"] = pd.to_numeric(df["Sex"], errors="coerce")

        print("\nSex column after preprocessing : ")
        print(df["Sex"].head(10))

    DisplayInfo("Step 4 : Data After Preprocessing")
    print(df.head())

    print("\nMissing Values After Preprocessing")   
    print(df.isnull().sum())

    # Encode Embarked Column - get_dummies() (One Hot Encoding)
    df = pd.get_dummies(df,columns=["Embarked"],drop_first=True)

    print("\nData After Encoding")
    print(df.head())

    print("Shape of Dataset : ",df.shape)

    # Convert Boolean Columns into integer
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)

    print("\nData After Converting Boolean to Integer")
    print(df.head())

    return df

# =========================================================
# Function Name :  TitanicSurvivalPrediction
# Description   :  Main pipeline controller -
#                  loads dataset, shows raw data,
#                  preprocesses and trains the model
# Input         :  DataPath (str) - File path to the Titanic CSV dataset
# Output        :  None
# Author        :  Devesh Manoj Narkhede
# =========================================================
def TitanicSurvivalPrediction(DataPath):
    DisplayInfo("Step 1 : Loading the dataset")
    df = pd.read_csv(DataPath)

    ShowData(df,"Initial Dataset")

    df = CleanTitanicData(df)

    TrainTitanicModel(df)

# =========================================================
# Function Name :  main
# Description   :  Entry point of the program
# Input         :  None
# Output        :  None
# Author        :  Devesh Manoj Narkhede
# =========================================================
def main():
    TitanicSurvivalPrediction("TitanicDataset.csv")

if __name__ =="__main__":
    main()