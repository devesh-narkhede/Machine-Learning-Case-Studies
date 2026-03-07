# ========================================================
# IRIS FLOWER CLASSIFICATION USING DECISION TREE
# ========================================================

# ========================================================
# Import Required Libraries
# ========================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# Decorative Border for Output
Border = "-" * 40


# ========================================================
# Step 1 : Load the Dataset
# ========================================================

print(Border)
print("Step 1 : Load the dataset")
print(Border)

DatasetPath = "iris.csv"
df = pd.read_csv(DatasetPath)

print("Dataset gets loaded successfully...")
print("Initial entries from dataset :")
print(df.head())


# ========================================================
# Step 2 : Exploratory Data Analysis (EDA)
# ========================================================

print(Border)
print("Step 2 : Data analysis")
print(Border)

print("Shape of dataset : ", df.shape)

print("Column Names : ", list(df.columns))

print("Missing values (Per Column)")
print(df.isnull().sum())

print("Class Distribution (Species count)")
print(df["species"].value_counts())

print("Statistical Report of dataset")
print(df.describe())


# ========================================================
# Step 3 : Define Independent & Dependent Variables
# ========================================================

print(Border)
print("Step 3 : Decide Independent & Dependent variables")
print(Border)

# X : Independent variables / Features
# Y : Dependent variable / Target label

feature_cols = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)"
]

X = df[feature_cols]
Y = df["species"]

print("X shape : ", X.shape)
print("Y shape : ", Y.shape)


# ========================================================
# Step 4 : Data Visualization
# ========================================================

print(Border)
print("Step 4 : Visualization of dataset")
print(Border)

# Scatter plot for petal length vs petal width
plt.figure(figsize=(7, 5))

for sp in df["species"].unique():
    temp = df[df["species"] == sp]
    plt.scatter(
        temp["petal length (cm)"],
        temp["petal width (cm)"],
        label=sp
    )

plt.title("Iris : Petal Length vs Petal Width")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.legend()
plt.grid(True)
plt.show()


# ========================================================
# Step 5 : Split Dataset into Training and Testing Data
# ========================================================

print(Border)
print("Step 5 : Split the dataset for training and testing")
print(Border)

# Total dataset : 150 samples
# Test size = 50%
# Train size = 50%

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

print("Data splitting completed")

print("X - Independent : ", X.shape)
print("Y - Dependent : ", Y.shape)

print("X_train : ", X_train.shape)
print("X_test : ", X_test.shape)

print("Y_train : ", Y_train.shape)
print("Y_test : ", Y_test.shape)


# ========================================================
# Step 6 : Build the Decision Tree Model
# ========================================================

print(Border)
print("Step 6 : Build the model")
print(Border)

print("We are going to use DecisionTreeClassifier")

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    random_state=42
)

print("Model successfully created : ", model)


# ========================================================
# Step 7 : Train the Model
# ========================================================

print(Border)
print("Step 7 : Train the model")
print(Border)

model.fit(X_train, Y_train)

print("Model training completed")


# ========================================================
# Step 8 : Evaluate the model
# ========================================================

print(Border)
print("Step 8 : Evaluate the model")
print(Border)

Y_pred = model.predict(X_test)

print("Model testing completed")

print("Expected answers : ")
print(Y_test)

print("Predicted answers : ")
print(Y_pred)


# ========================================================
# Step 9 : Evaluate the model performance
# ========================================================

print(Border)
print("Step 9 : Evaluate the model performance")
print(Border)

# Accuracy
accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy of model is : ", accuracy * 100)

# Confusion Matrix
cm = confusion_matrix(Y_test, Y_pred)
print("Confusion Matrix : ")
print(cm)

# Classification Report
print("Classification Report : ")
print(classification_report(Y_test, Y_pred))


# ========================================================
# Step 10 : Plot Confusion Matrix
# ========================================================

print(Border)
print("Step 10 : Plot confusion matrix")
print(Border)

data = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

data.plot()
plt.title("Confusion Matrix of Iris Dataset")
plt.show()