# ========================================================
# BREAST CANCER PREDICTION CASE STUDY
# USING RANDOM FOREST CLASSIFIER
# ========================================================

# ========================================================
# Import Required Libraries
# ========================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_curve,
    auc,
    classification_report,
)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib

# ========================================================
# File Paths
# ========================================================
INPUT_PATH = "breast-cancer-wisconsin.data"
OUTPUT_PATH = "breast-cancer-wisconsin.csv"
MODEL_PATH = "bc_rf_pipeline.joblib"

# ========================================================
# Headers
# ========================================================
HEADERS = [
    "CodeNumber", "ClumpThickness", "UniformityCellSize", "UniformityCellShape", "MarginalAdhesion",
    "SingleEpithelialCellSize", "BareNuclei", "BlandChromatin", "NormalNucleoli", "Mitoses", "CancerType"
]

# =========================================================
# Function Name :  read_data
# Description   :  Reads the data into pandas dataframe
# Input         :  path (str) - Path of CSV file
# Output        :  data - Pandas DataFrame
# Author        :  Devesh Manoj Narkhede
# =========================================================
def read_data(path):
    data = pd.read_csv(path, header=None)
    return data

# =========================================================
# Function Name :  get_headers
# Description   :  Returns dataset headers
# Input         :  dataset - Pandas DataFrame
# Output        :  Returns the header values
# Author        :  Devesh Manoj Narkhede
# =========================================================
def get_headers(dataset):
    return dataset.columns.values

# =========================================================
# Function Name :  add_headers
# Description   :  Adds the headers to the dataset
# Input         :  dataset - Pandas DataFrame
#                  headers - List of header names
# Output        :  Updated dataset with headers
# Author        :  Devesh Manoj Narkhede
# =========================================================
def add_headers(dataset, headers):
    dataset.columns = headers
    return dataset

# =========================================================
# Function Name :  data_file_to_csv
# Description   :  Converts raw .data file to CSV with headers
# Input         :  None
# Output        :  Writes the data to CSV file
# Author        :  Devesh Manoj Narkhede
# =========================================================
def data_file_to_csv():
    dataset = read_data(INPUT_PATH)
    dataset = add_headers(dataset, HEADERS)
    dataset.to_csv(OUTPUT_PATH, index=False)
    print("File saved ...!")

# =========================================================
# Function Name :  handle_missing_values_with_imputer
# Description   :  Filters missing values from the dataset
#                  Converts ? to NaN and casts features to numeric
#                  Imputation happens inside the Pipeline
# Input         :  df - Dataset with missing values
#                  feature_headers - List of feature column names
# Output        :  Dataset after handling missing values
# Author        :  Devesh Manoj Narkhede
# =========================================================
def handle_missing_values_with_imputer(df, feature_headers):
    # Replace ? in the whole dataframe
    df = df.replace('?', np.nan)

    # Cast features to numeric
    df[feature_headers] = df[feature_headers].apply(pd.to_numeric, errors='coerce')

    return df

# =========================================================
# Function Name :  split_dataset
# Description   :  Splits the dataset into training and testing
# Input         :  dataset - Pandas DataFrame
#                  train_percentage - Train split ratio
#                  feature_headers - List of feature column names
#                  target_header - Target column name
#                  random_state - Random state value
# Output        :  train_x, test_x, train_y, test_y
# Author        :  Devesh Manoj Narkhede
# =========================================================
def split_dataset(dataset, train_percentage, feature_headers, target_header, random_state=42):
    train_x, test_x, train_y, test_y = train_test_split(
        dataset[feature_headers], dataset[target_header],
        train_size=train_percentage, random_state=random_state, stratify=dataset[target_header]
    )
    return train_x, test_x, train_y, test_y

# =========================================================
# Function Name :  dataset_statistics
# Description   :  Prints basic statistics of the dataset
# Input         :  dataset - Pandas DataFrame
# Output        :  None
# Author        :  Devesh Manoj Narkhede
# =========================================================
def dataset_statistics(dataset):
    print(dataset.describe(include='all'))

# =========================================================
# Function Name :  build_pipeline
# Description   :  Builds a Pipeline with SimpleImputer and
#                  RandomForestClassifier
#                  SimpleImputer replaces missing with median
#                  RandomForestClassifier is robust baseline
# Input         :  None
# Output        :  pipe - Pipeline object
# Author        :  Devesh Manoj Narkhede
# =========================================================
def build_pipeline():
    pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            class_weight=None
        ))
    ])
    return pipe

# =========================================================
# Function Name :  train_pipeline
# Description   :  Trains the Pipeline on training data
# Input         :  pipeline - Pipeline object
#                  X_train - Training features
#                  y_train - Training labels
# Output        :  pipeline - Trained Pipeline object
# Author        :  Devesh Manoj Narkhede
# =========================================================
def train_pipeline(pipeline, X_train, y_train):
    pipeline.fit(X_train, y_train)
    return pipeline

# =========================================================
# Function Name :  save_model
# Description   :  Saves the trained model to disk
# Input         :  model - Trained model object
#                  path (str) - Path to save the model
# Output        :  None
# Author        :  Devesh Manoj Narkhede
# =========================================================
def save_model(model, path=MODEL_PATH):
    joblib.dump(model, path)
    print(f"Model saved to {path}")

# =========================================================
# Function Name :  load_model
# Description   :  Loads the trained model from disk
# Input         :  path (str) - Path to load the model from
# Output        :  model - Loaded model object
# Author        :  Devesh Manoj Narkhede
# =========================================================
def load_model(path=MODEL_PATH):
    model = joblib.load(path)
    print(f"Model loaded from {path}")
    return model

# =========================================================
# Function Name :  plot_confusion_matrix_matshow
# Description   :  Displays the Confusion Matrix using matshow
# Input         :  y_true - Actual labels
#                  y_pred - Predicted labels
#                  title (str) - Title of the plot
# Output        :  None
# Author        :  Devesh Manoj Narkhede
# =========================================================
def plot_confusion_matrix_matshow(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    cax = ax.matshow(cm)
    fig.colorbar(cax)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha='center', va='center')
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

# =========================================================
# Function Name :  plot_feature_importances
# Description   :  Displays the feature importance of the model
# Input         :  model - Trained model object
#                  feature_names - List of feature names
#                  title (str) - Title of the plot
# Output        :  None
# Author        :  Devesh Manoj Narkhede
# =========================================================
def plot_feature_importances(model, feature_names, title="Feature Importances (Random Forest)"):
    if hasattr(model, "named_steps") and "rf" in model.named_steps:
        rf = model.named_steps["rf"]
        importances = rf.feature_importances_
    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        print("Feature importances not available for this model.")
        return

    idx = np.argsort(importances)[::-1]
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(importances)), importances[idx])
    plt.xticks(range(len(importances)), [feature_names[i] for i in idx], rotation=45, ha='right')
    plt.ylabel("Importance")
    plt.title(title)
    plt.tight_layout()
    plt.show()

# =========================================================
# Function Name :  main
# Description   :  Main function from where execution starts
# Input         :  None
# Output        :  None
# Author        :  Devesh Manoj Narkhede
# =========================================================
def main():
    Border = "-"*40

    #----------------------------------------------
    # Step 1 : Load CSV
    #----------------------------------------------
    print(Border)
    print("Step 1 : Load CSV")
    print(Border)

    dataset = pd.read_csv(OUTPUT_PATH)

    #----------------------------------------------
    # Step 2 : Display Basic Statistics
    #----------------------------------------------
    print(Border)
    print("Step 2 : Display Basic Statistics")
    print(Border)

    dataset_statistics(dataset)

    #----------------------------------------------
    # Step 3 : Prepare Features and Target
    #----------------------------------------------
    print(Border)
    print("Step 3 : Prepare Features and Target")
    print(Border)

    feature_headers = HEADERS[1:-1]
    target_header = HEADERS[-1]

    print("Feature Headers : ", feature_headers)
    print("Target Header : ", target_header)

    #----------------------------------------------
    # Step 4 : Handle Missing Values
    #----------------------------------------------
    print(Border)
    print("Step 4 : Handle Missing Values")
    print(Border)

    dataset = handle_missing_values_with_imputer(dataset, feature_headers)
    print("Missing Values Handled Successfully...")

    #----------------------------------------------
    # Step 5 : Split Dataset
    #----------------------------------------------
    print(Border)
    print("Step 5 : Split Dataset")
    print(Border)

    train_x, test_x, train_y, test_y = split_dataset(dataset, 0.7, feature_headers, target_header)

    print("Train_x Shape :: ", train_x.shape)
    print("Train_y Shape :: ", train_y.shape)
    print("Test_x Shape :: ", test_x.shape)
    print("Test_y Shape :: ", test_y.shape)

    #----------------------------------------------
    # Step 6 : Build and Train Pipeline
    #----------------------------------------------
    print(Border)
    print("Step 6 : Build and Train Pipeline")
    print(Border)

    pipeline = build_pipeline()
    trained_model = train_pipeline(pipeline, train_x, train_y)
    print("Trained Pipeline :: ", trained_model)

    #----------------------------------------------
    # Step 7 : Predictions
    #----------------------------------------------
    print(Border)
    print("Step 7 : Predictions")
    print(Border)

    predictions = trained_model.predict(test_x)
    print("Predictions Done Successfully...")

    #----------------------------------------------
    # Step 8 : Evaluate Model
    #----------------------------------------------
    print(Border)
    print("Step 8 : Evaluate Model")
    print(Border)

    print("Train Accuracy :: ", accuracy_score(train_y, trained_model.predict(train_x)))
    print("Test Accuracy  :: ", accuracy_score(test_y, predictions))
    print("Classification Report:\n", classification_report(test_y, predictions))
    print("Confusion Matrix:\n", confusion_matrix(test_y, predictions))

    #----------------------------------------------
    # Step 9 : Plot Feature Importances
    #----------------------------------------------
    print(Border)
    print("Step 9 : Plot Feature Importances")
    print(Border)

    plot_feature_importances(trained_model, feature_headers, title="Feature Importances (RF)")

    #----------------------------------------------
    # Step 10 : Save Model
    #----------------------------------------------
    print(Border)
    print("Step 10 : Save Model")
    print(Border)

    save_model(trained_model, MODEL_PATH)

    #----------------------------------------------
    # Step 11 : Load Model and Test Sample
    #----------------------------------------------
    print(Border)
    print("Step 11 : Load Model and Test Sample")
    print(Border)

    loaded = load_model(MODEL_PATH)
    sample = test_x.iloc[[0]]
    pred_loaded = loaded.predict(sample)
    print(f"Loaded model prediction for first test sample: {pred_loaded[0]}")

# ========================================================
# Application Starter
# ========================================================
if __name__ == "__main__":
    main()