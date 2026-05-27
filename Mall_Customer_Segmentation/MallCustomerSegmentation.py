# ========================================================
# MALL CUSTOMER SEGMENTATION CASE STUDY
# USING K-MEANS CLUSTERING
# ========================================================

# ========================================================
# Import Required Libraries
# ========================================================
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# =========================================================
# Function Name :  main
# Description   :  Segments mall customers into clusters
#                  based on Annual Income and Spending Score
#                  using K-Means Clustering algorithm.
#                  Includes data loading, feature selection,
#                  scaling, elbow method and model training
# Input         :  None
# Output        :  Prints step-by-step results and displays
#                  Elbow Method plot
# Author        :  Devesh Manoj Narkhede
# =========================================================
def main():
    Border = "-"*40

    #------------------------------------------
    # Step 1 : Load the dataset
    #------------------------------------------

    print(Border)
    print("Step 1 : Load the dataset")
    print(Border)

    df = pd.read_csv("Mall_Customers.csv")

    print("First Few Records : ")
    print(df.head())

    print("Shape of Dataset : ")
    print(df.shape)

    print("Missing Values : ")
    print(df.isnull().sum())

    #------------------------------------------
    # Step 2 : Select Features (Independent)
    #------------------------------------------
    # CustomerID,Age,AnnualIncome,SpendingScore - Dataset Header

    print(Border)
    print("Step 2 : Select Features (Independent)")
    print(Border)

    X = df[["AnnualIncome","SpendingScore"]]

    print("Selected Features : ")
    print(X.head())

    print("Shape of selected features : ")
    print(X.shape)

    #------------------------------------------
    # Step 3 : Scale the data
    #------------------------------------------

    print(Border)
    print("Step 3 : Scale the data")
    print(Border)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Data After Scaling : ")
    print(X_scaled[:5])

    #------------------------------------------
    # Step 4 : Use Elbow Method
    #------------------------------------------

    print(Border)
    print("Step 4 : Use Elbow Method")
    print(Border)

    WCSS = []

    for i in range(1,11): 
        model = KMeans(n_clusters=i, random_state=42, n_init=10) #n_init=10 loop iterate 10 times atleast
        model.fit(X_scaled)
        WCSS.append(model.inertia_)

    plt.figure(figsize=(8,5))
    plt.plot(range(1,11),WCSS, marker = 'o')
    plt.xlabel("Number of Clusters")
    plt.ylabel("WCSS")
    plt.title("Elbow Method")
    plt.grid(True)
    plt.show()

    #------------------------------------------
    # Step 5 : Train the model
    #------------------------------------------

    print(Border)
    print("Step 5 : Train the model")
    print(Border)

    model = KMeans(n_clusters=4,random_state=42,n_init=10)
    clusters = model.fit_predict(X_scaled)

    df["clusters"] = clusters

    print("Dataset with Cluster")
    print(df.head(30))

if __name__ == "__main__":
    main()