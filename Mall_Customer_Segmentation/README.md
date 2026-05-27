# Mall Customer Segmentation using K-Means Clustering

## Description
Segmenting mall customers into groups based on Annual Income and Spending Score using K-Means Clustering algorithm with Elbow Method to find the optimal number of clusters.

## Machine Learning Type
| Field | Details |
|-------|---------|
| **Machine Learning Type** | Unsupervised Learning |
| **Problem Type** | Clustering |
| **Algorithm Used** | K-Means Clustering |
| **Optimal K Value** | 4 (found using Elbow Method) |
| **Performance Metric** | WCSS (Within Cluster Sum of Squares) |

## Dataset
- **File** : `Mall_Customers.csv`
- **Records** : 100
- **Features** : CustomerID, Age, AnnualIncome, SpendingScore
- **Selected Features** : AnnualIncome, SpendingScore
- **Target** : No target (Unsupervised Learning)

## Steps Covered
1. Load dataset
2. Select features (AnnualIncome and SpendingScore)
3. Scale the data using StandardScaler
4. Use Elbow Method to find optimal number of clusters (K=1 to 10)
5. Train the model with best K value (K=4)
6. Assign clusters to each customer

## Model Performance
| Metric | Value |
|--------|-------|
| Algorithm | K-Means Clustering |
| Best K Value | 4 |
| Total Customers | 100 |
| Total Clusters | 4 |

## Cluster Analysis
| Cluster | Type of Customer |
|---------|-----------------|
| 0 | High Income Low Spending |
| 1 | Low Income Low Spending |
| 2 | Low Income High Spending |
| 3 | High Income High Spending |

## Key Insight
- **Elbow Method** was used to find the optimal number of clusters
- **K = 4** was selected as the best value from Elbow Method
- **AnnualIncome and SpendingScore** were the most important features for segmentation
- Customers are segmented into 4 groups for targeted marketing strategy

## Technologies Used
- Python
- Pandas
- Matplotlib
- Scikit-learn

## How to Run
1. Clone the repository
2. Install dependencies
   pip install pandas matplotlib scikit-learn
3. Run the script
   python MallCustomerSegmentation.py

## Project Files
| File | Description |
|------|-------------|
| `MallCustomerSegmentation.py` | Main code file |
| `Mall_Customers.csv` | Dataset file |
| `README.md` | Project description |

## Author
**Devesh Manoj Narkhede**