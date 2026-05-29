# Wine Classification using K-Nearest Neighbors (KNN)

## Description
Classifying wine types based on chemical properties using K-Nearest Neighbors algorithm with hyperparameter tuning to find the best value of K.

## Machine Learning Type
| Field | Details |
|-------|---------|
| **Machine Learning Type** | Supervised Learning |
| **Problem Type** | Classification |
| **Algorithm Used** | K-Nearest Neighbors (KNN) |
| **Performance Metric** | Accuracy, Confusion Matrix, Classification Report |

## Dataset
- **File** : `WinePredictor.csv`
- **Records** : 178
- **Features** : Alcohol, Malic acid, Ash, Alcalinity of ash, Magnesium, Total phenols, Flavanoids, Nonflavanoid phenols, Proanthocyanins, Color intensity, Hue, OD280/OD315 of diluted wines, Proline
- **Target** : Class (1, 2, 3)

## Steps Covered
1. Load dataset from csv file
2. Clean the dataset by removing empty rows
3. Separate independent and dependent variables
4. Split dataset for training and testing
5. Feature scaling using StandardScaler
6. Explore multiple values of K (1 to 20)
7. Plot graph of K vs Accuracy
8. Find best value of K
9. Build final model using best value of K
10. Calculate final accuracy
11. Display confusion matrix
12. Display classification report

## Model Performance
| Metric | Value |
|--------|-------|
| Best K Value | 7 |
| Accuracy | 100% |

## Confusion Matrix
| | Predicted 1 | Predicted 2 | Predicted 3 |
|---|---|---|---|
| **Actual 1** | 12 | 0 | 0 |
| **Actual 2** | 0 | 14 | 0 |
| **Actual 3** | 0 | 0 | 10 |

## Classification Report
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 1 | 1.00 | 1.00 | 1.00 | 12 |
| 2 | 1.00 | 1.00 | 1.00 | 14 |
| 3 | 1.00 | 1.00 | 1.00 | 10 |
| **Accuracy** | | | **1.00** | **36** |

## Key Insight
- **Best K = 7** gave perfect classification on test data
- **100% accuracy** — model correctly classified all 36 test wine samples
- **Feature scaling** was critical for KNN to perform well

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
   python WineClassifier.py

## Author
**Devesh Manoj Narkhede**