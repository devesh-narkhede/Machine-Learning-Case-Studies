# Titanic Survival Prediction using Logistic Regression

## Description
Predicting whether a passenger survived the Titanic disaster based on features like Age, Fare, Sex, Pclass and Embarked using Logistic Regression.

## Machine Learning Type
| Field | Details |
|-------|---------|
| **Machine Learning Type** | Supervised Learning |
| **Problem Type** | Classification |
| **Algorithm Used** | Logistic Regression |
| **Performance Metric** | Accuracy, Confusion Matrix |

## Dataset
- **File** : `TitanicDataset.csv`
- **Records** : 1309
- **Features** : Age, Fare, Sex, Sibsp, Parch, Pclass, Embarked
- **Target** : Survived (0 = Not Survived, 1 = Survived)

## Steps Covered
1. Load dataset
2. Show basic dataset information
3. Remove unwanted columns (Passengerid, zero)
4. Handle missing values (Age, Fare, Embarked, Sex)
5. Encode categorical columns (One Hot Encoding on Embarked)
6. Convert boolean columns to integer
7. Split into features and labels
8. Split dataset for training and testing
9. Create and train Logistic Regression model
10. Preserve trained model as `titanic.pkl` using Joblib
11. Load preserved model from `titanic.pkl`
12. Predict on test data
13. Evaluate accuracy and confusion matrix

## Model Performance
| Metric | Value |
|--------|-------|
| Accuracy | 76.71% |

## Confusion Matrix
| | Predicted Not Survived | Predicted Survived |
|---|---|---|
| **Actual Not Survived** | 174 | 46 |
| **Actual Survived** | 15 | 27 |

## Model Coefficients
| Feature | Coefficient |
|---------|-------------|
| Age | -0.0263 |
| Fare | -0.0000 |
| Sex | 1.8459 |
| Sibsp | -0.1845 |
| Parch | -0.0488 |
| Pclass | -0.7639 |
| Embarked_1.0 | 0.0398 |
| Embarked_2.0 | -0.1109 |
| Intercept | 0.7204 |

## Key Insight
- **Sex** has the highest coefficient (1.84) — gender was the strongest survival factor
- **Pclass** (-0.76) — lower class passengers had less chance of survival
- **Age** (-0.026) — older passengers had slightly less chance of survival

## Saved Model
- Trained model is saved as `titanic.pkl` using Joblib
- Model is automatically loaded from `titanic.pkl` for prediction

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib

## How to Run
1. Clone the repository
2. Install dependencies
   pip install pandas numpy scikit-learn joblib
3. Run the script
   python TitanicSurvivalPrediction.py
4. Model will be saved automatically as `titanic.pkl`

## Project Files
| File | Description |
|------|-------------|
| `TitanicSurvivalPrediction.py` | Main code file |
| `TitanicDataset.csv` | Dataset file |
| `titanic.pkl` | Saved trained model |
| `README.md` | Project description |

## Author
**Devesh Manoj Narkhede**