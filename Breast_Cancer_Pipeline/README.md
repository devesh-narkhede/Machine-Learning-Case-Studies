# Breast Cancer Prediction using Random Forest & Pipeline

## Description
An end-to-end MLOps machine learning pipeline for predicting breast cancer diagnosis
(Benign vs Malignant) using Random Forest Classifier on the Breast Cancer Wisconsin dataset.
This project follows industrial best practices by automating preprocessing, training,
evaluation, and model saving and loading using Scikit-learn Pipeline and Joblib.

## Machine Learning Type
| Field | Details |
|-------|---------|
| **Machine Learning Type** | Supervised Learning |
| **Problem Type** | Classification |
| **Algorithm Used** | Random Forest Classifier |
| **Pipeline Steps** | SimpleImputer → RandomForestClassifier |
| **Performance Metric** | Accuracy, Confusion Matrix, Classification Report |

## MLOps Pipeline
| Stage | Tool | Description |
|-------|------|-------------|
| Data Preparation | Pandas | Load and clean dataset |
| Missing Value Handling | SimpleImputer | Replace missing values with median |
| Model Training | RandomForestClassifier | Train with 300 estimators |
| Model Evaluation | Scikit-learn Metrics | Accuracy, Confusion Matrix, Classification Report |
| Model Saving | Joblib | Save trained pipeline to disk |
| Model Loading | Joblib | Load model for future predictions |
| Feature Importance | Matplotlib | Visualize most important features |

## Dataset Information
| Field | Details |
|-------|---------|
| **File** | `breast-cancer-wisconsin.csv` |
| **Source** | UCI Machine Learning Repository |
| **Total Records** | 698 |
| **Training Records** | 488 (70%) |
| **Testing Records** | 210 (30%) |
| **Target** | CancerType (2 = Benign, 4 = Malignant) |

## Features (9)
| No | Feature |
|----|---------|
| 1 | ClumpThickness |
| 2 | UniformityCellSize |
| 3 | UniformityCellShape |
| 4 | MarginalAdhesion |
| 5 | SingleEpithelialCellSize |
| 6 | BareNuclei |
| 7 | BlandChromatin |
| 8 | NormalNucleoli |
| 9 | Mitoses |

## Steps Covered
1. Load CSV dataset
2. Display basic statistics
3. Prepare features and target
4. Handle missing values using SimpleImputer
5. Split dataset into 70% training and 30% testing
6. Build pipeline with SimpleImputer and RandomForestClassifier
7. Train the pipeline
8. Predictions on test data
9. Evaluate model accuracy and confusion matrix
10. Plot feature importances
11. Save model as `bc_rf_pipeline.joblib`
12. Load model and test sample prediction

## Model Performance
| Metric | Value |
|--------|-------|
| Train Accuracy | 100% |
| Test Accuracy | 97.14% |

## Confusion Matrix
| | Predicted Benign (2) | Predicted Malignant (4) |
|---|---|---|
| **Actual Benign (2)** | 135 | 2 |
| **Actual Malignant (4)** | 4 | 69 |

## Classification Report
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 2 (Benign) | 0.97 | 0.99 | 0.98 | 137 |
| 4 (Malignant) | 0.97 | 0.95 | 0.96 | 73 |
| **Accuracy** | | | **0.97** | **210** |

## Key Insight
- **Train Accuracy = 100%** — model perfectly learned training data
- **Test Accuracy = 97.14%** — model generalizes very well on unseen data
- Only **6 wrong predictions** out of 210 test records
- **BareNuclei** and **UniformityCellSize** are the most important features
- **Pipeline** ensures preprocessing and prediction are always consistent
- **Joblib** allows model reuse without retraining every time

## Saved Model
- Trained pipeline is saved as `bc_rf_pipeline.joblib` using Joblib
- Model is automatically loaded from `bc_rf_pipeline.joblib` for prediction
- No retraining needed — load and predict directly

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib

## How to Run
1. Clone the repository
2. Install dependencies
   pip install pandas numpy matplotlib scikit-learn joblib
3. Run the script
   python BreastCancerPrediction.py
4. Model will be saved automatically as `bc_rf_pipeline.joblib`

## Project Files
| File | Description |
|------|-------------|
| `BreastCancerPrediction.py` | Main code file |
| `breast-cancer-wisconsin.csv` | Dataset file |
| `bc_rf_pipeline.joblib` | Saved trained pipeline model |
| `README.md` | Project description |

## Author
**Devesh Manoj Narkhede**