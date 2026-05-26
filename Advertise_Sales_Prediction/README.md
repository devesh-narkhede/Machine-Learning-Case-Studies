# Advertising Sales Prediction using Multiple Linear Regression

## Description
Predicting product sales based on TV, Radio, and Newspaper advertising budgets using Multiple Linear Regression.

## Dataset
- **File** : `Advertising.csv`
- **Records** : 200
- **Features** : TV, Radio, Newspaper
- **Target** : Sales

## Steps Covered
1. Load dataset
2. Remove unwanted columns
3. Check missing values
4. Display statistical summary
5. Correlation between columns
6. Split into independent & dependent variables
7. Split dataset for training & testing
8. Create & train the model
9. Test the model
10. Evaluate the model
11. Calculate model coefficients
12. Compare actual and predicted values
13. Plot actual vs predicted

## Model Performance
| Metric | Value |
|--------|-------|
| MSE    | 3.174 |
| RMSE   | 1.781 |
| R² Score | 0.899 |

## Model Coefficients
| Feature   | Coefficient |
|-----------|-------------|
| TV        | 0.0447      |
| Radio     | 0.1892      |
| Newspaper | 0.0028      |
| Intercept | 2.9791      |

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## How to Run
1. Clone the repository
2. Install dependencies
   pip install pandas numpy matplotlib scikit-learn
3. Run the script
   python AdvertiseSalesPrediction.py

## Author
**Devesh Manoj Narkhede**