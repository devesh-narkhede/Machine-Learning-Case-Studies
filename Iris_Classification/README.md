# Iris Flower Classification Using Decision Tree

## Project Overview
I implemented a Machine Learning classification model using the Decision Tree algorithm to classify iris flowers into different species based on their measurements.

The model predicts the species using features such as sepal length, sepal width, petal length, and petal width.

---

## Machine Learning Type
- Machine Learning Type: Supervised Learning
- Problem Type: Classification
- Algorithm Used: Decision Tree Classifier

---

## Dataset Details
I used the Iris dataset, which contains:

- Total Records: 150
- Input Features: 4  
  - Sepal Length  
  - Sepal Width  
  - Petal Length  
  - Petal Width  

- Target Classes:
  - Setosa
  - Versicolor
  - Virginica

---

## Tools & Technologies Used
- Python
- Pandas – for data loading and preprocessing
- Matplotlib – for plotting graphs
- Seaborn – for data visualization
- Scikit-learn – for building and training the machine learning model

---

## Implementation Steps
1. Loaded the dataset using Pandas.
2. Performed basic data analysis and visualization.
3. Separated the feature variables (X) and the target variable (y).
4. Split the dataset into 80% training data and 20% testing data.
5. Built the Decision Tree Classifier model using Scikit-learn.
6. Trained the model using the training dataset.
7. Tested the model using the testing dataset.

---

## Model Configuration
- Algorithm: Decision Tree Classifier
- Criterion: Gini Index
- Maximum Depth: 5
- Random State: 42

---

## Conclusion

The Decision Tree model successfully classifies iris flowers into Setosa, Versicolor, and Virginica based on their sepal and petal measurements.

## Author
**Devesh Manoj Narkhede**