''' Breast Cancer modeling  Project '''

# Overview
This project analyzes the Breast Cancer dataset using machine learning techniques. It includes data preparation, feature selection, model tuning, and a Streamlit app for user interaction.

# Project Structure
The project consists of the following files:
1. breast_data.py: Loads and preprocesses the dataset.
2. `feature_selection.py`: Performs feature selection using `SelectKBest`.
3. `grid_search.py`: Performs Grid Search CV for hyperparameter tuning.
4. `ann_model.py`: Trains and evaluates an Artificial Neural Network (ANN) model.
5. `app.py`: A Streamlit app for user interaction and prediction.
6. `README.md`: This file, documenting the project.

---

## Dataset
The dataset used in this project is the ##Breast Cancer Wisconsin (Diagnostic) Dataset## from `sklearn.datasets`. It contains
 1. Features.
 2. Target: Binary classification (0 = Malignant, 1 = Benign)

## How to Run the Project

1. Install Python 3.x.
2. Install the required libraries:
   ```bash
   pip install numpy pandas scikit-learn streamlit
