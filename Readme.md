# End-to-End Machine Learning Project: Student Performance Indicator

This repository contains a highly detailed end-to-end machine learning workflow for predicting students' performance based on various demographic and academic features. The workflow covers all critical steps from problem definition to model deployment.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset Details](#dataset-details)
- [Project Workflow](#project-workflow)
  - [1. Data Collection](#1-data-collection)
  - [2. Exploratory Data Analysis (EDA)](#2-exploratory-data-analysis-eda)
  - [3. Data Preprocessing](#3-data-preprocessing)
  - [4. Feature Engineering](#4-feature-engineering)
  - [5. Model Building & Training](#5-model-building--training)
  - [6. Model Evaluation](#6-model-evaluation)
  - [7. Model Selection](#7-model-selection)
  - [8. Model Deployment Pipeline](#8-model-deployment-pipeline)
- [Results](#results)
- [File Structure](#file-structure)
- [How to Run](#how-to-run)
- [References](#references)

---

## Problem Statement

Predict the student's **math score** based on their demographic and academic information. The project investigates how features such as gender, ethnicity, parental education, lunch type, and test preparation course affect student performance.

---

## Dataset Details

- **Source:** [Kaggle: Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams?datasetId=74977)
- **Rows:** 1000
- **Columns:** 8
  - gender, race/ethnicity, parental level of education, lunch, test preparation course, math score, reading score, writing score

---

## Project Workflow

### 1. Data Collection

- **Download** the dataset from Kaggle.
- **Import** required libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, CatBoost, XGBoost.

### 2. Exploratory Data Analysis (EDA)

- **Variable exploration:** Understand distributions and relationships between variables.
- **Target variable:** `math_score`
- **Feature variables:** Categorical (gender, race_ethnicity, parental_level_of_education, lunch, test_preparation_course), Numerical (reading_score, writing_score).
- **Visualizations:** Histogram, boxplots, correlation matrix, pair plots.

### 3. Data Preprocessing

- **Handling missing values:** Ensure dataset has no missing values; otherwise, impute or drop as needed.
- **Categorical encoding:** Use appropriate encoders (e.g., OneHotEncoder or LabelEncoder) for categorical features.
- **Feature scaling:** Standardize/normalize numerical features.

### 4. Feature Engineering

- **Feature selection:** Identify important features for prediction.
- **Data splitting:** Split data into train and test sets (typically 80/20 split).

### 5. Model Building & Training

- **Algorithms used:**
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - Decision Tree Regressor
  - Random Forest Regressor
  - K-Nearest Neighbors Regressor
  - XGBoost Regressor
  - CatBoost Regressor
  - AdaBoost Regressor
- **Hyperparameter Tuning:** Use `RandomizedSearchCV` for parameter optimization (where applicable).
- **Pipeline:** Build a training pipeline to automate preprocessing and model training.

### 6. Model Evaluation

- **Metrics:** R² Score, Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE).
- **Evaluation function:** Custom evaluation function to show all metrics for each model on both training and testing sets.
- **Cross-validation:** Validate model generalizability.

### 7. Model Selection

- **Compare all models** based on R² score and error metrics.
- **Model leaderboard example (Test Set R² Score):**
  | Model Name              | R² Score (Test) |
  |------------------------ |:--------------:|
  | Ridge                   | 0.8806         |
  | Linear Regression       | 0.8803         |
  | CatBoosting Regressor   | 0.8516         |
  | AdaBoost Regressor      | 0.8498         |
  | Random Forest Regressor | 0.8473         |
  | Lasso                   | 0.8253         |
  | XGBRegressor            | 0.8216         |
  | K-Neighbors Regressor   | 0.7838         |
  | Decision Tree           | 0.7603         |

- **Best model:** Ridge/Linear Regression based on balanced performance and generalization.

### 8. Model Deployment Pipeline

- **Saving artifacts:** Store the trained model and preprocessor as `.pkl` files in the `artifacts/` directory.
- **Prediction pipeline:** `src/pipeline/predict_pipeline.py`
    - Loads model and preprocessor.
    - Handles missing values and formats categorical features.
    - Transforms input features and predicts math score.
    - Ensures robust handling for real-world prediction scenarios.

---

## Results

- **Best Model:** Ridge Regression / Linear Regression (Test R² ≈ 0.88)
- **Metrics:**
  - RMSE (Test): ~5.4
  - MAE (Test): ~4.2
- The model can explain ~88% of the variance in student math scores based on the input features.

---

## File Structure

```
.
├── notebook/
│   ├── 1 . EDA STUDENT PERFORMANCE .ipynb
│   └── 2. MODEL TRAINING.ipynb
├── src/
│   └── pipeline/
│       └── predict_pipeline.py
├── artifacts/
│   ├── model.pkl
│   └── preprocessor.pkl
├── Readme.md
```

---

## How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/Aryan235711/Machine-Learning-Project.git
   cd Machine-Learning-Project
   ```
2. **Install requirements**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run notebooks**
   - Explore EDA and training workflow in `notebook/` directory.
4. **Predict**
   - Use the prediction pipeline (`src/pipeline/predict_pipeline.py`) with new data.

---

## References

- Kaggle Dataset: [Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams?datasetId=74977)
- scikit-learn, xgboost, catboost official documentation

---

**Author:** Aryan235711
