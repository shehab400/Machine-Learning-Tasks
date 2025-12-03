# Assignment 4a: Loan Approval Prediction

## Overview
This assignment implements and compares classification algorithms for predicting loan approval decisions. The project demonstrates the application of Logistic Regression and Support Vector Machines (SVM) on a realistic loan dataset.

## Team Members
- **AbdelRahman Hesham Zakaria** (1210148)
- **Shehab Tarek Elhadary** (1210366)
- **Omar Walid Mohamed** (1210269)

## Files

### Notebooks
- **`task4a updated.ipynb`** - Complete implementation with model comparison

### Datasets
- **`loan_approval.csv`** - Loan application data
  - **Features**:
    - `name` - Applicant name (dropped during preprocessing)
    - `city` - City of residence (categorical)
    - `income` - Annual income (numerical)
    - `credit_score` - Credit rating (numerical)
    - `loan_amount` - Requested loan amount (numerical)
    - `years_employed` - Employment duration (numerical)
    - `points` - Internal scoring system (dropped due to perfect separability)
  - **Target**: `loan_approved` - Approval decision (True/False)

## Project Workflow

### 1. Data Loading & Exploration
```python
- Load dataset using pandas
- Inspect data types and structure
- Check for missing values
- Generate statistical summaries
```

### 2. Data Preprocessing
- **Drop irrelevant features**: Removed `name` column
- **Handle target variable**: Convert TRUE/FALSE strings to binary (1/0)
- **Feature selection**: Removed `points` column (perfectly separable, causing 100% accuracy)
- **Feature categorization**:
  - Categorical: `city`
  - Numerical: `income`, `credit_score`, `loan_amount`, `years_employed`

### 3. Data Splitting
- **Training set**: 80% of data
- **Test set**: 20% of data
- **Stratification**: Maintained class distribution using `stratify=y`

### 4. Feature Engineering Pipeline
```python
ColumnTransformer:
  - Numerical features: StandardScaler (mean=0, std=1)
  - Categorical features: OneHotEncoder (handles unknown categories)
```

## Models Implemented

### 1. Logistic Regression
**Concept**: Linear model for binary classification using sigmoid function

**Configuration**:
```python
LogisticRegression(max_iter=500)
```

**Characteristics**:
- Fast training and prediction
- Outputs probability scores
- Works well with linearly separable data
- Interpretable coefficients

### 2. Hard Margin SVM (Linear)
**Concept**: Finds optimal hyperplane with maximum margin, minimal tolerance for errors

**Configuration**:
```python
LinearSVC(C=1e10, max_iter=5000)
```

**Parameters**:
- `C=1e10` - Very large regularization parameter (almost no tolerance for misclassification)
- Creates hard decision boundaries
- Sensitive to outliers
- Seeks perfect separation

### 3. Soft Margin SVM with RBF Kernel
**Concept**: Allows some misclassification for better generalization using non-linear kernel

**Configuration**:
```python
SVC(kernel='rbf', C=1.0)
```

**Parameters**:
- `kernel='rbf'` - Radial Basis Function for non-linear boundaries
- `C=1.0` - Balanced regularization (default)
- More flexible decision boundaries
- Better generalization to unseen data

## Model Evaluation

### Metrics Used
1. **Classification Report**:
   - Precision: TP / (TP + FP)
   - Recall: TP / (TP + FN)
   - F1-Score: Harmonic mean of precision and recall
   - Support: Number of samples per class

2. **Confusion Matrix**: Visual representation of predictions vs actual
3. **Accuracy Score**: Overall correct predictions / total predictions

### Visualization
- Heatmap confusion matrices for all three models
- Side-by-side comparison using seaborn
- Color-coded: Blues (Logistic), Greens (Hard SVM), Oranges (Soft SVM)

## Key Findings

### Perfect Separability Issue
Initially, models achieved 100% accuracy due to the `points` feature being perfectly correlated with the target. **Pairplot analysis** revealed:
- `points` and `credit_score` perfectly separate approved/rejected loans
- This is unrealistic in production scenarios
- Solution: Dropped `points` column for realistic model comparison

### Model Comparison
After removing `points`:
```
Model                              | Accuracy
----------------------------------|----------
Logistic Regression               | ~XX%
Hard Margin SVM (C=1e10)          | ~XX%
Soft Margin SVM RBF (C=1.0)       | ~XX%
```
*(Values depend on data split)*

## Libraries Used
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
```

## How to Run

1. Install required packages:
   ```bash
   pip install pandas numpy scikit-learn seaborn matplotlib
   ```

2. Ensure `loan_approval.csv` is in the same directory

3. Run the notebook:
   ```bash
   jupyter notebook "task4a updated.ipynb"
   ```

## Key Learnings

### Technical
- **Feature selection matters**: Perfectly correlated features create unrealistic models
- **Kernel choice in SVM**: RBF kernel handles non-linear patterns better
- **Regularization (C parameter)**: Balance between margin maximization and error tolerance
- **Pipeline usage**: Ensures consistent preprocessing in train/test

### Practical
- Real-world loan approval involves complex, non-linear relationships
- Credit score and income are strong predictors
- Model interpretability vs performance tradeoff
- Importance of data exploration before modeling

## Challenges Encountered

1. **Perfect Separability**: Initial 100% accuracy revealed data leakage
2. **Solution**: Thorough EDA using pairplots to identify issue
3. **Preprocessing**: Handling mixed numerical/categorical features efficiently
4. **Class Imbalance**: Potential imbalance between approved/rejected loans

## Future Improvements

- [ ] Feature engineering (income-to-loan ratio, debt-to-income)
- [ ] Hyperparameter tuning (GridSearchCV, RandomizedSearchCV)
- [ ] Ensemble methods (Random Forest, Gradient Boosting)
- [ ] Cross-validation for robust evaluation
- [ ] ROC-AUC curve analysis
- [ ] Feature importance analysis
- [ ] Handle class imbalance if present (SMOTE, class weights)

---

**Business Impact**: Accurate loan approval prediction helps financial institutions minimize default risk while maximizing lending opportunities.
