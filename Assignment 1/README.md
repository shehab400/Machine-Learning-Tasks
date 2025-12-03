# Assignment 1: Data Preprocessing

## Overview
This assignment focuses on fundamental data preprocessing techniques essential for preparing raw data for machine learning models. Two comprehensive notebooks demonstrate various preprocessing methods on healthcare and insurance datasets.

## Team Members
- **AbdelRahman Hesham Zakaria** (1210148)
- **Shehab Tarek ElHadary** (1210366)
- **Omar Walid Mohamed** (1210269)

## Files

### Notebooks
- **`Preprocessing.ipynb`** - Initial preprocessing pipeline
- **`Preprocessing1.ipynb`** - Advanced preprocessing techniques

### Datasets
- **`healthcare-dataset-stroke-data.csv`** - Medical data for stroke prediction
  - Features: age, gender, hypertension, heart disease, BMI, glucose levels, etc.
  - Target: stroke occurrence (binary classification)
  
- **`medical_insurance.csv`** - Insurance charges dataset
  - Features: age, sex, BMI, children, smoker status, region
  - Target: insurance charges (regression)

## Key Concepts Covered

### 1. Data Exploration
- Loading and inspecting datasets
- Understanding data types and structure
- Statistical summaries (mean, median, std, quartiles)
- Identifying data distributions

### 2. Handling Missing Values
- Detecting null/missing values
- Imputation strategies:
  - Mean/median imputation for numerical features
  - Mode imputation for categorical features
  - Forward/backward fill
  - Dropping rows/columns with excessive missing data

### 3. Data Cleaning
- Removing duplicates
- Handling outliers using:
  - IQR (Interquartile Range) method
  - Z-score method
  - Visual inspection with box plots
- Fixing data type inconsistencies

### 4. Feature Encoding
- **Label Encoding** - Converting categorical variables to numerical (ordinal)
- **One-Hot Encoding** - Creating binary columns for categorical features
- **Binary Encoding** - For binary categorical variables (Yes/No, Male/Female)

### 5. Feature Scaling
- **Standardization (Z-score normalization)** - Mean=0, Std=1
- **Min-Max Scaling** - Scaling to range [0,1]
- **Robust Scaling** - Using median and IQR (robust to outliers)

### 6. Feature Engineering
- Creating new features from existing ones
- Binning continuous variables
- Extracting information from datetime features
- Polynomial features

### 7. Data Visualization
- Distribution plots (histograms, KDE)
- Correlation heatmaps
- Box plots for outlier detection
- Scatter plots for relationships
- Count plots for categorical data

## Libraries Used
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
```

## How to Run

1. Ensure all required libraries are installed:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

2. Place the CSV datasets in the same directory as the notebooks

3. Open the notebook:
   ```bash
   jupyter notebook Preprocessing.ipynb
   ```

4. Run cells sequentially (Shift + Enter)

## Key Takeaways

- **Data quality** is crucial - "Garbage in, garbage out"
- Different preprocessing techniques suit different data types and ML algorithms
- Always **explore before preprocessing** - understand your data first
- **Document your preprocessing steps** - reproducibility is key
- **Visualize at every step** - catch errors and understand transformations
- Consider the **downstream ML model** when choosing preprocessing methods

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Missing values in numerical columns | Use mean/median imputation |
| Missing values in categorical columns | Use mode imputation or create "Unknown" category |
| Outliers affecting model | Use robust scaling or remove extreme outliers |
| Categorical features | Apply one-hot encoding or label encoding |
| Different feature scales | Apply standardization or normalization |
| Imbalanced classes | Use SMOTE, class weights, or resampling |

## Next Steps

After preprocessing, the cleaned data is ready for:
- Exploratory Data Analysis (EDA)
- Feature selection
- Model training and evaluation
- Hyperparameter tuning

---

**Note**: The datasets contain sensitive health information and are used for educational purposes only.
