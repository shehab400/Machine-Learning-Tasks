# Assignment 5: Customer Churn Prediction

## Overview
This assignment focuses on predicting customer churn in a telecommunications company. Customer churn (attrition) is when customers stop doing business with a company. Predicting churn allows companies to take proactive retention measures.

## Team Members
- **AbdelRahman Hesham Zakaria** (1210148)
- **Shehab Tarek ElHadary** (1210366)
- **Omar Walid Mohamed** (1210269)

## Files

### Notebooks
- **`Assignment 5.ipynb`** - Complete churn prediction analysis

### Datasets
- **`churn_dataset.csv`** - Telecom customer data
  - **Customer Information**:
    - `customerID` - Unique identifier
    - `gender` - Male/Female
    - `SeniorCitizen` - Whether customer is senior (0/1)
    - `Partner` - Has partner (Yes/No)
    - `Dependents` - Has dependents (Yes/No)
    - `tenure` - Months with company
  
  - **Service Details**:
    - `PhoneService` - Has phone service
    - `MultipleLines` - Multiple phone lines
    - `InternetService` - DSL/Fiber optic/No
    - `OnlineSecurity` - Online security service
    - `OnlineBackup` - Online backup service
    - `DeviceProtection` - Device protection plan
    - `TechSupport` - Tech support service
    - `StreamingTV` - TV streaming service
    - `StreamingMovies` - Movie streaming service
  
  - **Billing Information**:
    - `Contract` - Month-to-month/One year/Two year
    - `PaperlessBilling` - Paperless billing (Yes/No)
    - `PaymentMethod` - Payment method type
    - `MonthlyCharges` - Monthly bill amount
    - `TotalCharges` - Total amount charged
  
  - **Target**:
    - `Churn` - Whether customer churned (Yes/No)

## Business Problem

### Why Predict Churn?
- **Cost**: Acquiring new customers is 5-25x more expensive than retaining existing ones
- **Revenue**: Churned customers represent lost revenue
- **Competitive advantage**: Proactive retention improves customer lifetime value

### Key Questions
1. What factors contribute most to customer churn?
2. Can we identify at-risk customers before they leave?
3. What retention strategies should be prioritized?

## Analysis Approach

### 1. Exploratory Data Analysis (EDA)
- **Churn rate analysis**: Calculate overall churn percentage
- **Demographic patterns**: Analyze churn by age, gender, family status
- **Service usage patterns**: Which services correlate with retention?
- **Contract analysis**: Impact of contract length on churn
- **Billing patterns**: Relationship between charges and churn

### 2. Data Preprocessing
```python
# Handle data issues
- Clean TotalCharges (convert to numeric, handle spaces)
- Handle missing values
- Remove customerID (not predictive)

# Feature engineering
- Create tenure groups (new, medium, long-term customers)
- Calculate average monthly charges
- Create service count features
- One-hot encode categorical variables

# Prepare for modeling
- Split train/test data
- Scale numerical features
- Handle class imbalance (if present)
```

### 3. Feature Analysis
Key churn indicators typically include:
- **Contract type**: Month-to-month customers churn more
- **Tenure**: New customers (< 6 months) at higher risk
- **Monthly charges**: Higher charges correlate with churn
- **Payment method**: Electronic check users churn more
- **Internet service**: Fiber optic users may have different patterns
- **Add-on services**: Lack of services like OnlineSecurity increases churn

### 4. Model Building
Likely models implemented:
- Logistic Regression (baseline)
- Decision Trees
- Random Forest
- Gradient Boosting (XGBoost/LightGBM)
- Neural Networks

### 5. Model Evaluation
- **Accuracy**: Overall correct predictions
- **Precision**: Of predicted churners, how many actually churned?
- **Recall**: Of actual churners, how many did we catch?
- **F1-Score**: Balance between precision and recall
- **AUC-ROC**: Model's ability to distinguish classes
- **Confusion Matrix**: Detailed prediction breakdown

## Key Metrics Focus

### Why Recall Matters
- **False Negatives are costly**: Missing a churner means losing revenue
- Better to contact non-churners (minor cost) than miss actual churners
- Typical target: Recall > 80%

### Precision-Recall Tradeoff
- High precision: Fewer false alarms, but miss some churners
- High recall: Catch most churners, but more false alarms
- Business decides based on retention campaign cost

## Expected Insights

### High-Risk Customer Profile
- Month-to-month contract
- Tenure < 6 months
- No add-on services
- Electronic check payment
- High monthly charges relative to services

### Retention Strategies
1. **Contract incentives**: Encourage annual contracts with discounts
2. **New customer onboarding**: Extra attention in first 6 months
3. **Service bundles**: Promote OnlineSecurity, TechSupport
4. **Payment simplification**: Move from electronic check
5. **Pricing optimization**: Review high-charge, low-service customers

## Libraries Used
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
```

## How to Run

1. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```

2. Ensure `churn_dataset.csv` is in the assignment directory

3. Launch notebook:
   ```bash
   jupyter notebook "Assignment 5.ipynb"
   ```

## Success Criteria

- **Model Performance**: AUC-ROC > 0.75
- **Business Value**: Identify top 20% at-risk customers
- **Actionable Insights**: Clear retention strategy recommendations
- **Interpretability**: Understand which features drive predictions

## Real-World Application

### Implementation Pipeline
1. **Score customers monthly**: Run model on active customer base
2. **Risk segmentation**: Low/Medium/High churn risk
3. **Targeted campaigns**: 
   - High risk: Personal outreach, retention offers
   - Medium risk: Automated emails with incentives
   - Low risk: Standard service communications
4. **Monitor results**: Track retention rate improvements
5. **Model retraining**: Update quarterly with new data

### Expected ROI
- 15-30% reduction in churn rate
- Millions in saved revenue
- Improved customer satisfaction scores
- Better resource allocation for retention teams

## Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| Class imbalance (more non-churners) | SMOTE, class weights, stratified sampling |
| TotalCharges has spaces/blanks | Convert to numeric, impute median |
| Categorical features | One-hot encoding, label encoding |
| Multicollinearity | Feature selection, PCA |
| Model interpretability | Feature importance plots, SHAP values |

## Extensions

- [ ] Deep learning models (Neural Networks)
- [ ] Time-series analysis (predict when churn will occur)
- [ ] Customer segmentation (cluster analysis)
- [ ] A/B testing retention strategies
- [ ] Survival analysis
- [ ] Real-time prediction API

---

**Business Impact**: Effective churn prediction can save millions in revenue and significantly improve customer lifetime value (CLV).
