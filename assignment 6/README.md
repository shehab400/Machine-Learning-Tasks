# Assignment 6: CART (Classification and Regression Trees)

## Overview
This assignment provides a comprehensive exploration of CART (Classification and Regression Trees), one of the fundamental machine learning algorithms. The report covers theoretical foundations, mathematical formulations, and practical implementations of decision tree algorithms.

## Team Members
- **AbdelRahman Hesham Zakaria** (1210148)
- **Shehab Tarek ElHadary** (1210366)
- **Omar Walid Mohamed** (1210269)

## Files

### Documentation
- **`CART ML Report.pdf`** - Comprehensive theoretical and practical report on CART

## Report Contents

### 1. Introduction
- Historical context: Breiman et al. (1984)
- Overview of CART as a versatile supervised learning algorithm
- Applications in both classification and regression tasks
- Importance of interpretability in machine learning

### 2. Core Concept: Binary Recursive Partitioning

#### Key Characteristics
- **Binary Splits**: Every node splits data into exactly two groups
- **Greedy Approach**: Locally optimal splits at each step
- **Recursive Process**: Continues until stopping criterion met
- **Tree Structure**: Flowchart-like decision-making process

#### How It Works
```
1. Start with entire dataset at root node
2. Evaluate all possible splits for each feature
3. Select split that maximizes information gain
4. Create two child nodes
5. Recursively repeat for each child
6. Stop when criteria met (max depth, min samples, purity)
```

### 3. Splitting Criteria

#### For Classification: Minimizing Impurity

**Gini Impurity (Default)**:
```
Gini = 1 - Σ(pi²)
```
- Where `pi` = proportion of class i in node
- Range: [0, 0.5] for binary classification
- 0 = pure node (all one class)
- 0.5 = maximum impurity (50/50 split)

**Entropy**:
```
Entropy = -Σ(pi × log2(pi))
```
- Measures uncertainty/disorder
- 0 = pure node
- 1 = maximum entropy (binary equal split)
- More computationally expensive than Gini

**Information Gain**:
```
Gain = Parent_Impurity - Weighted_Average(Children_Impurity)
```
- Quantifies improvement from split
- Higher gain = better split
- Basis for split selection

#### For Regression: Minimizing Variance

**Mean Squared Error (MSE)**:
```
MSE = (1/n) × Σ(yi - ŷ)²
```
- Goal: Create nodes with values close to their mean
- Minimizes prediction error
- Each leaf predicts mean of target values

### 4. Building a Tree: Step-by-Step Example

**Sample Process**:
1. Calculate root impurity (baseline)
2. For each feature:
   - For each possible split point:
     - Calculate information gain
     - Track best split
3. Execute best split
4. Repeat recursively for children
5. Stop when:
   - Maximum depth reached
   - Minimum samples in node
   - No further improvement
   - Node is pure

### 5. Controlling Complexity: Preventing Overfitting

#### Pre-Pruning (Early Stopping)

**Parameters**:
- `max_depth`: Limit tree depth (e.g., 5-10)
- `min_samples_split`: Minimum samples to split (e.g., 20)
- `min_samples_leaf`: Minimum samples in leaf (e.g., 10)
- `max_leaf_nodes`: Limit total leaves
- `min_impurity_decrease`: Minimum improvement required

**Benefits**:
- Faster training
- Prevents overly complex trees
- Reduces computational cost

#### Post-Pruning (Cost-Complexity Pruning)

**Process**:
1. Grow full tree
2. Calculate cost-complexity for each subtree
3. Remove nodes that don't justify complexity
4. Use cross-validation to select best α (complexity parameter)

**Formula**:
```
Cost_Complexity = Error + α × (Number_of_Leaves)
```

**Benefits**:
- Often achieves better performance than pre-pruning
- More principled approach
- Automatic complexity selection

### 6. Strengths and Limitations

#### Advantages ✓
- **Interpretable**: Easy to visualize and explain
- **No scaling required**: Handles features at different scales
- **Mixed data types**: Handles numerical and categorical
- **Non-parametric**: No assumptions about data distribution
- **Feature interactions**: Automatically captures interactions
- **Missing values**: Can handle with surrogate splits
- **Fast prediction**: O(log n) for balanced trees

#### Disadvantages ✗
- **Overfitting**: Easily creates overly complex trees
- **Instability**: Small data changes → different trees
- **Bias**: Biased toward features with more levels
- **Greedy algorithm**: May miss globally optimal tree
- **Weak for extrapolation**: Can't predict outside training range
- **Step-wise boundaries**: Only axis-aligned splits

### 7. Practical Implementation

#### Scikit-learn Example
```python
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# Classification
clf = DecisionTreeClassifier(
    criterion='gini',      # or 'entropy'
    max_depth=5,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42
)

# Regression
reg = DecisionTreeRegressor(
    criterion='squared_error',  # or 'absolute_error'
    max_depth=5,
    min_samples_split=20
)
```

#### Visualization
```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20,10))
plot_tree(clf, filled=True, feature_names=feature_names, 
          class_names=class_names, rounded=True)
plt.show()
```

### 8. Advanced Topics

#### Feature Importance
```python
importances = clf.feature_importances_
# Measures total reduction in impurity by each feature
```

#### Ensemble Methods (Built on CART)
- **Random Forest**: Multiple trees with randomness
- **Gradient Boosting**: Sequential trees correcting errors
- **AdaBoost**: Weighted combination of trees
- **XGBoost/LightGBM**: Optimized gradient boosting

## Key Formulas Summary

| Metric | Formula | Use Case |
|--------|---------|----------|
| Gini Impurity | 1 - Σ(pi²) | Classification splits |
| Entropy | -Σ(pi × log₂(pi)) | Classification splits |
| Information Gain | Parent_Impurity - Weighted_Children_Impurity | Split selection |
| MSE | (1/n) × Σ(yi - ŷ)² | Regression splits |
| Cost-Complexity | Error + α × Leaves | Pruning |

## Real-World Applications

### 1. Medical Diagnosis
- Interpretable rules for disease prediction
- Clinicians can understand decision logic
- Example: "If blood pressure > 140 AND age > 50 THEN high risk"

### 2. Credit Scoring
- Transparent loan approval decisions
- Regulatory compliance (explainability)
- Fair lending practices verification

### 3. Customer Segmentation
- Clear rules for customer categories
- Actionable marketing strategies
- Easy communication to business stakeholders

### 4. Fraud Detection
- Rule-based anomaly identification
- Fast real-time predictions
- Interpretable fraud patterns

## Comparison with Other Algorithms

| Feature | CART | Logistic Regression | SVM | Neural Networks |
|---------|------|-------------------|-----|-----------------|
| Interpretability | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐ |
| Non-linear | ⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Training Speed | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| Prediction Speed | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Overfitting Risk | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

## Conclusion

CART remains a cornerstone algorithm in machine learning due to its:
- **Simplicity**: Easy to understand and implement
- **Interpretability**: Visual and rule-based decisions
- **Versatility**: Works for classification and regression
- **Foundation**: Basis for powerful ensemble methods

While individual trees may not achieve state-of-the-art performance, they excel in:
- Exploratory analysis
- Feature selection
- Building blocks for ensembles
- Situations requiring interpretability

## References

1. Breiman, L., Friedman, J., Stone, C. J., & Olshen, R. A. (1984). *Classification and Regression Trees*
2. Scikit-learn Documentation: Decision Trees
3. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*

## Further Reading

- [ ] ID3 and C4.5 algorithms (predecessors to CART)
- [ ] Random Forests and feature importance
- [ ] Gradient Boosting Decision Trees
- [ ] Explainable AI with SHAP values for tree models
- [ ] Optimal decision trees (recent research)

---

**Academic Value**: Understanding CART is essential for mastering modern ensemble methods and interpretable machine learning.
