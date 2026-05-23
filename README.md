# Gene-Expression-using-knn

A machine learning project that predicts cancer presence based on gene expression levels using the K-Nearest Neighbors algorithm.

Dataset

- **File:** `gene_knn_dataset.csv`
- **Size:** 100 samples
- **Features:**
  - `Gene_One_Expression` — expression level of gene one
  - `Gene_Two_Expression` — expression level of gene two
- **Target:**
  - `Cancer_Present` — 0 (No Cancer) / 1 (Cancer)

---

**Exploratory Data Analysis**

- Scatterplot of Gene One vs Gene Two colored by cancer presence
- Pairplot to visualize feature relationships by class

---

Preprocessing

- Split data: **70% training / 30% testing** (`test_size=0.3, random_state=42`)
- Feature scaling using **StandardScaler** (zero mean, unit variance)

---

**Model 1 — Basic KNN (k=1)**

```python
KNeighborsClassifier(n_neighbors=1)
```

**Results**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0 (No Cancer) | 0.94 | 0.94 | 0.94 | 16 |
| 1 (Cancer) | 0.93 | 0.93 | 0.93 | 14 |
| **Overall Accuracy** | | | **0.93** | 30 |

- **Test Error Rate:** `0.067` (6.7%)

---

## Elbow Curve — Finding Optimal K

- Looped k from **1 to 29**
- Plotted **Error Rate vs Number of Neighbors**
- Helps identify the best k before overfitting

---

## Model 2 — Pipeline + GridSearchCV (Recommended)

Built a proper ML pipeline to prevent data leakage:

- **GridSearchCV** with `cv=5` cross-validation
- Searched k values from **1 to 19**
- Best k automatically selected

### Results

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0 (No Cancer) | 0.93 | 0.88 | 0.90 | 16 |
| 1 (Cancer) | 0.87 | 0.93 | 0.90 | 14 |
| **Overall Accuracy** | | | **0.90** | 30 |

> Accuracy is slightly lower than k=1 but more **reliable and generalizable** on unseen data.

---

## Predicting a New Patient

```python
new_patient = [[3.8, 6.4]]

full_cv_classifier.predict(new_patient)
# Output: [0]  →  No Cancer Detected

full_cv_classifier.predict_proba(new_patient)
# Output: [[0.833, 0.167]]  →  83.3% No Cancer, 16.7% Cancer
```

---

## Libraries Used

- `numpy`, `pandas` — data handling
- `matplotlib`, `seaborn` — visualization
- `scikit-learn` — model building, evaluation, and tuning
  - `KNeighborsClassifier`
  - `StandardScaler`
  - `train_test_split`
  - `GridSearchCV`
  - `Pipeline`
  - `confusion_matrix`, `classification_report`

---

## How to Run

1. Clone this repository
2. Upload `gene_knn_dataset.csv` to your environment
3. Open the notebook in **Google Colab** or **Jupyter Notebook**
4. Run all cells in order

---

## Key Takeaways

| Approach | Accuracy | Notes |
|----------|----------|-------|
| KNN k=1 (basic) | 93% | Prone to overfitting |
| KNN + GridSearchCV (pipeline) | 90% | More robust, production-ready |

> **Best practice:** Always scale inside the pipeline to avoid data leakage during cross-validation.
