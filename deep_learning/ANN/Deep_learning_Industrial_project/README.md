# Industrial Projects

**Marvellous Infosystems — OJT 2026**
**Trainee : Shubhada A. Palwe**

---

## About This Folder

This folder has a real industrial-style ML project. Instead of small toy datasets, we work with the actual UCI Wisconsin Breast Cancer dataset and build a proper production-quality pipeline — with missing value handling, a scikit-learn Pipeline, model saving, and full evaluation metrics.

File 47: Breast Cancer classification using Random Forest inside a sklearn Pipeline.

---

## File 47 — Breast Cancer Classification

This project predicts whether a tumour is **Benign (2)** or **Malignant (4)** based on 9 cell features from biopsy samples.

### Dataset — Wisconsin Breast Cancer

The dataset comes from the UCI Machine Learning Repository. It has 699 samples and 11 columns:

```
CodeNumber, ClumpThickness, UniformityCellSize, UniformityCellShape,
MarginalAdhesion, SingleEpithelialCellSize, BareNuclei,
BlandChromatin, NormalNucleoli, Mitoses, CancerType
```

- **CancerType** is the label: `2` = Benign, `4` = Malignant
- **CodeNumber** is just an ID — we drop it from features
- **BareNuclei** has some missing values stored as `"?"` — we handle those

---

### Step 1 — Read the Raw Data File

```python
HEADERS = ["CodeNumber", "ClumpThickness", ..., "CancerType"]

def data_file_to_csv():
    dataset = pd.read_csv("breast-cancer-wisconsin.data", header=None)
    dataset.columns = HEADERS
    dataset.to_csv("breast-cancer-wisconsin.csv", index=False)
```

The original `.data` file has no headers. We assign column names manually and save it as a proper CSV for easy reuse.

---

### Step 2 — Handle Missing Values

```python
def handle_missing_values_with_imputer(df, feature_headers):
    df = df.replace('?', np.nan)
    df[feature_headers] = df[feature_headers].apply(pd.to_numeric, errors='coerce')
    return df
```

The `BareNuclei` column has `"?"` where data is missing. We replace those with `np.nan` first, then convert all feature columns to numeric. The actual imputation (filling missing values with the median) happens inside the Pipeline.

Note: `errors='coerce'` means any value that can't be converted to a number becomes `NaN` instead of raising an error.

---

### Step 3 — Train/Test Split

```python
train_x, test_x, train_y, test_y = train_test_split(
    dataset[feature_headers], dataset[target_header],
    train_size=0.7, random_state=42, stratify=dataset[target_header]
)
```

70% training, 30% testing. `stratify=dataset[target_header]` ensures both splits have the same proportion of Benign and Malignant cases. Without stratify, a random split might accidentally put most malignant cases in training only.

---

### Step 4 — sklearn Pipeline

```python
pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("rf", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1))
])
```

A Pipeline chains preprocessing and model together. When we call `pipeline.fit(X_train, y_train)`:
1. `SimpleImputer` fills any remaining `NaN` values with the median of each column
2. `RandomForestClassifier` trains on the cleaned data

The benefit: when we later call `pipeline.predict(X_test)`, the same imputation automatically applies to test data too. No risk of accidentally forgetting to impute.

**Why Random Forest?**
- Works well with medical data (non-linear relationships between features)
- `n_estimators=300` means 300 decision trees are trained; they vote on the final prediction
- `n_jobs=-1` uses all available CPU cores — faster training

---

### Step 5 — Evaluation

```python
print("Train Accuracy ::", accuracy_score(train_y, trained_model.predict(train_x)))
print("Test Accuracy  ::", accuracy_score(test_y, predictions))
print("Classification Report:\n", classification_report(test_y, predictions))
print("Confusion Matrix:\n", confusion_matrix(test_y, predictions))
```

**Accuracy:** percentage of correct predictions overall.

**Classification Report:** for each class (2 and 4), shows:
- Precision: of all we predicted as class X, how many actually were class X
- Recall: of all actual class X, how many did we correctly predict
- F1-score: balance between precision and recall

For medical data, **recall for Malignant (4)** is the most important. Missing a malignant tumour (False Negative) is far more dangerous than a false alarm.

**Confusion Matrix:**
```
                Predicted Benign | Predicted Malignant
Actual Benign       TN           |        FP
Actual Malignant    FN           |        TP
```

---

### Step 6 — Feature Importance

```python
def plot_feature_importances(model, feature_names, title):
    rf = model.named_steps["rf"]
    importances = rf.feature_importances_
    idx = np.argsort(importances)[::-1]
    plt.bar(range(len(importances)), importances[idx])
```

Random Forest tells us which features matter most. `feature_importances_` is an array where each value shows how much that feature contributed to the predictions.

`model.named_steps["rf"]` accesses the Random Forest step inside the Pipeline by its name.

---

### Step 7 — Save and Load Model

```python
joblib.dump(trained_model, "bc_rf_pipeline.joblib")   # save
loaded = joblib.load("bc_rf_pipeline.joblib")          # load
```

`joblib` saves the entire Pipeline (imputer + RF model) to a file. When loaded later, it works exactly the same — both the imputation and the prediction. This is how trained models are deployed to production.

---

### How to Run

```bash
pip install pandas numpy scikit-learn matplotlib joblib

# First, put the dataset file in the same folder:
# breast-cancer-wisconsin.data
# breast-cancer-wisconsin.csv (if already converted)

python 47_IndustrialBreastCancer.py
```

The dataset is available at the UCI Machine Learning Repository:
https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original

---

### Key Terms

| Term | Meaning |
|------|---------|
| Pipeline | Chains multiple steps (imputer + model) into one object |
| SimpleImputer | Fills missing values with mean/median/mode |
| stratify | Keeps class ratio same in train/test split |
| joblib | Fast way to save Python objects (especially sklearn models) |
| feature_importances_ | Which features mattered most to the Random Forest |
| n_jobs=-1 | Use all CPU cores for training |

---

*Marvellous Infosystems — Applying ML to real medical data with a production-ready pipeline.*
