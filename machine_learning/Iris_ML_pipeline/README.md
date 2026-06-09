# On-the-Job Training (OJT) Report

---

| Field               | Details                                                  |
|---------------------|----------------------------------------------------------|
| **Trainee Name**    | Shubhada A. Palwe                                        |
| **Training Domain** | Machine Learning — Complete Supervised Classification    |
| **Case Study**      | Iris Flower Classification — End-to-End ML Pipeline      |
| **Dataset**         | iris.csv (150 samples, 4 features, 3 classes)            |
| **Model Used**      | Decision Tree Classifier (sklearn)                       |
| **Total Steps**     | 10 (across 10 incremental files)                         |
| **Tools Used**      | Python, pandas, matplotlib, seaborn, scikit-learn        |
|                                         

---

## Objective

To implement the **complete Machine Learning pipeline from scratch** — starting from loading a raw CSV file and ending with a trained, evaluated, and visualized classification model. Each step is coded incrementally so each file adds exactly one new concept to the previous one.

This is the **capstone case study** that brings together everything learned so far:
Pandas → DataFrames → Visualization → Train/Test Split → Model → Evaluation → Confusion Matrix.

---

## The Dataset: iris.csv

The Iris dataset contains **150 flower measurements** across 3 species. It is the "Hello World" of Machine Learning.

| Column | Type | Description |
|--------|------|-------------|
| sepal length (cm) | float | Length of the sepal |
| sepal width (cm) | float | Width of the sepal |
| petal length (cm) | float | Length of the petal |
| petal width (cm) | float | Width of the petal |
| species | string | **Target label** — setosa / versicolor / virginica |

**Total rows:** 150 | **Features (X):** 4 | **Labels (Y):** 3 classes | **Missing values:** None

---

## The Full ML Pipeline — Overview

```
Step 1  → Load Dataset (pd.read_csv)
Step 2  → Data Analysis / EDA (shape, nulls, describe, value_counts)
Step 3  → Define X and Y (features vs label)
Step 4  → Visualize Data (scatter plot colored by class)
Step 5  → Train / Test Split (80% train, 20% test)
Step 6  → Build Model (DecisionTreeClassifier with hyperparameters)
Step 7  → Train Model (.fit)
Step 8  → Make Predictions (.predict)
Step 9  → Evaluate Performance (accuracy, confusion matrix, report)
Step 10 → Visualize Confusion Matrix (ConfusionMatrixDisplay)
```

---

## Step-by-Step Breakdown

---

### Step 1 — `CaseStudyStep1.py`
**Load the Dataset**

The very first step in any ML project: get the data into Python as a DataFrame.

```python
DataSetPath = "iris.csv"
df = pd.read_csv(DataSetPath)
print(df.head())
```

**What `df.head()` shows:**

```
   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm) species
0                5.1               3.5                1.4               0.2  setosa
1                4.9               3.0                1.4               0.2  setosa
...
```

**What I learned:**
- `pd.read_csv()` reads any CSV file and returns a DataFrame automatically.
- `.head()` shows the first 5 rows — a quick sanity check that data loaded correctly.
- All imports are gathered at the top: pandas, matplotlib, seaborn, sklearn. This is professional Python style.

**Key concept:** Real ML projects use CSV files, not manually typed lists. `pd.read_csv()` is the most common way to load data.

---

### Step 2 — `CaseStudyStep2.py`
**Data Analysis / Exploratory Data Analysis (EDA)**

Before doing anything else, we must understand our data deeply. This step uses 4 key pandas tools:

```python
print("Shape of dataset:", df.shape)           # (150, 5)
print("Column Names:", list(df.columns))        # ['sepal length...', ..., 'species']
print(df.isnull().sum())                        # check for missing values
print(df["species"].value_counts())             # count per class
print(df.describe())                            # statistical summary
```

**What each line tells us:**

| Command | Output | What It Means |
|---------|--------|---------------|
| `df.shape` | `(150, 5)` | 150 rows, 5 columns |
| `list(df.columns)` | `['sepal length...', 'species']` | Column names as a list |
| `df.isnull().sum()` | `0` for all | No missing values — data is clean |
| `df["species"].value_counts()` | 50 each | All 3 classes are perfectly balanced |
| `df.describe()` | min, max, mean, std | Statistical profile of each feature |

**What I learned:**
- `df.shape` is a **property**, not a method — no brackets needed.
- `df.isnull().sum()` is the standard check for missing/null values. If any column shows a number > 0, data cleaning is needed.
- `value_counts()` on the label column checks class balance. Balanced classes (50/50/50 here) are ideal for training.
- `df.describe()` gives count, mean, std, min, 25%, 50%, 75%, max for every numeric column in one shot.

**Key concept:** EDA (Exploratory Data Analysis) is not optional — it's how you understand what you're working with before building anything.

---

### Step 3 — `CaseStudyStep3.py`
**Define X (Features) and Y (Label)**

Now we separate the data into inputs (X) and the answer we want to predict (Y).

```python
feature_cols = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)"
]

X = df[feature_cols]    # shape: (150, 4) — 150 rows, 4 features
Y = df["species"]       # shape: (150,)  — 150 labels

print("X shape:", X.shape)   # (150, 4)
print("Y shape:", Y.shape)   # (150,)
```

**What I learned:**
- `X` is a **DataFrame** (2D — rows × features). Shape `(150, 4)`.
- `Y` is a **Series** (1D — just the labels). Shape `(150,)`.
- The `feature_cols` list makes it easy to change which columns to include/exclude — important for feature selection later.
- We never include the label column in X — the model must not see the answer during training.

**Note:** The print statement in this file says "Step 2" instead of "Step 3" — a small typo that happens when coding incrementally. Easy to overlook when focused on logic.

**Key concept:** Separating X and Y is not just convention — it's a hard requirement. The model learns the mapping from X → Y.

---

### Step 4 — `CaseStudyStep4.py`
**Visualize the Dataset (Scatter Plot by Class)**

Before training, we visualize the data to see if the classes are separable — can a line (or tree) actually distinguish them?

```python
plt.figure(figsize=(7, 5))

for sp in df["species"].unique():           # loop over 3 species
    temp = df[df["species"] == sp]          # filter rows for that species
    plt.scatter(
        temp["petal length (cm)"],
        temp["petal width (cm)"],
        label=sp                            # legend label = species name
    )

plt.title("Iris: Petal length vs Petal width")
plt.xlabel("Petal length (cm)")
plt.ylabel("Petal width (cm)")
plt.legend()
plt.grid(True)
plt.show()
```

**What the plot reveals:**
- **Setosa** (small petal length + width) forms a tight, separate cluster.
- **Versicolor** and **Virginica** partially overlap — these are harder to separate.
- Petal length and width are the most **discriminative features** — they separate classes better than sepal measurements.

**What I learned:**
- Looping over `df["species"].unique()` automatically handles any number of classes — good general code.
- `df[df["species"] == sp]` is **boolean indexing** — filtering rows where a condition is True.
- `label=sp` connects each scatter group to the legend automatically.
- `plt.grid(True)` adds grid lines — makes reading exact values easier.

**Key concept:** If classes are visually separable in a scatter plot, a Decision Tree (or any classifier) will likely do well. This step gives us confidence before even touching the model.

---

### Step 5 — `CaseStudyStep5.py`
**Train / Test Split**

We cannot evaluate a model on data it was trained on. This step uses `train_test_split` to divide data into a training set and a held-out test set.

```python
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,       # 20% for testing, 80% for training
    random_state=42      # fixed seed → reproducible results every run
)
```

**Resulting shapes:**

| Variable | Shape | Meaning |
|----------|-------|---------|
| `X` | (150, 4) | Full feature set |
| `Y` | (150,) | Full label set |
| `X_train` | (120, 4) | 80% of features — for training |
| `X_test` | (30, 4) | 20% of features — for testing |
| `Y_train` | (120,) | 80% of labels — for training |
| `Y_test` | (30,) | 20% of labels — ground truth for evaluation |

**What I learned:**
- `train_test_split` returns 4 items in order: X_train, X_test, Y_train, Y_test.
- `test_size=0.2` means 20% goes to testing. 150 × 0.2 = 30 test samples, 120 for training.
- `random_state=42` is a seed that fixes which rows go to train/test. Without it, you'd get different splits every run and different results — impossible to compare.
- The function **shuffles** the data before splitting, so it doesn't just take the first 120 rows.

**Key concept:** `random_state=42` is a convention in ML — 42 is used everywhere so results are reproducible and shareable.

---

### Step 6 — `CaseStudyStep6.py`
**Build the Model (with Hyperparameters)**

We create the DecisionTreeClassifier object and configure it with hyperparameters before any training happens.

```python
model = DecisionTreeClassifier(
    criterion="gini",    # split quality measure
    max_depth=3,         # maximum tree depth — limits complexity
    random_state=42      # reproducibility
)
print("Model successfully created:", model)
```

**Understanding the hyperparameters:**

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `criterion="gini"` | Gini Impurity | Measures how "mixed" a node's classes are. Lower = purer split. |
| `max_depth=3` | 3 levels deep | Limits tree growth — prevents overfitting |
| `random_state=42` | Fixed seed | Makes tie-breaking in splits reproducible |

**What I learned:**
- Creating the model object does NOT train it — it just configures the algorithm with its settings.
- `max_depth` is a **hyperparameter** — a setting we choose before training, not something the model learns. Choosing hyperparameters carefully is called **hyperparameter tuning**.
- `criterion="gini"` uses Gini Impurity. The alternative is `entropy` (information gain). Both produce similar results.
- A tree with no `max_depth` grows until every leaf is pure — this **overfits** (memorizes training data, fails on new data).

**Key concept:** Hyperparameters control how a model learns. They are set by us, not learned from data. Choosing good hyperparameters is a major skill in ML.

---

### Step 7 — `CaseStudyStep7.py`
**Train the Model**

One line. This is where the actual learning happens.

```python
model.fit(X_train, Y_train)
print("Model training completed")
```

**What `.fit()` does internally:**
1. Looks at all 120 training samples.
2. Finds the best feature + threshold to split on at each node (e.g., "petal length < 2.45?").
3. Repeats until `max_depth=3` is reached or all leaves are pure.
4. Stores the learned tree structure inside the `model` object.

**What I learned:**
- `.fit(X_train, Y_train)` trains on the training set ONLY. `X_test` and `Y_test` are never seen here.
- After `.fit()`, the `model` object contains the full decision tree — all the learned if/else rules.
- Training on the Iris dataset takes milliseconds. Real-world datasets with millions of rows can take hours.
- The model is now ready to make predictions.

**Key concept:** `.fit()` = learning. The model reads the training data and builds its internal rules. After this, it knows how to classify new flowers.

---

### Step 8 — `CaseStudyStep8.py`
**Make Predictions (Evaluate)**

We use the trained model to predict species for the 30 test samples it has never seen.

```python
Y_pred = model.predict(X_test)

print("Expected answers:")
print(Y_test)        # ground truth from the CSV

print("Predicted answers:")
print(Y_pred)        # model's predictions
```

**What I learned:**
- `.predict(X_test)` returns an array of predicted labels, one per test sample.
- `Y_test` is the **actual** (ground truth) species. `Y_pred` is the **predicted** species.
- At this stage, we compare them visually — do the arrays match?
- `Y_pred.shape` is `(30,)` — 30 predictions for 30 test samples.
- The model predicts from patterns it learned in Step 7 — it has never seen these 30 rows before.

**Key concept:** Comparing `Y_test` vs `Y_pred` is the core of model evaluation. Every mismatch is a wrong prediction — an error we need to measure and minimize.

---

### Step 9 — `CaseStudyStep9.py`
**Model Performance Evaluation + Hyperparameter Experiment**

This is the most information-rich step. Three evaluation tools are used together, AND the hyperparameters are changed to experiment with model improvement.

**Changes from Step 8:**
- `test_size` changed from `0.2` → `0.5` (now 75 train / 75 test)
- `max_depth` changed from `3` → `5` (deeper, more complex tree)

```python
# Accuracy Score
accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy of model:", accuracy * 100)

# Confusion Matrix
cm = confusion_matrix(Y_test, Y_pred)
print("Confusion matrix:")
print(cm)

# Classification Report
print(classification_report(Y_test, Y_pred))
```

**Understanding each metric:**

**Accuracy Score:**
```
Accuracy = Correct predictions / Total predictions
         = (TP + TN) / Total
```
Simple but can be misleading with imbalanced classes.

**Confusion Matrix (3×3 for 3 classes):**
```
              Predicted
              setosa  versicolor  virginica
Actual setosa   [  ]       [  ]      [  ]
   versicolor   [  ]       [  ]      [  ]
    virginica   [  ]       [  ]      [  ]
```
Diagonal = correct predictions. Off-diagonal = errors (which class got confused with which).

**Classification Report — per-class metrics:**

| Metric | Formula | Meaning |
|--------|---------|---------|
| Precision | TP / (TP + FP) | Of all predicted as this class, how many actually are? |
| Recall | TP / (TP + FN) | Of all actual members of this class, how many did we catch? |
| F1-Score | 2 × P × R / (P + R) | Balanced measure of Precision and Recall |
| Support | Count | How many test samples belong to this class |

**What I learned about hyperparameter tuning:**
- Changing `test_size` to 0.5 gives the model less training data (75 instead of 120). This tests whether the model still learns well with less data.
- Increasing `max_depth` to 5 allows a more complex tree — which can be better OR cause overfitting.
- Comparing accuracy between Step 8 and Step 9 configurations tells you which settings work better.
- This is the core idea of **model improvement**: systematically changing settings and measuring the effect.

**Key concept:** Never rely on accuracy alone. The classification report gives a complete picture — especially important when one class is harder to predict than others.

---

### Step 10 — `CaseStudyStep10.py`
**Visualize the Confusion Matrix**

The final step turns the confusion matrix numbers into a visual heatmap with class labels.

```python
data = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_   # ['setosa', 'versicolor', 'virginica']
)
data.plot()
plt.title("Confusion Matrix of Iris Dataset")
plt.show()
```

**What the visualization adds:**
- Color coding: darker squares = more predictions in that cell.
- Actual class names instead of 0/1/2 numbers.
- Diagonal squares (correct predictions) are visually obvious.
- Off-diagonal squares (errors) immediately show which classes get confused.
- `model.classes_` automatically retrieves the class names from the trained model — no hardcoding needed.

**What I learned:**
- `ConfusionMatrixDisplay` is sklearn's built-in visual confusion matrix — much cleaner than building one manually.
- `model.classes_` gives `['setosa', 'versicolor', 'virginica']` — the unique class names in training order.
- Visual evaluation is as important as numeric evaluation. A plot communicates errors instantly.

**Key concept:** Always visualize your results. A confusion matrix plot is often more informative than a table of numbers — patterns of errors become immediately visible.

---

## Complete Pipeline Summary

| Step | File | ML Pipeline Stage | Key Code / Tool |
|------|------|------------------|----------------|
| 1 | CaseStudyStep1.py | Data Collection | `pd.read_csv()`, `df.head()` |
| 2 | CaseStudyStep2.py | Data Analysis (EDA) | `.shape`, `.isnull()`, `.describe()`, `value_counts()` |
| 3 | CaseStudyStep3.py | Feature / Label Split | `X = df[feature_cols]`, `Y = df["species"]` |
| 4 | CaseStudyStep4.py | Data Visualization | `plt.scatter()` loop colored by class |
| 5 | CaseStudyStep5.py | Train/Test Split | `train_test_split(X, Y, test_size=0.2, random_state=42)` |
| 6 | CaseStudyStep6.py | Model Selection | `DecisionTreeClassifier(criterion, max_depth)` |
| 7 | CaseStudyStep7.py | Model Training | `model.fit(X_train, Y_train)` |
| 8 | CaseStudyStep8.py | Prediction | `Y_pred = model.predict(X_test)` |
| 9 | CaseStudyStep9.py | Evaluation + Tuning | `accuracy_score`, `confusion_matrix`, `classification_report` |
| 10 | CaseStudyStep10.py | Visual Evaluation | `ConfusionMatrixDisplay` |

---

## Hyperparameter Experiments Across Steps

| Step | test_size | max_depth | Effect |
|------|-----------|-----------|--------|
| Steps 5–8 | 0.2 (20%) | 3 | Standard setup — 120 train, 30 test, shallow tree |
| Steps 9–10 | 0.5 (50%) | 5 | Experiment — 75 train, 75 test, deeper tree |

**Why this matters:** Changing these values and comparing the accuracy score is the simplest form of hyperparameter tuning. In professional ML, tools like `GridSearchCV` automate this process across many combinations.

---

## Key Concepts Introduced in This Case Study

| Concept | Where Introduced | Simple Explanation |
|---------|-----------------|-------------------|
| `pd.read_csv()` | Step 1 | Load any CSV file into a DataFrame |
| `df.isnull().sum()` | Step 2 | Count missing values per column |
| `df.describe()` | Step 2 | Statistical summary (mean, std, min, max) |
| `df.value_counts()` | Step 2 | Count occurrences of each category |
| Boolean indexing | Step 4 | Filter rows: `df[df["col"] == value]` |
| `train_test_split` | Step 5 | Split into training and test sets automatically |
| `random_state=42` | Step 5, 6 | Fix randomness for reproducible results |
| `criterion="gini"` | Step 6 | Gini Impurity — how mixed is each node? |
| `max_depth` | Step 6 | Limit tree complexity to prevent overfitting |
| `.fit()` | Step 7 | Train the model |
| `.predict()` | Step 8 | Make predictions on new data |
| `accuracy_score` | Step 9 | % of correct predictions |
| `classification_report` | Step 9 | Precision, Recall, F1 per class |
| `ConfusionMatrixDisplay` | Step 10 | Visual heatmap of the confusion matrix |
| `model.classes_` | Step 10 | Get class names from trained model |

---

## Observations & Reflections

- This case study is different from all previous ones — instead of learning one concept at a time, this one builds the **complete pipeline end-to-end**. Every previous topic (DataFrames, visualization, confusion matrix) is used here.
- The cumulative file structure (each step adds to the previous) is a smart way to learn. If something breaks, you know exactly which step introduced the problem.
- Step 9 was the most educational — changing `test_size` and `max_depth` and observing the effect on accuracy is the foundation of **hyperparameter tuning**. I now understand why data scientists spend so much time tweaking these values.
- `random_state=42` appears everywhere and I finally understand why — without it, every run gives different results and it's impossible to know if an improvement is real or just luck.
- The `classification_report` is much more useful than a single accuracy number. Seeing precision and recall per class shows exactly where the model struggles.
- Setosa is almost always classified perfectly (it's well-separated in feature space). Versicolor and Virginica sometimes get confused — which matches exactly what the scatter plot in Step 4 showed.

---

## What's Next

- Use `plot_tree()` from `sklearn.tree` to visualize the actual decision tree — see every if/else rule the model learned.
- Try `GridSearchCV` to automatically test many `max_depth` and `criterion` combinations and find the best one.
- Try other classifiers on the same data: `KNeighborsClassifier`, `RandomForestClassifier`, `SVC`.
- Use `cross_val_score` for k-fold cross-validation — a more reliable way to evaluate than a single train/test split.
- Apply this same 10-step pipeline to a different dataset (e.g., Titanic, Breast Cancer).

---

## Files

| File | Steps Covered | New Addition |
|------|--------------|--------------|
| `CaseStudyStep1.py` | Step 1 | Load CSV with `pd.read_csv()` |
| `CaseStudyStep2.py` | Steps 1–2 | EDA — shape, nulls, distribution, describe |
| `CaseStudyStep3.py` | Steps 1–3 | Define X (features) and Y (labels) |
| `CaseStudyStep4.py` | Steps 1–4 | Scatter plot visualization per class |
| `CaseStudyStep5.py` | Steps 1–5 | Train/test split (80/20) |
| `CaseStudyStep6.py` | Steps 1–6 | Build DecisionTree (gini, max_depth=3) |
| `CaseStudyStep7.py` | Steps 1–7 | Train the model with `.fit()` |
| `CaseStudyStep8.py` | Steps 1–8 | Predict with `.predict()`, compare Y_test vs Y_pred |
| `CaseStudyStep9.py` | Steps 1–9 | Accuracy, confusion matrix, classification report + tuning (50/50, depth=5) |
| `CaseStudyStep10.py` | Steps 1–10 | Visualize confusion matrix with `ConfusionMatrixDisplay` |


*OJT Report — Machine Learning Track | Iris Case Study: Complete Pipeline*
*Trainee: Shubhada A. Palwe*
