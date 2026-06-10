# On-the-Job Training (OJT) Report

---

| Field               | Details                                                             |
|---------------------|---------------------------------------------------------------------|
| **Trainee Name**    | Shubhada A Palwe                               |
| **Training Domain** | Machine Learning — Supervised Classification (Case Study)           |
| **Topic**           | Wine Quality Classifier — Full KNN Pipeline on Real-World Data      |
| **Tools Used**      | Python, pandas, scikit-learn (KNN, StandardScaler), matplotlib      |
| **Total Files**     | 5 (wineClassifierKNN1 →wineClassifierKNNModelVisulizationFinal)                  |
| **Dataset**         | WinePredictor.csv — 178 wine samples, 13 chemical features, 3 classes |
| **Pipeline Steps**  | 12 (Load → Clean → Split X/Y → Train/Test Split → Scale → Tune K → Plot → Best K → Final Model → Accuracy → Confusion Matrix → Report) |

---

## The Case Study

This is the most complete and professional project in the OJT so far.

Previous topics used either a **built-in sklearn dataset** (Iris — 150 rows, clean, ready to use) or a **tiny toy dataset** (5 data points for Linear Regression, 4 points for UserDefinedKNN). This case study introduces a **real-world CSV file** — data that had to be loaded, cleaned, inspected, and prepared before any model could be built.

The dataset is the **UCI Wine Recognition Dataset** — a classic benchmark in machine learning. It contains the results of a chemical analysis of 178 wines grown in Italy, all from the same region but from three different cultivars (grape varieties). The challenge: given 13 chemical measurements of a wine, correctly identify which of the 3 types it is.

This is not a 3-line sklearn experiment. It is a **12-step production-style pipeline** — the kind of structure used in real ML projects.

---

## The Dataset: WinePredictor.csv

| Property | Value |
|----------|-------|
| Rows | 178 wine samples |
| Features (X) | 13 chemical measurements |
| Target (Y) | `Class` — wine type 1, 2, or 3 |
| Source | UCI Machine Learning Repository |

**The 13 features:**

| # | Feature | What It Measures |
|---|---------|-----------------|
| 1 | Alcohol | Alcohol content percentage |
| 2 | Malic acid | Acidity (affects taste, aging) |
| 3 | Ash | Mineral content |
| 4 | Alcalinity of ash | Alkalinity of the ash |
| 5 | Magnesium | Magnesium concentration (mg/L) |
| 6 | Total phenols | Antioxidant compounds |
| 7 | Flavanoids | A type of phenol (affects color, taste) |
| 8 | Nonflavanoid phenols | Other phenol types |
| 9 | Proanthocyanins | Tannins (affects bitterness) |
| 10 | Color intensity | Depth of color |
| 11 | Hue | Color tone |
| 12 | OD280/OD315 | Protein content measurement |
| 13 | Proline | An amino acid (ranges from ~300 to 1700) |

**Why this dataset is challenging for KNN:** The features have vastly different scales. Proline values range from ~300 to 1700. Malic acid values range from 0.74 to 5.80. In raw form, Proline would completely dominate every distance calculation — a 1000-unit difference in Proline would drown out a 5-unit difference in Malic acid. This is why **feature scaling** (Step 5) is not optional here — it is essential.

---

## The 12-Step Pipeline

The entire case study follows one function: `MarvellousClassifier(Datapath)`. Each file adds more steps to this function until all 12 steps are complete in the final file.

---

### Step 1 — Load the Dataset (all files)

```python
df = pd.read_csv(Datapath)
print(df.head())
```

Unlike `load_iris()` which comes packaged in sklearn, this dataset is loaded from a **CSV file on disk**. `pd.read_csv()` reads it into a DataFrame. `df.head()` shows the first 5 rows — the first sanity check: does the data look right?

---

### Step 2 — Clean the Data (all files)

```python
df.dropna(inplace=True)
print("total records :", df.shape[0])
print("Total columns :", df.shape[0])   # ← bug: should be df.shape[1]
```

`df.dropna()` removes any row that has a missing value. The `inplace=True` means the original DataFrame is modified directly — no need to reassign.

**Bug spotted:** `df.shape[0]` is used for both "total records" and "total columns." The column count should use `df.shape[1]`. `df.shape` returns a tuple `(rows, columns)` — index 0 is rows, index 1 is columns. This prints the row count twice. The Iris dataset examples didn't need this check because `load_iris()` always returns clean data — this bug appeared because working with CSVs requires more careful inspection.

---

### Step 3 — Separate X and Y (`wineClassifierKNNEDA.py`)

```python
X = df.drop(columns=['Class'])   # all 13 features
Y = df['Class']                  # wine type (1, 2, or 3)

print("Shape of X :", X.shape)   # (178, 13)
print("Shape of Y :", Y.shape)   # (178,)
print("Input columns :", X.columns.to_list())
```

The `Class` column is the label — what we want to predict. Every other column is a feature — what the model uses to predict. `df.drop(columns=['Class'])` removes the target column and assigns the remaining 13 columns to X.

Note: the file is named `wineClassifierKNNEDA.py` suggesting Exploratory Data Analysis. In full practice, EDA would include `df.describe()` (statistics), `df['Class'].value_counts()` (class distribution), and correlation heatmaps. Here, EDA is limited to the X/Y shape inspection — a starting point for deeper analysis.

---

### Step 4 — Train/Test Split (`wineClassifierKNNModel.py` onwards)

```python
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)
```

**Two important additions compared to earlier projects:**

`random_state=42` — the split is fixed. Results are the same every run. (This was the lesson learned from KNN1.py vs KNN2.py.)

`stratify=Y` — **new concept introduced here.** Stratified splitting ensures each class (1, 2, 3) appears in the same proportion in both train and test sets. Without stratify, a random split might accidentally put all of class 3 into training and none into test — the model would never be evaluated on that class. Stratify prevents this.

**Split result:**
- X_train: ~142 rows (80%), X_test: ~36 rows (20%)
- Each class represented proportionally in both sets

---

### Step 5 — Feature Scaling (`wineClassifierKNNModel.py` onwards)

```python
scalar = StandardScaler()
X_train_scaled = scalar.fit_transform(X_train)
X_test_scaled  = scalar.fit_transform(X_test)   # ← important bug here
```

**Why KNN absolutely needs scaling:**

KNN classifies by distance. Consider two wine samples:
- Sample A: Proline = 1065, Alcohol = 14.23
- Sample B: Proline = 500, Alcohol = 13.20

Distance contribution from Proline: (1065 - 500)² = **319,225**
Distance contribution from Alcohol: (14.23 - 13.20)² = **1.06**

Proline contributes 300,000x more to the distance than Alcohol. The model essentially ignores every other feature and classifies wines purely by Proline. This is clearly wrong.

`StandardScaler` fixes this by transforming every feature to have **mean=0 and standard deviation=1**:
```
scaled_value = (original_value - mean) / std_deviation
```

After scaling, all 13 features contribute equally to the distance calculation.

**The bug — data leakage:**

`scalar.fit_transform(X_test)` refits the scaler on the test data, learning the test set's own mean and standard deviation. The correct code is `scalar.transform(X_test)` — apply the scaler that was already fitted on the *training* data.

Using `fit_transform` on the test set causes **data leakage**: the test set's statistics influence the scaling, which means the test set is no longer truly "unseen" data. In practice on this dataset the effect is small, but on noisy real-world data this mistake can give falsely optimistic accuracy numbers.

---

### Step 6 — Hyperparameter Tuning: K=1 to K=20 (`wineClassifierKNNModel.py` onwards)

```python
accuracy_scores = []
K_values = range(1, 21)

for k in K_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, Y_train)
    Y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(Y_test, Y_pred)
    accuracy_scores.append(accuracy)
```

Instead of picking one K and hoping it works, this loop tries **every K from 1 to 20** and records the accuracy for each. This is systematic hyperparameter tuning — finding the best K by evidence rather than guessing.

The results are stored in `accuracy_scores` — a list of 20 accuracy values, one per K.

---

### Step 7 — K vs Accuracy Plot (`wineClassifierKNNModelVisulization.py` onwards)

```python
plt.figure(figsize=(8, 5))
plt.plot(K_values, accuracy_scores, marker='o')
plt.title("k value vs Accuracy")
plt.xlabel("Value of K")
plt.ylabel("Accuracy")
plt.grid(True)
plt.xticks(list(K_values))
plt.show()
```

This plot makes the hyperparameter search **visual**. Each point on the graph shows the accuracy achieved by a specific K. The best K is the one at the highest point.

**Bug noted:** The print statement for Step 7 says `"step 6: Explore the multiple values of K"` — a copy-paste error. The step number in the print is wrong (should say "step 7"). The code itself is correct.

`plt.grid(True)` and `plt.xticks(list(K_values))` are quality-of-life additions — the grid makes it easier to read exact K values, and showing every K tick (1 through 20) makes the plot more informative.

---

### Steps 8–12 — Final Model, Evaluation (`wineClassifierKNNModelVisulizationFinal.py`)

The final file adds 5 more steps to complete the pipeline:

**Step 8 — Find Best K:**
```python
best_k = list(K_values)[accuracy_scores.index(max(accuracy_scores))]
print("Best value of K is:", best_k)
```
`max(accuracy_scores)` finds the highest accuracy. `.index(...)` finds its position in the list. `list(K_values)[index]` gets the corresponding K value. One line that picks the winning K automatically.

**Step 9 — Build Final Model with Best K:**
```python
final_model = KNeighborsClassifier(n_neighbors=best_k)
final_model.fit(X_train_scaled, Y_train)
Y_pred = final_model.predict(X_test_scaled)
```
The loop in Step 6 was for exploration. This step builds the *actual* final model using only the best K found.

**Step 10 — Final Accuracy:**
```python
accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy of model is:", accuracy * 100)
```

**Step 11 — Confusion Matrix:**
```python
cm = confusion_matrix(Y_test, Y_pred)
print(cm)
```
Shows a 3×3 matrix (3 wine classes). The diagonal values are correct predictions; off-diagonal values show which classes get confused with each other.

**Step 12 — Classification Report:**
```python
print(classification_report(Y_test, Y_pred))
```
Goes beyond accuracy to show **precision, recall, and F1-score** for each class individually:
- **Precision:** Of all the times the model said "Class 1," how often was it right?
- **Recall:** Of all actual Class 1 wines, how many did the model find?
- **F1-score:** Harmonic mean of precision and recall — the balanced single-number summary.

This is important for imbalanced datasets. A model that gets 95% accuracy but completely misses Class 3 (small class) would look good by accuracy alone — the classification report exposes this.

---

## The 5-File Build — What Each File Adds

| File | Steps Added | New Concept |
|------|-------------|-------------|
| `wineClassifierKNN1.py` | 1–2 | Load CSV with `pd.read_csv()`, clean with `dropna()` |
| `wineClassifierKNNEDA.py` | 3 | X/Y separation from DataFrame, `df.drop(columns=[...])` |
| `wineClassifierKNNModel.py` | 4–6 | `stratify=Y`, `StandardScaler`, K=1–20 tuning loop |
| `wineClassifierKNNModelVisulization.py` | 7 | K vs Accuracy line plot with `plt.grid`, `plt.xticks` |
| `wineClassifierKNNModelVisulizationFinal.py` | 8–12 | Best K selection, final model, confusion matrix, classification report |

---

## Bugs Found and What They Teach

| File | Bug | Correct Code | Lesson |
|------|-----|-------------|--------|
| `wineClassifierKNN1.py` | `df.shape[0]` used for both rows and columns | Second print should use `df.shape[1]` | `.shape` returns `(rows, cols)` — index 0 = rows, index 1 = cols |
| `wineClassifierKNNModel.py` | `scalar.fit_transform(X_test)` | Should be `scalar.transform(X_test)` | Fit the scaler only on training data; transform applies it to test data |
| `wineClassifierKNNModelVisulization.py` | Step 7 print says "step 6" | Should say "step 7" | Copy-paste error in print statement — cosmetic but worth fixing |

---

## New Concepts Introduced in This Case Study

| Concept | Where | Why It Matters |
|---------|-------|---------------|
| `pd.read_csv()` | Step 1 | Real datasets come from files, not sklearn |
| `df.dropna(inplace=True)` | Step 2 | Real data has missing values — must clean before use |
| `df.drop(columns=[...])` | Step 3 | Separating target column from feature columns |
| `stratify=Y` | Step 4 | Ensures class balance in train/test split |
| `StandardScaler` | Step 5 | Essential for KNN — without scaling, large-scale features dominate distances |
| K tuning loop (1–20) | Step 6 | Systematic hyperparameter search instead of guessing |
| K vs Accuracy plot | Step 7 | Visual hyperparameter selection |
| `accuracy_scores.index(max(...))` | Step 8 | Programmatic best-K selection |
| `confusion_matrix` | Step 11 | Detailed error breakdown — which classes get confused? |
| `classification_report` | Step 12 | Precision, recall, F1 per class — goes beyond accuracy |

---

## Why StandardScaler is Critical for KNN (Deep Dive)

This is the most important new concept in this case study.

Without scaling:

```
Distance formula uses: (Proline_A - Proline_B)²  +  (Alcohol_A - Alcohol_B)²  +  ...
                       ≈ (1065 - 500)²            +  (14.23 - 13.20)²          +  ...
                       ≈ 319,225                   +  1.06                      +  ...
```

Proline drowns everything else out. The model is effectively a "Proline Classifier" — it ignores the other 12 features.

After StandardScaler:
```
All features have mean=0, std=1
Distance formula treats all features equally
KNN actually uses all 13 chemical properties to distinguish wine types
```

The model becomes genuinely multi-dimensional — which is what we actually want.

---

## Observations & Reflections

- This case study felt like a major step up from everything before. For the first time, the data came from a CSV file — which means it could have been messy, it could have had missing rows, it could have had different column names. `df.dropna()` and `df.head()` were the first tools for checking that the data was what it was supposed to be.

- The `stratify=Y` parameter in `train_test_split` was completely new and, in hindsight, important. Without it, the random split could have put all 59 Class 1 wines into training and left Class 1 unrepresented in the test set. Stratify prevents evaluation failures from unbalanced splits.

- Discovering the `scalar.fit_transform(X_test)` bug while documenting this was a real lesson. The code runs without error — Python doesn't complain. The bug is *conceptual*: fitting the scaler on test data leaks information. This kind of bug only shows up when you understand what the code is supposed to do, not just what it does.

- The classification report output — with precision, recall, and F1 per class — felt much more honest than a single accuracy number. A single accuracy number could hide the fact that the model performs poorly on one specific wine class. The full report exposes that.

- Building the K vs Accuracy plot (Step 7) and then using code to automatically find the best K (Step 8) instead of just visually reading the graph is a small but meaningful step. The code picks the winner programmatically — no guessing, no manual reading.

---

## What's Next

- Fix the `scalar.fit_transform(X_test)` bug → replace with `scalar.transform(X_test)` and observe whether accuracy changes.
- Add proper EDA to `wineClassifierKNNEDA.py`: `df.describe()`, `df['Class'].value_counts()`, correlation heatmap with seaborn.
- Use `cross_val_score` instead of a single train/test split for a more reliable K selection.
- Try other classifiers (Decision Tree, SVM) on the same Wine dataset and compare classification reports.
- Explore `KNeighborsClassifier(metric='manhattan')` — Manhattan distance as an alternative to Euclidean for this dataset.

---

## Files

| File | Steps | Key Addition |
|------|-------|-------------|
| `wineClassifierKNN1.py` | 1–2 | CSV load + dropna |
| `wineClassifierKNNEDA.py` | 1–3 | X/Y split |
| `wineClassifierKNNModel.py` | 1–6 | stratify, StandardScaler, K tuning loop |
| `wineClassifierKNNModelVisulization.py` | 1–7 | K vs Accuracy plot |
| `wineClassifierKNNModelVisulizationFinal.py` | 1–12 | Best K, final model, confusion matrix, classification report |
| `WinePredictor.csv` | — | 178 wine samples, 13 features, 3 classes |


*OJT Report — Machine Learning Track | Topic: Wine Classifier — KNN Case Study*
*Trainee: Shubhada A. Palwe*
*Milestone: First real-world CSV pipeline — from raw data to classification report*
