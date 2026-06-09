# On-the-Job Training (OJT) Report

---

| Field               | Details                                        |
|---------------------|------------------------------------------------|
| **Trainee Name**    | Shubhada A. Palwe                              |
| **Training Domain** | Machine Learning — Data Handling & Evaluation  |
| **Topics Covered**  | Pandas Series · Confusion Matrix               |
| **Tools Used**      | Python, pandas, scikit-learn (sklearn)         |
| **Total Files**     | 6 (5 Pandas Series + 1 Confusion Matrix)       |
|                                   

---

## Objective

To learn two important building blocks of Machine Learning:
1. **Pandas Series** — how to store and organize data in Python before feeding it to a model.
2. **Confusion Matrix** — how to honestly evaluate how well a model is performing after predictions.

---

## Topic 1: Pandas Series

### What is a Pandas Series?

Think of a Pandas Series like a single column in an Excel sheet — it holds a list of values, and each value has an index (like a row number). It is the simplest data structure in the `pandas` library.

Before we can train any ML model, we need our data in a clean, organized structure. Pandas helps us do that.

---

### Learning Progression — Version by Version

---

#### Version 1 — `Pandas_series_1.py`
**Topic: Creating a list and converting it to a Series**

I started by creating a normal Python list and then converting it into a Pandas Series using `pd.Series()`.

```python
Data = [11, 21, 51, 101, 11]
print(Data)          # plain Python list

sobj = pd.Series(Data)
print(sobj)          # Pandas Series — now has index!
```

**What I noticed:**
When you print a normal list you get `[11, 21, 51, 101, 11]`. But when you print a Series, pandas automatically adds an index (0, 1, 2, 3, 4) alongside each value. That index is what makes it powerful.

**Key learning:** A Pandas Series = data + index. It's more than just a list.

---

#### Version 2 — `Pandas_series_2.py`
**Topic: Cleaner way to create a Series directly**

Instead of creating a list first and then converting it, I passed the data directly into `pd.Series()`.

```python
sobj = pd.Series([11, 21, 51, 101, 11])
print(sobj)
```

**Key learning:** You can create a Series in one line. Less code, same result. In Python, writing clean and concise code is good practice.

---

#### Version 3 — `Pandas_series_3.py`
**Topic: Float (decimal) data in a Series**

I changed the values from integers to floats (decimal numbers) to see how Series handles different data types.

```python
sobj = pd.Series([11.0, 21.0, 51.0, 101.0, 11.0])
print(sobj)
```

**What I noticed:**
The dtype (data type) shown at the bottom changed from `int64` to `float64`. A Series automatically detects the data type of its values.

**Key learning:** A Pandas Series stores the data type of its values. This matters because ML models often need specific data types (usually float).

---

#### Version 4 — `Pandas_series_4.py`
**Topic: Custom numeric indexing**

By default, pandas uses 0, 1, 2... as the index. But I can set my own custom numeric index.

```python
sobj = pd.Series([11.0, 21.0, 51.0, 101.0, 11.0], index=[5, 6, 7, 8, 9])
print(sobj)
```

**What I noticed:**
Now the index starts at 5 instead of 0. This is useful when your data originally had a different numbering system and you want to preserve it.

**Key learning:** You control the index. This helps when merging datasets or keeping track of original row numbers.

---

#### Version 5 — `Pandas_series_5.py`
**Topic: String (label) indexing — Series as a dictionary**

This was the most interesting version. I used text labels as the index, and then accessed values using those labels like dictionary keys.

```python
sobj = pd.Series([25000, 27000, 29000, 30000], index=["PPA", "LB", "Python", "React"])
print(sobj)
print(sobj["Python"])  # Access by label → 29000
```

**What I noticed:**
- A Series with string labels behaves like a Python dictionary — but with extra powers.
- I also learned that a Series should contain **homogeneous data** (all values same type). Mixing types can cause unexpected behavior.

**Key learning:** String-indexed Series = labeled data storage. Very useful for representing structured real-world data like salaries by skill, scores by student, prices by product.

---

### Pandas Series — Summary Table

| Version | Concept |
|---------|---------|
| 1 | List → Series conversion, auto index |
| 2 | Direct Series creation (one-liner) |
| 3 | Float dtype, automatic type detection |
| 4 | Custom numeric index |
| 5 | String index, label-based access, homogeneous data |

---

## Topic 2: Confusion Matrix

### `Confussion_Matrix.py`
**Topic: Evaluating a model's predictions honestly**

After a model makes predictions, we need to measure how good those predictions are. Accuracy alone is not enough — we need to know *what kind of mistakes* the model is making.

The **Confusion Matrix** shows us exactly that.

```python
from sklearn import confusion_matrix

Actual    = [1, 0, 1, 1, 1, 0, 1, 0, 0, 1]
Predicted = [1, 0, 0, 1, 1, 1, 1, 1, 0, 1]

con_mat = confusion_matrix(Actual, Predicted)
print(con_mat)
```

**What the output looks like:**

```
[[2  1]
 [1  6]]
```

**Reading the matrix:**

|                    | Predicted: 0 (Negative) | Predicted: 1 (Positive) |
|--------------------|------------------------|------------------------|
| **Actual: 0 (Negative)** | True Negative (TN) = 2  | False Positive (FP) = 1 |
| **Actual: 1 (Positive)** | False Negative (FN) = 1 | True Positive (TP) = 6  |

- **True Positive (TP):** Model said Positive, it IS Positive. ✅
- **True Negative (TN):** Model said Negative, it IS Negative. ✅
- **False Positive (FP):** Model said Positive, but it's actually Negative. ❌ (Type I Error)
- **False Negative (FN):** Model said Negative, but it's actually Positive. ❌ (Type II Error)

**Key learning:** A confusion matrix shows not just *how many* mistakes, but *what type* of mistakes. In medical diagnosis, a False Negative (missing a real patient) is far worse than a False Positive.

---

## ML Pipeline — Where These Topics Fit

| Step | Status |
|------|--------|
| 1. Data Collection |  Done |
| 2. Data Analysis |  Done — Pandas Series helps here |
| 3. Data Cleaning |  Done — Pandas helps here too |
| 4. Model Selection |  Done (previous case studies) |
| 5. Model Training | Done (previous case studies) |
| 6. Model Testing / Evaluation |  Done — **Confusion Matrix is Step 6** |
| 7. Model Improvement |  Next step |
| 8. Prediction / Deployment |  Next step |

---

## Observations & Reflections

- Pandas is not just for ML — it's a general-purpose data tool used in data science, finance, and analytics.
- The confusion matrix was eye-opening. Before this, I thought "accuracy" was the only way to judge a model. Now I realize accuracy can be misleading — a model can have 90% accuracy but still miss all the important cases.
- I noticed the import in `Confussion_Matrix.py` should be `from sklearn.metrics import confusion_matrix` — the correct submodule is `metrics`.

---

## What's Next

- Learn `pandas.DataFrame` — the 2D version of Series (rows + columns).
- Calculate **Precision**, **Recall**, and **F1 Score** from the confusion matrix.
- Use `sklearn.metrics.classification_report` for a full evaluation summary.
- Apply confusion matrix to the Iris classification model.

---

## Files

| File | Topic |
|------|-------|
| `Pandas_series_1.py` | List to Series, auto index |
| `Pandas_series_2.py` | Direct Series creation |
| `Pandas_series_3.py` | Float data type |
| `Pandas_series_4.py` | Custom numeric index |
| `Pandas_series_5.py` | String index, label access |
| `Confussion_Matrix.py` | Confusion matrix evaluation |

---

*OJT Report — Machine Learning Track | Topic: Pandas Series & Confusion Matrix*
*Trainee: Shubhada A. Palwe*
