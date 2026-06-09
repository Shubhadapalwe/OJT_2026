# On-the-Job Training (OJT) Report

---

| Field               | Details                          |
|---------------------|----------------------------------|
| **Trainee Name**    | _(Shubhada A. Palwe)_                    |
| **Training Domain** | Machine Learning — Supervised Classification |
| **Case Study**      | Iris Flower Classification       |
| **Tools Used**      | Python, scikit-learn (sklearn)   |
| **Total Versions**  | 5                                |
|                                                        |

---

## Objective

To explore and understand a real-world, standard ML dataset (the Iris dataset) using `scikit-learn`, and progressively learn how to load, inspect, and navigate structured data before building a classifier.

---

## About the Dataset

The **Iris dataset** is one of the most famous datasets in Machine Learning. It contains measurements of 150 iris flowers from 3 different species.

| Feature              | Description                         |
|----------------------|-------------------------------------|
| Sepal Length (cm)    | Length of the sepal leaf            |
| Sepal Width (cm)     | Width of the sepal leaf             |
| Petal Length (cm)    | Length of the petal                 |
| Petal Width (cm)     | Width of the petal                  |

**Target Classes (Dependent Variable):**

| Label | Species          |
|-------|-----------------|
| 0     | Iris Setosa      |
| 1     | Iris Versicolor  |
| 2     | Iris Virginica   |

Total samples: **150** (50 per class)

---

## Learning Progression — Version by Version

---

### Version 1 — `IrisClassification1.py`
**Topic: Loading a built-in dataset from sklearn**

**What I did:**
I discovered that `sklearn` comes with built-in datasets — no need to download or prepare data manually. I used `load_iris()` to load the entire dataset and just printed it to see what it looked like.

```python
from sklearn.datasets import load_iris
Dataset = load_iris()
print(Dataset)
```

**What I observed:**
The output was a large block of data — numbers, arrays, and keys. It was hard to read, but I could see data was actually there. I realized this is a `Bunch` object (like a dictionary) containing features, labels, and metadata.

**Key learning:** sklearn ships with ready-to-use datasets. `load_iris()` returns a structured object — not just a table.

---

### Version 2 — `IrisClassification2.py`
**Topic: Exploring dataset metadata — feature names and target names**

**What I did:**
Instead of printing the whole raw dataset, I started exploring its structure using `.feature_names` and `.target_names`.

```python
print(Dataset.feature_names)
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

print(Dataset.target_names)
# ['setosa' 'versicolor' 'virginica']
```

**What I observed:**
- There are **4 independent variables** (features) — all physical measurements of the flower.
- There are **3 dependent variables** (target classes) — the 3 species we want to classify.

**Key learning:** Before touching the data values, always inspect the *metadata* first. Knowing what columns mean and what you're predicting is essential.

---

### Version 3 — `IrisClassification3.py`
**Topic: Understanding dataset dimensions using `len()`**

**What I did:**
I added `len()` to check how many features and how many target classes exist.

```python
print("Length of independent variables:", len(Dataset.feature_names))  # 4
print("Length of dependent variables:",  len(Dataset.target_names))    # 3
```

**What I observed:**
- 4 features → the model will use 4 inputs to make a prediction.
- 3 target classes → this is a **multi-class classification** problem (not binary like Tennis vs Cricket).

**Key learning:** Checking dimensions is a basic but important habit. In larger datasets, you'd also check `Dataset.data.shape` to see how many rows and columns exist.

---

### Version 4 — `IrisClassification4.py`
**Topic: Accessing individual samples and labels by index**

**What I did:**
I started treating the dataset like an array — accessing specific rows of data and their corresponding labels using index positions.

```python
print(Dataset.data[0])    # First flower's measurements
print(Dataset.target[0])  # First flower's label → 0 (Setosa)

print(Dataset.target[50]) # → 1 (Versicolor — second class starts at index 50)
print(Dataset.target[100])# → 2 (Virginica — third class starts at index 100)
```

**What I observed:**
- The dataset is sorted by class: samples 0–49 are Setosa (label 0), 50–99 are Versicolor (label 1), 100–149 are Virginica (label 2).
- Each data row is an array of 4 float values: `[sepal_len, sepal_wid, petal_len, petal_wid]`.

**Key learning:** Datasets are indexed arrays. You can access any sample directly. The label at `Dataset.target[i]` corresponds to the features at `Dataset.data[i]`.

---

### Version 5 — `IrisClassification5.py`
**Topic: Iterating through the full dataset with formatted output**

**What I did:**
Instead of manually printing specific indexes, I looped through all 150 samples using `range(len(Dataset.target))` and printed each record in a readable, formatted way.

```python
Border = "-" * 40
print(Border)

for i in range(len(Dataset.target)):
    print("ID %d, features %s, label %s" % (i, Dataset.data[i], Dataset.target[i]))

print(Border)
```

**What I observed:**
- All 150 records printed in a clean, readable format.
- Label transitions were visible — 0s becoming 1s at index 50, 1s becoming 2s at index 100.
- Using `%d`, `%s` string formatting to mix integers and arrays in one print statement.

**Key learning:** Programmatic iteration with formatted output is how you *audit* and *verify* your dataset. This is part of the Data Analysis step in the ML pipeline. Never assume your data is clean — always look at it.

---

## Summary of Concepts Learned

| Concept | Understood In |
|---|---|
| Loading built-in sklearn datasets | Version 1 |
| Exploring metadata (feature/target names) | Version 2 |
| Checking dataset dimensions with `len()` | Version 3 |
| Index-based access to samples and labels | Version 4 |
| Iterating through full dataset, formatted print | Version 5 |

---

## ML Pipeline Progress Tracker

| Step | Status |
|------|--------|
| 1. Data Collection |  Done — used `load_iris()` |
| 2. Data Analysis | Done — inspected features, targets, dimensions, samples |
| 3. Data Cleaning |  Not needed (sklearn dataset is pre-cleaned) |
| 4. Model Selection |  Next step |
| 5. Model Training |  Next step |
| 6. Model Testing / Evaluation |  Next step |
| 7. Model Improvement | Next step |
| 8. Prediction / Deployment |  Next step |

---

## Observations & Reflections

The Iris dataset felt like a big step up from the Ball Classification case study. Here's what was different:

- **No manual encoding needed** — sklearn already stores the data as numbers.
- **Multi-class problem** — 3 output classes instead of 2 (Tennis/Cricket).
- **Real-world data** — 150 actual flower measurements collected by botanist Ronald Fisher in 1936.
- **Dataset exploration is its own skill** — I spent all 5 versions just *understanding* the data before touching a model. That's intentional. In real ML projects, data exploration takes more time than model training.

---

## What's Next

- Build a `DecisionTreeClassifier` on this dataset (applying what I learned in Ball Classification).
- Use `train_test_split` from `sklearn.model_selection` to split 150 samples properly.
- Measure accuracy with `accuracy_score`.
- Visualize the decision tree.

---

## Files

| File | Concept Practiced |
|---|---|
| `IrisClassification1.py` | Load dataset, print raw output |
| `IrisClassification2.py` | Inspect feature names and target names |
| `IrisClassification3.py` | Check dataset dimensions with `len()` |
| `IrisClassification4.py` | Index-based access to data and labels |
| `IrisClassification5.py` | Full dataset iteration with formatted output |

---

## Commit Message

```
feat: add Iris Classification ML case study — data exploration phase (OJT)
```

**Commit Description:**
```
Five incremental versions covering dataset loading and exploration
using sklearn's built-in Iris dataset.

Covers: load_iris(), feature/target metadata, dataset dimensions,
index-based access, and full dataset iteration with formatted output.

This is the data analysis phase of the ML pipeline — no model trained yet.
Tools: Python, scikit-learn
```

---

*OJT Report — Machine Learning Track | Case Study 2: Iris Classification*
