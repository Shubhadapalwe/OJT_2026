# On-the-Job Training (OJT) Report

---

| Field               | Details                                              |
|---------------------|------------------------------------------------------|
| **Trainee Name**    | Shubhada A. Palwe                                    |
| **Training Domain** | Machine Learning — Supervised Classification         |
| **Topic**           | Decision Tree Classifier with Visualization          |
| **Tools Used**      | Python, scikit-learn, matplotlib                     |
| **Total Files**     | 2 (DecisionTree1.py · DecisionTree2.py)             |
| **Dataset**         | Iris (built-in sklearn dataset)                      |

---

## Objective

To implement a **Decision Tree Classifier** on the Iris dataset using scikit-learn — first as a basic accuracy experiment, then extended with a full **tree visualization** using `plot_tree`. The two files show the natural next step after getting a model running: making it visible and interpretable.

---

## What is a Decision Tree?

A Decision Tree is a supervised learning algorithm that makes predictions by asking a series of yes/no questions about the features of the data. Starting from a root question, each answer leads to either another question (internal node) or a final prediction (leaf node).

For the Iris dataset with 4 features (sepal length, sepal width, petal length, petal width), the tree might ask:

```
Is petal length ≤ 2.45?
├── Yes → Setosa (100% certain)
└── No → Is petal width ≤ 1.75?
          ├── Yes → Versicolor
          └── No  → Virginica
```

Each split is chosen to best separate the classes — the algorithm finds the feature and threshold that produces the "purest" groups at each step.

Unlike KNN (which stores all data and computes distances at prediction time), a Decision Tree **builds a structure** during training. Prediction is then fast — just follow the branches.

---

## Decision Tree vs KNN — Quick Comparison

| Property | Decision Tree | KNN |
|----------|--------------|-----|
| Training | Builds a tree of if/else rules | Stores all data (no learning) |
| Prediction | Follow branches — fast | Calculate all distances — slower |
| Interpretable | Yes — you can read the tree | No — it's just distances |
| Type | Model-based | Instance-based / Lazy learner |

---

## File by File

---

### `DecisionTree1.py` — Basic Classifier

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data
Y = iris.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

model = DecisionTreeClassifier()   # object tayar kela
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy is:", accuracy * 100)
```

**What each line does:**

| Line | What It Does |
|------|-------------|
| `load_iris()` | Loads the 150-row Iris dataset with 4 features and 3 class labels |
| `train_test_split(X, Y, test_size=0.2)` | 80% training (120 rows), 20% testing (30 rows) |
| `DecisionTreeClassifier()` | Creates a model with all default settings (no depth limit, Gini impurity) |
| `model.fit(X_train, Y_train)` | Builds the actual tree — finds the best splits |
| `model.predict(X_test)` | Walks each test point down the tree to a leaf node |
| `accuracy_score(Y_test, Y_pred)` | Counts correct predictions / total predictions |

**Key observation:** No `random_state` is set on the `train_test_split`, so the 80/20 split is different every run. This means accuracy varies slightly each time the file runs. On the Iris dataset, the default Decision Tree typically achieves **90–100% accuracy**, but the exact number changes run to run.

**What `DecisionTreeClassifier()` defaults to:**
- `criterion='gini'` — uses Gini impurity to measure the quality of each split
- `max_depth=None` — the tree grows until every leaf is pure (can overfit on noisy data)
- `min_samples_split=2` — a node splits if it has at least 2 samples

---

### `DecisionTree2.py` — Tree Visualization

The second file adds one important capability: **seeing the tree**.

**New imports:**
```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
```

**New code added after prediction:**
```python
plt.figure(figsize=(12, 8))
plot_tree(model, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.title("Marvellous Decision Tree Classifier")
plt.show()
```

**What each parameter does:**

| Parameter | Value | Effect |
|-----------|-------|--------|
| `figsize=(12, 8)` | Width 12, Height 8 inches | Large enough to read all branches clearly |
| `filled=True` | Enabled | Nodes are colored by majority class — deeper color = purer node |
| `feature_names=iris.feature_names` | `['sepal length (cm)', ...]` | Shows actual feature names instead of `X[0]`, `X[1]` |
| `class_names=iris.target_names` | `['setosa', 'versicolor', 'virginica']` | Shows class names instead of `0`, `1`, `2` |
| `plt.title(...)` | `"Marvellous Decision Tree Classifier"` | Labels the plot |

**Why this matters:**

Decision Trees are uniquely interpretable — unlike KNN or Neural Networks, the trained model can be read like a flowchart. `plot_tree` makes this explicit: every split condition, every class count at each node, every leaf prediction is visible in one diagram.

Running `DecisionTree2.py` produces an image that answers: *why* did the model predict what it predicted? For any test point, you can trace its path from root to leaf and see exactly which conditions led to the classification.

---

## What Happens Inside `model.fit()`

When `DecisionTreeClassifier().fit(X_train, Y_train)` runs, the algorithm:

1. Looks at all 4 features and all possible split thresholds
2. Picks the split that minimizes **Gini impurity** — the one that creates the two "purest" groups
3. Recursively repeats on each sub-group until leaves are pure or the tree hits its limits
4. Stores the final structure as a tree of nodes

This is the opposite of KNN's `fit()` — KNN's fit does nothing (just stores data). Decision Tree's `fit()` does all the heavy lifting upfront.

---

## Gini Impurity — The Math Behind the Splits

At each node, the algorithm calculates Gini impurity to decide where to split:

```
Gini = 1 - Σ(pᵢ²)
```

Where `pᵢ` is the fraction of samples belonging to class `i` at that node.

- **Gini = 0** → perfectly pure node (all one class) → leaf, no more splitting needed
- **Gini = 0.5** → most mixed possible (50/50 split between 2 classes) → needs splitting

The algorithm always picks the feature + threshold combination that **minimizes** the weighted average Gini of the resulting two child nodes.

---

## Observations & Reflections

- The biggest insight from this topic: a Decision Tree makes its reasoning **visible**. After training, `plot_tree` shows exactly which features matter most and what thresholds the algorithm found — you can read it like a flowchart. No other ML algorithm studied so far gives you this level of transparency.

- `DecisionTreeClassifier()` with default settings works surprisingly well on the Iris dataset (90–100% accuracy with no tuning at all). This is because the Iris classes are fairly well separated in feature space — the tree can find clean boundaries.

- The `filled=True` parameter in `plot_tree` is worth remembering. The color depth shows how confident each node is — a deeply colored node is nearly pure, while a pale node is still mixed. This gives instant visual feedback on which parts of the tree are doing the most work.

- Without `random_state`, both files produce slightly different accuracy numbers each run because the train/test split changes. This is the same lesson learned in KNN1.py vs KNN2.py — reproducibility is not automatic.

- The title `"Marvellous Decision Tree Classifier"` on the plot is a small touch that feels intentional — it connects to the `MarvellousKNeighborsClassifier` naming style from the UserDefinedKNN files. Consistent naming shows ownership of the work.

---

## What's Next

- Add `random_state=42` to `train_test_split` to make results reproducible.
- Experiment with `max_depth` — limit the tree depth to prevent overfitting (e.g., `DecisionTreeClassifier(max_depth=3)`).
- Compare `criterion='gini'` vs `criterion='entropy'` — two different ways to measure split quality.
- Use `export_text(model, feature_names=iris.feature_names)` to print the tree as readable text rules instead of a graph.
- Try the model on a noisier dataset to see where an unlimited-depth tree starts to overfit.

---

## Files

| File | What It Does | Key Concept |
|------|-------------|-------------|
| `DecisionTree1.py` | Train + predict + print accuracy | Basic DecisionTreeClassifier pipeline |
| `DecisionTree2.py` | Adds `plot_tree` visualization | Seeing the tree — interpretability |


*OJT Report — Machine Learning Track | Topic: Decision Tree Classifier*
*Trainee: Shubhada A. Palwe*
