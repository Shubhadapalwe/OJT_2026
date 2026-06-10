# On-the-Job Training (OJT) Report

---

| Field               | Details                                              |
|---------------------|------------------------------------------------------|
| **Trainee Name**    | Shubhada A. Palwe                                    |
| **Training Domain** | Machine Learning — Supervised Classification         |
| **Topic**           | K-Nearest Neighbors (KNN) Algorithm                  |
| **Tools Used**      | Python, scikit-learn, math (built-in)                |
| **Total Files**     | 2 (KNN1.py · KNN2.py)                               |
| **Reference Doc**   | User Defined Implementation of KNN with Numeric Calculations.pdf |
                                    |

---

## Objective

To understand and implement the **K-Nearest Neighbors (KNN)** algorithm — first by working through the **mathematics by hand** (as shown in the reference PDF), and then by applying it using **scikit-learn** on the Iris dataset. The two Python files show a progression: basic implementation → hyperparameter experiment.

---

## What is KNN?

> *"KNN (K-Nearest Neighbors) is a simple, supervised machine learning algorithm used for both classification and regression tasks. However, it is more widely used for classification problems."*
> — Reference: *User Defined Implementation of KNN with Numeric Calculations*

KNN works differently from all other algorithms studied so far (like Decision Tree). There is **no model training** step. Instead:

- KNN **stores all training data** as-is.
- When a new point arrives, it **calculates the distance** to every stored point.
- It picks the **K closest** (nearest) points.
- It predicts the class by **majority vote** among those K neighbors.

This is why KNN is called a **lazy learner** — it does no work at training time, all work happens at prediction time.

---

## KNN vs Decision Tree — Key Difference

| Property | Decision Tree | KNN |
|----------|--------------|-----|
| Training | Builds a tree structure | Stores all data (no learning) |
| Prediction | Follows if/else rules | Calculates distances to all points |
| Speed (Training) | Slower | Instant |
| Speed (Prediction) | Fast | Slower (calculates all distances) |
| Type | Model-based | Instance-based / Lazy learner |

---

## How KNN Works — Step by Step

*(As described in the reference PDF)*

**Step 1** → Choose the number **K** (how many nearest neighbors to consider).

**Step 2** → For the new point, **calculate the distance** to every training point using Euclidean Distance.

**Step 3** → **Sort** all distances in ascending order. Pick the **K smallest** ones.

**Step 4** → **Majority vote** — whichever class appears most among the K neighbors is the predicted class.

---

## The Mathematics: Euclidean Distance

*(Source: Reference PDF — Mathematical Formula section)*

Euclidean Distance is the straight-line distance between two points — like using a ruler on a graph.

**For 2 dimensions (x, y):**

```
Distance = √( (x2 - x1)² + (y2 - y1)² )
```

**For n dimensions (general ML case):**

```
Distance = √( Σ (xi - yi)² )    for i = 1 to n
```

In the Iris dataset, each flower has 4 features → we calculate distance in **4-dimensional space**. The formula is the same, just with 4 terms under the square root instead of 2.

---

## Worked Example from the Reference PDF

*(Fully worked out with numbers — from the reference document)*

### Problem
Classify a new point **P(3, 3)** using **K = 3** from the following training data:

| Point | Coordinates (x, y) | Class |
|-------|-------------------|-------|
| A | (1, 2) | Red |
| B | (2, 3) | Red |
| C | (3, 1) | Blue |
| D | (6, 5) | Blue |

---

### Step 1: Calculate Euclidean Distance from P(3,3) to each point

**Distance to A(1, 2):**
```
d(P,A) = √( (3-1)² + (3-2)² )
       = √( 4 + 1 )
       = √5
       ≈ 2.24
```

**Distance to B(2, 3):**
```
d(P,B) = √( (3-2)² + (3-3)² )
       = √( 1 + 0 )
       = √1
       = 1.00
```

**Distance to C(3, 1):**
```
d(P,C) = √( (3-3)² + (3-1)² )
       = √( 0 + 4 )
       = √4
       = 2.00
```

**Distance to D(6, 5):**
```
d(P,D) = √( (3-6)² + (3-5)² )
       = √( 9 + 4 )
       = √13
       ≈ 3.61
```

---

### Step 2: Sort Distances (Ascending) and Pick K=3 Nearest

| Rank | Point | Distance | Class |
|------|-------|----------|-------|
| 1st (nearest) | B | 1.00 | Red |
| 2nd | C | 2.00 | Blue |
| 3rd | A | 2.24 | Red |
| 4th (excluded) | D | 3.61 | Blue |

K = 3 → We take the **top 3**: B, C, A

---

### Step 3: Majority Vote

| Class | Votes | Points |
|-------|-------|--------|
| Red | 2 | B, A |
| Blue | 1 | C |

**Red has more votes → Predicted class for P(3,3) = RED**

---

## User-Defined KNN Code (from PDF)

Before using sklearn, the reference PDF shows how to build KNN **from scratch in Python** — so we understand what sklearn is doing internally:

```python
import math

# Step 1: Euclidean distance function
def EucDistance(P1, P2):
    return math.sqrt((P1['x'] - P2['x'])**2 + (P1['y'] - P2['y'])**2)

def MarvellousKNN():
    # Step 0: Training data stored as list of dicts
    data = [
        {'point': 'A', 'x': 1, 'y': 2, 'label': 'Red'},
        {'point': 'B', 'x': 2, 'y': 3, 'label': 'Red'},
        {'point': 'C', 'x': 3, 'y': 1, 'label': 'Blue'},
        {'point': 'D', 'x': 6, 'y': 5, 'label': 'Blue'}
    ]

    new_point = {'x': 3, 'y': 3}   # point to classify

    # Step 1: Calculate distance from new_point to every training point
    for d in data:
        d['distance'] = EucDistance(d, new_point)

    # Step 2: Sort by distance (ascending)
    sorted_data = sorted(data, key=lambda item: item['distance'])

    # Step 3: Select K=3 nearest neighbors
    K = 3
    k_nearest = sorted_data[:K]

    # Step 4: Majority voting
    red  = sum(1 for i in k_nearest if i['label'] == 'Red')
    blue = sum(1 for i in k_nearest if i['label'] == 'Blue')

    if red > blue:
        print("Predicted class = Red")
    else:
        print("Predicted class = Blue")

MarvellousKNN()
```

**What each part does:**

| Code Section | What It Does | Equivalent in sklearn |
|---|---|---|
| `data = [...]` | Store all training points | `model.fit(X_train, Y_train)` |
| `EucDistance(P1, P2)` | Calculate distance manually | Done internally by KNN |
| `sorted(data, key=...)` | Sort by distance | Done internally |
| `sorted_data[:K]` | Pick K nearest | Done internally |
| `sum(1 for i in ...)` | Count votes per class | Done internally |
| `if red > blue` | Majority decision | `model.predict()` |

**Key learning:** sklearn's `KNeighborsClassifier` does all of this automatically — but the logic is exactly the same as the manual code above.

---

## sklearn Implementation — File by File

---

### `KNN1.py` — Basic KNN with K=3

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data      # shape (150, 4) — 4 features
Y = iris.target    # shape (150,)  — 3 class labels (0,1,2)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy is:", accuracy * 100)
```

**What I learned:**
- `KNeighborsClassifier(n_neighbors=3)` → K=3 means the model looks at 3 closest training points to vote.
- `model.fit()` here doesn't build a tree — it just **stores** the training data.
- `model.predict()` is when the real work happens: distances are calculated for each test point against all 120 training points.
- No `random_state` here → split is different every run → accuracy will vary slightly each time.

**Note:** Without `random_state`, this file gives a **different accuracy every run** — which makes it hard to evaluate or compare results.

---

### `KNN2.py` — Improved KNN with K=7 and random_state

```python
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42   # fixed split
)

model = KNeighborsClassifier(n_neighbors=7)   # K changed to 7
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy is:", accuracy * 100)
```

**What changed from KNN1.py:**

| Setting | KNN1.py | KNN2.py | Effect |
|---------|---------|---------|--------|
| `n_neighbors` | 3 | 7 | More voters → smoother decision boundary |
| `random_state` | Not set | 42 | Fixed split → reproducible accuracy |

**What I learned about choosing K:**
- **Small K (e.g., K=1 or K=3):** Very sensitive to individual points — can overfit. A single noisy point can change the prediction.
- **Large K (e.g., K=7 or K=11):** Smoother — considers more neighbors, less affected by noise. But too large and the model underfits (looks at irrelevant far-away points).
- **Odd K is preferred** for binary classification — avoids tie votes (e.g., 2 Red, 2 Blue).
- The best K is found through experimentation — this is **hyperparameter tuning**.

---

## KNN Hyperparameter: How K Affects the Model

*(Connecting the PDF math to practical coding)*

Imagine classifying P(3,3) from our worked example with different K values:

| K | Neighbors Selected | Votes | Prediction |
|---|-------------------|-------|------------|
| 1 | B (1.00) | Red:1, Blue:0 | Red |
| 3 | B, C, A | Red:2, Blue:1 | Red |
| 4 | B, C, A, D | Red:2, Blue:2 | **Tie!** (problem) |

This is exactly why **odd K values** are preferred for 2-class problems.

---

## KNN in the ML Pipeline

| ML Step | KNN's Approach |
|---------|---------------|
| Data Collection | Load Iris with `load_iris()` |
| Data Analysis | (Implicit — no EDA in these files) |
| Data Cleaning | Not needed (Iris is pre-cleaned) |
| Model Selection | Choose KNN + select K value |
| Model "Training" | `model.fit()` — just stores data |
| Prediction | `model.predict()` — calculates distances |
| Evaluation | `accuracy_score()` |

---

## Summary of Concepts Learned

| Concept | Explanation |
|---------|-------------|
| Lazy Learner | KNN stores data, does no training — all computation at prediction time |
| Euclidean Distance | `√Σ(xi - yi)²` — straight-line distance in n-dimensional space |
| K (hyperparameter) | Number of nearest neighbors to vote — must be chosen before training |
| Majority Vote | Most common class among K neighbors = predicted class |
| `n_neighbors` | The K value in sklearn's `KNeighborsClassifier` |
| `random_state=42` | Fixes train/test split so results are reproducible |
| Odd K | Preferred to avoid tie votes in binary classification |
| Overfitting (small K) | K=1 memorizes training data, poor on new data |
| Underfitting (large K) | Very large K looks at irrelevant distant points |

---

## Observations & Reflections

- KNN was the most **intuitive** algorithm so far. The idea — "look at your nearest neighbors to make a decision" — is something humans do naturally. If a new food looks, smells, and tastes like an apple, it's probably an apple.
- The reference PDF's manual calculation was extremely helpful. Working through the numbers (distance to A = 2.24, distance to B = 1.00, etc.) made me understand exactly what sklearn is doing inside `model.predict()`.
- The biggest difference from Decision Tree: there is no "model" to inspect or visualize. KNN doesn't learn rules — it memorizes data. This has a cost: predicting on large datasets is slow because distances must be recalculated every time.
- KNN1.py gave different accuracy each run because there's no `random_state`. KNN2.py fixed this. This showed me that reproducibility is not automatic — you have to set `random_state` deliberately.
- Choosing K=7 vs K=3 is a real decision with real consequences. On the Iris dataset, both give high accuracy (~96–100%), but on noisier datasets the difference matters a lot.

---

## What's Next

- Try multiple values of K (1, 3, 5, 7, 9, 11) and plot accuracy vs K to find the optimal value.
- Use `cross_val_score` to get a more reliable accuracy estimate.
- Learn about **Manhattan Distance** and **Minkowski Distance** — other distance metrics KNN can use.
- Apply KNN to a larger, noisier dataset to see where it struggles vs Decision Tree.
- Learn about **feature scaling** (StandardScaler) — KNN is sensitive to features with different units/ranges because distance is affected by scale.

---

## Files

| File | K Value | random_state | Key Concept |
|------|---------|-------------|-------------|
| `KNN1.py` | 3 | Not set (varies) | Basic KNN implementation |
| `KNN2.py` | 7 | 42 (fixed) | Hyperparameter experiment + reproducibility |
| `User Defined Implementation of KNN with Numeric Calculations.pdf` | 3 | Manual | Theory, math formula, step-by-step worked example |

---
---

*OJT Report — Machine Learning Track | Topic: K-Nearest Neighbors (KNN)*
*Trainee: Shubhada A. Palwe*
*Reference: User Defined Implementation of KNN with Numeric Calculations*
