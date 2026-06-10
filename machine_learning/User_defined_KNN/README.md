# On-the-Job Training (OJT) Report

---

| Field               | Details                                                    |
|---------------------|------------------------------------------------------------|
| **Trainee Name**    | Shubhada A. Palwe                                          |
| **Training Domain** | Machine Learning — Algorithm Design from Scratch           |
| **Topic**           | User-Defined KNN — Building `KNeighborsClassifier` by Hand |
| **Tools Used**      | Python (pure), `math` module, `numpy` (imported)           |
| **Total Files**     | 6 (UserDefinedKNN1.py → UserDefinedKNN6.py)               |
| **Achievement**     | Full KNN classifier written without any ML library         |

---

## The Challenge

Most ML tutorials say: *"import sklearn and call `.fit()`."*

This set of files takes the opposite path.

Instead of using a library, the goal here was to **build `KNeighborsClassifier` from zero** — understanding every internal step by implementing it manually in Python. The function is even named `MarvellousKNeighborsClassifier` — a deliberate mirror of sklearn's class name, but this one is 100% hand-built.

---

## The Problem Being Solved

All 6 files work on the same toy problem — a 2D classification task:

```
Training Data:
  Point A → (1, 2) → Red
  Point B → (2, 3) → Red
  Point C → (3, 1) → Blue
  Point D → (5, 6) → Blue

New Point: P(3, 3) → Predict its class
```

This is the same worked example from the KNN reference PDF. Building it from code (instead of just working through it on paper) is the real test of understanding.

---

## The 6-File Journey: Building KNN One Step at a Time

This is not 6 separate programs. It is **one program built incrementally** — each file adds exactly one new piece of the algorithm. Reading them in order shows how an algorithm is assembled, debugged, and completed.

---

### `UserDefinedKNN1.py` — Step 0: Data Structure

**What was added:** The skeleton of the function + the training data.

```python
def MarvellousKNeighborsClassifier():
    data = [
        {'point':'A', 'X':1, 'Y':2, 'label':'Red'},
        {'point':'B', 'X':2, 'Y':3, 'label':'Red'},
        {'point':'C', 'X':3, 'Y':1, 'label':'Blue'},
        {'point':'D', 'X':5, 'Y':6, 'label':'Blue'}
    ]
    for i in data:
        print(i)
```

**What this establishes:**
- Each training point is stored as a Python dictionary (key → value).
- This is exactly what sklearn does internally — it stores the training data and refers to it later.
- The `border = "-"*40` string creates clean console output — a small but professional formatting detail.

**What was learned:** KNN "training" is nothing more than storing data. There is no formula being learned, no weights being adjusted. Just: save the data and wait.

---

### `UserDefinedKNN2.py` — Step 1: The Distance Function (First Attempt)

**What was added:** `EucDistance(P1, P2)` — the mathematical heart of KNN.

```python
import math

def EucDistance(P1, P2):
    return math.sqrt(((P1['X'] - P2['X']) ** 2) + ((P1['Y'] - P2['Y']) ** 2))
    return Ans   # ← unreachable line
```

Also added: the test point and a first call to `EucDistance`.

```python
new_point = {'X': 3, 'Y': 3}
Result = EucDistance(data[0], new_point)
```

**Honest observation — there is a bug here:**
Line 11 is `return Ans` — but `Ans` was never assigned. The code still works because the `return math.sqrt(...)` on line 10 exits the function before reaching line 11. The second `return Ans` is unreachable dead code. This is a real debugging moment — noticing that the function runs correctly but contains a logic error that would crash if the first return were ever removed.

**What was learned:**
- Euclidean distance translates directly to Python: `(x2-x1)² + (y2-y1)²` → `math.sqrt(...)`.
- Testing the function on a single point (`data[0]`) before applying it to all points is good incremental development practice.

---

### `UserDefinedKNN3.py` — Step 2: Distance Calculation Loop (with a Print Bug)

**What was added:** A loop that calculates the distance from every training point to the new point.

```python
for d in data:
    d['distance'] = EucDistance(d, new_point)
    print(border)
    print("Calculated distance are:")
    print(border)
```

**The `EucDistance` function was also fixed:**
```python
def EucDistance(P1, P2):
    Ans = math.sqrt(((P1['X'] - P2['X']) ** 2) + ((P1['Y'] - P2['Y']) ** 2))
    return Ans   # ← now correct: Ans is assigned first, then returned
```

**Honest observation — another bug:**
The `print(border)` lines are **inside** the loop, so the "Calculated distance are:" header prints once per data point (4 times total) instead of once. The distances themselves aren't even being printed here yet. This shows the process of figuring out where print statements belong — inside vs outside a loop.

**The key technical step:**
```python
d['distance'] = EucDistance(d, new_point)
```
This adds a new key `'distance'` to each dictionary in-place — Python's mutability at work. After the loop, every item in `data` has a new `'distance'` field.

---

### `UserDefinedKNN4.py` — Step 3: Sorting + Clean Output

**What was added:** Fixed print positions, printed distances, and added sorting.

```python
# print distances (now outside the loop — correct)
for d in data:
    print(d)

# sort by distance — ascending
sorted_data = sorted(data, key=lambda item: item['distance'])

for d in sorted_data:
    print(d)
```

**This is the moment the algorithm becomes visible.** Running this file shows:
```
{'point': 'B', 'X': 2, 'Y': 3, 'label': 'Red',  'distance': 1.0}
{'point': 'C', 'X': 3, 'Y': 1, 'label': 'Blue', 'distance': 2.0}
{'point': 'A', 'X': 1, 'Y': 2, 'label': 'Red',  'distance': 2.236...}
{'point': 'D', 'X': 5, 'Y': 6, 'label': 'Blue', 'distance': 3.606...}
```

**What was learned:**
- `sorted(data, key=lambda item: item['distance'])` — using a lambda to sort a list of dictionaries by a specific key. This is a Python pattern worth remembering.
- The sorted output matches exactly what was calculated by hand in the reference PDF. The code is doing the same math — just faster.

---

### `UserDefinedKNN5.py` — Step 4: K Selection + Voting

**What was added:** Selecting the K nearest neighbors and counting votes.

```python
k = 3
nearest = sorted_data[:k]   # take the 3 closest

# voting
votes = {}
for neighbour in nearest:
    label = neighbour['label']
    votes[label] = votes.get(label, 0) + 1

for d in votes:
    print("Name:", d, "  Number of votes:", votes[d])
```

**Output at this stage:**
```
Name: Red    Number of votes: 2
Name: Blue   Number of votes: 1
```

**What was learned:**

`votes.get(label, 0) + 1` is an elegant pattern. Instead of checking "does this key exist?" with an if-statement, `.get(label, 0)` returns the current count (or 0 if the label hasn't been seen yet). Then +1 increments it. One line replaces what could have been 3–4 lines of if-else code.

The voting result matches the manual calculation: B (Red), C (Blue), A (Red) → Red wins 2–1.

At this point, the algorithm is 95% complete. The winner is known but not yet extracted from the `votes` dictionary programmatically.

---

### `UserDefinedKNN6.py` — Step 5: The Final Prediction

**What was added:** One line that completes the entire classifier.

```python
predicted_class = max(votes, key=votes.get)
print("Predicted class of (3,3) is:", predicted_class)
```

**Output:**
```
Predicted class of (3,3) is: Red
```

**This one line is the final answer.**

`max(votes, key=votes.get)` finds the key in the `votes` dictionary with the highest value — the class with the most votes. This is Python's built-in `max()` used with a key function — the same pattern as `sorted(data, key=lambda item: item['distance'])`, just applied differently.

The result matches the PDF's worked example perfectly.

---

## The Complete Algorithm: What Was Built

By the end of `UserDefinedKNN6.py`, the following fully functional KNN classifier exists:

| Step | Code | What It Does |
|------|------|--------------|
| Store data | `data = [{'X':..., 'Y':..., 'label':...}, ...]` | The "training" phase — just memory |
| Define distance | `EucDistance(P1, P2)` using `math.sqrt` | Euclidean formula in 2D |
| Calculate all distances | `d['distance'] = EucDistance(d, new_point)` | Distance from test point to every training point |
| Sort distances | `sorted(data, key=lambda item: item['distance'])` | Rank nearest to farthest |
| Select K | `nearest = sorted_data[:k]` | Keep only the K closest |
| Vote | `votes.get(label, 0) + 1` | Count votes per class |
| Predict | `max(votes, key=votes.get)` | Return the majority class |

This is exactly what `sklearn.neighbors.KNeighborsClassifier` does internally — written out explicitly, step by step.

---

## Comparing `MarvellousKNeighborsClassifier` to `sklearn`

| Operation | User-Defined Code | sklearn Equivalent |
|-----------|------------------|--------------------|
| Store training data | `data = [{'X':..., 'label':...}]` | `model.fit(X_train, Y_train)` |
| Euclidean distance | `math.sqrt(...)` manually | Computed internally |
| Sort distances | `sorted(data, key=lambda ...)` | Computed internally |
| Select K nearest | `sorted_data[:k]` | Computed internally |
| Majority vote | `votes.get(label, 0) + 1` | Computed internally |
| Return prediction | `max(votes, key=votes.get)` | `model.predict(X_test)` |

sklearn wraps all 6 steps into 3 lines. The user-defined version makes all 6 steps visible — which is the entire point of building it by hand.

---

## Bugs Found and Fixed During Development

Learning to code means learning to debug. These are the real mistakes that appeared in these files:

| File | Bug | Effect | Fix |
|------|-----|--------|-----|
| `UserDefinedKNN2.py` | `return Ans` after `return math.sqrt(...)` — `Ans` is never assigned | Unreachable dead code; `Ans` would cause `NameError` if reached | Fixed in File 3: assign `Ans = math.sqrt(...)`, then `return Ans` |
| `UserDefinedKNN3.py` | Print statements inside the `for d in data` loop | "Calculated distance are:" prints 4 times instead of once | Fixed in File 4: moved print statements outside the loop |

These are exactly the kinds of bugs every programmer encounters. Finding them is not a failure — it is the learning process itself.

---

## What the Function Name Tells Us

The function is called `MarvellousKNeighborsClassifier`.

sklearn's class is called `KNeighborsClassifier`.

Naming it this way is a deliberate choice — it shows that the goal was to build something equivalent to sklearn, not just write a script. The function is structured the same way (takes training data, has a test point, returns a prediction), named the same way, and produces the same result.

That confidence — giving your own code a name that stands alongside a professional library — is worth noting.

---

## Key Python Concepts Demonstrated

| Concept | Where It Appears |
|---------|-----------------|
| List of dictionaries | Training data structure |
| Dictionary mutation (`d['distance'] = ...`) | Adding computed fields to existing dicts |
| `math.sqrt()` | Euclidean distance calculation |
| `sorted(list, key=lambda ...)` | Sorting dicts by a computed field |
| List slicing (`[:k]`) | Selecting K nearest neighbors |
| `dict.get(key, default)` | Safe vote counting without if-else |
| `max(dict, key=dict.get)` | Finding the key with the highest value |

---

## Observations & Reflections

- Building this algorithm from scratch took 6 iterations — and that feels exactly right. Real development is not "write the perfect solution on the first try." It is: get something running, add the next piece, find the bug, fix it, add the next piece.

- The moment of greatest satisfaction was in `UserDefinedKNN4.py` — when the sorted distances printed in order (B=1.0, C=2.0, A=2.24, D=3.61) and matched the PDF's manual calculation exactly. The math worked. The code worked. They agreed.

- `max(votes, key=votes.get)` in `UserDefinedKNN6.py` is one of the best Python lines in this entire OJT series. It is short, readable, and exactly right. Finding that pattern felt like unlocking a better way to think.

- After doing this, using sklearn's `KNeighborsClassifier` feels very different. Before: it was a black box. Now: every line of `model.fit()` and `model.predict()` maps to a specific piece of code that was written by hand.

---

## What's Next

- Extend `MarvellousKNeighborsClassifier` to work with **n-dimensional data** (not just X, Y — but any number of features like the Iris dataset's 4 features).
- Add support for a **configurable K** passed as a parameter rather than hardcoded as `k=3`.
- Test with **different distance metrics** — Manhattan distance (`|x2-x1| + |y2-y1|`) instead of Euclidean.
- Apply the user-defined classifier to the **Iris dataset** (150 points, 4 features, 3 classes) and compare its accuracy to sklearn's `KNeighborsClassifier`.

---

## Files

| File | What It Adds | KNN Step |
|------|-------------|----------|
| `UserDefinedKNN1.py` | Data structure + print training data | Store data (training phase) |
| `UserDefinedKNN2.py` | `EucDistance()` function — first attempt | Distance formula |
| `UserDefinedKNN3.py` | Distance calculation loop + fix `EucDistance` | Calculate all distances |
| `UserDefinedKNN4.py` | Fixed print positions + sort by distance | Sort + rank |
| `UserDefinedKNN5.py` | K selection + majority voting | Voting |
| `UserDefinedKNN6.py` | `max(votes, key=votes.get)` — final prediction | **Complete classifier** |

---



*OJT Report — Machine Learning Track | Topic: User-Defined KNN*
*Trainee: Shubhada A. Palwe*
*Achievement: KNN classifier built from first principles — no sklearn, no shortcuts*
