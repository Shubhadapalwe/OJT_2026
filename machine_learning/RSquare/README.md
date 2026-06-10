# On-the-Job Training (OJT) Report

---

| Field               | Details                                                        |
|---------------------|----------------------------------------------------------------|
| **Trainee Name**    | Shubhada A. Palwe                                              |
| **Training Domain** | Machine Learning — Regression Evaluation                       |
| **Topic**           | R² (R-Squared) — Measuring How Good a Regression Line Is      |
| **Tools Used**      | Python, scikit-learn (`r2_score`)                              |
| **Total Files**     | 3 (Rsquare.py · RsquareX.py · RsquareXX.py)                  |
| **Connects To**     | `linear_regression/` — this is the planned next step from that topic |

---

## Why These Files Exist

At the end of `LinearRegressionGraphically.py`, three comments were left as future work:

```python
# logic to cal : R square
# logic for cal : yp
# for yp : same as
```

These 3 files fulfill that promise. They explore **R² (R-squared)** — the metric that answers the question the regression line alone cannot: *"How well does this line actually fit the data?"*

The same dataset from the linear regression topic is used throughout:

```
Y_actual = [3, 4, 2, 4, 5]
```

And the predicted values from the regression line `Y = 0.4X + 2.4` built in that topic:

```
For X = [1, 2, 3, 4, 5]:
Yp = [0.4×1+2.4, 0.4×2+2.4, 0.4×3+2.4, 0.4×4+2.4, 0.4×5+2.4]
   = [2.8,       3.2,        3.6,        4.0,        4.4]
```

These predicted values appear directly in `Rsquare.py` — confirming the connection between the two topics.

---

## What is R²?

After building a regression line, two natural questions arise:
1. How close are the predictions to the actual values?
2. Is this model *better* than just guessing the average every time?

R² answers both at once. Its formula is:

```
R² = 1 - (SS_res / SS_tot)

where:
  SS_res = Σ (Yᵢ - Ŷᵢ)²     ← sum of squared residuals (prediction errors)
  SS_tot = Σ (Yᵢ - Ȳ)²      ← total variance in Y (how spread out Y is)
  Ȳ      = mean of Y_actual
```

**Interpreting R²:**

| R² Value | Meaning |
|----------|---------|
| **1.0** | Perfect — predictions match actual values exactly |
| **0.7–0.9** | Strong — model explains most of the variance |
| **0.3–0.6** | Moderate — model has some predictive power |
| **0.0** | Useless — model is no better than predicting the mean every time |
| **Negative** | Worse than useless — model is worse than just guessing the mean |

The key insight: **R² can be negative.** A regression line that fits the data very poorly will have a larger SS_res than SS_tot — giving R² < 0. This means the model performs worse than the baseline of always predicting Ȳ.

---

## The 3-File Experiment

The 3 files use the same code structure — only `Y_predicted` changes. This makes R² the only variable, so its behavior is clear.

---

### `Rsquare.py` — The Regression Line Predictions

```python
Y_actual    = [3, 4, 2, 4, 5]
Y_predicted = [2.8, 3.2, 3.6, 4.0, 4.4]   # from Y = 0.4X + 2.4
```

These are the actual predicted values from the regression line built in the `linear_regression` topic. The experiment checks: "How good was the line we built?"

**Manual calculation:**

Ȳ = (3+4+2+4+5) / 5 = **3.6**

SS_tot = (3−3.6)² + (4−3.6)² + (2−3.6)² + (4−3.6)² + (5−3.6)²
       = 0.36 + 0.16 + 2.56 + 0.16 + 1.96 = **5.20**

SS_res = (3−2.8)² + (4−3.2)² + (2−3.6)² + (4−4.0)² + (5−4.4)²
       = 0.04 + 0.64 + 2.56 + 0.00 + 0.36 = **3.60**

```
R² = 1 - (3.60 / 5.20) = 1 - 0.692 ≈ 0.31
```

**Output:** `R Square value: 0.307...`

The regression line explains about **31% of the variance** in Y. That is a modest fit — the dataset Y = [3, 4, 2, 4, 5] has an outlier-like dip at X=3 (Y=2 when the trend suggests Y should be around 3.6), which reduces the line's fit significantly.

---

### `RsquareX.py` — Bad Predictions

```python
Y_actual    = [3, 4, 2, 4, 5]
Y_predicted = [1.8, 1.2, 3.6, 1.0, 2.4]   # poor, misaligned predictions
```

This experiment deliberately uses predictions that do not follow the data's pattern.

**Manual calculation:**

SS_res = (3−1.8)² + (4−1.2)² + (2−3.6)² + (4−1.0)² + (5−2.4)²
       = 1.44 + 7.84 + 2.56 + 9.00 + 6.76 = **27.60**

```
R² = 1 - (27.60 / 5.20) = 1 - 5.31 ≈ −4.31
```

**Output:** `R Square value: -4.307...`

R² is **negative**. This means the bad predictions have a total squared error (27.60) that is more than 5 times larger than the total variance in Y (5.20). The model is actively harmful — it would be better to always predict the mean (3.6) for every value than to use these predictions.

This negative result is one of the most important things R² can show: a model that looks like it's "doing something" can actually be worse than no model at all.

---

### `RsquareXX.py` — Perfect Predictions

```python
Y_actual    = [3, 4, 2, 4, 5]
Y_predicted = [3, 4, 2, 4, 5]   # identical to actual
```

This experiment sets `Y_predicted = Y_actual` — the theoretical upper bound of any regression model.

**Manual calculation:**

SS_res = (3−3)² + (4−4)² + (2−2)² + (4−4)² + (5−5)² = **0**

```
R² = 1 - (0 / 5.20) = 1 - 0 = 1.0
```

**Output:** `R Square value: 1.0`

R² = 1.0 means every actual value was predicted exactly. No error. In practice, R² = 1.0 is only achievable when the data has no noise and the model captures the exact underlying pattern — which almost never happens on real-world data. It exists here as a reference point: 1.0 is the ceiling.

---

## Comparing All Three Files

| File | Y_predicted | SS_res | R² | What It Shows |
|------|------------|--------|----|--------------|
| `RsquareXX.py` | Perfect = Y_actual | 0 | **1.0** | Upper bound — theoretical maximum |
| `Rsquare.py` | From regression line | 3.60 | **~0.31** | Actual line quality — moderate fit |
| `RsquareX.py` | Poor predictions | 27.60 | **~−4.31** | Lower bound — model worse than baseline |

Reading these three in this order gives a complete map of what R² can look like in practice.

---

## The Code Pattern — Same Structure, Different Data

All three files are identical in structure:

```python
from sklearn.metrics import r2_score

def main():
    Y_actual    = [...]    # same across all files
    Y_predicted = [...]    # only this changes

    r2 = r2_score(Y_actual, Y_predicted)
    print("Actual value : Y ", Y_actual)
    print("Predicted values : yp ", Y_predicted)
    print("R Square value : ", r2)
```

**What `r2_score(Y_actual, Y_predicted)` does internally:**
1. Calculates Ȳ (mean of Y_actual)
2. Calculates SS_tot = Σ(Yᵢ − Ȳ)²
3. Calculates SS_res = Σ(Yᵢ − Ŷᵢ)²
4. Returns `1 − (SS_res / SS_tot)`

This is the same formula that would have been built manually if the `# logic to cal: R square` comment in the previous topic had been continued by hand.

---

## R² vs Accuracy — Different Metrics for Different Problems

| | Accuracy | R² |
|--|----------|----|
| **Used for** | Classification | Regression |
| **Range** | 0% to 100% | −∞ to 1.0 |
| **Can be negative?** | No | Yes |
| **Perfect score** | 100% | 1.0 |
| **"No better than guessing"** | Same as class frequency | R² = 0 |
| **Example** | Iris classifier: 96% correct | Wine regression: R²=0.85 |

Both measure model quality — but in fundamentally different ways because the problems are different. Classification asks "right or wrong?" Regression asks "how far off?"

---

## Observations & Reflections

- The most surprising result was `RsquareX.py` — R² = −4.31. A negative R² feels counterintuitive at first. How can a metric go below zero? The formula makes it clear: if SS_res > SS_tot, the model's errors are larger than the variance in Y itself. The model is making things *worse*, not better. This is not just bad — it is actively misleading.

- `RsquareXX.py` (R² = 1.0) was expected, but it's still a useful anchor. It confirms the formula works correctly: zero error → R² = 1.0. In real projects, if R² ever comes out to exactly 1.0, it usually means the model was trained and tested on the same data (overfitting), not that it's actually perfect.

- `Rsquare.py` using the exact predicted values from the linear regression line (2.8, 3.2, 3.6, 4.0, 4.4) connects this topic directly back to the previous one. The line was built there; the quality of that line is measured here. R² ≈ 0.31 tells us the line from `LinearRegression2.py` is not a particularly strong fit — the data point Y=2 at X=3 doesn't follow the upward trend well, which reduces the score.

- Using `sklearn.metrics.r2_score` here, after computing slope and intercept manually in the previous topic, shows the natural progression: understand the formula by building it, then use the library function to apply it efficiently.

---

## What's Next

- Compute R² **manually** (without sklearn) to verify the formula: `1 - SS_res/SS_tot` — matching the pattern from `MarvellousPredictor`.
- Try improving the regression line on this dataset by adding a second-degree term (polynomial regression) and comparing R² values.
- Apply R² to a larger regression dataset (e.g., California Housing) to see how R² behaves at scale.
- Understand **Adjusted R²** — a version that penalizes adding unnecessary features, used in multiple linear regression.

---

## Files

| File | Y_predicted | R² Result | Concept Demonstrated |
|------|------------|-----------|---------------------|
| `Rsquare.py` | `[2.8, 3.2, 3.6, 4.0, 4.4]` (regression line) | ~0.31 | Evaluating the actual linear regression built previously |
| `RsquareX.py` | `[1.8, 1.2, 3.6, 1.0, 2.4]` (bad predictions) | ~−4.31 | R² can be negative — model worse than baseline |
| `RsquareXX.py` | `[3, 4, 2, 4, 5]` (perfect predictions) | 1.0 | R² upper bound — theoretical perfect fit |





*OJT Report — Machine Learning Track | Topic: R-Squared Evaluation Metric*
*Trainee: Shubhada A. Palwe*
*Connects to: `linear_regression/` — measures the line built in that topic*
