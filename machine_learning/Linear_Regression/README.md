# On-the-Job Training (OJT) Report

---

| Field               | Details                                                       |
|---------------------|---------------------------------------------------------------|
| **Trainee Name**    | Shubhada A. Palwe                                             |
| **Training Domain** | Machine Learning — Supervised Regression                      |
| **Topic**           | Linear Regression — Built from Mathematical First Principles  |
| **Tools Used**      | Python, `numpy`, `matplotlib`, `pandas` (imported)            |
| **Total Files**     | 3 (LinearRegression1.py · LinearRegression2.py · LinearRegressionGraphically.py) |
| **Achievement**     | Full regression line derived manually — no sklearn            |

---

## A New Kind of Problem

Every algorithm studied so far — Decision Tree, KNN, Ball Classification, Iris Classification — was solving a **classification** problem: given some inputs, predict which *category* a data point belongs to (Red/Blue, Setosa/Versicolor, etc.).

Linear Regression solves a different kind of problem: **regression** — given some inputs, predict a *number*.

Instead of asking "which class is this?", it asks: "what value will this have?"

- Classification: *What type of flower is this?* → Setosa / Versicolor / Virginica
- Regression: *What will this house sell for?* → ₹42,00,000

Linear Regression is the simplest regression algorithm. It finds the **best-fit straight line** through the data — and uses that line to predict any future value.

---

## The Problem Being Solved

All 3 files work on the same small dataset:

```
X (Independent variable) = [1, 2, 3, 4, 5]
Y (Dependent variable)   = [3, 4, 2, 4, 5]
```

X is the input (what we know). Y is the output (what we want to predict). The goal is to find a line `Y = mX + C` that fits these 5 points as closely as possible — and then use that line to predict Y for any new X value.

The function is named `MarvellousPredictor` — continuing the naming convention from `MarvellousKNeighborsClassifier` in the UserDefinedKNN series.

---

## The Math: Where Does the Line Come From?

A straight line has two parameters: **slope (m)** and **intercept (C)**.

```
Y = mX + C
```

The best-fit line minimizes the total error between the actual Y values and the line's predicted Y values. The formula that gives this optimal line analytically is called the **Least Squares** formula:

**Slope (m):**
```
m = Σ (Xᵢ - X̄)(Yᵢ - Ȳ)
    ─────────────────────
         Σ (Xᵢ - X̄)²
```

**Intercept (C):**
```
C = Ȳ - m × X̄
```

Where X̄ = mean of X, Ȳ = mean of Y.

These formulas are implemented by hand across the 3 files — no sklearn, no black box.

---

## The 3-File Build: Step by Step

---

### `LinearRegression1.py` — Step 0: Data + Means

**What was added:** The data, and the two means needed for every formula.

```python
X = [1, 2, 3, 4, 5]
Y = [3, 4, 2, 4, 5]

mean_x = np.mean(X)
mean_y = np.mean(Y)

print("X_MEAN is:", mean_x)   # 3.0
print("Y_MEAN is:", mean_y)   # 3.6
```

**Output:**
```
Values of independent variables : X [1, 2, 3, 4, 5]
Values of Dependent variables   : Y [3, 4, 2, 4, 5]
X_MEAN is : 3.0
Y_MEAN is : 3.6
```

**What was established:**

The means X̄ = 3.0 and Ȳ = 3.6 are the anchor of the entire calculation. Every formula for slope and intercept references these two values. Before computing anything, compute the means.

`np.mean()` handles the averaging in one call — but importantly, the expected values are commented right in the code (`# 3.0` and `# 3.6`). This shows that the math was verified by hand before writing the code.

---

### `LinearRegression2.py` — Step 1: Slope and Intercept

**What was added:** The full slope formula implemented as a loop, then the intercept.

```python
n = len(X)   # 5

# Y = mX + C
# formula for m = slope
# m = (Σ (X - X_bar) * (Y - Y_bar)) / (Σ (X - X_bar)²)
numerator   = 0
denominator = 0

for i in range(n):
    numerator   += (X[i] - mean_x) * (Y[i] - mean_y)
    denominator += (X[i] - mean_x) ** 2

m = numerator / denominator
print("Slope of Line i.e., (m) :", m)        # 0.4

C = mean_y - (m * mean_x)
print("Y-intercept of line i.e., C :", C)    # 2.4
```

**The calculation worked through manually:**

| i | Xᵢ | Yᵢ | (Xᵢ - X̄) | (Yᵢ - Ȳ) | (Xᵢ-X̄)(Yᵢ-Ȳ) | (Xᵢ-X̄)² |
|---|----|----|-----------|-----------|----------------|----------|
| 0 | 1  | 3  | -2        | -0.6      | 1.2            | 4        |
| 1 | 2  | 4  | -1        | 0.4       | -0.4           | 1        |
| 2 | 3  | 2  | 0         | -1.6      | 0.0            | 0        |
| 3 | 4  | 4  | 1         | 0.4       | 0.4            | 1        |
| 4 | 5  | 5  | 2         | 1.4       | 2.8            | 4        |
| **Σ** | | | | | **4.0** | **10** |

```
m = 4.0 / 10 = 0.4
C = 3.6 - (0.4 × 3.0) = 3.6 - 1.2 = 2.4
```

**The regression line equation: Y = 0.4X + 2.4**

The comments in the code (`# 0.4` and `# 2.4`) confirm the result was verified against manual calculation before committing.

**What was learned:** The slope formula is a ratio — numerator measures how much X and Y move together (covariance), denominator measures how spread out X is (variance). When both move together, the numerator is large and so is the slope. When X varies a lot but Y doesn't respond, the slope is small.

---

### `LinearRegressionGraphically.py` — Step 2: Visualizing the Line

**What was added:** Generating points on the regression line and plotting everything.

```python
x = np.linspace(1, 6, n)   # yane apan sangto 1 pasun 6 paryant generate kr
                             # (generates evenly spaced numbers from 1 to 6)
y = C + m * x

plt.plot(x, y, color='g', label="regression line")
plt.scatter(X, Y, color='r', label="Scatter plot")

plt.xlabel("X : Independent variables")
plt.ylabel("Y : Dependent Variables")
plt.legend()
plt.show()
```

**What each piece does:**

| Code | Purpose |
|------|---------|
| `np.linspace(1, 6, n)` | Generates 5 evenly spaced x-values from 1 to 6 — smooth line points |
| `y = C + m * x` | Applies the regression equation to all x-values at once (numpy vectorization) |
| `plt.plot(x, y, color='g')` | Draws the green regression line |
| `plt.scatter(X, Y, color='r')` | Draws the original 5 data points as red dots |
| `plt.legend()` | Adds the key (green=line, red=scatter) |

**The resulting plot shows:**
- 5 red dots: the actual data points
- 1 green line: `Y = 0.4X + 2.4` — the best-fit line through those points

The line doesn't pass through every point — it minimizes the *total* squared distance to all points. Some points are above the line, some below. That's expected and correct.

**The Marathi comment** `# yane apan sangto 1 pasun 6 paryant generate kr` translates to "this tells it to generate from 1 to 6" — a note written while learning what `np.linspace` does.

**What's planned next** (visible in the comments at the bottom of the file):
```python
# logic to cal : R square
# logic for cal : yp
# for yp : same as
```

R² (R-squared) measures how well the line fits the data — 1.0 = perfect fit, 0.0 = the line explains nothing. Yp (predicted Y) would apply the equation to generate predictions for new X values. These are the natural next steps.

---

## The Complete Algorithm Summary

By the end of `LinearRegressionGraphically.py`, a complete linear regression pipeline was built from scratch:

| Step | Code | Mathematical Meaning |
|------|------|---------------------|
| Load data | `X = [...]`, `Y = [...]` | Input: independent and dependent variables |
| Calculate means | `np.mean(X)`, `np.mean(Y)` | X̄ = 3.0, Ȳ = 3.6 |
| Calculate slope | `Σ(Xᵢ-X̄)(Yᵢ-Ȳ) / Σ(Xᵢ-X̄)²` | m = 0.4 |
| Calculate intercept | `C = mean_y - m * mean_x` | C = 2.4 |
| Generate line points | `np.linspace()` + `y = C + m*x` | The line Y = 0.4X + 2.4 |
| Visualize | `plt.plot()` + `plt.scatter()` | Green line + red data points |

This is exactly what `sklearn.linear_model.LinearRegression` does internally — written out step by step.

---

## `MarvellousPredictor` vs sklearn's `LinearRegression`

| Operation | User-Defined Code | sklearn Equivalent |
|-----------|------------------|--------------------|
| Store data | `X = [...]`, `Y = [...]` | `model.fit(X, Y)` |
| Calculate slope | `Σ(Xᵢ-X̄)(Yᵢ-Ȳ) / Σ(Xᵢ-X̄)²` | `model.coef_` |
| Calculate intercept | `Ȳ - m × X̄` | `model.intercept_` |
| Predict new values | `y = C + m * x` | `model.predict(X_new)` |
| Evaluate fit | *(R² planned)* | `model.score(X_test, Y_test)` |

---

## Classification vs Regression — The Big Picture

This topic marks a turning point in the OJT:

| | Classification | Regression |
|--|---------------|------------|
| **Output** | A category (Red, Blue, Setosa) | A number (price, score, value) |
| **Example** | "Is this email spam?" | "What will this house cost?" |
| **Measure of success** | Accuracy (% correct) | R² score, Mean Squared Error |
| **Algorithm here** | Decision Tree, KNN | Linear Regression |

---

## Observations & Reflections

- This is the first regression topic in the OJT — and the shift from classification to regression required a different way of thinking. Classification outputs a label; regression outputs a number on a continuous scale. The question changes from "which box?" to "where on the number line?"

- Working through the slope formula by hand (filling in the table row by row) before coding it made the loop feel very natural. When writing `numerator += (X[i] - mean_x) * (Y[i] - mean_y)`, it was clear exactly what each term represents — not just syntax.

- `np.linspace(1, 6, n)` was a new function. The Marathi note in the code (`yane apan sangto 1 pasun 6 paryant generate kr`) captures the exact moment of understanding what it does — generate equally spaced points on a number line for the plot.

- The commented-out next steps (`# logic to cal: R square`, `# logic for cal: yp`) at the bottom of the last file show where this is heading. R² is the natural follow-up — it answers "how good is this line actually?" as a number between 0 and 1.

- Naming the function `MarvellousPredictor` instead of just `linearRegression` or `main` gives it the same personality as `MarvellousKNeighborsClassifier` — it's a user-defined implementation that stands on its own.

---

## What's Next

- Calculate **predicted Y values (Yp)**: for each Xᵢ, compute Ŷᵢ = mXᵢ + C and compare to actual Yᵢ.
- Calculate **R² (R-squared)**: `1 - Σ(Yᵢ - Ŷᵢ)² / Σ(Yᵢ - Ȳ)²` — measures how well the line fits.
- Implement using **sklearn's `LinearRegression`** and compare `model.coef_` and `model.intercept_` to the manually computed m and C.
- Apply to a real dataset (e.g., Boston Housing, California Housing) where X has multiple features — this leads to **Multiple Linear Regression**.
- Add error visualization — draw vertical lines from each point to the regression line to show the residuals.

---

## Files

| File | What It Adds | Output |
|------|-------------|--------|
| `LinearRegression1.py` | Data setup + mean calculation | Prints X, Y, X̄=3.0, Ȳ=3.6 |
| `LinearRegression2.py` | Slope (m=0.4) + Intercept (C=2.4) | Prints the line equation parameters |
| `LinearRegressionGraphically.py` | `np.linspace` + `plt.plot` + `plt.scatter` | Displays regression line over scatter plot |



*OJT Report — Machine Learning Track | Topic: Linear Regression (User-Defined)*
*Trainee: Shubhada A. Palwe*
*Achievement: Regression line derived from Least Squares formula — no sklearn*
