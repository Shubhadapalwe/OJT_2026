# On-the-Job Training (OJT) Report

---

| Field               | Details                                                                   |
|---------------------|---------------------------------------------------------------------------|
| **Trainee Name**    | Shubhada A. Palwe 
                                                        |
| **Training Domain** | Machine Learning — Supervised Regression (Case Study) 
                    |
| **Topic**           | Advertising Sales Predictor — Multiple Linear Regression on 
Business Data |

| **Tools Used**      | Python, pandas, scikit-learn (`LinearRegression`, MSE, R²), matplotlib    |

| **Total Files**     | 5 (Advertising1 → AdvertisingCaseStudyModelbuldingVisualization)    
      |
| **Dataset**         | Advertising.csv — 200 records, 3 ad channels, 1 sales target              |

| **Pipeline Steps**  | 13 (Load → Clean → Missing Values → Stats → Correlation → X/Y Split → Train/Test → Train → Predict → Evaluate → Coefficients → Compare → Visualize) |

---

## The Case Study: A Business Question

Every business that spends money on advertising wants to know one thing: *"Which channel is actually driving sales?"*

A company ran advertising campaigns across three channels — TV, radio, and newspaper — and tracked sales results across 200 markets. The data is in `Advertising.csv`. The challenge: **build a model that predicts sales from advertising spend**, and use it to understand which channel has the most impact.

This is the first **regression case study** in the OJT — the equivalent of the Wine Classifier case study, but for regression instead of classification. Both follow a professional multi-step pipeline. The difference:

| | Wine Classifier (KNN) | Advertising (Linear Regression) |
|--|----------------------|----------------------------------|
| Problem type | Classification | Regression |
| Output | Wine class (1/2/3) | Sales figure (continuous number) |
| Algorithm | KNeighborsClassifier | LinearRegression |
| Evaluation | Accuracy, classification report | MSE, RMSE, R² |

---

## The Dataset: Advertising.csv

| Property | Value |
|----------|-------|
| Rows | 200 market records |
| Features (X) | TV, radio, newspaper (ad spend in $000s) |
| Target (Y) | sales (units sold in 000s) |
| Extra column | `Unnamed: 0` — a row index artifact from CSV export, must be removed |

**Sample data:**

| TV | radio | newspaper | sales |
|----|-------|-----------|-------|
| 230.1 | 37.8 | 69.2 | 22.1 |
| 44.5 | 39.3 | 45.1 | 10.4 |
| 17.2 | 45.9 | 69.3 | 9.3 |
| 151.5 | 41.3 | 58.5 | 18.5 |

**The business interpretation:** Each row is one market. The company spent $230,100 on TV, $37,800 on radio, and $69,200 on newspaper — and sold 22,100 units. The model learns from all 200 markets to find the equation that best predicts sales from spend.

---

## From Simple to Multiple Linear Regression

In the `linear_regression/` topic, the model was:

```
Y = mX + C          (1 feature — single line in 2D)
```

This case study uses **Multiple Linear Regression** — 3 features:

```
sales = (c₁ × TV) + (c₂ × radio) + (c₃ × newspaper) + intercept
```

Instead of a line in 2D space, this model fits a **hyperplane** through 4-dimensional data (TV, radio, newspaper, sales). The logic is the same — minimize squared errors — but sklearn handles the 3-coefficient math automatically. The coefficients `c₁, c₂, c₃` become the key output: they tell us which channel drives sales the most per dollar spent.

---

## The 5-File Build

---

### `Advertising1.py` — Step 0: Load and Clean

```python
df = pd.read_csv("Advertising.csv")
print(df.shape)   # (200, 5) — 5 columns including 'Unnamed: 0'

if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)
print(df.shape)   # (200, 4) — now 4 clean columns
```

**Why the `Unnamed: 0` column exists:** When a pandas DataFrame is saved to CSV without specifying `index=False`, it writes the row index (0, 1, 2...) as the first column. When re-loaded, pandas reads it as a regular data column named `Unnamed: 0`. It carries no information — it must be removed before analysis.

**The defensive check `if 'Unnamed: 0' in df.columns`:** Rather than blindly dropping the column (which would crash if it didn't exist), this check first confirms the column is present. It works whether or not the CSV was exported with an index.

This file is the minimal starting point — just confirming the data loads and the cleanup works.

---

### `Advertising2.py` — Step 1: Quick Model (Exploration)

```python
X = df[['TV', 'radio', 'newspaper']]
Y = df['sales']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

model = LinearRegression()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

print("Coefficient:", model.coef_)
print("Intercept:", model.intercept_)
```

This file skips a formal function structure and goes straight to building the model. It's an exploratory step — "can we get a model running quickly?" — before the proper pipeline is built in the case study files.

**New concepts introduced:**

`model.coef_` — returns an array of 3 coefficients, one per feature. For this dataset, the typical output is approximately:
```
[0.046  0.188  -0.001]
 TV     radio  newspaper
```

Meaning: for every $1,000 extra spent on TV → ~46 more units sold. For every $1,000 on radio → ~188 more units. For newspaper → effectively 0. This is a real business insight.

`model.intercept_` — the sales value when all ad spend is zero. On this dataset, approximately 2.9 (2,900 units baseline).

`test_size=0.1` — only 10% (20 rows) used for testing. This was later changed to 0.2 and then 0.05 across subsequent files as experiments continued.

---

### `AdvertisingCaseStudyAnalysis.py` — Steps 1–5: Real EDA

This file introduces `MarvellousAdvertise(datapath)` — the proper function structure used in all remaining files. It focuses entirely on understanding the data before any model is built.

**Step 3 — Check Missing Values:**
```python
print(df.isnull().sum())
```
Shows count of nulls per column. On this dataset: all zeros (no missing values). But checking is non-negotiable on real datasets — the Wine Classifier used `dropna()`, here we confirm there's nothing to drop.

**Step 4 — Statistical Summary:**
```python
print(df.describe())
```
Prints count, mean, std, min, 25%, 50%, 75%, max for every column. For this dataset:
- TV spend ranges from 0.7 to 296.4 ($000s) — high variance
- Newspaper spend has the highest standard deviation (~21.7)
- Sales range from 1.6 to 27.0

**Step 5 — Correlation Matrix:**
```python
print(df.corr())
```
Shows how strongly each feature is linearly related to every other column (including sales). Key findings:
- TV vs sales: correlation ≈ **0.78** — strong positive relationship
- radio vs sales: correlation ≈ **0.58** — moderate positive
- newspaper vs sales: correlation ≈ **0.23** — weak
- This already predicts what `model.coef_` will show: TV and radio matter most, newspaper least.

---

### `AdvertisingCaseStudyModelbulding.py` — Steps 1–12: Full Pipeline

This is the complete pipeline without visualization. Steps 6–12 are added to the EDA foundation.

**Step 6 — X/Y Split:**
```python
X = df[['TV', 'radio', 'newspaper']]
Y = df['sales']
```
Explicit column selection with a list of column names — more readable than `df.drop(columns=['sales'])`.

**Step 7 — Train/Test Split:**
```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
```
80/20 split — 160 training rows, 40 test rows.

**Bug spotted:** `print("Y_train shape", Y_train.shape)` appears twice in a row. The second should be `print("Y_test shape", Y_test.shape)`. A copy-paste error — harmless for the model but confusing for the output.

**Steps 8–9 — Train and Predict:**
```python
model = LinearRegression()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
```
Unlike `MarvellousPredictor` in the manual regression, sklearn handles the entire Least Squares calculation internally. `model.fit()` computes all 3 coefficients and the intercept simultaneously.

**Step 10 — Three Evaluation Metrics:**
```python
MSE  = mean_squared_error(Y_test, Y_pred)
RMSE = np.sqrt(MSE)
R2   = r2_score(Y_test, Y_pred)
```

Three metrics used together for the first time:

| Metric | Formula | What It Measures | Units |
|--------|---------|-----------------|-------|
| **MSE** | Σ(Yᵢ − Ŷᵢ)² / n | Average squared error | (sales units)² |
| **RMSE** | √MSE | Average error in original units | sales units (000s) |
| **R²** | 1 − SS_res/SS_tot | % variance explained by model | 0 to 1 |

RMSE is the most interpretable: if RMSE ≈ 1.5, the model's predictions are off by about 1,500 units on average. R² ≈ 0.90 on this dataset means the model explains 90% of the variance in sales — a strong fit.

**Step 11 — Coefficients with f-string:**
```python
for column, value in zip(X.columns, model.coef_):
    print(f"{column} : {value}")
print("Intercept:", model.intercept_)
```
`zip(X.columns, model.coef_)` pairs each column name with its coefficient — a clean way to display all 3 together. The f-string `f"{column} : {value}"` formats each pair into a readable line.

**Step 12 — Actual vs Predicted DataFrame:**
```python
Result = pd.DataFrame({
    'Actual sale':    Y_test.values,
    'Predicted sale': Y_pred
})
print(Result.head(10))
```
Side-by-side comparison of actual and predicted values. `Y_test.values` converts the pandas Series to a numpy array so it aligns correctly with `Y_pred`.

---

### `AdvertisingCaseStudyModelbuldingVisualization.py` — Step 13: Scatter Plot

The final file adds the diagnostic visualization and changes `test_size` from 0.2 to 0.05.

**`test_size=0.05` — why so small?**
With 200 rows, 5% = only 10 test samples. This likely happened to make the scatter plot less cluttered while experimenting. In production, 0.2 (40 samples) gives a more reliable evaluation.

**Step 13 — Actual vs Predicted Scatter Plot:**
```python
plt.figure(figsize=(8, 5))
plt.scatter(Y_test, Y_pred)
plt.xlabel("Actual sales")
plt.ylabel("Predicted sales")
plt.title("Actual sales vs predicted sales")
plt.grid(True)
plt.show()
```

This plot is a standard regression diagnostic. Each point represents one test sample — its x-coordinate is the actual sales, its y-coordinate is what the model predicted.

**Reading the plot:**
- If the model were perfect, every point would lie on the diagonal line y=x
- Points close to the diagonal = accurate predictions
- Points far from the diagonal = large prediction errors
- Systematic curves or patterns in the scatter = the model is missing a non-linear relationship

A good R² (≈0.90) would show points clustered tightly around the diagonal with no obvious pattern in the errors.

---

## What the Coefficients Mean (Business Interpretation)

When the model runs on this dataset, typical coefficients are:

```
TV        : ~0.046   → $1,000 more TV spend → +46 units sold
radio     : ~0.188   → $1,000 more radio spend → +188 units sold
newspaper : ~-0.001  → newspaper has essentially no effect
Intercept :  ~2.9    → baseline sales with zero advertising
```

**The business insight:** Radio gives the highest return per dollar spent. Newspaper is essentially wasted budget. TV has a positive but smaller effect per dollar (though total TV budgets are much larger).

This is why the model's coefficients matter more than its accuracy alone — the numbers answer a real business question.

---

## Bugs Found in the Code

| File | Bug | Fix | Lesson |
|------|-----|-----|--------|
| `AdvertisingCaseStudyModelbulding.py` | `print("Y_train shape", Y_train.shape)` repeated twice | Second print should be `Y_test.shape` | Copy-paste error in step printouts |
| `AdvertisingCaseStudyModelbuldingVisualization.py` | Same — `Y_train.shape` printed twice in Step 7 | Second should be `Y_test.shape` | Same copy-paste bug carried forward |
| All files | `test_size` changes: 0.1 → 0.2 → 0.05 | Settle on 0.2 for reliable evaluation | Experimentation visible in the code history |

---

## New Concepts in This Case Study

| Concept | File | Why It Matters |
|---------|------|---------------|
| `if 'Unnamed: 0' in df.columns` | Advertising1.py | Defensive column removal — won't crash if column absent |
| `model.coef_` (3 values) | Advertising2.py | Multiple regression returns one coefficient per feature |
| `df.isnull().sum()` | Analysis.py | Check null count per column — first step after loading CSV |
| `df.describe()` | Analysis.py | One-line statistical summary: mean, std, min, quartiles, max |
| `df.corr()` | Analysis.py | Feature correlation — preview which features matter before modeling |
| `mean_squared_error` | Modelbulding.py | Average squared prediction error |
| `RMSE = np.sqrt(MSE)` | Modelbulding.py | Error in original units — more interpretable than MSE |
| `zip(X.columns, model.coef_)` | Modelbulding.py | Pair column names with coefficients for readable output |
| `pd.DataFrame({'col1': ..., 'col2': ...})` | Modelbulding.py | Build result comparison table from arrays |
| Actual vs Predicted scatter | Visualization.py | Standard regression diagnostic — how close are predictions to reality? |

---

## Observations & Reflections

- The correlation matrix from Step 5 is the most underrated step in this pipeline. Before building any model, just looking at `df.corr()` already tells the story: TV and radio correlate strongly with sales, newspaper does not. The model coefficients confirm exactly what the correlation matrix predicted.

- `model.coef_` returning `[0.046, 0.188, -0.001]` was a genuine surprise. Newspaper advertising appears to have *negative* (or effectively zero) impact on sales in this dataset. A business decision — "stop buying newspaper ads" — could come directly from these three numbers.

- MSE, RMSE, and R² together give a more complete picture than any single metric. R² says "how much of the variance is explained." RMSE says "on average, how far off are predictions in real units." Both matter: a high R² with a large RMSE in a domain where small errors matter is still a problem.

- The `test_size` changed from 0.1 to 0.2 to 0.05 across the three files. This kind of experimentation is visible in the code history — it shows the process of trying different things to see how the output changes. With only 10 test samples (test_size=0.05), the R² result is less reliable because it depends heavily on which 10 rows happen to be selected.

- The actual vs predicted scatter plot (Step 13) is more informative than printing numbers. Seeing the cloud of points around the diagonal immediately shows how good the fit is — and whether errors are random (good) or systematic (bad).

---

## What's Next

- Add `plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--')` to draw the perfect-prediction diagonal on the scatter plot.
- Use `cross_val_score` to get a more reliable R² across multiple train/test splits instead of a single split.
- Try building the model with only TV and radio (dropping newspaper) and compare R² — hypothesis: it won't change much because newspaper coefficient is near zero.
- Explore **Residuals Plot** — `plt.scatter(Y_pred, Y_test - Y_pred)` — to check if errors are random or show a pattern.
- Apply **Polynomial Features** to capture any non-linear relationships in the data.

---

## Files

| File | Steps | Key Addition |
|------|-------|-------------|
| `Advertising1.py` | Load + clean | CSV load, `Unnamed: 0` removal |
| `Advertising2.py` | Load + clean + model | Quick X/Y split, LinearRegression, coef_ & intercept_ |
| `AdvertisingCaseStudyAnalysis.py` | 1–5 | `MarvellousAdvertise()` function, EDA: isnull, describe, corr |
| `AdvertisingCaseStudyModelbulding.py` | 1–13 (Step 13 empty) | Full pipeline: MSE/RMSE/R², f-string coefficients, result DataFrame |
| `AdvertisingCaseStudyModelbuldingVisualization.py` | 1–13 (complete) | Actual vs Predicted scatter plot |
| `Advertising.csv` | — | 200 records: TV/radio/newspaper spend + sales |


*OJT Report — Machine Learning Track | Topic: Advertising Case Study — Multiple Linear Regression*
*Trainee: Shubhada A. Palwe*
*Milestone: First regression case study — business question answered with model coefficients*
