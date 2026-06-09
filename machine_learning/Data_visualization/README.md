# On-the-Job Training (OJT) Report

---

| Field               | Details                                          |
|---------------------|--------------------------------------------------|
| **Trainee Name**    | Shubhada A. Palwe                                |
| **Training Domain** | Machine Learning — Data Visualization            |
| **Topic**           | Matplotlib & Seaborn — Plotting for ML           |
| **Tools Used**      | Python, matplotlib, seaborn, pandas              |
| **Total Files**     | 6                                                |
| **Date**            | _(Fill in date)_                                 |

---

## Objective

Before training a model, a data scientist must **visually understand the data** — its distribution, relationships between features, outliers, and class balance. This topic covers 6 essential plot types using **Seaborn** (built on top of Matplotlib) that are used every day in real ML projects.

---

## Why Visualize Data?

In Machine Learning, raw numbers alone don't tell the full story. A good visualization can instantly reveal:
- Are there any **outliers** that will confuse the model?
- Are the **features correlated** with each other?
- Is the **class distribution balanced** or skewed?
- How are values **spread** across the dataset?

These questions are answered before writing a single line of model code.

---

## Tools Used

| Library | Role |
|---------|------|
| `matplotlib.pyplot` | Core plotting engine — controls the display window |
| `seaborn` | High-level plotting library built on matplotlib — cleaner, smarter charts |
| `pandas` | Data structure (DataFrame) used as input for some plots |

---

## File-by-File Learning

---

### 1. `Matplotlib_histogram.py`
**Plot Type: Histogram**
**Purpose: Understand distribution of continuous (numeric) values**

A histogram groups continuous data into **bins** and shows how many values fall in each bin. It answers: *"Where are my values concentrated?"*

```python
sns.histplot(data=[10, 20, 30, 20, 20, 20, 30, 40])
plt.show()
```

**What I noticed:**
- The value `20` appears 4 times — the tallest bar.
- The x-axis shows the value ranges, the y-axis shows how many times they appear.
- Histograms are for **continuous (numeric) data** — age, weight, temperature, salary.

**Key learning:** Use a histogram when you want to see the **spread and frequency** of numeric data. It's the first plot to make when exploring a new dataset.

> *Tip from class notes: "If values are continuous, use histogram"*

---

### 2. `Matplotlib_CountPlot.py`
**Plot Type: Count Plot**
**Purpose: Count occurrences of categorical values**

A count plot is like a histogram, but for **categorical (text) data**. It counts how many times each category appears and draws a bar for it.

```python
sns.countplot(x=["A", "B", "A", "A", "B", "A", "C"])
plt.show()
```

**What I noticed:**
- A appears 4 times → tallest bar
- B appears 2 times → medium bar
- C appears 1 time → shortest bar
- Perfect for checking class balance in a classification dataset.

**Key learning:** Use a count plot when your data has **categories or labels** (not numbers). It immediately shows if one class dominates the others — a sign of class imbalance.

> *Tip from class notes: "If values are categorical, use countplot"*

---

### 3. `Matplotlib_CountPlotX.py`
**Plot Type: Count Plot (Real-World Example)**
**Purpose: Apply count plot to real data — programming language popularity**

This version applies the same count plot to a more realistic dataset — a list of programming languages.

```python
sns.countplot(x=["C", "C", "C++", "Java", "C++", "Python",
                  "Javascript", "C++", "Golang", "C"])
plt.show()
```

**What I noticed:**
- C++ appears 3 times → most popular in this sample
- C appears 2 times
- Java, Python, Javascript, Golang appear once each
- The chart instantly shows the distribution without any counting by hand.

**Key learning:** Count plots work on any categorical list — not just A/B/C test data, but real labels like countries, job titles, product categories, or class labels in an ML dataset.

---

### 4. `Matplotlib_Boxplot.py`
**Plot Type: Box Plot**
**Purpose: Detect outliers in data**

A box plot shows the **statistical summary** of a dataset in one picture — minimum, maximum, median, and quartiles. Most importantly, it visually highlights **outliers** (unusually high or low values).

```python
sns.boxplot(x=[10, 20, 30, 110])
plt.show()
```

**What I noticed:**
- The values 10, 20, 30 form the main box.
- The value `110` is plotted as a **separate dot** outside the whiskers — it is an **outlier**.
- Outliers can damage ML model training. A model might over-fit to that single unusual point.

**How to read a box plot:**

```
|---[  Q1 | Median | Q3  ]---|   o  ← outlier
 min                     max
```

| Part | Meaning |
|------|---------|
| Box (Q1 to Q3) | Middle 50% of data (Interquartile Range) |
| Line inside box | Median (middle value) |
| Whiskers | Min and max within normal range |
| Dot outside | Outlier — unusually different value |

**Key learning:** Always draw a box plot before training. If outliers exist, decide whether to remove them or keep them — they can significantly affect model accuracy.

---

### 5. `Matplotlib_Scatterplot.py`
**Plot Type: Scatter Plot**
**Purpose: Show relationship between two numeric variables**

A scatter plot places each data point as a dot on an X-Y grid. It reveals whether two variables have a **relationship (correlation)** — as one increases, does the other also increase?

```python
sns.scatterplot(x=[1, 2, 3], y=[3, 1, 4])
plt.show()
```

**What I noticed:**
- Each (x, y) pair becomes one dot on the chart.
- If dots form a line going up → **positive correlation** (both increase together).
- If dots form a line going down → **negative correlation** (one increases as other decreases).
- If dots are scattered randomly → **no correlation**.

**Key learning:** Scatter plots are used to check if two **features are related**. In ML, correlated features can help or hurt a model — it's important to know which features work together.

---

### 6. `Matplotlib_Heatmap.py`
**Plot Type: Heatmap (Correlation Matrix)**
**Purpose: Show correlation between ALL features at once**

A heatmap is the most powerful visualization for ML. It uses **color intensity** to show how strongly each pair of features is correlated — all at once in a single grid.

```python
dobj = pd.DataFrame({
    "A": [1, 2, 3],
    "B": [4, 5, 6],
    "C": [7, 8, 9]
})

sns.heatmap(dobj.corr(), annot=True)
plt.show()
```

**What I noticed:**
- `dobj.corr()` computes the **correlation matrix** — a table showing correlation between every pair of columns.
- `annot=True` prints the actual number inside each cell (e.g., 1.0, -0.5).
- Diagonal is always **1.0** — every feature is perfectly correlated with itself.
- A, B, C all increase together in this data → all values are **1.0** (perfectly correlated).

**How to read correlation values:**

| Value | Meaning |
|-------|---------|
| 1.0 | Perfect positive correlation |
| 0.5 to 0.9 | Strong positive correlation |
| 0.0 | No correlation |
| -0.5 to -0.9 | Strong negative correlation |
| -1.0 | Perfect negative correlation |

**Key learning:** Heatmaps show **feature correlation** — which features move together. If two features are perfectly correlated (value = 1.0), one of them is redundant and can be removed before training. This is called **feature selection**.

> *Class note: "yacha use Feature correlation display karayla hoto" = "It is used to display feature correlation"*

---

## When to Use Which Plot

| Plot | Data Type | Use When |
|------|-----------|----------|
| Histogram | Continuous (numeric) | See how values are spread |
| Count Plot | Categorical (text/labels) | Count class frequencies, check imbalance |
| Box Plot | Continuous (numeric) | Find outliers |
| Scatter Plot | Two numeric columns | Check relationship between 2 features |
| Heatmap | All numeric columns | Check correlations between all features |

---

## How Visualization Fits in the ML Pipeline

| ML Step | Plot Used |
|---------|-----------|
| **Data Analysis** | Histogram, Count Plot — understand what's in the data |
| **Data Cleaning** | Box Plot — find and remove outliers |
| **Feature Engineering** | Heatmap — identify correlated/redundant features |
| **Model Evaluation** | Count Plot — check if prediction classes are balanced |

---

## Observations & Reflections

- Before this topic I was jumping straight to model training. Now I know that **visualization comes first** — it's the "look before you leap" step of ML.
- The box plot was the most eye-opening. The `110` in the data was obviously wrong, and the plot flagged it immediately without any calculation.
- The heatmap feels like a cheat code — instead of checking each pair of features one by one with scatter plots, it shows all correlations at once.
- I noticed that seaborn does all the heavy lifting — it calculates counts, correlations, and quartiles automatically. Matplotlib just handles displaying the result.
- The class notes switching between English and Marathi (`jar countigus variable astil tr histogram`) show real thinking happening while learning — not just copying code.

---

## What's Next

- Apply these plots to a real dataset (Iris, CSV file).
- Use `hue` parameter in seaborn plots to color by class label (e.g., `sns.scatterplot(x=..., y=..., hue=target)`).
- Learn `plt.title()`, `plt.xlabel()`, `plt.ylabel()` to label charts properly.
- Use `sns.pairplot()` — automatically draws scatter plots for all pairs of features in a DataFrame.
- Save plots to image files using `plt.savefig("plot.png")`.

---

## Files

| File | Plot Type | Key Concept |
|------|-----------|-------------|
| `Matplotlib_histogram.py` | Histogram | Distribution of continuous values |
| `Matplotlib_CountPlot.py` | Count Plot | Frequency of categorical values (A/B/C) |
| `Matplotlib_CountPlotX.py` | Count Plot | Real-world categories (languages) |
| `Matplotlib_Boxplot.py` | Box Plot | Outlier detection |
| `Matplotlib_Scatterplot.py` | Scatter Plot | Relationship between 2 variables |
| `Matplotlib_Heatmap.py` | Heatmap | Feature correlation matrix |

---

*OJT Report — Machine Learning Track | Topic: Data Visualization*
*Trainee: Shubhada A. Palwe*
