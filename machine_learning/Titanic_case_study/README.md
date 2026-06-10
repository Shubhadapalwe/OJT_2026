# On-the-Job Training (OJT) Report

---

| Field               | Details                                                                         |
|---------------------|---------------------------------------------------------------------------------|
| **Trainee Name**    | Shubhada A. Palwe        
                                                       |
| **Training Domain** | Machine Learning — Supervised Classification (Case Study)                      |
| **Topic**           | Titanic Survival Predictor — Logistic Regression + Model Persistence            |
| **Tools Used**      | Python, pandas, numpy, scikit-learn (`LogisticRegression`, `accuracy_score`, `confusion_matrix`, `classification_report`), joblib |
| **Total Files**     | 5 Python files + 1 saved model (.pkl) + 1 dataset (.csv)                       |
| **Dataset**         | MarvellousTitanicDataset.csv — pre-encoded Titanic passenger data, binary Survived target |
| **Pipeline Steps**  | Load → Clean (4 strategies) → X/Y Split → Train/Test Split → Logistic Regression → Evaluate → **Save model (.pkl)** → **Load model → Predict** |

---

## The Case Study: Can a Machine Learn Who Survived?

On April 15, 1912, the RMS Titanic sank after hitting an iceberg. Of the 2,224 passengers and crew, 1,502 died. The tragedy is one of the most documented disasters in history — and the passenger data is one of the most studied datasets in all of machine learning.

The challenge: **given passenger information (age, fare, class, sex, port of embarkation), predict whether they survived.**

This is a **binary classification** problem: Survived = 1 (yes) or Survived = 0 (no). Unlike the Advertising case study which predicted a continuous number (sales), this model predicts a category. That makes all the difference in which algorithm is used.

| | Advertising (Linear Regression) | Titanic (Logistic Regression) |
|--|----------------------------------|-------------------------------|
| Problem type | Regression | Classification |
| Output | Sales figure (continuous) | Survived: 0 or 1 (binary) |
| Algorithm | LinearRegression | LogisticRegression |
| Evaluation | MSE, RMSE, R² | accuracy_score, confusion_matrix, classification_report |

**Why Logistic Regression — not a Decision Tree or KNN?**
Logistic Regression is specifically designed for binary outcomes. Despite the word "regression" in its name, it is a classifier. It uses the sigmoid function to output a probability between 0 and 1, then applies a threshold (default 0.5) to assign the final class label. It is fast, interpretable through its coefficients, and works well on structured tabular data like this.

**The new concept that sets this case study apart: model persistence.**
In every previous OJT file, the model was trained and used within the same script — run the script, get a prediction. Once the script ended, the model was gone. This case study introduces `joblib`: saving the trained model to a `.pkl` file, loading it back in a separate script, and predicting on new data. This is how real production systems work.

---

## The Dataset: MarvellousTitanicDataset.csv

| Property     | Value                                                                   |
|--------------|-------------------------------------------------------------------------|
| Format       | Pre-cleaned/encoded CSV — no raw strings like "male"/"female"          |
| Features (X) | Age, Fare, Sex (0/1), sibsp, Parch, Pclass, Embarked (encoded), zero   |
| Target (Y)   | Survived (0 = died, 1 = survived) — binary                             |
| Encoding     | Sex: 0=male, 1=female already numeric; Embarked: 0/1/2                 |
| Junk column  | `zero` — all zeros, no information, must be dropped                    |

**Sample rows:**

| Passengerid | Age | Fare    | Sex | sibsp | Parch | zero | Pclass | Embarked | Survived |
|-------------|-----|---------|-----|-------|-------|------|--------|----------|----------|
| 1           | 22  | 7.25    | 0   | 1     | 0     | 0    | 3      | 2        | 0        |
| 2           | 38  | 71.2833 | 1   | 1     | 0     | 0    | 1      | 0        | 1        |

**Business interpretation:** Passenger 1 — male (Sex=0), 22 years old, third class (Pclass=3), cheap fare ($7.25), embarked from port 2 — did not survive. Passenger 2 — female (Sex=1), 38 years old, first class (Pclass=1), expensive fare ($71.28), embarked from port 0 — survived. The pattern is already visible: women, first class, higher fare correlate with survival.

---

## The 5-File Build

---

### `TitanicLoadData.py` — Step 1: Load

This file introduces **professional function docstrings** — the most formally structured code in the entire OJT portfolio.

```python
def DisplayInfo(title):
    print("\n" + "="*70)
    print(title)
    print("="*70)

def ShowData(df, message):
    DisplayInfo(message)
    print("First 5 rows of dataset")
    print(df.head())
    print("\n Shape of Data set")
    print(df.shape)
    print("\n Column names")
    print(df.columns.tolist())
    print("\n Missing values in each column")
    print(df.isnull().sum())
```

**New pattern: helper functions.** Instead of inlining print logic, `DisplayInfo()` and `ShowData()` are reusable helpers that format any message or DataFrame consistently. Every subsequent file builds on these.

**Docstring format introduced:**
```
Function name : MarvellousTitanicLogistic
Description   : This function loads the titanic dataset and displays its basic structure
Parameter     : Datapath -- str -- path of csv file
Return        : df -- pandas.DataFrame -- loaded dataset
Date          : 14/03/2026
Author        : Shubhada Anil Palwe
```

This is the first time in the OJT that a function includes a formal docstring. It documents the parameter type, return type, date, and author — a professional engineering standard.

---

### `TitanicPreprocessing_2.py` — Step 2: Clean

This file introduces `CleanTitanicData(df)` — the most sophisticated preprocessing function in the entire OJT. It handles 4 different cleaning strategies in a single function.

**Strategy 1 — Drop useless columns defensively:**
```python
drop_columns = ["Passengerid", "zero", "Name", "Cabin"]
existing_columns = [col for col in drop_columns if col in df.columns]
df = df.drop(columns=existing_columns)
```
`Passengerid` is a row index — no predictive value. `zero` is all zeros. `Name` and `Cabin` are text/high-cardinality — if they exist in a fuller version of the dataset, they'd need NLP or be dropped. The list comprehension `[col for col in drop_columns if col in df.columns]` is an elegant defensive check: never crash even if a column doesn't exist.

**Strategy 2 — Numeric coerce + median fill for Age and Fare:**
```python
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
df["Age"] = df["Age"].fillna(df["Age"].median())
```
`pd.to_numeric(errors="coerce")` converts any non-numeric string to `NaN` instead of crashing. Then `fillna(median())` fills those NaN values. Median is used instead of mean because age/fare distributions are skewed — a few very high fares would pull the mean up unfairly.

**Strategy 3 — String cleaning + mode fill for Embarked:**
```python
df["Embarked"] = df["Embarked"].astype(str).str.strip()
df["Embarked"] = df["Embarked"].replace(['nan', 'None', ''], np.nan)
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
```
Even though `Embarked` is stored as numbers in this dataset, this handles the general case where port names ("S", "C", "Q") might appear as strings. `.mode()[0]` gets the most frequent value — for categorical columns, mode is the right fill strategy (unlike mean or median).

**Strategy 4 — One-hot encode Embarked:**
```python
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)
for col in df.columns:
    if df[col].dtype == bool:
        df[col] = df[col].astype(int)
```
`pd.get_dummies(drop_first=True)` converts a categorical column into binary (0/1) columns, dropping the first to avoid the dummy variable trap (multicollinearity). The `bool → int` loop that follows is important: `pd.get_dummies` in some pandas versions returns boolean columns, but sklearn expects integers.

---

### `TitanicTrainModel_4.py` — Steps 3–5: Train + Evaluate

**Logistic Regression — first appearance in the OJT:**
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)
```

`max_iter=1000` is critical. Logistic Regression uses an iterative optimization algorithm (gradient descent or LBFGS) to find the best coefficients. The default max_iter is 100, which often fails to converge on real datasets — sklearn raises a `ConvergenceWarning`. Setting it to 1000 gives the optimizer enough iterations to find the solution.

**Model coefficients — classification insight:**
```python
print("\n Intercept of model :")
print(model.intercept_)
print("\n Coeeficient of model")
for feature, coefocient in zip(X.columns, model.coef_[0]):
    print(feature, " : ", coefocient)
```

`model.coef_[0]` — note the `[0]` index. Unlike LinearRegression where `model.coef_` returns a 1D array, LogisticRegression returns a 2D array (one row per class). For binary classification, `coef_[0]` extracts the single row of feature weights. A positive coefficient means the feature increases the probability of survival; negative means it decreases it.

**Three evaluation metrics — consistent with full pipeline:**
```python
print("Accuracy : ", accuracy_score(Y_test, Y_pred))
print(confusion_matrix(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))
```

---

### `TitanicPreserveModel.py` — Step 6: Save Model

**The biggest new concept in this case study:**
```python
import joblib

def preserveModel(model, filename):
    joblib.dump(model, filename)
    print("Model preserve successfully with name :", filename)
```

`joblib.dump(model, filename)` serializes the entire trained model — all coefficients, intercept, scaling parameters, and internal state — into a `.pkl` (pickle) binary file. After this line runs, the model exists as a file on disk. The Python script can end, the computer can be restarted, and the model is still there.

**Why this matters:** Every previous case study trained the model and used it in the same run. If you wanted a prediction tomorrow, you'd have to retrain from scratch. `joblib` breaks that cycle. Train once, save, deploy anywhere.

This is how ML models go from notebooks to production. The saved file — `Marvelloustitanic.pkl` — is included in this folder as proof.

---

### `TitanictestModel.py` — Step 7: Load + Predict

```python
def LoadPreserveModel(filename):
    loaded_model = joblib.load(filename)
    print("Model successfully loaded")
    return loaded_model
```

`joblib.load(filename)` reconstructs the exact model that was saved — same coefficients, same class. The loaded model behaves identically to the original. No retraining needed.

```python
loaded_model = LoadPreserveModel("Marvelloustitanic.pkl")
Y_pred = loaded_model.predict(X_test)
accuracy = accuracy_score(Y_pred, Y_test)
cm = confusion_matrix(Y_pred, Y_test)
```

This file shows the full production pattern: load model → prepare new data → predict → evaluate. In a real deployment, `X_test` would be live passenger data, not data from the original split.

---

## New Concepts in This Case Study

| Concept | File | Why It Matters |
|---------|------|----------------|
| `LogisticRegression(max_iter=1000)` | TrainModel_4.py | First classification algorithm using probability + sigmoid function |
| `model.coef_[0]` (2D array) | TrainModel_4.py | Logistic Regression returns 2D coef; `[0]` extracts first row for binary |
| `joblib.dump(model, filename)` | PreserveModel.py | Serialize trained model to disk — train once, use forever |
| `joblib.load(filename)` | testModel.py | Load saved model without retraining |
| `.pkl` file format | testModel.py | Python pickle — standard ML model storage format |
| `pd.to_numeric(errors="coerce")` | Preprocessing_2.py | Convert column to numeric, turning bad values to NaN instead of crashing |
| `fillna(median())` vs `fillna(mode()[0])` | Preprocessing_2.py | Median for skewed numeric; mode for categorical |
| `pd.get_dummies(drop_first=True)` | Preprocessing_2.py | One-hot encode categorical without dummy variable trap |
| `bool → int` column conversion | Preprocessing_2.py | `get_dummies` sometimes returns bool; sklearn needs int |
| List comprehension for safe column drop | Preprocessing_2.py | `[col for col in list if col in df.columns]` — defensive and Pythonic |
| Professional function docstrings | LoadData.py | Parameter, Return, Date, Author — engineering documentation standard |
| `DisplayInfo()` / `ShowData()` helpers | LoadData.py | Reusable display functions — separation of concerns |

---

## What the Coefficients Mean (Business Interpretation)

When the model runs, typical logistic regression coefficients for this dataset:

```
Sex (female=1)  :  +high positive  → women had much higher survival probability
Pclass          :  negative        → lower class (higher Pclass number) = lower survival chance
Fare            :  positive        → higher fare = higher survival
Age             :  negative        → older passengers slightly less likely to survive
sibsp / Parch   :  mixed           → traveling with family had complex effects
```

The intercept is the log-odds of survival when all features are zero (the baseline).

**The historical truth encoded in data:** "women and children first" is visible in the positive Sex coefficient. First class passengers had better access to lifeboats — visible in the negative Pclass coefficient. The model learns history from numbers.

---

## Observations & Reflections

- Logistic Regression is the most counterintuitively named algorithm so far. It has "regression" in the name, outputs probabilities, but is used exclusively for classification. The key is the sigmoid function: it squishes any real number into a 0–1 range, which can then be interpreted as a probability and thresholded into a class label.

- `joblib` changes everything about how you think about ML. Before this case study, the model was a temporary object that existed only during the script run. Now the model is an artifact — something you can share, version, deploy, and reload. This is the difference between a notebook experiment and a real application.

- The preprocessing in `CleanTitanicData()` is the most realistic data cleaning in the entire OJT. Real-world datasets have columns with mixed types, missing values, string-encoded numbers, and categorical variables all at once. This function handles all of them — and does it defensively (checking before dropping, using errors="coerce" instead of crashing).

- The `confusion_matrix` argument order bug (`Y_pred, Y_test` instead of `Y_test, Y_pred`) is easy to miss because `accuracy_score` is symmetric — you get the same number either way. But the confusion matrix is not symmetric: rows and columns swap when arguments are reversed, meaning TP/FP/FN/TN labels become incorrect. This is the kind of subtle bug that passes all "does it run?" checks but produces wrong analysis.

- The `bool → int` conversion loop after `pd.get_dummies` is a sign of experience with pandas version differences. Newer pandas returns boolean columns from `get_dummies`, older returns int. The loop makes the code version-independent.

- `max_iter=1000` is worth noting: without it, on most real datasets sklearn will raise a ConvergenceWarning and return a partially-optimized model. The default of 100 iterations is almost always too few for Logistic Regression on real data.

---

## What's Next

- Apply `StandardScaler` before Logistic Regression — like in the Wine Classifier — to see if scaling improves accuracy or convergence speed.
- Use `cross_val_score` with `cv=5` for a more reliable accuracy estimate across 5 different train/test splits.
- Try `DecisionTreeClassifier` on the same Titanic data and compare accuracy — which algorithm fits this dataset better?
- Plot the confusion matrix as a heatmap using `sns.heatmap(cm, annot=True)` for a cleaner visual.
- Build a prediction function that takes a new passenger's data, runs it through the same `CleanTitanicData()` steps, loads the `.pkl` model, and returns a survival prediction — a complete end-to-end demo.
- Explore `GridSearchCV` to find the best `C` hyperparameter for Logistic Regression (the regularization strength).

---

## Files

| File | Steps | Key Addition |
|------|-------|-------------|
| `TitanicLoadData.py` | Load only | `DisplayInfo()`, `ShowData()` helpers; professional docstrings; `MarvellousTitanicLogistic()` |
| `TitanicPreprocessing_2.py` | Load + Clean | `CleanTitanicData()`: 4-strategy preprocessing, `pd.get_dummies`, `fillna(median/mode)`, `pd.to_numeric(errors="coerce")` |
| `TitanicTrainModel_4.py` | Full train + evaluate | `LogisticRegression(max_iter=1000)`, `model.coef_[0]`, accuracy + confusion_matrix + classification_report |
| `TitanicPreserveModel.py` | Train + Save | `joblib.dump(model, "Marvelloustitanic.pkl")` — model persistence |
| `TitanictestModel.py` | Load + Predict | `joblib.load()`, load-and-predict pattern |
| `Marvelloustitanic.pkl` | — | Actual saved trained model |
| `MarvellousTitanicDataset.csv` | — | Pre-encoded Titanic passenger data, 0/1 target |

---

*OJT Report — Machine Learning Track | Topic: Titanic Case Study — Logistic Regression + Model Persistence*
*Trainee: Shubhada A. Palwe*
*Milestone: First Logistic Regression model + first saved .pkl model in the OJT portfolio*
