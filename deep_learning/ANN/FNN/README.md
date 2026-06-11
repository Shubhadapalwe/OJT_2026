# On-the-Job Training (OJT) Report

---

| Field               | Details                                                                              |
|---------------------|--------------------------------------------------------------------------------------|
| **Trainee Name**    | Shubhada A. Palwe   
                                                                 |
| **Training Domain** | Deep Learning — Feedforward Neural Networks (Applied Case Studies)  
                |
| **Topic**           | FNN on Real Problems — Salary Prediction, Student Pass/Fail, Placement Prediction  
 |

| **Tools Used**      | Python, pandas, numpy, scikit-learn (`MLPRegressor`, `MLPClassifier`, `StandardScaler`), matplotlib, joblib |
| **Total Files**     | 4 (Files 18–21, continuing from series)                                             |
| **Pipeline Steps**  | Load → Preprocess → Scale → MLPRegressor/MLPClassifier → Evaluate → Visualize → Save → Load → Predict |

---

## The Topic: From Scratch to sklearn — Neural Networks on Real Problems

The previous topics (`ann_fundamentals`, `loss_and_backprop`) built every neuron, every weight update, and every gradient by hand in pure Python. That work was essential — it revealed exactly what happens inside a neural network.

This topic makes the leap to **sklearn's neural network implementation**: `MLPClassifier` and `MLPRegressor`. The mathematics is identical to what was built from scratch — weighted sums, activation functions, backpropagation, gradient descent — but sklearn handles all of it internally, allowing focus to shift to the ML pipeline: data preparation, scaling, evaluation, and model persistence.

**The FNN (Feedforward Neural Network) in sklearn:**

| Class | Problem | Output |
|-------|---------|--------|
| `MLPClassifier` | Classification (categories) | Class labels (0/1/2...) |
| `MLPRegressor` | Regression (numbers) | Continuous value |

Both use the same constructor pattern and differ only in the loss function sklearn uses internally (cross entropy for classifier, MSE for regressor) and the output activation (softmax/sigmoid for classifier, linear/identity for regressor).

This is also the first topic to use **two separate scalers** (one for X, one for y in regression), introduce `predict_proba()`, use `model.loss_curve_`, and save both the model and the scaler to disk.

---

## The 4-File Build

---

### `18_FNN_Salary_Prediction_Regression.py` — MLPRegressor: First Neural Network Regression

The first file in the OJT to use `MLPRegressor` — sklearn's neural network for continuous output prediction.

**Dataset:**
```python
# [Experience, Education Score, Skill Rating, Certificates]
X = [[1,5,4,0], [2,6,5,1], [3,6,6,1], [4,7,7,2], [5,7,8,2],
     [6,8,8,3], [7,8,9,3], [8,9,9,4], [10,9,10,5], [9,9,10,4]]

y = [22000, 26000, 32000, 40000, 47000,
     54000, 62000, 70000, 85000, 78000]
```
10 employees. Four input features. Salary target ranging from ₹22,000 to ₹85,000.

**Critical new pattern — scaling BOTH X and y:**
```python
y = np.array(y).reshape(-1, 1)   # must be 2D for scaler

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled  = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train).ravel()   # 2D → 1D after scaling
y_test_scaled  = scaler_y.transform(y_test).ravel()
```

**Why scale y in regression?** sklearn's `MLPRegressor` uses the identity function as the output activation — the raw weighted sum is the prediction. If y values are in the range 22,000–85,000 but weights are initialized near 0, the initial predictions will be near 0, creating enormous errors and unstable gradients. Scaling y to mean=0, std=1 brings targets into a range the network can match without exploding gradients.

**`y.reshape(-1, 1)`:** `StandardScaler` requires 2D input (`-1` means "however many rows", `1` means one column). After scaling, `.ravel()` flattens it back to 1D because `MLPRegressor.fit()` expects a 1D y array.

**Two separate scalers** (`scaler_X` and `scaler_y`) are needed because X and y have completely different distributions. They must never be mixed.

**Inverse transform — getting predictions back to real salary:**
```python
predictions = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
```
The model predicts in scaled space (mean=0, std=1). `inverse_transform` applies the reverse: `original = predicted × std + mean`. Without this step, the output would be meaningless numbers like `0.73` instead of `₹54,000`.

**MLPRegressor configuration:**
```python
model = MLPRegressor(
    hidden_layer_sizes=(6,),   # 1 hidden layer, 6 neurons
    activation='relu',
    solver='lbfgs',            # not adam — lbfgs is better for tiny datasets
    max_iter=5000,
    random_state=42
)
```

**`solver='lbfgs'` vs `solver='adam'`:** Adam is a stochastic optimizer that updates weights using small batches — designed for large datasets. LBFGS (Limited-memory Broyden–Fletcher–Goldfarb–Shanno) is a full-batch optimizer that uses second-order gradient information — more powerful on small datasets where it can see all data at once. With only 10 samples, LBFGS converges faster and more reliably.

**New employee prediction:**
```python
new_emp = [[5, 8, 9, 3]]
new_emp_scaled = scaler_X.transform(new_emp)    # use transform (not fit_transform!)
salary_scaled  = model.predict(new_emp_scaled)
salary = scaler_y.inverse_transform(salary_scaled.reshape(-1, 1))
```

---

### `19_FNN_Student_Result_Classification.py` — MLPClassifier: No Scaling Baseline

The first `MLPClassifier` in the Deep Learning track. Intentionally simple — **no scaling** — to establish a baseline before adding preprocessing in File 20.

**Dataset:** 15 students, 3 features: Study Hours, Attendance, Assignment Score. Target: 0=Fail, 1=Pass.

```python
model = MLPClassifier(
    hidden_layer_sizes=(5,),
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42
)
model.fit(X_train, y_train)   # raw unscaled data
```

**Why no scaling here?** This is a deliberate pedagogical choice — File 19 shows what the unscaled version produces, File 20 shows the improvement after scaling. The contrast between the two teaches *why* scaling matters rather than just mandating it.

**Evaluation:**
```python
accuracy = accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
```

**New student prediction — direct:**
```python
new_student = [[5, 75, 60]]
prediction = model.predict(new_student)
if prediction[0] == 1:
    print("PASS")
else:
    print("FAIL")
```
No transform needed here (since no scaler was fitted). This highlights the risk: if you forget to scale input consistently with how the model was trained, predictions are wrong. File 20 fixes this.

---

### `20_FNN_Student_Result_Classification_Scaling.py` — With Scaling + stratify

Same dataset and problem as File 19, but with two important additions:

**Addition 1 — `stratify=y` in train_test_split:**
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
```
`stratify=y` ensures the class ratio (Pass vs Fail) is preserved in both train and test sets. Without it, a random split might put all "Fail" students in training and all "Pass" in testing (or vice versa) — making the model appear to perform better or worse than it really does. With 15 students (only a few Fail cases), stratification is critical.

This concept appeared earlier in `wine_classifier_knn` — its reappearance here shows it's a standard professional practice, not a one-time trick.

**Addition 2 — StandardScaler on X:**
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
```
Attendance (35–90) and Study Hours (1–8) are on very different scales. Without scaling, the model treats Attendance as ~10× more important than Study Hours simply because of magnitude — not because it actually is. Scaling removes this bias.

**Solver change:**
```python
model = MLPClassifier(
    hidden_layer_sizes=(5,),
    activation='relu',
    solver='lbfgs',        # changed from 'adam' in File 19
    max_iter=2000,
    random_state=42
)
```
Switching from 'adam' to 'lbfgs' on the same small dataset — consistent with what File 18 established: lbfgs suits small data.

**`zero_division=0` in classification_report:**
```python
print(classification_report(y_test, y_pred, zero_division=0))
```
When a class appears in the test set but the model never predicts it (e.g., predicts all Pass, never Fail), sklearn raises a `UndefinedMetricWarning` and shows NaN in the report. `zero_division=0` replaces those NaN values with 0 instead of crashing — a defensive parameter for small imbalanced datasets.

**Prediction with consistent scaling:**
```python
new_student_scaled = scaler.transform([[5, 75, 60]])
prediction = model.predict(new_student_scaled)
```
The new student data passes through the *same* scaler that was fitted on training data. This is the correct pattern — and the contrast with File 19 makes the lesson clear.

---

### `21_FNN_Placement_Prediction.py` — Full 17-Step Professional Pipeline

The most complete FNN pipeline in the entire OJT portfolio — 17 numbered steps, CSV dataset, two hidden layers, probability scores, both model and scaler saved, and 4 graphs.

**Dataset from CSV:**
```python
data = pd.read_csv("placement_data.csv")
X = data[["Aptitude", "Coding", "Communication", "Academics", "Internship"]]
y = data["Placed"]
```
Five input features → binary target: 1=Placed, 0=Not Placed. The step-by-step EDA (shape, describe, isnull) mirrors the Machine Learning pipeline pattern from earlier case studies.

**Two hidden layers — the most complex network architecture in the OJT:**
```python
model = MLPClassifier(
    hidden_layer_sizes=(8, 4),   # Layer 1: 8 neurons, Layer 2: 4 neurons
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42
)
```
`hidden_layer_sizes=(8, 4)` creates a network with:
- Input layer: 5 neurons (one per feature)
- Hidden layer 1: 8 neurons (ReLU)
- Hidden layer 2: 4 neurons (ReLU)
- Output layer: 1 neuron (Sigmoid for binary)

The funnel shape (8→4) is a common architecture pattern: wider early layers detect many features, narrower later layers combine them into more abstract representations.

**`model.predict_proba()` — confidence scores:**
```python
y_prob = model.predict_proba(X_test_scaled)
print(y_prob[:5])
```
While `predict()` returns the class label (0 or 1), `predict_proba()` returns the *probability* of each class: `[[0.23, 0.77], [0.91, 0.09], ...]`. The second column is the probability of being Placed. This is more informative than a binary prediction — a student with 77% placement probability vs one with 51% are both predicted "Placed" but carry very different confidence.

**Saving BOTH model AND scaler:**
```python
joblib.dump(model, "placement_fnn_model.pkl")
joblib.dump(scaler, "placement_scaler.pkl")
```
This is the critical advancement over the Titanic case study (which only saved the model). The scaler must be saved alongside the model because any new prediction must be preprocessed with *the same scaling parameters* used during training (the same mean and std). Loading only the model without the scaler would produce wrong predictions.

**Loading and using both:**
```python
loaded_model  = joblib.load("placement_fnn_model.pkl")
loaded_scaler = joblib.load("placement_scaler.pkl")

new_student_scaled = loaded_scaler.transform(new_student)
new_prediction     = loaded_model.predict(new_student_scaled)
new_probability    = loaded_model.predict_proba(new_student_scaled)
```

**`model.loss_curve_` — training loss history:**
```python
plt.plot(model.loss_curve_)
plt.title("Training Loss Curve")
```
`loss_curve_` is a list that sklearn populates during `.fit()`, storing the loss value after each iteration. Plotting it shows whether training converged smoothly or oscillated. A good curve decreases steadily and flattens. This is the equivalent of the manually tracked `loss_list` in the ann_fundamentals training loop — but now sklearn tracks it automatically.

**4 graphs:**
- Graph 1: Bar chart — class distribution (Placed vs Not Placed) — checks for imbalance
- Graph 2: Scatter — Aptitude vs Coding, marker='o' for Placed, marker='x' for Not Placed
- Graph 3: `model.loss_curve_` — training convergence
- Graph 4: Actual vs Predicted line plot on test samples

---

## Key Progression: Files 19 → 20 → 21

| Feature | File 19 | File 20 | File 21 |
|---------|---------|---------|---------|
| Scaling | None | StandardScaler (X only) | StandardScaler (X only) |
| stratify | No | Yes | Yes |
| solver | adam | lbfgs | adam |
| Hidden layers | (5,) | (5,) | (8, 4) |
| predict_proba | No | No | Yes |
| Save model | No | No | Yes (.pkl) |
| Save scaler | No | No | Yes (.pkl) |
| Graphs | None | None | 4 |
| Dataset source | Inline | Inline | CSV file |

---

## New Concepts in This Topic

| Concept | File | Why It Matters |
|---------|------|----------------|
| `MLPRegressor` | File 18 | sklearn's neural network for continuous prediction |
| `MLPClassifier` | Files 19-21 | sklearn's neural network for classification |
| Two separate scalers (`scaler_X`, `scaler_y`) | File 18 | X and y have different distributions; never mix |
| `y.reshape(-1, 1)` | File 18 | StandardScaler requires 2D array input |
| `scaler_y.inverse_transform()` | File 18 | Convert scaled predictions back to real salary values |
| `solver='lbfgs'` vs `solver='adam'` | Files 18, 20 | lbfgs = better for small data; adam = better for large data |
| `hidden_layer_sizes=(6,)` | File 18 | Tuple controls layer count and neuron count |
| `hidden_layer_sizes=(8, 4)` | File 21 | Two hidden layers — funnel architecture |
| `stratify=y` | File 20 | Preserves class ratio in train/test split |
| `zero_division=0` | File 20 | Prevents NaN crash in classification_report on imbalanced small data |
| `model.predict_proba()` | File 21 | Returns confidence scores, not just class labels |
| `model.loss_curve_` | File 21 | sklearn stores loss history automatically during fit |
| `joblib.dump(scaler, "scaler.pkl")` | File 21 | Save scaler alongside model — mandatory for deployment |
| `pd.DataFrame([[...]], columns=[...])` | File 21 | Create proper named DataFrame for new prediction input |

---

## MLPRegressor vs MLPClassifier — Internal Differences

Both classes share the same neural network architecture. The differences are:

| | MLPRegressor | MLPClassifier |
|-|-------------|---------------|
| Output activation | Identity (linear) | Logistic (sigmoid) or Softmax |
| Loss function | MSE | Cross Entropy |
| predict() output | Continuous number | Class label |
| predict_proba() | Not available | Returns probability per class |
| y scaling | Required (manual) | Not required (labels are stable) |

---

## Observations & Reflections

- The two-scaler pattern in File 18 (`scaler_X` and `scaler_y`) was the most surprising thing to implement. In classification, you never scale the target — 0 and 1 are already meaningful. In regression, the target can be in any range and must be scaled for neural networks to train stably. The `inverse_transform` step at the end is easy to forget and produces completely wrong outputs when omitted.

- `solver='lbfgs'` felt unfamiliar at first. All the ann_fundamentals code used gradient descent (equivalent to 'sgd' or 'adam'). LBFGS is a fundamentally different optimizer — it approximates the second derivative to take smarter steps. It's computationally expensive per step but converges in far fewer iterations on small datasets. The choice of solver is as important as the choice of hidden layer architecture.

- File 19 vs File 20 is the most instructive comparison in this folder. The same model, same data, same architecture — but File 20 adds scaling and stratify. Seeing the accuracy difference (if any) makes the effect of preprocessing concrete rather than theoretical.

- Saving the scaler in File 21 (`joblib.dump(scaler, "placement_scaler.pkl")`) is the thing most beginners forget. A model deployed without its scaler is broken — any new data passed through the unscaled model will be in the wrong space. In production, model and scaler are versioned and deployed together.

- `model.loss_curve_` requires `solver='adam'` or `solver='sgd'` — it is not populated when `solver='lbfgs'`. This is why File 21 switches back to 'adam' (and uses a larger dataset) even though File 18 and 20 used 'lbfgs'. A flat or oscillating loss curve is often the first signal that learning rate or architecture needs adjustment.

---

## What's Next

- Add validation loss to File 21 — track both training loss (`model.loss_curve_`) and a separate validation loss to detect overfitting early.
- Use `GridSearchCV` to tune `hidden_layer_sizes`, `activation`, and `solver` systematically rather than choosing them manually.
- Apply the full File 21 pipeline to the Titanic dataset — compare `MLPClassifier` accuracy against the `LogisticRegression` from the Titanic case study.
- Add `confusion_matrix` heatmap using `seaborn.heatmap(cm, annot=True)` to File 21 for a cleaner visualization.
- Experiment with `hidden_layer_sizes=(16, 8, 4)` — three hidden layers — and observe if accuracy improves or overfits on the placement dataset.

---

## Files

| File | Key Addition |
|------|-------------|
| `18_FNN_Salary_Prediction_Regression.py` | `MLPRegressor`, two scalers (scaler_X, scaler_y), y.reshape(-1,1), inverse_transform, lbfgs, MAE evaluation |
| `19_FNN_Student_Result_Classification.py` | `MLPClassifier` baseline, no scaling, adam, accuracy + classification_report, PASS/FAIL output |
| `20_FNN_Student_Result_Classification_Scaling.py` | Adds StandardScaler, stratify=y, lbfgs, zero_division=0, scaler.transform on new student |
| `21_FNN_Placement_Prediction.py` | Full 17-step pipeline: CSV, EDA, (8,4) layers, predict_proba, loss_curve_, save model+scaler pkl, 4 graphs |

---

*OJT Report — Deep Learning Track | Topic: FNN Case Studies — Salary, Student Pass/Fail, Placement Prediction*
*Trainee: Shubhada A. Palwe*
*Milestone: First sklearn neural network on real problems — MLPRegressor, MLPClassifier, dual-pkl deployment pattern*
