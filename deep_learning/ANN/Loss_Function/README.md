# On-the-Job Training (OJT) Report

---

| Field               | Details                                                                              |
|---------------------|--------------------------------------------------------------------------------------|
| **Trainee Name**    | Shubhada A. Palwe                                                                    |
| **Training Domain** | Deep Learning — Loss Functions + Backpropagation                                    |
| **Topic**           | How a Neural Network Knows It's Wrong — MSE, MAE, BCE, Gradient Descent, Backprop  |
| **Tools Used**      | Python, math, matplotlib (`pyplot`, `FuncAnimation`)                                |
| **Total Files**     | 5 (Files 13–17, continuing from ann_fundamentals series)                            |
| **Pipeline Steps**  | MSE → MAE → Binary Cross Entropy → Backpropagation → Animated Backprop             |

---

## The Topic: What Comes After the Forward Pass?

In `ann_fundamentals`, the network learned to compute a prediction through forward propagation. But there was an unanswered question: *how does the network know whether its prediction is good or bad, and how does it improve?*

This topic answers that question in full.

There are two parts:

**Part 1 — Loss Functions (Files 13–15):** Measuring *how wrong* the prediction is. Three different formulas, three different use cases.

**Part 2 — Backpropagation (Files 16–17):** Using that error measurement to compute *which direction* to adjust each weight. The chain rule applied step by step.

Together, these two concepts form the complete training loop that powers every neural network:

```
Forward Pass → Predict → Compute Loss → Backpropagate → Update Weights → Repeat
```

This is not a library call. Every gradient is computed by hand, every derivative is labelled, every weight update is printed. By the end of File 17, there are no more hidden steps in how a neural network learns.

---

## Part 1: Loss Functions

A **loss function** (also called cost function) measures the distance between what the network predicted and what the correct answer actually is. The smaller the loss, the better the network is performing. Training is the process of minimizing this number.

Different problems use different loss functions:

| Loss Function | Problem Type | Formula |
|---------------|-------------|---------|
| MSE | Regression (continuous output) | Σ(yᵢ − ŷᵢ)² / n |
| MAE | Regression (robust to outliers) | Σ\|yᵢ − ŷᵢ\| / n |
| Binary Cross Entropy | Binary classification (0/1 output) | −[y·log(p) + (1−y)·log(1−p)] |

---

### `13_LossFunction_MSE.py` — Mean Squared Error

```python
def Marvellous_MSE(y_true, y_pred):
    n = len(y_true)
    total_error = 0
    for i in range(n):
        error = y_true[i] - y_pred[i]
        total_error += error ** 2   # Squared error
    return total_error / n

y_true = [10, 20, 30]
y_pred = [12, 18, 33]
```

**Manual calculation:**
```
Error 1: 10 - 12 = -2  → squared → 4
Error 2: 20 - 18 =  2  → squared → 4
Error 3: 30 - 33 = -3  → squared → 9
Total = 17   MSE = 17/3 = 5.67
```

**Why squared?** Two reasons: (1) squaring makes all errors positive, so underestimates and overestimates don't cancel each other. (2) squaring penalizes large errors *more* than small ones — a prediction that is 6 units off contributes 36 to the loss, while a prediction 2 units off contributes only 4. This makes the model try harder to fix its biggest mistakes.

**Why MSE is differentiable:** the squared term has a clean derivative (`2 × error`), which is what gradient descent needs. This is why MSE is the standard choice for regression and the foundation of backpropagation math.

---

### `14_LossFunction_MAE.py` — Mean Absolute Error

```python
def Marvellous_MAE(y_true, y_pred):
    n = len(y_true)
    total_error = 0
    for i in range(n):
        error = abs(y_true[i] - y_pred[i])   # abs() not **2
        total_error += error
    return total_error / n
```

**Same data, different result:**
```
|10 - 12| = 2
|20 - 18| = 2
|30 - 33| = 3
MAE = 7/3 = 2.33
```

**MSE vs MAE — the key difference:**

| | MSE | MAE |
|-|-----|-----|
| Outlier sensitivity | High — outliers get squared, dominating the loss | Low — all errors count equally |
| Gradient | Smooth (proportional to error) | Constant (always ±1 direction) |
| Use case | When large errors are especially bad | When outliers should not dominate |

With the same `y_pred=[12,18,33]`: MSE=5.67 but MAE=2.33. The prediction `33` vs actual `30` has a larger impact on MSE (9 vs 3) because it's squared. MAE treats it the same as the others — `abs(error)`.

**The `abs()` insight:** unlike squaring, absolute value is non-differentiable at zero (the V-shape has no derivative at the tip). This makes gradient-based optimization slightly harder with MAE — the gradient doesn't shrink as predictions get closer to the target.

---

### `15_LossFunction_Binary_Cross_Entropy.py` — Binary Cross Entropy

The most mathematically interesting loss function in the set. Used whenever the output is a probability (0 to 1) and the target is 0 or 1.

```python
def Marvellous_Binary_CrossEntropy(y_true, y_pred):
    n = len(y_true)
    total_loss = 0
    for i in range(n):
        y = y_true[i]
        p = y_pred[i]
        p = max(min(p, 0.999), 0.001)    # numerical safety clamp
        loss = -(y * math.log(p) + (1 - y) * math.log(1 - p))
        total_loss += loss
    return total_loss / n

y_true = [1, 0, 1]
y_pred = [0.9, 0.2, 0.8]
```

**The formula dissected:**

For a single sample: `Loss = −[y·log(p) + (1−y)·log(1−p)]`

- When `y=1` (actual is positive): the second term disappears (1−y=0). Loss = `−log(p)`. If the model predicted p=0.9, loss = −log(0.9) ≈ 0.105. If p=0.1, loss = −log(0.1) ≈ 2.30 — much larger penalty for wrong probability.
- When `y=0` (actual is negative): the first term disappears. Loss = `−log(1−p)`. If model predicted p=0.2 (correctly low), loss = −log(0.8) ≈ 0.22.

**`p = max(min(p, 0.999), 0.001)`** — This is numerical safety. `math.log(0)` is `-∞` and would crash the program. If the model is ever 100% confident (p=1.0 or p=0.0), this clamp prevents the infinity. In real libraries (TensorFlow, PyTorch), this is done internally with `epsilon`.

**Why not MSE for classification?** MSE with sigmoid outputs creates a gradient landscape with flat regions where the gradient nearly vanishes — training stalls. Cross entropy's log function has gradients that stay informative across the full 0-to-1 range.

---

## Part 2: Backpropagation

### `16_Gradient_Backprapogation.py` — The First Real Backprop

*(Note: filename has a typo — "Backprapogation" — carried from original)*

This is the most mathematically significant file in the OJT Deep Learning track. It implements true backpropagation using the **chain rule** — the exact algorithm used inside TensorFlow and PyTorch.

**Network:** single neuron, x1=1.0, x2=2.0, target=1.0, sigmoid output.

**The 4-step loop (10 epochs):**

```python
# Step 1: Forward Propagation
z = (x1 * w1) + (x2 * w2) + b
output = Marvellous_Sigmoid(z)

# Step 2: Loss (MSE with 0.5 factor)
loss = 0.5 * (target - output) ** 2

# Step 3: Backpropagation — Chain Rule
dL_doutput = output - target                      # ∂L/∂ŷ
doutput_dz = Marvellous_Sigmoid_Derivative(output) # ∂ŷ/∂z = ŷ(1-ŷ)
dL_dz = dL_doutput * doutput_dz                   # ∂L/∂z (chain rule)

dL_dw1 = dL_dz * x1   # ∂L/∂w1 = ∂L/∂z × ∂z/∂w1 = ∂L/∂z × x1
dL_dw2 = dL_dz * x2   # ∂L/∂w2 = ∂L/∂z × x2
dL_db  = dL_dz         # ∂L/∂b  = ∂L/∂z × 1

# Step 4: Gradient Descent
w1 = w1 - (learning_rate * dL_dw1)
w2 = w2 - (learning_rate * dL_dw2)
b  = b  - (learning_rate * dL_db)
```

**Why the 0.5 factor in loss?** `loss = 0.5 × (target − output)²`. The derivative of this is `−(target − output)` = `(output − target)`. Without the 0.5, the derivative would be `2 × (output − target)`. The 0.5 cancels the 2 and makes the gradient cleaner — a common math convention in backpropagation derivations.

**The chain rule unpacked:**

The gradient of the loss with respect to `w1` cannot be computed directly because `w1` doesn't appear in the loss formula directly — it appears through `z`, which appears through `output`, which appears in the loss. The chain rule says: multiply all the intermediate derivatives together.

```
∂L/∂w1 = (∂L/∂output) × (∂output/∂z) × (∂z/∂w1)
        =  (output − target)  ×  output(1−output)  ×  x1
```

This is `dL_doutput × doutput_dz × x1` — exactly what the code computes.

**`Marvellous_Sigmoid_Derivative(output)`:**
```python
def Marvellous_Sigmoid_Derivative(output):
    return output * (1 - output)
```
The derivative of sigmoid is `σ(z) × (1 − σ(z))`. Since `output = σ(z)` is already computed, the derivative is just `output × (1 − output)` — no need to recompute from z. This is why sigmoid's derivative is computationally cheap.

**Weight update direction:** `w = w − lr × gradient`. The minus sign is critical. The gradient points *uphill* on the loss surface. Subtracting it moves *downhill* — reducing the loss. This is gradient descent.

---

### `17_Gradient_Backprapogation_Graphical.py` — Animated Backprop

Identical math to File 16 (now 20 epochs) with two major additions: a `history` dictionary and a `FuncAnimation`.

**The `history` dictionary:**
```python
history.append({
    "epoch": epoch,
    "z": z, "output": output, "loss": loss,
    "w1": w1, "w2": w2, "b": b,
    "grad_w1": dL_dw1, "grad_w2": dL_dw2, "grad_b": dL_db
})
```
All 10 values per epoch are stored *before* the weight update. This is important: the animation shows what happened at each step in the order it happened — not the updated values.

**The animated neuron diagram:**
```python
draw_circle(x1_pos,   f"x1\n{x1}",                  "lightgreen")
draw_circle(x2_pos,   f"x2\n{x2}",                  "lightgreen")
draw_circle(bias_pos, f"b\n{round(data['b'], 3)}",   "khaki")
draw_circle(sum_pos,  f"z\n{round(data['z'], 3)}",   "lightskyblue")
draw_circle(out_pos,  f"ŷ\n{round(data['output'], 3)}", "salmon")
```

Each node is a colored circle (`plt.Circle`) with the current value displayed inside. Arrows between nodes are drawn with `ax.annotate(..., arrowprops=dict(arrowstyle="->"))`. The formula, loss, and all three gradients are displayed as text. As training progresses, you see w1 and w2 update each second while the loss value decreases.

**`interval=1000, repeat=False`** — one second per epoch, stops after 20 epochs (no loop). This is slower than File 12's 700ms to give time to read the gradient values.

---

## New Concepts in This Topic

| Concept | File | Why It Matters |
|---------|------|----------------|
| `error ** 2` (squared error) | File 13 | Penalizes large errors more; always positive; differentiable |
| `abs(error)` (absolute error) | File 14 | Robust to outliers; non-differentiable at 0 |
| `math.log(p)` in BCE | File 15 | Log of small probability = large loss; log of high probability ≈ 0 |
| `max(min(p, 0.999), 0.001)` | File 15 | Numerical clamp to prevent log(0) = -∞ crash |
| `0.5 × (target − output)²` | File 16 | The 0.5 factor cancels the 2 from differentiation |
| `Marvellous_Sigmoid_Derivative` | File 16 | σ'(z) = σ(z)(1−σ(z)) — cheapest derivative in DL |
| `dL_doutput`, `doutput_dz`, `dL_dz` | File 16 | Chain rule steps named explicitly |
| `dL_dw1 = dL_dz * x1` | File 16 | ∂z/∂w1 = x1 — derivative of z w.r.t. its weight |
| `w1 = w1 − lr × dL_dw1` | File 16 | Gradient descent: move opposite to gradient |
| `history.append({...})` | File 17 | Store full training state per epoch for animation |
| `plt.Circle(position, radius)` | File 17 | Drawing neuron nodes as circles |
| `ax.annotate(..., arrowprops)` | File 17 | Directional arrows between nodes |
| `interval=1000, repeat=False` | File 17 | Slower animation, single-play for readability |

---

## MSE vs MAE vs BCE — When to Use Which

| Situation | Right Choice | Why |
|-----------|-------------|-----|
| Predicting house prices (continuous) | MSE | Penalizes large errors more; smooth gradient |
| Predicting prices with some outlier properties | MAE | Outliers don't dominate the loss |
| Predicting spam/not spam (0 or 1) | Binary Cross Entropy | Designed for probabilities; gradients don't vanish near 0/1 |
| Neural network output layer is sigmoid | BCE | MSE + sigmoid creates vanishing gradients; BCE doesn't |

---

## Observations & Reflections

- The `0.5` factor in MSE loss (`loss = 0.5*(target-output)²`) confused me at first — it seems to make the loss artificially smaller. But it exists purely for mathematical convenience: it cancels the `2` that comes out of differentiating `(target-output)²`. The gradient `dL/doutput = output - target` is cleaner than `2*(output-target)`. This kind of mathematical convention appears constantly in deep learning papers.

- The chain rule in File 16 is the single most important piece of mathematics in all of deep learning. Every library — TensorFlow, PyTorch, JAX — is essentially a very efficient implementation of the chain rule across thousands of layers. Seeing it written out as three lines (`dL_doutput`, `doutput_dz`, `dL_dz`) makes the concept concrete before letting a library hide it.

- `max(min(p, 0.999), 0.001)` in the BCE file is a small but crucial detail. In production, a model that becomes overconfident (predicting exactly 0 or 1) would crash without this guard. PyTorch's `BCELoss` has this built in with a `clamp` operation internally.

- The `sigmoid derivative = output × (1 - output)` formula is elegant: it means the derivative is largest (0.25) when output is 0.5 (the model is maximally uncertain) and smallest when output is near 0 or 1 (the model is very confident). This is related to the vanishing gradient problem — in deep networks, chaining many sigmoid derivatives together produces very small numbers.

- File 17's animated diagram shows something File 16 can't: you can *watch* w1 and w2 converge. At epoch 1, w2 changes more than w1 because x2=2.0 is larger than x1=1.0 — the gradient `dL_dw = dL_dz × x` scales with the input magnitude. The animation makes this visible.

---

## What's Next

- Implement full **multi-layer backpropagation** using the chain rule across two layers — the hidden layer gradients require chaining through the output layer's weights.
- Add **Binary Cross Entropy** as the loss function in the backprop file and compare convergence with MSE.
- Try different **learning rates** (0.01, 0.5, 1.0) and observe overshoot vs slow convergence in the animated graph.
- Implement **batch gradient descent** — compute the average gradient across multiple samples before updating weights, rather than updating after every single sample.
- Use `numpy` matrix operations to compute gradients across all weights simultaneously: `dL_dW = X.T @ dL_dZ` — the vectorized backprop formula.

---

## Files

| File | Key Addition |
|------|-------------|
| `13_LossFunction_MSE.py` | `Marvellous_MSE()` — squared error loop, MSE=5.67 on sample data |
| `14_LossFunction_MAE.py` | `Marvellous_MAE()` — absolute error loop, MAE=2.33, robust to outliers |
| `15_LossFunction_Binary_Cross_Entropy.py` | BCE formula, `math.log(p)`, numerical clamp `max(min(p,0.999),0.001)` |
| `16_Gradient_Backprapogation.py` | Full backprop: chain rule, sigmoid derivative, dL/dw1, dL/dw2, dL/db, 10 epochs |
| `17_Gradient_Backprapogation_Graphical.py` | Same + `history` dict, colored neuron diagram, `ax.annotate` arrows, FuncAnimation (1s/epoch, no repeat) |

---

*OJT Report — Deep Learning Track | Topic: Loss Functions + Backpropagation*
*Trainee: Shubhada A. Palwe*
*Milestone: First complete backpropagation implementation — chain rule computed by hand, every gradient named and printed*
