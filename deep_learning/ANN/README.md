# On-the-Job Training (OJT) Report

---

| Field               | Details                                                                              |
|---------------------|--------------------------------------------------------------------------------------|
| **Trainee Name**    | Shubhada A. Palwe 
                                                                   |
| **Training Domain** | Deep Learning — Artificial Neural Networks (Fundamentals)
                           |
| **Topic**           | ANN from Scratch — Single Neuron → Layered Network → Training Loop → Animation      |
| **Tools Used**      | Python, numpy, math, matplotlib (`pyplot`, `FuncAnimation`), random                 |
| **Total Files**     | 11 (Files 1–9, 11, 12 — built incrementally)                                        |
| **Network Used**    | 2 Inputs → 2 Hidden Neurons (ReLU) → 1 Output Neuron (Sigmoid)                     |
| **Pipeline Steps**  | Single Neuron → ReLU → Sigmoid → Comparison → Layered ANN → Forward Pass → Training → Visualization → Animation |

---

## The Topic: Why Build a Neural Network From Scratch?

Everything in Deep Learning — TensorFlow, PyTorch, Keras — is built on top of a single idea: the artificial neuron. Before using a library that does the math automatically, it is essential to understand *what* the library is doing inside.

This topic builds that understanding through 11 files, moving from a single artificial neuron all the way to an animated neural network that updates its weights in real time during training. No shortcuts, no black boxes — every multiplication, every weight update, every graph is written by hand.

This is the first topic in the **Deep Learning** track, bridging everything learned in Machine Learning (sklearn models, train/test split, evaluation) with the internal mechanics of how modern AI systems actually work.

**What makes Deep Learning different from Machine Learning:**

| | Machine Learning (sklearn) | Deep Learning (ANN) |
|--|---------------------------|---------------------|
| Model | Decision Tree, KNN, LogReg | Layers of neurons |
| Feature engineering | Manual | Learned automatically |
| Training | fit() hides the math | Weights updated step by step |
| Interpretability | Coefficients, feature importances | Weights, activations |
| Scale | Thousands of rows | Millions of parameters |

---

## The Network Architecture (Used in Files 5–12)

```
Input Layer      Hidden Layer      Output Layer
    x1  ──w11──▶  Neuron H1 ──w_out1──▶
                  (ReLU)                  Neuron Out (Sigmoid) ──▶ ŷ
    x2  ──w21──▶  Neuron H2 ──w_out2──▶
```

- **2 inputs** (x1, x2) — features from data
- **2 hidden neurons** — each with 2 weights + 1 bias, ReLU activation
- **1 output neuron** — 2 weights + 1 bias, Sigmoid activation → probability (0 to 1)
- **Total parameters:** 4 hidden weights + 2 hidden biases + 2 output weights + 1 output bias = **9 trainable parameters**

---

## The 11-File Build

---

### `1_ANN_Single_Neuron.py` — The Core Calculation

The most fundamental file in the folder. No functions, no classes — just the raw math of one neuron.

```python
inputs  = np.array([2.0, 3.0, 4.0])
weights = np.array([0.5, 0.3, 0.2])
bias    = 1.0

weighted_sum = np.dot(inputs, weights) + bias
# = (2.0×0.5) + (3.0×0.3) + (4.0×0.2) + 1.0
# = 1.0 + 0.9 + 0.8 + 1.0 = 3.7

def relu(x):
    return max(0, x)

output = relu(weighted_sum)   # → 3.7
```

**`np.dot(inputs, weights)`** — numpy's dot product computes the weighted sum in one line instead of a manual loop. This is the same formula from the theory PDFs: `z = Σ(xᵢ·wᵢ) + b`.

**Why ReLU here:** 3.7 is positive, so ReLU passes it through unchanged. If Z were negative, the output would be 0 — the neuron "doesn't fire."

---

### `2_Activation_Relu.py` — ReLU + First Visualization

Introduces `Marvellous_neuron_forward()` — a proper function that encapsulates the full single-neuron computation.

```python
def relu(z):
    return max(0, z)

def Marvellous_neuron_forward(inputs, weights, bias):
    z = sum(w * x for w, x in zip(weights, inputs)) + bias
    y_hat = relu(z)
    return z, y_hat
```

**`zip(weights, inputs)`** — pairs each weight with its corresponding input, then the generator expression `sum(w * x for ...)` computes the weighted sum without indexing. Cleaner than a for loop with indices.

**`plot_relu()`** — uses `np.linspace(-10, 10, 200)` to generate 200 evenly spaced z values, then `np.maximum(0, z_values)` (vectorized ReLU) for the y-axis. The graph shows the characteristic "hockey stick" shape: flat at 0 for negative inputs, diagonal for positive.

Key visual detail: `plt.axvline(x=0, color="gray", linestyle="--")` draws the vertical dashed line at z=0 — showing exactly where ReLU switches behaviour.

---

### `3_Activation_Sigmoid.py` — Sigmoid + Probability Output

```python
import math

def sigmoid(z):
    return 1 / (1 + math.exp(-z))
```

`math.exp(-z)` is used for a single value (not vectorized). For plotting, `np.exp(-z_values)` is used instead — numpy handles arrays, math handles scalars.

**The sigmoid S-curve:** output ranges from 0 to 1 exclusively. When z=0, sigmoid=0.5. As z→+∞, sigmoid→1. As z→-∞, sigmoid→0. This is why sigmoid is used in the output layer for binary classification — the output is a probability.

The plot adds two horizontal reference lines at y=0 and y=1 (`plt.axhline`) to show the asymptotic bounds, making the S-shape and its limits visually clear.

---

### `4_Activation_Sigmpoid_vs_Relu.py` — Generic Neuron + Side-by-Side Plot

*(Note: filename has a typo — "Sigmpoid" — carried from the original file)*

The biggest conceptual jump in the first four files: the neuron becomes **generic** by accepting the activation function as a parameter.

```python
def Marvellous_neuron_forward(inputs, weights, bias, activation_func):
    z = sum(w * x for w, x in zip(weights, inputs)) + bias
    y_hat = activation_func(z)
    print("Activation Function :", activation_func.__name__)
    return z, y_hat

# Calling the same neuron with two different activations:
Marvellous_neuron_forward(inputs, weights, bias, sigmoid)
Marvellous_neuron_forward(inputs, weights, bias, relu)
```

**`activation_func.__name__`** — in Python, functions are objects. Every function has a `__name__` attribute that returns its name as a string. Passing `sigmoid` (the function itself, not `sigmoid()`) lets the neuron print `"sigmoid"` or `"relu"` depending on which was passed.

**The comparison plot** overlays both curves on the same axes — showing that ReLU is unbounded above zero while Sigmoid stays compressed between 0 and 1. This is why ReLU is used in hidden layers (avoids the vanishing gradient problem) and sigmoid in output layers (probability output).

---

### `5_Layered_Artificial_Neural_Network.py` — Full Layered ANN (Function-Based)

The most architecturally significant file in the folder. Seven named functions, each with a clear responsibility:

```
Marvellous_ReLU()                    ← activation
Marvellous_Sigmoid()                 ← activation
Marvellous_Calculate_Weighted_Sum()  ← z = Σ(w·x) + b
Marvellous_Display_Multiplication_Details()  ← step-by-step print
Marvellous_Process_Hidden_Layer()    ← loops over all hidden neurons
Marvellous_Process_Output_Layer()    ← output neuron with sigmoid
Marvellous_Display_Network_Summary() ← confidence %, prediction label
Marvellous_ANN_Forward_Pass()        ← orchestrates the full pass
```

**Hidden layer as a loop:**
```python
for neuron_index in range(len(hidden_weights)):
    z_value = Marvellous_Calculate_Weighted_Sum(inputs, hidden_weights[neuron_index], hidden_biases[neuron_index])
    activated_output = Marvellous_ReLU(z_value)
    hidden_outputs.append(activated_output)
```
Instead of manually computing H1 and H2 separately, a loop iterates over `hidden_weights` (a list of lists). This is the pattern that scales — adding a third hidden neuron means adding one more row to `hidden_weights`, not writing new code.

**Prediction with confidence:**
```python
print(f"Confidence Percentage: {final_output * 100:.2f}%")
if final_output >= 0.5:
    print("Prediction : Positive Class")
else:
    print("Prediction : Negative Class")
```
Sigmoid output × 100 = confidence percentage. The 0.5 threshold is the standard decision boundary for binary classification.

**Hard-coded weights:**
```
hidden_weights = [[0.5, -0.2], [0.8, 0.4]]
hidden_biases  = [0.1, -0.1]
output_weights = [1.0, -1.5]
output_bias    = 0.2
```
These are manually set (not yet trained). File 8 onward introduces training.

---

### `6_Layered_Artificial_Neural_Network_Simplified.py` — Same Network, No Functions

Identical 2→2→1 network but written as flat sequential code — no function calls. Extremely verbose, every step labelled and printed. Designed for classroom readability: you can follow the calculation top-to-bottom without jumping between functions.

Compare how H1 is computed vs File 5:
- File 5: `Marvellous_Calculate_Weighted_Sum(inputs, hidden_weights[0], hidden_biases[0])` — one line
- File 6: `m1 = x1*w11`, `m2 = x2*w12`, `z1 = m1+m2+b1`, `h1 = max(0,z1)` — four lines

The simplification is intentional: seeing the multiplication steps spelled out (x1×w11, x2×w12) makes it concrete before abstracting into a function. The same final output (h1, h2, y_hat) is produced by both files.

---

### `7_Layered_Artificial_Neural_Network_Graphical.py` — 4 Visualizations

Adds four graphs to the File 6 calculation:

**Graph 1 — ANN Structure Diagram:**
```python
plt.scatter(input_x, input_y, s=1200)   # nodes
plt.plot([input_x[i], hidden_x[j]], [input_y[i], hidden_y[j]])  # connections
plt.text(0.85, 2, f"x1\n{x1}", ...)   # node labels
```
This manually draws the network: circles at fixed coordinates for neurons, `plt.plot()` lines for connections, `plt.text()` for labels showing actual values inside nodes. The weight values are annotated on the connection lines.

**Graph 2 — Bar chart** of h1, h2, and final output values.

**Graph 3 — Sigmoid function** with the actual network output point marked as a dot on the curve (`plt.scatter(z_out, output)`). This shows where on the S-curve the specific prediction lands.

**Graph 4 — Flow diagram:** a line plot of `[x1, x2, h1, h2, final_output]` showing how values change as they pass through each layer — a simple but revealing "signal flow" visualization.

---

### `8_ANN_Trianing_Steps.py` — First Training Loop

**The biggest conceptual shift in the folder.** Everything up to here was a forward pass — inputs in, output out, weights fixed. File 8 introduces the weight update rule.

```python
weight = random.uniform(0, 1)   # random starting point
learning_rate = 0.1

for step in range(1, 11):        # 10 training steps
    predicted_output = x * weight          # FORWARD PASS
    error = actual_output - predicted_output  # ERROR
    loss = error ** 2                         # LOSS (squared)
    weight = weight + (learning_rate * error * x)  # WEIGHT UPDATE
```

**`random.uniform(0, 1)`** — the weight starts at a random value between 0 and 1. Every run gives a different starting weight, so the path to the solution looks different each time — but always converges to the same answer.

**The weight update rule:** `w_new = w_old + (lr × error × input)`. This is a simplified form of gradient descent. When the error is large (prediction far from actual), the update is large. When error is small (close to target), the update is small. After enough steps, weight converges to the value that minimizes error.

**Why `error × input`:** this is the derivative of the squared loss with respect to the weight, simplified. The input `x` appears because the prediction is `x × weight` — differentiating by `weight` brings `x` out.

With `x=2`, `actual_output=10`, the optimal weight is 5 (since 2×5=10). The training loop finds this.

---

### `9_ANN_Trianing_Steps_Graphical.py` — Training Visualized

Extends File 8 to 20 steps and adds three graphs that show the learning process visually:

**Graph 1 — Loss curve:** loss starts high, curves down toward zero. This is the fundamental picture of training — what every Deep Learning paper shows. Seeing it emerge from a 20-line script is powerful.

**Graph 2 — Prediction vs Actual:** the predicted output line starts far from the dashed actual-output line and gradually approaches it. This makes the convergence concrete.

**Graph 3 — Weight change:** the weight value moves from its random starting point toward the optimal value (5.0). The curve shows how quickly or slowly the adjustment happens with `learning_rate=0.1`.

Values are stored in lists during the loop:
```python
steps.append(step)
loss_list.append(loss)
weight_list.append(weight)
prediction_list.append(predicted_output)
```
Then plotted after training completes.

---

### `11_ANN_MultiNeuron_Trianing_Steps_Graphical.py` — Multi-Neuron Training

*(File 10 is not present — this follows file 9 directly)*

Scales training from a single weight to a full 2→2→1 network with 9 random weights. 50 training steps. Introduces a simplified backpropagation-like update:

```python
# Output layer (exact gradient):
w_out1 = w_out1 + (learning_rate * error * h1)
w_out2 = w_out2 + (learning_rate * error * h2)
b_out  = b_out  + (learning_rate * error)

# Hidden layer (approximate, scaled by 0.1):
if z1 > 0:   # only update if ReLU was active
    w11 = w11 + (learning_rate * error * x1 * w_out1 * 0.1)
    w12 = w12 + (learning_rate * error * x2 * w_out1 * 0.1)
```

**The `if z1 > 0` check** is the key insight: ReLU's derivative is 1 when z>0 and 0 when z≤0. Updating weights for a neuron that produced z≤0 (and therefore h=0) would be mathematically incorrect — those weights didn't contribute to the output. This check is the manual implementation of ReLU's gradient.

**6 graphs produced:** loss curve, prediction vs actual, hidden neuron activations h1/h2 over time, output weights change, hidden weights change, and a final ANN structure diagram with all learned weight values annotated.

---

### `12_ANN_MultiNeuron_Trianing_Steps_Graphical_Animation.py` — Animated Training

Identical to File 11 with one major addition: `matplotlib.animation.FuncAnimation` creates an animated visualization of the network updating in real time.

```python
from matplotlib.animation import FuncAnimation

def update(frame):
    ax.clear()
    # Redraw the ANN diagram with values from training step [frame]
    current_h1   = h1_list[frame]
    current_w11  = w11_list[frame]
    ...
    ax.text(hidden_x[0], hidden_y[0], f"h1\n{current_h1:.2f}", ...)

ani = FuncAnimation(fig, update, frames=len(steps), interval=700, repeat=True)
plt.show()
```

**How `FuncAnimation` works:** it calls `update(frame)` repeatedly, once per frame, where `frame` goes from 0 to `len(steps)-1`. Each call clears the axes and redraws the network with the values from that training step. At `interval=700`, each frame shows for 700ms — so you watch 50 training steps play out over ~35 seconds, weight values updating on the diagram in real time.

This is the most visually advanced code in the entire OJT portfolio across both Machine Learning and Deep Learning.

---

## New Concepts in This Topic

| Concept | File | Why It Matters |
|---------|------|----------------|
| `np.dot(inputs, weights)` | File 1 | Vectorized weighted sum — the core ANN operation |
| `sum(w * x for w, x in zip(...))` | File 2 | Generator expression weighted sum — Pythonic and clean |
| `math.exp(-z)` vs `np.exp(-z_values)` | File 3 | Scalar math vs vectorized numpy — different tools for different contexts |
| `activation_func` as parameter | File 4 | Functions are first-class objects in Python; `__name__` attribute |
| Hidden layer as a loop over weight matrix | File 5 | Scalable pattern — add neurons without rewriting code |
| `plt.scatter()` for ANN structure diagram | File 7 | Using matplotlib to draw network architecture manually |
| `plt.axvline()` / `plt.axhline()` | Files 2,3 | Reference lines on graphs |
| `random.uniform(0, 1)` for weight init | File 8 | Random initialization — weights must differ to break symmetry |
| Weight update: `w += lr * error * input` | File 8 | Gradient descent rule — foundation of all DL training |
| `if z > 0` for ReLU backprop | File 11 | ReLU gradient is 0 when neuron is inactive |
| `matplotlib.animation.FuncAnimation` | File 12 | Animated visualization of training dynamics |
| `ax.clear()` inside animation update | File 12 | Redraw strategy — clear then re-render each frame |

---

## The Learning Curve Across Files

```
File 1  → Raw numpy, single neuron, no functions
File 2  → First function (Marvellous_neuron_forward), ReLU graph
File 3  → Sigmoid function, probability output, S-curve graph
File 4  → Generic neuron (activation as parameter), side-by-side comparison
File 5  → Full layered ANN (7 functions), forward pass
File 6  → Same ANN, flat sequential code (classroom version)
File 7  → Same ANN + 4 visualizations including network diagram
File 8  → TRAINING introduced: random weight → error → update (10 steps)
File 9  → Training + 3 graphs: loss curve, prediction, weight change
File 11 → Multi-neuron training (9 weights), simplified backprop, 6 graphs
File 12 → Same + FuncAnimation — live animated training visualization
```

File 10 is not present in this folder (either skipped or not yet created).

---

## Observations & Reflections

- The jump from File 7 to File 8 is the most important conceptual transition in Deep Learning: from a static forward pass to a learning system. A network that can only do forward propagation is just a mathematical function with fixed outputs. Training is what makes it a learning system.

- The simplified backpropagation in File 11 (using the `0.1` scale factor for hidden layer updates) is not mathematically exact. Real backpropagation computes gradients precisely using the chain rule. But the simplified version captures the key ideas: output error flows backward, active neurons get updated, inactive neurons don't. It works well enough on simple data.

- `FuncAnimation` in File 12 is genuinely sophisticated — it requires understanding that `update()` is a callback, that list data must be stored during training before animation starts, and that `ax.clear()` is needed to prevent values accumulating across frames. This is a meaningful level of matplotlib mastery.

- File 5's architecture (all Marvellous_ functions) is more professional than files 6 and 7, but files 6 and 7 are more teachable. The tension between "clean code" and "readable step-by-step code" is real in educational contexts — and having both versions is the right approach.

- `random.uniform(0, 1)` for weight initialization is adequate for this toy example. In real deep learning, weights are initialized with Xavier/Glorot or He initialization to prevent gradients from vanishing or exploding across many layers.

---

## What's Next

- Implement proper backpropagation using the chain rule — compute ∂Loss/∂w for each weight exactly, without the 0.1 approximation factor.
- Add a validation set: split data into train/val and track both loss curves to detect overfitting.
- Replace manual weight updates with `numpy` matrix operations — compute all hidden neurons simultaneously as `H = ReLU(X @ W + b)` instead of one-by-one.
- Try training on real data (e.g., the Titanic or Advertising datasets) using this from-scratch ANN.
- Introduce `keras.Sequential` and compare its output with the manual results from File 5 — verify they produce the same forward pass given identical weights.

---

## Files

| File | Key Addition |
|------|-------------|
| `1_ANN_Single_Neuron.py` | np.dot weighted sum, relu=max(0,x), 7-step single neuron |
| `2_Activation_Relu.py` | `Marvellous_neuron_forward()`, zip generator sum, ReLU graph |
| `3_Activation_Sigmoid.py` | `sigmoid()`, math vs numpy, S-curve graph |
| `4_Activation_Sigmpoid_vs_Relu.py` | Generic neuron (activation as param), `__name__`, comparison plot |
| `5_Layered_Artificial_Neural_Network.py` | 7 Marvellous_ functions, full 2→2→1 ANN, confidence %, prediction label |
| `6_Layered_Artificial_Neural_Network_Simplified.py` | Same network, flat sequential, classroom-readable |
| `7_Layered_Artificial_Neural_Network_Graphical.py` | 4 graphs: ANN diagram, bar chart, sigmoid point, flow diagram |
| `8_ANN_Trianing_Steps.py` | First training loop: random weight, 10 steps, error → loss → update |
| `9_ANN_Trianing_Steps_Graphical.py` | 20-step training + 3 graphs: loss, prediction vs actual, weight change |
| `11_ANN_MultiNeuron_Trianing_Steps_Graphical.py` | 9-weight 2→2→1 training, ReLU backprop check, 6 graphs |
| `12_ANN_MultiNeuron_Trianing_Steps_Graphical_Animation.py` | FuncAnimation animated training — most advanced file in OJT |

---

*OJT Report — Deep Learning Track | Topic: ANN Fundamentals — Single Neuron to Animated Multi-Neuron Training*
*Trainee: Shubhada A. Palwe*
*Milestone: First Deep Learning topic — complete neural network built from scratch, forward pass + training + live animation*
