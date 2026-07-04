# CNN Pipeline — Complete Step by Step

**Marvellous Infosystems — OJT 2026**
**Trainee : Shubhada A. Palwe**

---

## About This Folder

This folder brings together everything we learned about CNNs and runs the complete pipeline end-to-end — Convolution → ReLU → Max Pooling → Flatten → Fully Connected → Prediction.

All 4 files run the same pipeline but at different levels of detail and visualization:
- File 39 : full pipeline, text + static matplotlib plots
- File 40 : same but animated — kernel visibly slides with time.sleep
- File 41 : only the internal steps (ReLU → Pool → Flatten → FC), no convolution
- File 42 : bigger image (6×6), bigger kernel (3×4), stride-1 pooling, graphical FC network diagram

---

## The 6-Step CNN Pipeline

All files implement these same steps:

```
Step 1 : INPUT IMAGE       → 5×5 binary matrix (0s and 1s)
Step 2 : CONVOLUTION       → slide 3×3 kernel → feature map
Step 3 : ReLU              → max(0, x) on every element
Step 4 : MAX POOLING       → 2×2 blocks, take the maximum
Step 5 : FLATTEN           → 2D array → 1D vector
Step 6 : FULLY CONNECTED   → flat × weights + bias = score
         DECISION          → score > 0 → "Vertical Line"
```

---

## File 39 — Complete Pipeline with Static Plots

### Input

```python
# Vertical Line
image = np.array([
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0]
], dtype=float)

# Horizontal Line
image = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
], dtype=float)
```

User chooses at runtime which image to test.

### Kernel

```python
kernel = np.array([
    [-1,  1, -1],
    [-1,  1, -1],
    [-1,  1, -1]
], dtype=float)
```

This kernel is designed to detect vertical lines. The centre column is +1 (rewards vertical bright pixels). The side columns are -1 (penalizes background pixels). When placed over a vertical stripe, the result is a large positive number.

### Convolution Function

```python
for i in range(output_rows):
    for j in range(output_cols):
        region = image[i:i+krows, j:j+kcols]   # extract 3×3 patch
        multiplication = region * kernel         # element-wise multiply
        result = np.sum(multiplication)          # sum all 9 values
        output[i][j] = result
```

Output size: `(5-3+1) × (5-3+1) = 3×3` feature map.

At each position, the function prints: the region selected, the kernel, the element-wise product, and the sum. Then shows a matplotlib plot of the feature map so far.

### ReLU Function

```python
output = np.maximum(0, data)
```

Simple. Every negative number becomes 0. Every positive stays. This removes the "noise" from the convolution — areas where the pattern was NOT detected had negative values, and we discard those.

### Pooling Function

```python
output_rows = rows // 2
output_cols = cols // 2

for i in range(0, rows, 2):
    for j in range(0, cols, 2):
        block = data[i:i+2, j:j+2]
        if block.shape != (2, 2):
            continue
        output[r][c] = np.max(block)
```

Takes non-overlapping 2×2 blocks and picks the maximum from each. A 3×3 input becomes about 1×1 (only one complete 2×2 block fits). The `if block.shape != (2,2): continue` handles edge cases where the block is incomplete.

### Flatten and FC

```python
flat = data.flatten()                    # 2D → 1D
weights = np.array([1, 1, 1, 1], dtype=float)
result = np.sum(flat * weights) + bias   # score
```

Flatten converts the 2D pooling output into a 1D vector. Then FC layer computes a score by multiplying each value with its weight and summing.

### Decision

```python
if score > 0:
    prediction = "Vertical Line"
else:
    prediction = "Horizontal Line"
```

The vertical-detection kernel gives positive scores for vertical lines and near-zero or negative for horizontal lines.

---

## File 40 — Animated Pipeline

Same as File 39 but with animation. The key addition is a yellow rectangle that shows which part of the image the kernel is currently processing:

```python
rectangle = plt.Rectangle((rect_x, rect_y), 3, 3,
                           fill=False, edgecolor='yellow', linewidth=3)
axes[0].add_patch(rectangle)
```

`plt.Rectangle((x, y), width, height)` draws a rectangle on the plot. We subtract 0.5 from the position because matplotlib pixel coordinates are centred at integer values.

At each step: 3 panels are shown — input image with yellow rectangle, kernel, and the growing feature map.

```python
time.sleep(delay)   # delay=1.0 for convolution, 1.5 for pooling
```

The animation pauses for 1 second at each position so we can see it clearly. Good for presentations or demos.

At the end, `Marvellous_ShowPipeline` shows all 4 stages (Input, Conv, ReLU, Pool) side by side in one figure.

---

## File 41 — Internal Steps Only (Starts from Feature Map)

This file starts from a pre-built feature map — skips convolution. Useful for focusing on what happens after convolution.

```python
feature_map = np.array([
    [-2, 5, 1, 2],
    [ 3,-1, 4, 1],
    [ 0, 2, 0, 6],
    [ 1, 3, 2, 0]
], dtype=float)
```

The mix of positive and negative values is intentional — it shows ReLU behavior clearly.

ReLU is printed value-by-value:
```
ReLU(-2) = 0
ReLU(5)  = 5
ReLU(3)  = 3
ReLU(-1) = 0
...
```

Pooling uses 4 explicitly named blocks instead of a loop:
```python
block1 = relu_output[0:2, 0:2]   # top-left
block2 = relu_output[0:2, 2:4]   # top-right
block3 = relu_output[2:4, 0:2]   # bottom-left
block4 = relu_output[2:4, 2:4]   # bottom-right
```

FC uses custom weights `[0.8, 0.5, 0.3, 0.9]` instead of all-ones. This shows that different features can have different importance.

This file has no matplotlib — pure console output. Good for running in environments without display.

---

## File 42 — BigData Version

This file uses a 6×6 image and a 3×4 kernel. The key differences:

**Bigger image:**
```python
image = np.array([
    [0, 0, 1, 1, 0, 0],
    ...
], dtype=float)
```
The vertical line is now 2 columns wide, matching the wider kernel.

**3×4 kernel:**
```python
kernel = np.array([
    [-1,  1,  1, -1],
    [-1,  1,  1, -1],
    [-1,  1,  1, -1]
], dtype=float)
```
Output size: `(6-3+1) × (6-4+1) = 4×3` feature map.

**Stride-1 pooling:**
```python
pool = Marvellous_Pooling(relu, pool_size=2, stride=1)
```
Default stride is 2 (halves dimensions). Stride 1 means the 2×2 window moves one pixel at a time — overlapping. Produces a larger output than stride 2.

**3-panel convolution display:** at each kernel position, shows Selected Region, Kernel, and Region × Kernel side by side in one figure.

**Explicit formula string:**
```python
terms = [f"({r}×{k})" for r, k in zip(flat_region, flat_kernel)]
print(" + ".join(terms))
```
Prints something like: `(0×-1) + (0×1) + (1×1) + (1×-1) + ...`

**Graphical FC network diagram:**
```python
plt.scatter(input_x, y_pos[i], s=1500, color='skyblue')   # flatten nodes
plt.scatter(hidden_x, 5, s=2500, color='lightgreen')       # FC neuron
plt.scatter(output_x, 5, s=3000, color='orange')           # output
plt.plot([...], [...], ...)                                 # connections with weight labels
```
Draws a visual neural network showing Flatten → FC → Output, with weight values on the connecting lines.

---

## File Comparison

| Feature | File 39 | File 40 | File 41 | File 42 |
|---------|---------|---------|---------|---------|
| Image size | 5×5 | 5×5 | 4×4 (pre-made) | 6×6 |
| Kernel size | 3×3 | 3×3 | (skip conv) | 3×4 |
| Pooling stride | 2 | 2 | 2 | 1 |
| Animation | No | Yes | No | No |
| matplotlib | Yes | Yes (animated) | No | Yes |
| FC diagram | No | No | No | Yes |

---

## How to Run

```bash
pip install numpy matplotlib
python 39_CNN_Convolution_ReLU_Pooling_FC.py
python 40_CNN_Convolution_ReLU_Pooling_FC_Graphical.py
python 41_CNN_Relu_Pooling_flattern.py
python 42_CNN_Convolution_ReLU_Pooling_FC_Graphical_BigData.py
```

Files 39, 40, 42 will ask: `1` for Vertical Line, `2` for Horizontal Line.
Close each matplotlib window to move to the next step.
File 41 has no plots — just prints to console.

---

*Marvellous Infosystems — Understanding every step of the CNN before using Keras.*
