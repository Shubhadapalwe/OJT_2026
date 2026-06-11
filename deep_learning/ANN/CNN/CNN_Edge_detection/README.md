# CNN Edge Detection — Manual Convolution to Real Image

## Overview

This folder builds CNN convolution understanding from scratch — no Keras, no TensorFlow, just numpy and raw math. Starting from a hand-crafted 6×6 image and a 3×3 kernel, I manually implemented convolution, then progressively added step-by-step display, animation, and finally applied real edge detection on an actual image using OpenCV.

---

## Files in This Folder

| File | Topic |
|------|-------|
| `25_CNN_Edge_Detection.py` | Manual convolution on 6×6 image — numpy only |
| `26_CNN_Edge_Detection_Result_Display.py` | Same + step-by-step print + matplotlib visualization |
| `27_CNN_Edge_Detection_Result_Animation.py` | Same + animated kernel sliding with red rectangle |
| `28_CNN_Edge_Detection_Cat.py` | Real image edge detection using OpenCV Canny |

---

## The Core Concept — Convolution

A convolution operation slides a kernel (filter) over an image, computing the dot product at each position. The result is a **feature map** that highlights specific patterns like edges.

```
Output size formula:
  output = (image_size - kernel_size + 1)
  6×6 image, 3×3 kernel → 4×4 feature map
```

---

## File 25 — Manual Convolution (Pure NumPy)

### What It Does
Creates a 6×6 image (top-half black, bottom-half white) and applies a horizontal edge-detection kernel manually using nested loops.

### Key Code
```python
import numpy as np

image = np.array([
    [0,   0,   0,   0,   0,   0],
    [0,   0,   0,   0,   0,   0],
    [0,   0,   0,   0,   0,   0],
    [255, 255, 255, 255, 255, 255],
    [255, 255, 255, 255, 255, 255],
    [255, 255, 255, 255, 255, 255]
])

kernel = np.array([
    [-1, -1, -1],
    [ 0,  0,  0],
    [ 1,  1,  1]
])

feature_map = np.zeros((4, 4))

for i in range(4):
    for j in range(4):
        region = image[i:i+3, j:j+3]
        feature_map[i][j] = np.sum(region * kernel)

print(feature_map)
```

### Why This Kernel?
The `-1, 0, +1` pattern detects horizontal edges — negative weights on top, positive on bottom. Where a bright-to-dark transition exists, the result is large (positive or negative).

---

## File 26 — Step-by-Step Display + Matplotlib

### What It Does
Same convolution as File 25, but with detailed step-by-step console output showing the selected region, element-wise multiplication, and running sum at each kernel position. Adds matplotlib visualization of the original image, kernel, and feature map.

### New Additions
```python
def Marvellous_Print_Matrix(title, matrix):
    print("\n" + "-" * 50)
    print(title)
    print(matrix)

# Shows every step:
# - "Kernel Position -> Row: 0 to 2, Column: 0 to 2"
# - Current 3x3 Region
# - Region * Kernel (element-wise)
# - Sum of all values
# - Feature Map Built So Far
```

### What I Learned
- Visualizing each step made the math tangible — I could see which region the kernel was "looking at" and why the edge lit up.
- High feature map values (like 765) appear exactly at the edge row between black and white.

---

## File 27 — Animated Kernel Sliding

### What It Does
The most visual version — for every kernel position, it renders a 4-panel matplotlib figure showing:
1. Original image with a **red rectangle** highlighting the active 3×3 region
2. Current 3×3 region extracted
3. The kernel
4. Feature map built so far

```python
from matplotlib.patches import Rectangle

# Red box showing current 3x3 region
rect = Rectangle((j - 0.5, i - 0.5), 3, 3,
                 linewidth=3, edgecolor='red', facecolor='none')
axes[0].add_patch(rect)
```

### What I Learned
- `Rectangle((x, y), width, height)` in matplotlib draws on the axes — the offset `-0.5` centers it on the cell.
- Seeing the kernel slide step-by-step made convolution feel intuitive, not abstract.
- 16 total positions (4×4 output) = 16 animation frames.

---

## File 28 — Real Image Edge Detection with OpenCV

### What It Does
Moves from synthetic 6×6 images to a real photograph. Uses OpenCV's `Canny` edge detection (which internally applies Gaussian blur → gradient → non-maximum suppression → hysteresis thresholding).

### Key Code
```python
import cv2

img = cv2.imread("sample.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Canny edge detection: low_threshold=100, high_threshold=200
edges = cv2.Canny(gray, 100, 200)

cv2.imshow("Original Image", img)
cv2.imshow("Grayscale Image", gray)
cv2.imshow("Edges", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Canny Thresholds
| Parameter | Role |
|-----------|------|
| `100` (low threshold) | Weak edges below this are discarded |
| `200` (high threshold) | Strong edges above this are kept |
| Between 100–200 | Kept only if connected to a strong edge |

### What I Learned
- `cv2.COLOR_BGR2GRAY` — OpenCV loads images as BGR (not RGB), so conversion order matters.
- `cv2.Canny` replaces our manual kernel with an industry-standard multi-step algorithm.
- `cv2.waitKey(0)` keeps the window open until a key is pressed.
- `cv2.destroyAllWindows()` is always needed to clean up the display windows.

---

## Progression in This Folder

```
25: Raw math — numpy convolution, no display
         ↓
26: Add print output + matplotlib static plots
         ↓
27: Add animation — red rectangle sliding over image
         ↓
28: Real image + OpenCV Canny (production-level tool)
```

---

## Libraries Used
- `numpy` — manual convolution math
- `matplotlib` — visualizing image, kernel, feature map
- `matplotlib.patches.Rectangle` — drawing the sliding kernel highlight
- `cv2` (OpenCV) — loading real images, Canny edge detection

---

## Key Takeaway

Convolution is just sliding a kernel across an image and computing dot products. Doing it manually with numpy first makes it clear why CNNs are powerful — the same operation, repeated with learned kernels, can detect faces, shapes, textures, and objects.

---

*OJT Training — Marvellous Infosystems | Deep Learning Track*
