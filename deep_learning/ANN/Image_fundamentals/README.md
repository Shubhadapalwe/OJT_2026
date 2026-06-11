# Image Fundamentals — How Computers See Images

## Table of Contents
1. [What This Folder Is About](#what-this-folder-is-about)
2. [Why Images Matter in Deep Learning](#why-images-matter-in-deep-learning)
3. [File 22 — Convert Image to 28×28 Grayscale](#file-22--convert-image-to-2828-grayscale)
4. [File 23 — Read and Display Grayscale Pixel Values](#file-23--read-and-display-grayscale-pixel-values)
5. [File 24 — Read and Display Color Image Pixels](#file-24--read-and-display-color-image-pixels)
6. [Grayscale vs Color — Side-by-Side Comparison](#grayscale-vs-color--side-by-side-comparison)
7. [Connection to CNN Input](#connection-to-cnn-input)
8. [Libraries Used](#libraries-used)

---

## What This Folder Is About

Before building a CNN or any image-based neural network, you must understand one fundamental truth:

> **A computer does not see images. It sees numbers.**

Every image — whether a photograph, a drawing, or a digit — is stored as a grid of numbers called **pixels**. This folder has 3 files that build that understanding step by step:

- File 22 teaches how to take any image and convert + resize it into the standard 28×28 grayscale format used by datasets like MNIST.
- File 23 teaches how to read those pixels as a numpy array and print every single value.
- File 24 extends this to color images and shows how RGB channels work.

---

## Why Images Matter in Deep Learning

When you feed an image into a neural network:
- The network does NOT receive a picture file
- It receives a **matrix of floating-point numbers**
- Each number represents the brightness of one pixel
- For color images, each pixel has 3 numbers (Red, Green, Blue)

This is why image preprocessing — resizing, converting to grayscale, normalizing — is always the first step in any computer vision project.

---

## File 22 — Convert Image to 28×28 Grayscale

### Full Code
```python
from PIL import Image

img = Image.open("digit.png")
img = img.convert("L")      # grayscale
img = img.resize((28, 28))

img.save("digit_28x28.png")

print(img.size)
```

### Line-by-Line Explanation

#### `from PIL import Image`
- **PIL** stands for **Python Imaging Library**.
- The modern version is called **Pillow** (installed via `pip install pillow`), but the import name is still `PIL`.
- `Image` is the main module inside PIL used for all image operations — opening, converting, resizing, saving.

#### `img = Image.open("digit.png")`
- Opens the image file `digit.png` from the current working directory.
- The file could be any format — `.png`, `.jpg`, `.bmp`, `.gif` — PIL handles all of them.
- This does NOT load the full image data into memory immediately; it's a lazy operation. Data is loaded when you actually use `img`.
- `img` is now a PIL Image object with properties like `.size`, `.mode`, `.format`.

#### `img = img.convert("L")`
- Converts the image to **grayscale**.
- `"L"` is a PIL mode code that stands for **Luminance** — a single-channel brightness value.
- Before conversion: image might be RGB (3 channels) or RGBA (4 channels, with transparency).
- After conversion: each pixel has only **1 value** ranging from 0 to 255.

**How the conversion works:**
PIL uses the standard luminance formula:
```
L = 0.299 × R + 0.587 × G + 0.114 × B
```
Green contributes the most (human eyes are most sensitive to green), blue the least.

**PIL Mode Reference:**
| Mode | Description | Channels |
|------|-------------|----------|
| `"RGB"` | Color image | 3 (Red, Green, Blue) |
| `"RGBA"` | Color with transparency | 4 (R, G, B, Alpha) |
| `"L"` | Grayscale (Luminance) | 1 |
| `"1"` | Black and white (binary) | 1 (0 or 255 only) |

#### `img = img.resize((28, 28))`
- Resizes the image to exactly **28 pixels wide and 28 pixels tall**.
- The argument is a tuple `(width, height)` — note: width comes first, then height.
- PIL uses **Lanczos resampling** by default for high-quality downscaling (smooth edges).
- Why 28×28? This is the **standard size for MNIST** — the most famous digit recognition dataset. All CNN demos and tutorials use 28×28 as the baseline size.

**What happens during resize:**
- If your original image is 500×400, PIL reduces it to 28×28.
- Pixel values are averaged/interpolated to represent the original content at the smaller size.

#### `img.save("digit_28x28.png")`
- Saves the converted and resized image to a new file called `digit_28x28.png`.
- PIL automatically detects the format from the file extension (`.png` = lossless, `.jpg` = compressed).
- This file is the output that File 23 reads.

#### `print(img.size)`
- Prints the final size of the image as a tuple.
- Expected output: `(28, 28)`
- Note: PIL's `.size` returns `(width, height)` — **not** `(height, width)`.
- NumPy's `.shape` returns `(height, width)` — the opposite order. This is a common source of confusion.

### Sample Output
```
(28, 28)
```

### What Was Learned From This File
- PIL is the standard Python library for image manipulation.
- `"L"` mode = grayscale. It reduces a 3-channel color image to a single brightness channel.
- 28×28 is the MNIST standard — understanding WHY this size is used connects the code to real datasets.
- `img.save()` writes the processed image to disk so it can be used in later files.
- PIL's `.size` returns `(width, height)` while numpy's `.shape` returns `(height, width)` — always check which convention you're using.

---

## File 23 — Read and Display Grayscale Pixel Values

### Full Code
```python
from PIL import Image
import numpy as np

# Load image
img = Image.open("digit_28x28.png")
img = img.convert("L")
img = img.resize((28, 28))

# Convert image into numpy array
pixels = np.array(img)

# Print shape
print("Image Size :", pixels.shape)

# Print pixel values
print("\nPixel Values:\n")
print(pixels)

# 0   → pure black
# 50  → dark gray
# 120 → medium gray
# 200 → light gray
# 255 → pure white
```

### Line-by-Line Explanation

#### `from PIL import Image` and `import numpy as np`
- PIL is needed to open and convert the image.
- NumPy is needed to convert it into a matrix of numbers that Python can work with mathematically.
- NumPy is the backbone of all scientific computing in Python — it provides fast n-dimensional arrays.

#### `img = Image.open("digit_28x28.png")`
- Loads the image saved by File 22 (`digit_28x28.png`).
- This file is already 28×28 grayscale from the previous step.

#### `img = img.convert("L")` and `img = img.resize((28, 28))`
- These lines are repeated here for safety — if someone runs this file directly without running File 22 first, the image is still converted and resized correctly.
- This is a good defensive programming practice.

#### `pixels = np.array(img)`
- This is the **key step** — it converts a PIL Image object into a **NumPy array**.
- For a grayscale (`"L"` mode) 28×28 image, the result is a **2D array** of shape `(28, 28)`.
- Each element is an integer from 0 to 255 representing the brightness of that pixel.
- After this line, you can do all standard numpy operations — slicing, arithmetic, reshaping, etc.

**What `np.array(img)` produces:**
```
pixels[0][0]   → brightness of top-left pixel
pixels[0][27]  → brightness of top-right pixel
pixels[27][0]  → brightness of bottom-left pixel
pixels[27][27] → brightness of bottom-right pixel
pixels[14][14] → brightness of center pixel
```

#### `print("Image Size :", pixels.shape)`
- `.shape` is a NumPy attribute that returns the dimensions of the array as a tuple.
- For a 28×28 grayscale image: output is `(28, 28)` meaning 28 rows × 28 columns.
- **Important:** NumPy shape is `(rows, columns)` = `(height, width)`.

#### `print(pixels)`
- Prints the full 28×28 matrix of pixel values.
- This is what the neural network actually receives as input.
- You will see a grid of numbers like:
```
[[  0   0   0  12  25 ...]
 [  0   0  15 200 255 ...]
 ...
 [  0   0   0   0   0 ...]]
```

### The Pixel Value Scale

Every pixel in a grayscale image is a number from 0 to 255:

| Value | Appearance | Meaning |
|-------|-----------|---------|
| `0` | ⬛ Pure black | No light at all |
| `50` | 🔲 Dark gray | Very little light |
| `120` | ◼ Medium gray | Half brightness |
| `200` | ◻ Light gray | Most of the way to white |
| `255` | ⬜ Pure white | Maximum brightness |

**Why 0 to 255?**
- Images are stored as **8-bit per channel**.
- 8 bits can store values from 0 to 2⁸−1 = 255.
- This is the standard for almost all consumer digital images (JPEG, PNG).

### Sample Output
```
Image Size : (28, 28)

Pixel Values:

[[  0   0   0   0   0   0   0   0   0   0  ...]
 [  0   0   0   0   0   0   0   0  12  34  ...]
 [  0   0   0   0   0   0  45 190 230 255  ...]
 ...
 [  0   0   0   0   0   0   0   0   0   0  ...]]
```

### What Was Learned From This File
- `np.array(img)` is the bridge between PIL images and numpy — it converts visual data to numerical data.
- A grayscale image becomes a 2D numpy array of shape `(height, width)`.
- Each pixel is a single integer 0–255 representing brightness.
- Printing `pixels` shows exactly what a neural network receives — not an image, but a matrix of numbers.
- NumPy indexing: `pixels[row][col]` — row is Y (top-to-bottom), col is X (left-to-right).

---

## File 24 — Read and Display Color Image Pixels

### Full Code
```python
# ------------------------------------------------------------
# Read Colored Image and Display Output
# ------------------------------------------------------------

from PIL import Image
import numpy as np

# Load image
img = Image.open("color.png")

# Resize image to 28x28
img = img.resize((28, 28))

# Convert image into numpy array
pixels = np.array(img)

# Basic Information
print("\n---------------------------------")
print("IMAGE INFORMATION")
print("---------------------------------")

print("Image Shape :", pixels.shape)
print("Height      :", pixels.shape[0], "Rows")
print("Width       :", pixels.shape[1], "Columns")
print("Channels    :", pixels.shape[2], "(R,G,B)")

total_pixels = pixels.shape[0] * pixels.shape[1]
print("Total Pixels:", total_pixels)

# Display Sample Pixel Values
print("\n---------------------------------")
print("SAMPLE PIXEL VALUES")
print("---------------------------------")

print("Pixel[0][0]   =", pixels[0][0])
print("Pixel[5][10]  =", pixels[5][10])
print("Pixel[10][10] =", pixels[10][10])
print("Pixel[15][15] =", pixels[15][15])
print("Pixel[20][20] =", pixels[20][20])

# Explain One Pixel
print("\n---------------------------------")
print("PIXEL MEANING")
print("---------------------------------")

r = pixels[10][10][0]
g = pixels[10][10][1]
b = pixels[10][10][2]

print("Pixel[10][10] contains:")
print("Red   =", r)
print("Green =", g)
print("Blue  =", b)

# Full Matrix
print("\n---------------------------------")
print("FIRST 5 ROWS OF IMAGE MATRIX")
print("---------------------------------")

print(pixels[:5])

print("\n---------------------------------")
print("FINAL UNDERSTANDING")
print("---------------------------------")
print("Image = Collection of Pixels")
print("Each Pixel = [R,G,B]")
print("Computer sees image as numbers")
print("CNN uses these numbers as input")
```

### Line-by-Line Explanation

#### `img = Image.open("color.png")`
- Opens a **color image** (RGB format — 3 channels: Red, Green, Blue).
- Unlike File 22 and 23, there is NO `.convert("L")` call — this intentionally keeps the image in color.
- PIL will load it as an `"RGB"` image by default for standard color PNGs.

#### `img = img.resize((28, 28))`
- Resizes to 28×28 just like before.
- The color channels are preserved — it's still a 3-channel image, just smaller.

#### `pixels = np.array(img)`
- For a **color** image, this produces a **3D array** — not 2D like grayscale.
- Shape: `(28, 28, 3)` — 28 rows, 28 columns, 3 channels (R, G, B).
- Each pixel is now an array of 3 values: `[R, G, B]`.

**Grayscale vs Color array structure:**
```
Grayscale:  pixels[row][col]        → single integer (brightness)
Color:      pixels[row][col]        → array of 3 integers [R, G, B]
Color:      pixels[row][col][0]     → Red value
Color:      pixels[row][col][1]     → Green value
Color:      pixels[row][col][2]     → Blue value
```

#### `print("Image Shape :", pixels.shape)`
- For a color image: output is `(28, 28, 3)`.
- Three dimensions: height=28, width=28, channels=3.

#### `print("Height      :", pixels.shape[0], "Rows")`
- `pixels.shape[0]` = number of rows = height = 28.

#### `print("Width       :", pixels.shape[1], "Columns")`
- `pixels.shape[1]` = number of columns = width = 28.

#### `print("Channels    :", pixels.shape[2], "(R,G,B)")`
- `pixels.shape[2]` = number of channels = 3 for RGB.
- This is the third dimension added by color images.
- A grayscale image has no third dimension (or it's 1 if explicitly set).

#### `total_pixels = pixels.shape[0] * pixels.shape[1]`
- Calculates total number of pixels: `28 × 28 = 784`.
- Note: this is the number of **pixel locations**, not the number of values.
- The number of values is `784 × 3 = 2352` (each pixel has 3 color values).

#### `print("Pixel[0][0] =", pixels[0][0])`
- Accesses the top-left pixel.
- For a color image this returns an array: e.g., `[255  12  45]`
- That means Red=255, Green=12, Blue=45 at that position.

#### `r = pixels[10][10][0]`, `g = pixels[10][10][1]`, `b = pixels[10][10][2]`
- Extracts the individual Red, Green, Blue values from the pixel at row 10, column 10.
- Index `[0]` = Red channel
- Index `[1]` = Green channel
- Index `[2]` = Blue channel
- This is how you extract a single color component from a specific pixel.

#### `print(pixels[:5])`
- `[:5]` is NumPy slice notation: select rows 0, 1, 2, 3, 4 (first 5 rows).
- Prints a 3D sub-array of shape `(5, 28, 3)`.
- Outputs 5 rows, each row containing 28 pixels, each pixel containing [R, G, B].

#### `print("Image = Collection of Pixels")` ... `print("CNN uses these numbers as input")`
- Summary lines explaining the conceptual connection between images and CNNs.
- A CNN's first layer (`Conv2D`) takes this 3D array as its input.

### Understanding the 3D Color Array

A 28×28 color image looks like this in memory:

```
pixels[0][0]  = [R, G, B]   ← top-left pixel
pixels[0][1]  = [R, G, B]   ← second pixel in first row
pixels[0][27] = [R, G, B]   ← last pixel in first row
pixels[1][0]  = [R, G, B]   ← first pixel in second row
...
pixels[27][27] = [R, G, B]  ← bottom-right pixel
```

Visually, the array has 3 "layers":

```
Layer 0 (Red):    [[R00, R01, ..., R027],
                   [R10, R11, ..., R127],
                   ...
                   [R270, R271, ..., R2727]]

Layer 1 (Green):  [[G00, G01, ..., G027],
                   ...                   ]

Layer 2 (Blue):   [[B00, B01, ..., B027],
                   ...                   ]
```

### Understanding RGB Colors

| Color | R | G | B |
|-------|---|---|---|
| Pure Red | 255 | 0 | 0 |
| Pure Green | 0 | 255 | 0 |
| Pure Blue | 0 | 0 | 255 |
| White | 255 | 255 | 255 |
| Black | 0 | 0 | 0 |
| Yellow | 255 | 255 | 0 |
| Purple | 128 | 0 | 128 |
| Orange | 255 | 165 | 0 |

### Sample Output
```
---------------------------------
IMAGE INFORMATION
---------------------------------
Image Shape : (28, 28, 3)
Height      : 28 Rows
Width       : 28 Columns
Channels    : 3 (R,G,B)
Total Pixels: 784

---------------------------------
SAMPLE PIXEL VALUES
---------------------------------
Pixel[0][0]   = [255  12  45]
Pixel[5][10]  = [100 200  50]
Pixel[10][10] = [180  90  30]
Pixel[15][15] = [ 60 130 200]
Pixel[20][20] = [220  80 160]

---------------------------------
PIXEL MEANING
---------------------------------
Pixel[10][10] contains:
Red   = 180
Green = 90
Blue  = 30

---------------------------------
FIRST 5 ROWS OF IMAGE MATRIX
---------------------------------
[[[255  12  45] [200  10  40] ... ]
 [[240  15  50] [198  12  42] ... ]
 ...
]

---------------------------------
FINAL UNDERSTANDING
---------------------------------
Image = Collection of Pixels
Each Pixel = [R,G,B]
Computer sees image as numbers
CNN uses these numbers as input
```

### What Was Learned From This File
- A color image loaded without `.convert("L")` gives a **3D numpy array** `(height, width, 3)`.
- `pixels.shape[2]` tells you the number of channels (3 for RGB).
- Individual channels accessed by third index: `[0]`=R, `[1]`=G, `[2]`=B.
- `pixels[:5]` slices the first 5 rows — NumPy slicing works on all dimensions.
- `total_pixels = shape[0] * shape[1]` = number of pixel locations (not values).
- The final print statements connect image data to CNN input — CNN literally reads these numbers.

---

## Grayscale vs Color — Side-by-Side Comparison

| Property | Grayscale | Color (RGB) |
|----------|-----------|-------------|
| PIL mode | `"L"` | `"RGB"` |
| Array dimensions | 2D | 3D |
| Array shape | `(28, 28)` | `(28, 28, 3)` |
| Values per pixel | 1 | 3 |
| Total values (28×28) | 784 | 2352 |
| Pixel access | `pixels[row][col]` → integer | `pixels[row][col]` → `[R,G,B]` |
| Red channel access | N/A | `pixels[row][col][0]` |
| Memory per image | 784 bytes | 2352 bytes |
| CNN input_shape | `(28, 28, 1)` | `(28, 28, 3)` |

---

## Connection to CNN Input

When you build a CNN in Keras, the `input_shape` parameter directly corresponds to what these files demonstrate:

```python
# For grayscale images (File 22, 23)
model.add(Conv2D(32, (3,3), input_shape=(28, 28, 1)))

# For color images (File 24)
model.add(Conv2D(32, (3,3), input_shape=(28, 28, 3)))
```

The CNN's first layer (`Conv2D`) slides kernels across the pixel grid — that is, it performs the exact same operation shown in the `cnn_edge_detection` folder, but on real image data and with learned kernels.

**Normalization** — In real CNN projects, pixel values are usually divided by 255 before feeding to the model:
```python
pixels = pixels / 255.0   # Scale to [0.0, 1.0]
```
This is called normalization. It helps gradient descent converge faster because all inputs are in the same small range.

---

## Libraries Used

### `PIL` (Pillow)
Installed via: `pip install pillow`

| Function | What It Does |
|----------|-------------|
| `Image.open(path)` | Opens image from file |
| `img.convert("L")` | Convert to grayscale (Luminance) |
| `img.resize((w, h))` | Resize to given width and height |
| `img.save(path)` | Save image to file |
| `img.size` | Returns `(width, height)` tuple |

### `numpy`
Installed via: `pip install numpy`

| Function/Property | What It Does |
|------------------|-------------|
| `np.array(img)` | Convert PIL image to numpy array |
| `pixels.shape` | Returns `(height, width)` or `(height, width, channels)` |
| `pixels[row][col]` | Access single pixel |
| `pixels[row][col][channel]` | Access single color channel of a pixel |
| `pixels[:5]` | Slice first 5 rows |
| `shape[0] * shape[1]` | Calculate total pixel count |

---

## Key Takeaways

1. **A computer sees numbers, not images.** Every image is a grid of pixel values (0–255).
2. **Grayscale = 2D array** — shape `(H, W)`. One brightness value per pixel.
3. **Color = 3D array** — shape `(H, W, 3)`. Three values per pixel: Red, Green, Blue.
4. **28×28 is the MNIST standard** — used by all digit/fashion recognition benchmarks.
5. **`np.array(img)` converts PIL to numpy** — this is the gateway to all mathematical operations.
6. **PIL's `.size` = `(width, height)`.** NumPy's `.shape` = `(height, width)`. They are reversed.
7. **Normalization** (`/255.0`) is needed before feeding pixel data to a neural network.
8. **`pixels.shape[2]`** tells you how many channels — 3 for color, absent for grayscale.

---

*OJT Training — Marvellous Infosystems | Deep Learning Track*
