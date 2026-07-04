# RNN Fundamentals

**Marvellous Infosystems — OJT 2026**
**Trainee : Shubhada A. Palwe**

---

## About This Folder

This folder is where we learn about Recurrent Neural Networks (RNNs). Unlike a regular dense network, RNNs are designed to work with sequences — text, time series, speech — where the order of data matters.

There are 3 files here:
- File 43 : Character prediction — given 3 characters, predict the next one
- File 44 : Sentiment analysis — is this sentence positive or negative?
- File 45 : Number sequence prediction — given the last 5 numbers, predict the next one

---

## Why Do We Need RNNs?

In a normal Dense/CNN layer, each input is treated independently. But for text like "I love this movie", the meaning depends on the order of words. A regular layer doesn't know that "love" comes after "I". RNNs solve this by keeping a hidden state — a kind of memory — that carries information from one step to the next.

```
Time step 1 → "I"    → hidden state h1
Time step 2 → "love" → hidden state h2 (uses h1)
Time step 3 → "this" → hidden state h3 (uses h2)
...
```

Each word is processed one at a time, and the hidden state carries what was seen before.

---

## File 43 — Character Prediction

Trains on the string `"hellohellohellohello"` and learns to predict the next character given 3 characters.

### Step 1 — Build Character Vocabulary

```python
text = "hellohellohellohello"
chars = sorted(list(set(text)))       # unique chars: ['e', 'h', 'l', 'o']
char_to_int = {c: i for i, c in enumerate(chars)}
int_to_char = {i: c for i, c in enumerate(chars)}
```

First we find all unique characters and assign each one an integer. `set(text)` removes duplicates, `sorted()` gives consistent ordering. So `{'e':0, 'h':1, 'l':2, 'o':3}` (approximate).

### Step 2 — Create Sequences

```python
seq_length = 3
for i in range(len(encoded) - seq_length):
    X.append(encoded[i:i+seq_length])   # e.g. [h, e, l]
    y.append(encoded[i+seq_length])     # next: l
```

We slide a window of 3 characters across the text. `X[0] = "hel"`, `y[0] = "l"`. `X[1] = "ell"`, `y[1] = "o"`. And so on.

### Step 3 — Reshape for RNN

```python
X = X.reshape((X.shape[0], X.shape[1], 1))
```

RNN layers expect shape `(samples, timesteps, features)`. Each character is 1 feature, so the last dimension is 1. This is important — if you skip this reshape you get a shape error.

### Step 4 — One-Hot Encode Output

```python
y = to_categorical(y, num_classes=len(chars))
```

The output is one of 4 characters, so we convert the integer labels to one-hot vectors: `2 → [0, 0, 1, 0]`. We use `categorical_crossentropy` as the loss.

### Model

```python
model = Sequential()
model.add(SimpleRNN(16, activation="tanh", input_shape=(seq_length, 1)))
model.add(Dense(len(chars), activation="softmax"))
```

`SimpleRNN(16)` — 16 hidden units. At each of the 3 timesteps, it processes the character and updates its hidden state. After the 3rd step, the hidden state is passed to the Dense layer.

`softmax` output gives a probability for each character. We pick the highest one with `np.argmax(pred)`.

**Expected result:** Input `["h", "e", "l"]` → Predicted: `"l"` (because in "hello", after "hel" comes "l").

---

## File 44 — Sentiment Analysis

Classifies 10 short sentences as Positive (1) or Negative (0).

### Step 1 — Tokenize Text

```python
tokenizer = Tokenizer(num_words=50)
tokenizer.fit_on_texts(sentences)
X = tokenizer.texts_to_sequences(sentences)
print("Word Index:", tokenizer.word_index)
```

Tokenizer converts words to integers. "I love this movie" might become `[1, 2, 3, 4]`. `fit_on_texts` builds the vocabulary. `texts_to_sequences` converts sentences to lists of integers.

`num_words=50` means only the top 50 most frequent words are kept. Rare words beyond that are ignored.

### Step 2 — Pad Sequences

```python
maxlen = 5
X = pad_sequences(X, maxlen=maxlen)
```

Different sentences have different lengths. "I love this movie" has 4 words, "What a fantastic experience" has 4 words too. But we need a fixed-length input for the RNN.

`pad_sequences` adds 0s at the start to make every sequence exactly length 5:
- `[1, 2, 3, 4]` → `[0, 1, 2, 3, 4]` (padded at the front)

### Model

```python
model = Sequential()
model.add(Embedding(input_dim=50, output_dim=8, input_length=maxlen))
model.add(SimpleRNN(8, activation="tanh"))
model.add(Dense(1, activation="sigmoid"))
```

**Embedding layer:** converts each integer word ID into an 8-dimensional vector. So word ID 2 → `[0.23, -0.11, 0.45, ...]` (8 numbers). These vectors are learned during training. This is better than raw integers because it captures meaning.

**SimpleRNN(8):** processes the 5 words one at a time, keeping a hidden state. After the 5th word, the hidden state captures the overall meaning of the sentence.

**Dense(1, sigmoid):** binary output — probability of being Positive.

Loss used: `binary_crossentropy` (same as we used in FNN for pass/fail prediction).

**Test:**
```python
test_sentences = ["I enjoyed this film", "I hated this film"]
```
Expected: Positive, Negative.

---

## File 45 — Number Sequence Prediction

Trains on the sequence 0, 1, 2, ..., 199 and learns to predict the next number given the previous 5.

### Step 1 — Normalize

```python
data = np.arange(0, 200, dtype=np.float32)
data_norm = data / max_val   # divide by 199 → values in [0, 1]
```

We normalize to [0, 1] so the RNN doesn't have to deal with large raw numbers. This makes training more stable.

### Step 2 — Sliding Window

```python
window = 5
for i in range(len(data_norm) - window):
    X.append(data_norm[i:i+window])    # [0,1,2,3,4] normalized
    y.append(data_norm[i+window])      # 5 normalized
```

We use a window of 5 timesteps to predict the 6th. So `X[0] = [0,1,2,3,4]`, `y[0] = 5` (all normalized).

`X = np.array(X)[..., np.newaxis]` — adds the features dimension. Shape becomes `(samples, 5, 1)`.

### Step 3 — Train/Test Split

```python
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
```

80% of the data for training, 20% for validation. This checks that the model didn't just memorize.

### Model

```python
model = Sequential([
    SimpleRNN(32, activation="tanh", input_shape=(window, 1)),
    Dense(1)
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005), loss="mse")
```

No activation on the Dense layer — this is regression (predicting a continuous number).

Loss: `mse` (mean squared error) — same as what we used in FNN regression.

### Predict Next Number

```python
def predict_next(seq):
    seq_norm = (seq / max_val).reshape(1, window, 1)   # normalize
    pred_norm = model.predict(seq_norm, verbose=0)[0, 0]
    return pred_norm * max_val                          # de-normalize
```

Note the de-normalization step. We normalized before training, so predictions come out normalized too. We multiply by `max_val` to get back the actual number.

**Test examples:**
```
Input: [95, 96, 97, 98, 99] → Predicted: ~100
Input: [7, 8, 9, 10, 11]   → Predicted: ~12
Input: [100, 101, 102, 103, 104] → Predicted: ~105
```

---

## Key Concepts

**SimpleRNN hidden state:** At each timestep, `h_t = tanh(W × x_t + U × h_{t-1} + b)`. The current input `x_t` and the previous hidden state `h_{t-1}` are both used. This is the "memory".

**Embedding layer:** Better than using raw integers. Converts word IDs to dense vectors where similar words end up close together in the vector space.

**Padding:** All sequences must be the same length. We pad shorter ones with 0s.

**Tanh activation:** RNNs almost always use tanh (instead of ReLU). Output range is (-1, 1) which prevents the hidden state from growing too large over many timesteps.

**Normalization in sequence tasks:** For numeric sequences, always normalize before feeding to the RNN, and de-normalize after getting predictions.

---

## File Comparison

| Feature | File 43 | File 44 | File 45 |
|---------|---------|---------|---------|
| Task | Char prediction | Sentiment | Number sequence |
| Input type | Characters | Words (text) | Numbers |
| Embedding | No | Yes | No |
| Output | Multi-class (4 chars) | Binary (pos/neg) | Continuous number |
| Loss | categorical_crossentropy | binary_crossentropy | mse |
| Output activation | Softmax | Sigmoid | None (linear) |

---

## How to Run

```bash
pip install tensorflow numpy
python 43_RNN_Character_Prediction.py
python 44_RNN_Sentiment_Analysis.py
python 45_RNN_Sequance_Prediction.py
```

No external datasets needed. All data is defined inside each file.

---

*Marvellous Infosystems — Learning to process sequences, one timestep at a time.*
