# RNN Sentiment Model

**Marvellous Infosystems — OJT 2026**
**Trainee : Shubhada A. Palwe**

---

## About This Folder

This is the third and final part of the RNN text series:

| Folder | Programs | What it covers |
|--------|----------|---------------|
| `rnn_text_preprocessing/` | 1–7 | Split → tokenize → vocab → sequences → padding |
| `rnn_embedding_internals/` | 8–14 | Vocab size, embedding, timesteps, hidden state, manual RNN math |
| `rnn_sentiment_model/` | 15–21 + graphical | Activations, loss, architecture, train, predict, complete app |

By the end of this folder we have a fully working RNN that reads restaurant review sentences and classifies them as Positive or Negative.

---

## File 63 — Program 15: Tanh Activation

```python
values = [-3, -1, 0, 1, 3]
for value in values:
    print("tanh :", np.tanh(value))
```

Output:
```
tanh(-3) = -0.9951
tanh(-1) = -0.7616
tanh(0)  =  0.0
tanh(1)  =  0.7616
tanh(3)  =  0.9951
```

`tanh` squashes any value to the range `(-1, +1)`. It's used inside SimpleRNN to control the hidden state. The key point: no matter how large the input gets, tanh never goes above 1 or below -1. This prevents the hidden state from exploding over many timesteps.

Note: tanh is symmetric around 0. For RNNs this is better than sigmoid (which is between 0 and 1) because centring around 0 helps gradients flow better during training.

---

## File 64 — Program 16: Sigmoid Activation

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

for value in values:
    result = sigmoid(value)
    decision = "Positive" if result >= 0.5 else "Negative"
```

Output:
```
sigmoid(-3) = 0.047  → Negative
sigmoid(-1) = 0.269  → Negative
sigmoid(0)  = 0.5    → Positive (boundary)
sigmoid(1)  = 0.731  → Positive
sigmoid(3)  = 0.953  → Positive
```

Sigmoid is used at the **output** layer for binary classification. It converts any number into a probability between 0 and 1. The decision boundary is at 0.5:
- `>= 0.5` → Positive (label 1)
- `< 0.5` → Negative (label 0)

Tanh is used for the hidden state (RNN internals). Sigmoid is used for the final output. Both squeeze values, but different ranges.

---

## File 65 — Program 17: Binary Cross Entropy Loss

```python
def binary_crossentropy(actual, predicted):
    return -(actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted))
```

6 examples are tested:

```
actual=1, predicted=0.90 → Loss = small  (correct, confident)
actual=1, predicted=0.60 → Loss = medium (correct, less confident)
actual=1, predicted=0.20 → Loss = large  (wrong, confident)
actual=0, predicted=0.10 → Loss = small  (correct, confident)
actual=0, predicted=0.40 → Loss = medium (correct, less confident)
actual=0, predicted=0.80 → Loss = large  (wrong, confident)
```

When actual=1: only `−log(predicted)` matters. `−log(0.9) ≈ 0.1` (small). `−log(0.2) ≈ 1.6` (large).

When actual=0: only `−log(1−predicted)` matters. `−log(0.9) ≈ 1.6` (large, wrong). `−log(0.9) ≈ 0.1` (small, correct).

This is what Keras uses when we write `loss="binary_crossentropy"`. Adam optimizer then adjusts weights to make this loss smaller each epoch.

---

## File 66 — Program 18: RNN Architecture

```python
vocab_size = 6
max_length = 4

model = Sequential()
model.add(Embedding(input_dim=6, output_dim=8))
model.add(SimpleRNN(units=8, activation="tanh", return_sequences=False))
model.add(Dense(1, activation="sigmoid"))
model.build(input_shape=(None, max_length))
model.summary()
```

**Architecture flow printed at the end:**
```
Input Sentence
→ Tokenization
→ Padding
→ Embedding Layer      (6 → 8: each of 6 tokens maps to 8-dim vector)
→ SimpleRNN Layer      (reads 4 timesteps, outputs final hidden state of 8 values)
→ Dense Layer          (8 → 1: single score)
→ Sigmoid Output       (score → probability)
```

`return_sequences=False` (default) means we only get the hidden state from the **last** timestep. We don't need all 4 intermediate hidden states — just the final one which summarises the whole sentence.

`model.build(input_shape=(None, max_length))` — `None` means any batch size. `max_length=4` means each input has 4 timesteps.

---

## File 67 — Program 19: Compile and Train

```python
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(X, y, epochs=800, verbose=0)

print("Final Loss    :", history.history["loss"][-1])
print("Final Accuracy:", history.history["accuracy"][-1])
```

`verbose=0` hides epoch-by-epoch output. We only see the final result.

**Why 800 epochs?** We only have 3 training sentences. With so little data, the model needs more rounds to converge. With a real dataset of thousands of sentences, even 10–20 epochs would be enough.

`history.history["loss"]` is a list of loss values — one per epoch. `[-1]` gives the last one.

The model should reach near-100% accuracy on the 3 training sentences.

---

## File 68 — Program 20: Prediction Flow

This is the cleanest file for understanding what happens when we predict on a new sentence. It shows the 5 steps inside the `Predict_Sentiment()` function:

```python
def Predict_Sentiment(text):
    # Step 1: text → sequence
    sequence = tokenizer.texts_to_sequences([text])
    # Step 2: show word → number mapping
    for word in text.split():
        print(f"{word} → {tokenizer.word_index[word]}")
    # Step 3: pad
    padded = pad_sequences(sequence, maxlen=4, padding="pre")
    # Step 4: predict (sigmoid value)
    prediction = model.predict(padded, verbose=0)
    # Step 5: decision (>= 0.5 → Positive)
    sentiment = "Positive" if prediction[0][0] >= 0.5 else "Negative"
```

Called for all 3 training sentences:
- `"food was good"` → Positive ✓
- `"food was bad"` → Negative ✓
- `"food was not good"` → Negative ✓

The important thing: the same tokenizer that was used for training is used at prediction time. If we created a new tokenizer, word indices would be different and predictions would be garbage.

---

## File 69 — Program 21: Complete Code (All Steps)

This is the clean, production-ready version. All steps in one file, no splits:

```
Step 1: Dataset (sentences + labels)
Step 2: Tokenization
Step 3: Text to sequences
Step 4: Padding
Step 5: Vocabulary size
Step 6: Build model
Step 7: Compile
Step 8: Train (800 epochs)
Step 9: Summary + print word index + print X and y
Step 10: Predict on all 3 training sentences
```

This is the file to share or submit. It contains everything needed from raw text to final prediction. The previous files (49–68) were teaching each step individually. This one does all of it cleanly in sequence.

---

## File 70 — Graphical Version (Full Mathematical Demo)

The most detailed file in the entire series. Goes through every calculation with matplotlib plots.

### 9 Steps:

**Step 1:** Input sentence `"food was not good"`, words split.

**Step 2:** Manual tokenization with `word_index` dictionary.

**Step 3:** Embedding lookup — shows the full 6×3 embedding matrix and does `embedding_matrix[token]` for each word. Shows resulting 4×3 input matrix `X`. Also plots it as a heatmap.

**Step 4:** RNN weight matrices explained:
```python
Wx = 3×3   # input weights (embedding_dim=3 → hidden_dim=3)
Wh = 3×3   # recurrent weights (hidden_dim → hidden_dim)
b  = 3     # bias vector
```

**Step 5:** Full RNN forward pass — at each of the 4 timesteps, shows:
- `Wx . x_t` computed row by row with every multiplication shown
- `Wh . h_previous` computed row by row
- Addition before tanh: `input_part + memory_part + b`
- `tanh(total)` for each dimension

Also plots hidden state values as a heatmap.

**Step 6:** Hidden state flow — shows how h changes from `[0,0,0]` after each word. Plots a flow diagram with arrows between word boxes.

**Step 7:** Dense layer:
```python
Wy = [-2.0, -1.5, -1.2]   # output weights
by = 1.0                    # output bias

z = Wy . h_final + by
```
Shows element-wise multiplication and sum. Plots a bar chart of each contribution to z.

**Step 8:** Sigmoid applied to z — shows manual `e^(-z)` calculation step by step. Plots the sigmoid curve with the actual point marked.

**Step 9:** Final decision: `prediction >= 0.5` → Positive / Negative.

**Step 10:** Summary figure — one matplotlib panel with the entire flow from sentence to prediction.

---

## The Three-Folder Series — Complete Picture

```
rnn_text_preprocessing/   (Programs 1–7)
 → Raw text → tokenize → pad → fixed-length integer sequences

rnn_embedding_internals/  (Programs 8–14)
 → vocab_size, embedding vectors, timesteps, hidden state, manual h_t formula

rnn_sentiment_model/      (Programs 15–21 + graphical)
 → tanh, sigmoid, BCE loss → build → compile → train → predict → complete app
```

---

## How to Run

```bash
pip install tensorflow numpy matplotlib

# Activation functions (no TF)
python 63_RNN_Tanh.py
python 64_RNN_Sigmoid.py
python 65_RNN_BinaryCrossEntropyLoass.py

# Model files (TF required)
python 66_RNN_Architecture.py
python 67_RNN_TrainModel.py
python 68_RNN_PredictionFlow.py
python 69_RNN_CompleteCode.py
python 70_RNN_Graphical_Version.py
```

Run File 69 if you want the complete working app in one shot. Run File 70 for the full graphical mathematical explanation.

---

*Marvellous Infosystems — From raw sentences to a trained RNN that understands sentiment.*
