# RNN Embedding and Internals

**Marvellous Infosystems — OJT 2026**
**Trainee : Shubhada A. Palwe**

---

## About This Folder

This folder continues from `rnn_text_preprocessing/` (Programs 1–7). There we converted text into padded sequences. Here (Programs 8–14) we go deeper — how does the Embedding layer work, what exactly happens at each time step, what is the hidden state, and how does the RNN actually compute its output mathematically.

| File | Program | Topic |
|------|---------|-------|
| 56 | Program 8 | Vocabulary size calculation |
| 57 | Program 9 | Output labels recap |
| 58 | Program 10 | Manual embedding (word → vector) |
| 59 | Program 11 | Keras Embedding layer |
| 60 | Program 12 | Time steps |
| 61 | Program 13 | Hidden state concept |
| 62 | Program 14 | Manual RNN calculation (h_t formula) |

Same dataset throughout: `["food was good", "food was bad", "food was not good"]`, labels `[1, 0, 0]`.

---

## File 56 — Program 8: Vocabulary Size

```python
vocab_size = len(word_index) + 1
```

We already know `word_index` gives us unique words. Our sentences have 5 unique words → `len(word_index) = 5`. But `vocab_size = 6`.

**Why + 1?** Because index 0 is reserved for padding. The Embedding layer needs to know about index 0 (the padding token), but it doesn't appear in `word_index`. So we always add 1.

```
word_index = {'food': 1, 'was': 2, 'good': 3, 'bad': 4, 'not': 5}
Unique words  = 5
Padding index = 0
vocab_size    = 5 + 1 = 6
```

This `vocab_size` is what we pass as `input_dim` to the Embedding layer in Program 11.

---

## File 57 — Program 9: Output Labels

Recaps the labels with a clearer if/else explanation:

```python
labels = [1, 0, 0]

for sentence, label in zip(sentences, labels):
    if label == 1:
        print("Meaning : Positive Sentiment")
    else:
        print("Meaning : Negative Sentiment")
```

This is a **binary classification** problem. The output layer of the RNN will have `Dense(1, activation='sigmoid')` — producing a value between 0 and 1. Above 0.5 → Positive. Below 0.5 → Negative.

Note: "food was not good" is Negative (0) even though it contains "good". The word "not" flips the meaning. A standard bag-of-words model might misclassify this because it ignores order. The RNN handles it correctly because it reads "not" before "good".

---

## File 58 — Program 10: Manual Embedding

Shows what an Embedding layer actually does — maps each word index to a fixed-size vector of numbers.

```python
embedding = {
    0: [0.0, 0.0, 0.0],   # padding
    1: [0.2, 0.4, 0.1],   # food
    2: [0.3, 0.1, 0.5],   # was
    3: [0.8, 0.7, 0.9],   # good
    4: [0.1, 0.2, 0.9],   # bad
    5: [0.9, 0.3, 0.2]    # not
}

sequence = [1, 2, 5, 3]   # "food was not good"
```

For each token in the sequence, we look up its vector:

```
Token 1 (food) → [0.2, 0.4, 0.1]
Token 2 (was)  → [0.3, 0.1, 0.5]
Token 5 (not)  → [0.9, 0.3, 0.2]
Token 3 (good) → [0.8, 0.7, 0.9]
```

The sentence `[1, 2, 5, 3]` becomes a 4×3 matrix — 4 words, each with a 3-number vector. The RNN reads one row at a time.

In the actual Keras model, these vectors are not hardcoded — they start random and are **learned during training** along with all the other weights. Words used in similar contexts end up with similar vectors.

---

## File 59 — Program 11: Keras Embedding Layer

Shows the real Keras Embedding layer and what it actually outputs.

```python
vocab_size = len(tokenizer.word_index) + 1   # 6
X = pad_sequences(sequences, maxlen=4, padding="pre")

embedding_model = Sequential()
embedding_model.add(Embedding(input_dim=vocab_size, output_dim=4))
embedding_model.build(input_shape=(None, max_length))
```

`input_dim=vocab_size` — how many unique tokens the embedding table has rows for (6 in our case).
`output_dim=4` — each token gets mapped to a vector of size 4.

After `model.predict(X)`, the output shape is `(3, 4, 4)`:
- 3 = number of sentences
- 4 = number of time steps (max_length)
- 4 = embedding vector size (output_dim)

So each sentence becomes a 4×4 matrix. The RNN then processes this matrix one row at a time (one time step at a time).

The file prints each position's token and its 4-number vector for the first sentence. These vectors are random since we only built an Embedding model without training it on sentiment.

---

## File 60 — Program 12: Time Steps

Makes the "time step" idea concrete:

```python
sentence = "food was not good"
words = sentence.split()

for index, word in enumerate(words):
    print("Time Step", index + 1, ":", word)
```

Output:
```
Time Step 1 : food
Time Step 2 : was
Time Step 3 : not
Time Step 4 : good
```

Each word = one time step. The RNN doesn't see all words at once — it reads them one by one, left to right. At Time Step 3 it knows about "food", "was", and "not". At Time Step 4 it knows about all 4 words.

This is the fundamental difference from a Dense layer, which sees all inputs at once with no concept of order.

---

## File 61 — Program 13: Hidden State Concept

Simulates the hidden state (memory) in plain text — no math, just the concept.

```python
hidden_state = "empty memory"

for index, word in enumerate(words):
    print("Current Word   :", word)
    print("Previous Memory:", hidden_state)
    hidden_state = "memory after reading '" + " ".join(words[:index+1]) + "'"
    print("Updated Memory :", hidden_state)
```

Output:
```
Time Step 1
Current Word    : food
Previous Memory : empty memory
Updated Memory  : memory after reading 'food'

Time Step 2
Current Word    : was
Previous Memory : memory after reading 'food'
Updated Memory  : memory after reading 'food was'
...
```

The hidden state carries a summary of everything seen so far. After the last word, the final hidden state represents the meaning of the whole sentence. This is what the Dense output layer uses to make the Positive/Negative prediction.

---

## File 62 — Program 14: Manual RNN Calculation

Shows the actual math. This is the most important file in this folder.

```python
inputs = [1, 2, 5, 3]   # token sequence for "food was not good"
hidden_state = 0         # initial hidden state

Wx = 0.5    # weight for current input
Wh = 0.8    # weight for previous hidden state
bias = 0.1

# Formula: h_t = tanh(Wx * x_t + Wh * h_{t-1} + bias)

for time_step, x in enumerate(inputs):
    previous_hidden_state = hidden_state
    total = Wx * x + Wh * previous_hidden_state + bias
    hidden_state = np.tanh(total)
```

Let's trace through manually:

**Time Step 1** (x=1, h_prev=0):
```
total = 0.5×1 + 0.8×0 + 0.1 = 0.6
h_t   = tanh(0.6) ≈ 0.537
```

**Time Step 2** (x=2, h_prev=0.537):
```
total = 0.5×2 + 0.8×0.537 + 0.1 = 1.53
h_t   = tanh(1.53) ≈ 0.909
```

**Time Step 3** (x=5, h_prev=0.909):
```
total = 0.5×5 + 0.8×0.909 + 0.1 = 3.327
h_t   = tanh(3.327) ≈ 0.997
```

**Time Step 4** (x=3, h_prev=0.997):
```
total = 0.5×3 + 0.8×0.997 + 0.1 = 2.398
h_t   = tanh(2.398) ≈ 0.983
```

The final hidden state (`≈0.983`) is fed to the Dense output layer for classification.

Note: this uses raw token integers, not embedding vectors. In a real Keras RNN, the input at each step is an embedding vector (e.g., 4 numbers for `output_dim=4`), not a single integer. The weight matrices would be larger accordingly. But the principle is exactly the same.

**`tanh` produces values in (-1, 1)** — this keeps the hidden state bounded even after many timesteps. This is better than letting values grow unboundedly.

---

## How the Pieces Fit Together

```
Padded Sequence  →  Embedding Layer  →  RNN Time Steps  →  Hidden State  →  Dense Output
[0, 1, 2, 5, 3]     Each token           h_t computed        Final h_t        0 or 1
                     → 4D vector          at each step        = sentence       (Neg/Pos)
                                          using formula        summary
```

---

## How to Run

```bash
# Files 57, 60, 61 — no imports needed
python 57_RNN_OutputLabels.py
python 60_RNN_TimeSteps.py
python 61_RNN_HiddnState.py

# Files 56, 58, 62 — need numpy only
python 56_RNN_VocabularySize.py
python 58_RNN_EmbeddingManual.py
python 62_RNN_ManualCalculations.py

# File 59 — needs TensorFlow
pip install tensorflow
python 59_RNN_EmbeddingKeras.py
```

---

*Marvellous Infosystems — Understanding what happens inside the RNN, one calculation at a time.*
