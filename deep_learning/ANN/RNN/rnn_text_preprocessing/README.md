# RNN Text Preprocessing


**Trainee : Shubhada A. Palwe**

---

## About This Folder

Before we feed text into an RNN, we have to convert it into numbers — because neural networks can only work with numbers, not words. This folder teaches that conversion process step by step.

There are 7 files here (Programs 1–7), each building on the previous one:

| File | Program | What it teaches |
|------|---------|----------------|
| 49 | Program 1 | Dataset and labels |
| 50 | Program 2 | Split sentence into words |
| 51 | Program 3 | Manual vocabulary creation |
| 52 | Program 4 | Keras Tokenizer (automatic vocab) |
| 53 | Program 5 | Text to sequences |
| 54 | Program 6 | Why padding is needed |
| 55 | Program 7 | Padding with Keras |

All 7 files use the same 3 sentences so we can see exactly how each step transforms the data.

---

## The Dataset Used in All Files

```python
sentences = [
    "food was good",    # label 1 → Positive
    "food was bad",     # label 0 → Negative
    "food was not good" # label 0 → Negative
]
labels = [1, 0, 0]
```

"food was not good" is Negative (0) because "not" reverses the meaning. This is why simple bag-of-words models struggle — word order matters. RNNs read words in sequence, so they can learn this.

---

## File 49 — Program 1: Dataset and Labels

This is the starting point. Just prints the sentences and labels with their meaning.

```python
for sentence, label in zip(sentences, labels):
    sentiment = "Positive" if label == 1 else "Negative"
    print("Sentence :", sentence)
    print("Label    :", label)
    print("Meaning  :", sentiment)
```

`zip(sentences, labels)` pairs each sentence with its label. The output shows us what we're trying to teach the RNN — for each sentence, is it positive (1) or negative (0)?

---

## File 50 — Program 2: Split Sentence into Words

Shows the very first step of text processing: splitting a sentence into individual words.

```python
sentence = "food was not good"
words = sentence.split()
# → ['food', 'was', 'not', 'good']
```

`str.split()` splits on whitespace by default. No arguments needed.

Output:
```
Position 1 : food
Position 2 : was
Position 3 : not
Position 4 : good
```

Conclusion printed in the file: "RNN processes sentence one word at a time." This is the key idea — each word is one timestep.

---

## File 51 — Program 3: Manual Vocabulary Creation

Builds the vocabulary (unique words) by hand, without any library.

```python
vocabulary = []

for sentence in sentences:
    words = sentence.split()
    for word in words:
        if word not in vocabulary:
            vocabulary.append(word)
```

This loops through every word in every sentence and adds it to the list only if it's not already there. Result: `['food', 'was', 'good', 'bad', 'not']` — 5 unique words.

Then assigns an index starting from 1:

```python
for index, word in enumerate(vocabulary, start=1):
    print(f"{word:10s} ---> {index}")
```

`enumerate(vocabulary, start=1)` gives pairs like `(1, 'food')`, `(2, 'was')`, etc. Starting from 1 because 0 is reserved for padding (we'll see this in File 54).

This manual approach helps us understand what Keras Tokenizer does internally.

---

## File 52 — Program 4: Keras Tokenization

Does the same thing as File 51 but automatically, using `Tokenizer`.

```python
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)

word_index = tokenizer.word_index
```

`fit_on_texts(sentences)` — scans all sentences, counts word frequencies, builds the vocabulary. More frequent words get lower index numbers.

`tokenizer.word_index` — a dictionary like `{'food': 1, 'was': 2, 'good': 3, 'bad': 4, 'not': 5}`.

Note: Keras Tokenizer sorts by frequency, so `food` gets index 1 (appears 3 times), `was` gets 2 (appears 3 times), and so on. The manual vocabulary in File 51 assigns indices in the order words were seen — so the order might differ.

---

## File 53 — Program 5: Text to Sequences

Converts each full sentence into a list of numbers.

```python
sequences = tokenizer.texts_to_sequences(sentences)
```

Each word is replaced with its index from `word_index`:

```
Sentence : food was good       → Sequence : [1, 2, 3]
Sentence : food was bad        → Sequence : [1, 2, 4]
Sentence : food was not good   → Sequence : [1, 2, 5, 3]
```

Now the text is numbers. But notice: the sequences have different lengths (3, 3, 4). This is the problem File 54 explains.

---

## File 54 — Program 6: Why Padding is Needed

Shows the problem directly — different length sequences.

```python
sequences = [
    [1, 2, 3],    # length 3
    [1, 2, 4],    # length 3
    [1, 2, 5, 3]  # length 4
]
```

Neural networks process input in batches. In a batch, all inputs must be the same shape. A 2D array like `[[1,2,3],[1,2,4],[1,2,5,3]]` can't even be created as a numpy array because the rows have different lengths.

The solution: add 0s to shorter sequences so all are the same length. This is called **padding**.

---

## File 55 — Program 7: Padding with Keras

Applies `pad_sequences` to fix the length problem.

```python
from tensorflow.keras.preprocessing.sequence import pad_sequences

max_length = 4

padded_sequences = pad_sequences(
    sequences,
    maxlen=max_length,
    padding="pre"
)
```

`maxlen=4` — all sequences become length 4.

`padding="pre"` — zeros are added at the **beginning** (pre = before). This is the default and usually preferred for RNNs because the actual words are at the end, closer to when the RNN makes its prediction.

Result:

```
Sentence          : food was good
Original Sequence : [1, 2, 3]
Padded Sequence   : [0, 1, 2, 3]    ← one 0 added at start

Sentence          : food was bad
Original Sequence : [1, 2, 4]
Padded Sequence   : [0, 1, 2, 4]    ← one 0 added at start

Sentence          : food was not good
Original Sequence : [1, 2, 5, 3]
Padded Sequence   : [1, 2, 5, 3]    ← already length 4, no change
```

Now all sequences are length 4 and can be stacked into a proper numpy array of shape `(3, 4)`.

`padding="post"` adds zeros at the end instead. Both work, but `pre` is more common for text classification.

---

## The Full Pipeline (Programs 1→7)

This is the complete text-to-numbers process we follow before training any RNN:

```
Raw Sentences
     ↓
Split into words (Program 2)
     ↓
Build vocabulary — unique word list (Program 3, 4)
     ↓
Convert each word to its index number (Program 4, 5)
     ↓
Recognize that sequences have different lengths (Program 6)
     ↓
Pad with zeros to make all sequences equal length (Program 7)
     ↓
Ready to feed into RNN / Embedding layer
```

After File 55, the padded array is what goes into `model.fit()`.

---

## How to Run

```bash
# Files 49, 50, 51, 54 — no imports needed
python 49_RNN_DatasetDisplay.py
python 50_RNN_Tokenization.py
python 51_RNN_VocabularyCreation.py
python 54_RNN_UseOfPadding.py

# Files 52, 53, 55 — need TensorFlow
pip install tensorflow
python 52_RNN_KerasTokenization.py
python 53_RNN_TexttoSequences.py
python 55_RNN_PaddingKeras.py
```

Run them in order (49 → 55). Each one shows one step of the pipeline.

---

## Quick Reference

| Concept | Code | What it does |
|---------|------|-------------|
| Split to words | `sentence.split()` | `"food was good"` → `['food','was','good']` |
| Build vocab | `Tokenizer().fit_on_texts()` | Creates `word_index` dictionary |
| Word → number | `texts_to_sequences()` | `"food was good"` → `[1, 2, 3]` |
| Equal length | `pad_sequences(X, maxlen=4)` | Adds 0s to shorter sequences |
| pre padding | `padding="pre"` | Zeros added at the start |
| post padding | `padding="post"` | Zeros added at the end |

---


