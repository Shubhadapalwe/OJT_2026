# Transformer Fundamentals

**Marvellous Infosystems — OJT 2026**
**Trainee : Shubhada A. Palwe**

---

## About This Folder

Transformers are the architecture behind ChatGPT, BERT, and most modern NLP. Unlike RNNs that read text one word at a time, Transformers look at all words at once using a mechanism called **Self-Attention**.

There are 8 files here — starting with manual numpy calculations and ending with three complete applications:

| File | Topic |
|------|-------|
| 71 | Self-Attention — full manual calculation step by step |
| 72 | Multi-Head Attention — two heads, different views |
| 73 | Positional Encoding — sin/cos formula |
| 74 | Transformer Encoder — complete encoder block |
| 75 | Transformer Decoder — masked attention + encoder-decoder attention |
| 76 | Sentiment Classification — Encoder-only (like BERT) |
| 77 | Neural Machine Translation — Encoder-Decoder (English → Marathi) |
| 78 | GPT-style Text Generation — Decoder-only |

---

## The Big Idea: Why Transformers?

RNN problem: "The cat, which chased the mouse that lived in the wall, sat on the mat." By the time the RNN reaches "sat", it has almost forgotten "cat" because it's so far back.

Transformer solution: Every word looks at every other word at the same time. "sat" can directly compare itself to "cat" no matter how far apart they are. This is Self-Attention.

---

## File 71 — Self-Attention (Manual)

Uses sentence `["I", "love", "AI"]` with 2D embeddings.

### Step 1: Input Embeddings
```
I    → [1, 0]
love → [0, 1]
AI   → [1, 1]
```

### Step 2: Weight Matrices
Three weight matrices: `Wq`, `Wk`, `Wv` (all identity here for simplicity).

- **Wq** → converts embedding to **Query** (what am I looking for?)
- **Wk** → converts embedding to **Key** (what do I contain?)
- **Wv** → converts embedding to **Value** (what do I give away?)

```python
Q = X @ Wq    # Query matrix
K = X @ Wk    # Key matrix
V = X @ Wv    # Value matrix
```

### Step 3: Attention Scores
```python
scores = Q @ K.T   # shape: (3, 3) — every word vs every word
```
Each value `scores[i][j]` measures how much word `i` should attend to word `j`.

### Step 4: Scale
```python
scaled_scores = scores / np.sqrt(d_k)   # d_k = 2 here
```
We divide by `sqrt(d_k)` to prevent scores from growing too large. Large scores make softmax produce near-zero gradients (training slows down).

### Step 5: Softmax (row-wise)
```python
attention_weights = softmax(scaled_scores)   # each row sums to 1
```
Now each row is a probability distribution — how much attention each word pays to every other word.

### Step 6: Output
```python
output = attention_weights @ V
```
Each word's output is a **weighted mix of all Value vectors**. A word that attended 80% to "AI" and 20% to itself gets 80% of AI's values and 20% of its own.

The file prints every single multiplication in Step 3 and Step 6, so we can trace the exact numbers.

**Key insight:** Before attention, "I" only knew about itself. After attention, "I" knows about "love" and "AI" too. Every word becomes **context-aware**.

---

## File 72 — Multi-Head Attention

Same sentence, but now embedding size is 4 and we run **two attention heads** in parallel.

```python
# Head 1: identity weights — sees input as-is
Wq1 = Wk1 = Wv1 = np.eye(4)

# Head 2: feature-swapping weights — sees input from different angle
Wq2 = Wk2 = Wv2 = np.array([[0,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,1,0]])
```

Each head runs the full attention pipeline (Q, K, V → scores → scale → softmax → output) independently.

**Concatenation:**
```python
final_output = np.concatenate((head1_output, head2_output), axis=1)
# shape: (3, 4) + (3, 4) → (3, 8)
```

Why multiple heads? One head might learn syntax (subject-verb agreement). Another might learn semantics (what words mean). Together they capture richer relationships. Real BERT uses 12 heads. GPT-3 uses 96.

---

## File 73 — Positional Encoding

Transformers have no built-in sense of word order (unlike RNNs). We add position information using a sin/cos formula.

```python
def positional_encoding(seq_len, d_model):
    PE = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            angle = pos / (10000 ** (i / d_model))
            PE[pos, i]   = np.sin(angle)   # even dimensions → sin
            PE[pos, i+1] = np.cos(angle)   # odd dimensions → cos
    return PE
```

For position 0: angle = 0, so `sin(0)=0`, `cos(0)=1`.
For position 1: angles vary by dimension — different frequencies.
For position 2: angles are 2× larger than position 1.

**Why sin and cos?** They produce unique patterns for every position and every dimension. The model can learn to decode position from the pattern.

**Final input to Transformer:**
```python
final_input = embedding + PE
```
Same shape — just element-wise addition. The embedding carries word meaning; PE carries word position.

The file prints each PE value with its formula: `PE[pos][i] = sin(pos / 10000^(i/d_model))`.

---

## File 74 — Transformer Encoder

Builds a complete encoder block using the components from Files 71–73.

**6 steps:**

```
Step 1: Input embeddings (X)
Step 2: Add positional encoding → encoder_input
Step 3: Self-Attention → attention_output
Step 4: Add & LayerNorm → norm1 = LayerNorm(encoder_input + attention_output)
Step 5: Feed Forward Network → ff_output
         Dense(4→4, ReLU) → Dense(4→4)
Step 6: Add & LayerNorm → encoder_output = LayerNorm(norm1 + ff_output)
```

**Layer Normalization:**
```python
def layer_norm(X):
    mean = X.mean(axis=1, keepdims=True)
    std  = X.std(axis=1, keepdims=True)
    return (X - mean) / (std + 1e-6)
```
Normalizes each row (each word's vector) to have mean=0 and std=1. This stabilizes training.

**The "Add" step** (residual connection): `norm1 = LayerNorm(encoder_input + attention_output)`. We add the original input back before normalizing. This prevents information loss — if attention learns nothing useful, the original embedding flows through unchanged.

The file ends with word-wise encoder output — each word has a richer, context-aware representation.

---

## File 75 — Transformer Decoder

Shows the decoder for translation: English "I love AI" → Marathi "मला AI आवडतो".

The decoder has **3 sub-layers** instead of 2:

**Sub-layer 1: Masked Self-Attention**
```python
mask = np.triu(np.ones(shape), k=1)
masked_scores[mask == 1] = -1e9   # effectively -infinity
```
Upper triangle of score matrix is masked. After softmax, `-1e9` becomes `≈0`. So word at position 1 can only attend to position 0 and itself — it cannot see future words. This is called **causal masking**.

Why? During training, the decoder sees the full target sentence. Without masking, word 2 could cheat by attending to word 3 (which it's trying to predict). Masking prevents this.

**Sub-layer 2: Encoder-Decoder Attention**
```python
Q = decoder_hidden @ Wq      # Q from decoder (what target word needs)
K = encoder_output @ Wk      # K from encoder (what source words have)
V = encoder_output @ Wv      # V from encoder (source word information)
```
This is the bridge between encoder and decoder. The decoder's output word queries the encoder's source representations to decide which source words to focus on.

**Sub-layer 3: Feed Forward Network** — same as encoder.

Final step: `logits = decoder_output @ W_linear` → `softmax` → `argmax` → predicted word.

---

## File 76 — Sentiment Classification (Encoder-Only)

Real Keras implementation using `layers.TextVectorization` and custom `TransformerEncoder` class.

**24 sentences** (12 positive, 12 negative about a course).

```python
vectorizer = layers.TextVectorization(max_tokens=1000, output_sequence_length=8)
vectorizer.adapt(sentences)
x_data = vectorizer(sentences)   # shape: (24, 8)
```

`TextVectorization` is the Keras version of `Tokenizer + pad_sequences`. `adapt()` builds the vocabulary. The output is already padded to length 8.

**Custom `TokenAndPositionEmbedding` layer:**
```python
token_embed    = Embedding(vocab_size, embed_dim)(x)
position_embed = Embedding(sequence_length, embed_dim)(positions)
return token_embed + position_embed
```
Using a learnable Embedding for positions (instead of the fixed sin/cos formula). Both approaches work.

**`TransformerEncoder` custom layer:**
```python
attention_output = MultiHeadAttention(num_heads=2)(x, x)
out1 = LayerNorm(x + attention_output)        # residual + norm
ffn_output = Dense(32, relu) → Dense(embed_dim)
return LayerNorm(out1 + ffn_output)           # residual + norm
```

**Full model:**
```
Input (8 tokens)
→ TokenAndPositionEmbedding
→ TransformerEncoder
→ GlobalAveragePooling1D   ← averages all 8 positions into one vector
→ Dense(32, relu)
→ Dropout(0.2)
→ Dense(1, sigmoid)        ← Positive / Negative
```

`GlobalAveragePooling1D` converts the (8, 16) encoder output into a single (16,) vector by averaging across the 8 positions. This gives us one summary vector for the whole sentence.

Trained for 80 epochs. Tests on 8 new sentences.

---

## File 77 — Neural Machine Translation (Encoder-Decoder)

English → Marathi translation using the full Encoder-Decoder Transformer.

**Dataset:** 10 sentence pairs. Marathi target sentences use `start` and `end` tokens.

```python
target_sentences = ["start " + sentence + " end" for sentence in marathi_sentences]
```

**Teacher Forcing:**
```python
decoder_input_data  = target_token_data[:, :-1]   # "start mala ai avadte"
decoder_target_data = target_token_data[:, 1:]     # "mala ai avadte end"
```
During training, the decoder receives the correct previous word at each step (not its own prediction). This makes training faster and more stable.

**Two separate inputs to the model:**
```python
model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.fit([encoder_input_data, decoder_input_data], decoder_target_data, ...)
```

**Autoregressive inference:**
```python
def translate_sentence(input_sentence):
    decoded_sentence = "start"
    for i in range(sequence_length):
        # predict next word
        # append to decoded_sentence
        # stop if "end" token predicted
```
At inference time, the model generates one word at a time, feeding each prediction back as the next decoder input.

---

## File 78 — GPT-Style Decoder-Only (Mini GPT)

Next-word prediction and text generation using a decoder-only Transformer (like GPT).

**Key difference from encoder-decoder:** No encoder, no cross-attention. Only Masked Self-Attention. This makes it purely generative.

**Data preparation:**
```python
x_data = tokenized_data[:, :-1]   # input: all words except last
y_data = tokenized_data[:, 1:]    # target: all words except first
```
For "i love artificial intelligence": input = "i love artificial", target = "love artificial intelligence". Predicts next word at every position simultaneously.

**Causal mask:**
```python
i = tf.range(seq_len)[:, None]
j = tf.range(seq_len)
mask = tf.cast(i >= j, dtype="int32")
```
Lower triangle is 1 (visible). Upper triangle is 0 (hidden). Position 0 sees only itself. Position 1 sees 0 and 1. Etc.

**Model (2 decoder blocks stacked):**
```
Input tokens
→ TokenAndPositionEmbedding
→ DecoderOnlyTransformerBlock (Masked Attention + FFN + LayerNorm)
→ DecoderOnlyTransformerBlock (same again — deeper)
→ Dense(vocab_size, softmax)   ← predict next word
```

**Text generation:**
```python
def generate_text(start_text, number_of_words=5):
    for i in range(number_of_words):
        next_word = predict_next_word(generated_text)
        generated_text = generated_text + " " + next_word
    return generated_text
```
Start with "i love" → predict "artificial" → append → predict "intelligence" → etc. This is **autoregressive generation** — exactly what GPT does.

---

## The Three Transformer Architectures

| Architecture | Files | Real-world example | What it does |
|---|---|---|---|
| Encoder-only | 71–74, 76 | BERT | Understands text (classification, Q&A) |
| Encoder-Decoder | 75, 77 | T5, original Transformer | Transforms one sequence to another (translation, summarization) |
| Decoder-only | 78 | GPT, LLaMA | Generates text (next word prediction) |

---

## How to Run

```bash
pip install tensorflow numpy

# Manual numpy demos (no training)
python 71_transformer_self_attention.py
python 72_transformer_Multihead_attention.py
python 73_transformer_poitionalencoder.py
python 74_transformer_encoder.py
python 75_transformer_decoder.py

# Full Keras models (training required, may take a few minutes)
python 76_Transformer_Sentiment_Classification_Encoder_Only.py
python 77_Transformer_Neural_Machine_Translation_Encoder_Decoder.py
python 78_Transformer_Generative_Pre_trained_Transformer_Decoder_Only.py
```

Files 71–75 run in seconds (pure numpy). Files 76–78 train neural networks and take longer.

---

*Marvellous Infosystems — From Q×K.T to GPT, understanding every step of the Transformer.*
