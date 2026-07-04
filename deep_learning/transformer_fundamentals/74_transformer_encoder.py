import numpy as np

np.set_printoptions(precision=3, suppress=True)

print("=" * 70)
print("        Marvellous Infosystems")
print("        Simple Transformer Encoder Demonstration")
print("=" * 70)

# ------------------------------------------------------------
# Step 1: Input Embeddings
# ------------------------------------------------------------

words = ["I", "love", "AI"]

X = np.array([
    [1.0, 0.0, 1.0, 0.0],   # I
    [0.0, 1.0, 0.0, 1.0],   # love
    [1.0, 1.0, 1.0, 1.0]    # AI
])

print("\nSTEP 1: Input Embeddings")
for word, vector in zip(words, X):
    print(f"{word:5s} -> {vector}")

# ------------------------------------------------------------
# Step 2: Positional Encoding
# ------------------------------------------------------------

def positional_encoding(seq_len, d_model):
    PE = np.zeros((seq_len, d_model))

    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            PE[pos, i] = np.sin(pos / (10000 ** (i / d_model)))

            if i + 1 < d_model:
                PE[pos, i + 1] = np.cos(pos / (10000 ** (i / d_model)))

    return PE

PE = positional_encoding(seq_len=3, d_model=4)

print("\nSTEP 2: Positional Encoding")
print(PE)

encoder_input = X + PE

print("\nEncoder Input = Embedding + Positional Encoding")
print(encoder_input)

# ------------------------------------------------------------
# Step 3: Self-Attention
# ------------------------------------------------------------

def softmax(row):
    exp_row = np.exp(row - np.max(row))
    return exp_row / exp_row.sum()

def self_attention(X):
    print("\nSTEP 3: SELF ATTENTION")
    print("-" * 70)

    # Simple identity weight matrices
    Wq = np.eye(4)
    Wk = np.eye(4)
    Wv = np.eye(4)

    Q = np.dot(X, Wq)
    K = np.dot(X, Wk)
    V = np.dot(X, Wv)

    print("\nQ Matrix")
    print(Q)

    print("\nK Matrix")
    print(K)

    print("\nV Matrix")
    print(V)

    scores = np.dot(Q, K.T)

    print("\nAttention Scores = Q × K.T")
    print(scores)

    dk = K.shape[1]
    scaled_scores = scores / np.sqrt(dk)

    print("\nScaled Scores = Scores / sqrt(dk)")
    print(scaled_scores)

    attention_weights = np.array([softmax(row) for row in scaled_scores])

    print("\nAttention Weights after Softmax")
    print(attention_weights)

    output = np.dot(attention_weights, V)

    print("\nSelf-Attention Output")
    print(output)

    return output

attention_output = self_attention(encoder_input)

# ------------------------------------------------------------
# Step 4: Add & Layer Normalization
# ------------------------------------------------------------

def layer_norm(X):
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)

    normalized = (X - mean) / (std + 1e-6)

    return normalized

print("\nSTEP 4: Add & Layer Normalization")
print("-" * 70)

add1 = encoder_input + attention_output

print("\nAdd Operation 1 = Encoder Input + Attention Output")
print(add1)

norm1 = layer_norm(add1)

print("\nLayer Normalization 1")
print(norm1)

# ------------------------------------------------------------
# Step 5: Feed Forward Network
# ------------------------------------------------------------

def feed_forward(X):
    print("\nSTEP 5: Feed Forward Network")
    print("-" * 70)

    W1 = np.array([
        [0.2, 0.4, 0.1, 0.3],
        [0.5, 0.1, 0.6, 0.2],
        [0.3, 0.7, 0.2, 0.5],
        [0.6, 0.2, 0.4, 0.1]
    ])

    W2 = np.array([
        [0.3, 0.2, 0.5, 0.1],
        [0.6, 0.4, 0.2, 0.3],
        [0.1, 0.7, 0.3, 0.5],
        [0.2, 0.1, 0.6, 0.4]
    ])

    hidden = np.dot(X, W1)

    print("\nDense Layer 1 Output")
    print(hidden)

    relu_output = np.maximum(0, hidden)

    print("\nAfter ReLU")
    print(relu_output)

    output = np.dot(relu_output, W2)

    print("\nDense Layer 2 Output")
    print(output)

    return output

ff_output = feed_forward(norm1)

# ------------------------------------------------------------
# Step 6: Add & Layer Normalization Again
# ------------------------------------------------------------

print("\nSTEP 6: Add & Layer Normalization Again")
print("-" * 70)

add2 = norm1 + ff_output

print("\nAdd Operation 2 = Norm1 + Feed Forward Output")
print(add2)

encoder_output = layer_norm(add2)

print("\nFinal Transformer Encoder Output")
print(encoder_output)

# ------------------------------------------------------------
# Final Word-wise Output
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL WORD-WISE ENCODER OUTPUT")
print("=" * 70)

for word, vector in zip(words, encoder_output):
    print(f"{word:5s} -> {vector}")
