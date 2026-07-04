import numpy as np

np.set_printoptions(precision=3, suppress=True)

print("=" * 70)
print("        Marvellous Infosystems")
print("        Detailed Manual Self Attention Calculation")
print("=" * 70)

# ------------------------------------------------------------
# Step 1: Input Embeddings
# ------------------------------------------------------------

words = ["I", "love", "AI"]

X = np.array([
    [1, 0],   # I
    [0, 1],   # love
    [1, 1]    # AI
], dtype=float)

print("\nSTEP 1: Input Embeddings")
print("-" * 70)

for word, vector in zip(words, X):
    print(f"{word:5s} -> {vector}")

print("\nMeaning:")
print("Each word is represented as a numeric vector.")
print("Here we have 3 words and each word has 2 features.")
print("Shape of X:", X.shape)


# ------------------------------------------------------------
# Step 2: Weight Matrices
# ------------------------------------------------------------

Wq = np.array([
    [1, 0],
    [0, 1]
], dtype=float)

Wk = np.array([
    [1, 0],
    [0, 1]
], dtype=float)

Wv = np.array([
    [1, 0],
    [0, 1]
], dtype=float)

print("\nSTEP 2: Weight Matrices")
print("-" * 70)

print("\nWq: Query Weight Matrix")
print(Wq)

print("\nWk: Key Weight Matrix")
print(Wk)

print("\nWv: Value Weight Matrix")
print(Wv)

print("\nMeaning:")
print("Wq converts input embeddings into Query vectors.")
print("Wk converts input embeddings into Key vectors.")
print("Wv converts input embeddings into Value vectors.")
print("In real Transformer models, these weights are learned during training.")


# ------------------------------------------------------------
# Step 3: Calculate Q, K, V
# ------------------------------------------------------------

Q = np.dot(X, Wq)
K = np.dot(X, Wk)
V = np.dot(X, Wv)

print("\nSTEP 3: Calculate Q, K, V")
print("-" * 70)

print("\nFormula:")
print("Q = X × Wq")
print("K = X × Wk")
print("V = X × Wv")

print("\nQuery Matrix Q")
for word, vector in zip(words, Q):
    print(f"{word:5s} -> {vector}")

print("\nKey Matrix K")
for word, vector in zip(words, K):
    print(f"{word:5s} -> {vector}")

print("\nValue Matrix V")
for word, vector in zip(words, V):
    print(f"{word:5s} -> {vector}")

print("\nMeaning:")
print("Q = what each word is searching for.")
print("K = what each word contains.")
print("V = what information each word gives.")


# ------------------------------------------------------------
# Step 4: Attention Scores
# ------------------------------------------------------------

KT = K.T
scores = np.dot(Q, KT)

print("\nSTEP 4: Attention Scores")
print("-" * 70)

print("\nFormula:")
print("Attention Scores = Q × K.T")

print("\nK Transpose")
print(KT)

print("\nAttention Score Matrix")
print(" " * 10 + "".join([f"{word:>10s}" for word in words]))

for i, row in enumerate(scores):
    print(f"{words[i]:>8s}  {row}")

print("\nMeaning:")
print("Each row represents one word that is searching.")
print("Each column represents the word being compared.")
print("Higher score means stronger relationship between words.")

print("\nManual Calculation of Score Matrix")
print("-" * 70)

for i, query_word in enumerate(words):
    print(f"\nFor word '{query_word}':")

    for j, key_word in enumerate(words):
        calculation = " + ".join(
            [f"{Q[i][k]:.0f}×{K[j][k]:.0f}" for k in range(K.shape[1])]
        )

        print(
            f"{query_word:5s} with {key_word:5s}: "
            f"{calculation} = {scores[i][j]:.0f}"
        )


# ------------------------------------------------------------
# Step 5: Scaling
# ------------------------------------------------------------

d_k = K.shape[1]
scale_factor = np.sqrt(d_k)
scaled_scores = scores / scale_factor

print("\nSTEP 5: Scaling")
print("-" * 70)

print(f"dk = dimension of each Key vector = {d_k}")
print(f"sqrt(dk) = sqrt({d_k}) = {scale_factor:.3f}")

print("\nFormula:")
print("Scaled Scores = Scores / sqrt(dk)")

print("\nScaled Score Matrix")
print(" " * 10 + "".join([f"{word:>10s}" for word in words]))

for i, row in enumerate(scaled_scores):
    print(f"{words[i]:>8s}  {row}")

print("\nManual Scaling")
for i in range(scores.shape[0]):
    for j in range(scores.shape[1]):
        print(
            f"Score[{words[i]}][{words[j]}] = "
            f"{scores[i][j]:.0f} / {scale_factor:.3f} = {scaled_scores[i][j]:.3f}"
        )

print("\nMeaning:")
print("Scaling reduces large score values before applying softmax.")
print("This makes softmax stable and avoids one value dominating too much.")


# ------------------------------------------------------------
# Step 6: Softmax Row-wise
# ------------------------------------------------------------

def softmax(row):
    exp_row = np.exp(row - np.max(row))
    return exp_row / exp_row.sum()

attention_weights = np.array([softmax(row) for row in scaled_scores])

print("\nSTEP 6: Apply Softmax Row-wise")
print("-" * 70)

print("\nFormula:")
print("Softmax(xi) = exp(xi) / sum(exp(all values in that row))")

print("\nAttention Weights Matrix")
print(" " * 10 + "".join([f"{word:>10s}" for word in words]))

for i, row in enumerate(attention_weights):
    print(f"{words[i]:>8s}  {row}")

print("\nRow-wise Interpretation")
for i, word in enumerate(words):
    print(f"\n{word} gives attention:")
    for j, other_word in enumerate(words):
        print(f"  {other_word:5s}: {attention_weights[i][j] * 100:.2f}%")
    print(f"  Row sum = {attention_weights[i].sum():.3f}")

print("\nMeaning:")
print("Each row is now converted into probabilities.")
print("Each word decides how much attention it gives to every word.")


# ------------------------------------------------------------
# Step 7: Final Self Attention Output
# ------------------------------------------------------------

output = np.dot(attention_weights, V)

print("\nSTEP 7: Final Self-Attention Output")
print("-" * 70)

print("\nFormula:")
print("Output = Attention Weights × V")

print("\nFinal Output Matrix")
for word, vector in zip(words, output):
    print(f"{word:5s} -> {vector}")

print("\nManual Output Calculation")
print("-" * 70)

for i, word in enumerate(words):
    print(f"\nFinal output for word '{word}':")

    for feature_index in range(V.shape[1]):
        terms = []
        total = 0

        for j in range(len(words)):
            value = attention_weights[i][j] * V[j][feature_index]
            total += value
            terms.append(
                f"{attention_weights[i][j]:.3f}×{V[j][feature_index]:.0f}"
            )

        print(
            f"Feature {feature_index + 1}: "
            + " + ".join(terms)
            + f" = {total:.3f}"
        )

print("\nMeaning of Final Output:")
print("Before attention, every word was independent.")
print("After attention, every word becomes context-aware.")
print("Each final vector contains information collected from all words.")

print("Final Word-wise Contextual Representation")
print("-" * 70)

for word, vector in zip(words, output):
    print(f"{word:5s} -> {vector}")