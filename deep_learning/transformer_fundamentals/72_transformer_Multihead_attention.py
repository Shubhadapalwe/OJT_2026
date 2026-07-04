import numpy as np

np.set_printoptions(precision=3, suppress=True)

print("=" * 70)
print("        Marvellous Infosystems")
print("        Detailed Multi Head Attention Demonstration")
print("=" * 70)

# ------------------------------------------------------------
# Step 1: Input Embeddings
# ------------------------------------------------------------

words = ["I", "love", "AI"]

X = np.array([
    [1, 0, 1, 0],   # I
    [0, 1, 0, 1],   # love
    [1, 1, 1, 1]    # AI
], dtype=float)

print("\nSTEP 1: Input Embeddings")
print("-" * 70)

for word, vector in zip(words, X):
    print(f"{word:5s} -> {vector}")

print("\nShape of X:", X.shape)
print("Meaning: 3 words, each word represented by 4 values")
print("So embedding dimension = 4")


# ------------------------------------------------------------
# Step 2: Define Weight Matrices for Two Heads
# ------------------------------------------------------------

print("\nSTEP 2: Weight Matrices")
print("-" * 70)

# Head 1: Identity weights
# This head sees the input as it is.
Wq1 = np.eye(4)
Wk1 = np.eye(4)
Wv1 = np.eye(4)

# Head 2: Feature swapping weights
# This head sees the same input from another perspective.
Wq2 = np.array([
    [0, 1, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
], dtype=float)

Wk2 = Wq2
Wv2 = Wq2

print("\nHead 1 Wq1, Wk1, Wv1 are Identity Matrices")
print(Wq1)

print("\nHead 2 Wq2, Wk2, Wv2")
print(Wq2)

print("\nMeaning:")
print("Head 1 keeps original features.")
print("Head 2 swaps features and observes the same sentence differently.")


# ------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------

def softmax(row):
    exp_row = np.exp(row - np.max(row))
    return exp_row / exp_row.sum()


def print_matrix_with_words(title, matrix, row_words=None, col_words=None):
    print(f"\n{title}")
    print("-" * 70)

    if col_words:
        header = " " * 10 + "".join([f"{word:>12s}" for word in col_words])
        print(header)

    for i, row in enumerate(matrix):
        row_name = row_words[i] if row_words else f"Row {i}"
        print(f"{row_name:>8s}  {row}")


# ------------------------------------------------------------
# Attention Function
# ------------------------------------------------------------

def attention(X, Wq, Wk, Wv, head_name):
    print("\n" + "=" * 70)
    print(f"{head_name}")
    print("=" * 70)

    # --------------------------------------------------------
    # Step A: Calculate Q, K, V
    # --------------------------------------------------------

    print("\nSTEP A: Calculate Q, K, V")
    print("-" * 70)

    Q = np.dot(X, Wq)
    K = np.dot(X, Wk)
    V = np.dot(X, Wv)

    print("\nFormula:")
    print("Q = X × Wq")
    print("K = X × Wk")
    print("V = X × Wv")

    print_matrix_with_words("Q Matrix", Q, words)
    print_matrix_with_words("K Matrix", K, words)
    print_matrix_with_words("V Matrix", V, words)

    print("\nMeaning:")
    print("Q tells what each word is searching for.")
    print("K tells what each word contains.")
    print("V tells what information each word gives.")

    # --------------------------------------------------------
    # Step B: Calculate Attention Scores
    # --------------------------------------------------------

    print("\nSTEP B: Calculate Attention Scores")
    print("-" * 70)

    KT = K.T
    scores = np.dot(Q, KT)

    print("\nFormula:")
    print("Scores = Q × K.T")

    print_matrix_with_words("K Transpose", KT)

    print_matrix_with_words(
        "Attention Score Matrix",
        scores,
        row_words=words,
        col_words=words
    )

    print("\nMeaning of Score Matrix:")
    print("Each row represents one word searching.")
    print("Each column represents the word being compared.")
    print("Higher score means stronger relationship.")

    # Manual calculation for first row
    print("\nManual Calculation for First Word:", words[0])
    for j, word in enumerate(words):
        calculation = " + ".join(
            [f"{Q[0][k]:.0f}×{K[j][k]:.0f}" for k in range(K.shape[1])]
        )
        print(f"{words[0]} with {word:5s}: {calculation} = {scores[0][j]:.0f}")

    # --------------------------------------------------------
    # Step C: Scaling
    # --------------------------------------------------------

    print("\nSTEP C: Scaling")
    print("-" * 70)

    dk = K.shape[1]
    scale_factor = np.sqrt(dk)

    scaled_scores = scores / scale_factor

    print(f"dk = dimension of key vector = {dk}")
    print(f"sqrt(dk) = sqrt({dk}) = {scale_factor:.3f}")
    print("\nFormula:")
    print("Scaled Scores = Scores / sqrt(dk)")

    print_matrix_with_words(
        "Scaled Score Matrix",
        scaled_scores,
        row_words=words,
        col_words=words
    )

    print("\nMeaning:")
    print("Scaling reduces large score values before softmax.")
    print("This helps softmax remain stable and balanced.")

    # --------------------------------------------------------
    # Step D: Softmax Row-wise
    # --------------------------------------------------------

    print("\nSTEP D: Apply Softmax Row-wise")
    print("-" * 70)

    weights = np.array([softmax(row) for row in scaled_scores])

    print_matrix_with_words(
        "Attention Weights Matrix",
        weights,
        row_words=words,
        col_words=words
    )

    print("\nRow-wise Interpretation:")
    for i, word in enumerate(words):
        print(f"\n{word} gives attention:")
        for j, other_word in enumerate(words):
            print(f"  {other_word:5s}: {weights[i][j] * 100:.2f}%")
        print(f"  Row sum = {weights[i].sum():.3f}")

    print("\nMeaning:")
    print("Each row becomes a probability distribution.")
    print("Each word decides how much attention to give to all words.")

    # --------------------------------------------------------
    # Step E: Final Output
    # --------------------------------------------------------

    print("\nSTEP E: Final Output of This Head")
    print("-" * 70)

    output = np.dot(weights, V)

    print("\nFormula:")
    print("Output = Attention Weights × V")

    print_matrix_with_words(
        "Output Matrix",
        output,
        row_words=words
    )

    print("\nManual Calculation for First Word Output:")

    for feature_index in range(V.shape[1]):
        terms = []
        total = 0

        for j in range(len(words)):
            value = weights[0][j] * V[j][feature_index]
            total += value
            terms.append(f"{weights[0][j]:.3f}×{V[j][feature_index]:.0f}")

        print(
            f"Feature {feature_index + 1}: "
            + " + ".join(terms)
            + f" = {total:.3f}"
        )

    print("\nMeaning:")
    print("The output vector of each word is now context-aware.")
    print("It contains information collected from all words based on attention weights.")

    return output


# ------------------------------------------------------------
# Step 3: Run Head 1 and Head 2
# ------------------------------------------------------------

head1 = attention(X, Wq1, Wk1, Wv1, "HEAD 1: Original Feature View")
head2 = attention(X, Wq2, Wk2, Wv2, "HEAD 2: Swapped Feature View")


# ------------------------------------------------------------
# Step 4: Concatenate Outputs of Both Heads
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL STEP: Concatenate Head Outputs")
print("=" * 70)

final_output = np.concatenate((head1, head2), axis=1)

print("\nFormula:")
print("Final Multi-Head Output = Concatenate(Head1 Output, Head2 Output)")

print_matrix_with_words("Head 1 Output", head1, words)
print_matrix_with_words("Head 2 Output", head2, words)
print_matrix_with_words("Final Multi-Head Attention Output", final_output, words)

print("\nFinal Word-wise Representation")
print("-" * 70)

for word, vector in zip(words, final_output):
    print(f"{word:5s} -> {vector}")

print("\nFinal Meaning:")
print("Head 1 gives one contextual view of the sentence.")
print("Head 2 gives another contextual view of the sentence.")
print("After concatenation, each word contains richer information from multiple heads.")
