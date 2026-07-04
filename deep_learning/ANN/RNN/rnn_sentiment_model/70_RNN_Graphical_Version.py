# ============================================================
# Program : RNN Sentiment Analysis with Detailed Mathematical
#           Calculations of Embedding, RNN, Dense and Sigmoid
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import math

np.set_printoptions(precision=4, suppress=True)

print("=" * 70)
print("              Marvellous Infosystems")
print("     RNN Sentiment Analysis - Full Mathematical Demo")
print("=" * 70)

# ------------------------------------------------------------
# STEP 1 : Input Sentence
# ------------------------------------------------------------

sentence = "food was not good"
words = sentence.split()

print("\nSTEP 1 : INPUT SENTENCE")
print("Sentence :", sentence)
print("Words    :", words)

# ------------------------------------------------------------
# STEP 2 : Tokenization
# ------------------------------------------------------------

word_index = {
    "food": 1,
    "was": 2,
    "good": 3,
    "bad": 4,
    "not": 5
}

tokens = [word_index[word] for word in words]

print("\nSTEP 2 : TOKENIZATION")
print("Word Index Table:")
for word, token in word_index.items():
    print(f"{word:5s} -> {token}")

print("\nSentence:")
print(sentence)

print("\nToken sequence:")
print(tokens)

# ------------------------------------------------------------
# STEP 3 : Embedding Layer
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 3 : EMBEDDING LAYER")
print("=" * 70)

print("""
Embedding layer does not perform multiplication for each word.
It works like a lookup table.

Token number is used as row number.
That row is picked from the embedding matrix.
""")

embedding_matrix = np.array([
    [0.0, 0.0, 0.0],   # index 0 padding
    [0.2, 0.4, 0.1],   # index 1 food
    [0.3, 0.1, 0.5],   # index 2 was
    [0.8, 0.7, 0.9],   # index 3 good
    [0.1, 0.2, 0.9],   # index 4 bad
    [0.9, 0.3, 0.2]    # index 5 not
])

index_to_word = {
    0: "padding",
    1: "food",
    2: "was",
    3: "good",
    4: "bad",
    5: "not"
}

print("Complete Embedding Matrix:")
print("Rows = token ids, Columns = embedding dimensions")
print(embedding_matrix)

print("\nEmbedding Lookup Calculation:")

embedded_sequence = []

for token in tokens:
    vector = embedding_matrix[token]
    embedded_sequence.append(vector)

    print(f"\nToken {token} represents word '{index_to_word[token]}'")
    print(f"Embedding vector = embedding_matrix[{token}]")
    print(f"Embedding vector = {vector}")

X = np.array(embedded_sequence)

print("\nFinal Embedded Input Matrix X:")
print(X)

print("\nShape Explanation:")
print("Number of words          =", X.shape[0])
print("Embedding size per word  =", X.shape[1])
print("Shape of X               =", X.shape)

# Graph : Embedding Matrix
plt.figure(figsize=(8, 4))
plt.imshow(X)
plt.colorbar()
plt.title("Embedding Matrix Used by RNN")
plt.xlabel("Embedding Dimensions")
plt.ylabel("Words")
plt.xticks([0, 1, 2], ["Dim 1", "Dim 2", "Dim 3"])
plt.yticks(range(len(words)), words)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        plt.text(j, i, str(X[i, j]), ha="center", va="center")

plt.show()

# ------------------------------------------------------------
# STEP 4 : RNN Weights
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 4 : RNN WEIGHT MATRICES")
print("=" * 70)

Wx = np.array([
    [0.5, 0.1, 0.3],
    [0.2, 0.4, 0.1],
    [0.3, 0.2, 0.5]
])

Wh = np.array([
    [0.4, 0.2, 0.1],
    [0.1, 0.5, 0.2],
    [0.3, 0.1, 0.4]
])

b = np.array([0.1, 0.1, 0.1])

print("""
RNN uses three important values:

1. Wx  : Input-to-hidden weight matrix
2. Wh  : Hidden-to-hidden weight matrix
3. b   : Bias vector
""")

print("Wx =")
print(Wx)

print("\nWh =")
print(Wh)

print("\nb =")
print(b)

print("""
RNN formula:

h_t = tanh(Wx . x_t + Wh . h_previous + b)

Meaning:

Current Hidden State =
tanh(Current Word Contribution + Previous Memory Contribution + Bias)
""")

# ------------------------------------------------------------
# Helper function to show matrix-vector multiplication
# ------------------------------------------------------------

def explain_matrix_vector_multiplication(matrix, vector, matrix_name, vector_name):
    result = np.dot(matrix, vector)

    print(f"\n{matrix_name} . {vector_name} calculation:")

    for i in range(matrix.shape[0]):
        expression_parts = []
        total = 0

        for j in range(matrix.shape[1]):
            multiplication = matrix[i][j] * vector[j]
            total += multiplication
            expression_parts.append(
                f"({matrix[i][j]:.2f} × {vector[j]:.2f})"
            )

        expression = " + ".join(expression_parts)
        print(f"Row {i+1}: {expression} = {total:.4f}")

    print(f"Result of {matrix_name} . {vector_name} = {result}")
    return result

# ------------------------------------------------------------
# STEP 5 : RNN Forward Pass
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 5 : RNN PROCESSING WITH COMPLETE CALCULATIONS")
print("=" * 70)

hidden_states = []
h_previous = np.array([0.0, 0.0, 0.0])

for time_step, word in enumerate(words):
    print("\n" + "-" * 70)
    print(f"TIME STEP {time_step + 1}")
    print("-" * 70)

    x_t = X[time_step]

    print("Current word              :", word)
    print("Current word vector x_t   :", x_t)
    print("Previous hidden state     :", h_previous)

    # Part 1: Wx . x_t
    input_part = explain_matrix_vector_multiplication(
        Wx, x_t, "Wx", "x_t"
    )

    # Part 2: Wh . h_previous
    memory_part = explain_matrix_vector_multiplication(
        Wh, h_previous, "Wh", "h_previous"
    )

    # Part 3: Add input_part + memory_part + bias
    total = input_part + memory_part + b

    print("\nAddition before tanh:")
    for i in range(len(total)):
        print(
            f"Value {i+1}: "
            f"{input_part[i]:.4f} + {memory_part[i]:.4f} + {b[i]:.4f} "
            f"= {total[i]:.4f}"
        )

    # Part 4: Apply tanh
    h_current = np.tanh(total)

    print("\nApply tanh activation:")
    for i in range(len(h_current)):
        print(f"tanh({total[i]:.4f}) = {h_current[i]:.4f}")

    print("\nCurrent hidden state h_current:")
    print(h_current)

    print("\nMeaning:")
    print(f"This hidden state stores memory after reading '{' '.join(words[:time_step+1])}'")

    hidden_states.append(h_current)
    h_previous = h_current

hidden_states = np.array(hidden_states)

# Graph : Hidden State Matrix
plt.figure(figsize=(8, 4))
plt.imshow(hidden_states)
plt.colorbar()
plt.title("Hidden State Values Generated by RNN")
plt.xlabel("Hidden State Dimensions")
plt.ylabel("Words / Time Steps")
plt.xticks([0, 1, 2], ["h_dim1", "h_dim2", "h_dim3"])
plt.yticks(range(len(words)), words)

for i in range(hidden_states.shape[0]):
    for j in range(hidden_states.shape[1]):
        plt.text(j, i, f"{hidden_states[i, j]:.2f}", ha="center", va="center")

plt.show()

# ------------------------------------------------------------
# STEP 6 : Hidden State Flow
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 6 : HIDDEN STATE FLOW")
print("=" * 70)

print("Initial hidden state:")
print("h0 =", np.array([0.0, 0.0, 0.0]))

for i, word in enumerate(words):
    print(f"\nAfter reading '{word}':")
    print(f"h{i+1} = {hidden_states[i]}")

plt.figure(figsize=(10, 4))
plt.title("RNN Memory Flow")

x_positions = [1, 3, 5, 7]

for i, word in enumerate(words):
    h = hidden_states[i]

    plt.text(
        x_positions[i],
        2,
        f"{word}\nh{i+1} = [{h[0]:.2f}, {h[1]:.2f}, {h[2]:.2f}]",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.5")
    )

    if i < len(words) - 1:
        plt.arrow(
            x_positions[i] + 0.6,
            2,
            0.8,
            0,
            head_width=0.1,
            head_length=0.2,
            length_includes_head=True
        )

plt.xlim(0, 8)
plt.ylim(1, 3)
plt.axis("off")
plt.show()

# ------------------------------------------------------------
# STEP 7 : Dense Layer
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 7 : DENSE LAYER CALCULATION")
print("=" * 70)

h_final = hidden_states[-1]

Wy = np.array([-2.0, -1.5, -1.2])
by = 1.0

print("Final hidden state from RNN:")
print("h_final =", h_final)

print("\nDense layer weights:")
print("Wy =", Wy)

print("\nDense layer bias:")
print("by =", by)

print("""
Dense formula:

z = Wy . h_final + by
""")

dense_products = Wy * h_final

print("Element-wise multiplication:")
for i in range(len(h_final)):
    print(
        f"({Wy[i]:.2f} × {h_final[i]:.4f}) = {dense_products[i]:.4f}"
    )

z = np.sum(dense_products) + by

print("\nAddition:")
print(
    f"z = {dense_products[0]:.4f} + "
    f"{dense_products[1]:.4f} + "
    f"{dense_products[2]:.4f} + {by:.4f}"
)

print("z =", z)

plt.figure(figsize=(7, 4))
labels = ["w1*h1", "w2*h2", "w3*h3", "bias"]
values = [dense_products[0], dense_products[1], dense_products[2], by]

plt.bar(labels, values)
plt.title("Dense Layer Calculation")
plt.ylabel("Contribution to z")

for i, value in enumerate(values):
    plt.text(i, value, f"{value:.2f}", ha="center")

plt.show()

# ------------------------------------------------------------
# STEP 8 : Sigmoid Calculation
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 8 : SIGMOID CALCULATION")
print("=" * 70)

print("""
Sigmoid converts raw score z into probability.

Formula:

sigmoid(z) = 1 / (1 + e^(-z))
""")

e_power = math.exp(-z)
denominator = 1 + e_power
prediction = 1 / denominator

print(f"z = {z:.4f}")
print(f"e^(-z) = e^({-z:.4f}) = {e_power:.4f}")
print(f"1 + e^(-z) = {denominator:.4f}")
print(f"sigmoid(z) = 1 / {denominator:.4f}")
print(f"sigmoid(z) = {prediction:.4f}")

# Sigmoid graph
z_values = np.linspace(-6, 6, 200)
sigmoid_values = 1 / (1 + np.exp(-z_values))

plt.figure(figsize=(8, 4))
plt.plot(z_values, sigmoid_values)
plt.axhline(0.5, linestyle="--")
plt.scatter([z], [prediction], s=100)
plt.title("Sigmoid Function")
plt.xlabel("z")
plt.ylabel("sigmoid(z)")
plt.text(z, prediction, f"  ({z:.2f}, {prediction:.2f})")
plt.show()

# ------------------------------------------------------------
# STEP 9 : Final Decision
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 9 : FINAL SENTIMENT DECISION")
print("=" * 70)

print("Decision rule:")
print("Prediction >= 0.5  -> Positive")
print("Prediction <  0.5  -> Negative")

if prediction >= 0.5:
    sentiment = "Positive"
    sentiment_value = 1
else:
    sentiment = "Negative"
    sentiment_value = 0

print("\nPrediction value :", prediction)
print("Final Sentiment  :", sentiment)
print("Sentiment Value  :", sentiment_value)

# ------------------------------------------------------------
# STEP 10 : Complete Summary
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("COMPLETE FLOW SUMMARY")
print("=" * 70)

print(f"""
Input Sentence:
{sentence}

Token Sequence:
{tokens}

Embedding Matrix:
{X}

Final Hidden State:
{h_final}

Dense Score:
z = {z:.4f}

Sigmoid Output:
prediction = {prediction:.4f}

Final Output:
{sentiment} Sentiment
""")

plt.figure(figsize=(10, 7))
plt.title("Complete RNN Sentiment Flow")

summary = f"""
Input:
food was not good

Tokenization:
{tokens}

Embedding:
Each token is replaced by its vector

RNN:
h_t = tanh(Wx.x_t + Wh.h_previous + b)

Final Hidden State:
[{h_final[0]:.2f}, {h_final[1]:.2f}, {h_final[2]:.2f}]

Dense:
z = Wy.h_final + by
z = {z:.2f}

Sigmoid:
prediction = {prediction:.2f}

Decision:
{sentiment} Sentiment
"""

plt.text(0.05, 0.5, summary, fontsize=12, va="center")
plt.axis("off")
plt.show()

print("\nConclusion:")
print("Embedding layer converts token IDs into vectors using lookup.")
print("RNN combines current word vector with previous hidden state.")
print("Dense layer converts final memory into a score.")
print("Sigmoid converts score into probability.")
print("Threshold converts probability into final sentiment.")