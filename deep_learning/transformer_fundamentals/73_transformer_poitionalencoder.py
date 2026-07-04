import numpy as np

np.set_printoptions(precision=4, suppress=True)

print("=" * 70)
print("        Marvellous Infosystems")
print("        Detailed Positional Encoding Demonstration")
print("=" * 70)

# ------------------------------------------------------------
# Step 1: Input Sentence & Embeddings
# ------------------------------------------------------------

words = ["I", "love", "AI"]

embedding = np.array([
    [0.2, 0.4, 0.1, 0.5],
    [0.8, 0.7, 0.9, 0.6],
    [0.5, 0.9, 0.7, 0.3]
])

print("\nSTEP 1: Word Embeddings")
print("-" * 70)

for word, vector in zip(words, embedding):
    print(f"{word:5s} -> {vector}")

print("\nMeaning:")
print("Each word is converted into a numeric vector.")
print("But this does NOT include position information.")

# ------------------------------------------------------------
# Step 2: Positional Encoding Function
# ------------------------------------------------------------

def positional_encoding(seq_len, d_model):
    PE = np.zeros((seq_len, d_model))

    print("\nSTEP 2: Calculating Positional Encoding")
    print("-" * 70)

    for pos in range(seq_len):
        print(f"\nPosition = {pos}")

        for i in range(0, d_model, 2):

            angle = pos / (10000 ** (i / d_model))

            sin_value = np.sin(angle)
            PE[pos, i] = sin_value

            print(f"  PE[{pos}][{i}] = sin({pos} / 10000^({i}/{d_model})) = {sin_value:.4f}")

            if i + 1 < d_model:
                cos_value = np.cos(angle)
                PE[pos, i + 1] = cos_value

                print(f"  PE[{pos}][{i+1}] = cos({pos} / 10000^({i}/{d_model})) = {cos_value:.4f}")

    return PE

# ------------------------------------------------------------
# Step 3: Generate Positional Encoding
# ------------------------------------------------------------

PE = positional_encoding(seq_len=3, d_model=4)

print("\nSTEP 3: Positional Encoding Matrix")
print("-" * 70)
print(PE)

print("\nMeaning:")
print("Each row represents position information of a word.")
print("Each column represents different frequency patterns.")

# ------------------------------------------------------------
# Step 4: Combine Embedding + Positional Encoding
# ------------------------------------------------------------

final_input = embedding + PE

print("\nSTEP 4: Final Input to Transformer")
print("-" * 70)

print("\nFormula:")
print("Final Input = Embedding + Positional Encoding")

print("\nFinal Matrix:")
print(final_input)

print("\nWord-wise Final Representation")

for word, emb, pe, final in zip(words, embedding, PE, final_input):
    print(f"\nWord: {word}")
    print(f"Embedding           : {emb}")
    print(f"Positional Encoding : {pe}")
    print(f"Final Input         : {final}")
