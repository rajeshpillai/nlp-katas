# Visualize Attention Weights

> Phase 9 — Transformer Architecture (Core Concepts) | Kata 9.1

---

## Concept & Intuition

### What problem are we solving?

In all previous phases, every word in a sentence received a fixed representation that did not depend on its neighbors. BoW and TF-IDF assigned each word a score based on frequency or rarity, but the word "bank" got the same score whether it appeared next to "river" or "financial." RNNs attempted to fix this by processing words sequentially, but they struggled with long-range dependencies because information had to pass through every intermediate step.

Self-attention solves this directly: **every token computes a weighted relationship with every other token in the sequence, in a single step.** The word "bank" can directly attend to "river" even if they are far apart, and the resulting representation of "bank" will be different from when it attends to "financial." This is the core mechanism that makes transformers work.

### Why earlier methods fail

**BoW and TF-IDF** treat each word independently. The representation of "bank" in "river bank" is identical to "bank" in "bank account." There is no mechanism for context to influence representation.

**RNNs** process tokens left-to-right (or both directions in BiRNNs). To connect the first word to the last in a 100-word sentence, information must pass through 99 sequential steps. At each step, some information is lost or distorted. This is the vanishing gradient problem in practice: distant words have a weak influence on each other.

**Self-attention** skips the sequential bottleneck entirely. Every token directly computes a relevance score with every other token. The word at position 1 can attend to the word at position 100 with a single matrix multiplication. No information passes through intermediate steps. No gradients vanish across long distances.

### Mental models

- **Attention as a soft lookup table**: Each token creates a "query" (what am I looking for?), and every token offers a "key" (what do I contain?). The dot product between query and key tells you how relevant each token is. High relevance means "pay more attention to this token."
- **Q, K, V as roles**: Every token plays three simultaneous roles. As a **Query**, it asks "who should I attend to?" As a **Key**, it advertises "here is what I contain." As a **Value**, it provides "here is the information I carry." Attention weights determine how much of each token's Value gets mixed into the final output.
- **Softmax as competition**: The attention weights for each query sum to 1.0 (via softmax). This means tokens compete for attention. If one key matches a query very well, it "wins" most of the attention, and other tokens get less. This creates a dynamic, context-dependent focus mechanism.
- **Scaling by sqrt(d_k)**: Without scaling, dot products between high-dimensional vectors grow large, pushing softmax into saturation (all weight on one token, none on others). Dividing by sqrt(d_k) keeps the values in a range where softmax produces meaningful distributions.

### Visual explanations

```
SELF-ATTENTION: Every token attends to every other token
=====================================================

  Input:  ["The", "cat", "sat", "on", "the", "bank"]

  Each token produces three vectors:
    Q (Query)  -- "What am I looking for?"
    K (Key)    -- "What do I contain?"
    V (Value)  -- "What information do I carry?"

  For each token, compute attention to ALL tokens:

        The   cat   sat   on   the   bank    <-- Keys
  The  [0.15  0.10  0.10  0.10  0.15  0.40]  <-- attention weights for "The"
  cat  [0.10  0.20  0.15  0.05  0.10  0.40]  <-- attention weights for "cat"
  sat  [0.10  0.30  0.20  0.10  0.10  0.20]  <-- attention weights for "sat"
  on   [0.10  0.10  0.15  0.15  0.10  0.40]  <-- attention weights for "on"
  the  [0.10  0.10  0.10  0.10  0.20  0.40]  <-- attention weights for "the"
  bank [0.15  0.15  0.10  0.10  0.15  0.35]  <-- attention weights for "bank"
   ^
   Queries

  Each row sums to 1.0 (softmax).
  Each row says: "For this query token, how much should I attend to each key?"


THE COMPUTATION
===============

  Step 1: Compute raw scores    Score = Q * K^T
  Step 2: Scale                 Score = Score / sqrt(d_k)
  Step 3: Apply softmax         Weights = softmax(Score)   [each row sums to 1]
  Step 4: Weighted sum of V     Output = Weights * V

  Q (seq_len x d_k)             K^T (d_k x seq_len)
  +-----------+                  +-----------+
  | q1 q1 q1 |                  | k1 k2 k3 |
  | q2 q2 q2 |   multiply -->   | k1 k2 k3 |
  | q3 q3 q3 |                  | k1 k2 k3 |
  +-----------+                  +-----------+
                    ||
                    vv
            Score (seq_len x seq_len)
            +-------------+
            | s11 s12 s13 |   <-- how much token 1 attends to 1, 2, 3
            | s21 s22 s23 |   <-- how much token 2 attends to 1, 2, 3
            | s31 s32 s33 |   <-- how much token 3 attends to 1, 2, 3
            +-------------+

  After softmax, each row is a probability distribution.
  Multiply by V to get context-aware representations.


WHY sqrt(d_k) SCALING?
======================

  Without scaling (d_k = 64):
    Dot products can reach values like 30-50.
    softmax([30, 50, 35]) --> [0.00, 1.00, 0.00]   (saturated!)

  With scaling (divide by sqrt(64) = 8):
    Values become 3.75, 6.25, 4.38
    softmax([3.75, 6.25, 4.38]) --> [0.06, 0.76, 0.18]  (meaningful distribution)
```

---

## Hands-on Exploration

1. Consider the sentence "The cat sat on the river bank." By hand, decide which words you think "bank" should attend to most. Now consider "The bank approved the loan." Would the attention pattern for "bank" differ? This is exactly what self-attention learns to do: different contexts produce different attention distributions for the same word.

2. In the attention weight matrix, each row sums to 1.0. What happens if one key matches a query extremely well (very high dot product)? After softmax, that key gets nearly all the weight, and the output for that query is dominated by a single token's value. Is this always desirable? Think of cases where attending broadly to many tokens would be better than focusing sharply on one.

3. The scaling factor is sqrt(d_k). What would happen if we used d_k instead (no square root)? What about no scaling at all? Run the code with different scaling values and observe how the attention distributions change. When the distribution is too "peaked" (one weight near 1.0, others near 0.0), the model cannot blend information from multiple sources.

---

## Live Code

```python
# --- Self-Attention: Scaled Dot-Product Attention from Scratch ---
# Using ONLY Python standard library

import math
import random

random.seed(42)


# ============================================================
# VECTOR AND MATRIX HELPERS
# ============================================================

def make_vector(dim):
    """Create a random vector of given dimension."""
    return [random.gauss(0, 1) for _ in range(dim)]


def make_matrix(rows, cols):
    """Create a random matrix (list of lists)."""
    return [make_vector(cols) for _ in range(rows)]


def dot(v1, v2):
    """Dot product of two vectors."""
    return sum(a * b for a, b in zip(v1, v2))


def mat_mul(A, B):
    """Multiply matrix A (m x n) by matrix B (n x p) -> (m x p)."""
    m = len(A)
    n = len(B)
    p = len(B[0])
    result = []
    for i in range(m):
        row = []
        for j in range(p):
            val = sum(A[i][k] * B[k][j] for k in range(n))
            row.append(val)
        result.append(row)
    return result


def transpose(M):
    """Transpose a matrix."""
    rows = len(M)
    cols = len(M[0])
    return [[M[i][j] for i in range(rows)] for j in range(cols)]


def softmax(values):
    """Compute softmax of a list of values (numerically stable)."""
    max_val = max(values)
    exps = [math.exp(v - max_val) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


def format_float(v, width=6):
    """Format a float to a fixed-width string."""
    return f"{v:>{width}.3f}"


# ============================================================
# SELF-ATTENTION COMPUTATION
# ============================================================

# Our sentence
words = ["The", "cat", "sat", "on", "the", "bank"]
seq_len = len(words)
d_model = 8   # embedding dimension
d_k = 4       # dimension of Q, K, V (smaller for clarity)

print("=" * 60)
print("SELF-ATTENTION: SCALED DOT-PRODUCT ATTENTION")
print("=" * 60)
print(f"\nSentence: {words}")
print(f"Sequence length: {seq_len}")
print(f"Embedding dimension (d_model): {d_model}")
print(f"Q/K/V dimension (d_k): {d_k}")

# Step 1: Create input embeddings (random for demonstration)
random.seed(42)
embeddings = make_matrix(seq_len, d_model)

print(f"\n--- Input Embeddings ({seq_len} x {d_model}) ---")
for i, word in enumerate(words):
    vec_str = " ".join(format_float(v) for v in embeddings[i])
    print(f"  {word:>4s}: [{vec_str}]")

# Step 2: Create weight matrices W_Q, W_K, W_V (d_model x d_k)
# These are the learned projections in a real transformer
W_Q = make_matrix(d_model, d_k)
W_K = make_matrix(d_model, d_k)
W_V = make_matrix(d_model, d_k)

# Step 3: Compute Q, K, V by projecting embeddings
# Q = embeddings * W_Q, etc.
Q = mat_mul(embeddings, W_Q)  # (seq_len x d_k)
K = mat_mul(embeddings, W_K)  # (seq_len x d_k)
V = mat_mul(embeddings, W_V)  # (seq_len x d_k)

print(f"\n--- Query Vectors Q ({seq_len} x {d_k}) ---")
for i, word in enumerate(words):
    vec_str = " ".join(format_float(v) for v in Q[i])
    print(f"  {word:>4s}: [{vec_str}]")

print(f"\n--- Key Vectors K ({seq_len} x {d_k}) ---")
for i, word in enumerate(words):
    vec_str = " ".join(format_float(v) for v in K[i])
    print(f"  {word:>4s}: [{vec_str}]")

print(f"\n--- Value Vectors V ({seq_len} x {d_k}) ---")
for i, word in enumerate(words):
    vec_str = " ".join(format_float(v) for v in V[i])
    print(f"  {word:>4s}: [{vec_str}]")

# Step 4: Compute attention scores: Q * K^T
K_T = transpose(K)
raw_scores = mat_mul(Q, K_T)  # (seq_len x seq_len)

print(f"\n--- Raw Attention Scores: Q * K^T ({seq_len} x {seq_len}) ---")
header = "        " + "".join(f"{w:>8s}" for w in words)
print(header)
for i, word in enumerate(words):
    row_str = "".join(format_float(raw_scores[i][j], 8) for j in range(seq_len))
    print(f"  {word:>4s}: {row_str}")

# Step 5: Scale by sqrt(d_k)
scale = math.sqrt(d_k)
scaled_scores = [[raw_scores[i][j] / scale
                   for j in range(seq_len)]
                  for i in range(seq_len)]

print(f"\n--- Scaled Scores: Q*K^T / sqrt({d_k}) = Q*K^T / {scale:.2f} ---")
header = "        " + "".join(f"{w:>8s}" for w in words)
print(header)
for i, word in enumerate(words):
    row_str = "".join(format_float(scaled_scores[i][j], 8) for j in range(seq_len))
    print(f"  {word:>4s}: {row_str}")

# Step 6: Apply softmax to each row
attention_weights = [softmax(row) for row in scaled_scores]

print(f"\n--- Attention Weights: softmax(scaled_scores) ---")
header = "        " + "".join(f"{w:>8s}" for w in words)
print(header)
for i, word in enumerate(words):
    row_str = "".join(format_float(attention_weights[i][j], 8)
                      for j in range(seq_len))
    row_sum = sum(attention_weights[i])
    print(f"  {word:>4s}: {row_str}  (sum={row_sum:.3f})")

# Step 7: Compute output: attention_weights * V
output = mat_mul(attention_weights, V)  # (seq_len x d_k)

print(f"\n--- Output: Attention_Weights * V ({seq_len} x {d_k}) ---")
for i, word in enumerate(words):
    vec_str = " ".join(format_float(v) for v in output[i])
    print(f"  {word:>4s}: [{vec_str}]")

# ============================================================
# ASCII HEATMAP OF ATTENTION WEIGHTS
# ============================================================

print("\n" + "=" * 60)
print("ATTENTION HEATMAP (ASCII)")
print("=" * 60)
print("\nBrightness: . < o < O < # (low to high attention)\n")

def attention_char(weight):
    """Map attention weight to ASCII character."""
    if weight < 0.10:
        return "  .  "
    elif weight < 0.18:
        return "  o  "
    elif weight < 0.25:
        return "  O  "
    else:
        return "  #  "

header = "  Query\\Key " + "".join(f"{w:>5s}" for w in words)
print(header)
print("  " + "-" * (11 + 5 * seq_len))
for i, word in enumerate(words):
    row_str = "".join(attention_char(attention_weights[i][j])
                      for j in range(seq_len))
    print(f"  {word:>8s}  |{row_str}")

print()

# ============================================================
# SHOW TOP ATTENDED TOKENS FOR EACH WORD
# ============================================================

print("=" * 60)
print("TOP ATTENDED TOKENS FOR EACH WORD")
print("=" * 60)

for i, word in enumerate(words):
    # Sort by attention weight descending
    ranked = sorted(range(seq_len),
                    key=lambda j: attention_weights[i][j],
                    reverse=True)
    print(f"\n  '{word}' attends most to:")
    for rank, j in enumerate(ranked[:3], 1):
        bar_len = int(attention_weights[i][j] * 30)
        bar = "#" * bar_len
        print(f"    {rank}. '{words[j]}' "
              f"(weight={attention_weights[i][j]:.3f}) {bar}")

# ============================================================
# DEMONSTRATE: EFFECT OF SCALING
# ============================================================

print("\n" + "=" * 60)
print("EFFECT OF SCALING ON ATTENTION DISTRIBUTION")
print("=" * 60)

# Use one row of scores as example
example_row = raw_scores[1]  # scores for "cat"
print(f"\nRaw scores for 'cat': "
      f"[{', '.join(f'{v:.2f}' for v in example_row)}]")

for scale_name, divisor in [("No scaling (div by 1)", 1.0),
                             ("sqrt(d_k) = 2.0", math.sqrt(d_k)),
                             ("d_k = 4.0", float(d_k))]:
    scaled = [v / divisor for v in example_row]
    weights = softmax(scaled)
    max_w = max(weights)
    min_w = min(weights)
    spread = max_w - min_w
    print(f"\n  {scale_name}:")
    print(f"    Weights: [{', '.join(f'{w:.3f}' for w in weights)}]")
    print(f"    Max={max_w:.3f}, Min={min_w:.3f}, Spread={spread:.3f}")
    if spread > 0.8:
        print(f"    --> Too peaked! One token dominates.")
    elif spread < 0.05:
        print(f"    --> Too flat! Nearly uniform, no focus.")
    else:
        print(f"    --> Good distribution. Meaningful attention pattern.")

# ============================================================
# VERIFY: OUTPUT IS CONTEXT-DEPENDENT
# ============================================================

print("\n" + "=" * 60)
print("KEY INSIGHT: OUTPUT DEPENDS ON CONTEXT")
print("=" * 60)

print(f"\nOriginal V for 'bank': "
      f"[{', '.join(format_float(v) for v in V[5])}]")
print(f"Output for 'bank':    "
      f"[{', '.join(format_float(v) for v in output[5])}]")
print(f"\nThese are DIFFERENT. The output for 'bank' is a weighted")
print(f"combination of ALL tokens' values, based on attention weights.")
print(f"'bank' now carries information from the words around it.")
print(f"\nThis is the core power of self-attention: every token's")
print(f"representation becomes context-dependent in a single step,")
print(f"with no sequential processing bottleneck.")
```

---

## Key Takeaways

- **Self-attention computes pairwise relevance between all tokens simultaneously.** Unlike RNNs, there is no sequential bottleneck. Every token can directly attend to every other token, regardless of distance in the sequence.
- **Q, K, V are three learned projections of the same input.** The Query asks "what am I looking for?", the Key advertises "what do I contain?", and the Value provides "what information should I contribute?" Attention weights are computed from Q and K, then applied to V.
- **Softmax creates a competition among keys.** Each query's attention weights sum to 1.0, forcing tokens to compete for influence. This means the output for each token is a weighted average of all values, with weights determined by relevance.
- **Scaling by sqrt(d_k) prevents softmax saturation.** Without scaling, large dot products push softmax toward one-hot distributions, destroying the ability to blend information from multiple tokens. The scale factor keeps attention distributions meaningful and learnable.
