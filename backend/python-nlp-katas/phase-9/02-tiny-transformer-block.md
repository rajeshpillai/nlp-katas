# Build a Tiny Transformer Block

> Phase 9 — Transformer Architecture (Core Concepts) | Kata 9.2

---

## Concept & Intuition

### What problem are we solving?

In Kata 9.1, we implemented self-attention: the ability for every token to attend to every other token. But self-attention alone is not enough. A single attention operation computes a weighted sum of values -- this is a linear operation. Without additional components, stacking attention layers would still only produce linear combinations. The transformer block adds the non-linear and stabilizing components that make deep, trainable networks possible.

A transformer encoder block combines four ideas: **multi-head self-attention** (attend to different things in parallel), **residual connections** (let gradients flow through unmodified), **layer normalization** (stabilize activations), and a **feed-forward network** (add non-linearity). Each component solves a specific problem, and together they form the building block that you can stack 6, 12, or 96 times to create models of increasing capacity.

### Why earlier methods fail

**Single-head attention** looks at the input from one "perspective." But language has many simultaneous relationships: syntactic (subject-verb agreement), semantic (word meaning), positional (nearby words), and referential (pronoun resolution). A single attention head must cram all of these into one set of weights. Multi-head attention solves this by running several attention operations in parallel, each with its own Q, K, V projections, then concatenating the results.

**Deep networks without residual connections** suffer from vanishing gradients. Information from the input must pass through many transformations, and errors in later layers struggle to propagate back to earlier ones. Residual connections add a shortcut: the output of each sub-layer is added to its input. This means gradients can flow directly through the addition, bypassing the transformation entirely if needed.

**Deep networks without normalization** suffer from internal covariate shift: the distribution of activations changes at each layer, making training unstable. Layer normalization standardizes activations within each layer, keeping values in a consistent range and enabling faster, more stable training.

**Attention alone is linear.** The weighted sum of values is a linear combination. The feed-forward network (two linear layers with a non-linear activation in between) adds the capacity to learn non-linear transformations of the attended representations.

### Mental models

- **Multi-head attention as parallel perspectives**: Imagine reading a sentence. One "head" focuses on syntactic structure (subject-verb pairs), another on semantic similarity (synonyms and related words), another on positional proximity (nearby words). Each head is a separate attention mechanism with its own learned weights. The results are concatenated to give each token a rich, multi-faceted representation.
- **Residual connections as safety nets**: Think of the residual connection as saying "at worst, pass through the original input unchanged." The transformation only needs to learn the *difference* (residual) from the input, which is easier to learn than the full output from scratch.
- **Layer normalization as recalibration**: After each sub-layer, normalize the activations so they have zero mean and unit variance. This prevents values from exploding or collapsing as they pass through many layers.
- **Feed-forward network as token-level processing**: Unlike attention, which mixes information across tokens, the FFN processes each token position independently. It applies the same transformation to every token, adding non-linear capacity.

### Visual explanations

```
TRANSFORMER ENCODER BLOCK
==========================

  Input Embeddings (seq_len x d_model)
         |
         v
  +-------------------------------+
  |   Multi-Head Self-Attention   |   <-- attend to all positions
  +-------------------------------+
         |
         +---> Add (residual)  <---+   <-- input + attention output
         |                         |
         v                         |
  +-------------------------------+
  |      Layer Normalization      |   <-- stabilize activations
  +-------------------------------+
         |
         v
  +-------------------------------+
  |    Feed-Forward Network       |   <-- two linear layers + ReLU
  |    Linear(d_model, d_ff)      |       expand, activate, compress
  |    ReLU                       |
  |    Linear(d_ff, d_model)      |
  +-------------------------------+
         |
         +---> Add (residual)  <---+   <-- FFN input + FFN output
         |                         |
         v                         |
  +-------------------------------+
  |      Layer Normalization      |   <-- stabilize again
  +-------------------------------+
         |
         v
  Output Embeddings (seq_len x d_model)
  (same shape as input, but context-aware!)


MULTI-HEAD ATTENTION (n_heads = 2, d_model = 8, d_k = 4)
=========================================================

  Input (seq_len x 8)
    |
    +----------+----------+
    |          |          |
  Head 1     Head 2     ...
  (Q1,K1,V1) (Q2,K2,V2)
  d_k=4      d_k=4
    |          |
  Attn 1     Attn 2     (each head: seq_len x d_k)
    |          |
    +----+-----+
         |
    Concatenate          (seq_len x d_model)
         |
    Linear (W_O)         (project back to d_model)
         |
    Output


LAYER NORMALIZATION (per token position)
=========================================

  Token vector: [3.2, -1.5, 0.8, 2.1, -0.3, 1.7, -2.4, 0.5]
                 |
                 v
  1. Compute mean:     mu = 0.5125
  2. Compute variance: var = 3.032
  3. Normalize:        x_i' = (x_i - mu) / sqrt(var + eps)
  4. Scale and shift:  out_i = gamma * x_i' + beta
                 |
                 v
  Normalized:   [1.54, -1.16, 0.17, 0.91, -0.47, 0.68, -1.67, -0.01]
                (zero mean, unit variance -- stable for next layer)


RESIDUAL CONNECTION (why it helps)
===================================

  Without residual:     output = F(x)
    - Gradient must flow entirely through F
    - If F has vanishing gradients, learning stops

  With residual:        output = x + F(x)
    - Gradient flows through BOTH paths
    - Even if F's gradient vanishes, the identity path carries it
    - F only needs to learn the DIFFERENCE from input
```

---

## Hands-on Exploration

1. The transformer block has two "Add & Norm" operations: one after multi-head attention, and one after the feed-forward network. Remove the residual connections in the code (set the output to just the sub-layer output, without adding the input). Run the code and compare the output values. Without residuals, the values tend to drift further from the input, making deep stacking unstable.

2. The feed-forward network expands the dimension from d_model to d_ff (typically 4x larger), applies a non-linearity, then compresses back. Why expand first? Think about it this way: the expansion creates a higher-dimensional space where the non-linearity can carve out more complex decision boundaries. Try changing d_ff from 16 to 8 (same as d_model) and observe whether the output changes significantly.

3. Multi-head attention splits d_model into n_heads parallel attention operations. With d_model=8 and n_heads=2, each head has d_k=4. What happens if you use n_heads=4 (each head gets d_k=2)? More heads means more "perspectives" but each perspective sees fewer dimensions. There is a tradeoff between the number of perspectives and the richness of each one.

---

## Live Code

```python
# --- Tiny Transformer Encoder Block from Scratch ---
# Using ONLY Python standard library

import math
import random

random.seed(42)


# ============================================================
# VECTOR AND MATRIX HELPERS
# ============================================================

def make_vector(dim):
    """Create a random vector sampled from N(0, 1)."""
    return [random.gauss(0, 1) for _ in range(dim)]


def make_matrix(rows, cols):
    """Create a random matrix."""
    return [make_vector(cols) for _ in range(rows)]


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
    """Compute softmax (numerically stable)."""
    max_val = max(values)
    exps = [math.exp(v - max_val) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


def relu(x):
    """ReLU activation: max(0, x)."""
    return max(0.0, x)


def vec_add(v1, v2):
    """Element-wise addition of two vectors."""
    return [a + b for a, b in zip(v1, v2)]


def mat_add(A, B):
    """Element-wise addition of two matrices."""
    return [vec_add(a, b) for a, b in zip(A, B)]


def format_vec(v, width=7, decimals=3):
    """Format a vector for printing."""
    return " ".join(f"{x:>{width}.{decimals}f}" for x in v)


# ============================================================
# LAYER NORMALIZATION (from scratch)
# ============================================================

def layer_norm(vector, gamma=None, beta=None, eps=1e-5):
    """
    Layer normalization for a single vector.
    Normalizes to zero mean and unit variance, then applies
    learnable scale (gamma) and shift (beta).
    """
    d = len(vector)

    # Compute mean
    mean = sum(vector) / d

    # Compute variance
    variance = sum((x - mean) ** 2 for x in vector) / d

    # Normalize
    normalized = [(x - mean) / math.sqrt(variance + eps) for x in vector]

    # Apply scale and shift (default: gamma=1, beta=0 -- identity)
    if gamma is None:
        gamma = [1.0] * d
    if beta is None:
        beta = [0.0] * d

    output = [g * n + b for g, n, b in zip(gamma, normalized, beta)]
    return output, mean, variance


def layer_norm_matrix(M, eps=1e-5):
    """Apply layer norm to each row of a matrix."""
    results = []
    stats = []
    for row in M:
        normed, mean, var = layer_norm(row, eps=eps)
        results.append(normed)
        stats.append((mean, var))
    return results, stats


# ============================================================
# SINGLE-HEAD ATTENTION
# ============================================================

def single_head_attention(X, W_Q, W_K, W_V, d_k):
    """
    Compute scaled dot-product attention for one head.
    X: input (seq_len x d_head_in)
    Returns: output (seq_len x d_k), attention weights
    """
    Q = mat_mul(X, W_Q)
    K = mat_mul(X, W_K)
    V = mat_mul(X, W_V)

    # Scores = Q * K^T / sqrt(d_k)
    K_T = transpose(K)
    scores = mat_mul(Q, K_T)
    scale = math.sqrt(d_k)
    scaled = [[s / scale for s in row] for row in scores]

    # Softmax each row
    weights = [softmax(row) for row in scaled]

    # Output = weights * V
    output = mat_mul(weights, V)
    return output, weights


# ============================================================
# MULTI-HEAD ATTENTION
# ============================================================

def multi_head_attention(X, n_heads, d_model):
    """
    Multi-head self-attention.
    Splits d_model into n_heads, runs attention on each,
    concatenates, and projects with W_O.
    """
    d_k = d_model // n_heads
    seq_len = len(X)

    all_head_outputs = []
    all_head_weights = []

    for h in range(n_heads):
        # Each head gets its own W_Q, W_K, W_V
        W_Q = make_matrix(d_model, d_k)
        W_K = make_matrix(d_model, d_k)
        W_V = make_matrix(d_model, d_k)

        head_out, head_weights = single_head_attention(X, W_Q, W_K, W_V, d_k)
        all_head_outputs.append(head_out)
        all_head_weights.append(head_weights)

    # Concatenate heads: for each position, concat all head outputs
    concat = []
    for i in range(seq_len):
        combined = []
        for h in range(n_heads):
            combined.extend(all_head_outputs[h][i])
        concat.append(combined)

    # Project through W_O (d_model x d_model)
    W_O = make_matrix(d_model, d_model)
    output = mat_mul(concat, W_O)

    return output, all_head_weights


# ============================================================
# FEED-FORWARD NETWORK
# ============================================================

def feed_forward(X, d_model, d_ff):
    """
    Position-wise feed-forward network.
    FFN(x) = ReLU(x * W1 + b1) * W2 + b2
    Applied independently to each token position.
    """
    W1 = make_matrix(d_model, d_ff)
    b1 = make_vector(d_ff)
    W2 = make_matrix(d_ff, d_model)
    b2 = make_vector(d_model)

    # First linear layer: X * W1 + b1
    hidden = mat_mul(X, W1)
    # Add bias and apply ReLU
    for i in range(len(hidden)):
        hidden[i] = [relu(hidden[i][j] + b1[j]) for j in range(d_ff)]

    # Second linear layer: hidden * W2 + b2
    output = mat_mul(hidden, W2)
    for i in range(len(output)):
        output[i] = [output[i][j] + b2[j] for j in range(d_model)]

    return output, hidden


# ============================================================
# FULL TRANSFORMER ENCODER BLOCK
# ============================================================

def transformer_encoder_block(X, n_heads, d_model, d_ff):
    """
    One transformer encoder block:
    1. Multi-head self-attention
    2. Add & Norm (residual + layer norm)
    3. Feed-forward network
    4. Add & Norm (residual + layer norm)
    """
    # --- Sub-layer 1: Multi-Head Self-Attention ---
    attn_output, attn_weights = multi_head_attention(X, n_heads, d_model)

    # Residual connection: add input to attention output
    residual_1 = mat_add(X, attn_output)

    # Layer normalization
    norm_1, norm_1_stats = layer_norm_matrix(residual_1)

    # --- Sub-layer 2: Feed-Forward Network ---
    ffn_output, ffn_hidden = feed_forward(norm_1, d_model, d_ff)

    # Residual connection: add norm_1 input to FFN output
    residual_2 = mat_add(norm_1, ffn_output)

    # Layer normalization
    norm_2, norm_2_stats = layer_norm_matrix(residual_2)

    return {
        "attn_output": attn_output,
        "attn_weights": attn_weights,
        "residual_1": residual_1,
        "norm_1": norm_1,
        "norm_1_stats": norm_1_stats,
        "ffn_hidden": ffn_hidden,
        "ffn_output": ffn_output,
        "residual_2": residual_2,
        "norm_2": norm_2,
        "norm_2_stats": norm_2_stats,
        "final_output": norm_2,
    }


# ============================================================
# RUN THE TRANSFORMER BLOCK
# ============================================================

# Configuration
words = ["The", "cat", "sat", "on"]
seq_len = len(words)
d_model = 8    # embedding dimension
n_heads = 2    # number of attention heads
d_ff = 16      # feed-forward hidden dimension
d_k = d_model // n_heads  # per-head dimension

print("=" * 60)
print("TINY TRANSFORMER ENCODER BLOCK")
print("=" * 60)
print(f"\nSentence:       {words}")
print(f"d_model:        {d_model}")
print(f"n_heads:        {n_heads}")
print(f"d_k (per head): {d_k}")
print(f"d_ff:           {d_ff}")

# Create input embeddings
random.seed(42)
X = make_matrix(seq_len, d_model)

print(f"\n--- Stage 1: Input Embeddings ({seq_len} x {d_model}) ---")
for i, word in enumerate(words):
    print(f"  {word:>4s}: [{format_vec(X[i])}]")

# Run the full block
result = transformer_encoder_block(X, n_heads, d_model, d_ff)

# --- Print intermediate representations ---

print(f"\n--- Stage 2: After Multi-Head Attention ({seq_len} x {d_model}) ---")
for i, word in enumerate(words):
    print(f"  {word:>4s}: [{format_vec(result['attn_output'][i])}]")

print(f"\n--- Stage 3: After Residual Connection (input + attention) ---")
for i, word in enumerate(words):
    print(f"  {word:>4s}: [{format_vec(result['residual_1'][i])}]")

print(f"\n--- Stage 4: After Layer Norm 1 ---")
for i, word in enumerate(words):
    mean, var = result['norm_1_stats'][i]
    print(f"  {word:>4s}: [{format_vec(result['norm_1'][i])}]"
          f"  (mu={mean:+.3f}, var={var:.3f})")

print(f"\n--- Stage 5: FFN Hidden Layer ({seq_len} x {d_ff}, after ReLU) ---")
for i, word in enumerate(words):
    # Show first 8 of d_ff=16 dims to keep output readable
    hidden_preview = result['ffn_hidden'][i][:8]
    zeros = sum(1 for v in result['ffn_hidden'][i] if v == 0.0)
    print(f"  {word:>4s}: [{format_vec(hidden_preview)}] ..."
          f"  ({zeros}/{d_ff} zeroed by ReLU)")

print(f"\n--- Stage 6: After FFN ({seq_len} x {d_model}) ---")
for i, word in enumerate(words):
    print(f"  {word:>4s}: [{format_vec(result['ffn_output'][i])}]")

print(f"\n--- Stage 7: After Residual Connection (norm_1 + FFN) ---")
for i, word in enumerate(words):
    print(f"  {word:>4s}: [{format_vec(result['residual_2'][i])}]")

print(f"\n--- Stage 8: Final Output (After Layer Norm 2) ---")
for i, word in enumerate(words):
    mean, var = result['norm_2_stats'][i]
    print(f"  {word:>4s}: [{format_vec(result['final_output'][i])}]"
          f"  (mu={mean:+.3f}, var={var:.3f})")

# ============================================================
# ATTENTION WEIGHTS PER HEAD
# ============================================================

print("\n" + "=" * 60)
print("ATTENTION WEIGHTS PER HEAD")
print("=" * 60)

for h in range(n_heads):
    print(f"\n  Head {h + 1}:")
    header = "          " + "".join(f"{w:>8s}" for w in words)
    print(header)
    for i, word in enumerate(words):
        row_str = "".join(f"{result['attn_weights'][h][i][j]:>8.3f}"
                          for j in range(seq_len))
        print(f"  {word:>6s}:  {row_str}")

print(f"\n  Each head learns a DIFFERENT attention pattern.")
print(f"  Head 1 might focus on nearby words, while Head 2")
print(f"  might focus on syntactically related words.")

# ============================================================
# LAYER NORM DEMONSTRATION
# ============================================================

print("\n" + "=" * 60)
print("LAYER NORMALIZATION IN DETAIL")
print("=" * 60)

example_vec = result['residual_1'][0]  # pre-norm vector for first token
normed, mean, var = layer_norm(example_vec)

print(f"\n  Before LayerNorm ('{words[0]}' after residual):")
print(f"    Values: [{format_vec(example_vec)}]")
print(f"    Mean:   {mean:+.4f}")
print(f"    Var:    {var:.4f}")
print(f"    Range:  [{min(example_vec):.3f}, {max(example_vec):.3f}]")

print(f"\n  After LayerNorm:")
print(f"    Values: [{format_vec(normed)}]")
normed_mean = sum(normed) / len(normed)
normed_var = sum((x - normed_mean) ** 2 for x in normed) / len(normed)
print(f"    Mean:   {normed_mean:+.4f}  (should be ~0)")
print(f"    Var:    {normed_var:.4f}  (should be ~1)")
print(f"    Range:  [{min(normed):.3f}, {max(normed):.3f}]")

# ============================================================
# COMPARE INPUT VS OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("INPUT vs OUTPUT: WHAT CHANGED?")
print("=" * 60)

print(f"\n  Shape preserved: ({seq_len} x {d_model}) -> ({seq_len} x {d_model})")
print(f"\n  But every token now carries information from all other tokens.")
print(f"  Let's measure how much each token changed:\n")

for i, word in enumerate(words):
    # Euclidean distance between input and output
    diff = [result['final_output'][i][j] - X[i][j]
            for j in range(d_model)]
    dist = math.sqrt(sum(d * d for d in diff))

    # Cosine similarity between input and output
    dot_prod = sum(a * b for a, b in zip(X[i], result['final_output'][i]))
    mag_in = math.sqrt(sum(a * a for a in X[i]))
    mag_out = math.sqrt(sum(a * a for a in result['final_output'][i]))
    cos_sim = dot_prod / (mag_in * mag_out) if (mag_in * mag_out) > 0 else 0

    print(f"  '{word}': distance={dist:.3f}, "
          f"cosine_similarity={cos_sim:.3f}")

print(f"\n  Same dimensionality, different content.")
print(f"  The transformer block transforms isolated embeddings")
print(f"  into context-aware representations, in one pass.")
```

---

## Key Takeaways

- **A transformer block has four components working together.** Multi-head self-attention captures relationships between tokens. Residual connections ensure gradient flow. Layer normalization stabilizes activations. The feed-forward network adds non-linear capacity. Remove any one, and the architecture degrades.
- **Multi-head attention provides multiple perspectives.** Each head can learn different types of relationships (syntactic, semantic, positional). The concatenated result gives each token a richer representation than any single head could provide.
- **Residual connections make depth possible.** Without them, stacking many layers would cause vanishing gradients. The identity shortcut guarantees that information and gradients can flow through the network, and each layer only needs to learn the incremental improvement (the residual).
- **The output has the same shape as the input, but different content.** A transformer block takes (seq_len x d_model) and produces (seq_len x d_model). This shape preservation is what allows blocks to be stacked arbitrarily deep, with each block refining the representations produced by the one before it.
