# Compare Encoder-Only vs Decoder-Only Tasks

> Phase 9 — Transformer Architecture (Core Concepts) | Kata 9.3

---

## Concept & Intuition

### What problem are we solving?

The original transformer paper ("Attention Is All You Need") introduced an encoder-decoder architecture for machine translation. Since then, the field has split into two dominant paradigms: **encoder-only** models (like BERT) that see the full input bidirectionally, and **decoder-only** models (like GPT) that see only previous tokens through causal masking. These are not just architectural choices -- they determine what tasks a model can naturally perform. Understanding the difference between bidirectional and causal attention is essential for understanding modern NLP.

The core question is: **should a token be allowed to look at future tokens?** For classification, yes -- you want to see the whole sentence before deciding. For generation, no -- you cannot look at words you have not yet produced. This single design choice -- the attention mask -- separates the two paradigms.

### Why earlier methods fail

**BoW and TF-IDF** have no concept of directionality at all. They process the entire document as a bag of features. This is fine for classification but provides no mechanism for sequential generation.

**RNNs** are inherently left-to-right (or right-to-left). A standard RNN can only use past context for prediction. Bidirectional RNNs (BiRNNs) can see both directions, but they process sequentially in each direction, making them slow and still limited by the vanishing gradient problem.

**Standard self-attention** (as in Kata 9.1) is bidirectional by default: every token attends to every other token. This is perfect for encoder tasks but violates the fundamental constraint of generation: you cannot attend to tokens that do not yet exist. Causal masking solves this by setting future positions to negative infinity before softmax, ensuring they receive zero attention weight.

### Mental models

- **Encoder as a reader**: The encoder reads the entire input, looking forward and backward, building a complete understanding. Think of reading an essay before writing a summary -- you need to see all of it first. This is bidirectional attention. Every token sees every other token.
- **Decoder as a writer**: The decoder produces output one token at a time, left to right. When writing the third word, it can look back at words one and two, but it cannot look ahead to word four (which does not exist yet). This is causal (masked) attention.
- **The attention mask as a visibility constraint**: The mask is a simple binary matrix. A 1 means "can attend," a 0 means "cannot attend." For encoders, the mask is all 1s. For decoders, it is a lower-triangular matrix (1s on and below the diagonal, 0s above).
- **Tasks follow from architecture**: Classification needs global context (encoder). Generation needs sequential prediction (decoder). Translation needs both: an encoder to understand the source, and a decoder to generate the target.

### Visual explanations

```
ENCODER: BIDIRECTIONAL ATTENTION (full visibility)
===================================================

  Sentence: ["The", "cat", "is", "happy"]

  Attention mask (1 = can attend, 0 = cannot):

         The  cat   is  happy
  The  [  1    1    1    1  ]   <-- "The" sees everything
  cat  [  1    1    1    1  ]   <-- "cat" sees everything
  is   [  1    1    1    1  ]   <-- "is" sees everything
  happy[  1    1    1    1  ]   <-- "happy" sees everything

  Every token has FULL context.
  Good for: classification, NER, similarity, sentence understanding.

  Example (BERT-style):
    Input:  "The movie was [MASK] entertaining"
    Task:   Predict the masked word
    Needs:  Both "The movie was" AND "entertaining" to predict "very"


DECODER: CAUSAL (MASKED) ATTENTION (left-to-right only)
========================================================

  Sentence: ["The", "cat", "is", "happy"]

  Attention mask:

         The  cat   is  happy
  The  [  1    0    0    0  ]   <-- "The" sees only itself
  cat  [  1    1    0    0  ]   <-- "cat" sees "The" and itself
  is   [  1    1    1    0  ]   <-- "is" sees "The", "cat", and itself
  happy[  1    1    1    1  ]   <-- "happy" sees everything before it

  Each token can ONLY see previous tokens (and itself).
  Good for: text generation, language modeling, completion.

  Example (GPT-style):
    Input:  "The cat is"
    Task:   Predict the next word
    Needs:  Only "The cat is" to predict "happy"
    Cannot look ahead to future tokens!


MASKING IN PRACTICE
===================

  Raw attention scores (before masking):

         The  cat   is  happy
  The  [ 2.1  0.8  1.3  0.5 ]
  cat  [ 1.0  3.2  0.7  1.1 ]
  is   [ 0.5  1.4  2.8  0.9 ]
  happy[ 0.3  0.6  1.2  2.5 ]

  After applying causal mask (set future positions to -inf):

         The   cat    is   happy
  The  [ 2.1  -inf  -inf  -inf ]   softmax -> [1.00  0.00  0.00  0.00]
  cat  [ 1.0   3.2  -inf  -inf ]   softmax -> [0.10  0.90  0.00  0.00]
  is   [ 0.5   1.4   2.8  -inf ]   softmax -> [0.07  0.17  0.70  0.00]
  happy[ 0.3   0.6   1.2   2.5 ]   softmax -> [0.05  0.07  0.12  0.76]

  -inf through softmax becomes 0.00 -- those positions are invisible.
```

---

## Hands-on Exploration

1. Consider the sentence "I went to the bank to deposit my check." For a classification task (is this about finance or geography?), which attention pattern is better -- bidirectional or causal? Now consider generating the next word after "I went to the bank to deposit my". Can you use bidirectional attention here? Why or why not?

2. Look at the causal attention mask. The last token ("happy") can see all previous tokens. The first token ("The") can only see itself. This means earlier tokens have less context than later tokens. How might this affect the quality of representations at different positions? In practice, models use positional encoding to help, but the asymmetry is fundamental to the causal architecture.

3. Some tasks seem to need both patterns. Machine translation reads the full source sentence (encoder, bidirectional), then generates the translation word by word (decoder, causal). The original transformer uses an encoder-decoder architecture for exactly this reason. Can you think of other tasks that naturally fit each architecture? List two tasks for encoder-only and two for decoder-only.

---

## Live Code

```python
# --- Encoder vs Decoder: Attention Masking and Task Differences ---
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
    """Compute softmax (numerically stable). Handles -inf correctly."""
    # Filter out -inf for max computation
    finite_vals = [v for v in values if v != float('-inf')]
    if not finite_vals:
        # All -inf: return uniform (edge case)
        return [1.0 / len(values)] * len(values)
    max_val = max(finite_vals)
    exps = []
    for v in values:
        if v == float('-inf'):
            exps.append(0.0)
        else:
            exps.append(math.exp(v - max_val))
    total = sum(exps)
    if total == 0:
        return [1.0 / len(values)] * len(values)
    return [e / total for e in exps]


def format_float(v, width=7, decimals=3):
    """Format a float for display."""
    return f"{v:>{width}.{decimals}f}"


# ============================================================
# ATTENTION WITH CONFIGURABLE MASKING
# ============================================================

def attention_with_mask(Q, K, V, mask, d_k):
    """
    Scaled dot-product attention with an explicit mask.
    mask[i][j] = 1 means token i CAN attend to token j.
    mask[i][j] = 0 means token i CANNOT attend to token j.
    """
    seq_len = len(Q)
    scale = math.sqrt(d_k)

    # Step 1: Compute raw scores Q * K^T
    K_T = transpose(K)
    scores = mat_mul(Q, K_T)

    # Step 2: Scale
    scaled = [[scores[i][j] / scale for j in range(seq_len)]
              for i in range(seq_len)]

    # Step 3: Apply mask (set masked positions to -inf)
    masked = []
    for i in range(seq_len):
        row = []
        for j in range(seq_len):
            if mask[i][j] == 0:
                row.append(float('-inf'))
            else:
                row.append(scaled[i][j])
        masked.append(row)

    # Step 4: Softmax
    weights = [softmax(row) for row in masked]

    # Step 5: Weighted sum of V
    output = mat_mul(weights, V)

    return output, weights, scores


# ============================================================
# MASK GENERATORS
# ============================================================

def encoder_mask(seq_len):
    """Bidirectional mask: every token sees every token."""
    return [[1] * seq_len for _ in range(seq_len)]


def decoder_mask(seq_len):
    """Causal mask: each token sees only itself and previous tokens."""
    mask = []
    for i in range(seq_len):
        row = []
        for j in range(seq_len):
            if j <= i:
                row.append(1)
            else:
                row.append(0)
        mask.append(row)
    return mask


# ============================================================
# SETUP
# ============================================================

words = ["The", "cat", "is", "happy"]
seq_len = len(words)
d_model = 8
d_k = 4

random.seed(42)

# Create embeddings
embeddings = make_matrix(seq_len, d_model)

# Create shared Q, K, V projection weights
# (same projections for both encoder and decoder, to make comparison fair)
W_Q = make_matrix(d_model, d_k)
W_K = make_matrix(d_model, d_k)
W_V = make_matrix(d_model, d_k)

# Project to Q, K, V
Q = mat_mul(embeddings, W_Q)
K = mat_mul(embeddings, W_K)
V = mat_mul(embeddings, W_V)

# ============================================================
# PRINT ATTENTION MASKS
# ============================================================

print("=" * 60)
print("ENCODER vs DECODER: ATTENTION MASKS")
print("=" * 60)

enc_mask = encoder_mask(seq_len)
dec_mask = decoder_mask(seq_len)

print("\n--- Encoder Mask (Bidirectional) ---")
print("  Every token sees every token.\n")
header = "          " + "".join(f"{w:>7s}" for w in words)
print(header)
for i, word in enumerate(words):
    row_str = "".join(f"{enc_mask[i][j]:>7d}" for j in range(seq_len))
    print(f"  {word:>6s}:  {row_str}")

print("\n--- Decoder Mask (Causal) ---")
print("  Each token sees only previous tokens and itself.\n")
header = "          " + "".join(f"{w:>7s}" for w in words)
print(header)
for i, word in enumerate(words):
    row_str = "".join(f"{dec_mask[i][j]:>7d}" for j in range(seq_len))
    print(f"  {word:>6s}:  {row_str}")

# ============================================================
# RUN BOTH ATTENTION PATTERNS
# ============================================================

print("\n" + "=" * 60)
print("ENCODER ATTENTION (Bidirectional)")
print("=" * 60)

enc_output, enc_weights, enc_scores = attention_with_mask(
    Q, K, V, enc_mask, d_k)

print("\n--- Attention Weights (encoder: full visibility) ---")
header = "          " + "".join(f"{w:>8s}" for w in words)
print(header)
for i, word in enumerate(words):
    row_str = "".join(format_float(enc_weights[i][j], 8)
                      for j in range(seq_len))
    print(f"  {word:>6s}:  {row_str}")

print("\n--- Encoder Output ---")
for i, word in enumerate(words):
    vec_str = " ".join(format_float(v) for v in enc_output[i])
    print(f"  {word:>6s}: [{vec_str}]")


print("\n" + "=" * 60)
print("DECODER ATTENTION (Causal)")
print("=" * 60)

dec_output, dec_weights, dec_scores = attention_with_mask(
    Q, K, V, dec_mask, d_k)

print("\n--- Attention Weights (decoder: causal masking) ---")
header = "          " + "".join(f"{w:>8s}" for w in words)
print(header)
for i, word in enumerate(words):
    row_str = "".join(format_float(dec_weights[i][j], 8)
                      for j in range(seq_len))
    future_blocked = seq_len - i - 1
    print(f"  {word:>6s}:  {row_str}"
          f"  ({future_blocked} future token(s) blocked)")

print("\n--- Decoder Output ---")
for i, word in enumerate(words):
    vec_str = " ".join(format_float(v) for v in dec_output[i])
    print(f"  {word:>6s}: [{vec_str}]")

# ============================================================
# COMPARE OUTPUTS: HOW MASKING CHANGES REPRESENTATIONS
# ============================================================

print("\n" + "=" * 60)
print("HOW MASKING CHANGES REPRESENTATIONS")
print("=" * 60)

print("\n  Same Q, K, V — different masks — different outputs.\n")

for i, word in enumerate(words):
    # Euclidean distance between encoder and decoder outputs
    diff = [enc_output[i][j] - dec_output[i][j] for j in range(d_k)]
    dist = math.sqrt(sum(d * d for d in diff))

    # How much context each version has
    enc_ctx = seq_len
    dec_ctx = i + 1

    print(f"  '{word}':")
    print(f"    Encoder context: {enc_ctx} tokens (all)")
    print(f"    Decoder context: {dec_ctx} token(s) (up to self)")
    print(f"    Output difference: {dist:.4f}")
    if i == seq_len - 1:
        print(f"    Note: Last token sees ALL tokens in both modes!")
    print()

# ============================================================
# TASK DEMONSTRATION: CLASSIFICATION (ENCODER-STYLE)
# ============================================================

print("=" * 60)
print("TASK 1: CLASSIFICATION (Encoder-Style)")
print("=" * 60)

print("""
  Classification needs to see the ENTIRE input before deciding.

  Example: Sentiment analysis
    Input:  "The cat is happy"
    Output: POSITIVE or NEGATIVE

  Strategy: Use encoder (bidirectional) attention, then aggregate
  all token representations into a single sentence vector.
""")

# Aggregate encoder output into a single vector (mean pooling)
sentence_vec = [0.0] * d_k
for i in range(seq_len):
    for j in range(d_k):
        sentence_vec[j] += enc_output[i][j]
sentence_vec = [v / seq_len for v in sentence_vec]

# Simple linear classifier: dot product with a "positive" direction
random.seed(99)
classifier_weights = make_vector(d_k)
score = sum(s * w for s, w in zip(sentence_vec, classifier_weights))
prediction = "POSITIVE" if score > 0 else "NEGATIVE"

print(f"  Sentence vector (mean pool): "
      f"[{' '.join(format_float(v) for v in sentence_vec)}]")
print(f"  Classifier score: {score:.4f}")
print(f"  Prediction: {prediction}")
print(f"\n  Key: The encoder saw ALL tokens bidirectionally.")
print(f"  'happy' influenced 'The', and 'The' influenced 'happy'.")

# ============================================================
# TASK DEMONSTRATION: GENERATION (DECODER-STYLE)
# ============================================================

print("\n" + "=" * 60)
print("TASK 2: NEXT-TOKEN PREDICTION (Decoder-Style)")
print("=" * 60)

print("""
  Generation produces tokens one at a time, left to right.
  At each step, the model can ONLY see previous tokens.

  Example: Language modeling
    Step 1: Input ["The"]           -> predict "cat"
    Step 2: Input ["The", "cat"]    -> predict "is"
    Step 3: Input ["The", "cat", "is"] -> predict "happy"
""")

# Simulate next-token prediction at each position
random.seed(77)
# Simple "vocabulary" for demonstration
vocab = ["the", "cat", "dog", "is", "was", "happy", "sad", "running"]
vocab_vectors = make_matrix(len(vocab), d_k)

print("  Simulated generation (using decoder attention):\n")

for step in range(seq_len - 1):
    # The decoder output at position 'step' is used to predict next token
    current_output = dec_output[step]
    context = " ".join(words[:step + 1])

    # Score each vocabulary word by similarity to decoder output
    scores = []
    for v_idx, v_vec in enumerate(vocab_vectors):
        score = sum(a * b for a, b in zip(current_output, v_vec))
        scores.append(score)

    # Apply softmax to get probabilities
    probs = softmax(scores)

    # Find top-3 predictions
    ranked = sorted(range(len(vocab)), key=lambda x: probs[x], reverse=True)

    print(f"  Step {step + 1}: Context = \"{context}\"")
    print(f"    Top predictions:")
    for rank in range(3):
        idx = ranked[rank]
        bar_len = int(probs[idx] * 30)
        bar = "#" * bar_len
        print(f"      {rank + 1}. '{vocab[idx]}'"
              f" (prob={probs[idx]:.3f}) {bar}")
    print(f"    Actual next: '{words[step + 1]}'\n")

print("  Key: At each step, the decoder ONLY sees previous tokens.")
print("  It cannot look ahead — causal masking enforces this.")

# ============================================================
# POSITIONAL ENCODING: WHY IT MATTERS
# ============================================================

print("\n" + "=" * 60)
print("POSITIONAL ENCODING: GIVING ORDER TO ATTENTION")
print("=" * 60)

print("""
  Self-attention has no built-in sense of word order.
  "The cat sat" and "sat cat The" produce the SAME attention
  scores if we just use word embeddings (the set of dot products
  is identical, just in different positions).

  Positional encoding adds a position-dependent signal to each
  embedding, so the model can distinguish word order.
""")

def positional_encoding(seq_len, d_model):
    """
    Sinusoidal positional encoding from 'Attention Is All You Need'.
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    PE = []
    for pos in range(seq_len):
        pe_row = []
        for i in range(d_model):
            if i % 2 == 0:
                val = math.sin(pos / (10000 ** (i / d_model)))
            else:
                val = math.cos(pos / (10000 ** ((i - 1) / d_model)))
            pe_row.append(val)
        PE.append(pe_row)
    return PE


PE = positional_encoding(seq_len, d_model)

print("  Positional encoding vectors (sinusoidal):\n")
for i, word in enumerate(words):
    vec_str = " ".join(format_float(v) for v in PE[i])
    print(f"    pos {i} ({word:>5s}): [{vec_str}]")

# Show that positions are distinguishable
print("\n  Distance between positional encodings:\n")
for i in range(seq_len):
    for j in range(i + 1, seq_len):
        dist = math.sqrt(sum((PE[i][k] - PE[j][k]) ** 2
                              for k in range(d_model)))
        print(f"    pos {i} <-> pos {j} "
              f"({words[i]:>5s} <-> {words[j]:<5s}): "
              f"distance = {dist:.4f}")

print("""
  Nearby positions have smaller distances than distant positions.
  This gives the model a smooth sense of relative ordering.
  Without positional encoding, attention treats input as a SET,
  not a SEQUENCE.
""")

# ============================================================
# PARALLELISM: WHY ATTENTION BEATS RNNs
# ============================================================

print("=" * 60)
print("PARALLELISM: WHY TRANSFORMERS ARE FAST")
print("=" * 60)

print(f"""
  RNN (sequential):
    Step 1: process "The"     -> h1
    Step 2: process "cat"     -> h2 (needs h1)
    Step 3: process "is"      -> h3 (needs h2)
    Step 4: process "happy"   -> h4 (needs h3)
    Total: 4 sequential steps. Cannot parallelize.

  Transformer attention (parallel):
    All Q*K^T dot products computed simultaneously.
    All softmax operations computed simultaneously.
    All weighted sums computed simultaneously.
    Total: 1 parallel step (with O(n^2) compute, but parallelizable).

  For a sequence of length {seq_len}:
    RNN steps:         {seq_len} (sequential, one after another)
    Attention steps:   1 (parallel, all at once on GPU)

  For a sequence of length 512:
    RNN steps:         512 (sequential)
    Attention steps:   1 (parallel, but 512x512 = 262,144 dot products)

  Transformers trade sequential time for parallel computation.
  This is why they train much faster on modern hardware (GPUs/TPUs)
  that excel at parallel matrix operations.
""")
```

---

## Key Takeaways

- **Encoder and decoder differ by a single mechanism: the attention mask.** Bidirectional (encoder) masks allow every token to see every other token, enabling full-context understanding. Causal (decoder) masks restrict each token to see only previous tokens, enabling sequential generation. The same attention computation, with different masks, produces fundamentally different behavior.
- **The attention mask determines which tasks a model can perform.** Encoder-only models excel at classification, NER, and similarity tasks where full context is available. Decoder-only models excel at text generation, completion, and language modeling where outputs are produced sequentially. Choosing the wrong architecture for a task creates a fundamental mismatch.
- **Positional encoding gives transformers a sense of order.** Pure self-attention is permutation-invariant: it treats the input as a set, not a sequence. Sinusoidal positional encodings add a position-dependent signal so the model can distinguish "the cat sat" from "sat the cat." Without this, word order information is lost entirely.
- **Transformers replace sequential processing with parallel computation.** RNNs must process tokens one at a time, creating a bottleneck proportional to sequence length. Transformers compute all pairwise attention scores simultaneously, trading sequential steps for parallelizable matrix operations. This is why transformers train orders of magnitude faster on modern hardware.
