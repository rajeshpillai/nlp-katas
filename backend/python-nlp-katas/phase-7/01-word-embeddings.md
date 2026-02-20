# Train Small Embedding-Based Models

> Phase 7 — Small Neural Text Models | Kata 7.1

---

## Concept & Intuition

### What problem are we solving?

Every text representation we have built so far --- Bag of Words, TF-IDF --- treats each word as an **independent dimension**. The word "happy" and the word "joyful" are as unrelated as "happy" and "concrete." Their vectors share no information. This is a fundamental limitation: BoW and TF-IDF cannot capture that words have **relationships**.

Word embeddings solve this by representing each word as a **dense vector** --- a short list of real numbers (typically 50 to 300 dimensions) --- where **similar words have similar vectors**. Instead of a sparse vector with 10,000 dimensions (one per vocabulary word), each word gets a compact vector of, say, 50 dimensions. These dimensions are not human-interpretable, but they encode patterns learned from how words co-occur in text.

The core insight: **words that appear in similar contexts tend to have similar meanings.** The word "king" appears near "throne", "crown", "rule." The word "queen" appears near... the same words. Therefore, "king" and "queen" should have similar vectors. This is the **distributional hypothesis** --- meaning is approximated by context.

### Why earlier methods fail

With BoW or TF-IDF, consider these two documents:

- Doc A: "The **happy** child played in the park"
- Doc B: "The **joyful** kid ran in the garden"

These documents are semantically very similar --- they describe the same scene. But BoW/TF-IDF sees almost zero overlap: "happy" vs "joyful", "child" vs "kid", "played" vs "ran", "park" vs "garden" --- all completely different dimensions. The cosine similarity will be near zero, driven only by shared words like "the" and "in."

The fundamental problem: **one-hot representations have no notion of word similarity.** Every word is equidistant from every other word. "Cat" is as far from "kitten" as it is from "democracy." This is clearly wrong, and no amount of TF-IDF weighting can fix it --- the geometry itself is broken.

### Mental models

- **Sparse vs dense**: BoW gives each word a unique axis in a 10,000-dimensional space. Embeddings compress words into a shared 50-dimensional space where proximity encodes similarity. Think of it as going from "one locker per student" (sparse) to "students sitting in a room where friends sit nearby" (dense).
- **Co-occurrence as meaning**: If two words repeatedly appear near the same neighbors, they are probably related. "Doctor" and "nurse" both appear near "hospital", "patient", "treatment." Their embeddings will be close.
- **Vector arithmetic**: Good embeddings support analogies. If "king" - "man" + "woman" gives a vector closest to "queen", the embedding space has captured gender as a direction. This is an emergent property, not something explicitly programmed.

### Visual explanations

```
SPARSE (BoW) vs DENSE (Embedding) Representations
===================================================

BoW / One-Hot (10,000 dimensions, mostly zeros):

  "cat"    = [0, 0, 1, 0, 0, 0, ..., 0, 0, 0]   (1 in position 42)
  "kitten" = [0, 0, 0, 0, 0, 0, ..., 1, 0, 0]   (1 in position 7831)
  "dog"    = [0, 0, 0, 1, 0, 0, ..., 0, 0, 0]   (1 in position 103)

  Distance(cat, kitten) = sqrt(2) = 1.414
  Distance(cat, dog)    = sqrt(2) = 1.414
  All words are EQUIDISTANT. No similarity captured.


Dense Embedding (50 dimensions, all values used):

  "cat"    = [0.23, -0.11, 0.85, 0.03, ..., 0.42]
  "kitten" = [0.21, -0.09, 0.82, 0.05, ..., 0.39]
  "dog"    = [0.19, -0.15, 0.78, 0.01, ..., 0.35]
  "table"  = [-0.52, 0.73, -0.11, 0.88, ..., -0.21]

  Distance(cat, kitten) = 0.08   (very close!)
  Distance(cat, dog)    = 0.15   (close --- both animals)
  Distance(cat, table)  = 1.87   (far apart --- unrelated)

  Similar words cluster together in the embedding space.


HOW CO-OCCURRENCE BUILDS EMBEDDINGS
====================================

Corpus: "the cat sat on the mat . the dog sat on the rug ."

Co-occurrence window (size 2 --- look 2 words left/right):

  "cat" neighbors: [the, sat, on]
  "dog" neighbors: [the, sat, on]

  "cat" and "dog" share the same neighbors!
  Therefore, they get similar embedding vectors.

  "mat" neighbors: [the, on, .]
  "rug" neighbors: [the, on, .]

  "mat" and "rug" also share neighbors --> similar vectors.


VECTOR ARITHMETIC (Analogy)
=============================

      king - man + woman = ???

           man -----> king
            |          |
            |  (same   |
            |  offset) |
            |          |
          woman ----> queen

  The vector from "man" to "king" encodes "royalty".
  Adding that same vector to "woman" lands near "queen".
  This works because the embedding space organizes concepts
  along consistent directions.
```

---

## Hands-on Exploration

1. Take the sample corpus from the code below. For the word "cat", list all words that appear within a window of 2 positions. Do the same for "dog." Compare the neighbor lists. Which words share the most neighbors? Predict which word pairs will end up with the highest cosine similarity in the embedding space, then run the code to check.

2. Modify the corpus in the code to add more sentences about food (e.g., "the chef cooked a delicious meal", "the cook prepared fresh food"). Run the code again. Do "chef" and "cook" end up with similar vectors? What happens if you add only one food sentence versus five? This demonstrates how **corpus size and content directly determine embedding quality**.

3. The code computes word analogies using vector arithmetic (A - B + C = ?). Try creating your own analogy by choosing three words from the corpus. What happens when the analogy involves words that do not have strong co-occurrence patterns? Why do analogies fail with small corpora but work with large ones?

---

## Live Code

```python
# --- Word Embeddings from Co-occurrence (Pure Python) ---
# Build dense word vectors from a corpus using co-occurrence statistics.
# No numpy, no gensim, no external libraries.

import math
import random
from collections import Counter

random.seed(42)


# ============================================================
# VECTOR MATH HELPERS (replacing numpy)
# ============================================================

def dot_product(v1, v2):
    """Dot product of two vectors."""
    return sum(a * b for a, b in zip(v1, v2))


def magnitude(v):
    """L2 norm of a vector."""
    return math.sqrt(sum(x * x for x in v))


def cosine_similarity(v1, v2):
    """Cosine similarity between two vectors."""
    d = dot_product(v1, v2)
    m1, m2 = magnitude(v1), magnitude(v2)
    if m1 == 0 or m2 == 0:
        return 0.0
    return d / (m1 * m2)


def vector_add(v1, v2):
    """Element-wise addition."""
    return [a + b for a, b in zip(v1, v2)]


def vector_subtract(v1, v2):
    """Element-wise subtraction."""
    return [a - b for a, b in zip(v1, v2)]


def scalar_multiply(scalar, v):
    """Multiply a vector by a scalar."""
    return [scalar * x for x in v]


# ============================================================
# CORPUS AND TOKENIZATION
# ============================================================

corpus = [
    "the cat sat on the mat in the room",
    "the cat chased the mouse around the house",
    "a small cat slept on the soft mat",
    "the dog sat on the rug in the hall",
    "the dog chased the ball around the yard",
    "a big dog slept on the warm rug",
    "the king ruled the land with wisdom",
    "the queen ruled the kingdom with grace",
    "the king wore a golden crown",
    "the queen wore a silver crown",
    "the man walked to the market",
    "the woman walked to the store",
    "the boy played in the park",
    "the girl played in the garden",
    "the cat and the dog sat together",
    "the king and the queen ruled together",
]


def tokenize(text):
    """Lowercase and extract alphabetic words."""
    result = []
    current = []
    for ch in text.lower():
        if ch.isalpha():
            current.append(ch)
        else:
            if current:
                result.append("".join(current))
                current = []
    if current:
        result.append("".join(current))
    return result


# Tokenize entire corpus
all_tokens = []
for sentence in corpus:
    all_tokens.extend(tokenize(sentence))

vocab = sorted(set(all_tokens))
word_to_idx = {w: i for i, w in enumerate(vocab)}
vocab_size = len(vocab)

print("=" * 60)
print("CORPUS AND VOCABULARY")
print("=" * 60)
print(f"  Sentences:     {len(corpus)}")
print(f"  Total tokens:  {len(all_tokens)}")
print(f"  Vocabulary:    {vocab_size} unique words")
print(f"  Words: {vocab}")
print()

# ============================================================
# STEP 1: BUILD CO-OCCURRENCE MATRIX
# ============================================================

WINDOW_SIZE = 2

# Initialize co-occurrence matrix as dict of dicts
cooccurrence = {}
for w in vocab:
    cooccurrence[w] = Counter()

# Slide window across each sentence
for sentence in corpus:
    tokens = tokenize(sentence)
    for i, target in enumerate(tokens):
        start = max(0, i - WINDOW_SIZE)
        end = min(len(tokens), i + WINDOW_SIZE + 1)
        for j in range(start, end):
            if i != j:
                context = tokens[j]
                cooccurrence[target][context] += 1

print("=" * 60)
print("CO-OCCURRENCE MATRIX (sample rows)")
print("=" * 60)

# Show co-occurrence for selected words
focus_words = ["cat", "dog", "king", "queen", "mat", "rug", "man", "woman"]
focus_words = [w for w in focus_words if w in word_to_idx]

for word in focus_words:
    neighbors = cooccurrence[word].most_common(8)
    neighbor_str = ", ".join(f"{w}:{c}" for w, c in neighbors)
    print(f"  {word:>8} -> {neighbor_str}")
print()

# ============================================================
# STEP 2: DIMENSIONALITY REDUCTION VIA RANDOM PROJECTION
# ============================================================
# We project the high-dimensional co-occurrence vectors into
# a lower-dimensional space using random projection.
# This is a simple but effective technique: multiply each
# co-occurrence vector by a random matrix.

EMBED_DIM = 20

# Create random projection matrix (vocab_size x EMBED_DIM)
# Each entry drawn from {-1, +1} (sparse random projection)
random.seed(42)
projection = []
for i in range(vocab_size):
    row = [random.choice([-1, 1]) for _ in range(EMBED_DIM)]
    projection.append(row)

# Build the co-occurrence vector for each word, then project
embeddings = {}
for word in vocab:
    # Full co-occurrence vector
    full_vec = [cooccurrence[word].get(w, 0) for w in vocab]

    # Project: embedding = full_vec @ projection
    embed = []
    for d in range(EMBED_DIM):
        val = sum(full_vec[i] * projection[i][d] for i in range(vocab_size))
        embed.append(val)

    # Normalize to unit length
    mag = magnitude(embed)
    if mag > 0:
        embed = [x / mag for x in embed]

    embeddings[word] = embed

print("=" * 60)
print(f"EMBEDDINGS ({EMBED_DIM}-dimensional, normalized)")
print("=" * 60)
for word in focus_words:
    vec_str = ", ".join(f"{x:+.3f}" for x in embeddings[word][:6])
    print(f"  {word:>8} -> [{vec_str}, ...]")
print()

# ============================================================
# STEP 3: MEASURE WORD SIMILARITY
# ============================================================

print("=" * 60)
print("WORD SIMILARITY (cosine similarity)")
print("=" * 60)

test_pairs = [
    ("cat", "dog"),
    ("cat", "mat"),
    ("king", "queen"),
    ("king", "man"),
    ("man", "woman"),
    ("mat", "rug"),
    ("boy", "girl"),
    ("cat", "king"),
    ("dog", "crown"),
]

test_pairs = [(a, b) for a, b in test_pairs
              if a in embeddings and b in embeddings]

for w1, w2 in test_pairs:
    sim = cosine_similarity(embeddings[w1], embeddings[w2])
    bar_len = int(max(0, sim) * 30)
    bar = "#" * bar_len
    print(f"  {w1:>8} <-> {w2:<8}  {sim:+.4f}  {bar}")
print()

# ============================================================
# STEP 4: FIND MOST SIMILAR WORDS
# ============================================================

print("=" * 60)
print("NEAREST NEIGHBORS")
print("=" * 60)


def most_similar(word, top_n=5):
    """Find the top_n most similar words to the given word."""
    if word not in embeddings:
        return []
    target = embeddings[word]
    scores = []
    for other in vocab:
        if other == word:
            continue
        sim = cosine_similarity(target, embeddings[other])
        scores.append((other, sim))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_n]


for word in ["cat", "king", "mat", "man"]:
    if word not in embeddings:
        continue
    neighbors = most_similar(word, top_n=5)
    neighbor_str = ", ".join(f"{w} ({s:+.3f})" for w, s in neighbors)
    print(f"  {word:>8} -> {neighbor_str}")
print()

# ============================================================
# STEP 5: WORD ANALOGIES (Vector Arithmetic)
# ============================================================

print("=" * 60)
print("WORD ANALOGIES (A - B + C = ?)")
print("=" * 60)


def analogy(a, b, c, top_n=3):
    """Compute A - B + C and find the closest word."""
    if a not in embeddings or b not in embeddings or c not in embeddings:
        return []
    # result_vec = embedding(A) - embedding(B) + embedding(C)
    result = vector_add(vector_subtract(embeddings[a], embeddings[b]),
                        embeddings[c])
    # Normalize
    mag = magnitude(result)
    if mag > 0:
        result = [x / mag for x in result]

    # Find closest words (excluding inputs)
    exclude = {a, b, c}
    scores = []
    for word in vocab:
        if word in exclude:
            continue
        sim = cosine_similarity(result, embeddings[word])
        scores.append((word, sim))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_n]


analogies = [
    ("king", "man", "woman", "queen"),    # king - man + woman = queen?
    ("queen", "woman", "man", "king"),     # queen - woman + man = king?
    ("cat", "mat", "rug", "dog"),          # cat - mat + rug = dog?
]

for a, b, c, expected in analogies:
    if a not in embeddings or b not in embeddings or c not in embeddings:
        continue
    results = analogy(a, b, c)
    result_str = ", ".join(f"{w} ({s:+.3f})" for w, s in results)
    got = results[0][0] if results else "?"
    match = "MATCH" if got == expected else "no match"
    print(f"  {a} - {b} + {c} = ?")
    print(f"    Expected: {expected}")
    print(f"    Got:      {result_str}  [{match}]")
    print()

# ============================================================
# STEP 6: WHY THIS WORKS --- THE DISTRIBUTIONAL HYPOTHESIS
# ============================================================

print("=" * 60)
print("WHY THIS WORKS")
print("=" * 60)
print()
print("  The Distributional Hypothesis:")
print("    'Words that occur in similar contexts tend to have")
print("     similar meanings.'  (Harris, 1954)")
print()
print("  In our corpus:")

for w1, w2 in [("cat", "dog"), ("king", "queen"), ("mat", "rug")]:
    if w1 not in cooccurrence or w2 not in cooccurrence:
        continue
    neighbors1 = set(cooccurrence[w1].keys())
    neighbors2 = set(cooccurrence[w2].keys())
    shared = neighbors1 & neighbors2
    print(f"    '{w1}' and '{w2}' share {len(shared)} context words: "
          f"{sorted(shared)[:8]}")

print()
print("  Shared context words --> similar co-occurrence vectors")
print("  --> similar embeddings --> high cosine similarity")
print()
print("  This is not 'understanding.' This is pattern matching")
print("  over word neighborhoods. But it captures something real")
print("  about how language organizes meaning.")
```

---

## Key Takeaways

- **Word embeddings are dense vectors that encode word relationships.** Unlike BoW/TF-IDF where every word is an independent dimension, embeddings place similar words close together in a shared vector space. "Cat" and "kitten" become neighbors, not strangers.
- **Co-occurrence statistics are the foundation of word embeddings.** Words that appear in the same contexts get similar vectors. This is the distributional hypothesis: meaning is approximated by the company a word keeps. No labeled data or explicit definitions are needed.
- **Vector arithmetic can encode semantic relationships.** Good embedding spaces organize concepts along consistent directions, enabling analogies like "king - man + woman = queen." This is an emergent property of learning from co-occurrence, not something explicitly programmed.
- **Neural models learn representations, not rules.** Unlike handcrafted features (stopword lists, stemming rules), embeddings are learned from data. The model discovers structure in language rather than being told what structure to look for. This is the paradigm shift from classical to neural NLP.
