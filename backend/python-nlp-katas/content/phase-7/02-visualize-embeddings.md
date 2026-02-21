# Visualize Embedding Spaces

> Phase 7 — Small Neural Text Models | Kata 7.2

---

## Concept & Intuition

### What problem are we solving?

In Kata 7.1, we built word embeddings --- dense vectors with 20 dimensions. We measured cosine similarity and saw that "cat" and "dog" are close while "cat" and "king" are far apart. But numbers alone do not build intuition. **We need to see the embedding space** to understand what these vectors actually look like.

Visualization answers critical questions: Do semantically related words actually cluster together? Are there visible groupings by category (animals, royalty, objects)? How does the structure of the embedding space reflect the structure of meaning in the corpus?

The problem is that our embeddings live in 20 dimensions, and we can only see 2. We need **dimensionality reduction** --- a way to project high-dimensional vectors onto a 2D plane while preserving the relative distances as much as possible.

### Why earlier methods fail

In Phase 2 (Kata 2.2), we visualized BoW vectors using bar charts --- one bar per vocabulary word. That worked because each dimension had a clear meaning (a specific word's count). But embedding dimensions have **no human-interpretable meaning**. Dimension 7 of an embedding is not "the cat dimension" or "the royalty dimension." It is a learned, abstract feature.

Bar charts of embedding values are meaningless. Seeing that "cat" has value 0.23 in dimension 3 and "dog" has value 0.19 in dimension 3 tells you nothing. What matters is the **relative positions** of words in the space --- which words are near each other and which are far apart. For that, we need scatter plots, not bar charts.

### Mental models

- **City map analogy**: Imagine embedding space as a city. Similar words live in the same neighborhood. Animals are in one district, royalty in another, furniture in a third. A 2D projection is like a flattened tourist map --- some distances get distorted, but the neighborhoods remain visible.
- **Shadow projection**: Imagine 20 flashlights shining on 20-dimensional word-objects from different angles. Each angle produces a different 2D shadow. Some shadows preserve the clustering structure well, others do not. PCA picks the angle that preserves the most variance --- the most informative shadow.
- **Lossy compression**: Going from 20D to 2D throws away information. Two words that are far apart in 20D might overlap in 2D, just like two buildings on opposite sides of a city can appear at the same point on a badly-angled photograph. The goal is to minimize this distortion.

### Visual explanations

```
FROM HIGH-DIMENSIONAL TO 2D
==============================

20D embedding space (cannot visualize directly):
  "cat"    = [0.23, -0.11, 0.85, 0.03, ..., 0.42]  (20 numbers)
  "dog"    = [0.19, -0.15, 0.78, 0.01, ..., 0.35]  (20 numbers)
  "king"   = [-0.41, 0.62, -0.08, 0.77, ..., -0.15] (20 numbers)

2D projection (can visualize):
  "cat"    = (2.3, 1.1)
  "dog"    = (2.1, 0.9)
  "king"   = (-1.8, -0.5)


WHAT A GOOD EMBEDDING VISUALIZATION LOOKS LIKE
================================================

        ^
        |          king *
        |        queen *
        |
        |    man *
        |  woman *
        |
   -----+----------------------------->
        |
        |             cat *
        |           dog *  kitten *
        |
        |      mat *
        |    rug *
        |

  Animals cluster together (bottom right).
  Royalty clusters together (top right).
  Objects cluster together (bottom left).
  People cluster in between.


WHAT A BAD PROJECTION LOOKS LIKE
==================================

        ^
        |    king *  cat *
        |         dog *
        |  queen *       mat *
        |      man *  rug *
        |    woman *
   -----+----------------------------->

  All words mixed together, no visible structure.
  This happens when you pick bad projection dimensions
  or when the embeddings themselves are poor quality.
```

---

## Hands-on Exploration

1. Run the code below and examine the ASCII scatter plot. Identify which words cluster together. Then mentally draw circles around each cluster and label them (e.g., "animals", "royalty", "objects"). Does the clustering match your intuition about word meaning? Are there any words placed in surprising positions?

2. The code uses the first two principal components for projection. Modify it to use dimensions 0 and 1, then dimensions 2 and 3, then dimensions 5 and 10 of the raw embedding vectors. How does the clustering quality change? This demonstrates that **not all dimensions carry equal information** --- PCA finds the most informative combination.

3. Add 5 new sentences to the corpus about a new topic (e.g., food: "the chef cooked a meal", "the cook prepared fresh soup"). Rebuild the embeddings and visualize again. Does a new cluster appear for food-related words? How many sentences do you need before the cluster becomes visible? This reveals the **minimum corpus size** needed for meaningful embeddings.

---

## Live Code

```python
# --- Visualize Word Embedding Spaces (Pure Python) ---
# Build embeddings from co-occurrence and project to 2D for ASCII plotting.
# No numpy, no matplotlib --- everything from scratch.

import math
import random
from collections import Counter

random.seed(42)


# ============================================================
# VECTOR MATH HELPERS
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


def vector_subtract(v1, v2):
    """Element-wise subtraction."""
    return [a - b for a, b in zip(v1, v2)]


def scalar_multiply(s, v):
    """Multiply vector by scalar."""
    return [s * x for x in v]


# ============================================================
# CORPUS AND EMBEDDING CONSTRUCTION
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
    "the man and the woman walked together",
    "the boy and the girl played together",
]


def tokenize(text):
    """Extract lowercase alphabetic words."""
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


all_tokens = []
for sentence in corpus:
    all_tokens.extend(tokenize(sentence))

vocab = sorted(set(all_tokens))
word_to_idx = {w: i for i, w in enumerate(vocab)}
vocab_size = len(vocab)

# Build co-occurrence matrix
WINDOW_SIZE = 2
cooccurrence = {w: Counter() for w in vocab}

for sentence in corpus:
    tokens = tokenize(sentence)
    for i, target in enumerate(tokens):
        start = max(0, i - WINDOW_SIZE)
        end = min(len(tokens), i + WINDOW_SIZE + 1)
        for j in range(start, end):
            if i != j:
                cooccurrence[target][tokens[j]] += 1

# Random projection to get embeddings
EMBED_DIM = 20
random.seed(42)
projection = [[random.choice([-1, 1]) for _ in range(EMBED_DIM)]
              for _ in range(vocab_size)]

embeddings = {}
for word in vocab:
    full_vec = [cooccurrence[word].get(w, 0) for w in vocab]
    embed = [sum(full_vec[i] * projection[i][d] for i in range(vocab_size))
             for d in range(EMBED_DIM)]
    mag = magnitude(embed)
    if mag > 0:
        embed = [x / mag for x in embed]
    embeddings[word] = embed

print("=" * 60)
print("WORD EMBEDDINGS BUILT")
print("=" * 60)
print(f"  Corpus: {len(corpus)} sentences, {vocab_size} unique words")
print(f"  Embedding dimensions: {EMBED_DIM}")
print()

# ============================================================
# SIMPLE PCA-LIKE PROJECTION TO 2D
# ============================================================
# We implement a basic power iteration method to find the
# top 2 principal directions of the embedding matrix.

# Step 1: Compute mean embedding
focus_words = ["cat", "dog", "mouse", "ball",
               "king", "queen", "crown", "kingdom",
               "man", "woman", "boy", "girl",
               "mat", "rug", "room", "hall",
               "park", "garden", "yard", "house"]
focus_words = [w for w in focus_words if w in embeddings]

mean_vec = [0.0] * EMBED_DIM
for w in focus_words:
    for d in range(EMBED_DIM):
        mean_vec[d] += embeddings[w][d]
mean_vec = [x / len(focus_words) for x in mean_vec]

# Step 2: Center the embeddings
centered = {}
for w in focus_words:
    centered[w] = vector_subtract(embeddings[w], mean_vec)


# Step 3: Power iteration to find principal components
def power_iteration(data_vecs, num_iterations=100):
    """Find the dominant direction via power iteration."""
    dim = len(data_vecs[0])
    # Random initial vector
    v = [random.gauss(0, 1) for _ in range(dim)]
    mag_v = magnitude(v)
    v = [x / mag_v for x in v]

    for _ in range(num_iterations):
        # Compute A^T A v where A is the data matrix
        new_v = [0.0] * dim
        for vec in data_vecs:
            proj = dot_product(vec, v)
            for d in range(dim):
                new_v[d] += proj * vec[d]
        mag_new = magnitude(new_v)
        if mag_new > 0:
            v = [x / mag_new for x in new_v]
    return v


data_list = [centered[w] for w in focus_words]

# First principal component
pc1 = power_iteration(data_list, num_iterations=200)

# Remove PC1 component from data to find PC2
deflated = []
for vec in data_list:
    proj = dot_product(vec, pc1)
    deflated.append([vec[d] - proj * pc1[d] for d in range(EMBED_DIM)])

pc2 = power_iteration(deflated, num_iterations=200)

# Project each word onto PC1 and PC2
coords_2d = {}
for w in focus_words:
    x = dot_product(centered[w], pc1)
    y = dot_product(centered[w], pc2)
    coords_2d[w] = (x, y)

print("=" * 60)
print("2D PROJECTIONS (PCA)")
print("=" * 60)
for w in focus_words:
    x, y = coords_2d[w]
    print(f"  {w:>10} -> ({x:+.4f}, {y:+.4f})")
print()

# ============================================================
# ASCII SCATTER PLOT
# ============================================================

# Define semantic categories for coloring
categories = {
    "animals": ["cat", "dog", "mouse"],
    "royalty": ["king", "queen", "crown", "kingdom"],
    "people":  ["man", "woman", "boy", "girl"],
    "objects": ["mat", "rug", "ball"],
    "places":  ["room", "hall", "park", "garden", "yard", "house"],
}

# Category symbol mapping
cat_symbols = {
    "animals": "A",
    "royalty": "R",
    "people":  "P",
    "objects": "O",
    "places":  "L",  # L for location
}

word_category = {}
for cat, words in categories.items():
    for w in words:
        if w in coords_2d:
            word_category[w] = cat

PLOT_WIDTH = 60
PLOT_HEIGHT = 30

# Find bounds
xs = [coords_2d[w][0] for w in focus_words]
ys = [coords_2d[w][1] for w in focus_words]
x_min, x_max = min(xs), max(xs)
y_min, y_max = min(ys), max(ys)

# Add padding
x_pad = (x_max - x_min) * 0.1 + 0.001
y_pad = (y_max - y_min) * 0.1 + 0.001
x_min -= x_pad
x_max += x_pad
y_min -= y_pad
y_max += y_pad


def to_grid(x, y):
    """Convert coordinates to grid position."""
    gx = int((x - x_min) / (x_max - x_min) * (PLOT_WIDTH - 1))
    gy = int((y - y_min) / (y_max - y_min) * (PLOT_HEIGHT - 1))
    gx = max(0, min(PLOT_WIDTH - 1, gx))
    gy = max(0, min(PLOT_HEIGHT - 1, gy))
    return gx, gy


print("=" * 60)
print("ASCII SCATTER PLOT OF EMBEDDING SPACE")
print("=" * 60)
print()
print("  Legend: A=animal  R=royalty  P=people  O=object  L=place")
print()

# Build grid
grid = [[" " for _ in range(PLOT_WIDTH)] for _ in range(PLOT_HEIGHT)]

# Place words on grid (store for label printing)
placed = {}
for w in focus_words:
    gx, gy = to_grid(coords_2d[w][0], coords_2d[w][1])
    cat = word_category.get(w, "?")
    symbol = cat_symbols.get(cat, "?")
    # Flip y so higher values are at top
    gy_flipped = PLOT_HEIGHT - 1 - gy
    key = (gx, gy_flipped)
    if key not in placed:
        grid[gy_flipped][gx] = symbol
        placed[key] = [w]
    else:
        placed[key].append(w)

# Print grid with border
print("  " + "-" * (PLOT_WIDTH + 2))
for row in grid:
    print("  |" + "".join(row) + "|")
print("  " + "-" * (PLOT_WIDTH + 2))
print()

# Print word positions with labels
print("  Word positions:")
for w in focus_words:
    x, y = coords_2d[w]
    cat = word_category.get(w, "?")
    symbol = cat_symbols.get(cat, "?")
    print(f"    [{symbol}] {w:>10}  ({x:+.3f}, {y:+.3f})  ({cat})")
print()

# ============================================================
# MATPLOTLIB SCATTER PLOT
# ============================================================
# Rich scatter plot (renders inline when matplotlib is available)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cat_colors = {
        "animals": "#e74c3c",   # red
        "royalty": "#8e44ad",   # purple
        "people":  "#2980b9",   # blue
        "objects": "#27ae60",   # green
        "places":  "#f39c12",   # orange
    }
    cat_markers = {
        "animals": "o",
        "royalty": "s",
        "people":  "^",
        "objects": "D",
        "places":  "p",
    }

    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot each category as a separate series for legend entries
    for cat_name, cat_word_list in categories.items():
        present = [w for w in cat_word_list if w in coords_2d]
        if not present:
            continue
        cat_xs = [coords_2d[w][0] for w in present]
        cat_ys = [coords_2d[w][1] for w in present]
        ax.scatter(
            cat_xs, cat_ys,
            c=cat_colors.get(cat_name, "#888888"),
            marker=cat_markers.get(cat_name, "o"),
            s=100, label=cat_name.capitalize(),
            edgecolors="white", linewidths=0.5, zorder=3,
        )

    # Annotate each point with its word label
    for w in focus_words:
        x, y = coords_2d[w]
        ax.annotate(
            w, (x, y),
            textcoords="offset points", xytext=(6, 6),
            fontsize=9, fontweight="bold",
            color="#333333",
        )

    ax.set_title("Word Embedding Space (2D Projection)", fontsize=14)
    ax.set_xlabel("Dimension 1 (PC1)")
    ax.set_ylabel("Dimension 2 (PC2)")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()  # Automatically renders as base64 image
except ImportError:
    pass  # matplotlib not available

# ============================================================
# CLUSTER ANALYSIS
# ============================================================

print("=" * 60)
print("CLUSTER ANALYSIS")
print("=" * 60)
print()

for cat_name, cat_words in categories.items():
    present = [w for w in cat_words if w in coords_2d]
    if len(present) < 2:
        continue

    # Compute average intra-cluster distance
    total_sim = 0.0
    count = 0
    for i in range(len(present)):
        for j in range(i + 1, len(present)):
            sim = cosine_similarity(embeddings[present[i]],
                                    embeddings[present[j]])
            total_sim += sim
            count += 1
    avg_sim = total_sim / count if count > 0 else 0

    # Compute centroid in 2D
    cx = sum(coords_2d[w][0] for w in present) / len(present)
    cy = sum(coords_2d[w][1] for w in present) / len(present)

    words_str = ", ".join(present)
    print(f"  {cat_name.upper():>10}: {words_str}")
    print(f"             Avg cosine similarity: {avg_sim:.4f}")
    print(f"             2D centroid: ({cx:+.3f}, {cy:+.3f})")
    print()

# ============================================================
# INTER-CLUSTER VS INTRA-CLUSTER DISTANCES
# ============================================================

print("=" * 60)
print("INTRA-CLUSTER vs INTER-CLUSTER SIMILARITY")
print("=" * 60)
print()
print("  Good embeddings: intra-cluster similarity >> inter-cluster")
print()

cat_names = [c for c in categories if len([w for w in categories[c]
             if w in embeddings]) >= 2]

# Intra-cluster: average similarity within each category
print("  INTRA-CLUSTER (within category):")
intra_scores = {}
for cat_name in cat_names:
    present = [w for w in categories[cat_name] if w in embeddings]
    total_sim = 0.0
    count = 0
    for i in range(len(present)):
        for j in range(i + 1, len(present)):
            sim = cosine_similarity(embeddings[present[i]],
                                    embeddings[present[j]])
            total_sim += sim
            count += 1
    avg = total_sim / count if count > 0 else 0
    intra_scores[cat_name] = avg
    bar = "#" * int(max(0, avg) * 30)
    print(f"    {cat_name:>10}: {avg:+.4f}  {bar}")

print()

# Inter-cluster: average similarity between different categories
print("  INTER-CLUSTER (between categories):")
for i in range(len(cat_names)):
    for j in range(i + 1, len(cat_names)):
        c1, c2 = cat_names[i], cat_names[j]
        words1 = [w for w in categories[c1] if w in embeddings]
        words2 = [w for w in categories[c2] if w in embeddings]
        total_sim = 0.0
        count = 0
        for w1 in words1:
            for w2 in words2:
                sim = cosine_similarity(embeddings[w1], embeddings[w2])
                total_sim += sim
                count += 1
        avg = total_sim / count if count > 0 else 0
        bar = "#" * int(max(0, avg) * 30)
        print(f"    {c1:>10} <-> {c2:<10}: {avg:+.4f}  {bar}")

print()
print("  If intra-cluster scores are consistently higher than")
print("  inter-cluster scores, the embedding space has meaningful")
print("  structure --- words are organized by semantic category.")

# ============================================================
# CORPUS QUALITY EXPERIMENT
# ============================================================

print()
print("=" * 60)
print("EMBEDDING QUALITY DEPENDS ON CORPUS")
print("=" * 60)
print()
print("  Key insight: embeddings are only as good as the corpus")
print("  they are built from.")
print()
print("  Our corpus has:")

word_freq = Counter(all_tokens)
for w in ["cat", "dog", "king", "queen", "mat", "rug"]:
    if w in word_freq:
        co_count = sum(cooccurrence[w].values())
        print(f"    '{w}': {word_freq[w]} occurrences, "
              f"{co_count} co-occurrence events")

print()
print("  With a larger corpus:")
print("    - More co-occurrence data -> more accurate vectors")
print("    - More diverse contexts -> richer representations")
print("    - Rare words get enough data to be placed correctly")
print()
print("  With our small corpus:")
print("    - Common words (the, on, in) dominate co-occurrence")
print("    - Rare words may be placed poorly")
print("    - Analogies may not work reliably")
print()
print("  This is why real embedding systems (Word2Vec, GloVe)")
print("  train on billions of words, not dozens of sentences.")
```

---

## Key Takeaways

- **Visualization reveals structure that numbers hide.** A table of cosine similarities tells you two words are close, but a scatter plot shows you the entire landscape --- clusters, outliers, and the overall geometry of the embedding space. Seeing is understanding.
- **Dimensionality reduction is lossy but essential.** Projecting 20 dimensions to 2 throws away information. Some close words may appear far apart, and some distant words may overlap. The goal is to preserve the most important structure, not all of it.
- **Semantic categories form visible clusters.** When embeddings are working well, words from the same category (animals, royalty, places) group together in the projected space. This clustering is not programmed --- it emerges from co-occurrence patterns in the corpus.
- **Embedding quality is bounded by corpus quality.** A small, narrow corpus produces noisy embeddings with weak clustering. A large, diverse corpus produces clean embeddings with sharp category boundaries. There is no substitute for sufficient data.
