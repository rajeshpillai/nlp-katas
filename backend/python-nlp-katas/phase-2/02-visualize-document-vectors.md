# Visualize Document Vectors

> Phase 2 — Bag of Words (BoW) | Kata 2.2

---

## Concept & Intuition

### What problem are we solving?

A BoW vector is a list of numbers. But what does it **look like**? When we say "documents are points in a high-dimensional space," that sounds abstract. This kata makes it concrete. By visualizing document vectors — even in ASCII — we build intuition for what "similarity in vector space" actually means.

When two documents share many of the same words, their vectors point in similar directions. When they share nothing, their vectors are orthogonal. Visualization makes this tangible before we ever compute a similarity metric.

### Why naive/earlier approaches fail

Looking at raw BoW vectors — long lists of numbers — tells you almost nothing at a glance:

- `[0, 1, 0, 0, 2, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]` — Is this similar to another document? Which words matter? How sparse is it?

Without visualization, you cannot develop intuition about the geometry of your text representations. You need a way to **see** the structure: which dimensions are active, which are zero, and how documents relate to each other.

### Mental models

- **The bookshelf**: Each word in the vocabulary is a shelf slot. Each document places books (word counts) on some shelves and leaves the rest empty. Similar documents fill similar shelves.
- **The fingerprint**: A BoW vector is a document's fingerprint — a pattern of peaks (frequent words) and flat zones (absent words). Comparing fingerprints reveals kinship.
- **Sparse skyline**: If you plot a BoW vector as a bar chart, most bars are height zero. The skyline is mostly flat with a few spikes. This sparsity pattern is the defining visual signature of BoW.

### Visual explanations

```
Document: "the cat sat on the mat"
Vocabulary: [cat, dog, mat, on, sat, the]

BoW vector: [1, 0, 1, 1, 1, 2]

ASCII bar chart:

  cat  |##          (1)
  dog  |            (0)
  mat  |##          (1)
  on   |##          (1)
  sat  |##          (1)
  the  |####        (2)

Compare with: "the dog ran"
BoW vector: [0, 1, 0, 0, 0, 1]

  cat  |            (0)
  dog  |##          (1)
  mat  |            (0)
  on   |            (0)
  sat  |            (0)
  the  |##          (1)

Notice: only "the" overlaps. The bar charts look very different.
The sparse regions (zeros) dominate both vectors.
```

---

## Hands-on Exploration

1. Pick three documents: two about the same topic and one about something completely different. Build their BoW vectors and draw the bar charts by hand. Notice how the two related documents share active dimensions while the unrelated one has peaks in different positions.

2. Count the zero entries in each vector and compute the sparsity percentage. Now imagine the vocabulary has 10,000 words and each document uses about 20 unique words. What would the sparsity be? (Answer: 99.8%). Think about what this means for storage and computation.

3. Look at the bar charts from exercise 1. Without computing any formula, use the visual overlap of the bars to guess which two documents are most similar. This visual intuition is exactly what cosine similarity formalizes (coming in Kata 2.3).

---

## Live Code

```python
# --- Visualize BoW document vectors with ASCII ---
# No external libraries — pure Python

import re
from collections import Counter


def tokenize(text):
    """Split text into lowercase words, removing punctuation."""
    return re.findall(r'[a-z]+', text.lower())


def build_vocabulary(corpus):
    """Build a sorted vocabulary from all documents."""
    vocab = set()
    for doc in corpus:
        vocab.update(tokenize(doc))
    return sorted(vocab)


def document_to_bow(doc, vocabulary):
    """Convert a single document to a BoW vector."""
    word_counts = Counter(tokenize(doc))
    return [word_counts.get(word, 0) for word in vocabulary]


# --- Corpus: mix of related and unrelated topics ---
corpus = [
    "The cat sat on the mat near the window",
    "A cat napped on a soft mat",
    "The stock market crashed on Monday morning",
    "Scientists discovered a new species of deep sea fish",
]

labels = ["Cat-mat (1)", "Cat-mat (2)", "Finance", "Science"]

vocabulary = build_vocabulary(corpus)
matrix = [document_to_bow(doc, vocabulary) for doc in corpus]

print(f"=== Vocabulary ({len(vocabulary)} words) ===")
print(f"  {vocabulary}")
print()


# --- ASCII bar chart for each document ---
def print_bar_chart(label, vector, vocabulary, bar_char="#", scale=2):
    """Print a horizontal ASCII bar chart of a BoW vector."""
    print(f"--- {label} ---")
    max_word_len = max(len(w) for w in vocabulary)
    for word, count in zip(vocabulary, vector):
        bar = bar_char * (count * scale)
        if count > 0:
            print(f"  {word:>{max_word_len}}  |{bar} ({count})")
        else:
            print(f"  {word:>{max_word_len}}  |")
    print()


print("=== Document Vectors as Bar Charts ===\n")
for label, vector in zip(labels, matrix):
    print_bar_chart(label, vector, vocabulary)


# --- Sparsity analysis per document ---
print("=== Sparsity Statistics ===")
print(f"  {'Document':<16} {'Non-zero':>10} {'Zero':>10} {'Total':>10} {'Sparsity':>10}")
print(f"  {'-' * 56}")
for label, vector in zip(labels, matrix):
    nonzero = sum(1 for v in vector if v > 0)
    zero = sum(1 for v in vector if v == 0)
    total = len(vector)
    sparsity = (zero / total) * 100
    print(f"  {label:<16} {nonzero:>10} {zero:>10} {total:>10} {sparsity:>9.1f}%")
print()


# --- Sparse vector display (only non-zero entries) ---
def print_sparse_vector(label, vector, vocabulary):
    """Show only the non-zero entries of a vector."""
    active = [(w, c) for w, c in zip(vocabulary, vector) if c > 0]
    parts = [f"{w}:{c}" for w, c in active]
    print(f"  {label:<16} -> {{ {', '.join(parts)} }}")


print("=== Sparse Representation (non-zero entries only) ===")
for label, vector in zip(labels, matrix):
    print_sparse_vector(label, vector, vocabulary)
print()
print("  In real systems, storing only non-zero entries saves enormous")
print("  memory. A 50,000-dim vector with 20 active words stores 20")
print("  entries instead of 50,000.")
print()


# --- Side-by-side comparison ---
def print_side_by_side(label_a, vec_a, label_b, vec_b, vocabulary):
    """Compare two document vectors side by side."""
    print(f"  {'Word':<12} {label_a:>12} {label_b:>12}   Overlap")
    print(f"  {'-' * 50}")
    max_word_len = 12
    overlaps = 0
    for word, a, b in zip(vocabulary, vec_a, vec_b):
        marker = ""
        if a > 0 and b > 0:
            marker = "  <-- shared"
            overlaps += 1
        a_str = str(a) if a > 0 else "."
        b_str = str(b) if b > 0 else "."
        print(f"  {word:<12} {a_str:>12} {b_str:>12}{marker}")
    total = len(vocabulary)
    print(f"\n  Shared dimensions: {overlaps}/{total}")
    print()


print("=== Side-by-Side Comparison ===\n")

# Similar documents
print("Similar topics (both about cats and mats):")
print_side_by_side(labels[0], matrix[0], labels[1], matrix[1], vocabulary)

# Different documents
print("Different topics (cat vs finance):")
print_side_by_side(labels[0], matrix[0], labels[2], matrix[2], vocabulary)

# Completely different documents
print("Unrelated topics (finance vs science):")
print_side_by_side(labels[2], matrix[2], labels[3], matrix[3], vocabulary)


# --- Visual overlap summary ---
print("=== Overlap Heatmap (shared non-zero dimensions) ===")
print(f"\n  {'':>16}", end="")
for label in labels:
    print(f"  {label:>14}", end="")
print()

for i, (label_i, vec_i) in enumerate(zip(labels, matrix)):
    print(f"  {label_i:>16}", end="")
    for j, (label_j, vec_j) in enumerate(zip(labels, matrix)):
        shared = sum(1 for a, b in zip(vec_i, vec_j) if a > 0 and b > 0)
        print(f"  {shared:>14}", end="")
    print()
print()
print("  Higher numbers = more vocabulary overlap = likely more similar.")
print("  This is a rough proxy — Kata 2.3 formalizes this with cosine similarity.")
```

---

## Key Takeaways

- **BoW vectors are mostly zeros.** Even with a small corpus, sparsity dominates. In real corpora with tens of thousands of vocabulary words, a typical document uses fewer than 1% of them. Visualization makes this unmistakable.
- **Similar documents light up similar dimensions.** When two documents share topic words, their bar charts have peaks in the same positions. Different topics produce peaks in different positions with no overlap.
- **Sparse representations are essential.** Storing only the non-zero entries of a vector saves orders of magnitude in memory and makes computation practical at scale.
- **Visual intuition precedes formal metrics.** Before computing cosine similarity or dot products, you can see similarity by comparing which dimensions are active. The math formalizes what the eye already notices.
