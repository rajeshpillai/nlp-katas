# Build Bag of Words from Scratch

> Phase 2 — Bag of Words (BoW) | Kata 2.1

---

## Concept & Intuition

### What problem are we solving?

Computers cannot process words directly. They need numbers. The Bag of Words model is the **first practical method** for converting text into numerical vectors that a machine can work with. Given a collection of documents, BoW answers: "Which words appear in each document, and how many times?"

This is the bridge from raw text to computation. Without it, you cannot compute similarity, classify documents, or search a corpus. Every technique that comes later — TF-IDF, embeddings, transformers — is solving the same fundamental problem: **how do we represent text as numbers?** BoW is the simplest, most transparent answer.

### Why naive/earlier approaches fail

Before BoW, you might try:

- **Exact string matching**: "the cat sat" vs "the cat sat on the mat" — these are different strings, but clearly related. String equality tells you nothing about partial overlap.
- **Character counting**: Counting characters gives you `{'t': 3, 'h': 2, ...}` — this destroys word-level meaning entirely. "dog bites man" and "man bites dog" would have identical character counts.
- **Fixed keyword lists**: You could check for specific words, but this requires knowing what to look for in advance and does not scale.

BoW solves this by building a **vocabulary from the data itself** and representing each document as a vector of word counts over that vocabulary.

### Mental models

- **The ballot box**: Imagine shredding each document into individual words and dropping them into labeled boxes (one box per unique word). The count in each box is the BoW vector.
- **The document-term matrix**: Rows are documents, columns are vocabulary words, and each cell holds a count. This is the fundamental data structure of classical text analysis.
- **Lossy compression**: BoW deliberately throws away word order. "Dog bites man" and "Man bites dog" produce identical vectors. This is a **design tradeoff**, not a bug.

### Visual explanations

```
Corpus:
  Doc 1: "the cat sat"
  Doc 2: "the cat sat on the mat"
  Doc 3: "the dog sat"

Step 1 — Build vocabulary (all unique words):
  ["cat", "dog", "mat", "on", "sat", "the"]

Step 2 — Count words per document:

              cat  dog  mat  on  sat  the
  Doc 1:  [    1    0    0    0    1    1  ]
  Doc 2:  [    1    0    1    1    1    2  ]
  Doc 3:  [    0    1    0    0    1    1  ]

This is the document-term matrix.

Step 3 — Notice sparsity:
  Doc 1 has 3 zeros out of 6 entries = 50% sparse
  Doc 3 has 3 zeros out of 6 entries = 50% sparse
  With real corpora (thousands of words), sparsity is typically 95-99%
```

---

## Hands-on Exploration

1. Take 3 short sentences about different topics. Build the vocabulary by hand (list every unique word). Then fill in the document-term matrix on paper. Notice how many cells are zero — this is sparsity, and it grows rapidly with vocabulary size.

2. Write two sentences that have **completely different meanings** but produce the **same BoW vector** (hint: rearrange the same words). This demonstrates the fundamental limitation of BoW: it discards word order entirely.

3. Add a fourth document to the corpus that shares no words with the others. Observe what happens to the matrix — an entire row of zeros (except for new columns). Think about what this means for comparing documents that use different vocabulary to express similar ideas.

---

## Live Code

```python
# --- Build a Bag of Words model from scratch ---
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


def build_document_term_matrix(corpus, vocabulary):
    """Build the full document-term matrix."""
    return [document_to_bow(doc, vocabulary) for doc in corpus]


# --- Our corpus ---
corpus = [
    "The cat sat on the mat",
    "The dog sat on the log",
    "The cat chased the dog around the yard",
    "A bird flew over the mat",
]

print("=== Corpus ===")
for i, doc in enumerate(corpus):
    print(f"  Doc {i + 1}: \"{doc}\"")
print()

# Step 1: Build vocabulary
vocabulary = build_vocabulary(corpus)
print(f"=== Vocabulary ({len(vocabulary)} words) ===")
print(f"  {vocabulary}")
print()

# Step 2: Build document-term matrix
matrix = build_document_term_matrix(corpus, vocabulary)

# Step 3: Display the matrix
print("=== Document-Term Matrix ===")
header = "         " + "".join(f"{w:>8}" for w in vocabulary)
print(header)
print("         " + "-" * (8 * len(vocabulary)))
for i, row in enumerate(matrix):
    row_str = "".join(f"{count:>8}" for count in row)
    print(f"  Doc {i + 1}: {row_str}")
print()

# Step 4: Analyze sparsity
print("=== Sparsity Analysis ===")
total_cells = 0
zero_cells = 0
for i, row in enumerate(matrix):
    zeros = row.count(0)
    total = len(row)
    total_cells += total
    zero_cells += zeros
    pct = (zeros / total) * 100
    print(f"  Doc {i + 1}: {zeros}/{total} entries are zero ({pct:.0f}% sparse)")

overall_sparsity = (zero_cells / total_cells) * 100
print(f"\n  Overall matrix sparsity: {zero_cells}/{total_cells} ({overall_sparsity:.0f}%)")
print()

# Step 5: Show the order-loss problem
print("=== BoW Loses Word Order ===")
sentence_a = "dog bites man"
sentence_b = "man bites dog"
vocab_ab = build_vocabulary([sentence_a, sentence_b])
vec_a = document_to_bow(sentence_a, vocab_ab)
vec_b = document_to_bow(sentence_b, vocab_ab)
print(f"  Vocabulary: {vocab_ab}")
print(f"  \"{sentence_a}\" -> {vec_a}")
print(f"  \"{sentence_b}\" -> {vec_b}")
print(f"  Vectors identical? {vec_a == vec_b}")
print(f"  Meanings identical? No! This is the fundamental BoW tradeoff.")
print()

# Step 6: Scaling intuition
print("=== Scaling Intuition ===")
print(f"  Corpus size: {len(corpus)} documents")
print(f"  Vocabulary size: {len(vocabulary)} words")
print(f"  Matrix dimensions: {len(corpus)} x {len(vocabulary)} = {len(corpus) * len(vocabulary)} cells")
print()
print("  In real-world corpora:")
print("    10,000 documents, 50,000 unique words = 500,000,000 cells")
print("    Most of those cells are 0. This is why sparsity matters.")
print("    Sparse representations save memory and computation.")
```

---

## Key Takeaways

- **BoW converts text into fixed-length numerical vectors** by counting word occurrences against a shared vocabulary. This is the foundation of all classical text analysis.
- **The document-term matrix is inherently sparse.** Each document uses only a tiny fraction of the total vocabulary, so most entries are zero. Real corpora are 95-99% sparse.
- **BoW deliberately discards word order.** "Dog bites man" and "Man bites dog" produce identical vectors. This is a known tradeoff: you gain simplicity and computational tractability at the cost of sequential meaning.
- **Vocabulary construction is a design choice.** What you include in the vocabulary (lowercasing, punctuation handling, stopword removal) directly shapes the representation. BoW does not "discover" meaning — it reflects the preprocessing decisions you make.
