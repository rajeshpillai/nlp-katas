# Compute Cosine Similarity Between Document Pairs

> Phase 4 — Similarity & Classical NLP Tasks | Kata 4.1

---

## Concept & Intuition

### What problem are we solving?

We have documents represented as numerical vectors (from BoW or TF-IDF). Now we need to answer a fundamental question: **how similar are two documents?** This is not a philosophical question — it is a geometric one. Once text becomes vectors, similarity becomes a measurement in space.

Cosine similarity measures the **angle** between two vectors, ignoring their magnitude. This is exactly what we want: a short document about "machine learning" and a long document about "machine learning" should be considered similar, even though the long one has much larger raw counts.

### Why naive/earlier approaches fail

**Raw word overlap** (counting shared words) fails because it does not account for word importance or document length. Two documents sharing the word "the" 50 times are not meaningfully similar.

**Euclidean distance** (straight-line distance between vector endpoints) fails because it is sensitive to magnitude. A document with word counts [2, 4, 6] and another with [20, 40, 60] are pointing in the exact same direction (same proportions of words), but Euclidean distance says they are far apart. They represent the same "topic mixture" at different scales.

**Jaccard similarity** (set overlap) fails because it treats every word as equally important and ignores frequency entirely. A word appearing once and a word appearing 100 times contribute equally.

### Mental models

- **Cosine similarity** = "How much do these vectors point in the same direction?" Range: -1 to 1 (for non-negative vectors like word counts: 0 to 1). Value of 1 means identical direction, 0 means completely orthogonal (no shared terms).
- **Euclidean distance** = "How far apart are the tips of these vectors?" Sensitive to scale. Good for same-length documents, misleading otherwise.
- **Jaccard similarity** = "What fraction of the combined vocabulary do they share?" Ignores frequency, only cares about presence/absence.

### Visual explanations

```
Cosine similarity measures the ANGLE, not the distance:

                        B (long doc)
                       /
                      /
                     /  angle = 0 degrees
                    /   cosine = 1.0 (identical direction!)
                   /
                  / B' (short doc, same topic)
                 /
                /
  Origin ------/----------------------------> A

  B and B' point in the SAME direction.
  Cosine similarity = 1.0 for both pairs (A,B) if they align.
  Euclidean distance between A and B is LARGE.
  Euclidean distance between A and B' is SMALL.
  But cosine treats them the same -- direction matters, not length.


Comparing the three metrics on two documents:

  Doc1 = "cat sat mat"       -> vector [1, 1, 1, 0, 0]
  Doc2 = "cat sat hat"       -> vector [1, 1, 0, 1, 0]
  Doc3 = "dog ran fast"      -> vector [0, 0, 0, 0, 0, 1, 1, 1]

  Cosine(Doc1, Doc2) = high   (share 2 of 3 terms, similar direction)
  Cosine(Doc1, Doc3) = 0.0    (no shared terms, orthogonal)

  Jaccard(Doc1, Doc2) = 2/4 = 0.50  (2 shared out of 4 unique words)
  Jaccard(Doc1, Doc3) = 0/6 = 0.00  (no shared words)

  Euclidean(Doc1, Doc2) = sqrt(2)   (small distance)
  Euclidean(Doc1, Doc3) = sqrt(6)   (larger distance)


The formula:

                    A . B           sum(a_i * b_i)
  cosine(A,B) = ----------- = -------------------------
                |A| * |B|     sqrt(sum(a_i^2)) * sqrt(sum(b_i^2))

  Numerator:   dot product (how much they overlap)
  Denominator: product of magnitudes (normalizes for length)
```

---

## Hands-on Exploration

1. Take two sentences: "the cat sat on the mat" and "the cat chased the rat". By hand, build their word-count vectors over the combined vocabulary, then compute cosine similarity using the formula. Verify your answer matches the code output.

2. Consider three documents of very different lengths: a tweet (10 words), a paragraph (100 words), and an essay (1000 words), all about the same topic. Predict: which similarity metric (cosine, Euclidean, Jaccard) will correctly identify them as similar? Run the code and check whether Euclidean distance is misleadingly large for the long document pair.

3. Create two documents that have high Jaccard similarity but low cosine similarity. Hint: think about documents that share many unique words but in very different proportions. Then create two documents with high cosine similarity but low Jaccard similarity. What does this tell you about when each metric is appropriate?

---

## Live Code

```python
# --- Cosine Similarity, Euclidean Distance, and Jaccard Similarity ---
# All implemented from scratch using only Python stdlib

import math
from collections import Counter


def tokenize(text):
    """Lowercase and split on non-alpha characters."""
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


def build_vectors(doc1_tokens, doc2_tokens):
    """Build aligned count vectors over the shared vocabulary."""
    vocab = sorted(set(doc1_tokens) | set(doc2_tokens))
    c1 = Counter(doc1_tokens)
    c2 = Counter(doc2_tokens)
    vec1 = [c1.get(w, 0) for w in vocab]
    vec2 = [c2.get(w, 0) for w in vocab]
    return vocab, vec1, vec2


def dot_product(v1, v2):
    """Compute dot product of two vectors."""
    return sum(a * b for a, b in zip(v1, v2))


def magnitude(v):
    """Compute the magnitude (L2 norm) of a vector."""
    return math.sqrt(sum(x * x for x in v))


def cosine_similarity(v1, v2):
    """Cosine similarity: dot(A,B) / (|A| * |B|)."""
    d = dot_product(v1, v2)
    m1 = magnitude(v1)
    m2 = magnitude(v2)
    if m1 == 0 or m2 == 0:
        return 0.0
    return d / (m1 * m2)


def euclidean_distance(v1, v2):
    """Euclidean distance: sqrt(sum((a-b)^2))."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))


def jaccard_similarity(tokens1, tokens2):
    """Jaccard similarity: |intersection| / |union| of token sets."""
    s1 = set(tokens1)
    s2 = set(tokens2)
    if not s1 and not s2:
        return 0.0
    intersection = s1 & s2
    union = s1 | s2
    return len(intersection) / len(union)


# --- Corpus ---
documents = {
    "Doc A": "The cat sat on the mat and watched the birds outside",
    "Doc B": "A cat was sitting on a mat looking at birds through the window",
    "Doc C": "Stock markets rallied today as investors grew confident",
    "Doc D": "The financial markets saw gains as stocks rose sharply",
    "Doc E": "The cat sat on the mat the cat sat on the mat",  # repeated
}

print("=" * 60)
print("DOCUMENTS")
print("=" * 60)
for name, text in documents.items():
    print(f"  {name}: {text}")
print()

# Tokenize all documents
tokenized = {name: tokenize(text) for name, text in documents.items()}

# --- Compare all pairs ---
names = list(documents.keys())

print("=" * 60)
print("PAIRWISE SIMILARITY COMPARISON")
print("=" * 60)
print(f"{'Pair':<12} {'Cosine':>8} {'Euclid':>8} {'Jaccard':>8}")
print("-" * 40)

for i in range(len(names)):
    for j in range(i + 1, len(names)):
        n1, n2 = names[i], names[j]
        t1, t2 = tokenized[n1], tokenized[n2]
        vocab, v1, v2 = build_vectors(t1, t2)

        cos = cosine_similarity(v1, v2)
        euc = euclidean_distance(v1, v2)
        jac = jaccard_similarity(t1, t2)

        label = f"{n1}-{n2}"
        print(f"{label:<12} {cos:>8.3f} {euc:>8.3f} {jac:>8.3f}")

print()

# Build cosine similarity matrix for heatmap
doc_names = list(documents.keys())
n_docs = len(doc_names)
cos_matrix = []
for i in range(n_docs):
    row = []
    for j in range(n_docs):
        if i == j:
            row.append(1.0)
        else:
            t1, t2 = tokenized[doc_names[i]], tokenized[doc_names[j]]
            _, v1, v2 = build_vectors(t1, t2)
            row.append(round(cosine_similarity(v1, v2), 3))
    cos_matrix.append(row)

show_chart({
    "type": "heatmap",
    "title": "Cosine Similarity Matrix (All Document Pairs)",
    "x_labels": doc_names,
    "y_labels": doc_names,
    "data": cos_matrix,
    "color_scale": "blue"
})

# --- Deep dive: Doc A vs Doc B ---
print("=" * 60)
print("DEEP DIVE: Doc A vs Doc B")
print("=" * 60)
vocab_ab, va, vb = build_vectors(tokenized["Doc A"], tokenized["Doc B"])
print(f"Vocabulary: {vocab_ab}")
print(f"Vec A:      {va}")
print(f"Vec B:      {vb}")
print(f"Dot product:    {dot_product(va, vb)}")
print(f"|A|:            {magnitude(va):.3f}")
print(f"|B|:            {magnitude(vb):.3f}")
print(f"Cosine sim:     {cosine_similarity(va, vb):.4f}")
print(f"Euclidean dist: {euclidean_distance(va, vb):.4f}")
print(f"Jaccard sim:    {jaccard_similarity(tokenized['Doc A'], tokenized['Doc B']):.4f}")
print()

# --- Magnitude sensitivity demo ---
print("=" * 60)
print("WHY COSINE BEATS EUCLIDEAN FOR DIFFERENT-LENGTH DOCS")
print("=" * 60)

short = "machine learning is powerful"
long_doc = " ".join(["machine learning is powerful"] * 10)

t_short = tokenize(short)
t_long = tokenize(long_doc)
_, v_short, v_long = build_vectors(t_short, t_long)

print(f"Short doc tokens: {len(t_short)}")
print(f"Long doc tokens:  {len(t_long)}")
print(f"Short vec: {v_short}")
print(f"Long vec:  {v_long}")
print(f"Cosine similarity:  {cosine_similarity(v_short, v_long):.4f}  (perfect!)")
print(f"Euclidean distance: {euclidean_distance(v_short, v_long):.4f}  (misleadingly large)")
print()
print("Both documents express the SAME idea.")
print("Cosine correctly gives 1.0 — same direction in vector space.")
print("Euclidean gives a large distance — fooled by magnitude difference.")
print()

# --- When Jaccard disagrees with Cosine ---
print("=" * 60)
print("WHEN JACCARD AND COSINE DISAGREE")
print("=" * 60)

# Same unique words, very different frequencies
d1 = "data data data data science"
d2 = "data science science science science"
t1 = tokenize(d1)
t2 = tokenize(d2)
_, v1, v2 = build_vectors(t1, t2)

print(f'Doc 1: "{d1}"')
print(f'Doc 2: "{d2}"')
print(f"Jaccard:  {jaccard_similarity(t1, t2):.4f}  (perfect — same word set!)")
print(f"Cosine:   {cosine_similarity(v1, v2):.4f}  (lower — different proportions)")
print()
print("Jaccard only sees presence/absence.")
print("Cosine sees that Doc1 emphasizes 'data' while Doc2 emphasizes 'science'.")
```

---

## Key Takeaways

- **Cosine similarity measures direction, not magnitude.** This makes it ideal for comparing documents of different lengths, because a long and short document about the same topic point in the same direction.
- **Euclidean distance is sensitive to scale.** Two documents with identical word proportions but different lengths will appear far apart, making it unreliable for raw term-count vectors.
- **Jaccard similarity ignores frequency entirely.** It only cares whether a word is present or absent, which loses valuable information about how much a document emphasizes certain terms.
- **Choosing the right similarity metric is a modeling decision.** Cosine similarity is the standard for text because NLP cares about what a document is *about* (direction), not how long it is (magnitude). Many NLP problems reduce to similarity in vector space.
