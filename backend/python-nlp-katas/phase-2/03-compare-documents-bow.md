# Compare Documents Using Bag of Words

> Phase 2 — Bag of Words (BoW) | Kata 2.3

---

## Concept & Intuition

### What problem are we solving?

Now that we can represent documents as numerical vectors (Kata 2.1) and visualize them (Kata 2.2), we need to **measure how similar two documents are**. This is the core operation behind text search, document clustering, duplicate detection, and recommendation systems.

Given two BoW vectors, we need a number that says: "these documents are very similar" or "these documents have nothing in common." The dot product and cosine similarity give us exactly that.

### Why naive/earlier approaches fail

- **Exact string matching**: "The cat sat on the mat" and "A cat napped on a soft mat" share topic words but are different strings. Exact match says they are 0% similar.
- **Word overlap count**: Counting shared words is a start, but it does not account for document length. A 1000-word document will share more words with anything simply because it has more words. We need a **normalized** measure.
- **Euclidean distance**: Measures the straight-line distance between vector endpoints. But two documents about the same topic, one short and one long, will be far apart in Euclidean space because the longer one has larger counts. We want to measure **direction**, not magnitude.

Cosine similarity solves the length problem by comparing the **angle** between vectors, not their distance.

### Mental models

- **The flashlight analogy**: Think of each document vector as a flashlight beam pointing in some direction in word-space. Cosine similarity measures how much the beams overlap. If they point the same way (similar topics), cosine is near 1. If they point in completely different directions (different topics), cosine is near 0.
- **Dot product as overlap**: The dot product multiplies matching dimensions and sums them. If both documents use the word "cat" frequently, that dimension contributes a large value. If one uses "cat" and the other does not, that dimension contributes zero.
- **Normalization as fairness**: Dividing by vector magnitudes ensures that a short email and a long essay about the same topic get a high similarity score. Without normalization, the essay always "wins."

### Visual explanations

```
Two vectors in 3D word-space:

  Vocabulary: [cat, dog, fish]

  Doc A: "cat cat dog"   -> [2, 1, 0]
  Doc B: "cat dog dog"   -> [1, 2, 0]
  Doc C: "fish fish"     -> [0, 0, 2]

  Dot product (A, B) = (2*1) + (1*2) + (0*0) = 4    (some overlap)
  Dot product (A, C) = (2*0) + (1*0) + (0*2) = 0    (no overlap)

  Cosine similarity:

             dot(A, B)          4          4
  cos(A,B) = ----------- = ----------- = ----- = 0.80
             |A| * |B|    sqrt(5)*sqrt(5)   5

             dot(A, C)          0
  cos(A,C) = ----------- = ----------- = 0.00
             |A| * |C|    sqrt(5)*sqrt(4)

  A and B are similar (both about cats and dogs).
  A and C are completely dissimilar (no shared words).

The BoW failure case:

  Doc X: "dog bites man"  -> [1, 1, 1]  (vocab: [bites, dog, man])
  Doc Y: "man bites dog"  -> [1, 1, 1]

  cos(X, Y) = 1.0  (identical vectors!)

  But the meanings are opposite. BoW cannot see word order.
```

---

## Hands-on Exploration

1. Pick three documents: two about cooking and one about sports. Compute their BoW vectors by hand, then calculate the dot product for each pair. Verify that the cooking documents have the highest dot product. Now compute cosine similarity and confirm the same ranking.

2. Write a pair of sentences with identical words in different order that mean very different things (e.g., "the teacher failed the student" vs "the student failed the teacher"). Compute their BoW vectors and cosine similarity. The similarity will be 1.0 despite different meanings. This is the most important limitation to internalize.

3. Take one of the cooking documents from exercise 1 and double every word count (imagine the document repeated twice). Compute Euclidean distance and cosine similarity to the other cooking document. Euclidean distance increases (the vectors are further apart), but cosine similarity stays the same (the angle has not changed). This demonstrates why cosine similarity is preferred for text.

---

## Live Code

```python
# --- Compare documents using dot product and cosine similarity ---
# No external libraries — pure Python

import re
import math
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


def dot_product(vec_a, vec_b):
    """Compute dot product of two vectors."""
    return sum(a * b for a, b in zip(vec_a, vec_b))


def magnitude(vec):
    """Compute the magnitude (L2 norm) of a vector."""
    return math.sqrt(sum(x * x for x in vec))


def cosine_similarity(vec_a, vec_b):
    """Compute cosine similarity between two vectors."""
    dot = dot_product(vec_a, vec_b)
    mag_a = magnitude(vec_a)
    mag_b = magnitude(vec_b)
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def euclidean_distance(vec_a, vec_b):
    """Compute Euclidean distance between two vectors."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec_a, vec_b)))


# --- Corpus with clear topic clusters ---
corpus = [
    "The cat sat on the warm mat by the fire",
    "A small cat napped on the soft mat",
    "The stock market index fell sharply on Monday",
    "Market traders were worried about the falling index",
    "The cat watched the fish in the bowl",
]

labels = ["Cat-mat (A)", "Cat-mat (B)", "Finance (C)", "Finance (D)", "Cat-fish (E)"]

vocabulary = build_vocabulary(corpus)
vectors = [document_to_bow(doc, vocabulary) for doc in corpus]

print(f"=== Corpus ({len(corpus)} documents, {len(vocabulary)} vocab words) ===")
for label, doc in zip(labels, corpus):
    print(f"  {label}: \"{doc}\"")
print()


# --- Pairwise similarity matrix ---
print("=== Cosine Similarity Matrix ===\n")
col_width = 14
print(f"  {'':>{col_width}}", end="")
for label in labels:
    print(f"  {label:>{col_width}}", end="")
print()

for i, (label_i, vec_i) in enumerate(zip(labels, vectors)):
    print(f"  {label_i:>{col_width}}", end="")
    for j, (label_j, vec_j) in enumerate(zip(labels, vectors)):
        sim = cosine_similarity(vec_i, vec_j)
        print(f"  {sim:>{col_width}.3f}", end="")
    print()
print()


# --- Top similar pairs ---
print("=== Most and Least Similar Pairs ===\n")
pairs = []
for i in range(len(corpus)):
    for j in range(i + 1, len(corpus)):
        sim = cosine_similarity(vectors[i], vectors[j])
        pairs.append((labels[i], labels[j], sim))

pairs.sort(key=lambda x: x[2], reverse=True)

print("  Ranked by cosine similarity (high to low):")
for label_a, label_b, sim in pairs:
    bar = "#" * int(sim * 30)
    print(f"    {label_a} vs {label_b}: {sim:.3f}  |{bar}")
print()


# --- Cosine vs Euclidean: the length problem ---
print("=== Cosine vs Euclidean Distance ===\n")
print("  What happens when we double one document's word counts?\n")

doc_short = "The cat sat on the mat"
doc_long = "The cat sat on the mat " * 3  # repeated 3 times

vocab_test = build_vocabulary([doc_short, doc_long, corpus[1]])
vec_short = document_to_bow(doc_short, vocab_test)
vec_long = document_to_bow(doc_long, vocab_test)
vec_other = document_to_bow(corpus[1], vocab_test)

cos_short = cosine_similarity(vec_short, vec_other)
cos_long = cosine_similarity(vec_long, vec_other)
euc_short = euclidean_distance(vec_short, vec_other)
euc_long = euclidean_distance(vec_long, vec_other)

print(f"  Original:  \"{doc_short}\"")
print(f"  Tripled:   \"{doc_short}\" (repeated 3x)")
print(f"  Compare to: \"{corpus[1]}\"")
print()
print(f"  {'Metric':<20} {'Original':>12} {'Tripled':>12}")
print(f"  {'-' * 44}")
print(f"  {'Cosine similarity':<20} {cos_short:>12.3f} {cos_long:>12.3f}")
print(f"  {'Euclidean distance':<20} {euc_short:>12.3f} {euc_long:>12.3f}")
print()
print("  Cosine similarity stays stable despite length change.")
print("  Euclidean distance increases because the vector is longer.")
print("  This is why cosine similarity is preferred for text comparison.")
print()


# --- The fundamental BoW failure: word order ---
print("=" * 55)
print("=== WHERE BoW COMPARISON FAILS: WORD ORDER ===")
print("=" * 55)
print()

order_pairs = [
    ("dog bites man", "man bites dog"),
    ("the teacher failed the student", "the student failed the teacher"),
    ("she saw him leave", "he saw her leave"),
]

for sent_a, sent_b in order_pairs:
    vocab_pair = build_vocabulary([sent_a, sent_b])
    vec_a = document_to_bow(sent_a, vocab_pair)
    vec_b = document_to_bow(sent_b, vocab_pair)
    sim = cosine_similarity(vec_a, vec_b)

    print(f"  A: \"{sent_a}\"")
    print(f"  B: \"{sent_b}\"")
    print(f"  BoW vectors identical? {vec_a == vec_b}")
    print(f"  Cosine similarity: {sim:.3f}")
    print(f"  Same meaning? NO — word order changes who does what to whom.")
    print()

print("  BoW represents text as an UNORDERED collection of words.")
print("  It captures WHAT words appear, not HOW they are arranged.")
print("  For tasks where order matters (sentiment, intent, narrative),")
print("  BoW similarity will mislead. This motivates sequence-aware")
print("  representations explored in later phases.")
print()


# --- When BoW comparison works well ---
print("=== WHERE BoW COMPARISON WORKS WELL ===\n")

topic_corpus = [
    "Python is a popular programming language for data science",
    "Data science uses Python and machine learning algorithms",
    "The cake recipe requires flour sugar eggs and butter",
    "Mix the flour and sugar then add the eggs to bake a cake",
]

topic_labels = ["Tech (A)", "Tech (B)", "Cooking (C)", "Cooking (D)"]
topic_vocab = build_vocabulary(topic_corpus)
topic_vecs = [document_to_bow(doc, topic_vocab) for doc in topic_corpus]

print("  Documents:")
for label, doc in zip(topic_labels, topic_corpus):
    print(f"    {label}: \"{doc}\"")
print()

print("  Cosine similarities:")
for i in range(len(topic_corpus)):
    for j in range(i + 1, len(topic_corpus)):
        sim = cosine_similarity(topic_vecs[i], topic_vecs[j])
        same_topic = "same" if (i < 2 and j < 2) or (i >= 2 and j >= 2) else "diff"
        marker = " <-- same topic" if same_topic == "same" else ""
        print(f"    {topic_labels[i]} vs {topic_labels[j]}: {sim:.3f}{marker}")
print()
print("  BoW works well for TOPIC detection: documents about the same")
print("  subject share vocabulary, and cosine similarity captures this.")
print("  It fails for MEANING distinctions that depend on word order.")
```

---

## Key Takeaways

- **Cosine similarity measures the angle between document vectors**, not their distance. This makes it robust to differences in document length — a short email and a long report about the same topic will score high.
- **BoW comparison works well for topic detection.** Documents about cooking share cooking words; documents about finance share finance words. Cosine similarity reliably separates topics based on vocabulary overlap.
- **BoW comparison fails when word order carries meaning.** "Dog bites man" and "Man bites dog" produce identical vectors and a cosine similarity of 1.0, despite meaning very different things. This is the central limitation of the Bag of Words model.
- **BoW ignores order but captures presence.** This tradeoff is acceptable for many tasks (search, clustering, topic modeling) but insufficient for tasks requiring sequential understanding (sentiment analysis, question answering, translation). Recognizing when BoW is enough — and when it is not — is a key skill in NLP.
