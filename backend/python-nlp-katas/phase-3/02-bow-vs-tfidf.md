# Compare Similarity Using BoW vs TF-IDF

> Phase 3 — TF-IDF | Kata 3.2

---

## Concept & Intuition

### What problem are we solving?

When we compare two documents, we need to measure how "similar" they are. In Phase 2, we learned to represent documents as vectors using Bag of Words (raw counts). We can then compute similarity between these vectors using cosine similarity. But BoW-based similarity has a critical flaw: **common words inflate similarity scores between unrelated documents**.

Two documents about completely different topics — say, cooking and astronomy — will still share hundreds of common words ("the", "is", "a", "with", "and"). Under BoW, these shared words contribute heavily to the similarity score, making every pair of documents look more alike than they really are.

TF-IDF fixes this by downweighting words that appear across many documents. The result: similarity scores become driven by **topically meaningful words**, not by the shared scaffolding of language.

### Why naive/earlier approaches fail

Consider these three documents:

- Doc A: "the cat sat on the mat on the floor"
- Doc B: "the dog sat on the rug on the porch"
- Doc C: "the cat chased a mouse across the yard"

Using raw BoW, Doc A and Doc B share "the" (3x each), "sat" (1x each), and "on" (2x each). That is 6 shared word occurrences. Doc A and Doc C share "the" (3x, 2x) and "cat" (1x each). The BoW similarity between A and B will be high — driven almost entirely by "the" and "on" — even though A is about a cat and B is about a dog.

With TF-IDF, "the" and "on" get suppressed (they appear in all documents), and the similarity between A and C rises because "cat" now carries real weight. The similarity between A and B drops because their shared words are all common ones.

### Mental models

- **BoW similarity is like comparing two people by how many English words they both know.** Everyone knows "the", "is", "and" — so everyone looks similar. TF-IDF is like comparing them by their specialized vocabulary: one knows "photosynthesis" and the other knows "jurisprudence" — now you can tell them apart.
- **Common words are noise in the similarity signal.** TF-IDF is a noise filter. After filtering, the remaining signal is what actually distinguishes documents from one another.
- **Cosine similarity measures direction, not magnitude.** Two vectors pointing the same way have cosine similarity = 1, regardless of length. This matters because documents of different lengths can still be about the same topic.

### Visual explanations

```
BOW vs TF-IDF SIMILARITY: WHY IT MATTERS
=========================================

Corpus:
  Doc A: "the cat sat on the mat"        (topic: cat)
  Doc B: "the dog sat on the rug"         (topic: dog)
  Doc C: "the cat played with the ball"   (topic: cat)

BoW Vectors (raw counts):
            the  cat  sat  on  mat  dog  rug  played  with  ball
  Doc A: [  2    1    1    1   1    0    0     0       0     0  ]
  Doc B: [  2    0    1    1   0    1    1     0       0     0  ]
  Doc C: [  2    1    0    0   0    0    0     1       1     1  ]

  cosine(A, B) using BoW:  HIGH   (shared: "the" x2, "sat", "on")
  cosine(A, C) using BoW:  MEDIUM (shared: "the" x2, "cat")

  BoW says A is MORE similar to B than to C!
  But A and C are both about cats. Something is wrong.

TF-IDF Vectors (common words suppressed):
            the  cat  sat  on  mat  dog  rug  played  with  ball
  Doc A: [ 0.0  0.14 0.14 0.14 0.27 0.0  0.0  0.0    0.0   0.0 ]
  Doc B: [ 0.0  0.0  0.14 0.14 0.0  0.27 0.27 0.0    0.0   0.0 ]
  Doc C: [ 0.0  0.14 0.0  0.0  0.0  0.0  0.0  0.27   0.27  0.27]

  cosine(A, B) using TF-IDF:  LOW    ("the" gone, only "sat"+"on" shared)
  cosine(A, C) using TF-IDF:  MEDIUM ("cat" now carries real weight)

  TF-IDF correctly identifies A and C as more topically related.
```

```
COSINE SIMILARITY FORMULA
=========================

                    A . B            sum(a_i * b_i)
  cosine(A, B) = --------- = -----------------------------
                 |A| * |B|   sqrt(sum(a_i^2)) * sqrt(sum(b_i^2))

  Result range: 0 (completely different) to 1 (identical direction)

  Example with 2D vectors:
    A = [3, 0]     (points right)
    B = [0, 4]     (points up)
    C = [2, 0]     (points right, shorter)

    cosine(A, B) = 0   (perpendicular — nothing in common)
    cosine(A, C) = 1   (same direction — identical topic, different lengths)
```

---

## Hands-on Exploration

1. Take the three documents from the visual explanation above. Compute the BoW cosine similarity between all three pairs (A-B, A-C, B-C) by hand. Verify that BoW makes A and B more similar than A and C. Then consider: what would happen if you added even more common words (e.g., "the the the") to every document?

2. Now remove "the", "on", "sat" (the words appearing in multiple documents) from the BoW vectors and recompute cosine similarity. How do the rankings change? Notice that manual stopword removal achieves something similar to what TF-IDF does automatically — but TF-IDF is more principled because it uses the corpus statistics rather than a fixed list.

3. Imagine a corpus of 1000 news articles. You want to find the article most similar to a query: "stock market crash recession". With BoW, articles mentioning "the" 50 times will look similar to everything. With TF-IDF, "stock", "market", "crash", and "recession" will dominate the comparison. Which approach would you trust for a search engine? Why?

---

## Live Code

```python
# --- Comparing document similarity: BoW vs TF-IDF ---
import math

# A small corpus where BoW gives misleading similarity
corpus = [
    "the cat sat on the mat in the room",           # Doc 0: about a cat
    "the dog sat on the rug in the hall",           # Doc 1: about a dog
    "the cat chased the mouse in the garden",       # Doc 2: about a cat
    "the dog fetched the stick in the park",        # Doc 3: about a dog
    "the bird sat on the branch in the tree",       # Doc 4: filler (shares structure)
    "the fish swam in the pond near the rocks",     # Doc 5: filler (shares structure)
]

print("=== CORPUS ===\n")
for i, doc in enumerate(corpus):
    print(f"  Doc {i}: \"{doc}\"")
print()

# Tokenize
documents = [doc.lower().split() for doc in corpus]

# Build vocabulary
vocabulary = sorted(set(word for doc in documents for word in doc))


# --- Bag of Words ---

def bow_vector(doc, vocab):
    """Build a raw count vector for a document."""
    counts = {}
    for word in doc:
        counts[word] = counts.get(word, 0) + 1
    return [counts.get(w, 0) for w in vocab]


# --- TF-IDF ---

def compute_tf(doc):
    """Term frequency: count / total words in document."""
    total = len(doc)
    counts = {}
    for word in doc:
        counts[word] = counts.get(word, 0) + 1
    return {word: count / total for word, count in counts.items()}

def compute_idf(documents, vocab):
    """Inverse document frequency: log(N / docs_containing_term)."""
    n = len(documents)
    idf = {}
    for word in vocab:
        doc_freq = sum(1 for doc in documents if word in doc)
        # Add check to avoid division by zero (shouldn't happen with our vocab)
        idf[word] = math.log(n / doc_freq) if doc_freq > 0 else 0
    return idf

def tfidf_vector(doc, vocab, idf):
    """Build a TF-IDF vector for a document."""
    tf = compute_tf(doc)
    return [tf.get(w, 0) * idf[w] for w in vocab]


# --- Cosine Similarity ---

def cosine_similarity(vec_a, vec_b):
    """Compute cosine similarity between two vectors."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a * a for a in vec_a))
    mag_b = math.sqrt(sum(b * b for b in vec_b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# Build all vectors
bow_vectors = [bow_vector(doc, vocabulary) for doc in documents]

idf = compute_idf(documents, vocabulary)
tfidf_vectors = [tfidf_vector(doc, vocabulary, idf) for doc in documents]


# --- Compare all pairs ---

print("=== PAIRWISE SIMILARITY ===\n")
print(f"  {'Pair':<15s} {'BoW Cosine':>12s} {'TF-IDF Cosine':>14s}  {'Better?'}")
print(f"  {'-'*15} {'-'*12} {'-'*14}  {'-'*30}")

pairs = [(0, 1), (0, 2), (1, 3), (0, 3), (1, 2)]

for i, j in pairs:
    bow_sim = cosine_similarity(bow_vectors[i], bow_vectors[j])
    tfidf_sim = cosine_similarity(tfidf_vectors[i], tfidf_vectors[j])

    # Determine which method gives a more meaningful result
    note = ""
    if (i, j) in [(0, 2)]:
        note = "<-- both about cats"
    elif (i, j) in [(1, 3)]:
        note = "<-- both about dogs"
    elif (i, j) in [(0, 1)]:
        note = "<-- cat vs dog (different topics)"
    elif (i, j) in [(0, 3), (1, 2)]:
        note = "<-- different topics"

    print(f"  Doc {i} vs Doc {j}  {bow_sim:12.4f} {tfidf_sim:14.4f}  {note}")

# --- Show why BoW is misleading ---

print("\n=== WHY BOW IS MISLEADING ===\n")
print("  Words shared between Doc 0 (cat) and Doc 1 (dog):")
shared_01 = set(documents[0]) & set(documents[1])
for word in sorted(shared_01):
    c0 = documents[0].count(word)
    c1 = documents[1].count(word)
    idf_val = idf.get(word, 0)
    print(f"    \"{word}\": count in Doc0={c0}, Doc1={c1}, IDF={idf_val:.4f}")

print("\n  Words shared between Doc 0 (cat) and Doc 2 (cat):")
shared_02 = set(documents[0]) & set(documents[2])
for word in sorted(shared_02):
    c0 = documents[0].count(word)
    c2 = documents[2].count(word)
    idf_val = idf.get(word, 0)
    print(f"    \"{word}\": count in Doc0={c0}, Doc2={c2}, IDF={idf_val:.4f}")

print()
print("  BoW similarity between Doc 0 and Doc 1 is inflated by shared")
print("  common words (\"the\", \"sat\", \"on\", \"in\") that appear everywhere.")
print()
print("  TF-IDF suppresses these common words and lets topic-specific words")
print("  like \"cat\", \"mat\", \"mouse\", \"garden\" drive the similarity score.")


# --- Show the IDF values to explain what's happening ---

print("\n=== IDF VALUES (what gets suppressed) ===\n")

idf_sorted = sorted(idf.items(), key=lambda x: x[1])
for word, score in idf_sorted:
    doc_freq = sum(1 for doc in documents if word in doc)
    bar = "#" * int(score * 20)
    label = "SUPPRESSED" if score == 0 else ("low" if score < 0.5 else "high")
    print(f"  {word:>10s}  IDF={score:.3f}  in {doc_freq}/{len(documents)} docs"
          f"  {bar}  ({label})")


# --- Final verdict ---

print("\n=== VERDICT ===\n")

bow_01 = cosine_similarity(bow_vectors[0], bow_vectors[1])
bow_02 = cosine_similarity(bow_vectors[0], bow_vectors[2])
tfidf_01 = cosine_similarity(tfidf_vectors[0], tfidf_vectors[1])
tfidf_02 = cosine_similarity(tfidf_vectors[0], tfidf_vectors[2])

print(f"  Question: Is Doc 0 (cat) more similar to Doc 1 (dog) or Doc 2 (cat)?")
print()
print(f"  BoW answer:    Doc 0-1 = {bow_01:.4f}, Doc 0-2 = {bow_02:.4f}", end="")
if bow_01 > bow_02:
    print("  --> BoW says Doc 1 (dog) is closer. WRONG.")
else:
    print("  --> BoW says Doc 2 (cat) is closer. Correct.")

print(f"  TF-IDF answer: Doc 0-1 = {tfidf_01:.4f}, Doc 0-2 = {tfidf_02:.4f}", end="")
if tfidf_02 > tfidf_01:
    print("  --> TF-IDF says Doc 2 (cat) is closer. Correct!")
else:
    print("  --> TF-IDF says Doc 1 (dog) is closer.")

print()
print("  TF-IDF produces more topically meaningful similarity scores")
print("  by suppressing words that carry no discriminative information.")
```

---

## Key Takeaways

- **BoW similarity is dominated by common words.** When two documents share frequent words like "the", "is", "on", their BoW cosine similarity is inflated regardless of topical relevance.
- **TF-IDF rescues similarity by downweighting noise.** Words appearing across many documents contribute less to the similarity score, letting topic-specific words drive the comparison.
- **Cosine similarity measures direction, not magnitude.** This makes it robust to document length differences — a short document and a long document about the same topic will still score high.
- **TF-IDF is a relevance heuristic, not "understanding."** It improves on BoW by using corpus statistics, but it still cannot capture synonymy, word order, or meaning. "Happy" and "joyful" remain completely unrelated in TF-IDF space.
