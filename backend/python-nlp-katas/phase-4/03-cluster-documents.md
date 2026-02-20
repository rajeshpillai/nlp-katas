# Cluster Documents by Topic

> Phase 4 — Similarity & Classical NLP Tasks | Kata 4.3

---

## Concept & Intuition

### What problem are we solving?

We have a pile of unlabeled documents and we want to **automatically group them by topic**. Nobody told us what the topics are. Nobody labeled anything. We just want the algorithm to discover that some documents are about sports, others about technology, and others about cooking — purely from the text.

This is **unsupervised learning**: finding structure in data without human labels. Document clustering converts each document into a TF-IDF vector and then groups vectors that are close together in space. Documents pointing in similar directions (high cosine similarity) end up in the same cluster.

### Why naive/earlier approaches fail

**Manual categorization** does not scale. A human can sort 50 documents, but not 50,000.

**Simple keyword rules** ("if the document contains 'goal', it's about sports") break immediately. "Goal" appears in business documents ("quarterly goals"), self-help articles, and game design discussions. Hard-coded rules are brittle and miss the nuance that comes from looking at the full distribution of words.

**Raw word overlap** without vectorization gives no way to measure "distance" between documents. We need documents in a shared vector space so that "close" and "far" have geometric meaning. TF-IDF vectors in a high-dimensional space give us exactly that.

### Mental models

- **K-means clustering** works by: (1) pick K random centroids, (2) assign each document to the nearest centroid, (3) recompute centroids as the mean of assigned documents, (4) repeat until stable. It is simple, interpretable, and surprisingly effective.
- **Centroid** = the "average" vector of a cluster. It represents the typical direction of documents in that group.
- **Convergence** = when no document changes its cluster assignment between iterations. The algorithm has found a stable grouping.
- **K** = the number of clusters. You must choose this in advance (a real limitation). Too few clusters merge distinct topics; too many split coherent topics.

### Visual explanations

```
K-means in 2D (imagine TF-IDF vectors projected to 2 dimensions):

  Step 1: Random centroids         Step 2: Assign to nearest
      *                                * x x
                                        x
          x  x                     o  x  x
        x      x                     x      x
                     x                            x
     o     x              o           o     x       o
        x    x                           x    x

  * = centroid                     x = assigned to *
  o = centroid                     (each point goes to nearest centroid)
  x = documents

  Step 3: Recompute centroids      Step 4: Reassign (converged!)
        x x                              x x
      *  x                             *  x
        x  x                             x  x
          x      x                         x      x
                            o                           o
            x                                x
         x    x                           x    x

  * moved to center of its           No points changed cluster.
  assigned points. o did too.         Algorithm stops. Done!


  Why cosine-based clustering works for text:

    Cluster 1 (Sports)          Cluster 2 (Tech)
    ┌─────────────────┐         ┌──────────────────┐
    │ "team won game"  │         │ "code runs fast"  │
    │ "scored a goal"  │         │ "debug the app"   │
    │ "final match"    │         │ "server crashed"  │
    └─────────────────┘         └──────────────────┘

    These vectors point in                These vectors point in
    a "sports" direction                  a "tech" direction
    in TF-IDF space.                      in TF-IDF space.

    The algorithm discovers this grouping WITHOUT knowing
    what "sports" or "tech" means — purely from word patterns.
```

---

## Hands-on Exploration

1. Look at the 12 documents in the code below. Before running it, manually sort them into 3 groups by topic. Then run the clustering algorithm and compare your grouping with the algorithm's output. Where do they agree? Where do they disagree? Why might the algorithm group things differently than you?

2. Change K from 3 to 2 and observe which topics get merged together. Then try K=4 and see if a coherent fourth cluster emerges or if a good cluster gets split. This demonstrates the fundamental challenge of choosing K.

3. Run the algorithm multiple times (it uses random initialization). Notice that you might get different clusterings each run. This non-determinism is a real property of k-means. Think about why random starting positions lead to different outcomes and what strategies might help (hint: the code runs multiple times and picks the best).

---

## Live Code

```python
# --- Document Clustering with K-Means on TF-IDF Vectors ---
# Pure Python stdlib — no external dependencies

import math
import random
from collections import Counter


def tokenize(text):
    """Lowercase and extract alphabetic tokens."""
    tokens = []
    current = []
    for ch in text.lower():
        if ch.isalpha():
            current.append(ch)
        else:
            if current:
                tokens.append("".join(current))
                current = []
    if current:
        tokens.append("".join(current))
    return tokens


STOPWORDS = {
    "a", "an", "the", "is", "it", "in", "on", "of", "to", "and",
    "for", "with", "that", "this", "are", "was", "be", "as", "at",
    "by", "or", "from", "but", "not", "have", "has", "had", "do",
    "does", "did", "will", "can", "so", "if", "its", "we", "they",
    "their", "about", "been", "more", "when", "also", "up", "into",
}


def preprocess(text):
    """Tokenize and remove stopwords."""
    return [t for t in tokenize(text) if t not in STOPWORDS]


def compute_tfidf(doc_tokens_list):
    """Compute TF-IDF vectors for a list of token lists."""
    # Build vocabulary
    vocab_set = set()
    for tokens in doc_tokens_list:
        vocab_set.update(tokens)
    vocab = sorted(vocab_set)
    word_to_idx = {w: i for i, w in enumerate(vocab)}

    n_docs = len(doc_tokens_list)
    n_words = len(vocab)

    # Compute document frequency for each word
    df = [0] * n_words
    for tokens in doc_tokens_list:
        seen = set(tokens)
        for word in seen:
            df[word_to_idx[word]] += 1

    # Compute IDF
    idf = [0.0] * n_words
    for i in range(n_words):
        if df[i] > 0:
            idf[i] = math.log(n_docs / df[i])

    # Compute TF-IDF vectors
    vectors = []
    for tokens in doc_tokens_list:
        tf = Counter(tokens)
        total = len(tokens) if tokens else 1
        vec = [0.0] * n_words
        for word, count in tf.items():
            idx = word_to_idx[word]
            vec[idx] = (count / total) * idf[idx]
        vectors.append(vec)

    return vocab, vectors


def cosine_similarity(v1, v2):
    """Cosine similarity between two vectors."""
    dot = sum(a * b for a, b in zip(v1, v2))
    m1 = math.sqrt(sum(a * a for a in v1))
    m2 = math.sqrt(sum(b * b for b in v2))
    if m1 == 0 or m2 == 0:
        return 0.0
    return dot / (m1 * m2)


def vector_add(v1, v2):
    """Element-wise addition of two vectors."""
    return [a + b for a, b in zip(v1, v2)]


def vector_scale(v, scalar):
    """Multiply a vector by a scalar."""
    return [x * scalar for x in v]


def kmeans_cosine(vectors, k, max_iters=50, n_runs=5):
    """
    K-means clustering using cosine similarity.

    Runs multiple times with different random seeds and returns
    the best clustering (highest average within-cluster similarity).
    """
    n = len(vectors)
    dim = len(vectors[0])
    best_assignment = None
    best_score = -1

    for run in range(n_runs):
        # Initialize: pick k random documents as centroids
        centroid_indices = random.sample(range(n), k)
        centroids = [vectors[i][:] for i in centroid_indices]

        assignment = [0] * n

        for iteration in range(max_iters):
            # Assign each document to the most similar centroid
            new_assignment = [0] * n
            for i in range(n):
                best_sim = -1
                best_cluster = 0
                for c in range(k):
                    sim = cosine_similarity(vectors[i], centroids[c])
                    if sim > best_sim:
                        best_sim = sim
                        best_cluster = c
                new_assignment[i] = best_cluster

            # Check convergence
            if new_assignment == assignment:
                break
            assignment = new_assignment

            # Recompute centroids as mean of assigned vectors
            new_centroids = [[0.0] * dim for _ in range(k)]
            counts = [0] * k
            for i in range(n):
                c = assignment[i]
                new_centroids[c] = vector_add(new_centroids[c], vectors[i])
                counts[c] += 1

            for c in range(k):
                if counts[c] > 0:
                    centroids[c] = vector_scale(new_centroids[c], 1.0 / counts[c])

        # Score this run: average within-cluster cosine similarity
        total_sim = 0.0
        count = 0
        for c in range(k):
            members = [i for i in range(n) if assignment[i] == c]
            for i in members:
                total_sim += cosine_similarity(vectors[i], centroids[c])
                count += 1
        avg_sim = total_sim / count if count > 0 else 0

        if avg_sim > best_score:
            best_score = avg_sim
            best_assignment = assignment[:]

    return best_assignment, best_score


def get_top_terms(cluster_doc_indices, doc_tokens_list, vocab, vectors, top_n=5):
    """Find the most representative terms for a cluster."""
    dim = len(vocab)
    centroid = [0.0] * dim
    for i in cluster_doc_indices:
        centroid = vector_add(centroid, vectors[i])
    if cluster_doc_indices:
        centroid = vector_scale(centroid, 1.0 / len(cluster_doc_indices))

    # Get top terms by centroid weight
    term_scores = [(vocab[i], centroid[i]) for i in range(dim) if centroid[i] > 0]
    term_scores.sort(key=lambda x: x[1], reverse=True)
    return term_scores[:top_n]


# --- Our corpus: 12 documents, 3 implicit topics ---
documents = {
    "D01": "The team scored three goals in the championship match",
    "D02": "Players trained hard for the upcoming tournament season",
    "D03": "The coach announced the starting lineup for the final game",
    "D04": "Fans cheered as the winning goal was scored in extra time",
    "D05": "New programming language features improve developer productivity",
    "D06": "The software update fixed critical security vulnerabilities",
    "D07": "Cloud computing platforms scale applications automatically",
    "D08": "Developers debug code using integrated testing frameworks",
    "D09": "The chef prepared a traditional recipe with fresh ingredients",
    "D10": "Baking bread requires precise measurements of flour and yeast",
    "D11": "The restaurant menu featured seasonal dishes and local produce",
    "D12": "Cooking techniques vary across different regional cuisines",
}

# Preprocess and vectorize
names = list(documents.keys())
texts = list(documents.values())
doc_tokens_list = [preprocess(text) for text in texts]
vocab, vectors = compute_tfidf(doc_tokens_list)

print("=" * 60)
print("CORPUS")
print("=" * 60)
for name, text in documents.items():
    print(f"  {name}: {text}")
print(f"\nVocabulary size: {len(vocab)} terms")
print(f"Documents: {len(documents)}")
print()

# --- Cluster into K=3 groups ---
K = 3
random.seed(42)  # for reproducibility
assignment, score = kmeans_cosine(vectors, K)

print("=" * 60)
print(f"CLUSTERING RESULTS (K={K})")
print("=" * 60)
print(f"Best average within-cluster similarity: {score:.4f}")
print()

for c in range(K):
    members = [i for i in range(len(names)) if assignment[i] == c]
    top_terms = get_top_terms(members, doc_tokens_list, vocab, vectors)

    term_str = ", ".join(f"{w} ({s:.3f})" for w, s in top_terms)
    print(f"--- Cluster {c + 1} ({len(members)} docs) ---")
    print(f"  Top terms: {term_str}")
    for i in members:
        print(f"    {names[i]}: {texts[i][:65]}...")
    print()

# --- Show pairwise similarity matrix for insight ---
print("=" * 60)
print("WITHIN-CLUSTER vs BETWEEN-CLUSTER SIMILARITY")
print("=" * 60)

# Compute average within-cluster similarity
for c in range(K):
    members = [i for i in range(len(names)) if assignment[i] == c]
    if len(members) < 2:
        continue
    sims = []
    for a in range(len(members)):
        for b in range(a + 1, len(members)):
            sims.append(cosine_similarity(vectors[members[a]], vectors[members[b]]))
    avg = sum(sims) / len(sims) if sims else 0
    print(f"  Cluster {c + 1} avg within-similarity: {avg:.4f}")

# Compute average between-cluster similarity
between_sims = []
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        if assignment[i] != assignment[j]:
            between_sims.append(cosine_similarity(vectors[i], vectors[j]))
avg_between = sum(between_sims) / len(between_sims) if between_sims else 0
print(f"  Average between-cluster similarity:    {avg_between:.4f}")
print()
print("Good clustering: within-cluster similarity >> between-cluster similarity")
print()

# --- Show what happens with different K values ---
print("=" * 60)
print("EFFECT OF CHOOSING DIFFERENT K")
print("=" * 60)

for k in [2, 3, 4, 5]:
    random.seed(42)
    asgn, sc = kmeans_cosine(vectors, k)
    cluster_sizes = [sum(1 for a in asgn if a == c) for c in range(k)]
    print(f"  K={k}: avg similarity={sc:.4f}, cluster sizes={cluster_sizes}")

print()
print("Note: K=3 matches our 3 natural topics (sports, tech, cooking).")
print("K=2 forces two topics to merge. K=5 splits coherent topics apart.")
print()
print("Many NLP problems reduce to similarity in vector space.")
print("Clustering discovers structure without labels — purely from")
print("the geometric arrangement of TF-IDF document vectors.")
```

---

## Key Takeaways

- **Clustering is unsupervised similarity.** Documents that point in similar directions in TF-IDF space naturally group together, revealing latent topics without any human labels.
- **K-means is simple but powerful.** The iterate-until-stable loop of "assign to nearest centroid, recompute centroids" discovers meaningful document groups from raw text vectors.
- **Choosing K is a real problem.** Too few clusters merge distinct topics; too many fragment coherent ones. There is no single right answer — K encodes your assumption about how many topics exist.
- **Good clusters have high internal similarity and low external similarity.** This is the geometric definition of "these documents belong together." Many NLP problems reduce to similarity in vector space.
