# Compare Neural vs TF-IDF Models

> Phase 7 — Small Neural Text Models | Kata 7.3

---

## Concept & Intuition

### What problem are we solving?

We now have two fundamentally different ways to represent text: **TF-IDF** (from Phase 3) and **embedding-based representations** (from Kata 7.1). Both turn documents into vectors. Both enable similarity measurement. But they capture **different kinds of information**, and each has strengths the other lacks.

This kata puts them head-to-head on the same documents. We will see cases where embeddings detect similarity that TF-IDF misses entirely (synonyms, paraphrases), and cases where TF-IDF outperforms embeddings (exact keyword matching, interpretability). The goal is not to declare a winner --- it is to understand **when to use which representation**.

### Why earlier methods fail

TF-IDF represents each document as a sparse vector where each dimension corresponds to a specific word. Two documents are similar only if they share **the exact same words**. This is powerful for keyword search but blind to meaning:

- "The automobile drove fast" and "The car sped quickly" share **zero** content words. TF-IDF similarity = near zero. But any human knows these sentences mean the same thing.

Embedding-based representations solve this by mapping each word to a dense vector where synonyms are close together. When we average word embeddings to represent a document, "automobile" and "car" contribute similar vectors, so the two documents end up close in embedding space.

But embeddings are not strictly better. They lose **specificity**:

- "Python programming language" and "Python snake species" would get similar embedding-averaged vectors because the word "python" has the same embedding regardless of context. TF-IDF, by contrast, can distinguish them through the presence of specific co-occurring keywords like "programming" vs "snake."

### Mental models

- **TF-IDF is a magnifying glass**: It sees exactly what words are on the page, with sharp detail. It cannot see meaning behind the words, but it never hallucinates relationships that are not there. High precision, limited recall.
- **Embeddings are blurry binoculars**: They see the broad semantic landscape --- that two sentences are about the same topic even with different words. But they lose fine detail. Two documents about very different things can look similar if they use related vocabulary.
- **The interpretability tradeoff**: With TF-IDF, you can point to exactly which words drove the similarity score. With embeddings, the score emerges from the interaction of 20 abstract dimensions --- correct but unexplainable. This matters in domains where decisions must be justified (medicine, law, finance).

### Visual explanations

```
TF-IDF vs EMBEDDING SIMILARITY
=================================

Doc A: "The car drove fast on the highway"
Doc B: "The automobile sped quickly on the road"
Doc C: "The stock market crashed today"

TF-IDF SPACE (each word = one dimension):

  Vocabulary: [automobile, car, crashed, drove, fast, highway,
               market, on, quickly, road, sped, stock, the, today]

  Doc A: [0, .18, 0, .18, .18, .26, 0, .07, 0, 0, 0, 0, .07, 0]
  Doc B: [.26, 0, 0, 0, 0, 0, 0, .07, .26, .26, .26, 0, .07, 0]
  Doc C: [0, 0, .26, 0, 0, 0, .26, 0, 0, 0, 0, .26, .07, .26]

  TF-IDF cosine(A, B) = LOW   (almost no shared words!)
  TF-IDF cosine(A, C) = LOW   (no shared content words)

  TF-IDF CANNOT see that A and B mean the same thing.
  "car" and "automobile" are different dimensions.


EMBEDDING SPACE (each word -> dense vector):

  embed("car")        = [0.8, 0.3, -0.1, ...]
  embed("automobile") = [0.7, 0.4, -0.2, ...]   <-- close to "car"!
  embed("drove")      = [0.2, 0.9,  0.1, ...]
  embed("sped")       = [0.3, 0.8,  0.0, ...]   <-- close to "drove"!
  embed("stock")      = [-0.5, -0.3, 0.8, ...]  <-- far from vehicles

  Doc vector = average of word embeddings

  Embedding cosine(A, B) = HIGH  (synonyms are close in embed space)
  Embedding cosine(A, C) = LOW   (semantically unrelated)

  Embeddings CAPTURE that A and B are paraphrases.


WHEN TF-IDF WINS
===================

  Query: "python programming language"

  Doc X: "Python is a popular programming language for data science"
  Doc Y: "The python snake is found in tropical forests"

  TF-IDF: "programming" and "language" match Doc X exactly.
          Doc Y has no match for these specific keywords.
          TF-IDF correctly ranks Doc X higher.

  Embeddings: "python" has ONE vector regardless of meaning.
              Both docs contribute a similar "python" component.
              Embeddings may fail to distinguish them clearly.

  TF-IDF's exact matching is a feature, not a bug.


THE TRADEOFF
==============

                    TF-IDF          Embeddings
                    ------          ----------
  Synonyms          misses them     captures them
  Paraphrases       misses them     captures them
  Exact keywords    excellent       loses specificity
  Interpretability  fully clear     opaque
  Corpus needed     just the docs   large training corpus
  Dimensionality    sparse, high    dense, low
  Computation       fast (sparse)   fast (dense, small)
```

---

## Hands-on Exploration

1. Look at the document pairs in the code below. Before running it, predict which pairs will have higher similarity under TF-IDF vs embeddings. Focus on pairs where one document uses synonyms of the other. Run the code and check your predictions. Were there any surprises?

2. The code includes a "keyword search" experiment. Try different queries: one with exact terms from a document, and one that describes the same concept with different words. Note how TF-IDF excels at the first and fails at the second, while embeddings do the opposite. What kind of search engine would you build for a legal document database? For a casual Q&A system?

3. Examine the interpretability section of the output. For the TF-IDF similarity score, the code shows exactly which words contributed. For the embedding similarity, it cannot. Imagine you are building a medical system that recommends similar patient records. A doctor asks "Why did you say these two records are similar?" Which representation lets you answer that question?

---

## Live Code

```python
# --- Compare Neural Embeddings vs TF-IDF (Pure Python) ---
# Side-by-side comparison on the same documents.
# No numpy, no sklearn --- everything from scratch.

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


def vector_add(v1, v2):
    """Element-wise addition."""
    return [a + b for a, b in zip(v1, v2)]


# ============================================================
# TOKENIZER
# ============================================================

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


# ============================================================
# TRAINING CORPUS (for building embeddings)
# ============================================================
# This corpus teaches the embedding space about word relationships.
# It must be larger than the test documents to learn useful patterns.

training_corpus = [
    # Vehicle / transportation context
    "the car drove fast on the highway",
    "the automobile sped quickly down the road",
    "a fast car raced on the track",
    "the vehicle moved quickly through traffic",
    "she drove her car to the store",
    "the automobile needed fuel for the road",
    # Animal context
    "the cat sat on the warm mat",
    "the kitten slept on the soft rug",
    "a small cat chased the mouse",
    "the dog ran across the green yard",
    "the puppy played on the soft rug",
    "a big dog chased the ball",
    # Happy / emotion context
    "the happy child played in the park",
    "the joyful kid ran in the garden",
    "a cheerful boy smiled at the sun",
    "the glad girl laughed with friends",
    "happiness filled the room with warmth",
    "joy spread through the entire family",
    # Food context
    "the chef cooked a delicious meal",
    "the cook prepared fresh food",
    "a tasty dish was served at dinner",
    "the meal was warm and satisfying",
    # Science context
    "the scientist studied the experiment results",
    "the researcher analyzed the data carefully",
    "an experiment revealed new findings",
    "the study produced interesting results",
    # General connective tissue
    "the man walked to the market",
    "the woman walked to the store",
    "they went to the big store together",
    "he drove to the market in his car",
]

# ============================================================
# BUILD WORD EMBEDDINGS FROM TRAINING CORPUS
# ============================================================

# Tokenize training corpus
all_training_tokens = []
for sentence in training_corpus:
    all_training_tokens.extend(tokenize(sentence))

embed_vocab = sorted(set(all_training_tokens))
embed_word_to_idx = {w: i for i, w in enumerate(embed_vocab)}
embed_vocab_size = len(embed_vocab)

# Build co-occurrence matrix
WINDOW_SIZE = 3
cooccurrence = {w: Counter() for w in embed_vocab}

for sentence in training_corpus:
    tokens = tokenize(sentence)
    for i, target in enumerate(tokens):
        start = max(0, i - WINDOW_SIZE)
        end = min(len(tokens), i + WINDOW_SIZE + 1)
        for j in range(start, end):
            if i != j:
                cooccurrence[target][tokens[j]] += 1

# Random projection for embeddings
EMBED_DIM = 25
random.seed(42)
proj_matrix = [[random.choice([-1, 1]) for _ in range(EMBED_DIM)]
               for _ in range(embed_vocab_size)]

word_embeddings = {}
for word in embed_vocab:
    full_vec = [cooccurrence[word].get(w, 0) for w in embed_vocab]
    embed = [sum(full_vec[i] * proj_matrix[i][d]
                 for i in range(embed_vocab_size))
             for d in range(EMBED_DIM)]
    mag = magnitude(embed)
    if mag > 0:
        embed = [x / mag for x in embed]
    word_embeddings[word] = embed

print("=" * 60)
print("SETUP COMPLETE")
print("=" * 60)
print(f"  Training corpus: {len(training_corpus)} sentences")
print(f"  Embedding vocabulary: {embed_vocab_size} words")
print(f"  Embedding dimensions: {EMBED_DIM}")
print()

# ============================================================
# TEST DOCUMENTS (what we will compare)
# ============================================================

test_documents = {
    "D1": "the car drove fast on the highway",
    "D2": "the automobile sped quickly on the road",
    "D3": "the cat sat on the soft mat",
    "D4": "the kitten slept on the warm rug",
    "D5": "the happy child played in the park",
    "D6": "the joyful kid ran in the garden",
    "D7": "the scientist studied the experiment",
    "D8": "the chef cooked a delicious meal",
}

print("=" * 60)
print("TEST DOCUMENTS")
print("=" * 60)
for name, text in test_documents.items():
    print(f"  {name}: {text}")
print()
print("  Paraphrase pairs: D1/D2 (vehicles), D3/D4 (pets), D5/D6 (kids)")
print("  Unrelated pairs:  D1/D7 (car vs science), D5/D8 (kids vs food)")
print()

# ============================================================
# METHOD 1: TF-IDF REPRESENTATION
# ============================================================

# Build vocabulary from test documents only
test_tokenized = {name: tokenize(text) for name, text in test_documents.items()}
tfidf_vocab = sorted(set(w for tokens in test_tokenized.values() for w in tokens))
num_docs = len(test_documents)


def compute_tf(tokens):
    """Term frequency: count / total."""
    total = len(tokens)
    counts = Counter(tokens)
    return {w: c / total for w, c in counts.items()}


def compute_idf(all_doc_tokens, vocabulary):
    """Inverse document frequency: log(N / df)."""
    n = len(all_doc_tokens)
    idf = {}
    for word in vocabulary:
        df = sum(1 for tokens in all_doc_tokens.values() if word in tokens)
        idf[word] = math.log(n / df) if df > 0 else 0
    return idf


idf = compute_idf(test_tokenized, tfidf_vocab)


def tfidf_vector(tokens, vocabulary, idf_scores):
    """Build a TF-IDF vector for a document."""
    tf = compute_tf(tokens)
    return [tf.get(w, 0) * idf_scores.get(w, 0) for w in vocabulary]


tfidf_vectors = {name: tfidf_vector(tokens, tfidf_vocab, idf)
                 for name, tokens in test_tokenized.items()}

# ============================================================
# METHOD 2: EMBEDDING REPRESENTATION (average of word vectors)
# ============================================================


def document_embedding(tokens, embeddings, dim):
    """Average word embeddings to get a document vector."""
    vec = [0.0] * dim
    count = 0
    for word in tokens:
        if word in embeddings:
            vec = vector_add(vec, embeddings[word])
            count += 1
    if count > 0:
        vec = [x / count for x in vec]
    return vec


embed_vectors = {name: document_embedding(tokens, word_embeddings, EMBED_DIM)
                 for name, tokens in test_tokenized.items()}

# ============================================================
# HEAD-TO-HEAD COMPARISON
# ============================================================

print("=" * 60)
print("HEAD-TO-HEAD: TF-IDF vs EMBEDDINGS")
print("=" * 60)

comparison_pairs = [
    ("D1", "D2", "car/automobile (paraphrase)"),
    ("D3", "D4", "cat/kitten (paraphrase)"),
    ("D5", "D6", "happy/joyful (paraphrase)"),
    ("D1", "D3", "car vs cat (different topics)"),
    ("D5", "D8", "kids vs food (different topics)"),
    ("D1", "D7", "car vs science (different topics)"),
    ("D7", "D8", "science vs food (different topics)"),
]

print(f"\n  {'Pair':<10} {'Description':<30} {'TF-IDF':>8} {'Embed':>8}  Winner")
print(f"  {'-'*10} {'-'*30} {'-'*8} {'-'*8}  {'-'*20}")

tfidf_wins = 0
embed_wins = 0
chart_labels = []
chart_tfidf = []
chart_embed = []

for n1, n2, desc in comparison_pairs:
    tfidf_sim = cosine_similarity(tfidf_vectors[n1], tfidf_vectors[n2])
    embed_sim = cosine_similarity(embed_vectors[n1], embed_vectors[n2])

    chart_labels.append(f"{n1}-{n2}")
    chart_tfidf.append(round(tfidf_sim, 4))
    chart_embed.append(round(embed_sim, 4))

    # Determine which is more appropriate
    is_paraphrase = "paraphrase" in desc
    is_different = "different" in desc

    if is_paraphrase:
        # For paraphrases, higher similarity is better
        winner = "EMBED" if embed_sim > tfidf_sim else "TF-IDF"
        if embed_sim > tfidf_sim:
            embed_wins += 1
        else:
            tfidf_wins += 1
    elif is_different:
        # For different topics, lower similarity is better
        winner = "TF-IDF" if tfidf_sim < embed_sim else "EMBED"
        if tfidf_sim <= embed_sim:
            tfidf_wins += 1
        else:
            embed_wins += 1
    else:
        winner = ""

    print(f"  {n1+'-'+n2:<10} {desc:<30} {tfidf_sim:>8.4f} {embed_sim:>8.4f}  {winner}")

print()

# --- Visualization: TF-IDF vs Embedding Similarity ---
show_chart({
    "type": "bar",
    "title": "TF-IDF vs Embedding Similarity (Head-to-Head)",
    "labels": chart_labels,
    "datasets": [
        {"label": "TF-IDF", "data": chart_tfidf, "color": "#f59e0b"},
        {"label": "Embedding", "data": chart_embed, "color": "#3b82f6"},
    ],
    "options": {"x_label": "Document Pair", "y_label": "Cosine Similarity"}
})

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = range(len(chart_labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - width/2 for i in x], chart_tfidf, width, label="TF-IDF", color="#f59e0b")
    ax.bar([i + width/2 for i in x], chart_embed, width, label="Embedding", color="#3b82f6")
    ax.set_xlabel("Document Pair")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("TF-IDF vs Embedding Similarity (Head-to-Head)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(chart_labels, rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.savefig("tfidf_vs_embedding.png", dpi=100)
    plt.close()
    print("  [Saved matplotlib chart: tfidf_vs_embedding.png]")
except ImportError:
    pass

# ============================================================
# WHERE EMBEDDINGS WIN: SYNONYM DETECTION
# ============================================================

print("=" * 60)
print("WHERE EMBEDDINGS WIN: SYNONYM DETECTION")
print("=" * 60)
print()
print("  D1: 'the car drove fast on the highway'")
print("  D2: 'the automobile sped quickly on the road'")
print()

tfidf_sim_12 = cosine_similarity(tfidf_vectors["D1"], tfidf_vectors["D2"])
embed_sim_12 = cosine_similarity(embed_vectors["D1"], embed_vectors["D2"])

print(f"  TF-IDF similarity: {tfidf_sim_12:.4f}")
print(f"  Embedding similarity: {embed_sim_12:.4f}")
print()

# Show why TF-IDF fails
shared = set(test_tokenized["D1"]) & set(test_tokenized["D2"])
d1_only = set(test_tokenized["D1"]) - set(test_tokenized["D2"])
d2_only = set(test_tokenized["D2"]) - set(test_tokenized["D1"])

print(f"  Shared words:  {sorted(shared)}")
print(f"  D1 only:       {sorted(d1_only)}")
print(f"  D2 only:       {sorted(d2_only)}")
print()
print("  TF-IDF sees almost no overlap because the content words differ.")
print("  'car' vs 'automobile', 'drove' vs 'sped', 'fast' vs 'quickly'")
print("  --- all different dimensions, zero contribution to similarity.")
print()
print("  Embeddings know these are synonyms because they appear in")
print("  similar contexts in the training corpus:")

synonym_pairs = [("car", "automobile"), ("fast", "quickly")]
for w1, w2 in synonym_pairs:
    if w1 in word_embeddings and w2 in word_embeddings:
        sim = cosine_similarity(word_embeddings[w1], word_embeddings[w2])
        print(f"    cosine('{w1}', '{w2}') = {sim:+.4f}")
print()

# ============================================================
# WHERE TF-IDF WINS: EXACT KEYWORD MATCHING
# ============================================================

print("=" * 60)
print("WHERE TF-IDF WINS: KEYWORD SEARCH")
print("=" * 60)
print()

queries = [
    "scientist experiment",
    "chef cooked meal",
    "car highway fast",
]

print("  Ranking documents for keyword queries:")
print()

for query_text in queries:
    query_tokens = tokenize(query_text)
    query_tfidf = tfidf_vector(query_tokens, tfidf_vocab, idf)
    query_embed = document_embedding(query_tokens, word_embeddings, EMBED_DIM)

    tfidf_scores = []
    embed_scores = []
    for name in test_documents:
        ts = cosine_similarity(query_tfidf, tfidf_vectors[name])
        es = cosine_similarity(query_embed, embed_vectors[name])
        tfidf_scores.append((name, ts))
        embed_scores.append((name, es))

    tfidf_scores.sort(key=lambda x: x[1], reverse=True)
    embed_scores.sort(key=lambda x: x[1], reverse=True)

    print(f"  Query: '{query_text}'")
    print(f"    TF-IDF ranking:     ", end="")
    for name, score in tfidf_scores[:3]:
        print(f"{name}({score:.3f}) ", end="")
    print()
    print(f"    Embedding ranking:  ", end="")
    for name, score in embed_scores[:3]:
        print(f"{name}({score:.3f}) ", end="")
    print()
    print()

print("  TF-IDF excels at exact keyword matching because each word")
print("  is its own dimension. If the query word appears in the")
print("  document, it contributes directly to the score.")
print()

# ============================================================
# INTERPRETABILITY COMPARISON
# ============================================================

print("=" * 60)
print("INTERPRETABILITY: CAN YOU EXPLAIN THE SCORE?")
print("=" * 60)
print()

n1, n2 = "D1", "D2"
print(f"  Comparing {n1} and {n2}:")
print(f"    {n1}: '{test_documents[n1]}'")
print(f"    {n2}: '{test_documents[n2]}'")
print()

# TF-IDF: show which words contribute to similarity
print("  TF-IDF explanation (word-by-word contribution to dot product):")
t1_tf = compute_tf(test_tokenized[n1])
t2_tf = compute_tf(test_tokenized[n2])
contributions = []
for word in tfidf_vocab:
    v1 = t1_tf.get(word, 0) * idf.get(word, 0)
    v2 = t2_tf.get(word, 0) * idf.get(word, 0)
    contrib = v1 * v2
    if contrib > 0:
        contributions.append((word, contrib, v1, v2))

contributions.sort(key=lambda x: x[1], reverse=True)
for word, contrib, v1, v2 in contributions:
    print(f"    '{word}': TF-IDF_D1={v1:.4f} * TF-IDF_D2={v2:.4f} "
          f"= {contrib:.6f}")

if not contributions:
    print("    (no shared content words with non-zero TF-IDF)")

print()
print("  Embedding explanation:")
embed_sim_val = cosine_similarity(embed_vectors[n1], embed_vectors[n2])
print(f"    Cosine similarity = {embed_sim_val:.4f}")
print("    Why? The embedding dimensions are abstract. We cannot")
print("    point to specific words or features that drove this score.")
print("    The similarity emerges from the interaction of {0} opaque".format(
    EMBED_DIM))
print("    dimensions. This is the cost of semantic richness.")
print()

# ============================================================
# SUMMARY SCORECARD
# ============================================================

print("=" * 60)
print("SUMMARY: WHEN TO USE WHICH")
print("=" * 60)
print()
print("  TASK                         BEST METHOD     WHY")
print("  " + "-" * 56)
print("  Synonym-aware search         Embeddings      Captures meaning, not words")
print("  Paraphrase detection         Embeddings      Different words, same idea")
print("  Exact keyword search         TF-IDF          Precise word matching")
print("  Explainable recommendations  TF-IDF          Word contributions are clear")
print("  Small corpus, no training    TF-IDF          Needs no pre-training")
print("  Cross-lingual similarity     Embeddings      With multilingual training")
print("  Document deduplication       Both            Use TF-IDF first, embed second")
print("  Topic clustering             Either          Depends on vocabulary overlap")
print()
print("  The right choice depends on your data, your task,")
print("  and whether you need to explain your results.")
print()
print("  In practice, many systems combine both:")
print("    1. TF-IDF for fast candidate retrieval (exact match)")
print("    2. Embeddings for re-ranking (semantic similarity)")
print("  This is the 'retrieve and re-rank' pattern used in")
print("  modern search engines and recommendation systems.")
```

---

## Key Takeaways

- **TF-IDF and embeddings capture different information.** TF-IDF measures lexical overlap --- whether documents share the same words. Embeddings measure semantic similarity --- whether documents express the same meaning. Neither is universally better; they answer different questions.
- **Embeddings detect synonyms and paraphrases that TF-IDF cannot.** Because word embeddings place "car" and "automobile" close together, embedding-based document similarity stays high even when no exact words are shared. This is the key advantage of dense representations over sparse ones.
- **TF-IDF offers interpretability that embeddings sacrifice.** With TF-IDF, you can point to exactly which words drove the similarity score and how much each contributed. With embeddings, the score is a black-box number emerging from abstract dimensions. In high-stakes domains, this transparency matters.
- **The best systems combine both approaches.** Use TF-IDF for fast, interpretable candidate retrieval, then use embeddings for semantic re-ranking. This "retrieve and re-rank" pattern captures the strengths of both representations while mitigating their individual weaknesses.
