# Pretraining vs Fine-Tuning

> Phase 10 — Modern NLP Pipelines (Awareness) | Kata 10.1

---

## Concept & Intuition

### What problem are we solving?

Training a model from scratch for every new NLP task is expensive and wasteful. A sentiment classifier needs to learn what words mean, how sentences work, and what sentiment is -- all from a small labeled dataset. A spam filter needs to learn the same language fundamentals, again from scratch. Every new task repeats the same expensive work of learning the structure of language.

The **two-stage paradigm** solves this: first, build general-purpose representations of language from a massive unlabeled corpus (pretraining), then adapt those representations for a specific task with a small labeled dataset (fine-tuning). This is an engineering technique for amortizing the cost of learning language structure across many downstream tasks.

### Why earlier methods fail

**Training from scratch** on small datasets produces poor representations. If you have 1,000 movie reviews for sentiment analysis, your model only sees the patterns present in those 1,000 examples. It cannot learn that "excellent" and "superb" are related unless both appear in the training data. It cannot learn sentence structure from a few hundred sentences.

**Classical methods like TF-IDF** do not learn representations at all -- they compute statistics. TF-IDF cannot discover that "terrible" and "awful" serve similar roles because it treats every word as an independent dimension. There is no mechanism for knowledge to transfer between tasks: a TF-IDF model trained on movie reviews tells you nothing about classifying support tickets.

**Fixed embeddings** (like Word2Vec) are a step forward -- they learn word relationships from large corpora. But the embeddings are frozen. The word "bank" gets the same vector whether it appears in a financial document or a geography textbook. There is no mechanism to adapt the representations for the task at hand.

### Mental models

- **Pretraining is like learning to read.** You spend years reading diverse text -- novels, newspapers, textbooks -- and develop a general understanding of how language works. This is expensive (years of effort) but the knowledge transfers to any reading task.
- **Fine-tuning is like studying for a specific exam.** You already know how to read. Now you spend a few hours focusing on chemistry. You are not learning the alphabet again -- you are adapting your existing knowledge to a specific domain.
- **Transfer learning is the core idea.** Representations learned from one task (predicting the next word in Wikipedia) transfer to another task (classifying sentiment in reviews) because both tasks require understanding language structure.
- **The pretrained model is a starting point, not a finished product.** It encodes general language patterns. Fine-tuning steers those patterns toward a specific objective.

### Visual explanations

```
THE TWO-STAGE PARADIGM
======================

Stage 1: PRETRAINING (expensive, done once)
  ┌──────────────────────────────────────────────┐
  │  Massive unlabeled corpus:                    │
  │    Wikipedia, books, web pages, news...       │
  │    Billions of words, diverse topics          │
  │                                               │
  │  Task: predict missing words / next word      │
  │    "The cat sat on the ___"  --> "mat"         │
  │    "Paris is the capital of ___" --> "France"  │
  │                                               │
  │  Result: general language representations     │
  │    - word relationships (synonym, antonym)    │
  │    - sentence structure (subject, verb, obj)  │
  │    - world knowledge (Paris -> France)        │
  └──────────────────────────────────────────────┘
               │
               │ pretrained representations
               ▼
Stage 2: FINE-TUNING (cheap, done per task)
  ┌──────────────────────────────────────────────┐
  │  Small labeled dataset:                       │
  │    1,000 movie reviews with sentiment labels  │
  │                                               │
  │  Task: classify positive / negative           │
  │                                               │
  │  The model already knows language.            │
  │  It just needs to learn what "sentiment" is.  │
  │                                               │
  │  Result: task-specific model                  │
  └──────────────────────────────────────────────┘


TRAINING FROM SCRATCH vs PRETRAINED + FINE-TUNED
=================================================

  From scratch (1,000 reviews):     Pretrained + fine-tuned:
  ┌──────────────────────┐          ┌──────────────────────┐
  │ Must learn:           │          │ Already knows:        │
  │  - what words mean    │          │  ✓ what words mean    │
  │  - how grammar works  │          │  ✓ how grammar works  │
  │  - what sentiment is  │          │  ✓ word relationships │
  │                       │          │                       │
  │ From only 1,000       │          │ Must learn:           │
  │ examples!             │          │  - what sentiment is  │
  │                       │          │                       │
  │ Result: poor          │          │ Result: good          │
  └──────────────────────┘          └──────────────────────┘


WHY KNOWLEDGE TRANSFERS
========================

  Pretrained on general text:
    "excellent" appears near "great", "superb", "outstanding"
    "terrible" appears near "awful", "horrible", "dreadful"

  These patterns transfer to sentiment:
    "excellent" --> positive (the model already knows it's a strong word)
    "terrible"  --> negative (the model already knows it's a negative word)

  A from-scratch model must discover these relationships
  from the small labeled dataset alone.
```

---

## Hands-on Exploration

1. The code below builds word co-occurrence representations from a general corpus, then uses them for a sentiment task. Before running it, predict: will the pretrained representations outperform random representations on the sentiment task? Why? Think about what information co-occurrence captures that random vectors miss.

2. After running the code, look at the "nearest neighbors" output for words like "excellent" and "terrible". Notice that pretrained representations group words with similar usage patterns together. Now consider: if you trained co-occurrence representations on only the 8 sentiment examples instead of the general corpus, would you get the same neighbor structure? Why not?

3. The code compares three approaches: random features, pretrained features, and pretrained + fine-tuned features. Think about a real-world scenario: you are building a customer support ticket classifier. You have 200 labeled tickets but access to millions of unlabeled support conversations. How would you apply the two-stage paradigm? What would pretraining look like? What would fine-tuning look like?

---

## Live Code

```python
# --- Pretraining vs Fine-Tuning: Transfer Learning Simulation ---
# Pure Python stdlib -- no external dependencies

import math
import random
from collections import Counter

random.seed(42)


# ============================================================
# STAGE 1: PRETRAINING -- Learn general word representations
# from a large diverse corpus using word co-occurrence
# ============================================================

print("=" * 65)
print("STAGE 1: PRETRAINING ON GENERAL CORPUS")
print("=" * 65)

# A diverse "general" corpus spanning multiple topics
general_corpus = [
    # Science
    "the experiment produced excellent results in the laboratory",
    "researchers discovered a remarkable new compound this year",
    "the study revealed surprising and wonderful findings",
    "scientists achieved outstanding progress in their field",
    "the results were terrible and the experiment failed completely",
    "poor methodology led to awful results in the study",
    "the research produced dreadful outcomes and was abandoned",
    "horrible execution ruined an otherwise promising experiment",
    # Daily life
    "the excellent restaurant served wonderful food all evening",
    "we had a remarkable experience at the outstanding hotel",
    "the superb performance received great applause from everyone",
    "the terrible service at the awful restaurant disappointed everyone",
    "it was a horrible experience with dreadful customer support",
    "poor quality products led to bad reviews from customers",
    # Nature
    "the garden produced excellent flowers in the warm spring",
    "remarkable birds with outstanding plumage visited the park",
    "the weather was terrible with awful storms all week",
    "poor conditions and bad weather ruined the outdoor event",
    # General patterns (to build co-occurrence structure)
    "this was an excellent and remarkable achievement for everyone",
    "the outstanding and superb quality impressed all visitors",
    "wonderful results came from great effort and dedication",
    "the terrible and horrible conditions caused awful problems",
    "poor planning and bad decisions led to dreadful outcomes",
    "awful management produced terrible results for the company",
]

print(f"\nGeneral corpus: {len(general_corpus)} sentences")
print("Topics: science, daily life, nature, general patterns")


def tokenize(text):
    """Lowercase and split into alphabetic tokens."""
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
    "for", "with", "that", "this", "are", "was", "be", "at", "by",
    "or", "from", "but", "not", "we", "all", "had", "has", "have",
}


def preprocess(text):
    return [t for t in tokenize(text) if t not in STOPWORDS]


# Build word co-occurrence matrix from general corpus
# (This is a simplified version of what real pretraining does)
def build_cooccurrence(sentences, window=3):
    """Count how often words appear near each other."""
    all_tokens = [preprocess(s) for s in sentences]
    vocab = sorted(set(w for tokens in all_tokens for w in tokens))
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    n = len(vocab)

    # Co-occurrence counts
    cooccur = [[0.0] * n for _ in range(n)]

    for tokens in all_tokens:
        for i, word in enumerate(tokens):
            wi = word_to_idx[word]
            for j in range(max(0, i - window), min(len(tokens), i + window + 1)):
                if i != j:
                    wj = word_to_idx[tokens[j]]
                    cooccur[wi][wj] += 1.0

    # Normalize rows to get representation vectors
    # (each word is represented by its co-occurrence distribution)
    vectors = {}
    for word in vocab:
        idx = word_to_idx[word]
        row = cooccur[idx]
        magnitude = math.sqrt(sum(x * x for x in row))
        if magnitude > 0:
            vectors[word] = [x / magnitude for x in row]
        else:
            vectors[word] = row
    return vocab, word_to_idx, vectors


vocab, word_to_idx, pretrained_vectors = build_cooccurrence(general_corpus)
print(f"Vocabulary size: {len(vocab)} words")
print(f"Representation dimension: {len(vocab)} (co-occurrence features)")


# Show what pretrained representations capture
def cosine_sim(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    m1 = math.sqrt(sum(a * a for a in v1))
    m2 = math.sqrt(sum(b * b for b in v2))
    if m1 == 0 or m2 == 0:
        return 0.0
    return dot / (m1 * m2)


def nearest_neighbors(word, vectors, top_n=5):
    """Find words with most similar co-occurrence patterns."""
    if word not in vectors:
        return []
    target = vectors[word]
    sims = []
    for other, vec in vectors.items():
        if other != word:
            sims.append((other, cosine_sim(target, vec)))
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:top_n]


print("\n--- Pretrained Nearest Neighbors ---")
print("(Words that appear in similar contexts cluster together)\n")

probe_words = ["excellent", "terrible", "results", "produced"]
for word in probe_words:
    neighbors = nearest_neighbors(word, pretrained_vectors, top_n=4)
    neighbor_str = ", ".join(f"{w} ({s:.3f})" for w, s in neighbors)
    print(f'  "{word}" --> {neighbor_str}')

print()
print("  Notice: positive words cluster with positive words,")
print("  negative words cluster with negative words.")
print("  The model learned this from co-occurrence, not from labels.")


# ============================================================
# STAGE 2: FINE-TUNING -- Adapt for sentiment classification
# ============================================================

print()
print("=" * 65)
print("STAGE 2: FINE-TUNING FOR SENTIMENT CLASSIFICATION")
print("=" * 65)

# Small labeled dataset for sentiment (this is all we have for the task)
sentiment_data = [
    ("the movie was excellent and the acting was superb", "positive"),
    ("a wonderful film with outstanding performances", "positive"),
    ("remarkable storytelling and great cinematography", "positive"),
    ("truly an excellent and remarkable piece of work", "positive"),
    ("the movie was terrible and the plot was awful", "negative"),
    ("horrible acting with dreadful dialogue throughout", "negative"),
    ("poor direction led to bad pacing and awful scenes", "negative"),
    ("a terrible waste of time with horrible characters", "negative"),
]

print(f"\nLabeled sentiment data: {len(sentiment_data)} examples")
for text, label in sentiment_data:
    print(f"  [{label:>8s}] {text}")


def document_vector(text, vectors):
    """Average word vectors to get a document representation."""
    tokens = preprocess(text)
    dim = len(next(iter(vectors.values())))
    doc_vec = [0.0] * dim
    count = 0
    for token in tokens:
        if token in vectors:
            for i in range(dim):
                doc_vec[i] += vectors[token][i]
            count += 1
    if count > 0:
        doc_vec = [x / count for x in doc_vec]
    return doc_vec


def simple_classifier(train_data, vectors, test_text):
    """
    Classify by comparing test document to average positive
    and average negative document vectors.
    This is a nearest-centroid classifier.
    """
    dim = len(next(iter(vectors.values())))
    pos_centroid = [0.0] * dim
    neg_centroid = [0.0] * dim
    pos_count = 0
    neg_count = 0

    for text, label in train_data:
        vec = document_vector(text, vectors)
        if label == "positive":
            for i in range(dim):
                pos_centroid[i] += vec[i]
            pos_count += 1
        else:
            for i in range(dim):
                neg_centroid[i] += vec[i]
            neg_count += 1

    if pos_count > 0:
        pos_centroid = [x / pos_count for x in pos_centroid]
    if neg_count > 0:
        neg_centroid = [x / neg_count for x in neg_centroid]

    test_vec = document_vector(test_text, vectors)
    pos_sim = cosine_sim(test_vec, pos_centroid)
    neg_sim = cosine_sim(test_vec, neg_centroid)

    return "positive" if pos_sim > neg_sim else "negative", pos_sim, neg_sim


# ============================================================
# COMPARISON: Random vs Pretrained vs Fine-Tuned
# ============================================================

print()
print("=" * 65)
print("COMPARISON: RANDOM vs PRETRAINED vs FINE-TUNED")
print("=" * 65)

# Generate random representations (no pretraining)
dim = len(vocab)
random_vectors = {}
for word in vocab:
    vec = [random.gauss(0, 1) for _ in range(dim)]
    magnitude = math.sqrt(sum(x * x for x in vec))
    random_vectors[word] = [x / magnitude for x in vec] if magnitude > 0 else vec

# Test sentences the model has NOT seen during training
test_sentences = [
    ("an excellent performance with remarkable skill", "positive"),
    ("superb quality and outstanding craftsmanship", "positive"),
    ("great results from wonderful teamwork", "positive"),
    ("terrible quality with awful workmanship", "negative"),
    ("horrible results from poor planning", "negative"),
    ("dreadful experience and bad service overall", "negative"),
]

print(f"\nTest sentences: {len(test_sentences)} (unseen during training)")
print()

# Evaluate each approach
approaches = [
    ("Random features", random_vectors),
    ("Pretrained features", pretrained_vectors),
]

approach_names = []
approach_accuracies = []

for approach_name, vectors in approaches:
    correct = 0
    print(f"--- {approach_name} ---")

    for text, true_label in test_sentences:
        pred, pos_s, neg_s = simple_classifier(sentiment_data, vectors, text)
        match = "OK" if pred == true_label else "WRONG"
        correct += 1 if pred == true_label else 0
        print(f"  [{match:>5s}] pred={pred:>8s} true={true_label:>8s}  "
              f"pos={pos_s:+.3f} neg={neg_s:+.3f}  "
              f"{text[:45]}...")

    accuracy = correct / len(test_sentences) * 100
    approach_names.append(approach_name)
    approach_accuracies.append(round(accuracy, 1))
    print(f"  Accuracy: {correct}/{len(test_sentences)} = {accuracy:.0f}%\n")

# --- Visualization: Random vs Pretrained Accuracy ---
show_chart({
    "type": "bar",
    "title": "Sentiment Classification Accuracy: Random vs Pretrained",
    "labels": approach_names,
    "datasets": [{"label": "Accuracy (%)", "data": approach_accuracies, "color": "#3b82f6"}],
    "options": {"x_label": "Approach", "y_label": "Accuracy (%)"}
})

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#ef4444", "#10b981"]
    ax.bar(approach_names, approach_accuracies, color=colors[:len(approach_names)])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Sentiment Classification: Random vs Pretrained")
    ax.set_ylim(0, 110)
    for i, v in enumerate(approach_accuracies):
        ax.text(i, v + 2, f"{v:.0f}%", ha="center", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("random_vs_pretrained_accuracy.png", dpi=100)
    plt.close()
    print("  [Saved matplotlib chart: random_vs_pretrained_accuracy.png]")
except ImportError:
    pass


# ============================================================
# DEMONSTRATE TRANSFER ACROSS DOMAINS
# ============================================================

print("=" * 65)
print("TRANSFER LEARNING: NEW DOMAIN, NO NEW TRAINING")
print("=" * 65)

# The general corpus was about science, restaurants, nature.
# Can the pretrained representations help with PRODUCT REVIEWS?
# (a domain never seen during pretraining or fine-tuning)

product_reviews = [
    ("the product quality was excellent and delivery was outstanding",
     "positive"),
    ("remarkable value with superb build quality overall",
     "positive"),
    ("terrible product with awful packaging and poor design",
     "negative"),
    ("horrible customer service and dreadful return policy",
     "negative"),
]

print("\nDomain: Product reviews (never seen during pretraining)")
print("Using sentiment model trained on movie review data.\n")

for approach_name, vectors in approaches:
    correct = 0
    print(f"--- {approach_name} ---")

    for text, true_label in product_reviews:
        pred, pos_s, neg_s = simple_classifier(sentiment_data, vectors, text)
        match = "OK" if pred == true_label else "WRONG"
        correct += 1 if pred == true_label else 0
        print(f"  [{match:>5s}] pred={pred:>8s}  {text[:50]}...")

    accuracy = correct / len(product_reviews) * 100
    print(f"  Accuracy: {correct}/{len(product_reviews)} = {accuracy:.0f}%\n")

print("The pretrained representations transfer across domains because")
print("they encode general language patterns (\"excellent\" is positive,")
print("\"terrible\" is negative) -- not domain-specific rules.")
print()
print("This is the core insight of the pretrain-then-fine-tune paradigm:")
print("learn language structure once, apply it to many tasks.")


# ============================================================
# WHAT PRETRAINING ACTUALLY CAPTURES
# ============================================================

print()
print("=" * 65)
print("WHAT PRETRAINING CAPTURES (REPRESENTATION ANALYSIS)")
print("=" * 65)

# Show the similarity structure that pretraining creates
word_groups = {
    "positive": ["excellent", "remarkable", "outstanding", "wonderful", "superb",
                 "great"],
    "negative": ["terrible", "awful", "horrible", "dreadful", "poor", "bad"],
}

print("\nWithin-group vs between-group similarity:")
print("(Higher within-group = better representation)\n")

for group_name, words in word_groups.items():
    present = [w for w in words if w in pretrained_vectors]
    if len(present) < 2:
        continue
    sims = []
    for i in range(len(present)):
        for j in range(i + 1, len(present)):
            s = cosine_sim(pretrained_vectors[present[i]],
                           pretrained_vectors[present[j]])
            sims.append(s)
    avg_sim = sum(sims) / len(sims) if sims else 0
    print(f"  {group_name:>8s} words avg similarity: {avg_sim:.4f}")
    print(f"           words: {', '.join(present)}")

# Cross-group similarity
cross_sims = []
pos_words = [w for w in word_groups["positive"] if w in pretrained_vectors]
neg_words = [w for w in word_groups["negative"] if w in pretrained_vectors]
for pw in pos_words:
    for nw in neg_words:
        cross_sims.append(cosine_sim(pretrained_vectors[pw],
                                     pretrained_vectors[nw]))
avg_cross = sum(cross_sims) / len(cross_sims) if cross_sims else 0
print(f"\n  positive vs negative avg similarity: {avg_cross:.4f}")
print()
print("  Within-group similarity > between-group similarity")
print("  --> Pretraining learned that positive words behave alike")
print("      and negative words behave alike, purely from")
print("      co-occurrence patterns. No labels were used.")

# --- Visualization: Within-group vs Between-group Similarity ---
# Collect the group averages for visualization
sim_group_labels = []
sim_group_values = []
sim_group_colors = []
for group_name, words in word_groups.items():
    present = [w for w in words if w in pretrained_vectors]
    if len(present) < 2:
        continue
    sims_inner = []
    for i in range(len(present)):
        for j in range(i + 1, len(present)):
            s = cosine_sim(pretrained_vectors[present[i]],
                           pretrained_vectors[present[j]])
            sims_inner.append(s)
    avg_inner = sum(sims_inner) / len(sims_inner) if sims_inner else 0
    sim_group_labels.append(f"{group_name} (within)")
    sim_group_values.append(round(avg_inner, 4))
    sim_group_colors.append("#10b981" if group_name == "positive" else "#ef4444")

sim_group_labels.append("pos vs neg (between)")
sim_group_values.append(round(avg_cross, 4))
sim_group_colors.append("#f59e0b")

show_chart({
    "type": "bar",
    "title": "Pretrained Representation: Within-Group vs Between-Group Similarity",
    "labels": sim_group_labels,
    "datasets": [{"label": "Avg Cosine Similarity", "data": sim_group_values, "color": "#3b82f6"}],
    "options": {"x_label": "Group Comparison", "y_label": "Average Cosine Similarity"}
})

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(sim_group_labels, sim_group_values, color=sim_group_colors)
    ax.set_ylabel("Average Cosine Similarity")
    ax.set_title("Within-Group vs Between-Group Similarity")
    for i, v in enumerate(sim_group_values):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig("within_vs_between_similarity.png", dpi=100)
    plt.close()
    print("  [Saved matplotlib chart: within_vs_between_similarity.png]")
except ImportError:
    pass

print()
print("Modern NLP stands on decades of classical foundations.")
print("Pretraining vs fine-tuning is an engineering strategy,")
print("not magic -- it reuses expensive computation efficiently.")
```

---

## Key Takeaways

- **Pretraining amortizes the cost of learning language structure.** Instead of learning what words mean from scratch for every task, a pretrained model captures general language patterns once and reuses them everywhere. This is an engineering efficiency, not artificial understanding.
- **Fine-tuning adapts general knowledge to specific tasks.** A small labeled dataset is enough when the model already knows language structure. Without pretraining, the same small dataset produces poor representations because there is not enough data to learn both language and the task.
- **Transfer learning works because language patterns are shared across domains.** The fact that "excellent" co-occurs with positive contexts in Wikipedia also holds in movie reviews and product descriptions. Pretrained representations encode these universal patterns.
- **The two-stage paradigm is a strategy, not a breakthrough in understanding.** Models do not "understand" language after pretraining. They have learned statistical regularities in word co-occurrence. Fine-tuning steers these regularities toward a task objective. The effectiveness comes from scale and reuse, not from comprehension.
