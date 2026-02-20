# Where LLMs Fit in the NLP Stack

> Phase 10 — Modern NLP Pipelines (Awareness) | Kata 10.3

---

## Concept & Intuition

### What problem are we solving?

Large Language Models (LLMs) dominate today's NLP landscape, but they did not appear from nothing. Every component of an LLM -- tokenization, embeddings, attention, pretraining -- was developed incrementally over decades. Without understanding the stack beneath them, LLMs are black boxes. With that understanding, they become predictable engineering artifacts whose strengths and limitations follow logically from their design.

This kata maps the entire NLP journey from raw text to modern LLMs, showing exactly where each technique fits, what it contributes, and what it cannot do. The goal is not to hype LLMs but to place them accurately on a continuum of increasingly sophisticated text representations.

### Why earlier methods fail

They do not fail universally. They fail at specific things:

- **BoW/TF-IDF** fail at capturing word order and meaning. "Dog bites man" and "Man bites dog" are identical under BoW. But they succeed at fast, interpretable document comparison.
- **Static embeddings** (Word2Vec) fail at polysemy. The word "bank" gets one vector regardless of context. But they succeed at capturing word relationships (king - man + woman = queen).
- **RNNs** fail at long-range dependencies and parallel training. But they succeed at modeling short sequences.
- **Basic transformers** fail at tasks requiring world knowledge not present in the input. But they succeed at capturing contextual relationships through attention.

Each technique has a specific failure mode that motivates the next. LLMs are the current endpoint of this chain -- not because they solve everything, but because they address the limitations of each predecessor. And they introduce their own limitations: computational cost, hallucination, opacity, and the inability to truly reason about novel situations.

### Mental models

- **The NLP stack is like a building.** Each floor depends on the floors below it. LLMs sit on the top floor, but they contain versions of every technique from every floor beneath them. Remove tokenization, and the building collapses. Remove attention, and it collapses. Every layer matters.
- **Each technique is a representation choice.** BoW represents text as word counts. TF-IDF represents text as weighted word importance. Embeddings represent words as dense vectors. Transformers represent tokens as context-dependent vectors. LLMs represent text as probability distributions over next tokens. The journey is one of increasingly rich representations.
- **LLMs are not magic -- they are scale.** Take a transformer, train it on more data with more parameters, and you get an LLM. The architecture is the same. The training objective (predict the next token) is the same. The difference is scale -- and scale produces emergent capabilities that smaller models lack.
- **Classical methods are still the right choice for many problems.** TF-IDF search is fast, interpretable, and works perfectly for keyword-based retrieval. Using an LLM for that task is like using a crane to pick up a pencil.

### Visual explanations

```
THE NLP STACK: FROM RAW TEXT TO LLMs
======================================

  Layer 10:  LLMs (GPT, etc.)
             Scale + pretraining + fine-tuning
             Can: generate, classify, translate, summarize, converse
             Cannot: guarantee factual accuracy, truly reason, explain itself
             ▲
  Layer 9:   Transformer Architecture
             Self-attention replaces recurrence
             Can: capture long-range dependencies in parallel
             Cannot: work without tokenization and embeddings below
             ▲
  Layer 8:   Context & Sequence Modeling
             Word meaning depends on surrounding words
             Can: distinguish "bank" (river) from "bank" (financial)
             Cannot: scale efficiently with RNNs (vanishing gradients)
             ▲
  Layer 7:   Neural Embeddings
             Dense vector representations learned from data
             Can: capture word similarity (king - man + woman = queen)
             Cannot: handle polysemy (one vector per word)
             ▲
  Layer 6:   Named Entity Recognition
             Extract structured information from text
             Can: find persons, locations, organizations
             Cannot: understand relationships between entities
             ▲
  Layer 5:   Tokenization (Deep Dive)
             Break text into model-friendly units (BPE, subwords)
             Can: handle any word, including unseen ones
             Cannot: capture meaning (purely mechanical splitting)
             ▲
  Layer 4:   Similarity & Classical Tasks
             Cosine similarity, search, clustering
             Can: find similar documents, group by topic
             Cannot: understand synonyms or paraphrases
             ▲
  Layer 3:   TF-IDF
             Weight words by importance (rare = important)
             Can: rank relevant documents, suppress common words
             Cannot: capture word order or word meaning
             ▲
  Layer 2:   Bag of Words
             First numerical representation of text
             Can: convert text to vectors, enable computation
             Cannot: distinguish "dog bites man" from "man bites dog"
             ▲
  Layer 1:   Text Preprocessing
             Lowercasing, stemming, stopwords, normalization
             Can: reduce noise, normalize surface variation
             Cannot: resolve ambiguity or capture meaning
             ▲
  Layer 0:   Language & Text (Foundations)
             Understanding raw material: ambiguity, noise, structure
             Can: frame the problem correctly
             Cannot: compute anything (this is where understanding starts)


THE CAPABILITY PYRAMID
=======================

  What each technique CAN do:

                    ┌───────────────┐
                    │     LLMs      │  Generate, classify, translate,
                    │               │  converse, summarize, few-shot learn
                  ┌─┴───────────────┴─┐
                  │   Transformers     │  Long-range context, parallel
                  │                    │  training, attention patterns
                ┌─┴────────────────────┴─┐
                │   Neural Embeddings     │  Word similarity, analogy,
                │                         │  dense representations
              ┌─┴─────────────────────────┴─┐
              │   TF-IDF / Classical NLP     │  Document similarity,
              │                              │  search, clustering
            ┌─┴──────────────────────────────┴─┐
            │   BoW + Preprocessing              │  Text-to-numbers,
            │                                    │  basic statistics
          ┌─┴────────────────────────────────────┴─┐
          │   Raw Text + Language Understanding      │  Problem framing,
          │                                          │  linguistic intuition
          └──────────────────────────────────────────┘

  Each layer INCLUDES everything below it.
  LLMs use tokenization, embeddings, attention -- ALL of it.


DECISION TREE: WHEN TO USE WHAT
=================================

  Start here: What is your task?
  │
  ├── Keyword search / document retrieval?
  │   └── Use TF-IDF.  Fast, interpretable, works well.
  │
  ├── Document classification with labeled data?
  │   ├── Thousands of labeled examples?
  │   │   └── TF-IDF + simple classifier.  Often sufficient.
  │   └── Few labeled examples (< 100)?
  │       └── Pretrained model + fine-tuning.  Leverages transfer.
  │
  ├── Named entity extraction?
  │   ├── Known entity patterns?
  │   │   └── Rule-based NER.  Fast, predictable.
  │   └── Diverse entity types?
  │       └── Pretrained encoder model + fine-tuning.
  │
  ├── Text generation / completion?
  │   └── Decoder-only model (LLM).  This is what they are built for.
  │
  ├── Semantic similarity (paraphrase detection)?
  │   └── Pretrained encoder model.  Captures meaning beyond keywords.
  │
  └── Do you need explainability?
      ├── Yes --> Classical methods (TF-IDF, rules).  Transparent.
      └── No  --> Neural / LLM.  Better performance, less transparency.
```

---

## Hands-on Exploration

1. The code below implements a simplified version of each technique from Phase 0 through Phase 10 and runs them all on the same corpus. Before running it, predict: which technique will produce the highest-quality document similarity rankings? Which will be fastest? After running, check your predictions. Notice that "best" depends on what you are optimizing for.

2. Look at the capability comparison table in the output. For each technique, identify one real-world task where it would be the best choice and one where it would be the worst choice. The point is not that newer techniques are always better -- it is that different techniques have different strengths.

3. The code includes a "decision tree" exercise at the end. Given three new tasks, it walks through the decision logic. Try adding your own task to the list. Trace through the decision tree in the ASCII diagram above and verify that the code's recommendation matches your manual trace.

---

## Live Code

```python
# --- Where LLMs Fit in the NLP Stack ---
# Pure Python stdlib -- no external dependencies
# Demonstrates every technique from Phase 0 to Phase 10

import math
import random
from collections import Counter, defaultdict

random.seed(42)


# ============================================================
# SHARED CORPUS AND UTILITIES
# ============================================================

corpus = [
    "the cat sat on the warm mat near the window",
    "a dog played fetch in the sunny park today",
    "the cat chased a quick mouse across the yard",
    "machine learning models process large datasets efficiently",
    "neural networks learn representations from training data",
    "deep learning requires significant computational resources",
    "the stock market experienced volatile trading sessions today",
    "investors analyzed quarterly earnings reports carefully",
    "economic indicators suggest moderate growth this quarter",
    "the chef prepared a delicious meal with fresh ingredients",
    "baking sourdough bread requires patience and warm water",
    "the restaurant menu featured seasonal dishes and local produce",
]

doc_labels = [
    "animals", "animals", "animals",
    "technology", "technology", "technology",
    "finance", "finance", "finance",
    "cooking", "cooking", "cooking",
]

print("=" * 65)
print("THE NLP STACK: EVERY TECHNIQUE ON THE SAME CORPUS")
print("=" * 65)
print(f"\nCorpus: {len(corpus)} documents across 4 topics")
for i, (doc, label) in enumerate(zip(corpus, doc_labels)):
    print(f"  D{i:02d} [{label:>10s}]: {doc}")


# --- Shared utilities ---

STOPWORDS = {
    "a", "an", "the", "is", "it", "in", "on", "of", "to", "and",
    "for", "with", "that", "this", "are", "was", "be", "at", "by",
    "or", "from", "but", "not", "we", "they", "their", "its",
}


def tokenize(text):
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


def preprocess(text):
    return [t for t in tokenize(text) if t not in STOPWORDS]


def cosine_sim(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    m1 = math.sqrt(sum(a * a for a in v1))
    m2 = math.sqrt(sum(b * b for b in v2))
    if m1 == 0 or m2 == 0:
        return 0.0
    return dot / (m1 * m2)


# ============================================================
# PHASE 0-1: RAW TEXT + PREPROCESSING
# ============================================================

print()
print("=" * 65)
print("PHASE 0-1: RAW TEXT + PREPROCESSING")
print("=" * 65)

doc_tokens = [preprocess(doc) for doc in corpus]

print(f"\nRaw:         \"{corpus[0]}\"")
print(f"Preprocessed: {doc_tokens[0]}")
print(f"\nPreprocessing removes noise (stopwords, case) but")
print(f"preserves content words. This is the foundation.")


# ============================================================
# PHASE 2: BAG OF WORDS
# ============================================================

print()
print("=" * 65)
print("PHASE 2: BAG OF WORDS")
print("=" * 65)

all_words = sorted(set(w for tokens in doc_tokens for w in tokens))
word_to_idx = {w: i for i, w in enumerate(all_words)}


def bow_vector(tokens):
    vec = [0] * len(all_words)
    for t in tokens:
        if t in word_to_idx:
            vec[word_to_idx[t]] += 1
    return vec


bow_vectors = [bow_vector(tokens) for tokens in doc_tokens]

print(f"\nVocabulary: {len(all_words)} words")
print(f"Vector dimension: {len(all_words)}")
print(f"Sparsity: most entries are 0")
print(f"\nBoW similarity (D00 cat vs D01 dog):  {cosine_sim(bow_vectors[0], bow_vectors[1]):.4f}")
print(f"BoW similarity (D00 cat vs D03 tech): {cosine_sim(bow_vectors[0], bow_vectors[3]):.4f}")
print(f"\nBoW captures word presence but ignores order and importance.")


# ============================================================
# PHASE 3: TF-IDF
# ============================================================

print()
print("=" * 65)
print("PHASE 3: TF-IDF")
print("=" * 65)

n_docs = len(doc_tokens)
n_words = len(all_words)

# Compute IDF
df = [0] * n_words
for tokens in doc_tokens:
    seen = set(tokens)
    for w in seen:
        df[word_to_idx[w]] += 1

idf = [0.0] * n_words
for i in range(n_words):
    if df[i] > 0:
        idf[i] = math.log(n_docs / df[i])


def tfidf_vector(tokens):
    tf = Counter(tokens)
    total = len(tokens) if tokens else 1
    vec = [0.0] * n_words
    for w, count in tf.items():
        if w in word_to_idx:
            idx = word_to_idx[w]
            vec[idx] = (count / total) * idf[idx]
    return vec


tfidf_vectors = [tfidf_vector(tokens) for tokens in doc_tokens]

print(f"\nTF-IDF similarity (D00 cat vs D02 cat):  {cosine_sim(tfidf_vectors[0], tfidf_vectors[2]):.4f}")
print(f"TF-IDF similarity (D00 cat vs D03 tech): {cosine_sim(tfidf_vectors[0], tfidf_vectors[3]):.4f}")
print(f"\nTF-IDF downweights common words, making topic-specific")
print(f"words drive similarity. An improvement over raw BoW.")


# ============================================================
# PHASE 4: SIMILARITY & SEARCH
# ============================================================

print()
print("=" * 65)
print("PHASE 4: SIMILARITY-BASED SEARCH")
print("=" * 65)

query = "cat chased mouse"
query_tokens = preprocess(query)
query_vec = tfidf_vector(query_tokens)

print(f'\nQuery: "{query}"')
print(f"Ranking documents by TF-IDF cosine similarity:\n")

rankings = [(i, cosine_sim(query_vec, tfidf_vectors[i])) for i in range(n_docs)]
rankings.sort(key=lambda x: x[1], reverse=True)

for rank, (idx, sim) in enumerate(rankings[:5], 1):
    print(f"  #{rank}: D{idx:02d} [{doc_labels[idx]:>10s}] sim={sim:.4f}  {corpus[idx][:50]}...")

print(f"\nSearch works by measuring distance in vector space.")
print(f"No \"understanding\" -- just geometric proximity.")


# ============================================================
# PHASE 5-6: TOKENIZATION & NER (Conceptual)
# ============================================================

print()
print("=" * 65)
print("PHASE 5-6: TOKENIZATION & NER (CONCEPTUAL)")
print("=" * 65)

# Demonstrate subword tokenization concept
print("\nSubword tokenization (BPE-style concept):")
example = "unbelievably"
print(f'  Word: "{example}"')
print(f'  Word-level:    ["{example}"]')
print(f'  Character:     {[ch for ch in example]}')
print(f'  Subword (sim): ["un", "believ", "ably"]')
print(f"\n  Subword tokenization balances vocabulary size with coverage.")
print(f"  LLMs use this -- it is a foundation layer, not optional.")

# Demonstrate NER concept
print(f"\nNamed Entity Recognition (rule-based concept):")
ner_text = "the chef prepared a delicious meal"
ner_tokens = tokenize(ner_text)
# Simple rule: words that appear as subjects in cooking context
entity_rules = {"chef": "ROLE", "meal": "FOOD"}
print(f'  Text: "{ner_text}"')
for token in ner_tokens:
    if token in entity_rules:
        print(f'  Found: "{token}" -> {entity_rules[token]}')
print(f"  Rule-based NER is fast but brittle. Neural NER is flexible but opaque.")


# ============================================================
# PHASE 7-8: NEURAL EMBEDDINGS & CONTEXT
# ============================================================

print()
print("=" * 65)
print("PHASE 7-8: NEURAL EMBEDDINGS & CONTEXT")
print("=" * 65)

# Simulate word embeddings via co-occurrence (simplified)
def build_embeddings(sentences, dim=20, window=2):
    """Build simple dense embeddings from co-occurrence + dimensionality reduction."""
    all_tok = [preprocess(s) for s in sentences]
    vocab_set = sorted(set(w for tokens in all_tok for w in tokens))
    w2i = {w: i for i, w in enumerate(vocab_set)}
    n = len(vocab_set)

    # Co-occurrence matrix
    cooccur = [[0.0] * n for _ in range(n)]
    for tokens in all_tok:
        for i, word in enumerate(tokens):
            wi = w2i[word]
            for j in range(max(0, i - window), min(len(tokens), i + window + 1)):
                if i != j:
                    wj = w2i[tokens[j]]
                    cooccur[wi][wj] += 1.0

    # Simple dimensionality reduction: random projection
    # (In real systems this would be SVD or neural training)
    random.seed(42)
    projection = [[random.gauss(0, 1) for _ in range(dim)] for _ in range(n)]

    embeddings = {}
    for word in vocab_set:
        wi = w2i[word]
        row = cooccur[wi]
        emb = [0.0] * dim
        for j in range(n):
            if row[j] > 0:
                for d in range(dim):
                    emb[d] += row[j] * projection[j][d]
        # Normalize
        mag = math.sqrt(sum(x * x for x in emb))
        if mag > 0:
            emb = [x / mag for x in emb]
        embeddings[word] = emb

    return embeddings


embeddings = build_embeddings(corpus, dim=20)

print(f"\nEmbedding dimension: 20 (dense, vs {len(all_words)} for BoW)")
print(f"Words with embeddings: {len(embeddings)}")

# Show word similarities with embeddings
print(f"\nWord similarities (embedding-based):")
word_pairs = [("cat", "dog"), ("cat", "market"), ("learning", "networks"),
              ("chef", "bread"), ("stock", "investors")]
for w1, w2 in word_pairs:
    if w1 in embeddings and w2 in embeddings:
        sim = cosine_sim(embeddings[w1], embeddings[w2])
        print(f'  "{w1}" vs "{w2}": {sim:+.4f}')

print(f"\nDense embeddings capture relationships that TF-IDF cannot:")
print(f"related words have similar vectors even if they never co-occur")
print(f"in the same document.")


# ============================================================
# PHASE 9-10: ATTENTION & LLM CONCEPTS
# ============================================================

print()
print("=" * 65)
print("PHASE 9-10: ATTENTION & LLM CONCEPTS")
print("=" * 65)

# Simulate simple attention: context-dependent word representation
def compute_attention_weights(tokens, embeddings):
    """
    Simplified self-attention: each word attends to all others
    based on embedding similarity (dot product).
    """
    n = len(tokens)
    vecs = []
    for t in tokens:
        if t in embeddings:
            vecs.append(embeddings[t])
        else:
            vecs.append([0.0] * 20)

    # Compute attention scores (dot product)
    scores = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            scores[i][j] = sum(a * b for a, b in zip(vecs[i], vecs[j]))

        # Softmax
        max_s = max(scores[i])
        exp_scores = [math.exp(s - max_s) for s in scores[i]]
        total = sum(exp_scores)
        if total > 0:
            scores[i] = [e / total for e in exp_scores]

    return scores


example = "cat sat on warm mat"
ex_tokens = preprocess(example)
if len(ex_tokens) > 1:
    attn = compute_attention_weights(ex_tokens, embeddings)

    print(f'\nSimulated self-attention for: "{example}"')
    print(f"\nAttention weights (which words attend to which):\n")

    # Header
    print(f"  {'':>10s}", end="")
    for t in ex_tokens:
        print(f"{t:>8s}", end="")
    print()

    for i, t in enumerate(ex_tokens):
        print(f"  {t:>10s}", end="")
        for j in range(len(ex_tokens)):
            weight = attn[i][j]
            print(f"  {weight:.3f}", end="")
        print()

    print(f"\n  Each row shows how much one word \"attends to\" each other word.")
    print(f"  Higher weight = more influence on the word's representation.")
    print(f"  This is the core mechanism of transformers and LLMs.")


# ============================================================
# THE FULL COMPARISON TABLE
# ============================================================

print()
print("=" * 65)
print("CAPABILITY COMPARISON: CLASSICAL TO MODERN")
print("=" * 65)

comparison = [
    ("BoW",
     "Word counts",
     "Document comparison",
     "No word order, no meaning",
     "microseconds"),
    ("TF-IDF",
     "Weighted word counts",
     "Search, classification",
     "No word order, no synonyms",
     "microseconds"),
    ("Static Embeddings",
     "Dense word vectors",
     "Word similarity, analogy",
     "No polysemy (one vec/word)",
     "milliseconds"),
    ("RNN/LSTM",
     "Sequential hidden states",
     "Sequence labeling",
     "Slow training, short memory",
     "seconds"),
    ("Transformer",
     "Context-dependent vectors",
     "All NLP tasks",
     "Quadratic attention cost",
     "seconds-minutes"),
    ("LLM",
     "Pretrained transformer",
     "Generation, few-shot tasks",
     "Expensive, hallucinations",
     "seconds-minutes"),
]

print()
print(f"  {'Technique':<20s} {'Representation':<25s} {'Speed':>15s}")
print(f"  {'-'*20} {'-'*25} {'-'*15}")
for name, rep, good, bad, speed in comparison:
    print(f"  {name:<20s} {rep:<25s} {speed:>15s}")

print()
print(f"  {'Technique':<20s} {'Good at':<28s} {'Limitation'}")
print(f"  {'-'*20} {'-'*28} {'-'*30}")
for name, rep, good, bad, speed in comparison:
    print(f"  {name:<20s} {good:<28s} {bad}")

print()
print("  Key insight: each technique is a different REPRESENTATION CHOICE.")
print("  Newer is not always better -- it depends on the task and constraints.")

# --- Visualization: Capability Comparison ---
# Assign a numeric capability score (1-6 scale) based on technique sophistication
capability_names = [name for name, rep, good, bad, speed in comparison]
# Score each technique on multiple dimensions for a grouped bar chart
# Scores: 1=minimal, 2=basic, 3=moderate, 4=good, 5=excellent
cap_word_meaning = [1, 1, 3, 4, 5, 5]    # Understanding word meaning
cap_word_order =   [1, 1, 1, 4, 5, 5]    # Handling word order
cap_speed =        [5, 5, 4, 2, 2, 2]    # Inference speed
cap_interpretability = [5, 5, 3, 2, 1, 1] # Interpretability

show_chart({
    "type": "bar",
    "title": "NLP Technique Capabilities (1=minimal, 5=excellent)",
    "labels": capability_names,
    "datasets": [
        {"label": "Word Meaning", "data": cap_word_meaning, "color": "#3b82f6"},
        {"label": "Word Order", "data": cap_word_order, "color": "#10b981"},
        {"label": "Speed", "data": cap_speed, "color": "#f59e0b"},
        {"label": "Interpretability", "data": cap_interpretability, "color": "#8b5cf6"},
    ],
    "options": {"x_label": "Technique", "y_label": "Score (1-5)"}
})

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))
    x = list(range(len(capability_names)))
    width = 0.18
    ax.bar([i - 1.5*width for i in x], cap_word_meaning, width, label="Word Meaning", color="#3b82f6")
    ax.bar([i - 0.5*width for i in x], cap_word_order, width, label="Word Order", color="#10b981")
    ax.bar([i + 0.5*width for i in x], cap_speed, width, label="Speed", color="#f59e0b")
    ax.bar([i + 1.5*width for i in x], cap_interpretability, width, label="Interpretability", color="#8b5cf6")
    ax.set_xlabel("Technique")
    ax.set_ylabel("Score (1-5)")
    ax.set_title("NLP Technique Capabilities")
    ax.set_xticks(x)
    ax.set_xticklabels(capability_names, rotation=30, ha="right")
    ax.legend()
    ax.set_ylim(0, 6)
    plt.tight_layout()
    plt.savefig("capability_comparison.png", dpi=100)
    plt.close()
    print("  [Saved matplotlib chart: capability_comparison.png]")
except ImportError:
    pass


# ============================================================
# PRACTICAL DECISION EXERCISE
# ============================================================

print()
print("=" * 65)
print("DECISION EXERCISE: CHOOSE THE RIGHT TOOL")
print("=" * 65)

decisions = [
    {
        "task": "Search 10,000 product descriptions by keyword",
        "requirements": ["fast response", "keyword matching", "interpretable"],
        "recommendation": "TF-IDF",
        "reason": "Fast, interpretable, excellent for keyword-based retrieval. "
                  "No need for neural models when keywords are sufficient.",
    },
    {
        "task": "Classify customer support tickets into 5 categories",
        "requirements": ["1,000 labeled examples", "high accuracy", "fast inference"],
        "recommendation": "TF-IDF + classifier (or pretrained encoder if accuracy critical)",
        "reason": "With 1,000 labeled examples, TF-IDF + logistic regression often "
                  "matches neural models. Try simple first.",
    },
    {
        "task": "Generate product descriptions from bullet points",
        "requirements": ["fluent text", "creative", "variable length"],
        "recommendation": "Decoder-only LLM",
        "reason": "Text generation requires autoregressive models. Classical methods "
                  "cannot generate fluent text. This is where LLMs excel.",
    },
    {
        "task": "Detect person and organization names in legal documents",
        "requirements": ["high precision", "domain-specific", "auditable"],
        "recommendation": "Pretrained encoder + fine-tuning (with rule-based fallback)",
        "reason": "NER benefits from bidirectional context (encoder). Fine-tuning on "
                  "legal text adapts to domain. Rules add auditability.",
    },
    {
        "task": "Find duplicate support tickets",
        "requirements": ["semantic matching", "beyond keyword overlap"],
        "recommendation": "Pretrained encoder embeddings + cosine similarity",
        "reason": "Paraphrases share meaning but not keywords. Encoder embeddings "
                  "capture semantic similarity that TF-IDF misses.",
    },
]

for i, d in enumerate(decisions, 1):
    print(f"\n  Task {i}: {d['task']}")
    print(f"  Requirements: {', '.join(d['requirements'])}")
    print(f"  --> Recommendation: {d['recommendation']}")
    print(f"      Why: {d['reason']}")


# ============================================================
# THE NLP JOURNEY: A REPRESENTATION PERSPECTIVE
# ============================================================

print()
print("=" * 65)
print("THE NLP JOURNEY: A REPRESENTATION PERSPECTIVE")
print("=" * 65)

journey = [
    ("Phase 0",  "Raw Text",
     "Symbolic characters",
     "We cannot compute on raw text."),
    ("Phase 1",  "Preprocessing",
     "Cleaned tokens",
     "Tokens are discrete. We need numbers."),
    ("Phase 2",  "Bag of Words",
     "Sparse count vectors",
     "Counts ignore word importance."),
    ("Phase 3",  "TF-IDF",
     "Sparse weighted vectors",
     "Weights ignore word meaning and order."),
    ("Phase 4",  "Similarity Tasks",
     "Cosine distance in TF-IDF space",
     "Still no word meaning or synonyms."),
    ("Phase 5",  "Tokenization",
     "Subword units",
     "Better input units, but still no meaning."),
    ("Phase 6",  "NER",
     "Token labels (B-PER, I-ORG)",
     "Labels depend on context we do not model well."),
    ("Phase 7",  "Embeddings",
     "Dense word vectors",
     "One vector per word -- no polysemy."),
    ("Phase 8",  "Context Models",
     "Context-dependent vectors",
     "RNNs struggle with long sequences."),
    ("Phase 9",  "Transformers",
     "Attention-based representations",
     "Need massive data to learn well."),
    ("Phase 10", "LLMs",
     "Pretrained attention at scale",
     "Expensive, opaque, can hallucinate."),
]

print(f"\n  {'Phase':<10s} {'Technique':<18s} {'Representation':<30s} {'Limitation'}")
print(f"  {'-'*10} {'-'*18} {'-'*30} {'-'*40}")
for phase, technique, representation, limitation in journey:
    print(f"  {phase:<10s} {technique:<18s} {representation:<30s} {limitation}")

print()
print("  Each phase solves the limitation of the previous one.")
print("  Each phase introduces a new limitation of its own.")
print("  LLMs are not the end of this journey -- they are the current point.")
print()
print("  Understanding these foundations makes you a better user of")
print("  modern tools. You know what they can do, what they cannot do,")
print("  and why they work the way they do.")
print()
print("  Modern NLP stands on decades of classical foundations.")
print("  LLMs did not replace this stack -- they absorbed it.")
```

---

## Key Takeaways

- **LLMs are the top of a stack, not a replacement for it.** Every LLM uses tokenization (Phase 5), embeddings (Phase 7), attention (Phase 9), and pretraining (Phase 10). Remove any layer and the system fails. Understanding the stack means understanding LLMs.
- **Each technique is a representation choice with specific tradeoffs.** BoW trades word order for simplicity. TF-IDF trades raw counts for relevance weighting. Embeddings trade interpretability for semantic density. LLMs trade efficiency for expressiveness. There is no universally "best" representation -- only the right one for your task and constraints.
- **Classical methods remain the right choice for many problems.** TF-IDF search is fast, interpretable, and effective for keyword retrieval. Using an LLM where TF-IDF suffices wastes computation and adds opacity. The decision tree is: start simple, add complexity only when simpler methods fail at your specific task.
- **LLMs do not "understand" language -- they compute increasingly rich statistical representations of it.** The journey from BoW to LLMs is one of richer representations, not of approaching comprehension. Each step captures more statistical structure. No step crosses from statistics into understanding. Knowing this protects you from both hype and disappointment.
