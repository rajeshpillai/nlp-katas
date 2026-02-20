# Build a Simple Text Search Engine

> Phase 4 — Similarity & Classical NLP Tasks | Kata 4.2

---

## Concept & Intuition

### What problem are we solving?

Given a collection of documents and a user's query, we want to **rank the documents by relevance** and return the best matches. This is the core problem behind every search engine, from a library catalog to a web search bar.

The key insight is that a query is just another document. We can represent both the query and every document as TF-IDF vectors, then use cosine similarity to rank documents by how closely they match the query. The highest-scoring documents are the most relevant results.

### Why naive/earlier approaches fail

**Exact string matching** fails because a search for "machine learning algorithms" should match a document containing "algorithms for learning on machines" even though the exact phrase never appears.

**Simple word overlap** (count shared words) fails because common words like "the" and "is" dominate the overlap count. A document full of stopwords will score high against any query.

**Raw Bag of Words** (without TF-IDF weighting) treats every word equally. A document matching the query on the word "the" is scored the same as one matching on "transformer" — clearly wrong. We need a way to say: rare, distinctive words matter more than common ones.

TF-IDF solves this by downweighting common words (high document frequency) and upweighting rare, discriminating words (low document frequency). Combined with cosine similarity, this gives us a principled ranking.

### Mental models

- **Indexing** = converting every document into a TF-IDF vector ahead of time. This is the offline/preprocessing step.
- **Querying** = converting the query into a TF-IDF vector using the same vocabulary, then computing cosine similarity against every document vector.
- **Ranking** = sorting documents by their cosine similarity score, highest first.
- **The vocabulary** is built from the corpus. Words not in the corpus cannot be searched for (this is a known limitation called the out-of-vocabulary problem).

### Visual explanations

```
Search engine pipeline:

  INDEXING (offline, done once):

    Corpus                    TF-IDF Vectors
    ┌─────────────────┐       ┌──────────────────┐
    │ Doc 1: "cat mat" │ ───> │ [0.0, 0.7, 0.7]  │
    │ Doc 2: "dog park"│ ───> │ [0.7, 0.0, 0.0]  │
    │ Doc 3: "cat dog" │ ───> │ [0.3, 0.3, 0.0]  │
    └─────────────────┘       └──────────────────┘
                                     |
                               Stored in index


  QUERYING (online, per search):

    Query: "cat"
         |
         v
    TF-IDF vector: [0.0, 1.0, 0.0]   (using same vocab)
         |
         v
    Cosine similarity against each doc:
         Doc 1: 0.71  <-- best match
         Doc 3: 0.45
         Doc 2: 0.00
         |
         v
    Ranked results:
         1. Doc 1 (score: 0.71)
         2. Doc 3 (score: 0.45)
         3. Doc 2 (score: 0.00)  <-- filtered out (no match)


  Why TF-IDF matters for search:

    Query: "rare disease symptoms"

    Without TF-IDF (raw counts):
      "the disease is a disease"       score: HIGH (word "disease" x2)
      "rare tropical disease symptoms" score: MEDIUM

    With TF-IDF:
      "rare tropical disease symptoms" score: HIGH
        ("rare" and "symptoms" are distinctive — high IDF)
      "the disease is a disease"       score: LOW
        ("the", "is", "a" have near-zero IDF)
```

---

## Hands-on Exploration

1. Take the five documents in the code below. Before running it, predict which document will rank highest for the query "natural language understanding". Think about which words are rare across the corpus and which are common. Run the code to check your prediction.

2. Try searching for a word that appears in every document (like "the" or "and" if present). Observe how TF-IDF gives it a near-zero weight, so no document scores highly. Then search for a word that appears in only one document and observe the sharp ranking. This is IDF in action.

3. Modify the corpus to add two more documents about a completely new topic (e.g., cooking recipes). Then search for a query that mixes topics (e.g., "learning to cook"). Observe how the engine handles multi-topic queries and which documents surface. What does this tell you about the limitations of keyword-based search?

---

## Live Code

```python
# --- A Simple Text Search Engine using TF-IDF + Cosine Similarity ---
# Pure Python stdlib — no external dependencies

import math
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


# --- Stopwords (small curated list) ---
STOPWORDS = {
    "a", "an", "the", "is", "it", "in", "on", "of", "to", "and",
    "for", "with", "that", "this", "are", "was", "be", "as", "at",
    "by", "or", "from", "but", "not", "have", "has", "had", "do",
    "does", "did", "will", "can", "so", "if", "its", "we", "they",
}


def preprocess(text):
    """Tokenize and remove stopwords."""
    return [t for t in tokenize(text) if t not in STOPWORDS]


class TFIDFSearchEngine:
    """A minimal search engine: index documents, query with ranking."""

    def __init__(self):
        self.documents = {}      # name -> original text
        self.doc_tokens = {}     # name -> token list
        self.vocab = []          # ordered vocabulary
        self.idf = {}            # word -> IDF score
        self.doc_vectors = {}    # name -> TF-IDF vector

    def index(self, documents):
        """Build the TF-IDF index from a dict of {name: text}."""
        self.documents = documents
        self.doc_tokens = {
            name: preprocess(text) for name, text in documents.items()
        }

        # Build vocabulary from all documents
        all_words = set()
        for tokens in self.doc_tokens.values():
            all_words.update(tokens)
        self.vocab = sorted(all_words)

        # Compute IDF: log(N / df) where df = docs containing the word
        n = len(documents)
        self.idf = {}
        for word in self.vocab:
            df = sum(
                1 for tokens in self.doc_tokens.values() if word in tokens
            )
            self.idf[word] = math.log(n / df) if df > 0 else 0.0

        # Build TF-IDF vector for each document
        for name, tokens in self.doc_tokens.items():
            self.doc_vectors[name] = self._tfidf_vector(tokens)

        print(f"Indexed {n} documents with {len(self.vocab)} unique terms.")

    def _tfidf_vector(self, tokens):
        """Compute TF-IDF vector for a token list."""
        tf = Counter(tokens)
        total = len(tokens) if tokens else 1
        vector = []
        for word in self.vocab:
            tf_score = tf.get(word, 0) / total
            idf_score = self.idf.get(word, 0.0)
            vector.append(tf_score * idf_score)
        return vector

    def _cosine_sim(self, v1, v2):
        """Cosine similarity between two vectors."""
        dot = sum(a * b for a, b in zip(v1, v2))
        m1 = math.sqrt(sum(a * a for a in v1))
        m2 = math.sqrt(sum(b * b for b in v2))
        if m1 == 0 or m2 == 0:
            return 0.0
        return dot / (m1 * m2)

    def search(self, query, top_k=5):
        """Search the index and return ranked results."""
        query_tokens = preprocess(query)
        query_vector = self._tfidf_vector(query_tokens)

        # Score every document
        scores = []
        for name, doc_vec in self.doc_vectors.items():
            sim = self._cosine_sim(query_vector, doc_vec)
            if sim > 0:
                scores.append((name, sim))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# --- Build our corpus ---
corpus = {
    "Doc 1": (
        "Natural language processing enables computers to understand "
        "human language and extract meaning from text data"
    ),
    "Doc 2": (
        "Machine learning algorithms learn patterns from data and make "
        "predictions without being explicitly programmed"
    ),
    "Doc 3": (
        "Deep learning neural networks process language using layers "
        "of transformations to capture complex patterns"
    ),
    "Doc 4": (
        "Text classification assigns labels to documents based on their "
        "content using statistical or machine learning methods"
    ),
    "Doc 5": (
        "Information retrieval systems search large collections of "
        "documents to find relevant results for user queries"
    ),
    "Doc 6": (
        "Sentiment analysis determines whether text expresses positive "
        "negative or neutral opinions about a subject"
    ),
    "Doc 7": (
        "Named entity recognition identifies and classifies proper nouns "
        "such as people places and organizations in text"
    ),
}

# --- Index the corpus ---
engine = TFIDFSearchEngine()
engine.index(corpus)
print()

# --- Run searches ---
queries = [
    "language understanding",
    "learning from data",
    "search and retrieval",
    "text classification methods",
    "neural network patterns",
]

for query in queries:
    print("=" * 55)
    print(f'  QUERY: "{query}"')
    print("=" * 55)
    results = engine.search(query, top_k=3)

    if not results:
        print("  No matching documents found.\n")
        continue

    for rank, (name, score) in enumerate(results, 1):
        print(f"  {rank}. [{score:.4f}] {name}: {corpus[name][:70]}...")
    print()

# --- Show IDF values for selected words ---
print("=" * 55)
print("  IDF VALUES (higher = rarer = more discriminating)")
print("=" * 55)
interesting_words = [
    "language", "learning", "text", "data", "patterns",
    "search", "sentiment", "neural", "classification",
]
for word in interesting_words:
    if word in engine.idf:
        df = sum(
            1 for tokens in engine.doc_tokens.values() if word in tokens
        )
        print(f"  {word:<18} IDF: {engine.idf[word]:.3f}  (appears in {df} docs)")

print()
print("Words appearing in fewer documents get higher IDF scores.")
print("These are the words that DISTINGUISH documents from each other.")
print("This is why TF-IDF search works: it rewards distinctive matches.")
```

---

## Key Takeaways

- **A search engine is a similarity problem.** The query is treated as a mini-document, vectorized the same way as the corpus, and compared via cosine similarity. No special magic — just geometry.
- **TF-IDF is the backbone of keyword search.** By upweighting rare terms and downweighting common ones, TF-IDF ensures that distinctive words drive the ranking, not filler words like "the" or "is".
- **Indexing separates offline cost from online speed.** Building TF-IDF vectors for all documents is done once. Each query only needs to vectorize the query and compute similarities — this is what makes search fast.
- **Keyword search has fundamental limits.** It cannot handle synonyms ("car" vs "automobile"), paraphrases, or semantic meaning. It only matches on shared vocabulary. Many NLP problems reduce to similarity in vector space, but the quality depends on the representation.
