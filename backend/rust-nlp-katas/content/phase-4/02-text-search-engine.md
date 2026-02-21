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
    +---------------------+   +----------------------+
    | Doc 1: "cat mat"    | -> | [0.0, 0.7, 0.7]     |
    | Doc 2: "dog park"   | -> | [0.7, 0.0, 0.0]     |
    | Doc 3: "cat dog"    | -> | [0.3, 0.3, 0.0]     |
    +---------------------+   +----------------------+
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

```rust
use std::collections::{HashMap, BTreeSet};

fn main() {
    // --- A Simple Text Search Engine using TF-IDF + Cosine Similarity ---
    // Pure Rust stdlib — no external dependencies

    fn tokenize(text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut current = String::new();
        for ch in text.to_lowercase().chars() {
            if ch.is_alphabetic() {
                current.push(ch);
            } else {
                if !current.is_empty() {
                    tokens.push(current.clone());
                    current.clear();
                }
            }
        }
        if !current.is_empty() {
            tokens.push(current);
        }
        tokens
    }

    // --- Stopwords (small curated list) ---
    let stopwords: BTreeSet<&str> = [
        "a", "an", "the", "is", "it", "in", "on", "of", "to", "and",
        "for", "with", "that", "this", "are", "was", "be", "as", "at",
        "by", "or", "from", "but", "not", "have", "has", "had", "do",
        "does", "did", "will", "can", "so", "if", "its", "we", "they",
    ].iter().copied().collect();

    let preprocess = |text: &str| -> Vec<String> {
        tokenize(text).into_iter()
            .filter(|t| !stopwords.contains(t.as_str()))
            .collect()
    };

    // --- TF-IDF Search Engine ---
    struct TFIDFSearchEngine {
        documents: Vec<(String, String)>,     // (name, original text)
        doc_tokens: Vec<Vec<String>>,         // token lists
        vocab: Vec<String>,                   // ordered vocabulary
        idf: HashMap<String, f64>,            // word -> IDF score
        doc_vectors: Vec<Vec<f64>>,           // TF-IDF vectors
    }

    impl TFIDFSearchEngine {
        fn new() -> Self {
            TFIDFSearchEngine {
                documents: Vec::new(),
                doc_tokens: Vec::new(),
                vocab: Vec::new(),
                idf: HashMap::new(),
                doc_vectors: Vec::new(),
            }
        }

        fn tfidf_vector(&self, tokens: &[String]) -> Vec<f64> {
            let mut tf: HashMap<&str, f64> = HashMap::new();
            for t in tokens { *tf.entry(t.as_str()).or_insert(0.0) += 1.0; }
            let total = if tokens.is_empty() { 1.0 } else { tokens.len() as f64 };
            self.vocab.iter().map(|word| {
                let tf_score = *tf.get(word.as_str()).unwrap_or(&0.0) / total;
                let idf_score = *self.idf.get(word.as_str()).unwrap_or(&0.0);
                tf_score * idf_score
            }).collect()
        }

        fn cosine_sim(v1: &[f64], v2: &[f64]) -> f64 {
            let dot: f64 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
            let m1: f64 = v1.iter().map(|a| a * a).sum::<f64>().sqrt();
            let m2: f64 = v2.iter().map(|b| b * b).sum::<f64>().sqrt();
            if m1 == 0.0 || m2 == 0.0 { return 0.0; }
            dot / (m1 * m2)
        }
    }

    let mut engine = TFIDFSearchEngine::new();

    // --- Build our corpus ---
    let corpus: Vec<(&str, &str)> = vec![
        ("Doc 1", "Natural language processing enables computers to understand human language and extract meaning from text data"),
        ("Doc 2", "Machine learning algorithms learn patterns from data and make predictions without being explicitly programmed"),
        ("Doc 3", "Deep learning neural networks process language using layers of transformations to capture complex patterns"),
        ("Doc 4", "Text classification assigns labels to documents based on their content using statistical or machine learning methods"),
        ("Doc 5", "Information retrieval systems search large collections of documents to find relevant results for user queries"),
        ("Doc 6", "Sentiment analysis determines whether text expresses positive negative or neutral opinions about a subject"),
        ("Doc 7", "Named entity recognition identifies and classifies proper nouns such as people places and organizations in text"),
    ];

    // --- Index the corpus ---
    engine.documents = corpus.iter().map(|(n, t)| (n.to_string(), t.to_string())).collect();
    engine.doc_tokens = corpus.iter().map(|(_, t)| preprocess(t)).collect();

    // Build vocabulary
    let mut all_words = BTreeSet::new();
    for tokens in &engine.doc_tokens {
        for t in tokens { all_words.insert(t.clone()); }
    }
    engine.vocab = all_words.into_iter().collect();

    // Compute IDF
    let n = corpus.len() as f64;
    for word in &engine.vocab {
        let df = engine.doc_tokens.iter()
            .filter(|tokens| tokens.contains(word))
            .count() as f64;
        let idf_val = if df > 0.0 { (n / df).ln() } else { 0.0 };
        engine.idf.insert(word.clone(), idf_val);
    }

    // Build TF-IDF vectors
    let tokens_clone: Vec<Vec<String>> = engine.doc_tokens.clone();
    engine.doc_vectors = tokens_clone.iter()
        .map(|tokens| engine.tfidf_vector(tokens))
        .collect();

    println!("Indexed {} documents with {} unique terms.", corpus.len(), engine.vocab.len());
    println!();

    // --- Run searches ---
    let queries = vec![
        "language understanding",
        "learning from data",
        "search and retrieval",
        "text classification methods",
        "neural network patterns",
    ];

    let query_colors = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"];

    for (qi, query) in queries.iter().enumerate() {
        println!("{}", "=".repeat(55));
        println!("  QUERY: \"{}\"", query);
        println!("{}", "=".repeat(55));

        let query_tokens = preprocess(query);
        let query_vector = engine.tfidf_vector(&query_tokens);

        // Score every document
        let mut scores: Vec<(usize, f64)> = Vec::new();
        for (i, doc_vec) in engine.doc_vectors.iter().enumerate() {
            let sim = TFIDFSearchEngine::cosine_sim(&query_vector, doc_vec);
            if sim > 0.0 {
                scores.push((i, sim));
            }
        }
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top_results: Vec<(usize, f64)> = scores.into_iter().take(3).collect();

        if top_results.is_empty() {
            println!("  No matching documents found.\n");
            continue;
        }

        for (rank, (idx, score)) in top_results.iter().enumerate() {
            let name = &engine.documents[*idx].0;
            let text = &engine.documents[*idx].1;
            let preview: String = text.chars().take(70).collect();
            println!("  {}. [{:.4}] {}: {}...", rank + 1, score, name, preview);
        }
        println!();

        // Visualize search result scores
        let result_labels: Vec<&str> = top_results.iter()
            .map(|(idx, _)| engine.documents[*idx].0.as_str())
            .collect();
        let result_scores: Vec<f64> = top_results.iter()
            .map(|(_, s)| (*s * 10000.0).round() / 10000.0)
            .collect();
        show_chart(&bar_chart(
            &format!("Search Results: \"{}\"", query),
            &result_labels,
            "Cosine Similarity",
            &result_scores,
            query_colors[qi % query_colors.len()],
        ));
    }

    // --- Show IDF values for selected words ---
    println!("{}", "=".repeat(55));
    println!("  IDF VALUES (higher = rarer = more discriminating)");
    println!("{}", "=".repeat(55));

    let interesting_words = [
        "language", "learning", "text", "data", "patterns",
        "search", "sentiment", "neural", "classification",
    ];
    for word in &interesting_words {
        if let Some(idf_val) = engine.idf.get(*word) {
            let df = engine.doc_tokens.iter()
                .filter(|tokens| tokens.iter().any(|t| t == *word))
                .count();
            println!("  {:<18} IDF: {:.3}  (appears in {} docs)", word, idf_val, df);
        }
    }

    println!();
    println!("Words appearing in fewer documents get higher IDF scores.");
    println!("These are the words that DISTINGUISH documents from each other.");
    println!("This is why TF-IDF search works: it rewards distinctive matches.");
}
```

---

## Key Takeaways

- **A search engine is a similarity problem.** The query is treated as a mini-document, vectorized the same way as the corpus, and compared via cosine similarity. No special magic — just geometry.
- **TF-IDF is the backbone of keyword search.** By upweighting rare terms and downweighting common ones, TF-IDF ensures that distinctive words drive the ranking, not filler words like "the" or "is".
- **Indexing separates offline cost from online speed.** Building TF-IDF vectors for all documents is done once. Each query only needs to vectorize the query and compute similarities — this is what makes search fast.
- **Keyword search has fundamental limits.** It cannot handle synonyms ("car" vs "automobile"), paraphrases, or semantic meaning. It only matches on shared vocabulary. Many NLP problems reduce to similarity in vector space, but the quality depends on the representation.
