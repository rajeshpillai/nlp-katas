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

```rust
use std::collections::HashMap;
use std::collections::BTreeSet;

// --- Build a Bag of Words model from scratch ---
// No external libraries — pure Rust

fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .chars()
        .map(|c| if c.is_ascii_alphabetic() { c } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .filter(|w| !w.is_empty())
        .map(|w| w.to_string())
        .collect()
}

fn build_vocabulary(corpus: &[&str]) -> Vec<String> {
    let mut vocab = BTreeSet::new();
    for doc in corpus {
        for word in tokenize(doc) {
            vocab.insert(word);
        }
    }
    vocab.into_iter().collect()
}

fn document_to_bow(doc: &str, vocabulary: &[String]) -> Vec<usize> {
    let mut word_counts: HashMap<String, usize> = HashMap::new();
    for word in tokenize(doc) {
        *word_counts.entry(word).or_insert(0) += 1;
    }
    vocabulary.iter()
        .map(|word| *word_counts.get(word).unwrap_or(&0))
        .collect()
}

fn build_document_term_matrix(corpus: &[&str], vocabulary: &[String]) -> Vec<Vec<usize>> {
    corpus.iter()
        .map(|doc| document_to_bow(doc, vocabulary))
        .collect()
}

fn main() {
    // --- Our corpus ---
    let corpus: Vec<&str> = vec![
        "The cat sat on the mat",
        "The dog sat on the log",
        "The cat chased the dog around the yard",
        "A bird flew over the mat",
    ];

    println!("=== Corpus ===");
    for (i, doc) in corpus.iter().enumerate() {
        println!("  Doc {}: \"{}\"", i + 1, doc);
    }
    println!();

    // Step 1: Build vocabulary
    let vocabulary = build_vocabulary(&corpus);
    println!("=== Vocabulary ({} words) ===", vocabulary.len());
    println!("  {:?}", vocabulary);
    println!();

    // Step 2: Build document-term matrix
    let matrix = build_document_term_matrix(&corpus, &vocabulary);

    // Step 3: Display the matrix
    println!("=== Document-Term Matrix ===");
    let header: String = vocabulary.iter()
        .map(|w| format!("{:>8}", w))
        .collect::<Vec<_>>()
        .join("");
    println!("         {}", header);
    println!("         {}", "-".repeat(8 * vocabulary.len()));
    for (i, row) in matrix.iter().enumerate() {
        let row_str: String = row.iter()
            .map(|count| format!("{:>8}", count))
            .collect::<Vec<_>>()
            .join("");
        println!("  Doc {}: {}", i + 1, row_str);
    }
    println!();

    // Heatmap visualization
    let x_labels: Vec<&str> = vocabulary.iter().map(|s| s.as_str()).collect();
    let y_labels: Vec<String> = (0..corpus.len()).map(|i| format!("Doc {}", i + 1)).collect();
    let y_label_refs: Vec<&str> = y_labels.iter().map(|s| s.as_str()).collect();
    let heatmap_data: Vec<Vec<f64>> = matrix.iter()
        .map(|row| row.iter().map(|&v| v as f64).collect())
        .collect();
    show_chart(&heatmap_chart(
        "Document-Term Matrix",
        &x_labels,
        &y_label_refs,
        &heatmap_data,
    ));

    // Step 4: Analyze sparsity
    println!("=== Sparsity Analysis ===");
    let mut total_cells = 0usize;
    let mut zero_cells = 0usize;
    for (i, row) in matrix.iter().enumerate() {
        let zeros = row.iter().filter(|&&c| c == 0).count();
        let total = row.len();
        total_cells += total;
        zero_cells += zeros;
        let pct = (zeros as f64 / total as f64) * 100.0;
        println!("  Doc {}: {}/{} entries are zero ({:.0}% sparse)", i + 1, zeros, total, pct);
    }

    let overall_sparsity = (zero_cells as f64 / total_cells as f64) * 100.0;
    println!("\n  Overall matrix sparsity: {}/{} ({:.0}%)", zero_cells, total_cells, overall_sparsity);
    println!();

    // Step 5: Show the order-loss problem
    println!("=== BoW Loses Word Order ===");
    let sentence_a = "dog bites man";
    let sentence_b = "man bites dog";
    let vocab_ab = build_vocabulary(&[sentence_a, sentence_b]);
    let vec_a = document_to_bow(sentence_a, &vocab_ab);
    let vec_b = document_to_bow(sentence_b, &vocab_ab);
    println!("  Vocabulary: {:?}", vocab_ab);
    println!("  \"{}\" -> {:?}", sentence_a, vec_a);
    println!("  \"{}\" -> {:?}", sentence_b, vec_b);
    println!("  Vectors identical? {}", vec_a == vec_b);
    println!("  Meanings identical? No! This is the fundamental BoW tradeoff.");
    println!();

    // Step 6: Scaling intuition
    println!("=== Scaling Intuition ===");
    println!("  Corpus size: {} documents", corpus.len());
    println!("  Vocabulary size: {} words", vocabulary.len());
    println!("  Matrix dimensions: {} x {} = {} cells",
        corpus.len(), vocabulary.len(), corpus.len() * vocabulary.len());
    println!();
    println!("  In real-world corpora:");
    println!("    10,000 documents, 50,000 unique words = 500,000,000 cells");
    println!("    Most of those cells are 0. This is why sparsity matters.");
    println!("    Sparse representations save memory and computation.");
}
```

---

## Key Takeaways

- **BoW converts text into fixed-length numerical vectors** by counting word occurrences against a shared vocabulary. This is the foundation of all classical text analysis.
- **The document-term matrix is inherently sparse.** Each document uses only a tiny fraction of the total vocabulary, so most entries are zero. Real corpora are 95-99% sparse.
- **BoW deliberately discards word order.** "Dog bites man" and "Man bites dog" produce identical vectors. This is a known tradeoff: you gain simplicity and computational tractability at the cost of sequential meaning.
- **Vocabulary construction is a design choice.** What you include in the vocabulary (lowercasing, punctuation handling, stopword removal) directly shapes the representation. BoW does not "discover" meaning — it reflects the preprocessing decisions you make.
