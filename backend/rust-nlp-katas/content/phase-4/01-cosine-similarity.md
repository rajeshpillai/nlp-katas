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

```rust
use std::collections::{HashMap, BTreeSet};

fn main() {
    // --- Cosine Similarity, Euclidean Distance, and Jaccard Similarity ---
    // All implemented from scratch using only Rust standard library

    /// Lowercase and split on non-alpha characters.
    fn tokenize(text: &str) -> Vec<String> {
        let mut result = Vec::new();
        let mut current = String::new();
        for ch in text.to_lowercase().chars() {
            if ch.is_alphabetic() {
                current.push(ch);
            } else {
                if !current.is_empty() {
                    result.push(current.clone());
                    current.clear();
                }
            }
        }
        if !current.is_empty() {
            result.push(current);
        }
        result
    }

    /// Build aligned count vectors over the shared vocabulary.
    fn build_vectors(doc1_tokens: &[String], doc2_tokens: &[String]) -> (Vec<String>, Vec<f64>, Vec<f64>) {
        let mut vocab_set = BTreeSet::new();
        for t in doc1_tokens.iter().chain(doc2_tokens.iter()) {
            vocab_set.insert(t.clone());
        }
        let vocab: Vec<String> = vocab_set.into_iter().collect();

        let mut c1: HashMap<&str, f64> = HashMap::new();
        for t in doc1_tokens { *c1.entry(t.as_str()).or_insert(0.0) += 1.0; }
        let mut c2: HashMap<&str, f64> = HashMap::new();
        for t in doc2_tokens { *c2.entry(t.as_str()).or_insert(0.0) += 1.0; }

        let vec1: Vec<f64> = vocab.iter().map(|w| *c1.get(w.as_str()).unwrap_or(&0.0)).collect();
        let vec2: Vec<f64> = vocab.iter().map(|w| *c2.get(w.as_str()).unwrap_or(&0.0)).collect();

        (vocab, vec1, vec2)
    }

    /// Compute dot product of two vectors.
    fn dot_product(v1: &[f64], v2: &[f64]) -> f64 {
        v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum()
    }

    /// Compute the magnitude (L2 norm) of a vector.
    fn magnitude(v: &[f64]) -> f64 {
        v.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Cosine similarity: dot(A,B) / (|A| * |B|).
    fn cosine_similarity(v1: &[f64], v2: &[f64]) -> f64 {
        let d = dot_product(v1, v2);
        let m1 = magnitude(v1);
        let m2 = magnitude(v2);
        if m1 == 0.0 || m2 == 0.0 { return 0.0; }
        d / (m1 * m2)
    }

    /// Euclidean distance: sqrt(sum((a-b)^2)).
    fn euclidean_distance(v1: &[f64], v2: &[f64]) -> f64 {
        v1.iter().zip(v2.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt()
    }

    /// Jaccard similarity: |intersection| / |union| of token sets.
    fn jaccard_similarity(tokens1: &[String], tokens2: &[String]) -> f64 {
        let s1: BTreeSet<&str> = tokens1.iter().map(|s| s.as_str()).collect();
        let s2: BTreeSet<&str> = tokens2.iter().map(|s| s.as_str()).collect();
        if s1.is_empty() && s2.is_empty() { return 0.0; }
        let intersection = s1.intersection(&s2).count();
        let union = s1.union(&s2).count();
        intersection as f64 / union as f64
    }

    // --- Corpus ---
    let documents: Vec<(&str, &str)> = vec![
        ("Doc A", "The cat sat on the mat and watched the birds outside"),
        ("Doc B", "A cat was sitting on a mat looking at birds through the window"),
        ("Doc C", "Stock markets rallied today as investors grew confident"),
        ("Doc D", "The financial markets saw gains as stocks rose sharply"),
        ("Doc E", "The cat sat on the mat the cat sat on the mat"),
    ];

    println!("{}", "=".repeat(60));
    println!("DOCUMENTS");
    println!("{}", "=".repeat(60));
    for (name, text) in &documents {
        println!("  {}: {}", name, text);
    }
    println!();

    // Tokenize all documents
    let tokenized: Vec<(&str, Vec<String>)> = documents.iter()
        .map(|(name, text)| (*name, tokenize(text)))
        .collect();

    // --- Compare all pairs ---
    let names: Vec<&str> = documents.iter().map(|(n, _)| *n).collect();

    println!("{}", "=".repeat(60));
    println!("PAIRWISE SIMILARITY COMPARISON");
    println!("{}", "=".repeat(60));
    println!("{:<12} {:>8} {:>8} {:>8}", "Pair", "Cosine", "Euclid", "Jaccard");
    println!("{}", "-".repeat(40));

    for i in 0..names.len() {
        for j in (i + 1)..names.len() {
            let t1 = &tokenized[i].1;
            let t2 = &tokenized[j].1;
            let (_vocab, v1, v2) = build_vectors(t1, t2);

            let cos = cosine_similarity(&v1, &v2);
            let euc = euclidean_distance(&v1, &v2);
            let jac = jaccard_similarity(t1, t2);

            let label = format!("{}-{}", names[i], names[j]);
            println!("{:<12} {:>8.3} {:>8.3} {:>8.3}", label, cos, euc, jac);
        }
    }
    println!();

    // Build cosine similarity matrix for heatmap
    let doc_names: Vec<&str> = documents.iter().map(|(n, _)| *n).collect();
    let n_docs = doc_names.len();
    let mut cos_matrix: Vec<Vec<f64>> = Vec::new();
    for i in 0..n_docs {
        let mut row = Vec::new();
        for j in 0..n_docs {
            if i == j {
                row.push(1.0);
            } else {
                let t1 = &tokenized[i].1;
                let t2 = &tokenized[j].1;
                let (_, v1, v2) = build_vectors(t1, t2);
                let val = (cosine_similarity(&v1, &v2) * 1000.0).round() / 1000.0;
                row.push(val);
            }
        }
        cos_matrix.push(row);
    }

    show_chart(&heatmap_chart(
        "Cosine Similarity Matrix (All Document Pairs)",
        &doc_names,
        &doc_names,
        &cos_matrix,
    ));

    // --- Deep dive: Doc A vs Doc B ---
    println!("{}", "=".repeat(60));
    println!("DEEP DIVE: Doc A vs Doc B");
    println!("{}", "=".repeat(60));
    let (vocab_ab, va, vb) = build_vectors(&tokenized[0].1, &tokenized[1].1);
    println!("Vocabulary: {:?}", vocab_ab);
    println!("Vec A:      {:?}", va.iter().map(|v| *v as i64).collect::<Vec<_>>());
    println!("Vec B:      {:?}", vb.iter().map(|v| *v as i64).collect::<Vec<_>>());
    println!("Dot product:    {}", dot_product(&va, &vb));
    println!("|A|:            {:.3}", magnitude(&va));
    println!("|B|:            {:.3}", magnitude(&vb));
    println!("Cosine sim:     {:.4}", cosine_similarity(&va, &vb));
    println!("Euclidean dist: {:.4}", euclidean_distance(&va, &vb));
    println!("Jaccard sim:    {:.4}", jaccard_similarity(&tokenized[0].1, &tokenized[1].1));
    println!();

    // --- Magnitude sensitivity demo ---
    println!("{}", "=".repeat(60));
    println!("WHY COSINE BEATS EUCLIDEAN FOR DIFFERENT-LENGTH DOCS");
    println!("{}", "=".repeat(60));

    let short = "machine learning is powerful";
    let long_doc: String = std::iter::repeat("machine learning is powerful")
        .take(10).collect::<Vec<_>>().join(" ");

    let t_short = tokenize(short);
    let t_long = tokenize(&long_doc);
    let (_, v_short, v_long) = build_vectors(&t_short, &t_long);

    println!("Short doc tokens: {}", t_short.len());
    println!("Long doc tokens:  {}", t_long.len());
    println!("Short vec: {:?}", v_short.iter().map(|v| *v as i64).collect::<Vec<_>>());
    println!("Long vec:  {:?}", v_long.iter().map(|v| *v as i64).collect::<Vec<_>>());
    println!("Cosine similarity:  {:.4}  (perfect!)", cosine_similarity(&v_short, &v_long));
    println!("Euclidean distance: {:.4}  (misleadingly large)", euclidean_distance(&v_short, &v_long));
    println!();
    println!("Both documents express the SAME idea.");
    println!("Cosine correctly gives 1.0 — same direction in vector space.");
    println!("Euclidean gives a large distance — fooled by magnitude difference.");
    println!();

    // --- When Jaccard disagrees with Cosine ---
    println!("{}", "=".repeat(60));
    println!("WHEN JACCARD AND COSINE DISAGREE");
    println!("{}", "=".repeat(60));

    let d1 = "data data data data science";
    let d2 = "data science science science science";
    let t1 = tokenize(d1);
    let t2 = tokenize(d2);
    let (_, v1, v2) = build_vectors(&t1, &t2);

    println!("Doc 1: \"{}\"", d1);
    println!("Doc 2: \"{}\"", d2);
    println!("Jaccard:  {:.4}  (perfect — same word set!)", jaccard_similarity(&t1, &t2));
    println!("Cosine:   {:.4}  (lower — different proportions)", cosine_similarity(&v1, &v2));
    println!();
    println!("Jaccard only sees presence/absence.");
    println!("Cosine sees that Doc1 emphasizes 'data' while Doc2 emphasizes 'science'.");
}
```

---

## Key Takeaways

- **Cosine similarity measures direction, not magnitude.** This makes it ideal for comparing documents of different lengths, because a long and short document about the same topic point in the same direction.
- **Euclidean distance is sensitive to scale.** Two documents with identical word proportions but different lengths will appear far apart, making it unreliable for raw term-count vectors.
- **Jaccard similarity ignores frequency entirely.** It only cares whether a word is present or absent, which loses valuable information about how much a document emphasizes certain terms.
- **Choosing the right similarity metric is a modeling decision.** Cosine similarity is the standard for text because NLP cares about what a document is *about* (direction), not how long it is (magnitude). Many NLP problems reduce to similarity in vector space.
