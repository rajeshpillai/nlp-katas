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

```rust
use std::collections::HashMap;
use std::collections::BTreeSet;
use std::collections::HashSet;

// --- Comparing document similarity: BoW vs TF-IDF ---

fn main() {
    // A small corpus where BoW gives misleading similarity
    let corpus = vec![
        "the cat sat on the mat in the room",           // Doc 0: about a cat
        "the dog sat on the rug in the hall",           // Doc 1: about a dog
        "the cat chased the mouse in the garden",       // Doc 2: about a cat
        "the dog fetched the stick in the park",        // Doc 3: about a dog
        "the bird sat on the branch in the tree",       // Doc 4: filler (shares structure)
        "the fish swam in the pond near the rocks",     // Doc 5: filler (shares structure)
    ];

    println!("=== CORPUS ===\n");
    for (i, doc) in corpus.iter().enumerate() {
        println!("  Doc {}: \"{}\"", i, doc);
    }
    println!();

    // Tokenize
    let documents: Vec<Vec<String>> = corpus.iter()
        .map(|doc| doc.to_lowercase().split_whitespace().map(|s| s.to_string()).collect())
        .collect();

    // Build vocabulary
    let mut vocab_set = BTreeSet::new();
    for doc in &documents {
        for word in doc {
            vocab_set.insert(word.clone());
        }
    }
    let vocabulary: Vec<String> = vocab_set.into_iter().collect();


    // --- Bag of Words ---

    fn bow_vector(doc: &[String], vocab: &[String]) -> Vec<f64> {
        let mut counts: HashMap<&str, usize> = HashMap::new();
        for word in doc {
            *counts.entry(word.as_str()).or_insert(0) += 1;
        }
        vocab.iter().map(|w| *counts.get(w.as_str()).unwrap_or(&0) as f64).collect()
    }


    // --- TF-IDF ---

    fn compute_tf(doc: &[String]) -> HashMap<String, f64> {
        let total = doc.len() as f64;
        let mut counts: HashMap<String, f64> = HashMap::new();
        for word in doc {
            *counts.entry(word.clone()).or_insert(0.0) += 1.0;
        }
        for val in counts.values_mut() {
            *val /= total;
        }
        counts
    }

    fn compute_idf(documents: &[Vec<String>], vocab: &[String]) -> HashMap<String, f64> {
        let n = documents.len() as f64;
        let mut idf = HashMap::new();
        for word in vocab {
            let doc_freq = documents.iter()
                .filter(|doc| doc.contains(word))
                .count() as f64;
            let val = if doc_freq > 0.0 { (n / doc_freq).ln() } else { 0.0 };
            idf.insert(word.clone(), val);
        }
        idf
    }

    fn tfidf_vector(doc: &[String], vocab: &[String], idf: &HashMap<String, f64>) -> Vec<f64> {
        let tf = compute_tf(doc);
        vocab.iter()
            .map(|w| tf.get(w).copied().unwrap_or(0.0) * idf.get(w).copied().unwrap_or(0.0))
            .collect()
    }


    // --- Cosine Similarity ---

    fn cosine_similarity(vec_a: &[f64], vec_b: &[f64]) -> f64 {
        let dot: f64 = vec_a.iter().zip(vec_b.iter()).map(|(a, b)| a * b).sum();
        let mag_a: f64 = vec_a.iter().map(|a| a * a).sum::<f64>().sqrt();
        let mag_b: f64 = vec_b.iter().map(|b| b * b).sum::<f64>().sqrt();
        if mag_a == 0.0 || mag_b == 0.0 {
            return 0.0;
        }
        dot / (mag_a * mag_b)
    }


    // Build all vectors
    let bow_vectors: Vec<Vec<f64>> = documents.iter()
        .map(|doc| bow_vector(doc, &vocabulary))
        .collect();

    let idf = compute_idf(&documents, &vocabulary);
    let tfidf_vectors: Vec<Vec<f64>> = documents.iter()
        .map(|doc| tfidf_vector(doc, &vocabulary, &idf))
        .collect();


    // --- Compare all pairs ---

    println!("=== PAIRWISE SIMILARITY ===\n");
    println!("  {:<15} {:>12} {:>14}  {}", "Pair", "BoW Cosine", "TF-IDF Cosine", "Better?");
    println!("  {} {} {}  {}", "-".repeat(15), "-".repeat(12), "-".repeat(14), "-".repeat(30));

    let pairs = vec![(0, 1), (0, 2), (1, 3), (0, 3), (1, 2)];

    let mut pair_labels: Vec<String> = Vec::new();
    let mut bow_scores: Vec<f64> = Vec::new();
    let mut tfidf_scores_list: Vec<f64> = Vec::new();

    for &(i, j) in &pairs {
        let bow_sim = cosine_similarity(&bow_vectors[i], &bow_vectors[j]);
        let tfidf_sim = cosine_similarity(&tfidf_vectors[i], &tfidf_vectors[j]);

        let note = match (i, j) {
            (0, 2) => "<-- both about cats",
            (1, 3) => "<-- both about dogs",
            (0, 1) => "<-- cat vs dog (different topics)",
            (0, 3) | (1, 2) => "<-- different topics",
            _ => "",
        };

        println!("  Doc {} vs Doc {}  {:12.4} {:14.4}  {}", i, j, bow_sim, tfidf_sim, note);

        pair_labels.push(format!("Doc {} vs Doc {}", i, j));
        bow_scores.push((bow_sim * 10000.0).round() / 10000.0);
        tfidf_scores_list.push((tfidf_sim * 10000.0).round() / 10000.0);
    }

    // Visualize BoW vs TF-IDF similarity as grouped bar chart
    let pair_label_refs: Vec<&str> = pair_labels.iter().map(|s| s.as_str()).collect();
    show_chart(&multi_bar_chart(
        "BoW vs TF-IDF Cosine Similarity",
        &pair_label_refs,
        &[
            ("BoW Cosine", &bow_scores, "#f59e0b"),
            ("TF-IDF Cosine", &tfidf_scores_list, "#3b82f6"),
        ],
    ));

    // --- Show why BoW is misleading ---

    println!("\n=== WHY BOW IS MISLEADING ===\n");
    println!("  Words shared between Doc 0 (cat) and Doc 1 (dog):");
    let shared_01: HashSet<&String> = documents[0].iter().collect::<HashSet<_>>()
        .intersection(&documents[1].iter().collect())
        .copied()
        .collect();
    let mut shared_01_sorted: Vec<&&String> = shared_01.iter().collect();
    shared_01_sorted.sort();
    for word in &shared_01_sorted {
        let c0 = documents[0].iter().filter(|w| w == *word).count();
        let c1 = documents[1].iter().filter(|w| w == *word).count();
        let idf_val = idf.get(word.as_str()).copied().unwrap_or(0.0);
        println!("    \"{}\": count in Doc0={}, Doc1={}, IDF={:.4}", word, c0, c1, idf_val);
    }

    println!("\n  Words shared between Doc 0 (cat) and Doc 2 (cat):");
    let shared_02: HashSet<&String> = documents[0].iter().collect::<HashSet<_>>()
        .intersection(&documents[2].iter().collect())
        .copied()
        .collect();
    let mut shared_02_sorted: Vec<&&String> = shared_02.iter().collect();
    shared_02_sorted.sort();
    for word in &shared_02_sorted {
        let c0 = documents[0].iter().filter(|w| w == *word).count();
        let c2 = documents[2].iter().filter(|w| w == *word).count();
        let idf_val = idf.get(word.as_str()).copied().unwrap_or(0.0);
        println!("    \"{}\": count in Doc0={}, Doc2={}, IDF={:.4}", word, c0, c2, idf_val);
    }

    println!();
    println!("  BoW similarity between Doc 0 and Doc 1 is inflated by shared");
    println!("  common words (\"the\", \"sat\", \"on\", \"in\") that appear everywhere.");
    println!();
    println!("  TF-IDF suppresses these common words and lets topic-specific words");
    println!("  like \"cat\", \"mat\", \"mouse\", \"garden\" drive the similarity score.");


    // --- Show the IDF values to explain what's happening ---

    println!("\n=== IDF VALUES (what gets suppressed) ===\n");

    let mut idf_sorted: Vec<(&String, &f64)> = idf.iter().collect();
    idf_sorted.sort_by(|a, b| a.1.partial_cmp(b.1).unwrap());

    for (word, &score) in &idf_sorted {
        let doc_freq = documents.iter()
            .filter(|doc| doc.contains(*word))
            .count();
        let bar = "#".repeat((score * 20.0) as usize);
        let label = if score == 0.0 { "SUPPRESSED" }
            else if score < 0.5 { "low" }
            else { "high" };
        println!("  {:>10}  IDF={:.3}  in {}/{} docs  {}  ({})",
            word, score, doc_freq, documents.len(), bar, label);
    }


    // --- Final verdict ---

    println!("\n=== VERDICT ===\n");

    let bow_01 = cosine_similarity(&bow_vectors[0], &bow_vectors[1]);
    let bow_02 = cosine_similarity(&bow_vectors[0], &bow_vectors[2]);
    let tfidf_01 = cosine_similarity(&tfidf_vectors[0], &tfidf_vectors[1]);
    let tfidf_02 = cosine_similarity(&tfidf_vectors[0], &tfidf_vectors[2]);

    println!("  Question: Is Doc 0 (cat) more similar to Doc 1 (dog) or Doc 2 (cat)?");
    println!();
    print!("  BoW answer:    Doc 0-1 = {:.4}, Doc 0-2 = {:.4}", bow_01, bow_02);
    if bow_01 > bow_02 {
        println!("  --> BoW says Doc 1 (dog) is closer. WRONG.");
    } else {
        println!("  --> BoW says Doc 2 (cat) is closer. Correct.");
    }

    print!("  TF-IDF answer: Doc 0-1 = {:.4}, Doc 0-2 = {:.4}", tfidf_01, tfidf_02);
    if tfidf_02 > tfidf_01 {
        println!("  --> TF-IDF says Doc 2 (cat) is closer. Correct!");
    } else {
        println!("  --> TF-IDF says Doc 1 (dog) is closer.");
    }

    println!();
    println!("  TF-IDF produces more topically meaningful similarity scores");
    println!("  by suppressing words that carry no discriminative information.");

    // Visualize the verdict
    show_chart(&multi_bar_chart(
        "Doc 0 (cat) - Similarity to Doc 1 (dog) vs Doc 2 (cat)",
        &["Doc 0 vs Doc 1 (dog)", "Doc 0 vs Doc 2 (cat)"],
        &[
            ("BoW Cosine", &[
                (bow_01 * 10000.0).round() / 10000.0,
                (bow_02 * 10000.0).round() / 10000.0,
            ], "#f59e0b"),
            ("TF-IDF Cosine", &[
                (tfidf_01 * 10000.0).round() / 10000.0,
                (tfidf_02 * 10000.0).round() / 10000.0,
            ], "#3b82f6"),
        ],
    ));
}
```

---

## Key Takeaways

- **BoW similarity is dominated by common words.** When two documents share frequent words like "the", "is", "on", their BoW cosine similarity is inflated regardless of topical relevance.
- **TF-IDF rescues similarity by downweighting noise.** Words appearing across many documents contribute less to the similarity score, letting topic-specific words drive the comparison.
- **Cosine similarity measures direction, not magnitude.** This makes it robust to document length differences — a short document and a long document about the same topic will still score high.
- **TF-IDF is a relevance heuristic, not "understanding."** It improves on BoW by using corpus statistics, but it still cannot capture synonymy, word order, or meaning. "Happy" and "joyful" remain completely unrelated in TF-IDF space.
