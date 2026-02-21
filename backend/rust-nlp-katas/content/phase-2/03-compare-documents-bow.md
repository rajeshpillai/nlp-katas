# Compare Documents Using Bag of Words

> Phase 2 — Bag of Words (BoW) | Kata 2.3

---

## Concept & Intuition

### What problem are we solving?

Now that we can represent documents as numerical vectors (Kata 2.1) and visualize them (Kata 2.2), we need to **measure how similar two documents are**. This is the core operation behind text search, document clustering, duplicate detection, and recommendation systems.

Given two BoW vectors, we need a number that says: "these documents are very similar" or "these documents have nothing in common." The dot product and cosine similarity give us exactly that.

### Why naive/earlier approaches fail

- **Exact string matching**: "The cat sat on the mat" and "A cat napped on a soft mat" share topic words but are different strings. Exact match says they are 0% similar.
- **Word overlap count**: Counting shared words is a start, but it does not account for document length. A 1000-word document will share more words with anything simply because it has more words. We need a **normalized** measure.
- **Euclidean distance**: Measures the straight-line distance between vector endpoints. But two documents about the same topic, one short and one long, will be far apart in Euclidean space because the longer one has larger counts. We want to measure **direction**, not magnitude.

Cosine similarity solves the length problem by comparing the **angle** between vectors, not their distance.

### Mental models

- **The flashlight analogy**: Think of each document vector as a flashlight beam pointing in some direction in word-space. Cosine similarity measures how much the beams overlap. If they point the same way (similar topics), cosine is near 1. If they point in completely different directions (different topics), cosine is near 0.
- **Dot product as overlap**: The dot product multiplies matching dimensions and sums them. If both documents use the word "cat" frequently, that dimension contributes a large value. If one uses "cat" and the other does not, that dimension contributes zero.
- **Normalization as fairness**: Dividing by vector magnitudes ensures that a short email and a long essay about the same topic get a high similarity score. Without normalization, the essay always "wins."

### Visual explanations

```
Two vectors in 3D word-space:

  Vocabulary: [cat, dog, fish]

  Doc A: "cat cat dog"   -> [2, 1, 0]
  Doc B: "cat dog dog"   -> [1, 2, 0]
  Doc C: "fish fish"     -> [0, 0, 2]

  Dot product (A, B) = (2*1) + (1*2) + (0*0) = 4    (some overlap)
  Dot product (A, C) = (2*0) + (1*0) + (0*2) = 0    (no overlap)

  Cosine similarity:

             dot(A, B)          4          4
  cos(A,B) = ----------- = ----------- = ----- = 0.80
             |A| * |B|    sqrt(5)*sqrt(5)   5

             dot(A, C)          0
  cos(A,C) = ----------- = ----------- = 0.00
             |A| * |C|    sqrt(5)*sqrt(4)

  A and B are similar (both about cats and dogs).
  A and C are completely dissimilar (no shared words).

The BoW failure case:

  Doc X: "dog bites man"  -> [1, 1, 1]  (vocab: [bites, dog, man])
  Doc Y: "man bites dog"  -> [1, 1, 1]

  cos(X, Y) = 1.0  (identical vectors!)

  But the meanings are opposite. BoW cannot see word order.
```

---

## Hands-on Exploration

1. Pick three documents: two about cooking and one about sports. Compute their BoW vectors by hand, then calculate the dot product for each pair. Verify that the cooking documents have the highest dot product. Now compute cosine similarity and confirm the same ranking.

2. Write a pair of sentences with identical words in different order that mean very different things (e.g., "the teacher failed the student" vs "the student failed the teacher"). Compute their BoW vectors and cosine similarity. The similarity will be 1.0 despite different meanings. This is the most important limitation to internalize.

3. Take one of the cooking documents from exercise 1 and double every word count (imagine the document repeated twice). Compute Euclidean distance and cosine similarity to the other cooking document. Euclidean distance increases (the vectors are further apart), but cosine similarity stays the same (the angle has not changed). This demonstrates why cosine similarity is preferred for text.

---

## Live Code

```rust
use std::collections::HashMap;
use std::collections::BTreeSet;

// --- Compare documents using dot product and cosine similarity ---
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

fn document_to_bow(doc: &str, vocabulary: &[String]) -> Vec<f64> {
    let mut word_counts: HashMap<String, usize> = HashMap::new();
    for word in tokenize(doc) {
        *word_counts.entry(word).or_insert(0) += 1;
    }
    vocabulary.iter()
        .map(|word| *word_counts.get(word).unwrap_or(&0) as f64)
        .collect()
}

fn dot_product(vec_a: &[f64], vec_b: &[f64]) -> f64 {
    vec_a.iter().zip(vec_b.iter()).map(|(a, b)| a * b).sum()
}

fn magnitude(vec: &[f64]) -> f64 {
    vec.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn cosine_similarity(vec_a: &[f64], vec_b: &[f64]) -> f64 {
    let dot = dot_product(vec_a, vec_b);
    let mag_a = magnitude(vec_a);
    let mag_b = magnitude(vec_b);
    if mag_a == 0.0 || mag_b == 0.0 {
        return 0.0;
    }
    dot / (mag_a * mag_b)
}

fn euclidean_distance(vec_a: &[f64], vec_b: &[f64]) -> f64 {
    vec_a.iter().zip(vec_b.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt()
}

fn main() {
    // --- Corpus with clear topic clusters ---
    let corpus: Vec<&str> = vec![
        "The cat sat on the warm mat by the fire",
        "A small cat napped on the soft mat",
        "The stock market index fell sharply on Monday",
        "Market traders were worried about the falling index",
        "The cat watched the fish in the bowl",
    ];

    let labels = vec!["Cat-mat (A)", "Cat-mat (B)", "Finance (C)", "Finance (D)", "Cat-fish (E)"];

    let vocabulary = build_vocabulary(&corpus);
    let vectors: Vec<Vec<f64>> = corpus.iter()
        .map(|doc| document_to_bow(doc, &vocabulary))
        .collect();

    println!("=== Corpus ({} documents, {} vocab words) ===", corpus.len(), vocabulary.len());
    for (label, doc) in labels.iter().zip(corpus.iter()) {
        println!("  {}: \"{}\"", label, doc);
    }
    println!();


    // --- Pairwise similarity matrix ---
    println!("=== Cosine Similarity Matrix ===\n");
    let col_width = 14;
    print!("  {:>width$}", "", width = col_width);
    for label in &labels {
        print!("  {:>width$}", label, width = col_width);
    }
    println!();

    let mut sim_matrix: Vec<Vec<f64>> = Vec::new();
    for (i, (label_i, vec_i)) in labels.iter().zip(vectors.iter()).enumerate() {
        let mut row: Vec<f64> = Vec::new();
        print!("  {:>width$}", label_i, width = col_width);
        for (j, vec_j) in vectors.iter().enumerate() {
            let sim = cosine_similarity(vec_i, vec_j);
            row.push((sim * 1000.0).round() / 1000.0);
            print!("  {:>width$.3}", sim, width = col_width);
        }
        sim_matrix.push(row);
        println!();
    }
    println!();

    let label_refs: Vec<&str> = labels.iter().copied().collect();
    show_chart(&heatmap_chart(
        "Document Cosine Similarity Matrix",
        &label_refs,
        &label_refs,
        &sim_matrix,
    ));


    // --- Top similar pairs ---
    println!("=== Most and Least Similar Pairs ===\n");
    let mut pairs: Vec<(&str, &str, f64)> = Vec::new();
    for i in 0..corpus.len() {
        for j in (i + 1)..corpus.len() {
            let sim = cosine_similarity(&vectors[i], &vectors[j]);
            pairs.push((labels[i], labels[j], sim));
        }
    }

    pairs.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

    println!("  Ranked by cosine similarity (high to low):");
    for (label_a, label_b, sim) in &pairs {
        let bar = "#".repeat((sim * 30.0) as usize);
        println!("    {} vs {}: {:.3}  |{}", label_a, label_b, sim, bar);
    }
    println!();


    // --- Cosine vs Euclidean: the length problem ---
    println!("=== Cosine vs Euclidean Distance ===\n");
    println!("  What happens when we double one document's word counts?\n");

    let doc_short = "The cat sat on the mat";
    let doc_long = "The cat sat on the mat The cat sat on the mat The cat sat on the mat";

    let test_corpus = vec![doc_short, doc_long, corpus[1]];
    let vocab_test = build_vocabulary(&test_corpus);
    let vec_short = document_to_bow(doc_short, &vocab_test);
    let vec_long = document_to_bow(doc_long, &vocab_test);
    let vec_other = document_to_bow(corpus[1], &vocab_test);

    let cos_short = cosine_similarity(&vec_short, &vec_other);
    let cos_long = cosine_similarity(&vec_long, &vec_other);
    let euc_short = euclidean_distance(&vec_short, &vec_other);
    let euc_long = euclidean_distance(&vec_long, &vec_other);

    println!("  Original:  \"{}\"", doc_short);
    println!("  Tripled:   \"{}\" (repeated 3x)", doc_short);
    println!("  Compare to: \"{}\"", corpus[1]);
    println!();
    println!("  {:<20} {:>12} {:>12}", "Metric", "Original", "Tripled");
    println!("  {}", "-".repeat(44));
    println!("  {:<20} {:>12.3} {:>12.3}", "Cosine similarity", cos_short, cos_long);
    println!("  {:<20} {:>12.3} {:>12.3}", "Euclidean distance", euc_short, euc_long);
    println!();
    println!("  Cosine similarity stays stable despite length change.");
    println!("  Euclidean distance increases because the vector is longer.");
    println!("  This is why cosine similarity is preferred for text comparison.");
    println!();


    // --- The fundamental BoW failure: word order ---
    println!("{}", "=".repeat(55));
    println!("=== WHERE BoW COMPARISON FAILS: WORD ORDER ===");
    println!("{}", "=".repeat(55));
    println!();

    let order_pairs = vec![
        ("dog bites man", "man bites dog"),
        ("the teacher failed the student", "the student failed the teacher"),
        ("she saw him leave", "he saw her leave"),
    ];

    for (sent_a, sent_b) in &order_pairs {
        let vocab_pair = build_vocabulary(&[sent_a, sent_b]);
        let vec_a = document_to_bow(sent_a, &vocab_pair);
        let vec_b = document_to_bow(sent_b, &vocab_pair);
        let sim = cosine_similarity(&vec_a, &vec_b);

        println!("  A: \"{}\"", sent_a);
        println!("  B: \"{}\"", sent_b);
        println!("  BoW vectors identical? {}", vec_a == vec_b);
        println!("  Cosine similarity: {:.3}", sim);
        println!("  Same meaning? NO — word order changes who does what to whom.");
        println!();
    }

    println!("  BoW represents text as an UNORDERED collection of words.");
    println!("  It captures WHAT words appear, not HOW they are arranged.");
    println!("  For tasks where order matters (sentiment, intent, narrative),");
    println!("  BoW similarity will mislead. This motivates sequence-aware");
    println!("  representations explored in later phases.");
    println!();


    // --- When BoW comparison works well ---
    println!("=== WHERE BoW COMPARISON WORKS WELL ===\n");

    let topic_corpus: Vec<&str> = vec![
        "Python is a popular programming language for data science",
        "Data science uses Python and machine learning algorithms",
        "The cake recipe requires flour sugar eggs and butter",
        "Mix the flour and sugar then add the eggs to bake a cake",
    ];

    let topic_labels = vec!["Tech (A)", "Tech (B)", "Cooking (C)", "Cooking (D)"];
    let topic_vocab = build_vocabulary(&topic_corpus);
    let topic_vecs: Vec<Vec<f64>> = topic_corpus.iter()
        .map(|doc| document_to_bow(doc, &topic_vocab))
        .collect();

    println!("  Documents:");
    for (label, doc) in topic_labels.iter().zip(topic_corpus.iter()) {
        println!("    {}: \"{}\"", label, doc);
    }
    println!();

    println!("  Cosine similarities:");
    for i in 0..topic_corpus.len() {
        for j in (i + 1)..topic_corpus.len() {
            let sim = cosine_similarity(&topic_vecs[i], &topic_vecs[j]);
            let same_topic = (i < 2 && j < 2) || (i >= 2 && j >= 2);
            let marker = if same_topic { " <-- same topic" } else { "" };
            println!("    {} vs {}: {:.3}{}", topic_labels[i], topic_labels[j], sim, marker);
        }
    }
    println!();
    println!("  BoW works well for TOPIC detection: documents about the same");
    println!("  subject share vocabulary, and cosine similarity captures this.");
    println!("  It fails for MEANING distinctions that depend on word order.");
}
```

---

## Key Takeaways

- **Cosine similarity measures the angle between document vectors**, not their distance. This makes it robust to differences in document length — a short email and a long report about the same topic will score high.
- **BoW comparison works well for topic detection.** Documents about cooking share cooking words; documents about finance share finance words. Cosine similarity reliably separates topics based on vocabulary overlap.
- **BoW comparison fails when word order carries meaning.** "Dog bites man" and "Man bites dog" produce identical vectors and a cosine similarity of 1.0, despite meaning very different things. This is the central limitation of the Bag of Words model.
- **BoW ignores order but captures presence.** This tradeoff is acceptable for many tasks (search, clustering, topic modeling) but insufficient for tasks requiring sequential understanding (sentiment analysis, question answering, translation). Recognizing when BoW is enough — and when it is not — is a key skill in NLP.
