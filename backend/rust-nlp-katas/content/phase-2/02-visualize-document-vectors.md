# Visualize Document Vectors

> Phase 2 — Bag of Words (BoW) | Kata 2.2

---

## Concept & Intuition

### What problem are we solving?

A BoW vector is a list of numbers. But what does it **look like**? When we say "documents are points in a high-dimensional space," that sounds abstract. This kata makes it concrete. By visualizing document vectors — even in ASCII — we build intuition for what "similarity in vector space" actually means.

When two documents share many of the same words, their vectors point in similar directions. When they share nothing, their vectors are orthogonal. Visualization makes this tangible before we ever compute a similarity metric.

### Why naive/earlier approaches fail

Looking at raw BoW vectors — long lists of numbers — tells you almost nothing at a glance:

- `[0, 1, 0, 0, 2, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]` — Is this similar to another document? Which words matter? How sparse is it?

Without visualization, you cannot develop intuition about the geometry of your text representations. You need a way to **see** the structure: which dimensions are active, which are zero, and how documents relate to each other.

### Mental models

- **The bookshelf**: Each word in the vocabulary is a shelf slot. Each document places books (word counts) on some shelves and leaves the rest empty. Similar documents fill similar shelves.
- **The fingerprint**: A BoW vector is a document's fingerprint — a pattern of peaks (frequent words) and flat zones (absent words). Comparing fingerprints reveals kinship.
- **Sparse skyline**: If you plot a BoW vector as a bar chart, most bars are height zero. The skyline is mostly flat with a few spikes. This sparsity pattern is the defining visual signature of BoW.

### Visual explanations

```
Document: "the cat sat on the mat"
Vocabulary: [cat, dog, mat, on, sat, the]

BoW vector: [1, 0, 1, 1, 1, 2]

ASCII bar chart:

  cat  |##          (1)
  dog  |            (0)
  mat  |##          (1)
  on   |##          (1)
  sat  |##          (1)
  the  |####        (2)

Compare with: "the dog ran"
BoW vector: [0, 1, 0, 0, 0, 1]

  cat  |            (0)
  dog  |##          (1)
  mat  |            (0)
  on   |            (0)
  sat  |            (0)
  the  |##          (1)

Notice: only "the" overlaps. The bar charts look very different.
The sparse regions (zeros) dominate both vectors.
```

---

## Hands-on Exploration

1. Pick three documents: two about the same topic and one about something completely different. Build their BoW vectors and draw the bar charts by hand. Notice how the two related documents share active dimensions while the unrelated one has peaks in different positions.

2. Count the zero entries in each vector and compute the sparsity percentage. Now imagine the vocabulary has 10,000 words and each document uses about 20 unique words. What would the sparsity be? (Answer: 99.8%). Think about what this means for storage and computation.

3. Look at the bar charts from exercise 1. Without computing any formula, use the visual overlap of the bars to guess which two documents are most similar. This visual intuition is exactly what cosine similarity formalizes (coming in Kata 2.3).

---

## Live Code

```rust
use std::collections::HashMap;
use std::collections::BTreeSet;

// --- Visualize BoW document vectors with ASCII ---
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

// --- Corpus: mix of related and unrelated topics ---
fn main() {
    let corpus: Vec<&str> = vec![
        "The cat sat on the mat near the window",
        "A cat napped on a soft mat",
        "The stock market crashed on Monday morning",
        "Scientists discovered a new species of deep sea fish",
    ];

    let labels = vec!["Cat-mat (1)", "Cat-mat (2)", "Finance", "Science"];

    let vocabulary = build_vocabulary(&corpus);
    let matrix: Vec<Vec<usize>> = corpus.iter()
        .map(|doc| document_to_bow(doc, &vocabulary))
        .collect();

    println!("=== Vocabulary ({} words) ===", vocabulary.len());
    println!("  {:?}", vocabulary);
    println!();


    // --- ASCII bar chart for each document ---
    fn print_bar_chart(label: &str, vector: &[usize], vocabulary: &[String], bar_char: &str, scale: usize) {
        println!("--- {} ---", label);
        let max_word_len = vocabulary.iter().map(|w| w.len()).max().unwrap_or(0);
        for (word, &count) in vocabulary.iter().zip(vector.iter()) {
            if count > 0 {
                let bar: String = bar_char.repeat(count * scale);
                println!("  {:>width$}  |{} ({})", word, bar, count, width = max_word_len);
            } else {
                println!("  {:>width$}  |", word, width = max_word_len);
            }
        }
        println!();
    }

    println!("=== Document Vectors as Bar Charts ===\n");
    for (label, vector) in labels.iter().zip(matrix.iter()) {
        print_bar_chart(label, vector, &vocabulary, "#", 2);

        // Rich bar chart for each document vector
        let active_words: Vec<&str> = vocabulary.iter().zip(vector.iter())
            .filter(|(_, &c)| c > 0)
            .map(|(w, _)| w.as_str())
            .collect();
        let active_counts: Vec<f64> = vector.iter()
            .filter(|&&c| c > 0)
            .map(|&c| c as f64)
            .collect();
        show_chart(&bar_chart(
            &format!("Document Vector: {}", label),
            &active_words,
            "Word Count",
            &active_counts,
            "#3b82f6",
        ));
    }


    // --- Sparsity analysis per document ---
    println!("=== Sparsity Statistics ===");
    println!("  {:<16} {:>10} {:>10} {:>10} {:>10}", "Document", "Non-zero", "Zero", "Total", "Sparsity");
    println!("  {}", "-".repeat(56));
    for (label, vector) in labels.iter().zip(matrix.iter()) {
        let nonzero = vector.iter().filter(|&&v| v > 0).count();
        let zero = vector.iter().filter(|&&v| v == 0).count();
        let total = vector.len();
        let sparsity = (zero as f64 / total as f64) * 100.0;
        println!("  {:<16} {:>10} {:>10} {:>10} {:>9.1}%", label, nonzero, zero, total, sparsity);
    }
    println!();

    // Rich chart: sparsity comparison across documents
    let sparsity_values: Vec<f64> = matrix.iter()
        .map(|vec| {
            let zero_count = vec.iter().filter(|&&v| v == 0).count();
            ((zero_count as f64 / vec.len() as f64) * 1000.0).round() / 10.0
        })
        .collect();

    let label_refs: Vec<&str> = labels.iter().copied().collect();
    show_chart(&bar_chart(
        "Sparsity Comparison Across Documents",
        &label_refs,
        "Sparsity (%)",
        &sparsity_values,
        "#f59e0b",
    ));


    // --- Sparse vector display (only non-zero entries) ---
    println!("=== Sparse Representation (non-zero entries only) ===");
    for (label, vector) in labels.iter().zip(matrix.iter()) {
        let active: Vec<String> = vocabulary.iter().zip(vector.iter())
            .filter(|(_, &c)| c > 0)
            .map(|(w, &c)| format!("{}:{}", w, c))
            .collect();
        println!("  {:<16} -> {{ {} }}", label, active.join(", "));
    }
    println!();
    println!("  In real systems, storing only non-zero entries saves enormous");
    println!("  memory. A 50,000-dim vector with 20 active words stores 20");
    println!("  entries instead of 50,000.");
    println!();


    // --- Side-by-side comparison ---
    fn print_side_by_side(label_a: &str, vec_a: &[usize], label_b: &str, vec_b: &[usize], vocabulary: &[String]) {
        println!("  {:<12} {:>12} {:>12}   Overlap", "Word", label_a, label_b);
        println!("  {}", "-".repeat(50));
        let mut overlaps = 0;
        for (word, (&a, &b)) in vocabulary.iter().zip(vec_a.iter().zip(vec_b.iter())) {
            let marker = if a > 0 && b > 0 {
                overlaps += 1;
                "  <-- shared"
            } else {
                ""
            };
            let a_str = if a > 0 { a.to_string() } else { ".".to_string() };
            let b_str = if b > 0 { b.to_string() } else { ".".to_string() };
            println!("  {:<12} {:>12} {:>12}{}", word, a_str, b_str, marker);
        }
        println!("\n  Shared dimensions: {}/{}", overlaps, vocabulary.len());
        println!();
    }


    println!("=== Side-by-Side Comparison ===\n");

    // Similar documents
    println!("Similar topics (both about cats and mats):");
    print_side_by_side(labels[0], &matrix[0], labels[1], &matrix[1], &vocabulary);

    // Different documents
    println!("Different topics (cat vs finance):");
    print_side_by_side(labels[0], &matrix[0], labels[2], &matrix[2], &vocabulary);

    // Completely different documents
    println!("Unrelated topics (finance vs science):");
    print_side_by_side(labels[2], &matrix[2], labels[3], &matrix[3], &vocabulary);


    // --- Visual overlap summary ---
    println!("=== Overlap Heatmap (shared non-zero dimensions) ===");
    print!("\n  {:>16}", "");
    for label in &labels {
        print!("  {:>14}", label);
    }
    println!();

    for (i, (label_i, vec_i)) in labels.iter().zip(matrix.iter()).enumerate() {
        print!("  {:>16}", label_i);
        for (j, vec_j) in matrix.iter().enumerate() {
            let shared = vec_i.iter().zip(vec_j.iter())
                .filter(|(&a, &b)| a > 0 && b > 0)
                .count();
            print!("  {:>14}", shared);
        }
        println!();
    }
    println!();
    println!("  Higher numbers = more vocabulary overlap = likely more similar.");
    println!("  This is a rough proxy — Kata 2.3 formalizes this with cosine similarity.");

    // Rich chart: overlap as grouped bar chart
    let colors = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444"];
    let mut overlap_datasets: Vec<(&str, Vec<f64>, &str)> = Vec::new();
    let mut overlap_data_storage: Vec<Vec<f64>> = Vec::new();

    for (i, vec_i) in matrix.iter().enumerate() {
        let overlaps: Vec<f64> = matrix.iter()
            .map(|vec_j| {
                vec_i.iter().zip(vec_j.iter())
                    .filter(|(&a, &b)| a > 0 && b > 0)
                    .count() as f64
            })
            .collect();
        overlap_data_storage.push(overlaps);
    }

    for (i, data) in overlap_data_storage.iter().enumerate() {
        overlap_datasets.push((labels[i], data.clone(), colors[i % colors.len()]));
    }

    let ds_refs: Vec<(&str, &[f64], &str)> = overlap_datasets.iter()
        .map(|(l, d, c)| (*l, d.as_slice(), *c))
        .collect();

    show_chart(&multi_bar_chart(
        "Vocabulary Overlap Between Documents",
        &label_refs,
        &ds_refs,
    ));
}
```

---

## Key Takeaways

- **BoW vectors are mostly zeros.** Even with a small corpus, sparsity dominates. In real corpora with tens of thousands of vocabulary words, a typical document uses fewer than 1% of them. Visualization makes this unmistakable.
- **Similar documents light up similar dimensions.** When two documents share topic words, their bar charts have peaks in the same positions. Different topics produce peaks in different positions with no overlap.
- **Sparse representations are essential.** Storing only the non-zero entries of a vector saves orders of magnitude in memory and makes computation practical at scale.
- **Visual intuition precedes formal metrics.** Before computing cosine similarity or dot products, you can see similarity by comparing which dimensions are active. The math formalizes what the eye already notices.
