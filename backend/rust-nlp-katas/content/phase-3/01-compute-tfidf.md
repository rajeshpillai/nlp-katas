# Compute TF-IDF Manually

> Phase 3 — TF-IDF | Kata 3.1

---

## Concept & Intuition

### What problem are we solving?

In Phase 2, we built Bag of Words (BoW) representations — raw counts of how often each word appears in a document. But raw counts treat all words equally. The word "the" appearing 10 times in a document does not make that document about "the." Meanwhile, the word "photosynthesis" appearing once is far more informative about the document's topic.

TF-IDF (Term Frequency — Inverse Document Frequency) solves this by asking two questions simultaneously:

1. **How often does this word appear in this document?** (Term Frequency)
2. **How rare is this word across all documents?** (Inverse Document Frequency)

Words that appear frequently in one document but rarely across the corpus get high scores. Words that appear everywhere get suppressed. This gives us a measure of **relevance**, not just presence.

### Why naive/earlier approaches fail

Raw word counts (BoW) have a fundamental flaw: **common words dominate the representation**.

Consider three documents about different topics. The words "the", "is", "and", "of" will have the highest counts in all three documents. When you compare these documents using raw counts, the similarity is driven by these shared function words, not by the topic-specific content words.

Stopword removal helps partially, but it requires a manually curated list and is language-specific. TF-IDF solves this more elegantly — it lets the corpus itself determine which words are informative.

### Mental models

- **TF-IDF as a signal-to-noise filter**: TF is the signal (this word matters in this document), IDF is the noise reduction (discount words that appear everywhere). Multiply them together and you get a cleaned-up importance score.
- **Rarity as information**: A word that appears in every document tells you nothing about any specific document. A word that appears in only one document tells you a lot. IDF quantifies this intuition using logarithms.
- **Three regimes of IDF**: A word in 1 of 100 documents has high IDF. A word in 50 of 100 has moderate IDF. A word in all 100 has IDF of zero — it carries no discriminative power.

### Visual explanations

```
THE TF-IDF FORMULA
==================

  TF-IDF(t, d) = TF(t, d) x IDF(t)

Where:
  TF(t, d)  = count of term t in document d / total terms in document d
  IDF(t)    = log( total number of documents / number of documents containing t )


EXAMPLE: Corpus of 3 documents
-------------------------------

  Doc 1: "the cat sat on the mat"
  Doc 2: "the dog sat on the rug"
  Doc 3: "the cat chased the dog"

  Word        Doc1  Doc2  Doc3  | Docs containing | IDF = log(3/n)
  --------    ----  ----  ----  | --------------- | --------------
  "the"        2     2     2   |        3         | log(3/3) = 0.00
  "cat"        1     0     1   |        2         | log(3/2) = 0.41
  "sat"        1     1     0   |        2         | log(3/2) = 0.41
  "on"         1     1     0   |        2         | log(3/2) = 0.41
  "mat"        1     0     0   |        1         | log(3/1) = 1.10
  "dog"        0     1     1   |        2         | log(3/2) = 0.41
  "rug"        0     1     0   |        1         | log(3/1) = 1.10
  "chased"     0     0     1   |        1         | log(3/1) = 1.10

  Notice: "the" gets IDF = 0 — it appears everywhere, so it's worthless.
          "mat", "rug", "chased" get high IDF — they're unique to one document.

TF-IDF FOR "cat" IN DOC 1:
  TF  = 1/6 = 0.167
  IDF = log(3/2) = 0.405
  TF-IDF = 0.167 x 0.405 = 0.068

TF-IDF FOR "the" IN DOC 1:
  TF  = 2/6 = 0.333
  IDF = log(3/3) = 0.000
  TF-IDF = 0.333 x 0.000 = 0.000   <-- suppressed despite high frequency!
```

```
WHY LOGARITHM IN IDF?
======================

Without log:    IDF("mat") = 3/1 = 3.0     IDF("cat") = 3/2 = 1.5
With log:       IDF("mat") = log(3/1) = 1.1  IDF("cat") = log(3/2) = 0.4

The logarithm compresses the scale. Without it, a word appearing in 1 of
1,000,000 documents would get IDF = 1,000,000 — an absurd weight. The log
keeps values manageable:

  log(1,000,000 / 1) = 13.8     (high, but not absurd)
  log(1,000,000 / 500,000) = 0.7  (moderate)
  log(1,000,000 / 1,000,000) = 0  (zero — as expected)

  Documents     Raw ratio     log(ratio)
  containing    (N/df)        ln(N/df)
  ----------    ---------     ----------
       1        1000000.0       13.82
      10         100000.0       11.51
     100          10000.0        9.21
    1000           1000.0        6.91
   10000            100.0        4.61
  100000             10.0        2.30
  500000              2.0        0.69
 1000000              1.0        0.00
```

---

## Hands-on Exploration

1. Take the three-document corpus shown above ("the cat sat on the mat", "the dog sat on the rug", "the cat chased the dog"). Compute by hand: TF for every word in Document 2, then IDF for every word in the vocabulary, then multiply to get TF-IDF. Which word has the highest TF-IDF in Document 2? Why does that make intuitive sense?

2. Create a fourth document that contains only common words: "the the the on on". Compute its TF-IDF vector. What happens? What does this tell you about documents composed entirely of frequent words?

3. Consider the word "cat" which appears in Doc 1 and Doc 3. Its IDF is the same regardless of which document we're computing TF-IDF for. Now imagine adding 10 more documents that all mention "cat". How would this change the IDF of "cat"? At what point does "cat" become as useless as "the" for distinguishing documents?

---

## Live Code

```rust
use std::collections::HashMap;
use std::collections::BTreeSet;

// --- Computing TF-IDF from scratch using only Rust stdlib ---

fn main() {
    // Our small corpus
    let corpus = vec![
        "the cat sat on the mat",
        "the dog sat on the rug",
        "the cat chased the dog",
    ];

    // Step 1: Tokenize each document into words
    let documents: Vec<Vec<String>> = corpus.iter()
        .map(|doc| doc.to_lowercase().split_whitespace().map(|s| s.to_string()).collect())
        .collect();

    println!("=== CORPUS ===\n");
    for (i, doc) in documents.iter().enumerate() {
        println!("  Doc {}: {:?}", i, doc);
    }

    // Step 2: Build vocabulary (all unique words across corpus)
    let mut vocab_set = BTreeSet::new();
    for doc in &documents {
        for word in doc {
            vocab_set.insert(word.clone());
        }
    }
    let vocabulary: Vec<String> = vocab_set.into_iter().collect();
    println!("\nVocabulary ({} words): {:?}\n", vocabulary.len(), vocabulary);

    // Step 3: Compute Term Frequency (TF)
    // TF(t, d) = count of t in d / total words in d

    fn compute_tf(document: &[String]) -> HashMap<String, f64> {
        let word_count = document.len() as f64;
        let mut counts: HashMap<String, f64> = HashMap::new();
        for word in document {
            *counts.entry(word.clone()).or_insert(0.0) += 1.0;
        }
        for val in counts.values_mut() {
            *val /= word_count;
        }
        counts
    }

    println!("=== TERM FREQUENCY (TF) ===\n");
    let mut tf_scores: Vec<HashMap<String, f64>> = Vec::new();
    for (i, doc) in documents.iter().enumerate() {
        let tf = compute_tf(doc);
        println!("  Doc {} ({} words):", i, doc.len());
        for word in &vocabulary {
            let score = tf.get(word).copied().unwrap_or(0.0);
            if score > 0.0 {
                let count = doc.iter().filter(|w| *w == word).count();
                println!("    {:>10}: count={}, TF={:.4}", word, count, score);
            }
        }
        println!();
        tf_scores.push(tf);
    }

    // Step 4: Compute Inverse Document Frequency (IDF)
    // IDF(t) = log(N / df(t))
    // where N = total documents, df(t) = documents containing term t

    let n = documents.len() as f64;
    let mut idf_scores: HashMap<String, f64> = HashMap::new();
    for word in &vocabulary {
        let doc_freq = documents.iter()
            .filter(|doc| doc.contains(word))
            .count() as f64;
        idf_scores.insert(word.clone(), (n / doc_freq).ln());
    }

    println!("=== INVERSE DOCUMENT FREQUENCY (IDF) ===\n");
    println!("  Total documents (N): {}\n", documents.len());
    for word in &vocabulary {
        let doc_freq = documents.iter()
            .filter(|doc| doc.contains(word))
            .count();
        println!("  {:>10}: appears in {}/{} docs,  IDF = log({}/{}) = {:.4}",
            word, doc_freq, documents.len(), documents.len(), doc_freq, idf_scores[word]);
    }

    // Step 5: Compute TF-IDF = TF * IDF
    println!("\n=== TF-IDF SCORES ===\n");

    let mut tfidf_all: Vec<HashMap<String, f64>> = Vec::new();
    let colors = ["#3b82f6", "#10b981", "#f59e0b"];

    for (i, doc) in documents.iter().enumerate() {
        println!("  Doc {}: \"{}\"", i, corpus[i]);
        let mut tfidf: HashMap<String, f64> = HashMap::new();
        for word in &vocabulary {
            let tf_val = tf_scores[i].get(word).copied().unwrap_or(0.0);
            let idf_val = idf_scores[word];
            tfidf.insert(word.clone(), tf_val * idf_val);
        }

        // Show non-zero scores sorted by value (descending)
        let mut nonzero: Vec<(String, f64)> = tfidf.iter()
            .filter(|(_, &s)| s > 0.0)
            .map(|(w, &s)| (w.clone(), s))
            .collect();
        nonzero.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        for (word, score) in &nonzero {
            let tf_val = tf_scores[i].get(word).copied().unwrap_or(0.0);
            println!("    {:>10}: TF={:.4} x IDF={:.4} = TF-IDF={:.4}",
                word, tf_val, idf_scores[word], score);
        }

        // Show suppressed words
        let zeros: Vec<&str> = tfidf.iter()
            .filter(|(w, &s)| s == 0.0 && doc.contains(w))
            .map(|(w, _)| w.as_str())
            .collect();
        if !zeros.is_empty() {
            let mut zeros_sorted = zeros.clone();
            zeros_sorted.sort();
            println!("    [suppressed]: {} (IDF=0, appear in all docs)",
                zeros_sorted.join(", "));
        }
        println!();

        // Visualize top TF-IDF words for this document
        if !nonzero.is_empty() {
            let chart_labels: Vec<&str> = nonzero.iter().map(|(w, _)| w.as_str()).collect();
            let chart_data: Vec<f64> = nonzero.iter().map(|(_, s)| (s * 10000.0).round() / 10000.0).collect();
            let title = format!("TF-IDF Scores - Doc {}: \"{}\"",
                i, &corpus[i][..corpus[i].len().min(30)]);
            show_chart(&bar_chart(
                &title,
                &chart_labels,
                "TF-IDF",
                &chart_data,
                colors[i % 3],
            ));
        }

        tfidf_all.push(tfidf);
    }

    // Step 6: Show the full document-term matrix
    println!("=== DOCUMENT-TERM MATRIX (TF-IDF) ===\n");

    let header: String = vocabulary.iter()
        .map(|w| format!("{:>10}", w))
        .collect::<Vec<_>>()
        .join("");
    println!("         {}", header);
    println!("         {}", "-".repeat(10 * vocabulary.len()));
    for (i, tfidf) in tfidf_all.iter().enumerate() {
        let mut row = format!("  Doc {}  ", i);
        for word in &vocabulary {
            let score = tfidf.get(word).copied().unwrap_or(0.0);
            row.push_str(&format!("{:10.4}", score));
        }
        println!("{}", row);
    }

    println!("\n=== KEY OBSERVATIONS ===\n");
    println!("  \"the\" has TF-IDF = 0 everywhere — it appears in ALL documents.");
    println!("  \"mat\", \"rug\", \"chased\" have the highest scores — they are UNIQUE.");
    println!("  TF-IDF automatically identifies what makes each document distinctive.");

    // Visualize IDF values across the vocabulary
    let mut idf_sorted_pairs: Vec<(&String, &f64)> = idf_scores.iter().collect();
    idf_sorted_pairs.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

    let idf_labels: Vec<&str> = idf_sorted_pairs.iter().map(|(w, _)| w.as_str()).collect();
    let idf_data: Vec<f64> = idf_sorted_pairs.iter().map(|(_, &s)| (s * 10000.0).round() / 10000.0).collect();
    show_chart(&bar_chart(
        "IDF Values Across Vocabulary",
        &idf_labels,
        "IDF",
        &idf_data,
        "#8b5cf6",
    ));
}
```

---

## Key Takeaways

- **TF-IDF combines two signals**: term frequency (local importance within a document) and inverse document frequency (global rarity across the corpus). Neither signal alone is sufficient.
- **Common words are automatically suppressed.** Words appearing in every document get IDF = 0, which zeroes out their TF-IDF score — no manual stopword list needed.
- **The logarithm in IDF is essential.** Without it, rare words would receive astronomically large weights that would dominate all other terms. The log compresses the scale to keep scores balanced.
- **TF-IDF is a relevance heuristic, not "understanding."** It measures statistical distinctiveness, not meaning. A word can have high TF-IDF simply because it is rare, even if it is not semantically important.
