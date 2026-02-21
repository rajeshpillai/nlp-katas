# Visualize Word Importance

> Phase 3 — TF-IDF | Kata 3.3

---

## Concept & Intuition

### What problem are we solving?

TF-IDF assigns a numerical score to every word in every document. But a table of numbers is hard to interpret. Which words does TF-IDF consider "important" for each document? Which words get suppressed? How do the importance rankings differ across documents in the same corpus?

Visualization makes the invisible visible. By plotting word importance as bar charts, we can immediately see the TF-IDF weighting at work: common words shrink to nothing, rare topic-specific words rise to the top, and each document gets a distinct "importance fingerprint."

This kata is about building intuition for what TF-IDF actually does — not through formulas, but through visual inspection of its output.

### Why naive/earlier approaches fail

Looking at raw TF-IDF numbers in a table is like reading a spreadsheet of pixel values instead of looking at an image. You can verify any individual number, but you cannot see the pattern. Humans are visual creatures — we spot trends, outliers, and clusters far faster in a chart than in a grid of decimals.

Without visualization:
- You might not notice that "the" gets a score of exactly 0 in every document
- You might miss that the top-ranked word in each document is the one that makes it unique
- You might not see how similar two documents' importance profiles are

### Mental models

- **TF-IDF as a spotlight**: Imagine each document on a dark stage. TF-IDF shines a spotlight on the words that make that document distinctive. Common words stay in shadow. Unique words glow brightly. The visualization shows you where the light falls.
- **Importance fingerprint**: Each document has a unique pattern of word importance scores. Like a fingerprint, this pattern identifies the document. Two documents about the same topic will have similar fingerprints. Visualization lets you compare fingerprints at a glance.
- **The long tail of language**: In any corpus, a few words appear everywhere (and get low TF-IDF), while most words appear rarely (and get high TF-IDF). Visualizing this distribution reveals the "long tail" of language — most of the vocabulary is rare, and rare words carry the most information.

### Visual explanations

```
WORD IMPORTANCE RANKING (what TF-IDF reveals)
==============================================

Document: "Solar energy powers millions of homes across the nation"

  Raw counts (BoW):                TF-IDF scores:
  ----------------                 ---------------
  the      ####  (common)          solar    ########  (rare, important)
  of       ###   (common)          energy   #######   (rare, important)
  solar    ##    (rare)            powers   ######    (rare, important)
  energy   ##    (rare)            homes    #####     (somewhat rare)
  powers   ##    (rare)            nation   #####     (somewhat rare)
  homes    ##    (rare)            millions ####      (somewhat rare)
  millions ##    (rare)            across   ##        (moderate)
  across   ##    (moderate)        of       #         (common, suppressed)
  nation   ##    (rare)            the               (ubiquitous, ZERO)

  BoW says "the" is the most          TF-IDF says "solar" and "energy"
  important word. That's useless.     are most important. That's useful.
```

```
COMPARING IMPORTANCE ACROSS DOCUMENTS
======================================

Doc 0 (about cooking):           Doc 1 (about astronomy):
  recipe   ########               telescope  #########
  flour    #######                 orbit      ########
  bake     ######                  planet     #######
  oven     #####                   galaxy     ######
  minutes  ####                    light      #####
  the                              the

Each document has its own "importance fingerprint."
The word "the" is invisible in both — TF-IDF killed it.
The top words immediately tell you what each document is about.
```

```
ASCII BAR CHART ANATOMY
========================

  word      |  score  |  bar
  ----------+---------+------------------------------
  recipe    |  0.231  |  ========================
  flour     |  0.198  |  ====================
  bake      |  0.154  |  ================
  ...       |  ...    |  ...
  the       |  0.000  |  (nothing — zero score)

  Bar length = score / max_score * bar_width
  This normalizes all bars relative to the top word.
```

---

## Hands-on Exploration

1. Take a document you care about (a favorite quote, a paragraph from a book, a news headline). Imagine placing it into a corpus of 100 random documents. Which words do you predict TF-IDF would rank highest? Which words would get suppressed? Write down your predictions, then check against the code output below.

2. Look at the output of the code and find two documents whose "importance fingerprints" overlap (they share high-scoring words). What does this tell you about their topical similarity? Now find two documents with completely different top words. How different are their topics?

3. Consider a failure case: a document containing a rare typo like "reciepe" instead of "recipe". TF-IDF will give "reciepe" a very high score because it appears in only one document. Is this word truly important? What does this reveal about the limitations of using rarity as a proxy for importance?

---

## Live Code

```rust
use std::collections::HashMap;
use std::collections::BTreeSet;
use std::collections::HashSet;

// --- Visualizing word importance with TF-IDF (ASCII bar charts) ---

fn main() {
    // A small corpus with distinct topics
    let corpus = vec![
        "the chef prepared a delicious pasta recipe with fresh tomato sauce",
        "the astronomer observed a distant galaxy through the new telescope",
        "the chef baked fresh bread and prepared a tomato basil soup recipe",
        "the telescope captured images of a distant planet orbiting a star",
        "the recipe called for fresh herbs and a pinch of sea salt",
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

    let mut vocab_set = BTreeSet::new();
    for doc in &documents {
        for word in doc {
            vocab_set.insert(word.clone());
        }
    }
    let vocabulary: Vec<String> = vocab_set.into_iter().collect();


    // --- Compute TF-IDF ---

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

    let idf = compute_idf(&documents, &vocabulary);


    // --- ASCII bar chart function ---

    fn ascii_bar(value: f64, max_value: f64, width: usize) -> String {
        if max_value == 0.0 {
            return String::new();
        }
        let length = ((value / max_value) * width as f64) as usize;
        "#".repeat(length)
    }


    // --- Visualize word importance for each document ---

    println!("{}", "=".repeat(70));
    println!("  WORD IMPORTANCE BY DOCUMENT (TF-IDF)");
    println!("{}", "=".repeat(70));

    for (i, doc) in documents.iter().enumerate() {
        let tf = compute_tf(doc);
        let mut tfidf: HashMap<String, f64> = HashMap::new();
        for word in &vocabulary {
            let tf_val = tf.get(word).copied().unwrap_or(0.0);
            let idf_val = idf.get(word).copied().unwrap_or(0.0);
            tfidf.insert(word.clone(), tf_val * idf_val);
        }

        // Sort by TF-IDF score descending
        let mut ranked: Vec<(String, f64)> = tfidf.iter()
            .map(|(w, &s)| (w.clone(), s))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Find max score for scaling bars
        let max_score = if ranked[0].1 > 0.0 { ranked[0].1 } else { 1.0 };

        println!("\n  Doc {}: \"{}\"", i, corpus[i]);
        println!("  {}", "-".repeat(64));

        let doc_words: HashSet<&String> = doc.iter().collect();
        for (word, score) in &ranked {
            if !doc_words.contains(word) {
                continue;
            }
            if *score == 0.0 {
                println!("    {:>14} | {:.4} |  (suppressed)", word, score);
            } else {
                let bar = ascii_bar(*score, max_score, 40);
                println!("    {:>14} | {:.4} | {}", word, score, bar);
            }
        }

        // Show suppressed words explicitly
        let suppressed: Vec<&String> = doc.iter()
            .filter(|w| tfidf.get(*w).copied().unwrap_or(0.0) == 0.0)
            .collect();
        if !suppressed.is_empty() {
            let mut unique_suppressed: Vec<&str> = suppressed.iter()
                .map(|s| s.as_str())
                .collect::<HashSet<_>>()
                .into_iter()
                .collect();
            unique_suppressed.sort();
            println!("    {:>14} |        |", "");
            println!("    {:>14} | {}", "[zero score]", unique_suppressed.join(", "));
        }

        // Rich bar chart for this document's top TF-IDF words
        let doc_words_set: HashSet<String> = doc.iter().cloned().collect();
        let mut doc_words_ranked: Vec<(String, f64)> = doc_words_set.iter()
            .filter(|w| tfidf.get(*w).copied().unwrap_or(0.0) > 0.0)
            .map(|w| (w.clone(), tfidf.get(w).copied().unwrap_or(0.0)))
            .collect();
        doc_words_ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top_words: Vec<&str> = doc_words_ranked.iter().take(10).map(|(w, _)| w.as_str()).collect();
        let top_scores: Vec<f64> = doc_words_ranked.iter().take(10).map(|(_, s)| (s * 10000.0).round() / 10000.0).collect();
        show_chart(&bar_chart(
            &format!("Top TF-IDF Words: Doc {}", i),
            &top_words,
            "TF-IDF Score",
            &top_scores,
            "#3b82f6",
        ));
    }


    // --- Compare top words across documents ---

    println!("\n{}", "=".repeat(70));
    println!("  TOP 3 MOST IMPORTANT WORDS PER DOCUMENT");
    println!("{}\n", "=".repeat(70));

    let mut all_top3_labels: Vec<String> = Vec::new();

    for (i, doc) in documents.iter().enumerate() {
        let tf = compute_tf(doc);
        let doc_words_set: HashSet<String> = doc.iter().cloned().collect();
        let mut tfidf_doc: Vec<(String, f64)> = doc_words_set.iter()
            .map(|w| {
                let score = tf.get(w).copied().unwrap_or(0.0) * idf.get(w).copied().unwrap_or(0.0);
                (w.clone(), score)
            })
            .collect();
        tfidf_doc.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top3: Vec<(String, f64)> = tfidf_doc.into_iter().take(3).collect();
        let words_str: String = top3.iter()
            .map(|(w, s)| format!("\"{}\" ({:.3})", w, s))
            .collect::<Vec<_>>()
            .join(", ");
        println!("  Doc {}: {}", i, words_str);

        for (w, _) in &top3 {
            if !all_top3_labels.contains(w) {
                all_top3_labels.push(w.clone());
            }
        }
    }

    println!();
    println!("  Notice how the top words immediately reveal each document's topic.");
    println!("  Documents 0, 2, 4 share cooking words. Documents 1, 3 share astronomy words.");

    // Rich bar chart comparing top words across all documents
    let doc_colors = ["#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6"];
    let mut all_top3_datasets: Vec<(&str, Vec<f64>, &str)> = Vec::new();
    let mut dataset_storage: Vec<Vec<f64>> = Vec::new();

    for (i, doc) in documents.iter().enumerate() {
        let tf = compute_tf(doc);
        let doc_words_set: HashSet<String> = doc.iter().cloned().collect();
        let mut tfidf_doc: HashMap<String, f64> = HashMap::new();
        for w in &doc_words_set {
            let score = tf.get(w).copied().unwrap_or(0.0) * idf.get(w).copied().unwrap_or(0.0);
            tfidf_doc.insert(w.clone(), score);
        }
        let scores: Vec<f64> = all_top3_labels.iter()
            .map(|w| (tfidf_doc.get(w).copied().unwrap_or(0.0) * 10000.0).round() / 10000.0)
            .collect();
        dataset_storage.push(scores);
    }

    let label_strs: Vec<String> = (0..documents.len()).map(|i| format!("Doc {}", i)).collect();
    for (i, data) in dataset_storage.iter().enumerate() {
        all_top3_datasets.push((
            label_strs[i].as_str(),
            data.clone(),
            doc_colors[i % doc_colors.len()],
        ));
    }

    let ds_refs: Vec<(&str, &[f64], &str)> = all_top3_datasets.iter()
        .map(|(l, d, c)| (*l, d.as_slice(), *c))
        .collect();
    let top3_label_refs: Vec<&str> = all_top3_labels.iter().map(|s| s.as_str()).collect();

    show_chart(&multi_bar_chart(
        "Top 3 Words Per Document (TF-IDF Scores)",
        &top3_label_refs,
        &ds_refs,
    ));


    // --- Show the IDF spectrum ---

    println!("\n{}", "=".repeat(70));
    println!("  IDF SPECTRUM: FROM COMMON TO RARE");
    println!("{}\n", "=".repeat(70));

    let mut idf_sorted: Vec<(&String, &f64)> = idf.iter().collect();
    idf_sorted.sort_by(|a, b| a.1.partial_cmp(b.1).unwrap());
    let max_idf = idf.values().cloned().fold(1.0f64, f64::max);

    for (word, &score) in &idf_sorted {
        let doc_freq = documents.iter()
            .filter(|doc| doc.contains(*word))
            .count();
        let bar = ascii_bar(score, max_idf, 35);
        let freq_label = format!("{}/{} docs", doc_freq, documents.len());
        println!("  {:>14}  IDF={:.3}  ({})  {}", word, score, freq_label, bar);
    }

    // Rich bar chart for the IDF spectrum
    let idf_words: Vec<&str> = idf_sorted.iter().map(|(w, _)| w.as_str()).collect();
    let idf_scores: Vec<f64> = idf_sorted.iter().map(|(_, &s)| (s * 1000.0).round() / 1000.0).collect();
    show_chart(&bar_chart(
        "IDF Spectrum: From Common to Rare",
        &idf_words,
        "IDF Score",
        &idf_scores,
        "#10b981",
    ));


    // --- Show importance distribution within a single document ---

    println!("\n{}", "=".repeat(70));
    println!("  IMPORTANCE DISTRIBUTION: DOC 0 (detailed)");
    println!("{}\n", "=".repeat(70));

    let doc_idx = 0;
    let doc = &documents[doc_idx];
    let tf = compute_tf(doc);
    let doc_words_set: HashSet<String> = doc.iter().cloned().collect();
    let mut tfidf_doc: HashMap<String, f64> = HashMap::new();
    for w in &doc_words_set {
        let score = tf.get(w).copied().unwrap_or(0.0) * idf.get(w).copied().unwrap_or(0.0);
        tfidf_doc.insert(w.clone(), score);
    }

    let mut ranked: Vec<(String, f64)> = tfidf_doc.iter()
        .map(|(w, &s)| (w.clone(), s))
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let max_score = if ranked[0].1 > 0.0 { ranked[0].1 } else { 1.0 };
    let total_tfidf: f64 = tfidf_doc.values().sum();

    println!("  Document: \"{}\"", corpus[doc_idx]);
    println!("  Total TF-IDF mass: {:.4}\n", total_tfidf);

    let mut cumulative = 0.0;
    for (word, score) in &ranked {
        cumulative += score;
        let pct = if total_tfidf > 0.0 { score / total_tfidf * 100.0 } else { 0.0 };
        let cum_pct = if total_tfidf > 0.0 { cumulative / total_tfidf * 100.0 } else { 0.0 };
        let bar = ascii_bar(*score, max_score, 30);
        let status = if *score == 0.0 { " <-- ZERO (appears everywhere)" } else { "" };
        println!("  {:>14}  {:.4}  ({:5.1}%)  cumul: {:5.1}%  {}{}",
            word, score, pct, cum_pct, bar, status);
    }

    println!("\n  The top few words carry most of the TF-IDF \"mass\".");
    println!("  Common words contribute nothing. This is by design.");

    // Rich bar chart for Doc 0 importance distribution
    let dist_words: Vec<&str> = ranked.iter().map(|(w, _)| w.as_str()).collect();
    let dist_scores: Vec<f64> = ranked.iter().map(|(_, s)| (s * 10000.0).round() / 10000.0).collect();
    show_chart(&bar_chart(
        &format!("Importance Distribution: Doc {} (TF-IDF Scores)", doc_idx),
        &dist_words,
        "TF-IDF Score",
        &dist_scores,
        "#f59e0b",
    ));


    // --- Side-by-side comparison of two documents ---

    println!("\n{}", "=".repeat(70));
    println!("  SIDE-BY-SIDE: COOKING DOC vs ASTRONOMY DOC");
    println!("{}\n", "=".repeat(70));

    fn get_tfidf_for_doc(
        doc_idx: usize,
        documents: &[Vec<String>],
        idf: &HashMap<String, f64>,
    ) -> HashMap<String, f64> {
        let doc = &documents[doc_idx];
        let total = doc.len() as f64;
        let mut counts: HashMap<String, f64> = HashMap::new();
        for word in doc {
            *counts.entry(word.clone()).or_insert(0.0) += 1.0;
        }
        let doc_words_set: HashSet<String> = doc.iter().cloned().collect();
        let mut result = HashMap::new();
        for w in &doc_words_set {
            let tf_val = counts.get(w).copied().unwrap_or(0.0) / total;
            let idf_val = idf.get(w).copied().unwrap_or(0.0);
            result.insert(w.clone(), tf_val * idf_val);
        }
        result
    }

    let cooking = get_tfidf_for_doc(0, &documents, &idf);
    let astronomy = get_tfidf_for_doc(1, &documents, &idf);

    let mut cooking_ranked: Vec<(String, f64)> = cooking.iter()
        .map(|(w, &s)| (w.clone(), s))
        .collect();
    cooking_ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut astro_ranked: Vec<(String, f64)> = astronomy.iter()
        .map(|(w, &s)| (w.clone(), s))
        .collect();
    astro_ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let max_len = cooking_ranked.len().max(astro_ranked.len());
    println!("  {:^32}  |  {:^32}", "Doc 0 (cooking)", "Doc 1 (astronomy)");
    println!("  {}  |  {}", "-".repeat(32), "-".repeat(32));

    let cooking_max = if cooking_ranked.is_empty() { 1.0 } else { cooking_ranked[0].1 };
    let astro_max = if astro_ranked.is_empty() { 1.0 } else { astro_ranked[0].1 };

    for idx in 0..max_len {
        let left = if idx < cooking_ranked.len() {
            let (w, s) = &cooking_ranked[idx];
            let bar = ascii_bar(*s, cooking_max, 14);
            format!("{:>10} {:.3} {}", w, s, bar)
        } else {
            String::new()
        };
        let right = if idx < astro_ranked.len() {
            let (w, s) = &astro_ranked[idx];
            let bar = ascii_bar(*s, astro_max, 14);
            format!("{:>10} {:.3} {}", w, s, bar)
        } else {
            String::new()
        };
        println!("  {:<32}  |  {:<32}", left, right);
    }

    println!();
    println!("  Each document has a completely different importance profile.");
    println!("  The visualization makes topical differences immediately obvious.");
    println!("  TF-IDF creates a unique \"fingerprint\" for each document's content.");

    // Rich bar chart: side-by-side comparison of cooking vs astronomy
    let cooking_top: Vec<(String, f64)> = cooking_ranked.iter().take(6).cloned().collect();
    let astro_top: Vec<(String, f64)> = astro_ranked.iter().take(6).cloned().collect();
    let mut all_compare_words: Vec<String> = Vec::new();
    for (w, _) in &cooking_top {
        if !all_compare_words.contains(w) {
            all_compare_words.push(w.clone());
        }
    }
    for (w, _) in &astro_top {
        if !all_compare_words.contains(w) {
            all_compare_words.push(w.clone());
        }
    }

    let compare_refs: Vec<&str> = all_compare_words.iter().map(|s| s.as_str()).collect();
    let cooking_data: Vec<f64> = all_compare_words.iter()
        .map(|w| (cooking.get(w).copied().unwrap_or(0.0) * 10000.0).round() / 10000.0)
        .collect();
    let astro_data: Vec<f64> = all_compare_words.iter()
        .map(|w| (astronomy.get(w).copied().unwrap_or(0.0) * 10000.0).round() / 10000.0)
        .collect();

    show_chart(&multi_bar_chart(
        "Side-by-Side: Cooking (Doc 0) vs Astronomy (Doc 1)",
        &compare_refs,
        &[
            ("Doc 0 (cooking)", &cooking_data, "#3b82f6"),
            ("Doc 1 (astronomy)", &astro_data, "#ef4444"),
        ],
    ));
}
```

---

## Key Takeaways

- **Visualization reveals what numbers hide.** A table of TF-IDF scores is hard to interpret, but an ASCII bar chart immediately shows which words matter and which are suppressed.
- **Each document has a unique importance fingerprint.** The top-ranked words in each document reveal its topic at a glance. Documents about similar topics share similar fingerprints.
- **Common words vanish under TF-IDF.** Words like "the" and "a" that appear in every document get a score of zero. The visualization makes this suppression obvious and dramatic.
- **TF-IDF is a relevance heuristic, not "understanding."** High TF-IDF does not mean a word is semantically important -- it means the word is statistically distinctive. Typos, jargon, and names all get high scores simply because they are rare, regardless of whether they carry real meaning.
