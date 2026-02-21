# Measure How Stopword Removal Changes Document Similarity

> Phase 1 — Text Preprocessing | Kata 1.3

---

## Concept & Intuition

### What problem are we solving?

Some words appear in almost every English sentence: "the", "is", "at", "on", "a", "and". These are called **stopwords**. They're grammatically necessary but often carry little meaning for tasks like search and classification.

Removing stopwords seems obvious — less noise, better results. But it's not that simple. Stopwords can carry meaning ("to be or not to be" loses everything if you remove stopwords), and removing them changes how similar two documents appear to be. This kata measures that change directly.

### Why naive approaches fail

- Removing all stopwords from "the president" gives "president" — fine
- Removing all stopwords from "not good" gives "good" — meaning is **inverted**
- Removing all stopwords from "to be or not to be" gives "" — the entire sentence is destroyed
- "The Who" (band name) becomes "Who" or nothing

Stopword lists are blunt instruments. The same word can be a stopword in one context and critical in another. "It" is usually meaningless, but in "It was the best of times" it's a stylistic choice; in "IT department" it's an acronym.

### Mental models

- **Signal-to-noise ratio**: Stopwords are high-frequency, low-information words. Removing them is like turning down the background noise — the signal (content words) becomes clearer. But turn down too much and you lose the music.
- **Document fingerprint**: Without stopwords, documents are reduced to their content words. Two documents about the same topic will have more overlapping words, making similarity easier to detect.
- **The 80/20 of language**: Roughly 50% of any English text is made up of about 100 common words. Removing them cuts the text in half while preserving most of the topical content.

### Visual explanations

```
Original:      "The cat sat on the mat and the dog sat on the rug"
Content words:  "cat sat mat dog sat rug"

Original:      "The cat is on the mat and the dog is on the rug"
Content words:  "cat mat dog rug"

Without stopwords, both sentences clearly share a topic (animals + furniture).
With stopwords, the shared words (the, on, the, and, the) dominate but add
no topical information.

Word frequency in a typical document:
  the  ████████████████████████████  (most frequent)
  is   ██████████████████
  a    █████████████████
  of   ████████████████
  and  ███████████████
  ---  (stopword boundary)  ---
  cat  ███
  sat  ██
  mat  ██
  dog  █
  rug  █

The top words tell you nothing about the topic.
The bottom words tell you everything.
```

---

## Hands-on Exploration

1. Take two paragraphs about the same topic — count overlapping words with and without stopwords
2. Find a sentence where removing stopwords changes or destroys the meaning
3. Compare the word frequency distribution before and after stopword removal

---

## Live Code

```rust
use std::collections::HashMap;
use std::collections::HashSet;

// --- How stopword removal changes document similarity ---

fn stopwords() -> HashSet<&'static str> {
    let words = [
        "a", "an", "the", "and", "or", "but", "is", "are", "was", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "will", "would", "could", "should", "may", "might", "shall", "can",
        "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
        "into", "through", "during", "before", "after", "above", "below",
        "it", "its", "this", "that", "these", "those", "i", "you", "he",
        "she", "we", "they", "me", "him", "her", "us", "them", "my", "your",
        "his", "our", "their", "what", "which", "who", "when", "where", "how",
        "not", "no", "nor", "so", "if", "then", "than", "too", "very",
        "just", "about", "up", "out", "also", "each", "all", "both", "only",
    ];
    words.iter().copied().collect()
}

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

fn remove_stopwords(words: &[String], stops: &HashSet<&str>) -> Vec<String> {
    words.iter()
        .filter(|w| !stops.contains(w.as_str()))
        .cloned()
        .collect()
}

fn word_overlap_similarity(words1: &[String], words2: &[String]) -> f64 {
    let set1: HashSet<&str> = words1.iter().map(|s| s.as_str()).collect();
    let set2: HashSet<&str> = words2.iter().map(|s| s.as_str()).collect();
    if set1.is_empty() && set2.is_empty() {
        return 1.0;
    }
    if set1.is_empty() || set2.is_empty() {
        return 0.0;
    }
    let intersection = set1.intersection(&set2).count();
    let union = set1.union(&set2).count();
    intersection as f64 / union as f64
}

fn main() {
    let stops = stopwords();

    // === Document pairs to compare ===

    let doc_pairs: Vec<(&str, &str, &str)> = vec![
        (
            "The cat sat on the mat in the living room",
            "A cat was sitting on a mat in a room",
            "Same topic, different function words",
        ),
        (
            "The stock market crashed and investors lost millions of dollars",
            "The financial market experienced a crash and investors suffered large losses",
            "Same topic, partially overlapping vocabulary",
        ),
        (
            "The quick brown fox jumps over the lazy dog",
            "The weather forecast predicts rain for the entire week",
            "Completely different topics",
        ),
        (
            "The president announced the new policy at the conference",
            "The CEO announced the new strategy at the meeting",
            "Similar structure, different domain",
        ),
    ];

    println!("=== Stopword Removal and Document Similarity ===\n");

    let mut chart_labels: Vec<String> = Vec::new();
    let mut sims_with: Vec<f64> = Vec::new();
    let mut sims_without: Vec<f64> = Vec::new();

    for (doc1, doc2, description) in &doc_pairs {
        let words1 = tokenize(doc1);
        let words2 = tokenize(doc2);
        let content1 = remove_stopwords(&words1, &stops);
        let content2 = remove_stopwords(&words2, &stops);

        let sim_with = word_overlap_similarity(&words1, &words2);
        let sim_without = word_overlap_similarity(&content1, &content2);

        chart_labels.push(description.to_string());
        sims_with.push((sim_with * 1000.0).round() / 1000.0);
        sims_without.push((sim_without * 1000.0).round() / 1000.0);

        println!("--- {} ---", description);
        println!("  Doc 1: '{}'", doc1);
        println!("  Doc 2: '{}'", doc2);
        println!("  With stopwords:    {} + {} words, similarity = {:.3}",
            words1.len(), words2.len(), sim_with);
        println!("  Without stopwords: {} + {} words, similarity = {:.3}",
            content1.len(), content2.len(), sim_without);
        let change = sim_without - sim_with;
        let direction = if change > 0.0 { "increased" } else if change < 0.0 { "decreased" } else { "unchanged" };
        println!("  Change: {:+.3} ({})", change, direction);
        println!();
    }

    let label_refs: Vec<&str> = chart_labels.iter().map(|s| s.as_str()).collect();
    show_chart(&multi_bar_chart(
        "Similarity: With vs Without Stopwords",
        &label_refs,
        &[
            ("With stopwords", &sims_with, "#3b82f6"),
            ("Without stopwords", &sims_without, "#10b981"),
        ],
    ));


    // === Show what stopwords dominate ===

    println!("\n=== Word Frequency: Before and After ===\n");

    let text = "The president of the United States announced that the new \
economic policy would be implemented by the end of the year. The policy \
is expected to have a significant impact on the economy and the job \
market. The president also stated that the government would provide \
additional support for small businesses and workers who have been \
affected by the recent economic downturn.";

    let words = tokenize(text);
    let content = remove_stopwords(&words, &stops);

    println!("Total words: {}", words.len());
    println!("After removing stopwords: {}", content.len());
    let removed = words.len() - content.len();
    let pct = (removed as f64 / words.len() as f64) * 100.0;
    println!("Stopwords removed: {} ({:.0}%)", removed, pct);
    println!();

    // Count word frequencies
    let mut freq: HashMap<&str, usize> = HashMap::new();
    for w in &words {
        *freq.entry(w.as_str()).or_insert(0) += 1;
    }
    let mut freq_sorted: Vec<(&&str, &usize)> = freq.iter().collect();
    freq_sorted.sort_by(|a, b| b.1.cmp(a.1));

    println!("Top 10 words WITH stopwords:");
    for (word, count) in freq_sorted.iter().take(10) {
        let bar = "\u{2588}".repeat(*count * 2);
        let is_stop = if stops.contains(**word) { " (stopword)" } else { "" };
        println!("  {:<12} {:>3}  {}{}", word, count, bar, is_stop);
    }

    println!();

    let mut content_freq: HashMap<&str, usize> = HashMap::new();
    for w in &content {
        *content_freq.entry(w.as_str()).or_insert(0) += 1;
    }
    let mut content_sorted: Vec<(&&str, &usize)> = content_freq.iter().collect();
    content_sorted.sort_by(|a, b| b.1.cmp(a.1));

    println!("Top 10 words WITHOUT stopwords:");
    for (word, count) in content_sorted.iter().take(10) {
        let bar = "\u{2588}".repeat(*count * 2);
        println!("  {:<12} {:>3}  {}", word, count, bar);
    }


    // === Danger cases: when removal destroys meaning ===

    println!("\n\n=== When Stopword Removal Goes Wrong ===\n");

    let danger_cases = vec![
        "To be or not to be",
        "This is not good at all",
        "I will not go",
        "Let it be",
        "It is what it is",
        "The Who released a new album",
    ];

    for text in &danger_cases {
        let words = tokenize(text);
        let content = remove_stopwords(&words, &stops);
        let result = if content.is_empty() {
            "(empty)".to_string()
        } else {
            content.join(" ")
        };
        println!("  '{}'", text);
        println!("  -> '{}'", result);
        println!();
    }

    println!("Stopword removal is not always safe.");
    println!("Negation words ('not', 'no') carry critical meaning.");
    println!("Some phrases are MADE of stopwords.");
}
```

---

## Key Takeaways

- **Stopword removal often improves similarity detection.** By removing high-frequency, low-information words, the remaining content words better represent the document's topic.
- **The effect is not uniform.** For similar documents, stopword removal typically increases measured similarity. For dissimilar documents, the effect varies.
- **Stopwords dominate text.** Roughly 40-60% of tokens in typical English text are stopwords. Removing them dramatically reduces the representation size.
- **Stopword removal can destroy meaning.** Negation words ("not", "no"), phrases made of function words ("to be or not to be"), and some proper nouns ("The Who") are casualties of blind stopword removal. Always consider your task before applying this step.
