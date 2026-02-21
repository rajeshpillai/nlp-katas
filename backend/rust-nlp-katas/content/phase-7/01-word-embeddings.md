# Train Small Embedding-Based Models

> Phase 7 — Small Neural Text Models | Kata 7.1

---

## Concept & Intuition

### What problem are we solving?

Every text representation we have built so far --- Bag of Words, TF-IDF --- treats each word as an **independent dimension**. The word "happy" and the word "joyful" are as unrelated as "happy" and "concrete." Their vectors share no information. This is a fundamental limitation: BoW and TF-IDF cannot capture that words have **relationships**.

Word embeddings solve this by representing each word as a **dense vector** --- a short list of real numbers (typically 50 to 300 dimensions) --- where **similar words have similar vectors**. Instead of a sparse vector with 10,000 dimensions (one per vocabulary word), each word gets a compact vector of, say, 50 dimensions. These dimensions are not human-interpretable, but they encode patterns learned from how words co-occur in text.

The core insight: **words that appear in similar contexts tend to have similar meanings.** The word "king" appears near "throne", "crown", "rule." The word "queen" appears near... the same words. Therefore, "king" and "queen" should have similar vectors. This is the **distributional hypothesis** --- meaning is approximated by context.

### Why earlier methods fail

With BoW or TF-IDF, consider these two documents:

- Doc A: "The **happy** child played in the park"
- Doc B: "The **joyful** kid ran in the garden"

These documents are semantically very similar --- they describe the same scene. But BoW/TF-IDF sees almost zero overlap: "happy" vs "joyful", "child" vs "kid", "played" vs "ran", "park" vs "garden" --- all completely different dimensions. The cosine similarity will be near zero, driven only by shared words like "the" and "in."

The fundamental problem: **one-hot representations have no notion of word similarity.** Every word is equidistant from every other word. "Cat" is as far from "kitten" as it is from "democracy." This is clearly wrong, and no amount of TF-IDF weighting can fix it --- the geometry itself is broken.

### Mental models

- **Sparse vs dense**: BoW gives each word a unique axis in a 10,000-dimensional space. Embeddings compress words into a shared 50-dimensional space where proximity encodes similarity. Think of it as going from "one locker per student" (sparse) to "students sitting in a room where friends sit nearby" (dense).
- **Co-occurrence as meaning**: If two words repeatedly appear near the same neighbors, they are probably related. "Doctor" and "nurse" both appear near "hospital", "patient", "treatment." Their embeddings will be close.
- **Vector arithmetic**: Good embeddings support analogies. If "king" - "man" + "woman" gives a vector closest to "queen", the embedding space has captured gender as a direction. This is an emergent property, not something explicitly programmed.

### Visual explanations

```
SPARSE (BoW) vs DENSE (Embedding) Representations
===================================================

BoW / One-Hot (10,000 dimensions, mostly zeros):

  "cat"    = [0, 0, 1, 0, 0, 0, ..., 0, 0, 0]   (1 in position 42)
  "kitten" = [0, 0, 0, 0, 0, 0, ..., 1, 0, 0]   (1 in position 7831)
  "dog"    = [0, 0, 0, 1, 0, 0, ..., 0, 0, 0]   (1 in position 103)

  Distance(cat, kitten) = sqrt(2) = 1.414
  Distance(cat, dog)    = sqrt(2) = 1.414
  All words are EQUIDISTANT. No similarity captured.


Dense Embedding (50 dimensions, all values used):

  "cat"    = [0.23, -0.11, 0.85, 0.03, ..., 0.42]
  "kitten" = [0.21, -0.09, 0.82, 0.05, ..., 0.39]
  "dog"    = [0.19, -0.15, 0.78, 0.01, ..., 0.35]
  "table"  = [-0.52, 0.73, -0.11, 0.88, ..., -0.21]

  Distance(cat, kitten) = 0.08   (very close!)
  Distance(cat, dog)    = 0.15   (close --- both animals)
  Distance(cat, table)  = 1.87   (far apart --- unrelated)

  Similar words cluster together in the embedding space.


HOW CO-OCCURRENCE BUILDS EMBEDDINGS
====================================

Corpus: "the cat sat on the mat . the dog sat on the rug ."

Co-occurrence window (size 2 --- look 2 words left/right):

  "cat" neighbors: [the, sat, on]
  "dog" neighbors: [the, sat, on]

  "cat" and "dog" share the same neighbors!
  Therefore, they get similar embedding vectors.

  "mat" neighbors: [the, on, .]
  "rug" neighbors: [the, on, .]

  "mat" and "rug" also share neighbors --> similar vectors.


VECTOR ARITHMETIC (Analogy)
=============================

      king - man + woman = ???

           man -----> king
            |          |
            |  (same   |
            |  offset) |
            |          |
          woman ----> queen

  The vector from "man" to "king" encodes "royalty".
  Adding that same vector to "woman" lands near "queen".
  This works because the embedding space organizes concepts
  along consistent directions.
```

---

## Hands-on Exploration

1. Take the sample corpus from the code below. For the word "cat", list all words that appear within a window of 2 positions. Do the same for "dog." Compare the neighbor lists. Which words share the most neighbors? Predict which word pairs will end up with the highest cosine similarity in the embedding space, then run the code to check.

2. Modify the corpus in the code to add more sentences about food (e.g., "the chef cooked a delicious meal", "the cook prepared fresh food"). Run the code again. Do "chef" and "cook" end up with similar vectors? What happens if you add only one food sentence versus five? This demonstrates how **corpus size and content directly determine embedding quality**.

3. The code computes word analogies using vector arithmetic (A - B + C = ?). Try creating your own analogy by choosing three words from the corpus. What happens when the analogy involves words that do not have strong co-occurrence patterns? Why do analogies fail with small corpora but work with large ones?

---

## Live Code

```rust
use std::collections::HashMap;

fn main() {
    // --- Word Embeddings from Co-occurrence (Pure Rust) ---
    // Build dense word vectors from a corpus using co-occurrence statistics.
    // No external crates.

    let mut rng = Rng::new(42);

    // ============================================================
    // VECTOR MATH HELPERS
    // ============================================================

    fn dot_product(v1: &[f64], v2: &[f64]) -> f64 {
        v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum()
    }

    fn magnitude(v: &[f64]) -> f64 {
        v.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    fn cosine_similarity(v1: &[f64], v2: &[f64]) -> f64 {
        let d = dot_product(v1, v2);
        let m1 = magnitude(v1);
        let m2 = magnitude(v2);
        if m1 == 0.0 || m2 == 0.0 { return 0.0; }
        d / (m1 * m2)
    }

    fn vector_add(v1: &[f64], v2: &[f64]) -> Vec<f64> {
        v1.iter().zip(v2.iter()).map(|(a, b)| a + b).collect()
    }

    fn vector_subtract(v1: &[f64], v2: &[f64]) -> Vec<f64> {
        v1.iter().zip(v2.iter()).map(|(a, b)| a - b).collect()
    }

    // ============================================================
    // CORPUS AND TOKENIZATION
    // ============================================================

    let corpus: Vec<&str> = vec![
        "the cat sat on the mat in the room",
        "the cat chased the mouse around the house",
        "a small cat slept on the soft mat",
        "the dog sat on the rug in the hall",
        "the dog chased the ball around the yard",
        "a big dog slept on the warm rug",
        "the king ruled the land with wisdom",
        "the queen ruled the kingdom with grace",
        "the king wore a golden crown",
        "the queen wore a silver crown",
        "the man walked to the market",
        "the woman walked to the store",
        "the boy played in the park",
        "the girl played in the garden",
        "the cat and the dog sat together",
        "the king and the queen ruled together",
    ];

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

    // Tokenize entire corpus
    let mut all_tokens: Vec<String> = Vec::new();
    for sentence in &corpus {
        all_tokens.extend(tokenize(sentence));
    }

    let mut vocab_set: Vec<String> = all_tokens.iter().cloned().collect::<std::collections::BTreeSet<_>>().into_iter().collect();
    vocab_set.sort();
    let vocab = vocab_set;
    let word_to_idx: HashMap<String, usize> = vocab.iter().enumerate().map(|(i, w)| (w.clone(), i)).collect();
    let vocab_size = vocab.len();

    println!("{}", "=".repeat(60));
    println!("CORPUS AND VOCABULARY");
    println!("{}", "=".repeat(60));
    println!("  Sentences:     {}", corpus.len());
    println!("  Total tokens:  {}", all_tokens.len());
    println!("  Vocabulary:    {} unique words", vocab_size);
    println!("  Words: {:?}", vocab);
    println!();

    // ============================================================
    // STEP 1: BUILD CO-OCCURRENCE MATRIX
    // ============================================================

    let window_size: usize = 2;

    // Initialize co-occurrence matrix as HashMap of HashMaps
    let mut cooccurrence: HashMap<String, HashMap<String, usize>> = HashMap::new();
    for w in &vocab {
        cooccurrence.insert(w.clone(), HashMap::new());
    }

    // Slide window across each sentence
    for sentence in &corpus {
        let tokens = tokenize(sentence);
        for (i, target) in tokens.iter().enumerate() {
            let start = if i >= window_size { i - window_size } else { 0 };
            let end = (i + window_size + 1).min(tokens.len());
            for j in start..end {
                if i != j {
                    let context = &tokens[j];
                    *cooccurrence.get_mut(target).unwrap().entry(context.clone()).or_insert(0) += 1;
                }
            }
        }
    }

    println!("{}", "=".repeat(60));
    println!("CO-OCCURRENCE MATRIX (sample rows)");
    println!("{}", "=".repeat(60));

    let focus_words: Vec<&str> = vec!["cat", "dog", "king", "queen", "mat", "rug", "man", "woman"]
        .into_iter().filter(|w| word_to_idx.contains_key(*w)).collect();

    for word in &focus_words {
        let wstr = word.to_string();
        if let Some(counts) = cooccurrence.get(&wstr) {
            let mut neighbors: Vec<(&String, &usize)> = counts.iter().collect();
            neighbors.sort_by(|a, b| b.1.cmp(a.1));
            let neighbor_str: String = neighbors.iter().take(8)
                .map(|(w, c)| format!("{}:{}", w, c)).collect::<Vec<_>>().join(", ");
            println!("  {:>8} -> {}", word, neighbor_str);
        }
    }
    println!();

    // ============================================================
    // STEP 2: DIMENSIONALITY REDUCTION VIA RANDOM PROJECTION
    // ============================================================

    let embed_dim: usize = 20;

    // Create random projection matrix (vocab_size x embed_dim)
    let mut projection: Vec<Vec<f64>> = Vec::new();
    for _ in 0..vocab_size {
        let row: Vec<f64> = (0..embed_dim).map(|_| if rng.next_f64() < 0.5 { -1.0 } else { 1.0 }).collect();
        projection.push(row);
    }

    // Build the co-occurrence vector for each word, then project
    let mut embeddings: HashMap<String, Vec<f64>> = HashMap::new();
    for word in &vocab {
        let co = cooccurrence.get(word).unwrap();
        let full_vec: Vec<f64> = vocab.iter().map(|w| *co.get(w).unwrap_or(&0) as f64).collect();

        // Project: embedding = full_vec @ projection
        let mut embed: Vec<f64> = vec![0.0; embed_dim];
        for d in 0..embed_dim {
            let mut val = 0.0;
            for i in 0..vocab_size {
                val += full_vec[i] * projection[i][d];
            }
            embed[d] = val;
        }

        // Normalize to unit length
        let mag = magnitude(&embed);
        if mag > 0.0 {
            embed = embed.iter().map(|x| x / mag).collect();
        }

        embeddings.insert(word.clone(), embed);
    }

    println!("{}", "=".repeat(60));
    println!("EMBEDDINGS ({}-dimensional, normalized)", embed_dim);
    println!("{}", "=".repeat(60));
    for word in &focus_words {
        let embed = &embeddings[*word];
        let vec_str: String = embed.iter().take(6).map(|x| format!("{:+.3}", x)).collect::<Vec<_>>().join(", ");
        println!("  {:>8} -> [{}, ...]", word, vec_str);
    }
    println!();

    // ============================================================
    // STEP 3: MEASURE WORD SIMILARITY
    // ============================================================

    println!("{}", "=".repeat(60));
    println!("WORD SIMILARITY (cosine similarity)");
    println!("{}", "=".repeat(60));

    let test_pairs: Vec<(&str, &str)> = vec![
        ("cat", "dog"), ("cat", "mat"), ("king", "queen"),
        ("king", "man"), ("man", "woman"), ("mat", "rug"),
        ("boy", "girl"), ("cat", "king"), ("dog", "crown"),
    ];

    let test_pairs: Vec<(&str, &str)> = test_pairs.into_iter()
        .filter(|(a, b)| embeddings.contains_key(*a) && embeddings.contains_key(*b))
        .collect();

    let mut similarity_labels: Vec<String> = Vec::new();
    let mut similarity_values: Vec<f64> = Vec::new();
    for (w1, w2) in &test_pairs {
        let sim = cosine_similarity(&embeddings[*w1], &embeddings[*w2]);
        let bar_len = (sim.max(0.0) * 30.0) as usize;
        let bar: String = "#".repeat(bar_len);
        println!("  {:>8} <-> {:<8}  {:+.4}  {}", w1, w2, sim, bar);
        similarity_labels.push(format!("{}-{}", w1, w2));
        similarity_values.push((sim * 10000.0).round() / 10000.0);
    }
    println!();

    // --- Visualization: Word Pair Similarity ---
    {
        let labels: Vec<&str> = similarity_labels.iter().map(|s| s.as_str()).collect();
        show_chart(&bar_chart(
            "Word Pair Cosine Similarity (Embedding-Based)",
            &labels, "Cosine Similarity", &similarity_values, "#3b82f6",
        ));
    }

    // ============================================================
    // STEP 4: FIND MOST SIMILAR WORDS
    // ============================================================

    println!("{}", "=".repeat(60));
    println!("NEAREST NEIGHBORS");
    println!("{}", "=".repeat(60));

    fn most_similar(word: &str, embeddings: &HashMap<String, Vec<f64>>, vocab: &[String], top_n: usize) -> Vec<(String, f64)> {
        fn cosine_sim_inner(v1: &[f64], v2: &[f64]) -> f64 {
            let d: f64 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
            let m1: f64 = v1.iter().map(|x| x * x).sum::<f64>().sqrt();
            let m2: f64 = v2.iter().map(|x| x * x).sum::<f64>().sqrt();
            if m1 == 0.0 || m2 == 0.0 { return 0.0; }
            d / (m1 * m2)
        }
        if !embeddings.contains_key(word) { return Vec::new(); }
        let target = &embeddings[word];
        let mut scores: Vec<(String, f64)> = vocab.iter()
            .filter(|w| w.as_str() != word)
            .map(|w| (w.clone(), cosine_sim_inner(target, &embeddings[w])))
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scores.into_iter().take(top_n).collect()
    }

    let nn_words = ["cat", "king", "mat", "man"];
    let nn_colors = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6"];
    for (idx, word) in nn_words.iter().enumerate() {
        if !embeddings.contains_key(*word) { continue; }
        let neighbors = most_similar(word, &embeddings, &vocab, 5);
        let neighbor_str: String = neighbors.iter()
            .map(|(w, s)| format!("{} ({:+.3})", w, s)).collect::<Vec<_>>().join(", ");
        println!("  {:>8} -> {}", word, neighbor_str);

        // Visualization per word
        let nn_labels: Vec<&str> = neighbors.iter().map(|(w, _)| w.as_str()).collect();
        let nn_scores: Vec<f64> = neighbors.iter().map(|(_, s)| (*s * 1000.0).round() / 1000.0).collect();
        show_chart(&bar_chart(
            &format!("Nearest Neighbors: '{}'", word),
            &nn_labels, "Similarity", &nn_scores, nn_colors[idx % nn_colors.len()],
        ));
    }
    println!();

    // ============================================================
    // STEP 5: WORD ANALOGIES (Vector Arithmetic)
    // ============================================================

    println!("{}", "=".repeat(60));
    println!("WORD ANALOGIES (A - B + C = ?)");
    println!("{}", "=".repeat(60));

    fn analogy(a: &str, b: &str, c: &str, embeddings: &HashMap<String, Vec<f64>>, vocab: &[String], top_n: usize) -> Vec<(String, f64)> {
        fn cs(v1: &[f64], v2: &[f64]) -> f64 {
            let d: f64 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
            let m1: f64 = v1.iter().map(|x| x * x).sum::<f64>().sqrt();
            let m2: f64 = v2.iter().map(|x| x * x).sum::<f64>().sqrt();
            if m1 == 0.0 || m2 == 0.0 { return 0.0; }
            d / (m1 * m2)
        }
        fn v_add(v1: &[f64], v2: &[f64]) -> Vec<f64> { v1.iter().zip(v2.iter()).map(|(a, b)| a + b).collect() }
        fn v_sub(v1: &[f64], v2: &[f64]) -> Vec<f64> { v1.iter().zip(v2.iter()).map(|(a, b)| a - b).collect() }
        fn mag(v: &[f64]) -> f64 { v.iter().map(|x| x * x).sum::<f64>().sqrt() }

        if !embeddings.contains_key(a) || !embeddings.contains_key(b) || !embeddings.contains_key(c) {
            return Vec::new();
        }
        let mut result = v_add(&v_sub(&embeddings[a], &embeddings[b]), &embeddings[c]);
        let m = mag(&result);
        if m > 0.0 { result = result.iter().map(|x| x / m).collect(); }

        let exclude: std::collections::HashSet<&str> = [a, b, c].iter().copied().collect();
        let mut scores: Vec<(String, f64)> = vocab.iter()
            .filter(|w| !exclude.contains(w.as_str()))
            .map(|w| (w.clone(), cs(&result, &embeddings[w])))
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scores.into_iter().take(top_n).collect()
    }

    let analogies_list: Vec<(&str, &str, &str, &str)> = vec![
        ("king", "man", "woman", "queen"),
        ("queen", "woman", "man", "king"),
        ("cat", "mat", "rug", "dog"),
    ];

    for (a, b, c, expected) in &analogies_list {
        if !embeddings.contains_key(*a) || !embeddings.contains_key(*b) || !embeddings.contains_key(*c) { continue; }
        let results = analogy(a, b, c, &embeddings, &vocab, 3);
        let result_str: String = results.iter()
            .map(|(w, s)| format!("{} ({:+.3})", w, s)).collect::<Vec<_>>().join(", ");
        let got = results.first().map(|(w, _)| w.as_str()).unwrap_or("?");
        let match_str = if got == *expected { "MATCH" } else { "no match" };
        println!("  {} - {} + {} = ?", a, b, c);
        println!("    Expected: {}", expected);
        println!("    Got:      {}  [{}]", result_str, match_str);
        println!();
    }

    // ============================================================
    // STEP 6: WHY THIS WORKS --- THE DISTRIBUTIONAL HYPOTHESIS
    // ============================================================

    println!("{}", "=".repeat(60));
    println!("WHY THIS WORKS");
    println!("{}", "=".repeat(60));
    println!();
    println!("  The Distributional Hypothesis:");
    println!("    'Words that occur in similar contexts tend to have");
    println!("     similar meanings.'  (Harris, 1954)");
    println!();
    println!("  In our corpus:");

    for (w1, w2) in &[("cat", "dog"), ("king", "queen"), ("mat", "rug")] {
        let w1s = w1.to_string();
        let w2s = w2.to_string();
        if let (Some(c1), Some(c2)) = (cooccurrence.get(&w1s), cooccurrence.get(&w2s)) {
            let n1: std::collections::HashSet<&String> = c1.keys().collect();
            let n2: std::collections::HashSet<&String> = c2.keys().collect();
            let shared: Vec<&&String> = n1.intersection(&n2).collect();
            let mut shared_sorted: Vec<String> = shared.iter().map(|s| s.to_string()).collect();
            shared_sorted.sort();
            shared_sorted.truncate(8);
            println!("    '{}' and '{}' share {} context words: {:?}", w1, w2, shared.len(), shared_sorted);
        }
    }

    println!();
    println!("  Shared context words --> similar co-occurrence vectors");
    println!("  --> similar embeddings --> high cosine similarity");
    println!();
    println!("  This is not 'understanding.' This is pattern matching");
    println!("  over word neighborhoods. But it captures something real");
    println!("  about how language organizes meaning.");
}
```

---

## Key Takeaways

- **Word embeddings are dense vectors that encode word relationships.** Unlike BoW/TF-IDF where every word is an independent dimension, embeddings place similar words close together in a shared vector space. "Cat" and "kitten" become neighbors, not strangers.
- **Co-occurrence statistics are the foundation of word embeddings.** Words that appear in the same contexts get similar vectors. This is the distributional hypothesis: meaning is approximated by the company a word keeps. No labeled data or explicit definitions are needed.
- **Vector arithmetic can encode semantic relationships.** Good embedding spaces organize concepts along consistent directions, enabling analogies like "king - man + woman = queen." This is an emergent property of learning from co-occurrence, not something explicitly programmed.
- **Neural models learn representations, not rules.** Unlike handcrafted features (stopword lists, stemming rules), embeddings are learned from data. The model discovers structure in language rather than being told what structure to look for. This is the paradigm shift from classical to neural NLP.
