# Compare Neural vs TF-IDF Models

> Phase 7 — Small Neural Text Models | Kata 7.3

---

## Concept & Intuition

### What problem are we solving?

We now have two fundamentally different ways to represent text: **TF-IDF** (from Phase 3) and **embedding-based representations** (from Kata 7.1). Both turn documents into vectors. Both enable similarity measurement. But they capture **different kinds of information**, and each has strengths the other lacks.

This kata puts them head-to-head on the same documents. We will see cases where embeddings detect similarity that TF-IDF misses entirely (synonyms, paraphrases), and cases where TF-IDF outperforms embeddings (exact keyword matching, interpretability). The goal is not to declare a winner --- it is to understand **when to use which representation**.

### Why earlier methods fail

TF-IDF represents each document as a sparse vector where each dimension corresponds to a specific word. Two documents are similar only if they share **the exact same words**. This is powerful for keyword search but blind to meaning:

- "The automobile drove fast" and "The car sped quickly" share **zero** content words. TF-IDF similarity = near zero. But any human knows these sentences mean the same thing.

Embedding-based representations solve this by mapping each word to a dense vector where synonyms are close together. When we average word embeddings to represent a document, "automobile" and "car" contribute similar vectors, so the two documents end up close in embedding space.

But embeddings are not strictly better. They lose **specificity**:

- "Python programming language" and "Python snake species" would get similar embedding-averaged vectors because the word "python" has the same embedding regardless of context. TF-IDF, by contrast, can distinguish them through the presence of specific co-occurring keywords like "programming" vs "snake."

### Mental models

- **TF-IDF is a magnifying glass**: It sees exactly what words are on the page, with sharp detail. It cannot see meaning behind the words, but it never hallucinates relationships that are not there. High precision, limited recall.
- **Embeddings are blurry binoculars**: They see the broad semantic landscape --- that two sentences are about the same topic even with different words. But they lose fine detail. Two documents about very different things can look similar if they use related vocabulary.
- **The interpretability tradeoff**: With TF-IDF, you can point to exactly which words drove the similarity score. With embeddings, the score emerges from the interaction of 20 abstract dimensions --- correct but unexplainable. This matters in domains where decisions must be justified (medicine, law, finance).

### Visual explanations

```
TF-IDF vs EMBEDDING SIMILARITY
=================================

Doc A: "The car drove fast on the highway"
Doc B: "The automobile sped quickly on the road"
Doc C: "The stock market crashed today"

TF-IDF SPACE (each word = one dimension):

  Vocabulary: [automobile, car, crashed, drove, fast, highway,
               market, on, quickly, road, sped, stock, the, today]

  Doc A: [0, .18, 0, .18, .18, .26, 0, .07, 0, 0, 0, 0, .07, 0]
  Doc B: [.26, 0, 0, 0, 0, 0, 0, .07, .26, .26, .26, 0, .07, 0]
  Doc C: [0, 0, .26, 0, 0, 0, .26, 0, 0, 0, 0, .26, .07, .26]

  TF-IDF cosine(A, B) = LOW   (almost no shared words!)
  TF-IDF cosine(A, C) = LOW   (no shared content words)

  TF-IDF CANNOT see that A and B mean the same thing.
  "car" and "automobile" are different dimensions.


EMBEDDING SPACE (each word -> dense vector):

  embed("car")        = [0.8, 0.3, -0.1, ...]
  embed("automobile") = [0.7, 0.4, -0.2, ...]   <-- close to "car"!
  embed("drove")      = [0.2, 0.9,  0.1, ...]
  embed("sped")       = [0.3, 0.8,  0.0, ...]   <-- close to "drove"!
  embed("stock")      = [-0.5, -0.3, 0.8, ...]  <-- far from vehicles

  Doc vector = average of word embeddings

  Embedding cosine(A, B) = HIGH  (synonyms are close in embed space)
  Embedding cosine(A, C) = LOW   (semantically unrelated)

  Embeddings CAPTURE that A and B are paraphrases.


WHEN TF-IDF WINS
===================

  Query: "python programming language"

  Doc X: "Python is a popular programming language for data science"
  Doc Y: "The python snake is found in tropical forests"

  TF-IDF: "programming" and "language" match Doc X exactly.
          Doc Y has no match for these specific keywords.
          TF-IDF correctly ranks Doc X higher.

  Embeddings: "python" has ONE vector regardless of meaning.
              Both docs contribute a similar "python" component.
              Embeddings may fail to distinguish them clearly.

  TF-IDF's exact matching is a feature, not a bug.


THE TRADEOFF
==============

                    TF-IDF          Embeddings
                    ------          ----------
  Synonyms          misses them     captures them
  Paraphrases       misses them     captures them
  Exact keywords    excellent       loses specificity
  Interpretability  fully clear     opaque
  Corpus needed     just the docs   large training corpus
  Dimensionality    sparse, high    dense, low
  Computation       fast (sparse)   fast (dense, small)
```

---

## Hands-on Exploration

1. Look at the document pairs in the code below. Before running it, predict which pairs will have higher similarity under TF-IDF vs embeddings. Focus on pairs where one document uses synonyms of the other. Run the code and check your predictions. Were there any surprises?

2. The code includes a "keyword search" experiment. Try different queries: one with exact terms from a document, and one that describes the same concept with different words. Note how TF-IDF excels at the first and fails at the second, while embeddings do the opposite. What kind of search engine would you build for a legal document database? For a casual Q&A system?

3. Examine the interpretability section of the output. For the TF-IDF similarity score, the code shows exactly which words contributed. For the embedding similarity, it cannot. Imagine you are building a medical system that recommends similar patient records. A doctor asks "Why did you say these two records are similar?" Which representation lets you answer that question?

---

## Live Code

```rust
use std::collections::HashMap;

fn main() {
    // --- Compare Neural Embeddings vs TF-IDF (Pure Rust) ---
    // Side-by-side comparison on the same documents.
    // No external crates --- everything from scratch.

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

    // ============================================================
    // TOKENIZER
    // ============================================================

    fn tokenize(text: &str) -> Vec<String> {
        let mut result = Vec::new();
        let mut current = String::new();
        for ch in text.to_lowercase().chars() {
            if ch.is_alphabetic() {
                current.push(ch);
            } else if !current.is_empty() {
                result.push(current.clone());
                current.clear();
            }
        }
        if !current.is_empty() { result.push(current); }
        result
    }

    // ============================================================
    // TRAINING CORPUS (for building embeddings)
    // ============================================================

    let training_corpus: Vec<&str> = vec![
        "the car drove fast on the highway",
        "the automobile sped quickly down the road",
        "a fast car raced on the track",
        "the vehicle moved quickly through traffic",
        "she drove her car to the store",
        "the automobile needed fuel for the road",
        "the cat sat on the warm mat",
        "the kitten slept on the soft rug",
        "a small cat chased the mouse",
        "the dog ran across the green yard",
        "the puppy played on the soft rug",
        "a big dog chased the ball",
        "the happy child played in the park",
        "the joyful kid ran in the garden",
        "a cheerful boy smiled at the sun",
        "the glad girl laughed with friends",
        "happiness filled the room with warmth",
        "joy spread through the entire family",
        "the chef cooked a delicious meal",
        "the cook prepared fresh food",
        "a tasty dish was served at dinner",
        "the meal was warm and satisfying",
        "the scientist studied the experiment results",
        "the researcher analyzed the data carefully",
        "an experiment revealed new findings",
        "the study produced interesting results",
        "the man walked to the market",
        "the woman walked to the store",
        "they went to the big store together",
        "he drove to the market in his car",
    ];

    // ============================================================
    // BUILD WORD EMBEDDINGS FROM TRAINING CORPUS
    // ============================================================

    let mut all_training_tokens: Vec<String> = Vec::new();
    for sentence in &training_corpus {
        all_training_tokens.extend(tokenize(sentence));
    }

    let embed_vocab: Vec<String> = {
        let set: std::collections::BTreeSet<String> = all_training_tokens.iter().cloned().collect();
        set.into_iter().collect()
    };
    let embed_word_to_idx: HashMap<String, usize> = embed_vocab.iter().enumerate().map(|(i, w)| (w.clone(), i)).collect();
    let embed_vocab_size = embed_vocab.len();

    // Build co-occurrence matrix
    let window_size: usize = 3;
    let mut cooccurrence: HashMap<String, HashMap<String, usize>> = HashMap::new();
    for w in &embed_vocab { cooccurrence.insert(w.clone(), HashMap::new()); }

    for sentence in &training_corpus {
        let tokens = tokenize(sentence);
        for (i, target) in tokens.iter().enumerate() {
            let start = if i >= window_size { i - window_size } else { 0 };
            let end = (i + window_size + 1).min(tokens.len());
            for j in start..end {
                if i != j {
                    *cooccurrence.get_mut(target).unwrap().entry(tokens[j].clone()).or_insert(0) += 1;
                }
            }
        }
    }

    // Random projection for embeddings
    let embed_dim: usize = 25;
    let mut proj_matrix: Vec<Vec<f64>> = Vec::new();
    for _ in 0..embed_vocab_size {
        let row: Vec<f64> = (0..embed_dim).map(|_| if rng.next_f64() < 0.5 { -1.0 } else { 1.0 }).collect();
        proj_matrix.push(row);
    }

    let mut word_embeddings: HashMap<String, Vec<f64>> = HashMap::new();
    for word in &embed_vocab {
        let co = cooccurrence.get(word).unwrap();
        let full_vec: Vec<f64> = embed_vocab.iter().map(|w| *co.get(w).unwrap_or(&0) as f64).collect();
        let mut embed: Vec<f64> = (0..embed_dim).map(|d| {
            (0..embed_vocab_size).map(|i| full_vec[i] * proj_matrix[i][d]).sum()
        }).collect();
        let mag = magnitude(&embed);
        if mag > 0.0 { embed = embed.iter().map(|x| x / mag).collect(); }
        word_embeddings.insert(word.clone(), embed);
    }

    println!("{}", "=".repeat(60));
    println!("SETUP COMPLETE");
    println!("{}", "=".repeat(60));
    println!("  Training corpus: {} sentences", training_corpus.len());
    println!("  Embedding vocabulary: {} words", embed_vocab_size);
    println!("  Embedding dimensions: {}", embed_dim);
    println!();

    // ============================================================
    // TEST DOCUMENTS
    // ============================================================

    let test_documents: Vec<(&str, &str)> = vec![
        ("D1", "the car drove fast on the highway"),
        ("D2", "the automobile sped quickly on the road"),
        ("D3", "the cat sat on the soft mat"),
        ("D4", "the kitten slept on the warm rug"),
        ("D5", "the happy child played in the park"),
        ("D6", "the joyful kid ran in the garden"),
        ("D7", "the scientist studied the experiment"),
        ("D8", "the chef cooked a delicious meal"),
    ];

    println!("{}", "=".repeat(60));
    println!("TEST DOCUMENTS");
    println!("{}", "=".repeat(60));
    for (name, text) in &test_documents {
        println!("  {}: {}", name, text);
    }
    println!();
    println!("  Paraphrase pairs: D1/D2 (vehicles), D3/D4 (pets), D5/D6 (kids)");
    println!("  Unrelated pairs:  D1/D7 (car vs science), D5/D8 (kids vs food)");
    println!();

    // ============================================================
    // METHOD 1: TF-IDF REPRESENTATION
    // ============================================================

    let test_tokenized: Vec<(&str, Vec<String>)> = test_documents.iter()
        .map(|(name, text)| (*name, tokenize(text))).collect();

    let tfidf_vocab: Vec<String> = {
        let set: std::collections::BTreeSet<String> = test_tokenized.iter()
            .flat_map(|(_, tokens)| tokens.iter().cloned()).collect();
        set.into_iter().collect()
    };
    let num_docs = test_documents.len();

    fn compute_tf(tokens: &[String]) -> HashMap<String, f64> {
        let total = tokens.len() as f64;
        let mut counts: HashMap<String, usize> = HashMap::new();
        for t in tokens { *counts.entry(t.clone()).or_insert(0) += 1; }
        counts.into_iter().map(|(w, c)| (w, c as f64 / total)).collect()
    }

    fn compute_idf(all_doc_tokens: &[(&str, Vec<String>)], vocabulary: &[String]) -> HashMap<String, f64> {
        let n = all_doc_tokens.len() as f64;
        let mut idf = HashMap::new();
        for word in vocabulary {
            let df = all_doc_tokens.iter().filter(|(_, tokens)| tokens.contains(word)).count() as f64;
            idf.insert(word.clone(), if df > 0.0 { (n / df).ln() } else { 0.0 });
        }
        idf
    }

    let idf = compute_idf(&test_tokenized, &tfidf_vocab);

    fn tfidf_vector(tokens: &[String], vocabulary: &[String], idf_scores: &HashMap<String, f64>) -> Vec<f64> {
        let tf = compute_tf(tokens);
        vocabulary.iter().map(|w| tf.get(w).unwrap_or(&0.0) * idf_scores.get(w).unwrap_or(&0.0)).collect()
    }

    let tfidf_vectors: HashMap<&str, Vec<f64>> = test_tokenized.iter()
        .map(|(name, tokens)| (*name, tfidf_vector(tokens, &tfidf_vocab, &idf))).collect();

    // ============================================================
    // METHOD 2: EMBEDDING REPRESENTATION
    // ============================================================

    fn document_embedding(tokens: &[String], embeddings: &HashMap<String, Vec<f64>>, dim: usize) -> Vec<f64> {
        let mut vec = vec![0.0; dim];
        let mut count = 0;
        for word in tokens {
            if let Some(emb) = embeddings.get(word) {
                for (i, v) in emb.iter().enumerate() { vec[i] += v; }
                count += 1;
            }
        }
        if count > 0 { vec = vec.iter().map(|x| x / count as f64).collect(); }
        vec
    }

    let embed_vectors: HashMap<&str, Vec<f64>> = test_tokenized.iter()
        .map(|(name, tokens)| (*name, document_embedding(tokens, &word_embeddings, embed_dim))).collect();

    // ============================================================
    // HEAD-TO-HEAD COMPARISON
    // ============================================================

    println!("{}", "=".repeat(60));
    println!("HEAD-TO-HEAD: TF-IDF vs EMBEDDINGS");
    println!("{}", "=".repeat(60));

    let comparison_pairs: Vec<(&str, &str, &str)> = vec![
        ("D1", "D2", "car/automobile (paraphrase)"),
        ("D3", "D4", "cat/kitten (paraphrase)"),
        ("D5", "D6", "happy/joyful (paraphrase)"),
        ("D1", "D3", "car vs cat (different topics)"),
        ("D5", "D8", "kids vs food (different topics)"),
        ("D1", "D7", "car vs science (different topics)"),
        ("D7", "D8", "science vs food (different topics)"),
    ];

    println!("\n  {:<10} {:<30} {:>8} {:>8}  Winner", "Pair", "Description", "TF-IDF", "Embed");
    println!("  {} {} {} {}  {}", "-".repeat(10), "-".repeat(30), "-".repeat(8), "-".repeat(8), "-".repeat(20));

    let mut chart_labels: Vec<String> = Vec::new();
    let mut chart_tfidf: Vec<f64> = Vec::new();
    let mut chart_embed: Vec<f64> = Vec::new();

    for (n1, n2, desc) in &comparison_pairs {
        let tfidf_sim = cosine_similarity(&tfidf_vectors[n1], &tfidf_vectors[n2]);
        let embed_sim = cosine_similarity(&embed_vectors[n1], &embed_vectors[n2]);

        chart_labels.push(format!("{}-{}", n1, n2));
        chart_tfidf.push((tfidf_sim * 10000.0).round() / 10000.0);
        chart_embed.push((embed_sim * 10000.0).round() / 10000.0);

        let is_paraphrase = desc.contains("paraphrase");
        let is_different = desc.contains("different");

        let winner = if is_paraphrase {
            if embed_sim > tfidf_sim { "EMBED" } else { "TF-IDF" }
        } else if is_different {
            if tfidf_sim < embed_sim { "TF-IDF" } else { "EMBED" }
        } else { "" };

        println!("  {:<10} {:<30} {:>8.4} {:>8.4}  {}", format!("{}-{}", n1, n2), desc, tfidf_sim, embed_sim, winner);
    }
    println!();

    // --- Visualization ---
    {
        let labels: Vec<&str> = chart_labels.iter().map(|s| s.as_str()).collect();
        show_chart(&multi_bar_chart(
            "TF-IDF vs Embedding Similarity (Head-to-Head)",
            &labels,
            &[
                ("TF-IDF", chart_tfidf.as_slice(), "#f59e0b"),
                ("Embedding", chart_embed.as_slice(), "#3b82f6"),
            ],
        ));
    }

    // ============================================================
    // WHERE EMBEDDINGS WIN: SYNONYM DETECTION
    // ============================================================

    println!("{}", "=".repeat(60));
    println!("WHERE EMBEDDINGS WIN: SYNONYM DETECTION");
    println!("{}", "=".repeat(60));
    println!();
    println!("  D1: 'the car drove fast on the highway'");
    println!("  D2: 'the automobile sped quickly on the road'");
    println!();

    let tfidf_sim_12 = cosine_similarity(&tfidf_vectors["D1"], &tfidf_vectors["D2"]);
    let embed_sim_12 = cosine_similarity(&embed_vectors["D1"], &embed_vectors["D2"]);

    println!("  TF-IDF similarity: {:.4}", tfidf_sim_12);
    println!("  Embedding similarity: {:.4}", embed_sim_12);
    println!();

    let d1_tokens: std::collections::HashSet<String> = test_tokenized.iter().find(|(n, _)| *n == "D1").unwrap().1.iter().cloned().collect();
    let d2_tokens: std::collections::HashSet<String> = test_tokenized.iter().find(|(n, _)| *n == "D2").unwrap().1.iter().cloned().collect();
    let shared: Vec<&String> = d1_tokens.intersection(&d2_tokens).collect();
    let d1_only: Vec<&String> = d1_tokens.difference(&d2_tokens).collect();
    let d2_only: Vec<&String> = d2_tokens.difference(&d1_tokens).collect();

    println!("  Shared words:  {:?}", shared);
    println!("  D1 only:       {:?}", d1_only);
    println!("  D2 only:       {:?}", d2_only);
    println!();
    println!("  TF-IDF sees almost no overlap because the content words differ.");
    println!("  'car' vs 'automobile', 'drove' vs 'sped', 'fast' vs 'quickly'");
    println!("  --- all different dimensions, zero contribution to similarity.");
    println!();
    println!("  Embeddings know these are synonyms because they appear in");
    println!("  similar contexts in the training corpus:");

    for (w1, w2) in &[("car", "automobile"), ("fast", "quickly")] {
        if word_embeddings.contains_key(*w1) && word_embeddings.contains_key(*w2) {
            let sim = cosine_similarity(&word_embeddings[*w1], &word_embeddings[*w2]);
            println!("    cosine('{}', '{}') = {:+.4}", w1, w2, sim);
        }
    }
    println!();

    // ============================================================
    // WHERE TF-IDF WINS: KEYWORD SEARCH
    // ============================================================

    println!("{}", "=".repeat(60));
    println!("WHERE TF-IDF WINS: KEYWORD SEARCH");
    println!("{}", "=".repeat(60));
    println!();

    let queries = ["scientist experiment", "chef cooked meal", "car highway fast"];
    println!("  Ranking documents for keyword queries:");
    println!();

    for query_text in &queries {
        let query_tokens = tokenize(query_text);
        let query_tfidf = tfidf_vector(&query_tokens, &tfidf_vocab, &idf);
        let query_embed = document_embedding(&query_tokens, &word_embeddings, embed_dim);

        let mut tfidf_scores: Vec<(&str, f64)> = test_documents.iter()
            .map(|(name, _)| (*name, cosine_similarity(&query_tfidf, &tfidf_vectors[name]))).collect();
        let mut embed_scores: Vec<(&str, f64)> = test_documents.iter()
            .map(|(name, _)| (*name, cosine_similarity(&query_embed, &embed_vectors[name]))).collect();

        tfidf_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        embed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        println!("  Query: '{}'", query_text);
        print!("    TF-IDF ranking:     ");
        for (name, score) in tfidf_scores.iter().take(3) { print!("{}({:.3}) ", name, score); }
        println!();
        print!("    Embedding ranking:  ");
        for (name, score) in embed_scores.iter().take(3) { print!("{}({:.3}) ", name, score); }
        println!();
        println!();
    }

    println!("  TF-IDF excels at exact keyword matching because each word");
    println!("  is its own dimension. If the query word appears in the");
    println!("  document, it contributes directly to the score.");
    println!();

    // ============================================================
    // INTERPRETABILITY COMPARISON
    // ============================================================

    println!("{}", "=".repeat(60));
    println!("INTERPRETABILITY: CAN YOU EXPLAIN THE SCORE?");
    println!("{}", "=".repeat(60));
    println!();

    let (n1, n2) = ("D1", "D2");
    let d1_text = test_documents.iter().find(|(n, _)| *n == "D1").unwrap().1;
    let d2_text = test_documents.iter().find(|(n, _)| *n == "D2").unwrap().1;
    println!("  Comparing {} and {}:", n1, n2);
    println!("    {}: '{}'", n1, d1_text);
    println!("    {}: '{}'", n2, d2_text);
    println!();

    println!("  TF-IDF explanation (word-by-word contribution to dot product):");
    let t1_tf = compute_tf(&test_tokenized.iter().find(|(n, _)| *n == n1).unwrap().1);
    let t2_tf = compute_tf(&test_tokenized.iter().find(|(n, _)| *n == n2).unwrap().1);
    let mut contributions: Vec<(String, f64, f64, f64)> = Vec::new();
    for word in &tfidf_vocab {
        let v1 = t1_tf.get(word).unwrap_or(&0.0) * idf.get(word).unwrap_or(&0.0);
        let v2 = t2_tf.get(word).unwrap_or(&0.0) * idf.get(word).unwrap_or(&0.0);
        let contrib = v1 * v2;
        if contrib > 0.0 { contributions.push((word.clone(), contrib, v1, v2)); }
    }
    contributions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for (word, contrib, v1, v2) in &contributions {
        println!("    '{}': TF-IDF_D1={:.4} * TF-IDF_D2={:.4} = {:.6}", word, v1, v2, contrib);
    }
    if contributions.is_empty() {
        println!("    (no shared content words with non-zero TF-IDF)");
    }

    println!();
    println!("  Embedding explanation:");
    let embed_sim_val = cosine_similarity(&embed_vectors[n1], &embed_vectors[n2]);
    println!("    Cosine similarity = {:.4}", embed_sim_val);
    println!("    Why? The embedding dimensions are abstract. We cannot");
    println!("    point to specific words or features that drove this score.");
    println!("    The similarity emerges from the interaction of {} opaque", embed_dim);
    println!("    dimensions. This is the cost of semantic richness.");
    println!();

    // ============================================================
    // SUMMARY SCORECARD
    // ============================================================

    println!("{}", "=".repeat(60));
    println!("SUMMARY: WHEN TO USE WHICH");
    println!("{}", "=".repeat(60));
    println!();
    println!("  TASK                         BEST METHOD     WHY");
    println!("  {}", "-".repeat(56));
    println!("  Synonym-aware search         Embeddings      Captures meaning, not words");
    println!("  Paraphrase detection         Embeddings      Different words, same idea");
    println!("  Exact keyword search         TF-IDF          Precise word matching");
    println!("  Explainable recommendations  TF-IDF          Word contributions are clear");
    println!("  Small corpus, no training    TF-IDF          Needs no pre-training");
    println!("  Cross-lingual similarity     Embeddings      With multilingual training");
    println!("  Document deduplication       Both            Use TF-IDF first, embed second");
    println!("  Topic clustering             Either          Depends on vocabulary overlap");
    println!();
    println!("  The right choice depends on your data, your task,");
    println!("  and whether you need to explain your results.");
    println!();
    println!("  In practice, many systems combine both:");
    println!("    1. TF-IDF for fast candidate retrieval (exact match)");
    println!("    2. Embeddings for re-ranking (semantic similarity)");
    println!("  This is the 'retrieve and re-rank' pattern used in");
    println!("  modern search engines and recommendation systems.");
}
```

---

## Key Takeaways

- **TF-IDF and embeddings capture different information.** TF-IDF measures lexical overlap --- whether documents share the same words. Embeddings measure semantic similarity --- whether documents express the same meaning. Neither is universally better; they answer different questions.
- **Embeddings detect synonyms and paraphrases that TF-IDF cannot.** Because word embeddings place "car" and "automobile" close together, embedding-based document similarity stays high even when no exact words are shared. This is the key advantage of dense representations over sparse ones.
- **TF-IDF offers interpretability that embeddings sacrifice.** With TF-IDF, you can point to exactly which words drove the similarity score and how much each contributed. With embeddings, the score is a black-box number emerging from abstract dimensions. In high-stakes domains, this transparency matters.
- **The best systems combine both approaches.** Use TF-IDF for fast, interpretable candidate retrieval, then use embeddings for semantic re-ranking. This "retrieve and re-rank" pattern captures the strengths of both representations while mitigating their individual weaknesses.
