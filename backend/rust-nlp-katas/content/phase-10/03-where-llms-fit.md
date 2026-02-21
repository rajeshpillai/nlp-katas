# Where LLMs Fit in the NLP Stack

> Phase 10 — Modern NLP Pipelines (Awareness) | Kata 10.3

---

## Concept & Intuition

### What problem are we solving?

Large Language Models (LLMs) dominate today's NLP landscape, but they did not appear from nothing. Every component of an LLM -- tokenization, embeddings, attention, pretraining -- was developed incrementally over decades. Without understanding the stack beneath them, LLMs are black boxes. With that understanding, they become predictable engineering artifacts whose strengths and limitations follow logically from their design.

This kata maps the entire NLP journey from raw text to modern LLMs, showing exactly where each technique fits, what it contributes, and what it cannot do. The goal is not to hype LLMs but to place them accurately on a continuum of increasingly sophisticated text representations.

### Why earlier methods fail

They do not fail universally. They fail at specific things:

- **BoW/TF-IDF** fail at capturing word order and meaning. "Dog bites man" and "Man bites dog" are identical under BoW. But they succeed at fast, interpretable document comparison.
- **Static embeddings** (Word2Vec) fail at polysemy. The word "bank" gets one vector regardless of context. But they succeed at capturing word relationships (king - man + woman = queen).
- **RNNs** fail at long-range dependencies and parallel training. But they succeed at modeling short sequences.
- **Basic transformers** fail at tasks requiring world knowledge not present in the input. But they succeed at capturing contextual relationships through attention.

Each technique has a specific failure mode that motivates the next. LLMs are the current endpoint of this chain -- not because they solve everything, but because they address the limitations of each predecessor. And they introduce their own limitations: computational cost, hallucination, opacity, and the inability to truly reason about novel situations.

### Mental models

- **The NLP stack is like a building.** Each floor depends on the floors below it. LLMs sit on the top floor, but they contain versions of every technique from every floor beneath them. Remove tokenization, and the building collapses. Remove attention, and it collapses. Every layer matters.
- **Each technique is a representation choice.** BoW represents text as word counts. TF-IDF represents text as weighted word importance. Embeddings represent words as dense vectors. Transformers represent tokens as context-dependent vectors. LLMs represent text as probability distributions over next tokens. The journey is one of increasingly rich representations.
- **LLMs are not magic -- they are scale.** Take a transformer, train it on more data with more parameters, and you get an LLM. The architecture is the same. The training objective (predict the next token) is the same. The difference is scale -- and scale produces emergent capabilities that smaller models lack.
- **Classical methods are still the right choice for many problems.** TF-IDF search is fast, interpretable, and works perfectly for keyword-based retrieval. Using an LLM for that task is like using a crane to pick up a pencil.

### Visual explanations

```
THE NLP STACK: FROM RAW TEXT TO LLMs
======================================

  Layer 10:  LLMs (GPT, etc.)
             Scale + pretraining + fine-tuning
             Can: generate, classify, translate, summarize, converse
             Cannot: guarantee factual accuracy, truly reason, explain itself
             ^
  Layer 9:   Transformer Architecture
             Self-attention replaces recurrence
             Can: capture long-range dependencies in parallel
             Cannot: work without tokenization and embeddings below
             ^
  Layer 8:   Context & Sequence Modeling
             Word meaning depends on surrounding words
             Can: distinguish "bank" (river) from "bank" (financial)
             Cannot: scale efficiently with RNNs (vanishing gradients)
             ^
  Layer 7:   Neural Embeddings
             Dense vector representations learned from data
             Can: capture word similarity (king - man + woman = queen)
             Cannot: handle polysemy (one vector per word)
             ^
  Layer 6:   Named Entity Recognition
             Extract structured information from text
             Can: find persons, locations, organizations
             Cannot: understand relationships between entities
             ^
  Layer 5:   Tokenization (Deep Dive)
             Break text into model-friendly units (BPE, subwords)
             Can: handle any word, including unseen ones
             Cannot: capture meaning (purely mechanical splitting)
             ^
  Layer 4:   Similarity & Classical Tasks
             Cosine similarity, search, clustering
             Can: find similar documents, group by topic
             Cannot: understand synonyms or paraphrases
             ^
  Layer 3:   TF-IDF
             Weight words by importance (rare = important)
             Can: rank relevant documents, suppress common words
             Cannot: capture word order or word meaning
             ^
  Layer 2:   Bag of Words
             First numerical representation of text
             Can: convert text to vectors, enable computation
             Cannot: distinguish "dog bites man" from "man bites dog"
             ^
  Layer 1:   Text Preprocessing
             Lowercasing, stemming, stopwords, normalization
             Can: reduce noise, normalize surface variation
             Cannot: resolve ambiguity or capture meaning
             ^
  Layer 0:   Language & Text (Foundations)
             Understanding raw material: ambiguity, noise, structure
             Can: frame the problem correctly
             Cannot: compute anything (this is where understanding starts)


THE CAPABILITY PYRAMID
=======================

  What each technique CAN do:

                    +---------------+
                    |     LLMs      |  Generate, classify, translate,
                    |               |  converse, summarize, few-shot learn
                  +-+---------------+-+
                  |   Transformers     |  Long-range context, parallel
                  |                    |  training, attention patterns
                +-+--------------------+-+
                |   Neural Embeddings     |  Word similarity, analogy,
                |                         |  dense representations
              +-+-------------------------+-+
              |   TF-IDF / Classical NLP     |  Document similarity,
              |                              |  search, clustering
            +-+------------------------------+-+
            |   BoW + Preprocessing              |  Text-to-numbers,
            |                                    |  basic statistics
          +-+------------------------------------+-+
          |   Raw Text + Language Understanding      |  Problem framing,
          |                                          |  linguistic intuition
          +------------------------------------------+

  Each layer INCLUDES everything below it.
  LLMs use tokenization, embeddings, attention -- ALL of it.


DECISION TREE: WHEN TO USE WHAT
=================================

  Start here: What is your task?
  |
  +-- Keyword search / document retrieval?
  |   +-- Use TF-IDF.  Fast, interpretable, works well.
  |
  +-- Document classification with labeled data?
  |   +-- Thousands of labeled examples?
  |   |   +-- TF-IDF + simple classifier.  Often sufficient.
  |   +-- Few labeled examples (< 100)?
  |       +-- Pretrained model + fine-tuning.  Leverages transfer.
  |
  +-- Named entity extraction?
  |   +-- Known entity patterns?
  |   |   +-- Rule-based NER.  Fast, predictable.
  |   +-- Diverse entity types?
  |       +-- Pretrained encoder model + fine-tuning.
  |
  +-- Text generation / completion?
  |   +-- Decoder-only model (LLM).  This is what they are built for.
  |
  +-- Semantic similarity (paraphrase detection)?
  |   +-- Pretrained encoder model.  Captures meaning beyond keywords.
  |
  +-- Do you need explainability?
      +-- Yes --> Classical methods (TF-IDF, rules).  Transparent.
      +-- No  --> Neural / LLM.  Better performance, less transparency.
```

---

## Hands-on Exploration

1. The code below implements a simplified version of each technique from Phase 0 through Phase 10 and runs them all on the same corpus. Before running it, predict: which technique will produce the highest-quality document similarity rankings? Which will be fastest? After running, check your predictions. Notice that "best" depends on what you are optimizing for.

2. Look at the capability comparison table in the output. For each technique, identify one real-world task where it would be the best choice and one where it would be the worst choice. The point is not that newer techniques are always better -- it is that different techniques have different strengths.

3. The code includes a "decision tree" exercise at the end. Given three new tasks, it walks through the decision logic. Try adding your own task to the list. Trace through the decision tree in the ASCII diagram above and verify that the code's recommendation matches your manual trace.

---

## Live Code

```rust
// --- Where LLMs Fit in the NLP Stack ---
// Pure Rust stdlib -- no external dependencies
// Demonstrates every technique from Phase 0 to Phase 10

use std::collections::HashMap;

fn main() {
    let mut rng = Rng::new(42);

    // ============================================================
    // SHARED CORPUS AND UTILITIES
    // ============================================================

    let corpus = vec![
        "the cat sat on the warm mat near the window",
        "a dog played fetch in the sunny park today",
        "the cat chased a quick mouse across the yard",
        "machine learning models process large datasets efficiently",
        "neural networks learn representations from training data",
        "deep learning requires significant computational resources",
        "the stock market experienced volatile trading sessions today",
        "investors analyzed quarterly earnings reports carefully",
        "economic indicators suggest moderate growth this quarter",
        "the chef prepared a delicious meal with fresh ingredients",
        "baking sourdough bread requires patience and warm water",
        "the restaurant menu featured seasonal dishes and local produce",
    ];

    let doc_labels = vec![
        "animals", "animals", "animals",
        "technology", "technology", "technology",
        "finance", "finance", "finance",
        "cooking", "cooking", "cooking",
    ];

    println!("{}", "=".repeat(65));
    println!("THE NLP STACK: EVERY TECHNIQUE ON THE SAME CORPUS");
    println!("{}", "=".repeat(65));
    println!("\nCorpus: {} documents across 4 topics", corpus.len());
    for (i, (doc, label)) in corpus.iter().zip(doc_labels.iter()).enumerate() {
        println!("  D{:02} [{:>10}]: {}", i, label, doc);
    }

    // --- Shared utilities ---

    let stopwords: std::collections::HashSet<&str> = [
        "a", "an", "the", "is", "it", "in", "on", "of", "to", "and",
        "for", "with", "that", "this", "are", "was", "be", "at", "by",
        "or", "from", "but", "not", "we", "they", "their", "its",
    ].iter().copied().collect();

    fn tokenize(text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut current = String::new();
        for ch in text.to_lowercase().chars() {
            if ch.is_alphabetic() {
                current.push(ch);
            } else {
                if !current.is_empty() {
                    tokens.push(current.clone());
                    current.clear();
                }
            }
        }
        if !current.is_empty() {
            tokens.push(current);
        }
        tokens
    }

    let preprocess = |text: &str| -> Vec<String> {
        tokenize(text).into_iter()
            .filter(|t| !stopwords.contains(t.as_str()))
            .collect()
    };

    fn cosine_sim(v1: &[f64], v2: &[f64]) -> f64 {
        let dot: f64 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
        let m1: f64 = v1.iter().map(|a| a * a).sum::<f64>().sqrt();
        let m2: f64 = v2.iter().map(|b| b * b).sum::<f64>().sqrt();
        if m1 == 0.0 || m2 == 0.0 { 0.0 } else { dot / (m1 * m2) }
    }

    // ============================================================
    // PHASE 0-1: RAW TEXT + PREPROCESSING
    // ============================================================

    println!();
    println!("{}", "=".repeat(65));
    println!("PHASE 0-1: RAW TEXT + PREPROCESSING");
    println!("{}", "=".repeat(65));

    let doc_tokens: Vec<Vec<String>> = corpus.iter().map(|doc| preprocess(doc)).collect();

    println!("\nRaw:         \"{}\"", corpus[0]);
    println!("Preprocessed: {:?}", doc_tokens[0]);
    println!("\nPreprocessing removes noise (stopwords, case) but");
    println!("preserves content words. This is the foundation.");

    // ============================================================
    // PHASE 2: BAG OF WORDS
    // ============================================================

    println!();
    println!("{}", "=".repeat(65));
    println!("PHASE 2: BAG OF WORDS");
    println!("{}", "=".repeat(65));

    let mut all_words_set = std::collections::BTreeSet::new();
    for tokens in &doc_tokens {
        for t in tokens {
            all_words_set.insert(t.clone());
        }
    }
    let all_words: Vec<String> = all_words_set.into_iter().collect();
    let word_to_idx: HashMap<String, usize> = all_words.iter()
        .enumerate()
        .map(|(i, w)| (w.clone(), i))
        .collect();

    let bow_vector = |tokens: &[String]| -> Vec<f64> {
        let mut vec = vec![0.0; all_words.len()];
        for t in tokens {
            if let Some(&idx) = word_to_idx.get(t) {
                vec[idx] += 1.0;
            }
        }
        vec
    };

    let bow_vectors: Vec<Vec<f64>> = doc_tokens.iter().map(|t| bow_vector(t)).collect();

    println!("\nVocabulary: {} words", all_words.len());
    println!("Vector dimension: {}", all_words.len());
    println!("Sparsity: most entries are 0");
    println!("\nBoW similarity (D00 cat vs D01 dog):  {:.4}",
        cosine_sim(&bow_vectors[0], &bow_vectors[1]));
    println!("BoW similarity (D00 cat vs D03 tech): {:.4}",
        cosine_sim(&bow_vectors[0], &bow_vectors[3]));
    println!("\nBoW captures word presence but ignores order and importance.");

    // ============================================================
    // PHASE 3: TF-IDF
    // ============================================================

    println!();
    println!("{}", "=".repeat(65));
    println!("PHASE 3: TF-IDF");
    println!("{}", "=".repeat(65));

    let n_docs = doc_tokens.len();
    let n_words = all_words.len();

    // Compute IDF
    let mut df = vec![0usize; n_words];
    for tokens in &doc_tokens {
        let seen: std::collections::HashSet<&String> = tokens.iter().collect();
        for w in seen {
            if let Some(&idx) = word_to_idx.get(w) {
                df[idx] += 1;
            }
        }
    }

    let idf: Vec<f64> = (0..n_words).map(|i| {
        if df[i] > 0 { (n_docs as f64 / df[i] as f64).ln() } else { 0.0 }
    }).collect();

    let tfidf_vector = |tokens: &[String]| -> Vec<f64> {
        let mut tf: HashMap<String, usize> = HashMap::new();
        for t in tokens { *tf.entry(t.clone()).or_insert(0) += 1; }
        let total = if tokens.is_empty() { 1 } else { tokens.len() };
        let mut vec = vec![0.0; n_words];
        for (w, count) in &tf {
            if let Some(&idx) = word_to_idx.get(w) {
                vec[idx] = (*count as f64 / total as f64) * idf[idx];
            }
        }
        vec
    };

    let tfidf_vectors: Vec<Vec<f64>> = doc_tokens.iter().map(|t| tfidf_vector(t)).collect();

    println!("\nTF-IDF similarity (D00 cat vs D02 cat):  {:.4}",
        cosine_sim(&tfidf_vectors[0], &tfidf_vectors[2]));
    println!("TF-IDF similarity (D00 cat vs D03 tech): {:.4}",
        cosine_sim(&tfidf_vectors[0], &tfidf_vectors[3]));
    println!("\nTF-IDF downweights common words, making topic-specific");
    println!("words drive similarity. An improvement over raw BoW.");

    // ============================================================
    // PHASE 4: SIMILARITY & SEARCH
    // ============================================================

    println!();
    println!("{}", "=".repeat(65));
    println!("PHASE 4: SIMILARITY-BASED SEARCH");
    println!("{}", "=".repeat(65));

    let query = "cat chased mouse";
    let query_tokens = preprocess(query);
    let query_vec = tfidf_vector(&query_tokens);

    println!("\nQuery: \"{}\"", query);
    println!("Ranking documents by TF-IDF cosine similarity:\n");

    let mut rankings: Vec<(usize, f64)> = (0..n_docs)
        .map(|i| (i, cosine_sim(&query_vec, &tfidf_vectors[i])))
        .collect();
    rankings.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for (rank, (idx, sim)) in rankings.iter().take(5).enumerate() {
        let truncated: String = corpus[*idx].chars().take(50).collect();
        println!("  #{}: D{:02} [{:>10}] sim={:.4}  {}...",
            rank + 1, idx, doc_labels[*idx], sim, truncated);
    }

    println!("\nSearch works by measuring distance in vector space.");
    println!("No \"understanding\" -- just geometric proximity.");

    // ============================================================
    // PHASE 5-6: TOKENIZATION & NER (Conceptual)
    // ============================================================

    println!();
    println!("{}", "=".repeat(65));
    println!("PHASE 5-6: TOKENIZATION & NER (CONCEPTUAL)");
    println!("{}", "=".repeat(65));

    // Demonstrate subword tokenization concept
    println!("\nSubword tokenization (BPE-style concept):");
    let example = "unbelievably";
    println!("  Word: \"{}\"", example);
    println!("  Word-level:    [\"{}\"]", example);
    let chars: Vec<String> = example.chars().map(|c| format!("'{}'", c)).collect();
    println!("  Character:     [{}]", chars.join(", "));
    println!("  Subword (sim): [\"un\", \"believ\", \"ably\"]");
    println!("\n  Subword tokenization balances vocabulary size with coverage.");
    println!("  LLMs use this -- it is a foundation layer, not optional.");

    // Demonstrate NER concept
    println!("\nNamed Entity Recognition (rule-based concept):");
    let ner_text = "the chef prepared a delicious meal";
    let ner_tokens = tokenize(ner_text);
    let mut entity_rules: HashMap<&str, &str> = HashMap::new();
    entity_rules.insert("chef", "ROLE");
    entity_rules.insert("meal", "FOOD");
    println!("  Text: \"{}\"", ner_text);
    for token in &ner_tokens {
        if let Some(entity) = entity_rules.get(token.as_str()) {
            println!("  Found: \"{}\" -> {}", token, entity);
        }
    }
    println!("  Rule-based NER is fast but brittle. Neural NER is flexible but opaque.");

    // ============================================================
    // PHASE 7-8: NEURAL EMBEDDINGS & CONTEXT
    // ============================================================

    println!();
    println!("{}", "=".repeat(65));
    println!("PHASE 7-8: NEURAL EMBEDDINGS & CONTEXT");
    println!("{}", "=".repeat(65));

    // Simulate word embeddings via co-occurrence (simplified)
    fn build_embeddings(
        sentences: &[&str],
        stopwords: &std::collections::HashSet<&str>,
        dim: usize,
        window: usize,
        rng: &mut Rng,
    ) -> HashMap<String, Vec<f64>> {
        let all_tok: Vec<Vec<String>> = sentences.iter()
            .map(|s| {
                let mut tokens = Vec::new();
                let mut current = String::new();
                for ch in s.to_lowercase().chars() {
                    if ch.is_alphabetic() {
                        current.push(ch);
                    } else {
                        if !current.is_empty() {
                            if !stopwords.contains(current.as_str()) {
                                tokens.push(current.clone());
                            }
                            current.clear();
                        }
                    }
                }
                if !current.is_empty() && !stopwords.contains(current.as_str()) {
                    tokens.push(current);
                }
                tokens
            })
            .collect();

        let mut vocab_set = std::collections::BTreeSet::new();
        for tokens in &all_tok {
            for t in tokens { vocab_set.insert(t.clone()); }
        }
        let vocab: Vec<String> = vocab_set.into_iter().collect();
        let w2i: HashMap<String, usize> = vocab.iter()
            .enumerate().map(|(i, w)| (w.clone(), i)).collect();
        let n = vocab.len();

        // Co-occurrence matrix
        let mut cooccur = vec![vec![0.0; n]; n];
        for tokens in &all_tok {
            for (i, word) in tokens.iter().enumerate() {
                let wi = w2i[word];
                let start = if i >= window { i - window } else { 0 };
                let end = (i + window + 1).min(tokens.len());
                for j in start..end {
                    if i != j {
                        let wj = w2i[&tokens[j]];
                        cooccur[wi][wj] += 1.0;
                    }
                }
            }
        }

        // Simple dimensionality reduction: random projection
        let projection: Vec<Vec<f64>> = (0..n)
            .map(|_| (0..dim).map(|_| rng.gauss(0.0, 1.0)).collect())
            .collect();

        let mut embeddings = HashMap::new();
        for word in &vocab {
            let wi = w2i[word];
            let row = &cooccur[wi];
            let mut emb = vec![0.0; dim];
            for j in 0..n {
                if row[j] > 0.0 {
                    for d in 0..dim {
                        emb[d] += row[j] * projection[j][d];
                    }
                }
            }
            // Normalize
            let mag: f64 = emb.iter().map(|x| x * x).sum::<f64>().sqrt();
            if mag > 0.0 {
                for d in 0..dim { emb[d] /= mag; }
            }
            embeddings.insert(word.clone(), emb);
        }

        embeddings
    }

    rng = Rng::new(42);
    let embeddings = build_embeddings(&corpus, &stopwords, 20, 2, &mut rng);

    println!("\nEmbedding dimension: 20 (dense, vs {} for BoW)", all_words.len());
    println!("Words with embeddings: {}", embeddings.len());

    // Show word similarities with embeddings
    println!("\nWord similarities (embedding-based):");
    let word_pairs: Vec<(&str, &str)> = vec![
        ("cat", "dog"), ("cat", "market"), ("learning", "networks"),
        ("chef", "bread"), ("stock", "investors"),
    ];
    for (w1, w2) in &word_pairs {
        if let (Some(v1), Some(v2)) = (embeddings.get(*w1), embeddings.get(*w2)) {
            let sim = cosine_sim(v1, v2);
            println!("  \"{}\" vs \"{}\": {:+.4}", w1, w2, sim);
        }
    }

    println!("\nDense embeddings capture relationships that TF-IDF cannot:");
    println!("related words have similar vectors even if they never co-occur");
    println!("in the same document.");

    // ============================================================
    // PHASE 9-10: ATTENTION & LLM CONCEPTS
    // ============================================================

    println!();
    println!("{}", "=".repeat(65));
    println!("PHASE 9-10: ATTENTION & LLM CONCEPTS");
    println!("{}", "=".repeat(65));

    // Simulate simple attention: context-dependent word representation
    fn compute_attention_weights(
        tokens: &[String], embeddings: &HashMap<String, Vec<f64>>,
    ) -> Vec<Vec<f64>> {
        let n = tokens.len();
        let dim = 20;
        let vecs: Vec<Vec<f64>> = tokens.iter().map(|t| {
            embeddings.get(t).cloned().unwrap_or_else(|| vec![0.0; dim])
        }).collect();

        // Compute attention scores (dot product)
        let mut scores = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                scores[i][j] = vecs[i].iter().zip(vecs[j].iter())
                    .map(|(a, b)| a * b).sum();
            }

            // Softmax
            let max_s = scores[i].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_scores: Vec<f64> = scores[i].iter().map(|s| (s - max_s).exp()).collect();
            let total: f64 = exp_scores.iter().sum();
            if total > 0.0 {
                scores[i] = exp_scores.iter().map(|e| e / total).collect();
            }
        }

        scores
    }

    let example_text = "cat sat on warm mat";
    let ex_tokens = preprocess(example_text);
    if ex_tokens.len() > 1 {
        let attn = compute_attention_weights(&ex_tokens, &embeddings);

        println!("\nSimulated self-attention for: \"{}\"", example_text);
        println!("\nAttention weights (which words attend to which):\n");

        // Header
        print!("  {:>10}", "");
        for t in &ex_tokens { print!("{:>8}", t); }
        println!();

        for (i, t) in ex_tokens.iter().enumerate() {
            print!("  {:>10}", t);
            for j in 0..ex_tokens.len() {
                print!("  {:.3}", attn[i][j]);
            }
            println!();
        }

        println!("\n  Each row shows how much one word \"attends to\" each other word.");
        println!("  Higher weight = more influence on the word's representation.");
        println!("  This is the core mechanism of transformers and LLMs.");
    }

    // ============================================================
    // THE FULL COMPARISON TABLE
    // ============================================================

    println!();
    println!("{}", "=".repeat(65));
    println!("CAPABILITY COMPARISON: CLASSICAL TO MODERN");
    println!("{}", "=".repeat(65));

    let comparison: Vec<(&str, &str, &str, &str, &str)> = vec![
        ("BoW", "Word counts", "Document comparison",
         "No word order, no meaning", "microseconds"),
        ("TF-IDF", "Weighted word counts", "Search, classification",
         "No word order, no synonyms", "microseconds"),
        ("Static Embeddings", "Dense word vectors", "Word similarity, analogy",
         "No polysemy (one vec/word)", "milliseconds"),
        ("RNN/LSTM", "Sequential hidden states", "Sequence labeling",
         "Slow training, short memory", "seconds"),
        ("Transformer", "Context-dependent vectors", "All NLP tasks",
         "Quadratic attention cost", "seconds-minutes"),
        ("LLM", "Pretrained transformer", "Generation, few-shot tasks",
         "Expensive, hallucinations", "seconds-minutes"),
    ];

    println!();
    println!("  {:<20} {:<25} {:>15}", "Technique", "Representation", "Speed");
    println!("  {} {} {}", "-".repeat(20), "-".repeat(25), "-".repeat(15));
    for (name, rep, _good, _bad, speed) in &comparison {
        println!("  {:<20} {:<25} {:>15}", name, rep, speed);
    }

    println!();
    println!("  {:<20} {:<28} {}", "Technique", "Good at", "Limitation");
    println!("  {} {} {}", "-".repeat(20), "-".repeat(28), "-".repeat(30));
    for (name, _rep, good, bad, _speed) in &comparison {
        println!("  {:<20} {:<28} {}", name, good, bad);
    }

    println!();
    println!("  Key insight: each technique is a different REPRESENTATION CHOICE.");
    println!("  Newer is not always better -- it depends on the task and constraints.");

    // --- Visualization: Capability Comparison ---
    let capability_names: Vec<&str> = comparison.iter().map(|(name, _, _, _, _)| *name).collect();
    let cap_word_meaning: Vec<f64> = vec![1.0, 1.0, 3.0, 4.0, 5.0, 5.0];
    let cap_word_order: Vec<f64> =   vec![1.0, 1.0, 1.0, 4.0, 5.0, 5.0];
    let cap_speed: Vec<f64> =        vec![5.0, 5.0, 4.0, 2.0, 2.0, 2.0];
    let cap_interpretability: Vec<f64> = vec![5.0, 5.0, 3.0, 2.0, 1.0, 1.0];

    show_chart(&multi_bar_chart(
        "NLP Technique Capabilities (1=minimal, 5=excellent)",
        &capability_names,
        &[
            ("Word Meaning", &cap_word_meaning, "#3b82f6"),
            ("Word Order", &cap_word_order, "#10b981"),
            ("Speed", &cap_speed, "#f59e0b"),
            ("Interpretability", &cap_interpretability, "#8b5cf6"),
        ],
    ));

    // ============================================================
    // PRACTICAL DECISION EXERCISE
    // ============================================================

    println!();
    println!("{}", "=".repeat(65));
    println!("DECISION EXERCISE: CHOOSE THE RIGHT TOOL");
    println!("{}", "=".repeat(65));

    let decisions: Vec<(&str, &[&str], &str, &str)> = vec![
        (
            "Search 10,000 product descriptions by keyword",
            &["fast response", "keyword matching", "interpretable"],
            "TF-IDF",
            "Fast, interpretable, excellent for keyword-based retrieval. \
             No need for neural models when keywords are sufficient.",
        ),
        (
            "Classify customer support tickets into 5 categories",
            &["1,000 labeled examples", "high accuracy", "fast inference"],
            "TF-IDF + classifier (or pretrained encoder if accuracy critical)",
            "With 1,000 labeled examples, TF-IDF + logistic regression often \
             matches neural models. Try simple first.",
        ),
        (
            "Generate product descriptions from bullet points",
            &["fluent text", "creative", "variable length"],
            "Decoder-only LLM",
            "Text generation requires autoregressive models. Classical methods \
             cannot generate fluent text. This is where LLMs excel.",
        ),
        (
            "Detect person and organization names in legal documents",
            &["high precision", "domain-specific", "auditable"],
            "Pretrained encoder + fine-tuning (with rule-based fallback)",
            "NER benefits from bidirectional context (encoder). Fine-tuning on \
             legal text adapts to domain. Rules add auditability.",
        ),
        (
            "Find duplicate support tickets",
            &["semantic matching", "beyond keyword overlap"],
            "Pretrained encoder embeddings + cosine similarity",
            "Paraphrases share meaning but not keywords. Encoder embeddings \
             capture semantic similarity that TF-IDF misses.",
        ),
    ];

    for (i, (task, requirements, recommendation, reason)) in decisions.iter().enumerate() {
        println!("\n  Task {}: {}", i + 1, task);
        println!("  Requirements: {}", requirements.join(", "));
        println!("  --> Recommendation: {}", recommendation);
        println!("      Why: {}", reason);
    }

    // ============================================================
    // THE NLP JOURNEY: A REPRESENTATION PERSPECTIVE
    // ============================================================

    println!();
    println!("{}", "=".repeat(65));
    println!("THE NLP JOURNEY: A REPRESENTATION PERSPECTIVE");
    println!("{}", "=".repeat(65));

    let journey: Vec<(&str, &str, &str, &str)> = vec![
        ("Phase 0",  "Raw Text", "Symbolic characters",
         "We cannot compute on raw text."),
        ("Phase 1",  "Preprocessing", "Cleaned tokens",
         "Tokens are discrete. We need numbers."),
        ("Phase 2",  "Bag of Words", "Sparse count vectors",
         "Counts ignore word importance."),
        ("Phase 3",  "TF-IDF", "Sparse weighted vectors",
         "Weights ignore word meaning and order."),
        ("Phase 4",  "Similarity Tasks", "Cosine distance in TF-IDF space",
         "Still no word meaning or synonyms."),
        ("Phase 5",  "Tokenization", "Subword units",
         "Better input units, but still no meaning."),
        ("Phase 6",  "NER", "Token labels (B-PER, I-ORG)",
         "Labels depend on context we do not model well."),
        ("Phase 7",  "Embeddings", "Dense word vectors",
         "One vector per word -- no polysemy."),
        ("Phase 8",  "Context Models", "Context-dependent vectors",
         "RNNs struggle with long sequences."),
        ("Phase 9",  "Transformers", "Attention-based representations",
         "Need massive data to learn well."),
        ("Phase 10", "LLMs", "Pretrained attention at scale",
         "Expensive, opaque, can hallucinate."),
    ];

    println!("\n  {:<10} {:<18} {:<30} {}",
        "Phase", "Technique", "Representation", "Limitation");
    println!("  {} {} {} {}",
        "-".repeat(10), "-".repeat(18), "-".repeat(30), "-".repeat(40));
    for (phase, technique, representation, limitation) in &journey {
        println!("  {:<10} {:<18} {:<30} {}", phase, technique, representation, limitation);
    }

    println!();
    println!("  Each phase solves the limitation of the previous one.");
    println!("  Each phase introduces a new limitation of its own.");
    println!("  LLMs are not the end of this journey -- they are the current point.");
    println!();
    println!("  Understanding these foundations makes you a better user of");
    println!("  modern tools. You know what they can do, what they cannot do,");
    println!("  and why they work the way they do.");
    println!();
    println!("  Modern NLP stands on decades of classical foundations.");
    println!("  LLMs did not replace this stack -- they absorbed it.");
}
```

---

## Key Takeaways

- **LLMs are the top of a stack, not a replacement for it.** Every LLM uses tokenization (Phase 5), embeddings (Phase 7), attention (Phase 9), and pretraining (Phase 10). Remove any layer and the system fails. Understanding the stack means understanding LLMs.
- **Each technique is a representation choice with specific tradeoffs.** BoW trades word order for simplicity. TF-IDF trades raw counts for relevance weighting. Embeddings trade interpretability for semantic density. LLMs trade efficiency for expressiveness. There is no universally "best" representation -- only the right one for your task and constraints.
- **Classical methods remain the right choice for many problems.** TF-IDF search is fast, interpretable, and effective for keyword retrieval. Using an LLM where TF-IDF suffices wastes computation and adds opacity. The decision tree is: start simple, add complexity only when simpler methods fail at your specific task.
- **LLMs do not "understand" language -- they compute increasingly rich statistical representations of it.** The journey from BoW to LLMs is one of richer representations, not of approaching comprehension. Each step captures more statistical structure. No step crosses from statistics into understanding. Knowing this protects you from both hype and disappointment.
