# Visualize Attention Weights

> Phase 9 — Transformer Architecture (Core Concepts) | Kata 9.1

---

## Concept & Intuition

### What problem are we solving?

In all previous phases, every word in a sentence received a fixed representation that did not depend on its neighbors. BoW and TF-IDF assigned each word a score based on frequency or rarity, but the word "bank" got the same score whether it appeared next to "river" or "financial." RNNs attempted to fix this by processing words sequentially, but they struggled with long-range dependencies because information had to pass through every intermediate step.

Self-attention solves this directly: **every token computes a weighted relationship with every other token in the sequence, in a single step.** The word "bank" can directly attend to "river" even if they are far apart, and the resulting representation of "bank" will be different from when it attends to "financial." This is the core mechanism that makes transformers work.

### Why earlier methods fail

**BoW and TF-IDF** treat each word independently. The representation of "bank" in "river bank" is identical to "bank" in "bank account." There is no mechanism for context to influence representation.

**RNNs** process tokens left-to-right (or both directions in BiRNNs). To connect the first word to the last in a 100-word sentence, information must pass through 99 sequential steps. At each step, some information is lost or distorted. This is the vanishing gradient problem in practice: distant words have a weak influence on each other.

**Self-attention** skips the sequential bottleneck entirely. Every token directly computes a relevance score with every other token. The word at position 1 can attend to the word at position 100 with a single matrix multiplication. No information passes through intermediate steps. No gradients vanish across long distances.

### Mental models

- **Attention as a soft lookup table**: Each token creates a "query" (what am I looking for?), and every token offers a "key" (what do I contain?). The dot product between query and key tells you how relevant each token is. High relevance means "pay more attention to this token."
- **Q, K, V as roles**: Every token plays three simultaneous roles. As a **Query**, it asks "who should I attend to?" As a **Key**, it advertises "here is what I contain." As a **Value**, it provides "here is the information I carry." Attention weights determine how much of each token's Value gets mixed into the final output.
- **Softmax as competition**: The attention weights for each query sum to 1.0 (via softmax). This means tokens compete for attention. If one key matches a query very well, it "wins" most of the attention, and other tokens get less. This creates a dynamic, context-dependent focus mechanism.
- **Scaling by sqrt(d_k)**: Without scaling, dot products between high-dimensional vectors grow large, pushing softmax into saturation (all weight on one token, none on others). Dividing by sqrt(d_k) keeps the values in a range where softmax produces meaningful distributions.

### Visual explanations

```
SELF-ATTENTION: Every token attends to every other token
=====================================================

  Input:  ["The", "cat", "sat", "on", "the", "bank"]

  Each token produces three vectors:
    Q (Query)  -- "What am I looking for?"
    K (Key)    -- "What do I contain?"
    V (Value)  -- "What information do I carry?"

  For each token, compute attention to ALL tokens:

        The   cat   sat   on   the   bank    <-- Keys
  The  [0.15  0.10  0.10  0.10  0.15  0.40]  <-- attention weights for "The"
  cat  [0.10  0.20  0.15  0.05  0.10  0.40]  <-- attention weights for "cat"
  sat  [0.10  0.30  0.20  0.10  0.10  0.20]  <-- attention weights for "sat"
  on   [0.10  0.10  0.15  0.15  0.10  0.40]  <-- attention weights for "on"
  the  [0.10  0.10  0.10  0.10  0.20  0.40]  <-- attention weights for "the"
  bank [0.15  0.15  0.10  0.10  0.15  0.35]  <-- attention weights for "bank"
   ^
   Queries

  Each row sums to 1.0 (softmax).
  Each row says: "For this query token, how much should I attend to each key?"


THE COMPUTATION
===============

  Step 1: Compute raw scores    Score = Q * K^T
  Step 2: Scale                 Score = Score / sqrt(d_k)
  Step 3: Apply softmax         Weights = softmax(Score)   [each row sums to 1]
  Step 4: Weighted sum of V     Output = Weights * V

  Q (seq_len x d_k)             K^T (d_k x seq_len)
  +-----------+                  +-----------+
  | q1 q1 q1 |                  | k1 k2 k3 |
  | q2 q2 q2 |   multiply -->   | k1 k2 k3 |
  | q3 q3 q3 |                  | k1 k2 k3 |
  +-----------+                  +-----------+
                    ||
                    vv
            Score (seq_len x seq_len)
            +-------------+
            | s11 s12 s13 |   <-- how much token 1 attends to 1, 2, 3
            | s21 s22 s23 |   <-- how much token 2 attends to 1, 2, 3
            | s31 s32 s33 |   <-- how much token 3 attends to 1, 2, 3
            +-------------+

  After softmax, each row is a probability distribution.
  Multiply by V to get context-aware representations.


WHY sqrt(d_k) SCALING?
======================

  Without scaling (d_k = 64):
    Dot products can reach values like 30-50.
    softmax([30, 50, 35]) --> [0.00, 1.00, 0.00]   (saturated!)

  With scaling (divide by sqrt(64) = 8):
    Values become 3.75, 6.25, 4.38
    softmax([3.75, 6.25, 4.38]) --> [0.06, 0.76, 0.18]  (meaningful distribution)
```

---

## Hands-on Exploration

1. Consider the sentence "The cat sat on the river bank." By hand, decide which words you think "bank" should attend to most. Now consider "The bank approved the loan." Would the attention pattern for "bank" differ? This is exactly what self-attention learns to do: different contexts produce different attention distributions for the same word.

2. In the attention weight matrix, each row sums to 1.0. What happens if one key matches a query extremely well (very high dot product)? After softmax, that key gets nearly all the weight, and the output for that query is dominated by a single token's value. Is this always desirable? Think of cases where attending broadly to many tokens would be better than focusing sharply on one.

3. The scaling factor is sqrt(d_k). What would happen if we used d_k instead (no square root)? What about no scaling at all? Run the code with different scaling values and observe how the attention distributions change. When the distribution is too "peaked" (one weight near 1.0, others near 0.0), the model cannot blend information from multiple sources.

---

## Live Code

```rust
// --- Self-Attention: Scaled Dot-Product Attention from Scratch ---
// Using ONLY Rust standard library

fn main() {
    let mut rng = Rng::new(42);

    // ============================================================
    // VECTOR AND MATRIX HELPERS
    // ============================================================

    fn make_vector(dim: usize, rng: &mut Rng) -> Vec<f64> {
        (0..dim).map(|_| rng.gauss(0.0, 1.0)).collect()
    }

    fn make_matrix(rows: usize, cols: usize, rng: &mut Rng) -> Vec<Vec<f64>> {
        (0..rows).map(|_| make_vector(cols, rng)).collect()
    }

    fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let m = a.len();
        let n = b.len();
        let p = b[0].len();
        let mut result = vec![vec![0.0; p]; m];
        for i in 0..m {
            for j in 0..p {
                let mut val = 0.0;
                for k in 0..n {
                    val += a[i][k] * b[k][j];
                }
                result[i][j] = val;
            }
        }
        result
    }

    fn transpose(m: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let rows = m.len();
        let cols = m[0].len();
        let mut result = vec![vec![0.0; rows]; cols];
        for i in 0..rows {
            for j in 0..cols {
                result[j][i] = m[i][j];
            }
        }
        result
    }

    fn softmax(values: &[f64]) -> Vec<f64> {
        let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = values.iter().map(|v| (v - max_val).exp()).collect();
        let total: f64 = exps.iter().sum();
        exps.iter().map(|e| e / total).collect()
    }

    fn format_float(v: f64, width: usize) -> String {
        format!("{:>width$.3}", v, width = width)
    }

    // ============================================================
    // SELF-ATTENTION COMPUTATION
    // ============================================================

    // Our sentence
    let words = vec!["The", "cat", "sat", "on", "the", "bank"];
    let seq_len = words.len();
    let d_model: usize = 8;   // embedding dimension
    let d_k: usize = 4;       // dimension of Q, K, V (smaller for clarity)

    println!("{}", "=".repeat(60));
    println!("SELF-ATTENTION: SCALED DOT-PRODUCT ATTENTION");
    println!("{}", "=".repeat(60));
    println!("\nSentence: {:?}", words);
    println!("Sequence length: {}", seq_len);
    println!("Embedding dimension (d_model): {}", d_model);
    println!("Q/K/V dimension (d_k): {}", d_k);

    // Step 1: Create input embeddings (random for demonstration)
    rng = Rng::new(42);
    let embeddings = make_matrix(seq_len, d_model, &mut rng);

    println!("\n--- Input Embeddings ({} x {}) ---", seq_len, d_model);
    for (i, word) in words.iter().enumerate() {
        let vec_str: String = embeddings[i].iter()
            .map(|v| format_float(*v, 6))
            .collect::<Vec<_>>()
            .join(" ");
        println!("  {:>4}: [{}]", word, vec_str);
    }

    // Step 2: Create weight matrices W_Q, W_K, W_V (d_model x d_k)
    let w_q = make_matrix(d_model, d_k, &mut rng);
    let w_k = make_matrix(d_model, d_k, &mut rng);
    let w_v = make_matrix(d_model, d_k, &mut rng);

    // Step 3: Compute Q, K, V by projecting embeddings
    let q = mat_mul(&embeddings, &w_q);  // (seq_len x d_k)
    let k = mat_mul(&embeddings, &w_k);  // (seq_len x d_k)
    let v = mat_mul(&embeddings, &w_v);  // (seq_len x d_k)

    println!("\n--- Query Vectors Q ({} x {}) ---", seq_len, d_k);
    for (i, word) in words.iter().enumerate() {
        let vec_str: String = q[i].iter()
            .map(|v| format_float(*v, 6))
            .collect::<Vec<_>>()
            .join(" ");
        println!("  {:>4}: [{}]", word, vec_str);
    }

    println!("\n--- Key Vectors K ({} x {}) ---", seq_len, d_k);
    for (i, word) in words.iter().enumerate() {
        let vec_str: String = k[i].iter()
            .map(|v| format_float(*v, 6))
            .collect::<Vec<_>>()
            .join(" ");
        println!("  {:>4}: [{}]", word, vec_str);
    }

    println!("\n--- Value Vectors V ({} x {}) ---", seq_len, d_k);
    for (i, word) in words.iter().enumerate() {
        let vec_str: String = v[i].iter()
            .map(|v| format_float(*v, 6))
            .collect::<Vec<_>>()
            .join(" ");
        println!("  {:>4}: [{}]", word, vec_str);
    }

    // Step 4: Compute attention scores: Q * K^T
    let k_t = transpose(&k);
    let raw_scores = mat_mul(&q, &k_t);  // (seq_len x seq_len)

    println!("\n--- Raw Attention Scores: Q * K^T ({} x {}) ---", seq_len, seq_len);
    let header: String = "        ".to_string()
        + &words.iter().map(|w| format!("{:>8}", w)).collect::<String>();
    println!("{}", header);
    for (i, word) in words.iter().enumerate() {
        let row_str: String = (0..seq_len)
            .map(|j| format_float(raw_scores[i][j], 8))
            .collect::<String>();
        println!("  {:>4}: {}", word, row_str);
    }

    // Step 5: Scale by sqrt(d_k)
    let scale = (d_k as f64).sqrt();
    let scaled_scores: Vec<Vec<f64>> = raw_scores.iter()
        .map(|row| row.iter().map(|v| v / scale).collect())
        .collect();

    println!("\n--- Scaled Scores: Q*K^T / sqrt({}) = Q*K^T / {:.2} ---", d_k, scale);
    let header: String = "        ".to_string()
        + &words.iter().map(|w| format!("{:>8}", w)).collect::<String>();
    println!("{}", header);
    for (i, word) in words.iter().enumerate() {
        let row_str: String = (0..seq_len)
            .map(|j| format_float(scaled_scores[i][j], 8))
            .collect::<String>();
        println!("  {:>4}: {}", word, row_str);
    }

    // Step 6: Apply softmax to each row
    let attention_weights: Vec<Vec<f64>> = scaled_scores.iter()
        .map(|row| softmax(row))
        .collect();

    println!("\n--- Attention Weights: softmax(scaled_scores) ---");
    let header: String = "        ".to_string()
        + &words.iter().map(|w| format!("{:>8}", w)).collect::<String>();
    println!("{}", header);
    for (i, word) in words.iter().enumerate() {
        let row_str: String = (0..seq_len)
            .map(|j| format_float(attention_weights[i][j], 8))
            .collect::<String>();
        let row_sum: f64 = attention_weights[i].iter().sum();
        println!("  {:>4}: {}  (sum={:.3})", word, row_str, row_sum);
    }

    // Step 7: Compute output: attention_weights * V
    let output = mat_mul(&attention_weights, &v);  // (seq_len x d_k)

    println!("\n--- Output: Attention_Weights * V ({} x {}) ---", seq_len, d_k);
    for (i, word) in words.iter().enumerate() {
        let vec_str: String = output[i].iter()
            .map(|v| format_float(*v, 6))
            .collect::<Vec<_>>()
            .join(" ");
        println!("  {:>4}: [{}]", word, vec_str);
    }

    // ============================================================
    // ASCII HEATMAP OF ATTENTION WEIGHTS
    // ============================================================

    println!("\n{}", "=".repeat(60));
    println!("ATTENTION HEATMAP (ASCII)");
    println!("{}", "=".repeat(60));
    println!("\nBrightness: . < o < O < # (low to high attention)\n");

    fn attention_char(weight: f64) -> &'static str {
        if weight < 0.10 {
            "  .  "
        } else if weight < 0.18 {
            "  o  "
        } else if weight < 0.25 {
            "  O  "
        } else {
            "  #  "
        }
    }

    let header: String = "  Query\\Key ".to_string()
        + &words.iter().map(|w| format!("{:>5}", w)).collect::<String>();
    println!("{}", header);
    println!("  {}", "-".repeat(11 + 5 * seq_len));
    for (i, word) in words.iter().enumerate() {
        let row_str: String = (0..seq_len)
            .map(|j| attention_char(attention_weights[i][j]))
            .collect::<String>();
        println!("  {:>8}  |{}", word, row_str);
    }

    println!();

    // Rich heatmap visualization of attention weights
    let words_ref: Vec<&str> = words.iter().copied().collect();
    show_chart(&heatmap_chart(
        "Self-Attention Weights",
        &words_ref,
        &words_ref,
        &attention_weights,
    ));

    // ============================================================
    // SHOW TOP ATTENDED TOKENS FOR EACH WORD
    // ============================================================

    println!("{}", "=".repeat(60));
    println!("TOP ATTENDED TOKENS FOR EACH WORD");
    println!("{}", "=".repeat(60));

    for (i, word) in words.iter().enumerate() {
        // Sort by attention weight descending
        let mut ranked: Vec<usize> = (0..seq_len).collect();
        ranked.sort_by(|&a, &b| attention_weights[i][b]
            .partial_cmp(&attention_weights[i][a]).unwrap());

        println!("\n  '{}' attends most to:", word);
        for (rank, &j) in ranked.iter().take(3).enumerate() {
            let bar_len = (attention_weights[i][j] * 30.0) as usize;
            let bar: String = "#".repeat(bar_len);
            println!("    {}. '{}' (weight={:.3}) {}",
                rank + 1, words[j], attention_weights[i][j], bar);
        }
    }

    // ============================================================
    // DEMONSTRATE: EFFECT OF SCALING
    // ============================================================

    println!("\n{}", "=".repeat(60));
    println!("EFFECT OF SCALING ON ATTENTION DISTRIBUTION");
    println!("{}", "=".repeat(60));

    // Use one row of scores as example
    let example_row = &raw_scores[1];  // scores for "cat"
    let example_str: String = example_row.iter()
        .map(|v| format!("{:.2}", v))
        .collect::<Vec<_>>()
        .join(", ");
    println!("\nRaw scores for 'cat': [{}]", example_str);

    let scale_configs: Vec<(&str, f64)> = vec![
        ("No scaling (div by 1)", 1.0),
        ("sqrt(d_k) = 2.0", (d_k as f64).sqrt()),
        ("d_k = 4.0", d_k as f64),
    ];

    for (scale_name, divisor) in &scale_configs {
        let scaled: Vec<f64> = example_row.iter().map(|v| v / divisor).collect();
        let weights = softmax(&scaled);
        let max_w = weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_w = weights.iter().cloned().fold(f64::INFINITY, f64::min);
        let spread = max_w - min_w;
        println!("\n  {}:", scale_name);
        let weights_str: String = weights.iter()
            .map(|w| format!("{:.3}", w))
            .collect::<Vec<_>>()
            .join(", ");
        println!("    Weights: [{}]", weights_str);
        println!("    Max={:.3}, Min={:.3}, Spread={:.3}", max_w, min_w, spread);
        if spread > 0.8 {
            println!("    --> Too peaked! One token dominates.");
        } else if spread < 0.05 {
            println!("    --> Too flat! Nearly uniform, no focus.");
        } else {
            println!("    --> Good distribution. Meaningful attention pattern.");
        }
    }

    // ============================================================
    // VERIFY: OUTPUT IS CONTEXT-DEPENDENT
    // ============================================================

    println!("\n{}", "=".repeat(60));
    println!("KEY INSIGHT: OUTPUT DEPENDS ON CONTEXT");
    println!("{}", "=".repeat(60));

    let v_bank_str: String = v[5].iter()
        .map(|val| format_float(*val, 6))
        .collect::<Vec<_>>()
        .join(", ");
    let out_bank_str: String = output[5].iter()
        .map(|val| format_float(*val, 6))
        .collect::<Vec<_>>()
        .join(", ");
    println!("\nOriginal V for 'bank': [{}]", v_bank_str);
    println!("Output for 'bank':    [{}]", out_bank_str);
    println!("\nThese are DIFFERENT. The output for 'bank' is a weighted");
    println!("combination of ALL tokens' values, based on attention weights.");
    println!("'bank' now carries information from the words around it.");
    println!("\nThis is the core power of self-attention: every token's");
    println!("representation becomes context-dependent in a single step,");
    println!("with no sequential processing bottleneck.");
}
```

---

## Key Takeaways

- **Self-attention computes pairwise relevance between all tokens simultaneously.** Unlike RNNs, there is no sequential bottleneck. Every token can directly attend to every other token, regardless of distance in the sequence.
- **Q, K, V are three learned projections of the same input.** The Query asks "what am I looking for?", the Key advertises "what do I contain?", and the Value provides "what information should I contribute?" Attention weights are computed from Q and K, then applied to V.
- **Softmax creates a competition among keys.** Each query's attention weights sum to 1.0, forcing tokens to compete for influence. This means the output for each token is a weighted average of all values, with weights determined by relevance.
- **Scaling by sqrt(d_k) prevents softmax saturation.** Without scaling, large dot products push softmax toward one-hot distributions, destroying the ability to blend information from multiple tokens. The scale factor keeps attention distributions meaningful and learnable.
