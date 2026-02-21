# Build a Tiny Transformer Block

> Phase 9 — Transformer Architecture (Core Concepts) | Kata 9.2

---

## Concept & Intuition

### What problem are we solving?

In Kata 9.1, we implemented self-attention: the ability for every token to attend to every other token. But self-attention alone is not enough. A single attention operation computes a weighted sum of values -- this is a linear operation. Without additional components, stacking attention layers would still only produce linear combinations. The transformer block adds the non-linear and stabilizing components that make deep, trainable networks possible.

A transformer encoder block combines four ideas: **multi-head self-attention** (attend to different things in parallel), **residual connections** (let gradients flow through unmodified), **layer normalization** (stabilize activations), and a **feed-forward network** (add non-linearity). Each component solves a specific problem, and together they form the building block that you can stack 6, 12, or 96 times to create models of increasing capacity.

### Why earlier methods fail

**Single-head attention** looks at the input from one "perspective." But language has many simultaneous relationships: syntactic (subject-verb agreement), semantic (word meaning), positional (nearby words), and referential (pronoun resolution). A single attention head must cram all of these into one set of weights. Multi-head attention solves this by running several attention operations in parallel, each with its own Q, K, V projections, then concatenating the results.

**Deep networks without residual connections** suffer from vanishing gradients. Information from the input must pass through many transformations, and errors in later layers struggle to propagate back to earlier ones. Residual connections add a shortcut: the output of each sub-layer is added to its input. This means gradients can flow directly through the addition, bypassing the transformation entirely if needed.

**Deep networks without normalization** suffer from internal covariate shift: the distribution of activations changes at each layer, making training unstable. Layer normalization standardizes activations within each layer, keeping values in a consistent range and enabling faster, more stable training.

**Attention alone is linear.** The weighted sum of values is a linear combination. The feed-forward network (two linear layers with a non-linear activation in between) adds the capacity to learn non-linear transformations of the attended representations.

### Mental models

- **Multi-head attention as parallel perspectives**: Imagine reading a sentence. One "head" focuses on syntactic structure (subject-verb pairs), another on semantic similarity (synonyms and related words), another on positional proximity (nearby words). Each head is a separate attention mechanism with its own learned weights. The results are concatenated to give each token a rich, multi-faceted representation.
- **Residual connections as safety nets**: Think of the residual connection as saying "at worst, pass through the original input unchanged." The transformation only needs to learn the *difference* (residual) from the input, which is easier to learn than the full output from scratch.
- **Layer normalization as recalibration**: After each sub-layer, normalize the activations so they have zero mean and unit variance. This prevents values from exploding or collapsing as they pass through many layers.
- **Feed-forward network as token-level processing**: Unlike attention, which mixes information across tokens, the FFN processes each token position independently. It applies the same transformation to every token, adding non-linear capacity.

### Visual explanations

```
TRANSFORMER ENCODER BLOCK
==========================

  Input Embeddings (seq_len x d_model)
         |
         v
  +-------------------------------+
  |   Multi-Head Self-Attention   |   <-- attend to all positions
  +-------------------------------+
         |
         +---> Add (residual)  <---+   <-- input + attention output
         |                         |
         v                         |
  +-------------------------------+
  |      Layer Normalization      |   <-- stabilize activations
  +-------------------------------+
         |
         v
  +-------------------------------+
  |    Feed-Forward Network       |   <-- two linear layers + ReLU
  |    Linear(d_model, d_ff)      |       expand, activate, compress
  |    ReLU                       |
  |    Linear(d_ff, d_model)      |
  +-------------------------------+
         |
         +---> Add (residual)  <---+   <-- FFN input + FFN output
         |                         |
         v                         |
  +-------------------------------+
  |      Layer Normalization      |   <-- stabilize again
  +-------------------------------+
         |
         v
  Output Embeddings (seq_len x d_model)
  (same shape as input, but context-aware!)


MULTI-HEAD ATTENTION (n_heads = 2, d_model = 8, d_k = 4)
=========================================================

  Input (seq_len x 8)
    |
    +----------+----------+
    |          |          |
  Head 1     Head 2     ...
  (Q1,K1,V1) (Q2,K2,V2)
  d_k=4      d_k=4
    |          |
  Attn 1     Attn 2     (each head: seq_len x d_k)
    |          |
    +----+-----+
         |
    Concatenate          (seq_len x d_model)
         |
    Linear (W_O)         (project back to d_model)
         |
    Output


LAYER NORMALIZATION (per token position)
=========================================

  Token vector: [3.2, -1.5, 0.8, 2.1, -0.3, 1.7, -2.4, 0.5]
                 |
                 v
  1. Compute mean:     mu = 0.5125
  2. Compute variance: var = 3.032
  3. Normalize:        x_i' = (x_i - mu) / sqrt(var + eps)
  4. Scale and shift:  out_i = gamma * x_i' + beta
                 |
                 v
  Normalized:   [1.54, -1.16, 0.17, 0.91, -0.47, 0.68, -1.67, -0.01]
                (zero mean, unit variance -- stable for next layer)


RESIDUAL CONNECTION (why it helps)
===================================

  Without residual:     output = F(x)
    - Gradient must flow entirely through F
    - If F has vanishing gradients, learning stops

  With residual:        output = x + F(x)
    - Gradient flows through BOTH paths
    - Even if F's gradient vanishes, the identity path carries it
    - F only needs to learn the DIFFERENCE from input
```

---

## Hands-on Exploration

1. The transformer block has two "Add & Norm" operations: one after multi-head attention, and one after the feed-forward network. Remove the residual connections in the code (set the output to just the sub-layer output, without adding the input). Run the code and compare the output values. Without residuals, the values tend to drift further from the input, making deep stacking unstable.

2. The feed-forward network expands the dimension from d_model to d_ff (typically 4x larger), applies a non-linearity, then compresses back. Why expand first? Think about it this way: the expansion creates a higher-dimensional space where the non-linearity can carve out more complex decision boundaries. Try changing d_ff from 16 to 8 (same as d_model) and observe whether the output changes significantly.

3. Multi-head attention splits d_model into n_heads parallel attention operations. With d_model=8 and n_heads=2, each head has d_k=4. What happens if you use n_heads=4 (each head gets d_k=2)? More heads means more "perspectives" but each perspective sees fewer dimensions. There is a tradeoff between the number of perspectives and the richness of each one.

---

## Live Code

```rust
// --- Tiny Transformer Encoder Block from Scratch ---
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

    fn relu(x: f64) -> f64 {
        if x > 0.0 { x } else { 0.0 }
    }

    fn vec_add(v1: &[f64], v2: &[f64]) -> Vec<f64> {
        v1.iter().zip(v2.iter()).map(|(a, b)| a + b).collect()
    }

    fn mat_add(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
        a.iter().zip(b.iter()).map(|(ra, rb)| vec_add(ra, rb)).collect()
    }

    fn format_vec(v: &[f64]) -> String {
        v.iter().map(|x| format!("{:>7.3}", x)).collect::<Vec<_>>().join(" ")
    }

    // ============================================================
    // LAYER NORMALIZATION (from scratch)
    // ============================================================

    fn layer_norm(vector: &[f64]) -> (Vec<f64>, f64, f64) {
        let d = vector.len() as f64;
        let eps = 1e-5;

        // Compute mean
        let mean: f64 = vector.iter().sum::<f64>() / d;

        // Compute variance
        let variance: f64 = vector.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / d;

        // Normalize (gamma=1, beta=0 -- identity)
        let normalized: Vec<f64> = vector.iter()
            .map(|x| (x - mean) / (variance + eps).sqrt())
            .collect();

        (normalized, mean, variance)
    }

    fn layer_norm_matrix(m: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<(f64, f64)>) {
        let mut results = Vec::new();
        let mut stats = Vec::new();
        for row in m {
            let (normed, mean, var) = layer_norm(row);
            results.push(normed);
            stats.push((mean, var));
        }
        (results, stats)
    }

    // ============================================================
    // SINGLE-HEAD ATTENTION
    // ============================================================

    fn single_head_attention(
        x: &[Vec<f64>], w_q: &[Vec<f64>], w_k: &[Vec<f64>], w_v: &[Vec<f64>], d_k: usize
    ) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let q = mat_mul(x, w_q);
        let k = mat_mul(x, w_k);
        let v = mat_mul(x, w_v);

        // Scores = Q * K^T / sqrt(d_k)
        let k_t = transpose(&k);
        let scores = mat_mul(&q, &k_t);
        let scale = (d_k as f64).sqrt();
        let scaled: Vec<Vec<f64>> = scores.iter()
            .map(|row| row.iter().map(|s| s / scale).collect())
            .collect();

        // Softmax each row
        let weights: Vec<Vec<f64>> = scaled.iter().map(|row| softmax(row)).collect();

        // Output = weights * V
        let output = mat_mul(&weights, &v);
        (output, weights)
    }

    // ============================================================
    // MULTI-HEAD ATTENTION
    // ============================================================

    fn multi_head_attention(
        x: &[Vec<f64>], n_heads: usize, d_model: usize, rng: &mut Rng
    ) -> (Vec<Vec<f64>>, Vec<Vec<Vec<f64>>>) {
        let d_k = d_model / n_heads;
        let seq_len = x.len();

        let mut all_head_outputs = Vec::new();
        let mut all_head_weights = Vec::new();

        for _h in 0..n_heads {
            let w_q = make_matrix(d_model, d_k, rng);
            let w_k = make_matrix(d_model, d_k, rng);
            let w_v = make_matrix(d_model, d_k, rng);

            let (head_out, head_weights) = single_head_attention(x, &w_q, &w_k, &w_v, d_k);
            all_head_outputs.push(head_out);
            all_head_weights.push(head_weights);
        }

        // Concatenate heads: for each position, concat all head outputs
        let mut concat = Vec::new();
        for i in 0..seq_len {
            let mut combined = Vec::new();
            for h in 0..n_heads {
                combined.extend_from_slice(&all_head_outputs[h][i]);
            }
            concat.push(combined);
        }

        // Project through W_O (d_model x d_model)
        let w_o = make_matrix(d_model, d_model, rng);
        let output = mat_mul(&concat, &w_o);

        (output, all_head_weights)
    }

    // ============================================================
    // FEED-FORWARD NETWORK
    // ============================================================

    fn feed_forward(
        x: &[Vec<f64>], d_model: usize, d_ff: usize, rng: &mut Rng
    ) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let w1 = make_matrix(d_model, d_ff, rng);
        let b1 = make_vector(d_ff, rng);
        let w2 = make_matrix(d_ff, d_model, rng);
        let b2 = make_vector(d_model, rng);

        // First linear layer: X * W1 + b1
        let mut hidden = mat_mul(x, &w1);
        // Add bias and apply ReLU
        for i in 0..hidden.len() {
            for j in 0..d_ff {
                hidden[i][j] = relu(hidden[i][j] + b1[j]);
            }
        }

        // Second linear layer: hidden * W2 + b2
        let mut output = mat_mul(&hidden, &w2);
        for i in 0..output.len() {
            for j in 0..d_model {
                output[i][j] += b2[j];
            }
        }

        (output, hidden)
    }

    // ============================================================
    // RUN THE TRANSFORMER BLOCK
    // ============================================================

    // Configuration
    let words = vec!["The", "cat", "sat", "on"];
    let seq_len = words.len();
    let d_model: usize = 8;     // embedding dimension
    let n_heads: usize = 2;     // number of attention heads
    let d_ff: usize = 16;       // feed-forward hidden dimension
    let d_k = d_model / n_heads; // per-head dimension

    println!("{}", "=".repeat(60));
    println!("TINY TRANSFORMER ENCODER BLOCK");
    println!("{}", "=".repeat(60));
    println!("\nSentence:       {:?}", words);
    println!("d_model:        {}", d_model);
    println!("n_heads:        {}", n_heads);
    println!("d_k (per head): {}", d_k);
    println!("d_ff:           {}", d_ff);

    // Create input embeddings
    rng = Rng::new(42);
    let x = make_matrix(seq_len, d_model, &mut rng);

    println!("\n--- Stage 1: Input Embeddings ({} x {}) ---", seq_len, d_model);
    for (i, word) in words.iter().enumerate() {
        println!("  {:>4}: [{}]", word, format_vec(&x[i]));
    }

    // --- Sub-layer 1: Multi-Head Self-Attention ---
    let (attn_output, attn_weights) = multi_head_attention(&x, n_heads, d_model, &mut rng);

    // Residual connection: add input to attention output
    let residual_1 = mat_add(&x, &attn_output);

    // Layer normalization
    let (norm_1, norm_1_stats) = layer_norm_matrix(&residual_1);

    // --- Sub-layer 2: Feed-Forward Network ---
    let (ffn_output, ffn_hidden) = feed_forward(&norm_1, d_model, d_ff, &mut rng);

    // Residual connection: add norm_1 input to FFN output
    let residual_2 = mat_add(&norm_1, &ffn_output);

    // Layer normalization
    let (norm_2, norm_2_stats) = layer_norm_matrix(&residual_2);

    // --- Print intermediate representations ---

    println!("\n--- Stage 2: After Multi-Head Attention ({} x {}) ---", seq_len, d_model);
    for (i, word) in words.iter().enumerate() {
        println!("  {:>4}: [{}]", word, format_vec(&attn_output[i]));
    }

    println!("\n--- Stage 3: After Residual Connection (input + attention) ---");
    for (i, word) in words.iter().enumerate() {
        println!("  {:>4}: [{}]", word, format_vec(&residual_1[i]));
    }

    println!("\n--- Stage 4: After Layer Norm 1 ---");
    for (i, word) in words.iter().enumerate() {
        let (mean, var) = norm_1_stats[i];
        println!("  {:>4}: [{}]  (mu={:+.3}, var={:.3})", word, format_vec(&norm_1[i]), mean, var);
    }

    println!("\n--- Stage 5: FFN Hidden Layer ({} x {}, after ReLU) ---", seq_len, d_ff);
    for (i, word) in words.iter().enumerate() {
        let hidden_preview: Vec<f64> = ffn_hidden[i].iter().take(8).copied().collect();
        let zeros = ffn_hidden[i].iter().filter(|&&v| v == 0.0).count();
        println!("  {:>4}: [{}] ...  ({}/{} zeroed by ReLU)",
            word, format_vec(&hidden_preview), zeros, d_ff);
    }

    println!("\n--- Stage 6: After FFN ({} x {}) ---", seq_len, d_model);
    for (i, word) in words.iter().enumerate() {
        println!("  {:>4}: [{}]", word, format_vec(&ffn_output[i]));
    }

    println!("\n--- Stage 7: After Residual Connection (norm_1 + FFN) ---");
    for (i, word) in words.iter().enumerate() {
        println!("  {:>4}: [{}]", word, format_vec(&residual_2[i]));
    }

    println!("\n--- Stage 8: Final Output (After Layer Norm 2) ---");
    for (i, word) in words.iter().enumerate() {
        let (mean, var) = norm_2_stats[i];
        println!("  {:>4}: [{}]  (mu={:+.3}, var={:.3})", word, format_vec(&norm_2[i]), mean, var);
    }

    // ============================================================
    // ATTENTION WEIGHTS PER HEAD
    // ============================================================

    println!("\n{}", "=".repeat(60));
    println!("ATTENTION WEIGHTS PER HEAD");
    println!("{}", "=".repeat(60));

    for h in 0..n_heads {
        println!("\n  Head {}:", h + 1);
        let header: String = "          ".to_string()
            + &words.iter().map(|w| format!("{:>8}", w)).collect::<String>();
        println!("{}", header);
        for (i, word) in words.iter().enumerate() {
            let row_str: String = (0..seq_len)
                .map(|j| format!("{:>8.3}", attn_weights[h][i][j]))
                .collect::<String>();
            println!("  {:>6}:  {}", word, row_str);
        }
    }

    println!("\n  Each head learns a DIFFERENT attention pattern.");
    println!("  Head 1 might focus on nearby words, while Head 2");
    println!("  might focus on syntactically related words.");

    // --- Visualization: Attention Heatmaps per Head ---
    let words_ref: Vec<&str> = words.iter().copied().collect();
    for h in 0..n_heads {
        let head_weights: Vec<Vec<f64>> = (0..seq_len)
            .map(|i| (0..seq_len)
                .map(|j| (attn_weights[h][i][j] * 1000.0).round() / 1000.0)
                .collect())
            .collect();
        show_chart(&heatmap_chart(
            &format!("Attention Weights: Head {}", h + 1),
            &words_ref,
            &words_ref,
            &head_weights,
        ));
    }

    // ============================================================
    // LAYER NORM DEMONSTRATION
    // ============================================================

    println!("\n{}", "=".repeat(60));
    println!("LAYER NORMALIZATION IN DETAIL");
    println!("{}", "=".repeat(60));

    let example_vec = &residual_1[0];  // pre-norm vector for first token
    let (normed, mean, var) = layer_norm(example_vec);

    println!("\n  Before LayerNorm ('{}' after residual):", words[0]);
    println!("    Values: [{}]", format_vec(example_vec));
    println!("    Mean:   {:+.4}", mean);
    println!("    Var:    {:.4}", var);
    let min_val = example_vec.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = example_vec.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    println!("    Range:  [{:.3}, {:.3}]", min_val, max_val);

    println!("\n  After LayerNorm:");
    println!("    Values: [{}]", format_vec(&normed));
    let normed_mean: f64 = normed.iter().sum::<f64>() / normed.len() as f64;
    let normed_var: f64 = normed.iter().map(|x| (x - normed_mean).powi(2)).sum::<f64>()
        / normed.len() as f64;
    println!("    Mean:   {:+.4}  (should be ~0)", normed_mean);
    println!("    Var:    {:.4}  (should be ~1)", normed_var);
    let normed_min = normed.iter().cloned().fold(f64::INFINITY, f64::min);
    let normed_max = normed.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    println!("    Range:  [{:.3}, {:.3}]", normed_min, normed_max);

    // ============================================================
    // COMPARE INPUT VS OUTPUT
    // ============================================================

    println!("\n{}", "=".repeat(60));
    println!("INPUT vs OUTPUT: WHAT CHANGED?");
    println!("{}", "=".repeat(60));

    println!("\n  Shape preserved: ({} x {}) -> ({} x {})", seq_len, d_model, seq_len, d_model);
    println!("\n  But every token now carries information from all other tokens.");
    println!("  Let's measure how much each token changed:\n");

    for (i, word) in words.iter().enumerate() {
        // Euclidean distance between input and output
        let diff: Vec<f64> = (0..d_model)
            .map(|j| norm_2[i][j] - x[i][j])
            .collect();
        let dist: f64 = diff.iter().map(|d| d * d).sum::<f64>().sqrt();

        // Cosine similarity between input and output
        let dot_prod: f64 = x[i].iter().zip(norm_2[i].iter())
            .map(|(a, b)| a * b).sum();
        let mag_in: f64 = x[i].iter().map(|a| a * a).sum::<f64>().sqrt();
        let mag_out: f64 = norm_2[i].iter().map(|a| a * a).sum::<f64>().sqrt();
        let cos_sim = if mag_in * mag_out > 0.0 {
            dot_prod / (mag_in * mag_out)
        } else {
            0.0
        };

        println!("  '{}': distance={:.3}, cosine_similarity={:.3}", word, dist, cos_sim);
    }

    println!("\n  Same dimensionality, different content.");
    println!("  The transformer block transforms isolated embeddings");
    println!("  into context-aware representations, in one pass.");
}
```

---

## Key Takeaways

- **A transformer block has four components working together.** Multi-head self-attention captures relationships between tokens. Residual connections ensure gradient flow. Layer normalization stabilizes activations. The feed-forward network adds non-linear capacity. Remove any one, and the architecture degrades.
- **Multi-head attention provides multiple perspectives.** Each head can learn different types of relationships (syntactic, semantic, positional). The concatenated result gives each token a richer representation than any single head could provide.
- **Residual connections make depth possible.** Without them, stacking many layers would cause vanishing gradients. The identity shortcut guarantees that information and gradients can flow through the network, and each layer only needs to learn the incremental improvement (the residual).
- **The output has the same shape as the input, but different content.** A transformer block takes (seq_len x d_model) and produces (seq_len x d_model). This shape preservation is what allows blocks to be stacked arbitrarily deep, with each block refining the representations produced by the one before it.
