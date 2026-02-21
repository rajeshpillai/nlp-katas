# Compare Encoder-Only vs Decoder-Only Tasks

> Phase 9 — Transformer Architecture (Core Concepts) | Kata 9.3

---

## Concept & Intuition

### What problem are we solving?

The original transformer paper ("Attention Is All You Need") introduced an encoder-decoder architecture for machine translation. Since then, the field has split into two dominant paradigms: **encoder-only** models (like BERT) that see the full input bidirectionally, and **decoder-only** models (like GPT) that see only previous tokens through causal masking. These are not just architectural choices -- they determine what tasks a model can naturally perform. Understanding the difference between bidirectional and causal attention is essential for understanding modern NLP.

The core question is: **should a token be allowed to look at future tokens?** For classification, yes -- you want to see the whole sentence before deciding. For generation, no -- you cannot look at words you have not yet produced. This single design choice -- the attention mask -- separates the two paradigms.

### Why earlier methods fail

**BoW and TF-IDF** have no concept of directionality at all. They process the entire document as a bag of features. This is fine for classification but provides no mechanism for sequential generation.

**RNNs** are inherently left-to-right (or right-to-left). A standard RNN can only use past context for prediction. Bidirectional RNNs (BiRNNs) can see both directions, but they process sequentially in each direction, making them slow and still limited by the vanishing gradient problem.

**Standard self-attention** (as in Kata 9.1) is bidirectional by default: every token attends to every other token. This is perfect for encoder tasks but violates the fundamental constraint of generation: you cannot attend to tokens that do not yet exist. Causal masking solves this by setting future positions to negative infinity before softmax, ensuring they receive zero attention weight.

### Mental models

- **Encoder as a reader**: The encoder reads the entire input, looking forward and backward, building a complete understanding. Think of reading an essay before writing a summary -- you need to see all of it first. This is bidirectional attention. Every token sees every other token.
- **Decoder as a writer**: The decoder produces output one token at a time, left to right. When writing the third word, it can look back at words one and two, but it cannot look ahead to word four (which does not exist yet). This is causal (masked) attention.
- **The attention mask as a visibility constraint**: The mask is a simple binary matrix. A 1 means "can attend," a 0 means "cannot attend." For encoders, the mask is all 1s. For decoders, it is a lower-triangular matrix (1s on and below the diagonal, 0s above).
- **Tasks follow from architecture**: Classification needs global context (encoder). Generation needs sequential prediction (decoder). Translation needs both: an encoder to understand the source, and a decoder to generate the target.

### Visual explanations

```
ENCODER: BIDIRECTIONAL ATTENTION (full visibility)
===================================================

  Sentence: ["The", "cat", "is", "happy"]

  Attention mask (1 = can attend, 0 = cannot):

         The  cat   is  happy
  The  [  1    1    1    1  ]   <-- "The" sees everything
  cat  [  1    1    1    1  ]   <-- "cat" sees everything
  is   [  1    1    1    1  ]   <-- "is" sees everything
  happy[  1    1    1    1  ]   <-- "happy" sees everything

  Every token has FULL context.
  Good for: classification, NER, similarity, sentence understanding.

  Example (BERT-style):
    Input:  "The movie was [MASK] entertaining"
    Task:   Predict the masked word
    Needs:  Both "The movie was" AND "entertaining" to predict "very"


DECODER: CAUSAL (MASKED) ATTENTION (left-to-right only)
========================================================

  Sentence: ["The", "cat", "is", "happy"]

  Attention mask:

         The  cat   is  happy
  The  [  1    0    0    0  ]   <-- "The" sees only itself
  cat  [  1    1    0    0  ]   <-- "cat" sees "The" and itself
  is   [  1    1    1    0  ]   <-- "is" sees "The", "cat", and itself
  happy[  1    1    1    1  ]   <-- "happy" sees everything before it

  Each token can ONLY see previous tokens (and itself).
  Good for: text generation, language modeling, completion.

  Example (GPT-style):
    Input:  "The cat is"
    Task:   Predict the next word
    Needs:  Only "The cat is" to predict "happy"
    Cannot look ahead to future tokens!


MASKING IN PRACTICE
===================

  Raw attention scores (before masking):

         The  cat   is  happy
  The  [ 2.1  0.8  1.3  0.5 ]
  cat  [ 1.0  3.2  0.7  1.1 ]
  is   [ 0.5  1.4  2.8  0.9 ]
  happy[ 0.3  0.6  1.2  2.5 ]

  After applying causal mask (set future positions to -inf):

         The   cat    is   happy
  The  [ 2.1  -inf  -inf  -inf ]   softmax -> [1.00  0.00  0.00  0.00]
  cat  [ 1.0   3.2  -inf  -inf ]   softmax -> [0.10  0.90  0.00  0.00]
  is   [ 0.5   1.4   2.8  -inf ]   softmax -> [0.07  0.17  0.70  0.00]
  happy[ 0.3   0.6   1.2   2.5 ]   softmax -> [0.05  0.07  0.12  0.76]

  -inf through softmax becomes 0.00 -- those positions are invisible.
```

---

## Hands-on Exploration

1. Consider the sentence "I went to the bank to deposit my check." For a classification task (is this about finance or geography?), which attention pattern is better -- bidirectional or causal? Now consider generating the next word after "I went to the bank to deposit my". Can you use bidirectional attention here? Why or why not?

2. Look at the causal attention mask. The last token ("happy") can see all previous tokens. The first token ("The") can only see itself. This means earlier tokens have less context than later tokens. How might this affect the quality of representations at different positions? In practice, models use positional encoding to help, but the asymmetry is fundamental to the causal architecture.

3. Some tasks seem to need both patterns. Machine translation reads the full source sentence (encoder, bidirectional), then generates the translation word by word (decoder, causal). The original transformer uses an encoder-decoder architecture for exactly this reason. Can you think of other tasks that naturally fit each architecture? List two tasks for encoder-only and two for decoder-only.

---

## Live Code

```rust
// --- Encoder vs Decoder: Attention Masking and Task Differences ---
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
        // Filter out -inf for max computation
        let finite_vals: Vec<f64> = values.iter().copied()
            .filter(|v| *v != f64::NEG_INFINITY)
            .collect();
        if finite_vals.is_empty() {
            let n = values.len() as f64;
            return vec![1.0 / n; values.len()];
        }
        let max_val = finite_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = values.iter().map(|v| {
            if *v == f64::NEG_INFINITY { 0.0 } else { (v - max_val).exp() }
        }).collect();
        let total: f64 = exps.iter().sum();
        if total == 0.0 {
            let n = values.len() as f64;
            return vec![1.0 / n; values.len()];
        }
        exps.iter().map(|e| e / total).collect()
    }

    fn format_float(v: f64, width: usize) -> String {
        format!("{:>width$.3}", v, width = width)
    }

    // ============================================================
    // ATTENTION WITH CONFIGURABLE MASKING
    // ============================================================

    fn attention_with_mask(
        q: &[Vec<f64>], k: &[Vec<f64>], v: &[Vec<f64>],
        mask: &[Vec<i32>], d_k: usize,
    ) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let seq_len = q.len();
        let scale = (d_k as f64).sqrt();

        // Step 1: Compute raw scores Q * K^T
        let k_t = transpose(k);
        let scores = mat_mul(q, &k_t);

        // Step 2: Scale
        let scaled: Vec<Vec<f64>> = scores.iter()
            .map(|row| row.iter().map(|s| s / scale).collect())
            .collect();

        // Step 3: Apply mask (set masked positions to -inf)
        let mut masked = vec![vec![0.0; seq_len]; seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                if mask[i][j] == 0 {
                    masked[i][j] = f64::NEG_INFINITY;
                } else {
                    masked[i][j] = scaled[i][j];
                }
            }
        }

        // Step 4: Softmax
        let weights: Vec<Vec<f64>> = masked.iter().map(|row| softmax(row)).collect();

        // Step 5: Weighted sum of V
        let output = mat_mul(&weights, v);

        (output, weights, scores)
    }

    // ============================================================
    // MASK GENERATORS
    // ============================================================

    fn encoder_mask(seq_len: usize) -> Vec<Vec<i32>> {
        vec![vec![1; seq_len]; seq_len]
    }

    fn decoder_mask(seq_len: usize) -> Vec<Vec<i32>> {
        let mut mask = vec![vec![0; seq_len]; seq_len];
        for i in 0..seq_len {
            for j in 0..=i {
                mask[i][j] = 1;
            }
        }
        mask
    }

    // ============================================================
    // SETUP
    // ============================================================

    let words = vec!["The", "cat", "is", "happy"];
    let seq_len = words.len();
    let d_model: usize = 8;
    let d_k: usize = 4;

    rng = Rng::new(42);

    // Create embeddings
    let embeddings = make_matrix(seq_len, d_model, &mut rng);

    // Create shared Q, K, V projection weights
    let w_q = make_matrix(d_model, d_k, &mut rng);
    let w_k = make_matrix(d_model, d_k, &mut rng);
    let w_v = make_matrix(d_model, d_k, &mut rng);

    // Project to Q, K, V
    let q = mat_mul(&embeddings, &w_q);
    let k = mat_mul(&embeddings, &w_k);
    let v = mat_mul(&embeddings, &w_v);

    // ============================================================
    // PRINT ATTENTION MASKS
    // ============================================================

    println!("{}", "=".repeat(60));
    println!("ENCODER vs DECODER: ATTENTION MASKS");
    println!("{}", "=".repeat(60));

    let enc_mask = encoder_mask(seq_len);
    let dec_mask = decoder_mask(seq_len);

    println!("\n--- Encoder Mask (Bidirectional) ---");
    println!("  Every token sees every token.\n");
    let header: String = "          ".to_string()
        + &words.iter().map(|w| format!("{:>7}", w)).collect::<String>();
    println!("{}", header);
    for (i, word) in words.iter().enumerate() {
        let row_str: String = (0..seq_len)
            .map(|j| format!("{:>7}", enc_mask[i][j]))
            .collect::<String>();
        println!("  {:>6}:  {}", word, row_str);
    }

    println!("\n--- Decoder Mask (Causal) ---");
    println!("  Each token sees only previous tokens and itself.\n");
    let header: String = "          ".to_string()
        + &words.iter().map(|w| format!("{:>7}", w)).collect::<String>();
    println!("{}", header);
    for (i, word) in words.iter().enumerate() {
        let row_str: String = (0..seq_len)
            .map(|j| format!("{:>7}", dec_mask[i][j]))
            .collect::<String>();
        println!("  {:>6}:  {}", word, row_str);
    }

    // --- Visualization: Encoder vs Decoder Attention Masks ---
    let words_ref: Vec<&str> = words.iter().copied().collect();
    let enc_mask_f64: Vec<Vec<f64>> = enc_mask.iter()
        .map(|row| row.iter().map(|&v| v as f64).collect())
        .collect();
    show_chart(&heatmap_chart(
        "Encoder Mask (Bidirectional — Full Visibility)",
        &words_ref,
        &words_ref,
        &enc_mask_f64,
    ));

    let dec_mask_f64: Vec<Vec<f64>> = dec_mask.iter()
        .map(|row| row.iter().map(|&v| v as f64).collect())
        .collect();
    show_chart(&heatmap_chart(
        "Decoder Mask (Causal — Left-to-Right Only)",
        &words_ref,
        &words_ref,
        &dec_mask_f64,
    ));

    // ============================================================
    // RUN BOTH ATTENTION PATTERNS
    // ============================================================

    println!("\n{}", "=".repeat(60));
    println!("ENCODER ATTENTION (Bidirectional)");
    println!("{}", "=".repeat(60));

    let (enc_output, enc_weights, _enc_scores) = attention_with_mask(
        &q, &k, &v, &enc_mask, d_k);

    println!("\n--- Attention Weights (encoder: full visibility) ---");
    let header: String = "          ".to_string()
        + &words.iter().map(|w| format!("{:>8}", w)).collect::<String>();
    println!("{}", header);
    for (i, word) in words.iter().enumerate() {
        let row_str: String = (0..seq_len)
            .map(|j| format_float(enc_weights[i][j], 8))
            .collect::<String>();
        println!("  {:>6}:  {}", word, row_str);
    }

    println!("\n--- Encoder Output ---");
    for (i, word) in words.iter().enumerate() {
        let vec_str: String = enc_output[i].iter()
            .map(|v| format_float(*v, 7))
            .collect::<Vec<_>>()
            .join(" ");
        println!("  {:>6}: [{}]", word, vec_str);
    }

    // --- Visualization: Encoder Attention Weights ---
    let enc_weights_data: Vec<Vec<f64>> = (0..seq_len)
        .map(|i| (0..seq_len)
            .map(|j| (enc_weights[i][j] * 1000.0).round() / 1000.0)
            .collect())
        .collect();
    show_chart(&heatmap_chart(
        "Encoder Attention Weights (Bidirectional)",
        &words_ref,
        &words_ref,
        &enc_weights_data,
    ));

    println!("\n{}", "=".repeat(60));
    println!("DECODER ATTENTION (Causal)");
    println!("{}", "=".repeat(60));

    let (dec_output, dec_weights, _dec_scores) = attention_with_mask(
        &q, &k, &v, &dec_mask, d_k);

    println!("\n--- Attention Weights (decoder: causal masking) ---");
    let header: String = "          ".to_string()
        + &words.iter().map(|w| format!("{:>8}", w)).collect::<String>();
    println!("{}", header);
    for (i, word) in words.iter().enumerate() {
        let row_str: String = (0..seq_len)
            .map(|j| format_float(dec_weights[i][j], 8))
            .collect::<String>();
        let future_blocked = seq_len - i - 1;
        println!("  {:>6}:  {}  ({} future token(s) blocked)",
            word, row_str, future_blocked);
    }

    println!("\n--- Decoder Output ---");
    for (i, word) in words.iter().enumerate() {
        let vec_str: String = dec_output[i].iter()
            .map(|v| format_float(*v, 7))
            .collect::<Vec<_>>()
            .join(" ");
        println!("  {:>6}: [{}]", word, vec_str);
    }

    // --- Visualization: Decoder Attention Weights ---
    let dec_weights_data: Vec<Vec<f64>> = (0..seq_len)
        .map(|i| (0..seq_len)
            .map(|j| (dec_weights[i][j] * 1000.0).round() / 1000.0)
            .collect())
        .collect();
    show_chart(&heatmap_chart(
        "Decoder Attention Weights (Causal Masking)",
        &words_ref,
        &words_ref,
        &dec_weights_data,
    ));

    // ============================================================
    // COMPARE OUTPUTS: HOW MASKING CHANGES REPRESENTATIONS
    // ============================================================

    println!("\n{}", "=".repeat(60));
    println!("HOW MASKING CHANGES REPRESENTATIONS");
    println!("{}", "=".repeat(60));

    println!("\n  Same Q, K, V — different masks — different outputs.\n");

    for (i, word) in words.iter().enumerate() {
        // Euclidean distance between encoder and decoder outputs
        let diff: Vec<f64> = (0..d_k)
            .map(|j| enc_output[i][j] - dec_output[i][j])
            .collect();
        let dist: f64 = diff.iter().map(|d| d * d).sum::<f64>().sqrt();

        // How much context each version has
        let enc_ctx = seq_len;
        let dec_ctx = i + 1;

        println!("  '{}':", word);
        println!("    Encoder context: {} tokens (all)", enc_ctx);
        println!("    Decoder context: {} token(s) (up to self)", dec_ctx);
        println!("    Output difference: {:.4}", dist);
        if i == seq_len - 1 {
            println!("    Note: Last token sees ALL tokens in both modes!");
        }
        println!();
    }

    // ============================================================
    // TASK DEMONSTRATION: CLASSIFICATION (ENCODER-STYLE)
    // ============================================================

    println!("{}", "=".repeat(60));
    println!("TASK 1: CLASSIFICATION (Encoder-Style)");
    println!("{}", "=".repeat(60));

    println!("\n  Classification needs to see the ENTIRE input before deciding.");
    println!();
    println!("  Example: Sentiment analysis");
    println!("    Input:  \"The cat is happy\"");
    println!("    Output: POSITIVE or NEGATIVE");
    println!();
    println!("  Strategy: Use encoder (bidirectional) attention, then aggregate");
    println!("  all token representations into a single sentence vector.");
    println!();

    // Aggregate encoder output into a single vector (mean pooling)
    let mut sentence_vec = vec![0.0; d_k];
    for i in 0..seq_len {
        for j in 0..d_k {
            sentence_vec[j] += enc_output[i][j];
        }
    }
    for j in 0..d_k {
        sentence_vec[j] /= seq_len as f64;
    }

    // Simple linear classifier: dot product with a "positive" direction
    rng = Rng::new(99);
    let classifier_weights = make_vector(d_k, &mut rng);
    let score: f64 = sentence_vec.iter().zip(classifier_weights.iter())
        .map(|(s, w)| s * w).sum();
    let prediction = if score > 0.0 { "POSITIVE" } else { "NEGATIVE" };

    let sv_str: String = sentence_vec.iter()
        .map(|v| format_float(*v, 7))
        .collect::<Vec<_>>()
        .join(" ");
    println!("  Sentence vector (mean pool): [{}]", sv_str);
    println!("  Classifier score: {:.4}", score);
    println!("  Prediction: {}", prediction);
    println!("\n  Key: The encoder saw ALL tokens bidirectionally.");
    println!("  'happy' influenced 'The', and 'The' influenced 'happy'.");

    // ============================================================
    // TASK DEMONSTRATION: GENERATION (DECODER-STYLE)
    // ============================================================

    println!("\n{}", "=".repeat(60));
    println!("TASK 2: NEXT-TOKEN PREDICTION (Decoder-Style)");
    println!("{}", "=".repeat(60));

    println!("\n  Generation produces tokens one at a time, left to right.");
    println!("  At each step, the model can ONLY see previous tokens.");
    println!();
    println!("  Example: Language modeling");
    println!("    Step 1: Input [\"The\"]           -> predict \"cat\"");
    println!("    Step 2: Input [\"The\", \"cat\"]    -> predict \"is\"");
    println!("    Step 3: Input [\"The\", \"cat\", \"is\"] -> predict \"happy\"");
    println!();

    // Simulate next-token prediction at each position
    rng = Rng::new(77);
    // Simple "vocabulary" for demonstration
    let vocab = vec!["the", "cat", "dog", "is", "was", "happy", "sad", "running"];
    let vocab_vectors = make_matrix(vocab.len(), d_k, &mut rng);

    println!("  Simulated generation (using decoder attention):\n");

    for step in 0..(seq_len - 1) {
        // The decoder output at position 'step' is used to predict next token
        let current_output = &dec_output[step];
        let context: String = words[..=step].join(" ");

        // Score each vocabulary word by similarity to decoder output
        let scores: Vec<f64> = vocab_vectors.iter()
            .map(|v_vec| current_output.iter().zip(v_vec.iter())
                .map(|(a, b)| a * b).sum())
            .collect();

        // Apply softmax to get probabilities
        let probs = softmax(&scores);

        // Find top-3 predictions
        let mut ranked: Vec<usize> = (0..vocab.len()).collect();
        ranked.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());

        println!("  Step {}: Context = \"{}\"", step + 1, context);
        println!("    Top predictions:");
        for rank in 0..3 {
            let idx = ranked[rank];
            let bar_len = (probs[idx] * 30.0) as usize;
            let bar: String = "#".repeat(bar_len);
            println!("      {}. '{}' (prob={:.3}) {}",
                rank + 1, vocab[idx], probs[idx], bar);
        }
        println!("    Actual next: '{}'\n", words[step + 1]);
    }

    println!("  Key: At each step, the decoder ONLY sees previous tokens.");
    println!("  It cannot look ahead — causal masking enforces this.");

    // ============================================================
    // POSITIONAL ENCODING: WHY IT MATTERS
    // ============================================================

    println!("\n{}", "=".repeat(60));
    println!("POSITIONAL ENCODING: GIVING ORDER TO ATTENTION");
    println!("{}", "=".repeat(60));

    println!();
    println!("  Self-attention has no built-in sense of word order.");
    println!("  \"The cat sat\" and \"sat cat The\" produce the SAME attention");
    println!("  scores if we just use word embeddings (the set of dot products");
    println!("  is identical, just in different positions).");
    println!();
    println!("  Positional encoding adds a position-dependent signal to each");
    println!("  embedding, so the model can distinguish word order.");
    println!();

    fn positional_encoding(seq_len: usize, d_model: usize) -> Vec<Vec<f64>> {
        let mut pe = vec![vec![0.0; d_model]; seq_len];
        for pos in 0..seq_len {
            for i in 0..d_model {
                if i % 2 == 0 {
                    pe[pos][i] = (pos as f64 / 10000_f64.powf(i as f64 / d_model as f64)).sin();
                } else {
                    pe[pos][i] = (pos as f64 / 10000_f64.powf((i - 1) as f64 / d_model as f64)).cos();
                }
            }
        }
        pe
    }

    let pe = positional_encoding(seq_len, d_model);

    println!("  Positional encoding vectors (sinusoidal):\n");
    for (i, word) in words.iter().enumerate() {
        let vec_str: String = pe[i].iter()
            .map(|v| format_float(*v, 7))
            .collect::<Vec<_>>()
            .join(" ");
        println!("    pos {} ({:>5}): [{}]", i, word, vec_str);
    }

    // Show that positions are distinguishable
    println!("\n  Distance between positional encodings:\n");
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            let dist: f64 = (0..d_model)
                .map(|k| (pe[i][k] - pe[j][k]).powi(2))
                .sum::<f64>()
                .sqrt();
            println!("    pos {} <-> pos {} ({:>5} <-> {:<5}): distance = {:.4}",
                i, j, words[i], words[j], dist);
        }
    }

    println!();
    println!("  Nearby positions have smaller distances than distant positions.");
    println!("  This gives the model a smooth sense of relative ordering.");
    println!("  Without positional encoding, attention treats input as a SET,");
    println!("  not a SEQUENCE.");
    println!();

    // ============================================================
    // PARALLELISM: WHY ATTENTION BEATS RNNs
    // ============================================================

    println!("{}", "=".repeat(60));
    println!("PARALLELISM: WHY TRANSFORMERS ARE FAST");
    println!("{}", "=".repeat(60));

    println!();
    println!("  RNN (sequential):");
    println!("    Step 1: process \"The\"     -> h1");
    println!("    Step 2: process \"cat\"     -> h2 (needs h1)");
    println!("    Step 3: process \"is\"      -> h3 (needs h2)");
    println!("    Step 4: process \"happy\"   -> h4 (needs h3)");
    println!("    Total: 4 sequential steps. Cannot parallelize.");
    println!();
    println!("  Transformer attention (parallel):");
    println!("    All Q*K^T dot products computed simultaneously.");
    println!("    All softmax operations computed simultaneously.");
    println!("    All weighted sums computed simultaneously.");
    println!("    Total: 1 parallel step (with O(n^2) compute, but parallelizable).");
    println!();
    println!("  For a sequence of length {}:", seq_len);
    println!("    RNN steps:         {} (sequential, one after another)", seq_len);
    println!("    Attention steps:   1 (parallel, all at once on GPU)");
    println!();
    println!("  For a sequence of length 512:");
    println!("    RNN steps:         512 (sequential)");
    println!("    Attention steps:   1 (parallel, but 512x512 = 262,144 dot products)");
    println!();
    println!("  Transformers trade sequential time for parallel computation.");
    println!("  This is why they train much faster on modern hardware (GPUs/TPUs)");
    println!("  that excel at parallel matrix operations.");
    println!();
}
```

---

## Key Takeaways

- **Encoder and decoder differ by a single mechanism: the attention mask.** Bidirectional (encoder) masks allow every token to see every other token, enabling full-context understanding. Causal (decoder) masks restrict each token to see only previous tokens, enabling sequential generation. The same attention computation, with different masks, produces fundamentally different behavior.
- **The attention mask determines which tasks a model can perform.** Encoder-only models excel at classification, NER, and similarity tasks where full context is available. Decoder-only models excel at text generation, completion, and language modeling where outputs are produced sequentially. Choosing the wrong architecture for a task creates a fundamental mismatch.
- **Positional encoding gives transformers a sense of order.** Pure self-attention is permutation-invariant: it treats the input as a set, not a sequence. Sinusoidal positional encodings add a position-dependent signal so the model can distinguish "the cat sat" from "sat the cat." Without this, word order information is lost entirely.
- **Transformers replace sequential processing with parallel computation.** RNNs must process tokens one at a time, creating a bottleneck proportional to sequence length. Transformers compute all pairwise attention scores simultaneously, trading sequential steps for parallelizable matrix operations. This is why transformers train orders of magnitude faster on modern hardware.
