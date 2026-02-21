# Encoder-Only vs Decoder-Only Models

> Phase 10 — Modern NLP Pipelines (Awareness) | Kata 10.2

---

## Concept & Intuition

### What problem are we solving?

Not all NLP tasks are the same. Some tasks require **understanding an entire input** (classify this email, find entities in this sentence, measure similarity between two documents). Other tasks require **generating new text** (complete this sentence, write the next paragraph, translate to another language). These two families of tasks have fundamentally different computational requirements.

Modern transformer-based architectures split along this line. **Encoder-only models** (BERT-style) process the full input bidirectionally -- every word can attend to every other word. **Decoder-only models** (GPT-style) process text left-to-right -- each word can only see what came before it. This is not an arbitrary design choice. It reflects what each architecture is optimized to do.

### Why earlier methods fail

**Classical methods** (BoW, TF-IDF) have no notion of directionality at all. They treat documents as unordered bags of features. This works for simple classification but cannot generate coherent text or model sequential dependencies.

**Simple sequence models** (basic RNNs) process text in one direction but struggle with long-range dependencies. They also cannot be efficiently parallelized during training, making them impractical at scale.

**A single architecture for all tasks** forces compromises. If you give the model access to the full input (bidirectional), it cannot generate text autoregressively because it would "see the future." If you restrict it to left-to-right processing, it loses information from the right context when doing classification. The split into encoder-only and decoder-only architectures resolves this tension.

### Mental models

- **Encoder-only (BERT-style) is like reading a complete document.** You see all the words at once and form a comprehensive understanding. You can answer questions about the text, classify its topic, find named entities -- anything that requires analyzing the full input.
- **Decoder-only (GPT-style) is like writing a sentence word by word.** You have written "The cat sat on the" and must predict the next word. You can only use what came before. This constraint is what makes text generation possible -- each prediction extends the sequence.
- **Masked language modeling** (encoder-only training) is like a fill-in-the-blank test. Given "The ___ sat on the mat", predict the masked word using context from both sides. This forces the model to build rich bidirectional representations.
- **Autoregressive modeling** (decoder-only training) is like finishing someone's sentence. Given "The cat sat on the", predict "mat". The model learns to generate text that follows naturally from what came before.

### Visual explanations

```
ENCODER-ONLY (BERT-style): BIDIRECTIONAL ATTENTION
====================================================

  Input:  "The  cat  [MASK]  on  the  mat"

  Each word attends to EVERY other word:

    The  --> cat, [MASK], on, the, mat
    cat  --> The, [MASK], on, the, mat
  [MASK] --> The, cat, on, the, mat         <-- uses BOTH sides
    on   --> The, cat, [MASK], the, mat
    the  --> The, cat, [MASK], on, mat
    mat  --> The, cat, [MASK], on, the

  Prediction for [MASK]: "sat"
  (uses left context "The cat" AND right context "on the mat")

  Good for: classification, NER, similarity, question answering
  Bad for:  text generation (cannot mask the future during generation)


DECODER-ONLY (GPT-style): LEFT-TO-RIGHT ATTENTION
====================================================

  Input:  "The  cat  sat  on  the  ___"

  Each word can ONLY attend to previous words:

    The  --> (nothing -- first word)
    cat  --> The
    sat  --> The, cat
    on   --> The, cat, sat
    the  --> The, cat, sat, on
    ___  --> The, cat, sat, on, the        <-- only left context

  Prediction for ___: "mat"
  (uses only preceding context -- cannot peek ahead)

  Good for: text generation, completion, conversation
  Bad for:  tasks needing full bidirectional context


TASK ALIGNMENT
===============

  Task                    Best architecture    Why
  ---------------------   ------------------   ---------------------
  Text classification     Encoder-only         Needs full input
  Named entity recog.     Encoder-only         Needs surrounding context
  Sentence similarity     Encoder-only         Needs to represent whole input
  Sentiment analysis      Encoder-only         Needs full sentence meaning

  Text completion         Decoder-only         Generates word by word
  Story generation        Decoder-only         Extends text sequentially
  Code completion         Decoder-only         Predicts next tokens
  Conversation            Decoder-only         Generates responses

  Translation             Encoder-Decoder      Reads source, generates target
  Summarization           Encoder-Decoder      Reads document, generates summary
```

---

## Hands-on Exploration

1. The code below simulates both masked language modeling (encoder-style) and autoregressive generation (decoder-style) using simple co-occurrence statistics. Run it and observe: when predicting a masked word, having context from BOTH sides produces better predictions than using only the left side. Why does bidirectional context help for fill-in-the-blank but create problems for generation?

2. Look at the autoregressive generation output. The model generates text one word at a time, each time choosing from words that commonly follow the current context. Notice how the generated text stays somewhat coherent locally but may drift topically. This is a fundamental property of left-to-right generation. What would happen if the context window were larger? Smaller?

3. Consider a real-world task: extracting product names from customer support emails. Would you want an encoder-only or decoder-only architecture? Now consider a different task: generating auto-reply suggestions for those same emails. How does the task requirement change which architecture fits better? Map three tasks from your own work to the correct architecture family.

---

## Live Code

```rust
// --- Encoder-Only vs Decoder-Only: Simulation with Co-occurrence ---
// Pure Rust stdlib -- no external dependencies

use std::collections::HashMap;

fn main() {
    let mut rng = Rng::new(42);

    // ============================================================
    // BUILD A LANGUAGE MODEL FROM CORPUS
    // ============================================================

    println!("{}", "=".repeat(65));
    println!("BUILDING LANGUAGE MODEL FROM CORPUS");
    println!("{}", "=".repeat(65));

    // A corpus with enough structure to demonstrate both approaches
    let corpus = vec![
        "the cat sat on the warm mat near the window",
        "the dog sat on the soft rug near the door",
        "the cat chased the small mouse across the yard",
        "the dog fetched the red ball across the park",
        "the cat slept on the warm blanket all morning",
        "the dog played on the green grass all afternoon",
        "the bird sang on the tall tree near the pond",
        "the fish swam in the clear pond near the rocks",
        "the cat watched the bird from the window ledge",
        "the dog watched the cat from the garden fence",
        "a warm breeze blew across the green yard today",
        "the small bird flew across the clear blue sky",
        "the red ball rolled across the soft green grass",
        "the tall tree stood near the old garden fence",
    ];

    println!("\nCorpus: {} sentences", corpus.len());

    fn tokenize(text: &str) -> Vec<String> {
        text.to_lowercase().split_whitespace().map(|s| s.to_string()).collect()
    }

    // Build bigram model (word pairs) for autoregressive prediction
    // Build context-window model for masked prediction
    let mut bigram_counts: HashMap<String, HashMap<String, usize>> = HashMap::new();
    let mut context_counts: HashMap<String, HashMap<String, usize>> = HashMap::new();
    let mut word_freq: HashMap<String, usize> = HashMap::new();

    let window: usize = 2;

    for sentence in &corpus {
        let tokens = tokenize(sentence);
        for (i, token) in tokens.iter().enumerate() {
            *word_freq.entry(token.clone()).or_insert(0) += 1;

            // Bigram: current word predicts next word
            if i < tokens.len() - 1 {
                *bigram_counts
                    .entry(token.clone())
                    .or_insert_with(HashMap::new)
                    .entry(tokens[i + 1].clone())
                    .or_insert(0) += 1;
            }

            // Context window: surrounding words predict center word
            let start = if i >= window { i - window } else { 0 };
            let end = (i + window + 1).min(tokens.len());
            for j in start..end {
                if i != j {
                    *context_counts
                        .entry(tokens[j].clone())
                        .or_insert_with(HashMap::new)
                        .entry(token.clone())
                        .or_insert(0) += 1;
                }
            }
        }
    }

    let mut vocab: Vec<String> = word_freq.keys().cloned().collect();
    vocab.sort();
    let total_tokens: usize = word_freq.values().sum();
    println!("Vocabulary: {} words", vocab.len());
    println!("Total tokens: {}", total_tokens);

    // ============================================================
    // ENCODER-STYLE: MASKED LANGUAGE MODEL (BIDIRECTIONAL)
    // ============================================================

    println!();
    println!("{}", "=".repeat(65));
    println!("ENCODER-STYLE: MASKED LANGUAGE MODEL (BIDIRECTIONAL)");
    println!("{}", "=".repeat(65));
    println!();
    println!("Task: predict the [MASK]ed word using context from BOTH sides.");
    println!("This simulates how encoder-only models like BERT are trained.\n");

    fn predict_masked_bidirectional(
        tokens: &[String], mask_pos: usize,
        context_counts: &HashMap<String, HashMap<String, usize>>,
        top_n: usize,
    ) -> Vec<(String, f64)> {
        let mut candidate_scores: HashMap<String, usize> = HashMap::new();

        for (i, token) in tokens.iter().enumerate() {
            if i == mask_pos { continue; }
            if let Some(counts) = context_counts.get(token.as_str()) {
                for (candidate, count) in counts {
                    *candidate_scores.entry(candidate.clone()).or_insert(0) += count;
                }
            }
        }

        let total: usize = candidate_scores.values().sum();
        let mut results: Vec<(String, f64)> = candidate_scores.iter()
            .map(|(w, s)| (w.clone(), if total > 0 { *s as f64 / total as f64 } else { 0.0 }))
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.into_iter().take(top_n).collect()
    }

    fn predict_masked_left_only(
        tokens: &[String], mask_pos: usize,
        context_counts: &HashMap<String, HashMap<String, usize>>,
        top_n: usize,
    ) -> Vec<(String, f64)> {
        let mut candidate_scores: HashMap<String, usize> = HashMap::new();

        for i in 0..mask_pos {
            let token = &tokens[i];
            if let Some(counts) = context_counts.get(token.as_str()) {
                for (candidate, count) in counts {
                    *candidate_scores.entry(candidate.clone()).or_insert(0) += count;
                }
            }
        }

        let total: usize = candidate_scores.values().sum();
        let mut results: Vec<(String, f64)> = candidate_scores.iter()
            .map(|(w, s)| (w.clone(), if total > 0 { *s as f64 / total as f64 } else { 0.0 }))
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.into_iter().take(top_n).collect()
    }

    // Test masked prediction on several examples
    let mask_tests: Vec<(&str, usize, &str)> = vec![
        ("the cat [MASK] on the warm mat", 2, "sat"),
        ("the [MASK] chased the small mouse", 1, "cat"),
        ("the dog played on the [MASK] grass", 5, "green"),
        ("the bird sang on the tall [MASK]", 6, "tree"),
        ("the cat watched the [MASK] from the window", 4, "bird"),
    ];

    for (sentence, mask_pos, true_word) in &mask_tests {
        let mut tokens: Vec<String> = sentence.replace("[MASK]", "[mask]")
            .to_lowercase().split_whitespace()
            .map(|s| s.to_string()).collect();
        tokens[*mask_pos] = "[mask]".to_string();

        println!("  Sentence: \"{}\"", sentence);
        println!("  True word: \"{}\"", true_word);

        // Bidirectional prediction (encoder-style)
        let bi_preds = predict_masked_bidirectional(&tokens, *mask_pos, &context_counts, 5);
        let bi_str: String = bi_preds.iter()
            .map(|(w, p)| format!("{} ({:.3})", w, p))
            .collect::<Vec<_>>()
            .join(", ");
        let bi_rank = bi_preds.iter().position(|(w, _)| w == true_word)
            .map(|i| format!("{}", i + 1))
            .unwrap_or_else(|| ">5".to_string());

        // Left-only prediction
        let left_preds = predict_masked_left_only(&tokens, *mask_pos, &context_counts, 5);
        let left_str: String = left_preds.iter()
            .map(|(w, p)| format!("{} ({:.3})", w, p))
            .collect::<Vec<_>>()
            .join(", ");
        let left_rank = left_preds.iter().position(|(w, _)| w == true_word)
            .map(|i| format!("{}", i + 1))
            .unwrap_or_else(|| ">5".to_string());

        println!("  Bidirectional: [{}]  (true word rank: {})", bi_str, bi_rank);
        println!("  Left-only:     [{}]  (true word rank: {})", left_str, left_rank);
        println!();
    }

    println!("  Bidirectional context (encoder-style) produces better predictions");
    println!("  because it uses evidence from BOTH sides of the masked word.");
    println!("  Left-only context misses crucial information from the right.");

    // ============================================================
    // DECODER-STYLE: AUTOREGRESSIVE GENERATION (LEFT-TO-RIGHT)
    // ============================================================

    println!();
    println!("{}", "=".repeat(65));
    println!("DECODER-STYLE: AUTOREGRESSIVE GENERATION (LEFT-TO-RIGHT)");
    println!("{}", "=".repeat(65));
    println!();
    println!("Task: generate text by predicting one word at a time.");
    println!("Each word is chosen based ONLY on preceding words.");
    println!("This simulates how decoder-only models like GPT generate text.\n");

    fn generate_autoregressive(
        seed_tokens: &[&str],
        bigram_counts: &HashMap<String, HashMap<String, usize>>,
        max_length: usize,
        temperature: f64,
        rng: &mut Rng,
    ) -> (Vec<String>, Vec<(String, Vec<(String, f64)>, String)>) {
        let mut tokens: Vec<String> = seed_tokens.iter().map(|s| s.to_string()).collect();
        let mut history = Vec::new();

        for _step in 0..max_length {
            let last_word = tokens.last().unwrap().clone();
            let candidates = match bigram_counts.get(&last_word) {
                Some(c) => c,
                None => break,
            };
            if candidates.is_empty() { break; }

            // Compute probabilities
            let total: usize = candidates.values().sum();
            let mut probs: Vec<(String, f64)> = candidates.iter()
                .map(|(w, c)| (w.clone(), *c as f64 / total as f64))
                .collect();
            probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let chosen;
            if temperature <= 0.01 {
                // Greedy: pick the most likely word
                chosen = probs[0].0.clone();
            } else {
                // Weighted random sampling
                let weights: Vec<f64> = probs.iter()
                    .map(|(_, p)| p.powf(1.0 / temperature))
                    .collect();
                let total_w: f64 = weights.iter().sum();
                let weights_norm: Vec<f64> = weights.iter().map(|w| w / total_w).collect();

                let r = rng.next_f64();
                let mut cumulative = 0.0;
                chosen = {
                    let mut result = probs[0].0.clone();
                    for (i, w) in weights_norm.iter().enumerate() {
                        cumulative += w;
                        if r <= cumulative {
                            result = probs[i].0.clone();
                            break;
                        }
                    }
                    result
                };
            }

            let top3: Vec<(String, f64)> = probs.iter().take(3).cloned().collect();
            history.push((last_word, top3, chosen.clone()));
            tokens.push(chosen);
        }

        (tokens, history)
    }

    // Generate from several prompts
    let prompts: Vec<Vec<&str>> = vec![
        vec!["the", "cat"],
        vec!["the", "dog"],
        vec!["the", "bird"],
        vec!["a", "warm"],
    ];

    for prompt in &prompts {
        let prompt_str: String = prompt.join(" ");
        println!("  Prompt: \"{}\"", prompt_str);

        // Greedy generation
        let (tokens_greedy, _) = generate_autoregressive(
            prompt, &bigram_counts, 8, 0.01, &mut rng);
        println!("  Greedy:  \"{}\"", tokens_greedy.join(" "));

        // Sampled generation (more variety)
        rng = Rng::new(42);
        let (tokens_sampled, _) = generate_autoregressive(
            prompt, &bigram_counts, 8, 0.8, &mut rng);
        println!("  Sampled: \"{}\"", tokens_sampled.join(" "));

        println!();
    }

    // Show the step-by-step generation process for one example
    println!("--- Step-by-Step Generation (decoder-style) ---\n");
    let prompt = vec!["the", "cat"];
    rng = Rng::new(42);
    let (tokens, history) = generate_autoregressive(
        &prompt, &bigram_counts, 6, 0.01, &mut rng);

    println!("  Starting with: \"{}\"", prompt.join(" "));
    println!();
    for (step, (last_word, top_preds, chosen)) in history.iter().enumerate() {
        let context_so_far: String = {
            let mut parts: Vec<String> = prompt.iter().map(|s| s.to_string()).collect();
            for h in history.iter().take(step) {
                parts.push(h.2.clone());
            }
            parts.join(" ")
        };
        let pred_str: String = top_preds.iter()
            .map(|(w, p)| format!("\"{}\" ({:.2})", w, p))
            .collect::<Vec<_>>()
            .join(", ");
        println!("  Step {}: context = \"{}\"", step + 1, context_so_far);
        println!("          candidates: [{}]", pred_str);
        println!("          chose: \"{}\"", chosen);
        println!();
    }

    println!("  Final: \"{}\"\n", tokens.join(" "));
    println!("  Each word is predicted from ONLY the preceding context.");
    println!("  The model cannot look ahead -- this is the autoregressive constraint");
    println!("  that makes text generation possible.");

    // ============================================================
    // COMPARISON: SAME TEXT, DIFFERENT ARCHITECTURES
    // ============================================================

    println!();
    println!("{}", "=".repeat(65));
    println!("COMPARISON: SAME TEXT, DIFFERENT TASKS");
    println!("{}", "=".repeat(65));

    let test_sentence = "the cat sat on the warm mat";
    let test_tokens = tokenize(test_sentence);

    println!("\n  Input: \"{}\"\n", test_sentence);

    // Encoder-style: classify / analyze the full input
    println!("  ENCODER-STYLE (analyze full input):");
    println!("  The model sees ALL words simultaneously.");
    println!("  It can answer: What is the subject? Where did it sit?\n");

    // Show attention pattern: each word attends to all others
    println!("  Attention pattern (bidirectional):");
    print!("  {:>8}", "");
    for t in &test_tokens { print!("{:>6}", t); }
    println!();
    for (_i, t) in test_tokens.iter().enumerate() {
        print!("  {:>8}", t);
        for _j in 0..test_tokens.len() {
            print!("{:>6}", "  *  ");
        }
        println!();
    }

    println!();

    // Decoder-style: generate continuations
    println!("  DECODER-STYLE (generate from prefix):");
    println!("  The model sees words LEFT-TO-RIGHT only.");
    println!("  It can answer: What word comes next?\n");

    // Show causal attention pattern
    println!("  Attention pattern (causal / left-to-right):");
    print!("  {:>8}", "");
    for t in &test_tokens { print!("{:>6}", t); }
    println!();
    for (i, t) in test_tokens.iter().enumerate() {
        print!("  {:>8}", t);
        for j in 0..test_tokens.len() {
            if j <= i {
                print!("{:>6}", "  *  ");
            } else {
                print!("{:>6}", "  -  ");
            }
        }
        println!();
    }

    println!();
    println!("  *  = can attend     -  = masked (cannot see)");
    println!();
    println!("  Encoder sees the full triangle: every word connects to every word.");
    println!("  Decoder sees the lower triangle: each word only connects to the past.");

    // ============================================================
    // WHEN TO USE WHICH ARCHITECTURE
    // ============================================================

    println!();
    println!("{}", "=".repeat(65));
    println!("DECISION GUIDE: WHICH ARCHITECTURE FOR WHICH TASK?");
    println!("{}", "=".repeat(65));

    let tasks: Vec<(&str, &str, &str)> = vec![
        ("Classify email as spam/not spam", "Encoder-only",
         "Needs to analyze the full email content"),
        ("Extract person names from text", "Encoder-only",
         "Each token needs context from both sides"),
        ("Measure similarity of two docs", "Encoder-only",
         "Must represent complete documents"),
        ("Complete a code snippet", "Decoder-only",
         "Must generate tokens left-to-right"),
        ("Write a product description", "Decoder-only",
         "Must generate coherent text from a prompt"),
        ("Answer questions in a chatbot", "Decoder-only",
         "Must generate response text sequentially"),
        ("Translate English to French", "Encoder-Decoder",
         "Must read full source, then generate target"),
        ("Summarize a long article", "Encoder-Decoder",
         "Must comprehend input, then generate summary"),
    ];

    println!();
    println!("  {:<38} {:<18} {}", "Task", "Architecture", "Reason");
    println!("  {} {} {}", "-".repeat(38), "-".repeat(18), "-".repeat(40));
    for (task, arch, reason) in &tasks {
        println!("  {:<38} {:<18} {}", task, arch, reason);
    }

    println!();
    println!("  The choice is not about which is \"better\" -- it is about");
    println!("  which computational structure matches the task requirements.");
    println!();
    println!("  Encoder-only: compute rich representations of existing text.");
    println!("  Decoder-only: compute the next token given preceding tokens.");
    println!("  These are different computations, suited to different problems.");
}
```

---

## Key Takeaways

- **Encoder-only models compute bidirectional representations.** Every token attends to every other token, producing representations that capture the full context of the input. This makes them well-suited for classification, entity recognition, and similarity tasks where the model must analyze a complete input.
- **Decoder-only models compute left-to-right predictions.** Each token can only attend to previous tokens, which is the constraint that makes sequential text generation possible. The model predicts one token at a time, extending the sequence autoregressively.
- **The architecture choice is driven by the task, not by preference.** Classification needs bidirectional context (encoder). Generation needs autoregressive prediction (decoder). Translation needs both (encoder-decoder). These are engineering decisions based on computational requirements.
- **Neither architecture "understands" language.** Both compute statistical patterns over token sequences. The encoder computes patterns using full bidirectional context. The decoder computes patterns using left-to-right context. The difference is structural -- what information flows where -- not a difference in comprehension.
