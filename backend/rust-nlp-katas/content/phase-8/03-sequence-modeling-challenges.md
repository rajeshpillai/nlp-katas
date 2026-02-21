# Demonstrate Sequence Modeling Challenges

> Phase 8 — Context & Sequence Modeling | Kata 8.3

---

## Concept & Intuition

### What problem are we solving?

In Kata 8.1, we showed that word order carries meaning and BoW throws it away. In Kata 8.2, we showed that context resolves ambiguity and fixed representations miss it. The natural next step is: **build a model that processes text as a sequence**, updating its internal state word by word, so that each position carries information about everything that came before.

This is exactly what Recurrent Neural Networks (RNNs) do. An RNN reads one token at a time, maintains a hidden state vector, and updates that state at each step. The state at position t is a function of the current word and the state at position t-1. In theory, the state at the final position encodes information about the entire sequence.

But RNNs have a critical flaw: **information from early tokens fades as the sequence grows**. Each state update involves multiplying by a weight matrix, and when these multiplications are repeated many times, the contribution of early tokens either vanishes (shrinks to zero) or explodes (grows uncontrollably). This is called the **vanishing/exploding gradient problem**, and it means RNNs struggle with **long-range dependencies** -- relating words that are far apart in a sequence.

This kata demonstrates these challenges with concrete numerical examples, using plain Rust arithmetic to show how repeated multiplication causes signals to decay, and why this motivated the invention of attention mechanisms.

### Why earlier methods fail

**BoW and TF-IDF** do not process sequences at all. They produce a single unordered vector for the entire document. No temporal processing means no ability to track how meaning builds through a sentence.

**N-grams** capture local order in fixed windows, but as shown in Kata 8.1, they cannot capture long-range dependencies and suffer from exponential vocabulary growth.

**Simple windowed averaging** (Kata 8.2) is not truly sequential. It looks at a fixed window around each token but does not accumulate information across the entire sequence. Position 50 in a document cannot "see" position 1 unless the window is 50 words wide.

**RNNs** were designed to solve this: process text sequentially, carry information forward through a hidden state. But the vanishing gradient problem means they fail at the very thing they were designed to do -- maintaining long-range memory. This is not a minor technical issue; it is a fundamental mathematical limitation of sequential state updates through multiplication.

### Mental models

- **An RNN is like a game of telephone.** Each person (token position) receives a message (hidden state), modifies it slightly, and passes it on. After enough people, the original message is lost. Early information fades because it passes through too many transformations.
- **Vanishing gradients = forgetting.** When you multiply a number less than 1 by itself many times, it shrinks toward zero. If the hidden state is scaled down at each step, contributions from early tokens become negligible by the end of a long sequence.
- **Exploding gradients = instability.** When you multiply a number greater than 1 by itself many times, it grows without bound. The model's hidden state becomes numerically unstable, and outputs become meaningless.
- **Attention is direct access.** Instead of passing information through a chain of intermediate states, attention lets any position directly look at any other position. Position 50 can directly attend to position 1 without the signal passing through 49 intermediate transformations.

### Visual explanations

```
HOW AN RNN PROCESSES A SEQUENCE
================================

  Input: "The cat sat on the mat"
  Tokens: [the, cat, sat, on, the, mat]

  h_0 = [0, 0, 0]                (initial hidden state)

  Step 1: h_1 = f(h_0, "the")    (update state with first word)
  Step 2: h_2 = f(h_1, "cat")    (update state with second word)
  Step 3: h_3 = f(h_2, "sat")    (update state with third word)
  Step 4: h_4 = f(h_3, "on")     (update state with fourth word)
  Step 5: h_5 = f(h_4, "the")    (update state with fifth word)
  Step 6: h_6 = f(h_5, "mat")    (final state -- should encode whole sentence)

  Each h_t depends on ALL previous tokens... in theory.
  In practice, h_6 is dominated by recent tokens ("the", "mat")
  and has nearly forgotten early tokens ("the", "cat").


THE VANISHING SIGNAL PROBLEM
==============================

  Simplified: h_t = w * h_{t-1}   (multiply by weight w at each step)

  If w = 0.9 (slightly less than 1):

    Step  1: 0.9^1  = 0.9000
    Step  5: 0.9^5  = 0.5905
    Step 10: 0.9^10 = 0.3487
    Step 20: 0.9^20 = 0.1216
    Step 50: 0.9^50 = 0.0052   <-- almost zero!

  After 50 steps, the original signal retains only 0.5% of its strength.

  If w = 1.1 (slightly more than 1):

    Step  1: 1.1^1  = 1.1000
    Step  5: 1.1^5  = 1.6105
    Step 10: 1.1^10 = 2.5937
    Step 20: 1.1^20 = 6.7275
    Step 50: 1.1^50 = 117.39   <-- exploded!


RNN vs ATTENTION: HOW THEY ACCESS EARLY TOKENS
================================================

  RNN (sequential access):

    Token 1 ---> Token 2 ---> Token 3 ---> ... ---> Token 50
        h_1 -------> h_2 -------> h_3 -------> ... -------> h_50

    To use info from Token 1 at Token 50:
    Signal must survive 49 multiplicative transformations.
    Path length: 49 steps.  Signal decay: w^49


  Attention (direct access):

    Token 1 -----------------------------------------------+
    Token 2 -------------------------------------------+   |
    Token 3 ---------------------------------------+   |   |
    ...                                            |   |   |
    Token 50 <-------------------------------------+---+---+

    To use info from Token 1 at Token 50:
    Direct connection. No intermediate transformations.
    Path length: 1 step.  No signal decay.

  This is why attention solves the long-range dependency problem.
```

---

## Hands-on Exploration

1. Take a calculator (or Rust) and compute 0.95^n for n = 1, 5, 10, 20, 50, 100, 200. At what point does the value drop below 0.01 (1% of the original signal)? Now repeat with 0.99 -- even "almost 1" still decays. What does this tell you about an RNN's ability to remember the first word of a 200-word paragraph?

2. Consider the sentence: "The students who attended the lecture that the professor who won the award gave last semester passed the exam." The subject "students" and the verb "passed" are 15 words apart, separated by two nested relative clauses. Trace how an RNN would process this. At which step does it encounter "students"? At which step does it encounter "passed"? How many state updates separate them? If each update multiplies the hidden state by 0.9, how much of the "students" signal remains when "passed" is processed?

3. Design a sentence where the meaning of the last word depends critically on the first word, with at least 10 words in between. For example: "The key that was hidden under the third rock from the left opens [what?]." Now consider: would a BoW model, an n-gram model, or an RNN handle this best? Which would fail? Would attention help?

---

## Live Code

```rust
use std::collections::HashMap;

fn main() {
    // --- Sequence Modeling Challenges ---
    // Only Rust stdlib -- no external crates

    let mut rng = Rng::new(42);

    // ============================================================
    println!("{}", "=".repeat(65));
    println!("PART 1: A SIMPLE RNN-LIKE COMPUTATION");
    println!("{}", "=".repeat(65));
    println!();
    println!("An RNN updates a hidden state at each step:");
    println!("  h_t = tanh(W_h * h_{{t-1}} + W_x * x_t)");
    println!();
    println!("We will simulate this with scalar values for clarity.");
    println!();

    fn tanh(x: f64) -> f64 {
        if x > 20.0 { return 1.0; }
        if x < -20.0 { return -1.0; }
        let e_pos = x.exp();
        let e_neg = (-x).exp();
        (e_pos - e_neg) / (e_pos + e_neg)
    }

    let word_values: HashMap<&str, f64> = vec![
        ("the", 0.1), ("cat", 0.8), ("sat", 0.5), ("on", 0.2),
        ("mat", 0.7), ("and", 0.05), ("watched", 0.6), ("birds", 0.9),
        ("dog", 0.75), ("chased", 0.65), ("mouse", 0.85),
    ].into_iter().collect();

    let sentence = "the cat sat on the mat and watched the birds";
    let tokens: Vec<&str> = sentence.split_whitespace().collect();

    let w_h = 0.7_f64;
    let w_x = 0.5_f64;
    let mut h = 0.0_f64;

    println!("Sentence: \"{}\"", sentence);
    println!("Weights: W_h = {}, W_x = {}", w_h, w_x);
    println!("Initial state: h_0 = {}", h);
    println!();
    println!("  {:>4}  {:>10}  {:>6}  {:>16}  {:>12}", "Step", "Token", "x_t", "W_h*h + W_x*x", "h_t (tanh)");
    println!("  {}  {}  {}  {}  {}", "-".repeat(4), "-".repeat(10), "-".repeat(6), "-".repeat(16), "-".repeat(12));

    let mut states: Vec<f64> = vec![h];
    for (t, token) in tokens.iter().enumerate() {
        let x_t = *word_values.get(token).unwrap_or(&0.3);
        let pre_activation = w_h * h + w_x * x_t;
        h = tanh(pre_activation);
        states.push(h);
        println!("  {:>4}  {:>10}  {:>6.2}  {:>16.4}  {:>12.4}", t + 1, token, x_t, pre_activation, h);
    }

    println!();
    println!("Final hidden state: {:.4}", h);
    println!("This single number supposedly encodes the entire sentence.");
    println!();

    // ============================================================
    println!("{}", "=".repeat(65));
    println!("PART 2: THE VANISHING SIGNAL -- REPEATED MULTIPLICATION");
    println!("{}", "=".repeat(65));
    println!();
    println!("Core problem: when we multiply by a factor < 1 repeatedly,");
    println!("signals from early steps shrink toward zero.");
    println!();

    let weights = [0.5, 0.7, 0.9, 0.95, 0.99, 1.0, 1.01, 1.05, 1.1];
    let steps_list = [1, 5, 10, 20, 50, 100];

    print!("  {:>8}", "Weight");
    for s in &steps_list { print!("  {:>10}", format!("Step {}", s)); }
    println!();
    print!("  {}", "-".repeat(8));
    for _ in &steps_list { print!("  {}", "-".repeat(10)); }
    println!();

    for w in &weights {
        print!("  {:>8.2}", w);
        for s in &steps_list {
            let val = w.powi(*s as i32);
            if val.abs() > 999999.0 {
                print!("  {:>10}", "EXPLODED");
            } else if val.abs() < 0.0001 {
                print!("  {:>10}", "~0.0000");
            } else {
                print!("  {:>10.4}", val);
            }
        }
        println!();
    }

    println!();
    println!("Observations:");
    println!("  - w < 1.0: signal VANISHES (shrinks to zero)");
    println!("  - w = 1.0: signal preserved perfectly (but this is unstable)");
    println!("  - w > 1.0: signal EXPLODES (grows without bound)");
    println!("  - There is NO safe fixed multiplier for long sequences.");
    println!();

    // ============================================================
    println!("{}", "=".repeat(65));
    println!("PART 3: MEASURING INFORMATION RETENTION IN OUR RNN");
    println!("{}", "=".repeat(65));
    println!();
    println!("How much does the first word influence the final hidden state?");
    println!("We test by changing the first word and seeing how much");
    println!("the final state changes.");
    println!();

    fn run_rnn(tokens: &[&str], word_vals: &HashMap<&str, f64>, w_h: f64, w_x: f64) -> Vec<f64> {
        fn tanh_inner(x: f64) -> f64 {
            if x > 20.0 { return 1.0; } if x < -20.0 { return -1.0; }
            let ep = x.exp(); let en = (-x).exp(); (ep - en) / (ep + en)
        }
        let mut h = 0.0;
        let mut states = vec![h];
        for token in tokens {
            let x_t = *word_vals.get(token).unwrap_or(&0.3);
            h = tanh_inner(w_h * h + w_x * x_t);
            states.push(h);
        }
        states
    }

    let original = "the cat sat on the mat and watched the birds";
    let original_tokens: Vec<&str> = original.split_whitespace().collect();
    let original_states = run_rnn(&original_tokens, &word_values, w_h, w_x);
    let original_final = *original_states.last().unwrap();

    println!("Original sentence: \"{}\"", original);
    println!("Original final state: {:.6}", original_final);
    println!();

    let positions_to_test = [0, 1, 4, 8, 9];
    println!("{:>10}  {:>10}  {:>12}  {:>10}  {:>10}  {:>10}", "Position", "Original", "Changed to", "New final", "Abs diff", "Influence");
    println!("{}  {}  {}  {}  {}  {}", "-".repeat(10), "-".repeat(10), "-".repeat(12), "-".repeat(10), "-".repeat(10), "-".repeat(10));

    for pos in &positions_to_test {
        let original_word = original_tokens[*pos];
        let replacement = if original_word != "mouse" { "mouse" } else { "dog" };
        let mut modified_tokens = original_tokens.clone();
        modified_tokens[*pos] = replacement;
        let modified_states = run_rnn(&modified_tokens, &word_values, w_h, w_x);
        let modified_final = *modified_states.last().unwrap();
        let diff = (original_final - modified_final).abs();
        let bar_len = (diff * 200.0) as usize;
        let bar: String = "#".repeat(bar_len.min(40));
        println!("{:>10}  {:>10}  {:>12}  {:>10.6}  {:>10.6}  {}", pos, original_word, replacement, modified_final, diff, bar);
    }

    println!();
    println!("Notice: changing EARLY words has LESS impact on the final state");
    println!("than changing LATE words. The RNN forgets the beginning!");
    println!();

    // ============================================================
    println!("{}", "=".repeat(65));
    println!("PART 4: LONG-RANGE DEPENDENCIES");
    println!("{}", "=".repeat(65));
    println!();
    println!("Language often requires connecting words far apart.");
    println!("We will show that our RNN loses this connection.");
    println!();

    let examples: Vec<(&str, &str, &str, &str, &str)> = vec![
        ("The dogs", "that the man who lived in the big house near the old church", "bark loudly", "barks loudly", "plural 'dogs' requires 'bark' (no -s), not 'barks'"),
        ("The key", "which was hidden under the third rock from the left by the garden", "opens the door", "eat the cake", "'key' should associate with 'opens', not 'eat'"),
        ("The musician", "who performed at the concert in the park on Saturday evening", "played beautifully", "dissolved quickly", "'musician' should associate with 'played'"),
    ];

    for (prefix, middle, correct_ending, _wrong_ending, explanation) in &examples {
        let full_correct = format!("{} {} {}", prefix, middle, correct_ending);
        let tokens_correct: Vec<&str> = full_correct.split_whitespace().collect();
        let first_word = prefix.to_lowercase().split_whitespace().nth(1).unwrap_or("?").to_string();
        let last_word = correct_ending.to_lowercase().split_whitespace().next().unwrap_or("?").to_string();
        let distance = tokens_correct.len().saturating_sub(2);

        println!("  Subject: \"{}\"", prefix);
        println!("  Filler:  \"{}\"", middle);
        println!("  Correct: \"{}\"", correct_ending);
        println!("  Wrong:   \"{}\"", _wrong_ending);
        println!("  Key link: '{}' <-> '{}' ({} words apart)", first_word, last_word, distance);
        println!("  Why: {}", explanation);

        let decay_09 = 0.9_f64.powi(distance as i32);
        let decay_095 = 0.95_f64.powi(distance as i32);
        println!("  Signal retention (w=0.90): {:.4} ({:.1}%)", decay_09, decay_09 * 100.0);
        println!("  Signal retention (w=0.95): {:.4} ({:.1}%)", decay_095, decay_095 * 100.0);
        println!();
    }

    println!("The farther apart the related words, the weaker the connection.");
    println!("RNNs struggle precisely because these dependencies are common.");
    println!();

    // ============================================================
    println!("{}", "=".repeat(65));
    println!("PART 5: VISUALIZING STATE DECAY ACROSS A LONG SEQUENCE");
    println!("{}", "=".repeat(65));
    println!();

    let sequence_length: usize = 40;
    let initial_signal = 1.0_f64;
    let decay_factor = 0.85_f64;

    println!("Tracking how a signal from position 0 decays over {} steps", sequence_length);
    println!("Decay factor per step: {}", decay_factor);
    println!();

    let mut signal = initial_signal;
    let mut decay_positions: Vec<String> = Vec::new();
    let mut decay_signals: Vec<f64> = Vec::new();
    println!("  {:>8}  {:>8}  {:>10}  Visualization", "Position", "Signal", "Remaining");
    println!("  {}  {}  {}  {}", "-".repeat(8), "-".repeat(8), "-".repeat(10), "-".repeat(40));

    for pos in 0..=sequence_length {
        let pct = (signal / initial_signal) * 100.0;
        let bar_len = (pct / 2.5) as usize;
        let bar: String = "#".repeat(bar_len.max(0));

        decay_positions.push(pos.to_string());
        decay_signals.push((signal * 10000.0).round() / 10000.0);

        if pos % 4 == 0 || pos <= 3 || pos == sequence_length {
            println!("  {:>8}  {:>8.4}  {:>9.1}%  |{}", pos, signal, pct, bar);
        }
        signal *= decay_factor;
    }
    println!();

    let thresholds = [0.5, 0.1, 0.01, 0.001];
    println!("Signal drops below threshold at:");
    for thresh in &thresholds {
        if decay_factor > 0.0 && decay_factor < 1.0 {
            let steps_needed = thresh.ln() / decay_factor.ln();
            println!("  {:>6.1}% signal remaining at step {:>6.1}", thresh * 100.0, steps_needed);
        }
    }
    println!();

    // --- Visualization ---
    {
        let labels: Vec<&str> = decay_positions.iter().map(|s| s.as_str()).collect();
        show_chart(&line_chart(
            &format!("Signal Decay Over Sequence (decay factor = {})", decay_factor),
            &labels, "Signal Strength", &decay_signals, "#ef4444",
        ));
    }

    // ============================================================
    println!("{}", "=".repeat(65));
    println!("PART 6: WHY ATTENTION SOLVES THIS");
    println!("{}", "=".repeat(65));
    println!();
    println!("The fundamental problem:");
    println!("  RNN path from token 1 to token N: N-1 sequential steps");
    println!("  Each step multiplies the signal by a factor ~ w");
    println!("  After N steps, signal strength ~ w^(N-1)");
    println!();
    println!("The attention solution:");
    println!("  Attention path from token 1 to token N: 1 direct step");
    println!("  No intermediate multiplications");
    println!("  Signal strength: preserved fully");
    println!();

    println!("  {:>12}  {:>10}  {:>12}  {:>10}  {:>12}", "Seq Length", "RNN Path", "RNN Signal", "Attn Path", "Attn Signal");
    println!("  {}  {}  {}  {}  {}", "-".repeat(12), "-".repeat(10), "-".repeat(12), "-".repeat(10), "-".repeat(12));

    let mut rnn_vs_attn_labels: Vec<String> = Vec::new();
    let mut rnn_vs_attn_rnn: Vec<f64> = Vec::new();
    let mut rnn_vs_attn_attn: Vec<f64> = Vec::new();

    for n in &[5, 10, 20, 50, 100, 200, 500] {
        let rnn_path = n - 1;
        let rnn_signal = 0.95_f64.powi(rnn_path as i32);
        let attn_path = 1;
        let attn_signal = 1.0_f64;

        rnn_vs_attn_labels.push(n.to_string());
        rnn_vs_attn_rnn.push((rnn_signal * 1000000.0).round() / 1000000.0);
        rnn_vs_attn_attn.push(attn_signal);

        let rnn_str = if rnn_signal > 0.000001 { format!("{:.6}", rnn_signal) } else { "~0.000000".to_string() };
        println!("  {:>12}  {:>10}  {:>12}  {:>10}  {:>12.6}", n, rnn_path, rnn_str, attn_path, attn_signal);
    }

    println!();
    println!("At sequence length 100: RNN retains 0.6% of the original signal.");
    println!("At sequence length 500: RNN retains effectively 0% of the signal.");
    println!("Attention retains 100% at ANY distance.");

    // --- Visualization ---
    {
        let labels: Vec<&str> = rnn_vs_attn_labels.iter().map(|s| s.as_str()).collect();
        let datasets: Vec<(&str, &[f64], &str)> = vec![
            ("RNN (w=0.95)", rnn_vs_attn_rnn.as_slice(), "#ef4444"),
            ("Attention", rnn_vs_attn_attn.as_slice(), "#10b981"),
        ];
        show_chart(&multi_bar_chart(
            "RNN vs Attention: Signal Retention by Sequence Length",
            &labels, &datasets,
        ));
    }

    println!();
    println!("This is why transformers replaced RNNs for most NLP tasks.");
    println!("Attention provides direct connections between all positions,");
    println!("eliminating the long-range dependency problem entirely.");
    println!();

    // ============================================================
    println!("{}", "=".repeat(65));
    println!("SUMMARY: THE PATH FROM BOW TO ATTENTION");
    println!("{}", "=".repeat(65));
    println!();
    println!("  BoW/TF-IDF:  No order.     No context.   No sequence.");
    println!("  N-grams:     Local order.  Local context. No long-range.");
    println!("  RNNs:        Full order.   Sequential.    Vanishing signal.");
    println!("  Attention:   Full order.   Direct access. No decay.");
    println!();
    println!("Each method solves the previous method's biggest limitation.");
    println!("Attention (and Transformers) solve the vanishing signal problem");
    println!("by letting every token directly attend to every other token.");
    println!("This is the subject of Phase 9.");
}
```

---

## Key Takeaways

- **Sequential processing matters because language builds meaning over time.** Each word modifies, extends, or reverses what came before. A model that reads word by word and maintains a running state can in principle capture these dynamics -- but only if the state faithfully preserves earlier information.
- **Repeated multiplication causes signals to vanish or explode.** An RNN updates its hidden state by multiplying by weight matrices at each step. After many steps, early signals are multiplied by the weight factor raised to a high power. If the factor is less than 1, the signal vanishes; if greater than 1, it explodes. There is no stable fixed multiplier for arbitrary-length sequences.
- **Long-range dependencies are pervasive in natural language.** Subjects and verbs, pronouns and antecedents, negation and the words it scopes over -- these relationships routinely span 10, 20, or 50+ words. RNNs struggle precisely because these common patterns require remembering information across many sequential steps.
- **Attention eliminates the long path problem.** Instead of routing information through a chain of N-1 intermediate states, attention mechanisms create direct connections between any two positions. The path length drops from N-1 to 1, and signals are preserved regardless of sequence length. This is the core motivation for the Transformer architecture explored in Phase 9.
