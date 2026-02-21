# Apply a Preprocessing Pipeline to Raw Text

> Phase 1 — Text Preprocessing | Kata 1.1

---

## Concept & Intuition

### What problem are we solving?

Raw text is inconsistent. The same word appears as "Hello", "hello", "HELLO". Sentences carry punctuation that adds nothing to meaning for most tasks. Whitespace is erratic. Before text can become numbers (which is the entire point of NLP), it must be **normalized** — brought into a consistent, predictable form.

A preprocessing pipeline is a sequence of transformations applied to raw text. Each step makes a decision about what to keep and what to discard. These decisions are not neutral — they encode assumptions about what matters.

### Why naive approaches fail

You might think: "just lowercase everything and strip punctuation." But consider:

- Lowercasing "US" (country) makes it "us" (pronoun) — meaning is lost
- Removing punctuation from "Dr. Smith" gives "Dr Smith" — minor. But from "3.14" gives "314" — catastrophic
- Stripping whitespace from "New York" gives "NewYork" — the entity is destroyed

There is no universal "correct" pipeline. Every choice is a tradeoff between simplifying the text and preserving meaning.

### Mental models

- **Pipeline as a funnel**: Raw text goes in, normalized text comes out. Each step narrows the variation but also narrows the information.
- **Lossy compression**: Preprocessing is like compressing an image — you reduce size (variation) but you can't get the original back. Choose your compression carefully.
- **Order matters**: Lowercasing before tokenization produces different results than tokenizing first. Each step transforms the input for the next step.

### Visual explanations

```
Raw text:   "  The Quick Brown FOX jumped over the lazy dog!!! 😊  "
                |
         [strip whitespace]
                |
            "The Quick Brown FOX jumped over the lazy dog!!! 😊"
                |
            [lowercase]
                |
            "the quick brown fox jumped over the lazy dog!!! 😊"
                |
         [remove punctuation]
                |
            "the quick brown fox jumped over the lazy dog  😊"
                |
          [remove emoji]
                |
            "the quick brown fox jumped over the lazy dog"
                |
        [normalize whitespace]
                |
            "the quick brown fox jumped over the lazy dog"

Each step is a decision. Each decision loses something.
```

---

## Hands-on Exploration

1. Take a messy sentence (with mixed case, punctuation, extra spaces) and apply each preprocessing step one at a time — observe what changes at each stage
2. Find a sentence where lowercasing changes the meaning (hint: proper nouns, acronyms)
3. Find a sentence where removing punctuation changes the meaning (hint: "Let's eat, grandma" vs "Let's eat grandma")

---

## Live Code

```rust
// --- Building a text preprocessing pipeline from scratch ---

// Step 1: Individual preprocessing functions

fn strip_whitespace(text: &str) -> String {
    text.trim().to_string()
}

fn lowercase(text: &str) -> String {
    text.to_lowercase()
}

fn remove_punctuation(text: &str) -> String {
    text.chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace() || *c == '_')
        .collect()
}

fn normalize_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<&str>>().join(" ")
}

#[allow(dead_code)]
fn remove_digits(text: &str) -> String {
    text.chars().filter(|c| !c.is_ascii_digit()).collect()
}

// Step 2: Build a pipeline — a list of functions applied in order

fn preprocess(text: &str, steps: &[(&str, fn(&str) -> String)]) -> String {
    println!("  Original:  '{}'", text);
    let mut current = text.to_string();
    for (name, step) in steps {
        current = step(&current);
        println!("  After {:.<25} '{}'", name, current);
    }
    println!();
    current
}

fn main() {
    // Step 3: Apply to messy real-world text

    let samples = vec![
        "  The Quick Brown FOX jumped over the lazy dog!!!  ",
        "  Dr. Smith earned $3.14 million in Q3 2024.  ",
        "I can't believe it's not butter!!! #food @brand  ",
        "NEW YORK -- The U.S. economy grew 2.5% last quarter...",
        "   Hello,    World!!   How's   it   going???   ",
    ];

    let pipeline: Vec<(&str, fn(&str) -> String)> = vec![
        ("strip_whitespace", strip_whitespace),
        ("lowercase", lowercase),
        ("remove_punctuation", remove_punctuation),
        ("normalize_whitespace", normalize_whitespace),
    ];

    println!("=== Standard Pipeline ===\n");
    for sample in &samples {
        preprocess(sample, &pipeline);
    }


    // --- Visualize token counts through the pipeline for one sample ---
    let demo_text = samples[0];
    let mut step_labels: Vec<String> = vec!["Raw".to_string()];
    let mut step_counts: Vec<f64> = vec![demo_text.split_whitespace().count() as f64];

    let mut current = demo_text.to_string();
    for (name, step) in &pipeline {
        current = step(&current);
        let label = name.replace("_", " ");
        // Capitalize first letter of each word
        let label: String = label.split_whitespace()
            .map(|w| {
                let mut c = w.chars();
                match c.next() {
                    None => String::new(),
                    Some(f) => f.to_uppercase().to_string() + &c.as_str().to_lowercase(),
                }
            })
            .collect::<Vec<_>>()
            .join(" ");
        step_labels.push(label);
        step_counts.push(current.split_whitespace().count() as f64);
    }

    let label_refs: Vec<&str> = step_labels.iter().map(|s| s.as_str()).collect();
    show_chart(&bar_chart(
        "Token Count Through Preprocessing Pipeline",
        &label_refs,
        "Token count",
        &step_counts,
        "#3b82f6",
    ));


    // Step 4: Compare different pipeline orders

    println!("\n=== Order Matters ===\n");

    let text = "The U.S. has 50 states.";

    println!("Pipeline A: lowercase -> remove punctuation -> normalize whitespace");
    let mut a = text.to_string();
    println!("  Original:  '{}'", a);
    a = lowercase(&a);
    println!("  After lowercase................. '{}'", a);
    a = remove_punctuation(&a);
    println!("  After remove_punctuation........ '{}'", a);
    a = normalize_whitespace(&a);
    println!("  After normalize_whitespace...... '{}'", a);
    println!();

    println!("Pipeline B: remove punctuation -> lowercase -> normalize whitespace");
    let mut b = text.to_string();
    println!("  Original:  '{}'", b);
    b = remove_punctuation(&b);
    println!("  After remove_punctuation........ '{}'", b);
    b = lowercase(&b);
    println!("  After lowercase................. '{}'", b);
    b = normalize_whitespace(&b);
    println!("  After normalize_whitespace...... '{}'", b);
    println!();

    println!("Same result? {}", a == b);
    println!();


    // Step 5: Show what preprocessing destroys

    println!("\n=== What Gets Lost ===\n");

    let problem_cases = vec![
        ("US vs us", "The US economy is strong."),
        ("Acronyms", "I work at NASA."),
        ("Decimal numbers", "Pi is approximately 3.14159."),
        ("Meaningful punctuation", "Let's eat, grandma!"),
        ("Contractions", "I can't, won't, and shouldn't."),
    ];

    for (label, text) in &problem_cases {
        let cleaned = preprocess(text, &pipeline);
        println!("  Issue ({}):", label);
        println!("    Before: '{}'", text);
        println!("    After:  '{}'", cleaned);
        println!();
    }
}
```

---

## Key Takeaways

- **Preprocessing is not neutral.** Every step encodes an assumption about what matters in the text.
- **Order matters.** The sequence of transformations affects the final result.
- **There is no universal pipeline.** The right pipeline depends on your task — sentiment analysis, search, classification, and translation all need different preprocessing.
- **Preprocessing is lossy.** Once information is removed, it cannot be recovered. Err on the side of keeping information unless you're certain it's noise.
