# Identify Noise in Real-World Text

> Phase 0 — Language & Text (Foundations) | Kata 0.2

---

## Concept & Intuition

### What problem are we solving?

Text in the real world is messy. It contains typos, inconsistent casing, HTML tags, encoding artifacts, extra whitespace, slang, abbreviations, and emoji. Before any NLP system can work with text, we need to understand **what noise looks like** and how it distorts meaning.

This kata is about seeing text the way a computer sees it — as a raw sequence of characters — and understanding how far that is from what a human reads.

### Why naive approaches fail

If you treat text as-is, you'll find that:
- "Hello", "hello", "HELLO", and "  hello  " are all different strings
- "don't" might appear as "don\u2019t", "don't", or "don&rsquo;t" depending on the source
- A product review scraped from a webpage might contain `<br>` tags, `&amp;` entities, and invisible Unicode characters

A system that doesn't account for noise will treat these as completely different inputs, even though a human reads them identically.

### Mental models

- **Signal vs noise**: The meaning you want is the signal. Everything else — formatting artifacts, encoding issues, typos — is noise.
- **The telephone game**: Every time text passes through a system (web scraping, copy-paste, database storage, API response), it can pick up noise.
- **Garbage in, garbage out**: If you feed noisy text to an NLP model, the model learns the noise, not the signal.

### Visual explanations

```
The same review, from different sources:

Source 1 (clean):
  "This product is great! Highly recommend it."

Source 2 (web scrape):
  "<p>This product is great!&nbsp;Highly recommend it.</p>\n\n"

Source 3 (social media):
  "this product is GREAT!!! highly reccomend it 😍😍"

Source 4 (encoding issue):
  "This product is great\ufffd Highly recommend it."

Same meaning. Four different strings. A computer sees four different inputs.
```

---

## Hands-on Exploration

1. Copy text from a webpage and paste it into a plain text editor — notice hidden characters
2. Compare the same message written formally vs as a text message
3. Try to write code that checks if two noisy strings "mean the same thing" — notice how many edge cases arise

---

## Live Code

```rust
use std::collections::HashSet;

fn main() {
    // --- Types of noise in real-world text ---

    // 1. Casing inconsistency
    let texts_casing = vec!["Hello World", "hello world", "HELLO WORLD", "hElLo WoRlD"];
    println!("--- Casing ---");
    for t in &texts_casing {
        let unique_chars: HashSet<char> = t.chars().collect();
        println!("  '{}' -> unique chars: {}", t, unique_chars.len());
    }
    let unique_raw: HashSet<&&str> = texts_casing.iter().collect();
    let unique_lower: HashSet<String> = texts_casing.iter().map(|t| t.to_lowercase()).collect();
    println!("  All equal? {}", unique_raw.len() == 1);
    println!("  All equal (lowered)? {}", unique_lower.len() == 1);
    println!();

    // 2. Whitespace noise
    let texts_whitespace = vec![
        "hello world",
        "  hello   world  ",
        "hello\tworld",
        "hello\n\nworld",
    ];
    println!("--- Whitespace ---");
    for t in &texts_whitespace {
        println!("  {:?}", t);
    }

    // Simple normalization: split and rejoin
    let normalized: Vec<String> = texts_whitespace
        .iter()
        .map(|t| t.split_whitespace().collect::<Vec<_>>().join(" "))
        .collect();
    println!("  After normalization: {:?}", normalized);
    let unique_norm: HashSet<&String> = normalized.iter().collect();
    println!("  All equal? {}", unique_norm.len() == 1);
    println!();

    // 3. Encoding artifacts and special characters
    let texts_encoding = vec![
        "don't",           // straight apostrophe
        "don\u{2019}t",   // curly apostrophe (Unicode RIGHT SINGLE QUOTATION MARK)
        "don&rsquo;t",    // HTML entity
    ];
    println!("--- Encoding ---");
    for t in &texts_encoding {
        let bytes: Vec<String> = t.bytes().map(|b| format!("{:#x}", b)).collect();
        println!("  '{}' -> bytes: {:?}", t, bytes);
    }
    println!();

    // 4. HTML/markup artifacts
    let html_text = r#"<div class="review">
  <p>This product is <b>great</b>!&nbsp;Highly recommend&hellip;</p>
  <br/>
  <span>★★★★★</span>
</div>"#;

    println!("--- HTML noise ---");
    let preview: String = html_text.chars().take(60).collect();
    println!("  Raw: {:?}...", preview);
    println!();

    let cleaned = strip_html(html_text);
    println!("  Cleaned: '{}'", cleaned);
    println!();

    // 5. Social media noise
    let social_texts = vec![
        "this is AMAZINGGG!!! 😍😍😍 #bestproduct @company",
        "ngl this product slaps fr fr no cap 💯",
        "Soooo good!!!! Would def reccomend 2 everyone",
        "worst. product. ever. do NOT buy 🚫🚫🚫",
    ];

    println!("--- Social media text ---");
    for t in &social_texts {
        let words: Vec<&str> = t.split_whitespace().collect();
        let has_emoji = t.chars().any(|c| c as u32 > 127);
        let has_hashtag = words.iter().any(|w| w.starts_with('#'));
        let has_mention = words.iter().any(|w| w.starts_with('@'));
        let repeated_chars = has_repeated_chars(t, 3);
        println!("  '{}'", t);
        println!("    emoji: {}, hashtag: {}, mention: {}, repeated chars: {}",
            has_emoji, has_hashtag, has_mention, repeated_chars);
    }
    println!();

    // 6. Putting it together: a noise detection function
    let test_inputs = vec![
        "Clean simple text here.",
        "  MESSY   text\twith\n\nproblems  ",
        "<p>HTML &amp; entities</p>",
        "sooooo gooood!!! 😍😍",
    ];

    println!("--- Noise analysis ---");
    for text in &test_inputs {
        let issues = analyze_noise(text);
        let label = if issues.is_empty() {
            "clean".to_string()
        } else {
            issues.join(", ")
        };
        println!("  '{}' -> {}", text, label);
    }
}

/// Remove HTML tags and decode common entities.
fn strip_html(text: &str) -> String {
    let mut result = String::new();
    let mut in_tag = false;
    for c in text.chars() {
        match c {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => result.push(c),
            _ => {}
        }
    }
    // Decode common HTML entities
    let result = result
        .replace("&nbsp;", " ")
        .replace("&hellip;", "...")
        .replace("&amp;", "&")
        .replace("&rsquo;", "\u{2019}");
    // Normalize whitespace
    result.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Check if text contains a character repeated n or more times consecutively.
fn has_repeated_chars(text: &str, min_repeat: usize) -> bool {
    let chars: Vec<char> = text.chars().collect();
    let mut count = 1;
    for i in 1..chars.len() {
        if chars[i] == chars[i - 1] {
            count += 1;
            if count >= min_repeat {
                return true;
            }
        } else {
            count = 1;
        }
    }
    false
}

/// Detect common types of noise in text.
fn analyze_noise(text: &str) -> Vec<&str> {
    let mut issues = Vec::new();

    if text != text.trim() {
        issues.push("leading/trailing whitespace");
    }
    if text.contains("  ") || text.contains('\t') || text.contains('\n') {
        issues.push("irregular whitespace");
    }
    if text.contains('<') && text.contains('>') {
        issues.push("HTML tags");
    }
    if text.contains('&') && text.contains(';') {
        issues.push("HTML entities");
    }
    if has_repeated_chars(text, 3) {
        issues.push("repeated characters");
    }
    if text.chars().any(|c| c as u32 > 127) {
        issues.push("non-ASCII characters");
    }
    let has_upper = text.chars().any(|c| c.is_uppercase());
    let has_lower = text.chars().any(|c| c.is_lowercase());
    if has_upper && has_lower {
        issues.push("mixed casing");
    }

    issues
}
```

---

## Key Takeaways

- **Real-world text is noisy.** Casing, whitespace, encoding, HTML artifacts, and social media conventions all distort the raw signal.
- **Noise is not random** — it follows patterns (web scraping produces HTML noise, social media produces emoji and slang).
- **A computer sees strings, not meaning.** "Hello" and "hello" are as different to a computer as "Hello" and "Goodbye."
- **Noise detection comes before noise removal.** You must understand what noise exists before deciding how to handle it — some "noise" (like casing or emoji) might carry meaning you want to preserve.
