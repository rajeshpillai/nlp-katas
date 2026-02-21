# Compare Structured vs Unstructured Text

> Phase 0 — Language & Text (Foundations) | Kata 0.3

---

## Concept & Intuition

### What problem are we solving?

Not all text is created equal. A CSV file with columns "name, age, city" is easy for a computer to parse — every value has a fixed position and type. A customer email saying "Hi, I'm 30 and I live in Mumbai" contains the same information, but extracting it requires **understanding language**.

NLP exists because most of the world's text is **unstructured** — it doesn't fit into neat rows and columns. Understanding the spectrum from structured to unstructured data is essential to knowing when you need NLP and when you don't.

### Why naive approaches fail

With structured data, you can write exact rules: "column 3 is the city." With unstructured text, there are no columns. The city could appear anywhere, in any format:
- "I live in Mumbai"
- "Mumbai-based professional"
- "currently residing in Mumbai, India"
- "from Bombay" (same city, old name)

Simple pattern matching breaks because natural language has infinite ways to express the same fact.

### Mental models

- **Structured data** = a filled form. Every field has a label and a slot. Easy to extract.
- **Unstructured data** = a conversation. Information is embedded in context. Hard to extract.
- **Semi-structured data** = a form with a "comments" box. Part of it is organized, part is free text.
- **The NLP spectrum**: The less structure your data has, the more NLP you need.

### Visual explanations

```
Structured (easy to parse):
  ┌──────────┬─────┬──────────┐
  │ Name     │ Age │ City     │
  ├──────────┼─────┼──────────┤
  │ Alice    │  30 │ Mumbai   │
  │ Bob      │  25 │ Delhi    │
  └──────────┴─────┴──────────┘

Semi-structured (partially parseable):
  {
    "name": "Alice",
    "bio": "30 year old developer from Mumbai who loves hiking"
  }
  → "name" is easy. Extracting age and city from "bio" requires NLP.

Unstructured (requires NLP):
  "Hi! I'm Alice. I moved to Mumbai three years ago after
   finishing my degree. I'm turning 31 next month."
  → Name, city, age are all embedded in prose.
```

---

## Hands-on Exploration

1. Take a restaurant review and try to extract the restaurant name, rating, and cuisine — notice how much harder it is than reading a CSV
2. Compare how many lines of code it takes to extract a name from a JSON object vs from a paragraph of text
3. Find an example of semi-structured data (like an HTML page) and identify which parts are structured and which are free text

---

## Live Code

```rust
fn main() {
    // --- Structured data: trivial to extract ---

    let csv_data = "name,age,city\nAlice,30,Mumbai\nBob,25,Delhi\nCharlie,35,Bangalore";

    println!("--- Structured (CSV) ---");
    let mut lines = csv_data.lines();
    let _header = lines.next(); // skip header
    for line in lines {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() == 3 {
            println!("  {} is {} years old, lives in {}", fields[0], fields[1], fields[2]);
        }
    }
    println!();

    // JSON-like structured data (using simple tuples since no serde)
    let json_data: Vec<(&str, u32, &str)> = vec![
        ("Alice", 30, "Mumbai"),
        ("Bob", 25, "Delhi"),
    ];

    println!("--- Structured (JSON-like) ---");
    for (name, age, city) in &json_data {
        println!("  {} is {} years old, lives in {}", name, age, city);
    }
    println!();


    // --- Unstructured data: same information, but embedded in prose ---

    let unstructured_texts = vec![
        "Hi, I'm Alice! I'm 30 years old and I live in Mumbai.",
        "Bob here. 25, based in Delhi. Love cricket and coding.",
        "My name is Charlie, I'm a 35-year-old engineer from Bangalore.",
    ];

    println!("--- Unstructured (free text) ---");
    for text in &unstructured_texts {
        println!("  Text: '{}'", text);
        println!("  How do we extract name, age, city from this?");
        println!();
    }


    // --- Attempting extraction from unstructured text (naive) ---

    println!("--- Naive extraction attempt ---");
    for text in &unstructured_texts {
        let result = naive_extract(text);
        let preview: String = text.chars().take(50).collect();
        println!("  Text: '{}...'", preview);
        println!("  Extracted: name={}, age={}, city={}", result.0, result.1, result.2);
        println!();
    }


    // --- Semi-structured: mix of both ---

    let semi_structured: Vec<(&str, bool, &str)> = vec![
        ("alice_m", true, "30 year old developer from Mumbai. Love hiking and coffee."),
        ("bob_d", false, "Delhi boy, 25. Into cricket, coding, and street food."),
    ];

    println!("--- Semi-structured ---");
    for (username, verified, bio) in &semi_structured {
        println!("  Username: {} (structured)", username);
        println!("  Verified: {} (structured)", verified);
        println!("  Bio: '{}' (unstructured — needs NLP)", bio);
        println!();
    }


    // --- The key comparison ---
    println!("--- Effort comparison ---");
    println!();
    println!("  Structured (CSV/JSON):");
    println!("    Code to extract city: fields[2]");
    println!("    Lines of code: 1");
    println!("    Accuracy: 100%");
    println!();
    println!("  Unstructured (free text):");
    println!("    Code to extract city: pattern matching, city lists, NLP models...");
    println!("    Lines of code: 10-100+");
    println!("    Accuracy: varies, never 100%");
    println!();
    println!("  This gap is why NLP exists.");
}

/// Try to extract name, age, city with simple patterns.
fn naive_extract(text: &str) -> (&str, String, &str) {
    // Try to find age: look for 1-2 digit numbers
    let age = extract_age(text);

    // Try to find city (hardcoded list — doesn't scale)
    let cities = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata"];
    let text_lower = text.to_lowercase();
    let city = cities
        .iter()
        .find(|c| text_lower.contains(&c.to_lowercase()))
        .unwrap_or(&"?");

    // Name? Almost impossible without NLP
    let name = "?";

    (name, age, city)
}

fn extract_age(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i].is_ascii_digit() {
            let start = i;
            while i < chars.len() && chars[i].is_ascii_digit() {
                i += 1;
            }
            let num_str: String = chars[start..i].iter().collect();
            if let Ok(n) = num_str.parse::<u32>() {
                if n >= 10 && n <= 99 {
                    return num_str;
                }
            }
        } else {
            i += 1;
        }
    }
    "?".to_string()
}
```

---

## Key Takeaways

- **Structured data has explicit labels** (column names, JSON keys). Extraction is trivial.
- **Unstructured text encodes information in prose.** Extracting it requires understanding language — this is the domain of NLP.
- **Semi-structured data is the most common** in practice — a JSON response with a free-text "description" field, an HTML page with navigation and articles.
- **NLP is expensive.** If your data is structured, use structured tools (SQL, data frames). Reach for NLP only when the information is embedded in natural language.
- **Language is symbolic, contextual, and lossy when digitized.** The same fact can be expressed in infinite ways, and converting it to a structured format always loses something.
