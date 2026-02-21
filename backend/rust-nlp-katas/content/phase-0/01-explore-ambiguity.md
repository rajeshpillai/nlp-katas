# Explore Ambiguity in Sentences

> Phase 0 — Language & Text (Foundations) | Kata 0.1

---

## Concept & Intuition

### What problem are we solving?

Human language is deeply ambiguous. The same word can mean completely different things depending on context. The same sentence can be parsed in multiple ways. Before we write any NLP code, we need to understand **why this is hard** — and it starts with ambiguity.

Computers work with precise, unambiguous instructions. Language is the opposite: meaning is fluid, context-dependent, and often requires world knowledge to resolve. This gap is the central challenge of NLP.

### Why naive approaches fail

A simple dictionary lookup — "bank" means "a financial institution" — fails immediately when someone says "I sat on the river bank." A word-counting approach can't tell the difference between:

- "The chicken is ready to eat" (the chicken is cooked, we can eat it)
- "The chicken is ready to eat" (the chicken is hungry, it wants food)

The words are **identical**. The meanings are **different**. No surface-level analysis can resolve this.

### Mental models

- **Lexical ambiguity**: One word, multiple meanings → "bat" (animal vs cricket bat)
- **Syntactic ambiguity**: One sentence, multiple parses → "I saw the man with the telescope" (who has the telescope?)
- **Semantic ambiguity**: One sentence, multiple interpretations → "Every student read a book" (same book or different books?)
- **Pragmatic ambiguity**: Meaning depends on situation → "Can you pass the salt?" (question about ability vs request)

### Visual explanations

```
Lexical ambiguity:

  "bank"
    ├── financial institution    ("I went to the bank to deposit money")
    ├── river edge               ("We sat on the bank of the river")
    └── to rely on               ("You can bank on it")

Syntactic ambiguity:

  "I saw the man with the telescope"
    ├── I used a telescope to see the man
    └── I saw a man who had a telescope

  Same words, same order — two completely different meanings.
```

---

## Hands-on Exploration

1. Take the word "light" and write 3 sentences where it means completely different things
2. Find a sentence that can be parsed two ways (like "I saw the man with the telescope")
3. Try to write a rule that picks the correct meaning — notice how quickly it gets complicated

---

## Live Code

```rust
fn main() {
    // --- Ambiguity in language: same words, different meanings ---

    // Lexical ambiguity: one word, multiple meanings
    let ambiguous_words: Vec<(&str, Vec<&str>)> = vec![
        ("bank", vec![
            "I deposited money at the bank.",          // financial institution
            "We had a picnic on the river bank.",       // edge of a river
            "You can bank on me to help.",              // to rely on
        ]),
        ("bat", vec![
            "The bat flew out of the cave at dusk.",    // animal
            "He swung the bat and hit the ball.",       // sports equipment
        ]),
        ("light", vec![
            "Turn on the light, please.",               // illumination
            "This bag is very light.",                   // not heavy
            "We should light the candles.",              // to ignite
        ]),
    ];

    for (word, sentences) in &ambiguous_words {
        println!("Word: \"{}\"", word);
        for (i, sentence) in sentences.iter().enumerate() {
            println!("  Meaning {}: {}", i + 1, sentence);
        }
        println!();
    }


    // --- Syntactic ambiguity: same sentence, different parses ---
    let syntactic_examples: Vec<(&str, Vec<&str>)> = vec![
        (
            "I saw the man with the telescope.",
            vec![
                "I used a telescope to see the man.",
                "I saw a man who was carrying a telescope.",
            ],
        ),
        (
            "Visiting relatives can be boring.",
            vec![
                "The act of visiting your relatives is boring.",
                "Relatives who visit you can be boring.",
            ],
        ),
        (
            "The chicken is ready to eat.",
            vec![
                "The chicken (food) is prepared — we can eat it.",
                "The chicken (animal) is hungry — it wants to eat.",
            ],
        ),
    ];

    println!("--- Syntactic Ambiguity ---\n");
    for (sentence, readings) in &syntactic_examples {
        println!("Sentence: \"{}\"", sentence);
        for (i, reading) in readings.iter().enumerate() {
            println!("  Reading {}: {}", i + 1, reading);
        }
        println!();
    }


    // --- Why this matters for NLP ---
    // A simple word-matching approach can't distinguish these meanings.
    // Let's demonstrate:

    let documents = vec![
        "I deposited money at the bank.",
        "We walked along the river bank.",
        "The bank approved my loan application.",
        "Erosion is damaging the bank of the stream.",
    ];

    let query = "bank";
    let results: Vec<&&str> = documents
        .iter()
        .filter(|doc| doc.to_lowercase().contains(&query.to_lowercase()))
        .collect();

    println!("--- Naive search for 'bank' ---\n");
    println!("Query: '{}'", query);
    println!("Results: {} documents matched", results.len());
    for r in &results {
        println!("  - {}", r);
    }
    println!();
    println!("Problem: ALL documents matched.");
    println!("The search cannot tell financial 'bank' from river 'bank'.");
    println!("This is why NLP needs more than word matching.");
}
```

---

## Key Takeaways

- **Language is inherently ambiguous.** The same word or sentence can carry multiple meanings.
- **Context resolves ambiguity** — but context is hard to capture computationally.
- **Naive word matching fails** because it treats words as flat symbols, ignoring meaning.
- **Ambiguity is not a bug in language** — it's a feature. Language is efficient *because* it reuses words and relies on context. NLP must deal with this reality.
