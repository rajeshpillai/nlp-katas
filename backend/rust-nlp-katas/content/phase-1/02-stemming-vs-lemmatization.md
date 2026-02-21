# Compare Stemming vs Lemmatization Output

> Phase 1 — Text Preprocessing | Kata 1.2

---

## Concept & Intuition

### What problem are we solving?

Words come in many forms: "running", "runs", "ran", "runner". For many NLP tasks, we want to treat these as variations of the same concept. But how do we reduce words to a common root?

There are two approaches:
- **Stemming**: Chop off suffixes using rules. Fast, crude, often produces non-words.
- **Lemmatization**: Use vocabulary and grammar to find the dictionary form. Slower, accurate, always produces real words.

Understanding the difference — and when each is appropriate — is fundamental to NLP preprocessing.

### Why naive approaches fail

You might think: "just strip common endings like -ing, -ed, -s." But:

- "caring" stems to "car" (wrong — dropped too much)
- "university" and "universe" stem to the same root "univers" (wrong — different concepts)
- "better" should reduce to "good" (no suffix rule can do this)
- "was" should reduce to "be" (irregular — needs linguistic knowledge)

Stemming is a blunt instrument. Lemmatization is precise but requires more machinery. Neither is universally correct.

### Mental models

- **Stemming = hedge trimmer**: Cuts aggressively, fast, doesn't care about shape. Good enough for many jobs.
- **Lemmatization = sculptor**: Carefully carves to the correct form. Slower, but the result is a real word.
- **Normalization spectrum**: Raw word → Stem (approximate root) → Lemma (dictionary form). More normalization = less variation, but also less distinction.

### Visual explanations

```
Word: "running"

Stemming (rule-based):
  running → remove "ning"? No.
  running → remove "ing"  → "runn" → "run"
  Result: "run" (happens to be correct!)

Word: "caring"

Stemming (rule-based):
  caring → remove "ing" → "car"
  Result: "car" (WRONG — should be "care")

Lemmatization (vocabulary lookup):
  caring → verb → base form → "care"
  Result: "care" (correct)

Word: "better"

Stemming: "better" → remove "er" → "bett" (nonsense)
Lemmatization: "better" → adjective → base form → "good" (correct)
```

---

## Hands-on Exploration

1. Take a paragraph and stem every word — count how many non-words you get
2. Try the same paragraph with lemmatization — compare the results
3. Find 5 words where stemming and lemmatization give different results
4. Find a case where stemming accidentally merges two unrelated words into the same stem

---

## Live Code

```rust
use std::collections::HashMap;
use std::collections::HashSet;

// --- Stemming vs Lemmatization: understanding the tradeoff ---

// We'll implement a simple suffix-stripping stemmer from scratch,
// then compare it with a more sophisticated approach.

// === STEMMING: Rule-based suffix stripping ===

fn simple_stem(word: &str) -> String {
    let word = word.to_lowercase();

    // Rule set (order matters — most specific first)
    let suffixes: Vec<(&str, &str)> = vec![
        ("ational", "ate"),    // relational -> relate
        ("tional", "tion"),    // conditional -> condition
        ("iveness", "ive"),    // effectiveness -> effective
        ("fulness", "ful"),    // hopefulness -> hopeful
        ("ousli", "ous"),      // dangerously -> dangerous
        ("ation", "ate"),      // information -> inform... wait, "inforate"?
        ("ness", ""),          // happiness -> happi
        ("ment", ""),          // movement -> move (if long enough)
        ("ling", ""),          // darling -> dar (oops)
        ("ally", ""),          // finally -> fin (oops)
        ("ing", ""),           // running -> runn
        ("ies", "i"),          // ponies -> poni
        ("ed", ""),            // jumped -> jump
        ("ly", ""),            // slowly -> slow
        ("er", ""),            // runner -> runn
        ("s", ""),             // cats -> cat
    ];

    for (suffix, replacement) in &suffixes {
        if word.ends_with(suffix) && word.len() > suffix.len() + 2 {
            let stem = &word[..word.len() - suffix.len()];
            return format!("{}{}", stem, replacement);
        }
    }

    word
}


// === LEMMATIZATION: Vocabulary-based normalization ===

// A real lemmatizer uses a dictionary + part-of-speech tags.
// Here we simulate one with a lookup table to show the concept.

fn build_lemma_table() -> HashMap<&'static str, &'static str> {
    let mut table = HashMap::new();

    // Verb forms
    for (form, base) in &[
        ("running", "run"), ("runs", "run"), ("ran", "run"),
        ("swimming", "swim"), ("swims", "swim"), ("swam", "swim"),
        ("caring", "care"), ("cares", "care"), ("cared", "care"),
        ("having", "have"), ("has", "have"), ("had", "have"),
        ("doing", "do"), ("does", "do"), ("did", "do"),
        ("going", "go"), ("goes", "go"), ("went", "go"), ("gone", "go"),
        ("was", "be"), ("were", "be"), ("been", "be"), ("being", "be"),
        ("is", "be"), ("are", "be"), ("am", "be"),
        ("said", "say"), ("says", "say"), ("saying", "say"),
        ("took", "take"), ("takes", "take"), ("taking", "take"), ("taken", "take"),
    ] {
        table.insert(*form, *base);
    }

    // Irregular adjectives
    for (form, base) in &[
        ("better", "good"), ("best", "good"),
        ("worse", "bad"), ("worst", "bad"),
        ("more", "much"), ("most", "much"),
        ("larger", "large"), ("largest", "large"),
    ] {
        table.insert(*form, *base);
    }

    // Noun plurals (irregular)
    for (form, base) in &[
        ("mice", "mouse"), ("geese", "goose"),
        ("children", "child"), ("teeth", "tooth"),
        ("feet", "foot"), ("men", "man"), ("women", "woman"),
        ("people", "person"),
    ] {
        table.insert(*form, *base);
    }

    // Regular forms
    for (form, base) in &[
        ("cats", "cat"), ("dogs", "dog"),
        ("jumping", "jump"), ("jumped", "jump"), ("jumps", "jump"),
        ("happiness", "happiness"), ("sadly", "sad"),
        ("studies", "study"), ("studied", "study"), ("studying", "study"),
        ("universities", "university"),
    ] {
        table.insert(*form, *base);
    }

    table
}

fn simple_lemmatize(word: &str, table: &HashMap<&str, &str>) -> String {
    let lower = word.to_lowercase();
    match table.get(lower.as_str()) {
        Some(lemma) => lemma.to_string(),
        None => lower,
    }
}


fn main() {
    let lemma_table = build_lemma_table();

    // === COMPARISON ===

    println!("=== Stemming vs Lemmatization: Head to Head ===\n");

    let test_words = vec![
        "running", "ran", "runs",
        "caring", "cares",
        "better", "best",
        "studies", "studying", "studied",
        "mice", "children", "geese",
        "universities", "universe",
        "was", "were", "been",
        "happiness", "sadly",
        "jumping", "jumped",
    ];

    println!("{:<16} {:<16} {:<16} Match?", "Word", "Stem", "Lemma");
    println!("{}", "-".repeat(60));
    for word in &test_words {
        let stem = simple_stem(word);
        let lemma = simple_lemmatize(word, &lemma_table);
        let match_str = if stem == lemma { "yes" } else { "NO" };
        println!("{:<16} {:<16} {:<16} {}", word, stem, lemma, match_str);
    }

    println!();

    // === WHERE STEMMING FAILS ===

    println!("=== Stemming Failures ===\n");

    let failure_cases = vec![
        ("caring", "Should reduce to \"care\", not \"car\""),
        ("university", "Should NOT merge with \"universe\""),
        ("better", "Should reduce to \"good\" (irregular)"),
        ("was", "Should reduce to \"be\" (irregular verb)"),
        ("mice", "Should reduce to \"mouse\" (irregular plural)"),
    ];

    for (word, explanation) in &failure_cases {
        let stem = simple_stem(word);
        let lemma = simple_lemmatize(word, &lemma_table);
        println!("  Word: '{}'", word);
        println!("    Stem:  '{}'", stem);
        println!("    Lemma: '{}'", lemma);
        println!("    Issue: {}", explanation);
        println!();
    }


    // === PRACTICAL IMPACT: How does this affect text matching? ===

    println!("=== Impact on Document Matching ===\n");

    let doc1 = "The children were running and swimming in the pool";
    let doc2 = "A child runs and swims every day";

    let words1: Vec<String> = doc1.to_lowercase().split_whitespace().map(|s| s.to_string()).collect();
    let words2: Vec<String> = doc2.to_lowercase().split_whitespace().map(|s| s.to_string()).collect();

    let stems1: HashSet<String> = words1.iter().map(|w| simple_stem(w)).collect();
    let stems2: HashSet<String> = words2.iter().map(|w| simple_stem(w)).collect();

    let lemmas1: HashSet<String> = words1.iter().map(|w| simple_lemmatize(w, &lemma_table)).collect();
    let lemmas2: HashSet<String> = words2.iter().map(|w| simple_lemmatize(w, &lemma_table)).collect();

    let raw_set1: HashSet<String> = words1.iter().cloned().collect();
    let raw_set2: HashSet<String> = words2.iter().cloned().collect();
    let raw_overlap: HashSet<&String> = raw_set1.intersection(&raw_set2).collect();
    let stem_overlap: HashSet<&String> = stems1.intersection(&stems2).collect();
    let lemma_overlap: HashSet<&String> = lemmas1.intersection(&lemmas2).collect();

    println!("Doc 1: '{}'", doc1);
    println!("Doc 2: '{}'", doc2);
    println!();
    if raw_overlap.is_empty() {
        println!("  Raw word overlap:   {{none}}");
    } else {
        let mut raw_vec: Vec<&&String> = raw_overlap.iter().collect();
        raw_vec.sort();
        println!("  Raw word overlap:   {:?}", raw_vec.iter().map(|s| s.as_str()).collect::<Vec<_>>());
    }
    let mut stem_vec: Vec<&&String> = stem_overlap.iter().collect();
    stem_vec.sort();
    println!("  Stem overlap:       {:?}", stem_vec.iter().map(|s| s.as_str()).collect::<Vec<_>>());
    let mut lemma_vec: Vec<&&String> = lemma_overlap.iter().collect();
    lemma_vec.sort();
    println!("  Lemma overlap:      {:?}", lemma_vec.iter().map(|s| s.as_str()).collect::<Vec<_>>());
    println!();
    println!("Lemmatization captures that 'children/child', 'were/be',");
    println!("'running/run', and 'swimming/swim' are the same concepts.");
}
```

---

## Key Takeaways

- **Stemming is fast but crude.** It chops suffixes using rules, often producing non-words ("univers", "car" from "caring"). It works well for search and information retrieval where approximate matching is acceptable.
- **Lemmatization is accurate but heavier.** It uses vocabulary and grammar knowledge to produce real dictionary forms. It's better when you need precise word meanings.
- **Neither is always right.** The choice depends on your task: search engines typically use stemming (speed matters, approximate is fine); question answering systems typically use lemmatization (precision matters).
- **Irregular forms break stemming.** English has hundreds of irregular verbs, plurals, and adjectives that no suffix rule can handle.
