# Rule-Based Named Entity Recognition

> Phase 6 — Named Entity Recognition (NER) | Kata 6.1

---

## Concept & Intuition

### What problem are we solving?

Text contains **named entities** — references to specific people, organizations, locations, dates, and other real-world objects. Extracting these entities from unstructured text is called Named Entity Recognition (NER). It turns prose into structured data.

Consider this sentence: "Priya Sharma joined Infosys in Bangalore last March." A human immediately spots three entities: a person, a company, and a city. But to a computer, these are just tokens in a string. NER is the task of finding where entities begin and end (the **boundary** problem) and what type they are (the **labeling** problem).

In this kata, we build NER using **rules** — handcrafted patterns that exploit regularities in how entities appear in text. Rules are the oldest and most transparent approach to NER, and understanding them is essential before moving to statistical methods.

### Why earlier methods fail

Bag of Words and TF-IDF treat every word as an independent feature. They can tell you that "Bangalore" appears in a document, but they cannot tell you:
- That "Bangalore" is a LOCATION
- That "New York" is a single entity spanning two tokens
- That "Apple" in "Apple released a new phone" is an ORGANIZATION, not a fruit

Earlier representations flatten text into word counts or weighted scores. NER requires understanding **which specific tokens** in a sentence refer to real-world entities, and that requires looking at the tokens themselves, their shape, their context, and their position.

### Mental models

- **Entity = span + label**: An entity is not just a word — it is a contiguous span of text (one or more tokens) assigned a category. "New York City" is one LOCATION entity spanning three tokens.
- **Gazetteer**: A lookup table of known entities. If you have a list of 10,000 city names, you can match them against text. Fast and precise, but incomplete — new entities appear constantly.
- **Pattern rules**: Entities follow surface patterns. Person names are often capitalized. Organizations often end with "Inc.", "Ltd.", "Corp." Locations often follow prepositions like "in" or "from."
- **The boundary problem**: Deciding where an entity starts and ends is harder than it looks. In "the University of California, Berkeley," is the entity "University of California" or "University of California, Berkeley"?

### Visual explanations

```
Input text:
  "Priya Sharma joined Infosys in Bangalore last March."

NER output:
   Priya Sharma   joined   Infosys   in   Bangalore   last   March
   [--- PERSON --]          [ORG  ]       [LOCATION]         [DATE]

How rules work:

  Rule 1: Capitalized word sequences -> candidate PERSON
    "Priya Sharma" -> two consecutive capitalized words -> PERSON candidate

  Rule 2: Gazetteer lookup -> ORGANIZATION
    "Infosys" -> found in company list -> ORGANIZATION

  Rule 3: Preposition + capitalized word -> LOCATION candidate
    "in Bangalore" -> "in" + capitalized word -> LOCATION candidate

  Rule 4: Month names -> DATE
    "March" -> found in month list -> DATE

Where rules break:

  "Paris Hilton went to Paris."
   [--- PERSON --]         [LOC]
   Rule 1 says: two capitalized words -> PERSON (correct for "Paris Hilton")
   Rule 3 says: "to" + capitalized word -> LOCATION (correct for "Paris")
   But Rule 1 alone would also match "Paris" as part of a person name.
   Without context, rules cannot disambiguate.
```

---

## Hands-on Exploration

1. Take the sentence "Dr. Amara Khan works at the World Health Organization in Geneva." Identify all entities by hand, marking the exact span and type of each. Notice how "World Health Organization" is one entity spanning three words — boundary detection is not trivial.

2. Write down a set of rules that would correctly extract PERSON, ORGANIZATION, and LOCATION from 5 news-like sentences of your choice. Now find a sentence where your rules produce a wrong answer. How many rules did it take before you hit a failure case?

3. Consider the word "Washington" in these three contexts: "Washington signed the treaty," "Washington approved the bill," "I flew to Washington." The same word is a PERSON, an ORGANIZATION (metonym for government), and a LOCATION. Can any fixed rule handle all three? What additional information would you need?

---

## Live Code

```rust
use std::collections::{HashMap, BTreeSet};

fn main() {
    // ============================================================
    // RULE-BASED NAMED ENTITY RECOGNITION
    // ============================================================
    // We build a NER system using only handcrafted rules:
    //   - Gazetteers (lookup lists of known entities)
    //   - Capitalization patterns
    //   - Contextual clues (words that appear near entities)
    //   - Manual string matching for dates
    // ============================================================

    // --- Gazetteers: lists of known entities ---

    let person_titles: BTreeSet<&str> = ["mr", "mrs", "ms", "dr", "prof", "sir", "lady", "captain", "judge"]
        .iter().copied().collect();

    let known_orgs: BTreeSet<&str> = [
        "infosys", "google", "microsoft", "amazon", "tesla", "apple",
        "united nations", "world health organization", "european union",
        "reserve bank of india", "supreme court", "nasa", "unesco",
        "red cross", "greenpeace", "amnesty international",
    ].iter().copied().collect();

    let org_suffixes: BTreeSet<&str> = ["inc", "ltd", "corp", "llc", "co", "plc", "foundation", "institute", "association"]
        .iter().copied().collect();

    let known_locations: BTreeSet<&str> = [
        "mumbai", "delhi", "bangalore", "chennai", "kolkata", "hyderabad",
        "new york", "london", "paris", "tokyo", "berlin", "sydney",
        "california", "texas", "india", "china", "germany", "brazil",
        "geneva", "washington", "singapore", "toronto", "dubai",
    ].iter().copied().collect();

    let location_prepositions: BTreeSet<&str> = ["in", "at", "from", "to", "near", "across", "through"]
        .iter().copied().collect();

    let months: BTreeSet<&str> = [
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
    ].iter().copied().collect();

    // --- Tokenizer: split text preserving positions ---
    #[derive(Clone, Debug)]
    struct Token {
        text: String,
        start: usize,
        end: usize,
    }

    fn tokenize_pos(text: &str) -> Vec<Token> {
        let mut tokens = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            let ch = chars[i];
            if ch.is_alphanumeric() {
                let start = i;
                let mut word = String::new();
                while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '.') {
                    // Handle abbreviation dots within token (e.g., "Dr.")
                    if chars[i] == '.' && i + 1 < chars.len() && chars[i + 1].is_alphanumeric() {
                        word.push(chars[i]);
                    } else if chars[i] == '.' {
                        word.push(chars[i]);
                        i += 1;
                        break;
                    } else {
                        word.push(chars[i]);
                    }
                    i += 1;
                }
                tokens.push(Token { text: word, start, end: i });
            } else if ".,;:!?".contains(ch) {
                tokens.push(Token { text: ch.to_string(), start: i, end: i + 1 });
                i += 1;
            } else {
                i += 1;
            }
        }
        tokens
    }

    #[derive(Clone, Debug)]
    struct Entity {
        text: String,
        start: usize,
        end: usize,
        label: String,
        rule: String,
    }

    fn find_entities(
        text: &str,
        person_titles: &BTreeSet<&str>,
        known_orgs: &BTreeSet<&str>,
        org_suffixes: &BTreeSet<&str>,
        known_locations: &BTreeSet<&str>,
        location_prepositions: &BTreeSet<&str>,
        months: &BTreeSet<&str>,
    ) -> Vec<Entity> {
        let tokens = tokenize_pos(text);
        let mut entities: Vec<Entity> = Vec::new();
        let text_lower = text.to_lowercase();

        // --- Rule 1: Multi-word organization gazetteer lookup ---
        for org in known_orgs {
            let org_lower = org.to_lowercase();
            let mut search_from = 0;
            while let Some(pos) = text_lower[search_from..].find(&org_lower) {
                let start = search_from + pos;
                let end = start + org.len();
                let matched_text = &text[start..end];
                entities.push(Entity {
                    text: matched_text.to_string(), start, end,
                    label: "ORG".to_string(), rule: "org_gazetteer".to_string(),
                });
                search_from = end;
            }
        }

        // --- Rule 2: Organization suffix patterns ---
        for (i, tok) in tokens.iter().enumerate() {
            let tok_clean = tok.text.to_lowercase().trim_end_matches('.').to_string();
            if org_suffixes.contains(tok_clean.as_str()) {
                let mut span_start = i;
                let mut j = i as isize - 1;
                while j >= 0 {
                    let prev = &tokens[j as usize];
                    if prev.text.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                        span_start = j as usize;
                        j -= 1;
                    } else {
                        break;
                    }
                }
                if span_start < i {
                    let span_text = &text[tokens[span_start].start..tokens[i].end];
                    entities.push(Entity {
                        text: span_text.to_string(),
                        start: tokens[span_start].start,
                        end: tokens[i].end,
                        label: "ORG".to_string(),
                        rule: "org_suffix".to_string(),
                    });
                }
            }
        }

        // --- Rule 3: Title + capitalized words -> PERSON ---
        let mut used_indices: BTreeSet<usize> = BTreeSet::new();
        for (i, tok) in tokens.iter().enumerate() {
            let tok_clean = tok.text.to_lowercase().trim_end_matches('.').to_string();
            if person_titles.contains(tok_clean.as_str()) {
                let mut j = i + 1;
                while j < tokens.len() && tokens[j].text.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                    j += 1;
                }
                if j > i + 1 {
                    let span_text = &text[tokens[i].start..tokens[j - 1].end];
                    entities.push(Entity {
                        text: span_text.to_string(),
                        start: tokens[i].start,
                        end: tokens[j - 1].end,
                        label: "PERSON".to_string(),
                        rule: "title_pattern".to_string(),
                    });
                    for k in i..j {
                        used_indices.insert(k);
                    }
                }
            }
        }

        // --- Rule 4: Location preposition + capitalized word(s) -> LOCATION ---
        for (i, tok) in tokens.iter().enumerate() {
            if location_prepositions.contains(tok.text.to_lowercase().as_str()) {
                let mut span_tokens = Vec::new();
                let mut j = i + 1;
                while j < tokens.len() && tokens[j].text.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                    span_tokens.push(j);
                    j += 1;
                }
                if !span_tokens.is_empty() {
                    let candidate: String = span_tokens.iter()
                        .map(|&idx| tokens[idx].text.to_lowercase())
                        .collect::<Vec<_>>().join(" ");
                    if known_locations.contains(candidate.as_str()) || span_tokens.len() == 1 {
                        let first = span_tokens[0];
                        let last = *span_tokens.last().unwrap();
                        let span_text = &text[tokens[first].start..tokens[last].end];
                        entities.push(Entity {
                            text: span_text.to_string(),
                            start: tokens[first].start,
                            end: tokens[last].end,
                            label: "LOCATION".to_string(),
                            rule: "prep_location".to_string(),
                        });
                    }
                }
            }
        }

        // --- Rule 5: Known location gazetteer ---
        for loc in known_locations {
            let loc_lower = loc.to_lowercase();
            let mut search_from = 0;
            while let Some(pos) = text_lower[search_from..].find(&loc_lower) {
                let start = search_from + pos;
                let end = start + loc.len();
                // Check word boundaries
                let before_ok = start == 0 || !text.as_bytes()[start - 1].is_ascii_alphanumeric();
                let after_ok = end >= text.len() || !text.as_bytes()[end].is_ascii_alphanumeric();
                if before_ok && after_ok {
                    let already_covered = entities.iter().any(|e| e.start <= start && e.end >= end);
                    if !already_covered {
                        let matched_text = &text[start..end];
                        entities.push(Entity {
                            text: matched_text.to_string(), start, end,
                            label: "LOCATION".to_string(), rule: "loc_gazetteer".to_string(),
                        });
                    }
                }
                search_from = end;
            }
        }

        // --- Rule 6: Date patterns (month names) ---
        for month in months {
            let month_lower = month.to_lowercase();
            let mut search_from = 0;
            while let Some(pos) = text_lower[search_from..].find(&month_lower) {
                let start = search_from + pos;
                let mut end = start + month.len();
                // Check word boundary
                let before_ok = start == 0 || !text.as_bytes()[start - 1].is_ascii_alphanumeric();
                let after_ok = end >= text.len() || !text.as_bytes()[end].is_ascii_alphanumeric();
                if before_ok && after_ok {
                    // Try to extend with day number and year
                    let rest = &text[end..];
                    let rest_trimmed = rest.trim_start();
                    let skip = rest.len() - rest_trimmed.len();
                    // Check for day number
                    let mut day_end = 0;
                    for ch in rest_trimmed.chars() {
                        if ch.is_ascii_digit() { day_end += 1; } else { break; }
                    }
                    if day_end > 0 && day_end <= 2 {
                        end = end + skip + day_end;
                        // Check for year
                        let rest2 = &text[end..];
                        let rest2_trimmed = rest2.trim_start_matches(|c: char| c == ',' || c == ' ');
                        let skip2 = rest2.len() - rest2_trimmed.len();
                        let mut year_end = 0;
                        for ch in rest2_trimmed.chars() {
                            if ch.is_ascii_digit() { year_end += 1; } else { break; }
                        }
                        if year_end == 4 {
                            end = end + skip2 + year_end;
                        }
                    }
                    let matched_text = &text[start..end];
                    entities.push(Entity {
                        text: matched_text.to_string(), start, end,
                        label: "DATE".to_string(), rule: "month_pattern".to_string(),
                    });
                }
                search_from = start + month.len();
            }
        }

        // --- Deduplicate: keep longest span at each position ---
        entities.sort_by(|a, b| a.start.cmp(&b.start).then((b.end - b.start).cmp(&(a.end - a.start))));
        let mut filtered = Vec::new();
        let mut covered_until: usize = 0;
        for e in &entities {
            if e.start >= covered_until {
                filtered.push(e.clone());
                covered_until = e.end;
            }
        }

        filtered
    }

    fn display_entities(text: &str, entities: &[Entity]) {
        if entities.is_empty() {
            println!("  Text: {}", text);
            println!("  Entities: (none found)");
            return;
        }
        println!("  Text: {}", text);
        for e in entities {
            println!("    [{:>8}]  \"{}\"  (rule: {})", e.label, e.text, e.rule);
        }
    }

    // ============================================================
    // TEST ON SAMPLE SENTENCES
    // ============================================================

    let test_sentences = vec![
        "Dr. Amara Khan works at the World Health Organization in Geneva.",
        "Infosys announced a new office in Bangalore on March 15, 2025.",
        "Prof. Raj Mehta presented at the United Nations summit in New York.",
        "Tesla CEO visited the Google campus in California last January.",
        "The Reserve Bank of India raised interest rates in Mumbai.",
    ];

    println!("{}", "=".repeat(65));
    println!("  RULE-BASED NER: SUCCESSFUL EXTRACTIONS");
    println!("{}", "=".repeat(65));
    println!();

    for sent in &test_sentences {
        let entities = find_entities(sent, &person_titles, &known_orgs, &org_suffixes, &known_locations, &location_prepositions, &months);
        display_entities(sent, &entities);
        println!();
    }

    // ============================================================
    // WHERE RULES FAIL
    // ============================================================

    let tricky_sentences = vec![
        "May I go to Paris in May?",
        "Apple released a new product.",
        "Jordan visited Jordan last summer.",
        "just landed in mumbai, meeting priya tomorrow at infosys office",
        "The president of the United States addressed Congress in Washington.",
    ];

    let expected_failures = vec![
        "Expected: LOCATION 'Paris', DATE 'May' (end). Problem: first 'May' is a verb, not a date.",
        "Expected: ORG 'Apple'. Problem: no title/suffix, not in org gazetteer as standalone capitalized word.",
        "Expected: PERSON 'Jordan' (first), LOCATION 'Jordan' (second). Problem: same word, two types.",
        "Expected: LOCATION 'mumbai', PERSON 'priya', ORG 'infosys'. Problem: no capitalization signals.",
        "Expected: ORG 'United States', ORG 'Congress', LOCATION 'Washington'. Problem: overlapping rules.",
    ];

    println!("{}", "=".repeat(65));
    println!("  WHERE RULES FAIL");
    println!("{}", "=".repeat(65));
    println!();

    for (sent, note) in tricky_sentences.iter().zip(expected_failures.iter()) {
        let entities = find_entities(sent, &person_titles, &known_orgs, &org_suffixes, &known_locations, &location_prepositions, &months);
        display_entities(sent, &entities);
        println!("    NOTE: {}", note);
        println!();
    }

    // ============================================================
    // RULE COVERAGE ANALYSIS
    // ============================================================

    println!("{}", "=".repeat(65));
    println!("  RULE COVERAGE ANALYSIS");
    println!("{}", "=".repeat(65));
    println!();

    let all_sentences: Vec<&str> = test_sentences.iter().chain(tricky_sentences.iter()).copied().collect();
    let mut rule_counts: HashMap<String, usize> = HashMap::new();
    let mut label_counts: HashMap<String, usize> = HashMap::new();
    let mut total_entities = 0;

    for sent in &all_sentences {
        let entities = find_entities(sent, &person_titles, &known_orgs, &org_suffixes, &known_locations, &location_prepositions, &months);
        for e in &entities {
            *rule_counts.entry(e.rule.clone()).or_insert(0) += 1;
            *label_counts.entry(e.label.clone()).or_insert(0) += 1;
            total_entities += 1;
        }
    }

    println!("  Total entities found: {}", total_entities);
    println!();
    println!("  Entities by type:");
    let mut label_sorted: Vec<(&String, &usize)> = label_counts.iter().collect();
    label_sorted.sort_by(|a, b| b.1.cmp(a.1));
    for (label, count) in &label_sorted {
        let bar: String = "#".repeat(**count);
        println!("    {:>8}: {:>3}  {}", label, count, bar);
    }

    // --- Entity Count by Type Chart ---
    let type_labels: Vec<&str> = label_sorted.iter().map(|(l, _)| l.as_str()).collect();
    let type_values: Vec<f64> = label_sorted.iter().map(|(_, c)| **c as f64).collect();
    show_chart(&bar_chart(
        "Entity Counts by Type",
        &type_labels,
        "Count",
        &type_values,
        "#3b82f6",
    ));

    println!();
    println!("  Entities by rule:");
    let mut rule_sorted: Vec<(&String, &usize)> = rule_counts.iter().collect();
    rule_sorted.sort_by(|a, b| b.1.cmp(a.1));
    for (rule, count) in &rule_sorted {
        let bar: String = "#".repeat(**count);
        println!("    {:>16}: {:>3}  {}", rule, count, bar);
    }

    // --- Entity Count by Rule Chart ---
    let rule_labels: Vec<&str> = rule_sorted.iter().map(|(r, _)| r.as_str()).collect();
    let rule_values: Vec<f64> = rule_sorted.iter().map(|(_, c)| **c as f64).collect();
    show_chart(&bar_chart(
        "Entity Counts by Rule",
        &rule_labels,
        "Count",
        &rule_values,
        "#f59e0b",
    ));

    println!();
    println!("  Key observations:");
    println!("    - Gazetteers provide high precision but miss unknown entities");
    println!("    - Pattern rules (title, suffix) catch novel entities but produce false positives");
    println!("    - Lowercased text defeats capitalization-based rules entirely");
    println!("    - Ambiguous words (May, Apple, Jordan) require context, not patterns");
    println!();
    println!("  Rule-based NER is transparent and fast,");
    println!("  but it is brittle — every new edge case requires a new rule.");
}
```

---

## Key Takeaways

- **NER is a boundary + labeling task.** Finding where an entity starts and ends is just as important as determining its type. "World Health Organization" is one entity, not three.
- **Rules exploit surface patterns** — capitalization, known word lists, contextual prepositions, and suffixes. They are transparent and require no training data, but they break on ambiguous words, missing capitalization, and novel entities.
- **Gazetteers are precise but incomplete.** A list of 10,000 cities will catch "Mumbai" but miss "Koramangala" (a neighborhood). Lists can never be exhaustive for a living language.
- **Ambiguity is the fundamental limit of rules.** "May" (month vs. verb), "Apple" (company vs. fruit), "Jordan" (person vs. country) — resolving these requires context that no fixed pattern can capture. This motivates statistical approaches.
