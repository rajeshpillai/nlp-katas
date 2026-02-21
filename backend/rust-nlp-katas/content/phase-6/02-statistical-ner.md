# Simple ML-Based Named Entity Recognition

> Phase 6 — Named Entity Recognition (NER) | Kata 6.2

---

## Concept & Intuition

### What problem are we solving?

In Kata 6.1, we built NER with handcrafted rules. Rules are transparent but brittle — every new edge case demands a new rule, and ambiguous words like "May" or "Apple" cannot be resolved by patterns alone. The question becomes: can we **learn** to recognize entities from patterns in data, instead of writing each pattern by hand?

Statistical NER shifts the work from **writing rules** to **designing features**. Instead of saying "if a word is capitalized and follows 'Dr.', it is a PERSON," we say "extract the capitalization status, the previous word, the word shape, and the position — then let a scoring system decide." The features capture the same intuitions as rules, but they combine flexibly and can handle novel cases that no single rule anticipated.

This kata builds a feature-based NER system from scratch. We do not use any machine learning library. Instead, we use a simple weighted scoring approach that demonstrates the core idea: **features plus weights equal predictions**.

### Why earlier methods fail

Rule-based NER fails for three fundamental reasons:

1. **Rigid boundaries**: A rule either fires or it does not. There is no notion of "mostly looks like a person name." Statistical approaches assign confidence scores.
2. **No feature interaction**: Rules check conditions independently. But entity recognition often requires combining multiple weak signals — a word is capitalized AND follows a preposition AND is at the end of a sentence. Rules require explicit enumeration of every useful combination.
3. **No generalization**: A rule for "Dr. + Name" does not help with "CEO + Name" unless you add another rule. Feature-based systems generalize because the feature "preceded by a title-like word" covers both cases.

### Mental models

- **Features as lenses**: Each feature is a different way of looking at a token. Capitalization is one lens. Word shape is another. Context words are another. No single lens is sufficient, but together they form a clear picture.
- **Voting by features**: Think of each feature as a voter. "This word is capitalized" votes weakly for PERSON. "The previous word is 'in'" votes for LOCATION. "The word shape is Xxxxx" votes for a named entity of some kind. The entity type with the most weighted votes wins.
- **Context windows**: A token's identity depends on its neighbors. The word "Bank" means different things after "River" vs. after "Central." Feature-based NER captures this by including features from surrounding tokens.

### Visual explanations

```
Sentence: "Dr. Amara Khan works at Infosys in Bangalore."

Token-level features for "Khan":
  +-----------------------------------------------+
  | Feature               | Value                 |
  +-----------------------------------------------+
  | is_capitalized        | True                  |
  | is_all_caps           | False                 |
  | word_shape            | Xxxx                  |
  | word_length           | 4                     |
  | position_in_sentence  | 0.28  (early)         |
  | prev_word             | "Amara"               |
  | prev_is_capitalized   | True                  |
  | prev_prev_word        | "Dr."                 |
  | next_word             | "works"               |
  | next_is_capitalized   | False                 |
  | has_title_prefix      | True  (Dr. nearby)    |
  | in_location_gazetteer | False                 |
  | in_org_gazetteer      | False                 |
  +-----------------------------------------------+

Feature voting for "Khan":
  PERSON:    is_capitalized (+2) + prev_capitalized (+1) + has_title (+3) = +6
  LOCATION:  is_capitalized (+2) + in_loc_gazetteer (0)                  = +2
  ORG:       is_capitalized (+2) + in_org_gazetteer (0)                  = +2
  O (none):  next_is_lowercase (+1)                                      = +1

  Winner: PERSON (score 6)

Comparison with rules (Kata 6.1):
  Rule system: "Dr." + capitalized word -> PERSON  (binary: fires or not)
  Feature system: combines 10+ signals with weights (graded: confidence score)
```

---

## Hands-on Exploration

1. Pick the word "Washington" in three different sentences (a person, a location, a government reference). For each usage, list what features would differ between the three contexts. Which features are most discriminative? Notice that no single feature resolves the ambiguity — it takes a combination.

2. Design a feature set for detecting ORGANIZATION entities. Think about: what word shapes do company names have? What suffixes appear? What words typically come before or after organization names? Write at least 8 features.

3. Take two sentences where rule-based NER from Kata 6.1 failed (e.g., "Apple released a new product" or lowercased social media text). What features could a statistical system use that rules could not? Consider features like "the word after this one is a verb associated with companies" or "this word appears in a tech-news context."

---

## Live Code

```rust
use std::collections::{HashMap, BTreeSet};

fn main() {
    // ============================================================
    // FEATURE-BASED NER (NO EXTERNAL LIBRARIES)
    // ============================================================
    // Instead of writing rules, we extract features for each token
    // and use weighted scoring to predict entity types.
    //
    // This demonstrates the core statistical NER idea:
    //   features + weights = predictions
    // ============================================================

    // --- Gazetteers (reused from Kata 6.1 as features, not rules) ---

    let person_titles: BTreeSet<&str> = ["mr", "mrs", "ms", "dr", "prof", "sir", "captain", "judge", "ceo", "president"]
        .iter().copied().collect();

    let known_orgs: BTreeSet<&str> = [
        "infosys", "google", "microsoft", "amazon", "tesla", "apple",
        "united nations", "world health organization", "nasa", "unesco",
    ].iter().copied().collect();

    let known_locations: BTreeSet<&str> = [
        "mumbai", "delhi", "bangalore", "chennai", "kolkata", "hyderabad",
        "new york", "london", "paris", "tokyo", "berlin", "sydney",
        "california", "texas", "india", "china", "geneva", "washington",
        "singapore", "toronto", "dubai", "jordan",
    ].iter().copied().collect();

    let org_suffixes: BTreeSet<&str> = ["inc", "ltd", "corp", "llc", "co", "foundation", "institute"]
        .iter().copied().collect();

    let person_name_indicators: BTreeSet<&str> = ["said", "told", "announced", "visited", "met", "joined", "presented"]
        .iter().copied().collect();

    let org_context_words: BTreeSet<&str> = ["company", "firm", "corporation", "released", "announced", "acquired", "hired"]
        .iter().copied().collect();

    let location_context_words: BTreeSet<&str> = ["in", "from", "to", "at", "near", "visited", "landed", "moved", "based"]
        .iter().copied().collect();

    let month_names: BTreeSet<&str> = [
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
    ].iter().copied().collect();

    /// Convert a word to its shape pattern.
    fn get_word_shape(word: &str) -> String {
        let mut shape = Vec::new();
        for ch in word.chars() {
            let s = if ch.is_uppercase() { 'X' }
                    else if ch.is_lowercase() { 'x' }
                    else if ch.is_ascii_digit() { 'd' }
                    else { ch };
            if shape.is_empty() || *shape.last().unwrap() != s {
                shape.push(s);
            }
        }
        shape.into_iter().collect()
    }

    /// Split text into word tokens.
    fn tokenize(text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            let ch = chars[i];
            if ch.is_alphanumeric() {
                let mut word = String::new();
                while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '.') {
                    word.push(chars[i]);
                    i += 1;
                }
                tokens.push(word);
            } else if ".,;:!?".contains(ch) {
                tokens.push(ch.to_string());
                i += 1;
            } else {
                i += 1;
            }
        }
        tokens
    }

    /// Extract features for the token at the given index.
    fn extract_features(
        tokens: &[String], index: usize,
        person_titles: &BTreeSet<&str>,
        known_orgs: &BTreeSet<&str>,
        known_locations: &BTreeSet<&str>,
        org_suffixes: &BTreeSet<&str>,
        location_context_words: &BTreeSet<&str>,
        org_context_words: &BTreeSet<&str>,
        person_name_indicators: &BTreeSet<&str>,
        month_names: &BTreeSet<&str>,
    ) -> HashMap<String, f64> {
        let word = &tokens[index];
        let word_lower = word.to_lowercase().trim_end_matches('.').to_string();
        let n = tokens.len();

        let prev_word = if index > 0 { tokens[index - 1].clone() } else { "<START>".to_string() };
        let prev2_word = if index > 1 { tokens[index - 2].clone() } else { "<START>".to_string() };
        let next_word = if index < n - 1 { tokens[index + 1].clone() } else { "<END>".to_string() };

        let mut features = HashMap::new();

        // Word-level features
        let is_cap = word.chars().next().map(|c| c.is_uppercase()).unwrap_or(false);
        features.insert("is_capitalized".to_string(), if is_cap { 1.0 } else { 0.0 });
        features.insert("is_all_caps".to_string(), if word.chars().all(|c| c.is_uppercase()) && word.len() > 1 { 1.0 } else { 0.0 });
        features.insert("is_all_lower".to_string(), if word.chars().all(|c| c.is_lowercase()) { 1.0 } else { 0.0 });
        features.insert("word_length".to_string(), word.len() as f64);
        features.insert("position_ratio".to_string(), (index as f64 / n.max(1) as f64 * 100.0).round() / 100.0);
        features.insert("is_first_token".to_string(), if index == 0 { 1.0 } else { 0.0 });

        // Gazetteer features
        features.insert("in_person_titles".to_string(), if person_titles.contains(word_lower.as_str()) { 1.0 } else { 0.0 });
        features.insert("in_org_gazetteer".to_string(), if known_orgs.contains(word_lower.as_str()) { 1.0 } else { 0.0 });
        features.insert("in_loc_gazetteer".to_string(), if known_locations.contains(word_lower.as_str()) { 1.0 } else { 0.0 });
        features.insert("has_org_suffix".to_string(), if org_suffixes.contains(word_lower.as_str()) { 1.0 } else { 0.0 });
        features.insert("is_month".to_string(), if month_names.contains(word_lower.as_str()) { 1.0 } else { 0.0 });

        // Context features (previous word)
        let prev_lower = prev_word.to_lowercase().trim_end_matches('.').to_string();
        features.insert("prev_is_capitalized".to_string(),
            if prev_word != "<START>" && prev_word.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) { 1.0 } else { 0.0 });
        features.insert("prev_is_title".to_string(), if person_titles.contains(prev_lower.as_str()) { 1.0 } else { 0.0 });
        features.insert("prev_is_loc_prep".to_string(), if location_context_words.contains(prev_lower.as_str()) { 1.0 } else { 0.0 });
        features.insert("prev_is_org_context".to_string(), if org_context_words.contains(prev_lower.as_str()) { 1.0 } else { 0.0 });

        // Context features (prev-prev word)
        let prev2_lower = prev2_word.to_lowercase().trim_end_matches('.').to_string();
        features.insert("prev2_is_title".to_string(), if person_titles.contains(prev2_lower.as_str()) { 1.0 } else { 0.0 });

        // Context features (next word)
        let next_lower = next_word.to_lowercase().trim_end_matches('.').to_string();
        features.insert("next_is_capitalized".to_string(),
            if next_word != "<END>" && next_word.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) { 1.0 } else { 0.0 });
        features.insert("next_is_person_verb".to_string(), if person_name_indicators.contains(next_lower.as_str()) { 1.0 } else { 0.0 });
        features.insert("next_is_org_context".to_string(), if org_context_words.contains(next_lower.as_str()) { 1.0 } else { 0.0 });
        features.insert("next_is_loc_prep".to_string(), if location_context_words.contains(next_lower.as_str()) { 1.0 } else { 0.0 });

        features
    }

    // --- Weighted scoring: manually assigned weights ---
    struct WeightTable {
        weights: HashMap<String, HashMap<String, f64>>,
    }

    impl WeightTable {
        fn new() -> Self {
            let mut weights = HashMap::new();

            let mut person = HashMap::new();
            person.insert("is_capitalized".to_string(), 2.0);
            person.insert("is_all_caps".to_string(), -0.5);
            person.insert("is_all_lower".to_string(), -3.0);
            person.insert("prev_is_capitalized".to_string(), 1.5);
            person.insert("prev_is_title".to_string(), 4.0);
            person.insert("prev2_is_title".to_string(), 2.5);
            person.insert("next_is_person_verb".to_string(), 3.0);
            person.insert("next_is_capitalized".to_string(), 1.0);
            person.insert("in_loc_gazetteer".to_string(), -2.0);
            person.insert("in_org_gazetteer".to_string(), -2.0);
            person.insert("is_month".to_string(), -3.0);
            person.insert("is_first_token".to_string(), -0.5);
            weights.insert("PERSON".to_string(), person);

            let mut org = HashMap::new();
            org.insert("is_capitalized".to_string(), 1.5);
            org.insert("is_all_caps".to_string(), 2.0);
            org.insert("is_all_lower".to_string(), -2.0);
            org.insert("in_org_gazetteer".to_string(), 5.0);
            org.insert("has_org_suffix".to_string(), 4.0);
            org.insert("next_is_org_context".to_string(), 3.0);
            org.insert("prev_is_org_context".to_string(), 2.0);
            org.insert("in_loc_gazetteer".to_string(), -1.5);
            org.insert("prev_is_title".to_string(), -1.0);
            org.insert("is_month".to_string(), -3.0);
            weights.insert("ORG".to_string(), org);

            let mut loc = HashMap::new();
            loc.insert("is_capitalized".to_string(), 1.5);
            loc.insert("is_all_lower".to_string(), -2.0);
            loc.insert("in_loc_gazetteer".to_string(), 5.0);
            loc.insert("prev_is_loc_prep".to_string(), 4.0);
            loc.insert("next_is_loc_prep".to_string(), 1.5);
            loc.insert("in_org_gazetteer".to_string(), -1.5);
            loc.insert("prev_is_title".to_string(), -2.0);
            loc.insert("next_is_person_verb".to_string(), -1.5);
            loc.insert("is_month".to_string(), -3.0);
            weights.insert("LOCATION".to_string(), loc);

            let mut date = HashMap::new();
            date.insert("is_month".to_string(), 6.0);
            date.insert("is_capitalized".to_string(), -0.5);
            date.insert("in_loc_gazetteer".to_string(), -2.0);
            date.insert("in_org_gazetteer".to_string(), -2.0);
            date.insert("prev_is_title".to_string(), -3.0);
            weights.insert("DATE".to_string(), date);

            WeightTable { weights }
        }

        fn score(&self, features: &HashMap<String, f64>, entity_type: &str) -> (f64, Vec<(String, f64)>) {
            let empty = HashMap::new();
            let w = self.weights.get(entity_type).unwrap_or(&empty);
            let mut score = 0.0;
            let mut contributing = Vec::new();
            for (feat_name, feat_value) in features {
                if let Some(&weight) = w.get(feat_name) {
                    if *feat_value == 1.0 {
                        // Boolean feature that is true
                        score += weight;
                        contributing.push((feat_name.clone(), weight));
                    } else if *feat_value > 0.0 && feat_name != "is_capitalized" && feat_name != "is_all_caps"
                        && feat_name != "is_all_lower" && feat_name != "is_first_token"
                        && feat_name != "in_person_titles" && feat_name != "in_org_gazetteer"
                        && feat_name != "in_loc_gazetteer" && feat_name != "has_org_suffix"
                        && feat_name != "is_month" && feat_name != "prev_is_capitalized"
                        && feat_name != "prev_is_title" && feat_name != "prev_is_loc_prep"
                        && feat_name != "prev_is_org_context" && feat_name != "prev2_is_title"
                        && feat_name != "next_is_capitalized" && feat_name != "next_is_person_verb"
                        && feat_name != "next_is_org_context" && feat_name != "next_is_loc_prep" {
                        // Numeric feature
                        score += weight * 0.5;
                        contributing.push((feat_name.clone(), weight * 0.5));
                    }
                }
            }
            (score, contributing)
        }
    }

    let weight_table = WeightTable::new();
    let entity_threshold = 3.0;
    let entity_types = ["PERSON", "ORG", "LOCATION", "DATE"];

    struct Prediction {
        token: String,
        label: String,
        score: f64,
        details: Vec<(String, f64)>,
        all_scores: HashMap<String, f64>,
    }

    let predict_entities = |text: &str| -> Vec<Prediction> {
        let tokens = tokenize(text);
        let mut results = Vec::new();

        for (i, token) in tokens.iter().enumerate() {
            if !token.chars().next().map(|c| c.is_alphabetic()).unwrap_or(false) { continue; }

            let features = extract_features(&tokens, i,
                &person_titles, &known_orgs, &known_locations, &org_suffixes,
                &location_context_words, &org_context_words, &person_name_indicators, &month_names);

            let mut scores = HashMap::new();
            let mut details_map = HashMap::new();
            for etype in &entity_types {
                let (score, contrib) = weight_table.score(&features, etype);
                scores.insert(etype.to_string(), score);
                details_map.insert(etype.to_string(), contrib);
            }

            let best_type = entity_types.iter()
                .max_by(|a, b| scores[**a].partial_cmp(&scores[**b]).unwrap())
                .unwrap();
            let best_score = scores[*best_type];

            if best_score >= entity_threshold {
                let rounded_scores: HashMap<String, f64> = scores.iter()
                    .map(|(k, v)| (k.clone(), (*v * 10.0).round() / 10.0))
                    .collect();
                results.push(Prediction {
                    token: token.clone(),
                    label: best_type.to_string(),
                    score: (best_score * 10.0).round() / 10.0,
                    details: details_map.remove(*best_type).unwrap_or_default(),
                    all_scores: rounded_scores,
                });
            }
        }
        results
    };

    let display_predictions = |text: &str, results: &[Prediction], show_scores: bool| {
        println!("  Text: {}", text);
        if results.is_empty() {
            println!("  Entities: (none found)");
            return;
        }
        for r in results {
            let score_info = if show_scores { format!("  score={}", r.score) } else { String::new() };
            println!("    [{:>8}]  \"{}\"{}", r.label, r.token, score_info);
        }
        if show_scores {
            println!("    Score breakdown:");
            for r in results {
                let mut sorted_details = r.details.clone();
                sorted_details.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
                let top: Vec<String> = sorted_details.iter().take(3)
                    .map(|(f, w)| format!("{}={:+.1}", f, w))
                    .collect();
                println!("      \"{}\" -> {}: {}", r.token, r.label, top.join(", "));
            }
        }
    };

    // ============================================================
    // TEST: SENTENCES THAT WORK WELL
    // ============================================================

    let test_sentences = vec![
        "Dr. Amara Khan works at Infosys in Bangalore.",
        "Prof. Raj Mehta presented at NASA in Washington.",
        "Tesla announced a new factory in Berlin on March 15.",
        "Ms. Priya Nair visited Google in California last June.",
        "The CEO joined Microsoft in Tokyo.",
    ];

    println!("{}", "=".repeat(65));
    println!("  FEATURE-BASED NER: PREDICTIONS WITH SCORING");
    println!("{}", "=".repeat(65));
    println!();

    for sent in &test_sentences {
        let results = predict_entities(sent);
        display_predictions(sent, &results, true);
        println!();
    }

    // ============================================================
    // COMPARISON: FEATURE-BASED vs RULE-BASED ON TRICKY CASES
    // ============================================================

    let tricky_sentences: Vec<(&str, &str)> = vec![
        ("May I go to Paris in May?",
         "Rule-based: both 'May' tagged as DATE. Feature-based: first 'May' is sentence-initial + not preceded by prep = lower DATE score."),
        ("Jordan visited Jordan last summer.",
         "Rule-based: cannot distinguish. Feature-based: first 'Jordan' + verb 'visited' = PERSON signal; 'last summer' context for second."),
        ("Apple released a new product.",
         "Rule-based: misses 'Apple' (not in org list as standalone). Feature-based: 'released' is org-context = ORG signal."),
        ("Dr. Smith met Prof. Lee in Geneva.",
         "Both systems handle titles well, but feature-based also scores confidence."),
    ];

    println!("{}", "=".repeat(65));
    println!("  COMPARISON: FEATURE-BASED vs RULE-BASED");
    println!("{}", "=".repeat(65));
    println!();

    for (sent, comparison_note) in &tricky_sentences {
        let results = predict_entities(sent);
        display_predictions(sent, &results, true);
        println!("    COMPARISON: {}", comparison_note);
        println!();
    }

    // ============================================================
    // FEATURE ANALYSIS: WHAT FEATURES MATTER MOST?
    // ============================================================

    println!("{}", "=".repeat(65));
    println!("  FEATURE IMPORTANCE ANALYSIS");
    println!("{}", "=".repeat(65));
    println!();

    let mut feature_fires: HashMap<String, HashMap<String, usize>> = HashMap::new();
    let all_test: Vec<&str> = test_sentences.iter().chain(tricky_sentences.iter().map(|(s, _)| s)).copied().collect();

    for sent in &all_test {
        let tokens = tokenize(sent);
        for (i, token) in tokens.iter().enumerate() {
            if !token.chars().next().map(|c| c.is_alphabetic()).unwrap_or(false) { continue; }
            let features = extract_features(&tokens, i,
                &person_titles, &known_orgs, &known_locations, &org_suffixes,
                &location_context_words, &org_context_words, &person_name_indicators, &month_names);
            for etype in &entity_types {
                let (_, contrib) = weight_table.score(&features, etype);
                for (feat_name, _) in &contrib {
                    *feature_fires.entry(etype.to_string()).or_insert_with(HashMap::new)
                        .entry(feat_name.clone()).or_insert(0) += 1;
                }
            }
        }
    }

    println!("  How often each feature contributes to each entity type:\n");
    for etype in &entity_types {
        if let Some(fires) = feature_fires.get(*etype) {
            let mut sorted_feats: Vec<(&String, &usize)> = fires.iter().collect();
            sorted_feats.sort_by(|a, b| b.1.cmp(a.1));
            println!("  {}:", etype);
            for (feat, count) in sorted_feats.iter().take(5) {
                let bar: String = "#".repeat(**count);
                println!("    {:<25} fires {:>2}x  {}", feat, count, bar);
            }
            println!();
        }
    }

    println!("  Key differences from rule-based NER:");
    println!("    - Features combine multiple weak signals (no single feature decides)");
    println!("    - Context features (prev_word, next_word) help disambiguate");
    println!("    - Scores provide confidence, not just binary yes/no");
    println!("    - The same feature set handles all entity types (rules need separate logic)");
    println!();
    println!("  Limitation of this approach:");
    println!("    - Weights are hand-tuned (a real ML system learns them from data)");
    println!("    - Single-token entities only (multi-token spans need sequence labeling)");
    println!("    - Feature engineering still requires domain knowledge");
    println!("    - A real statistical NER uses BIO tagging and sequence models");
}
```

---

## Key Takeaways

- **Features replace rules.** Instead of writing "if capitalized and after a title, then PERSON," we extract features (is_capitalized, prev_is_title) and let weighted scoring combine them. This is more flexible and generalizes better to unseen text.
- **Context windows are powerful.** Looking at the words before and after a token provides signals that the token alone cannot. "Apple released" (ORG context) vs. "ate an apple" (not an entity) — the verb after the word changes everything.
- **Scores express confidence, not certainty.** A rule fires or it does not. A feature-based system produces a score, which lets you set thresholds, rank alternatives, and understand why a decision was made. This transparency is critical for error analysis.
- **Feature engineering is the bottleneck.** The system is only as good as the features you design. This motivates the move to neural methods (Phase 7+) that learn features automatically from data, instead of requiring human-designed feature templates.
