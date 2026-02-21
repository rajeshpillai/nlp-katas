# Show How Word Order Changes Meaning (BoW Failure Cases)

> Phase 8 — Context & Sequence Modeling | Kata 8.1

---

## Concept & Intuition

### What problem are we solving?

Throughout Phases 2-4, we represented text as Bag of Words and TF-IDF vectors. These representations deliberately **discard word order** -- they treat a document as an unordered collection of words. For many tasks (topic classification, document retrieval), this works surprisingly well. But there is an entire class of problems where **the order of words is the meaning itself**, and BoW/TF-IDF are provably blind to this.

Consider: "The dog bit the man" and "The man bit the dog." These sentences contain exactly the same words in exactly the same frequencies. Their BoW vectors are identical. Their TF-IDF vectors are identical. Their cosine similarity is 1.0 -- perfect similarity. Yet they describe completely different events with completely different consequences.

This is not a rare edge case. Language is full of constructions where swapping, reordering, or rearranging words changes meaning while preserving the exact same vocabulary. Any representation that ignores order will confuse these cases, and no amount of weighting or normalization can fix it -- the information is simply not captured.

### Why earlier methods fail

**Bag of Words** counts how many times each word appears. "Dog bites man" and "Man bites dog" produce the vector [dog:1, bites:1, man:1] in both cases. The representation is literally identical for two different events.

**TF-IDF** refines BoW by weighting words based on their corpus-level importance. But it still builds a vector from word frequencies. If two documents have identical word frequencies, TF-IDF cannot distinguish them. The IDF weights are the same for both because the words are the same.

**Cosine similarity** measures the angle between vectors. If two vectors are identical, the angle is zero and cosine similarity is 1.0. No similarity metric can rescue us here -- the problem is in the representation, not the metric.

The fundamental issue: **BoW and TF-IDF are permutation-invariant**. They produce the same output regardless of how you rearrange the words. Word order is not noise -- it is signal, and these representations throw it away.

### Mental models

- **BoW is like a recipe's ingredient list without instructions.** "Flour, eggs, sugar, butter" could be a cake, a pancake, or a cookie. The ingredients are the same, but the process (order of operations) determines the outcome. BoW gives you the ingredient list and throws away the recipe.
- **Word order encodes grammatical relations.** In "The cat chased the mouse," word order tells us who is doing the chasing and who is being chased. English relies heavily on word order for this (unlike Latin, which uses word endings). BoW destroys these relations.
- **N-grams are a patch, not a cure.** Bigrams (word pairs) capture *some* local order, but the vocabulary grows quadratically, most bigrams are sparse, and long-range dependencies remain invisible.

### Visual explanations

```
THE WORD ORDER PROBLEM
======================

Sentence A: "The dog bit the man"
Sentence B: "The man bit the dog"

BoW vectors (same vocabulary: {bit, dog, man, the}):

  Sentence A: [bit:1, dog:1, man:1, the:2]
  Sentence B: [bit:1, dog:1, man:1, the:2]   <-- IDENTICAL!

  cosine(A, B) = 1.000  (perfect similarity)

  But A means: dog is the biter, man is the victim
       B means: man is the biter, dog is the victim

  These are opposite events with identical representations.


HOW N-GRAMS PARTIALLY FIX THIS
================================

Unigrams (BoW):
  A: {the, dog, bit, the, man}      -> {the:2, dog:1, bit:1, man:1}
  B: {the, man, bit, the, dog}      -> {the:2, dog:1, bit:1, man:1}
  SAME.

Bigrams (consecutive pairs):
  A: {the_dog, dog_bit, bit_the, the_man}
  B: {the_man, man_bit, bit_the, the_dog}
  DIFFERENT! "dog_bit" != "man_bit", "the_dog" != "the_man" (as bigrams)

  Now A and B have different representations.
  But bigrams only see pairs -- they cannot capture longer patterns.


THE VOCABULARY EXPLOSION
========================

  Vocabulary size V = 10,000 words

  Unigrams:  10,000 possible features
  Bigrams:   10,000 x 10,000 = 100,000,000 possible features
  Trigrams:  10,000^3 = 1,000,000,000,000 possible features

  Most of these will never appear -> extreme sparsity
  And we STILL cannot capture dependencies beyond 3 words apart.

  "The man who lives next door bit the dog"
    -> "man" and "bit" are 6 words apart
    -> no trigram connects them
    -> we still do not know WHO did the biting
```

---

## Hands-on Exploration

1. Write five pairs of sentences where the same words appear in different orders with different meanings. For each pair, verify by hand that their BoW vectors are identical. Then write the bigram sets for each pair and check whether bigrams successfully distinguish them. Are there any pairs where even bigrams fail?

2. Take the sentence "I saw her duck under the table." This sentence is ambiguous even with word order preserved (did she duck, or did I see her pet duck?). Now scramble the words randomly. How many distinct meanings can you construct by reordering the same six words? What does this tell you about how much information word order carries?

3. Consider a document classification task: sorting movie reviews into positive and negative. Come up with examples where BoW would misclassify a review because of word order. For instance, "This movie is not good" vs "This good movie is not..." Then consider: would bigrams fix all of these cases? What about negation that spans several words, like "I would not say this movie is anywhere close to good"?

---

## Live Code

```rust
use std::collections::HashMap;

fn main() {
    // --- Word Order Matters: BoW Failure Cases and N-gram Partial Fixes ---
    // Only Rust stdlib -- no external crates

    fn tokenize(text: &str) -> Vec<String> {
        let mut result = Vec::new();
        let mut current = String::new();
        for ch in text.to_lowercase().chars() {
            if ch.is_alphabetic() {
                current.push(ch);
            } else if !current.is_empty() {
                result.push(current.clone());
                current.clear();
            }
        }
        if !current.is_empty() { result.push(current); }
        result
    }

    fn build_bow(tokens: &[String]) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for t in tokens { *counts.entry(t.clone()).or_insert(0) += 1; }
        counts
    }

    fn bow_vector(tokens: &[String], vocab: &[String]) -> Vec<f64> {
        let counts = build_bow(tokens);
        vocab.iter().map(|w| *counts.get(w).unwrap_or(&0) as f64).collect()
    }

    fn cosine_similarity(v1: &[f64], v2: &[f64]) -> f64 {
        let dot: f64 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
        let mag1: f64 = v1.iter().map(|a| a * a).sum::<f64>().sqrt();
        let mag2: f64 = v2.iter().map(|b| b * b).sum::<f64>().sqrt();
        if mag1 == 0.0 || mag2 == 0.0 { return 0.0; }
        dot / (mag1 * mag2)
    }

    fn get_ngrams(tokens: &[String], n: usize) -> Vec<String> {
        if tokens.len() < n { return Vec::new(); }
        (0..=(tokens.len() - n)).map(|i| tokens[i..i + n].join("_")).collect()
    }

    fn ngram_vector(ngrams: &[String], vocab: &[String]) -> Vec<f64> {
        let mut counts = HashMap::new();
        for ng in ngrams { *counts.entry(ng.clone()).or_insert(0) += 1; }
        vocab.iter().map(|ng| *counts.get(ng).unwrap_or(&0) as f64).collect()
    }

    // ============================================================
    println!("{}", "=".repeat(65));
    println!("PART 1: SENTENCES WITH SAME WORDS, DIFFERENT MEANINGS");
    println!("{}", "=".repeat(65));

    let pairs: Vec<(&str, &str)> = vec![
        ("The dog bit the man", "The man bit the dog"),
        ("She gave him the book", "He gave her the book"),
        ("The teacher failed the student", "The student failed the teacher"),
        ("Only I love you", "I love only you"),
        ("The cat chased the mouse", "The mouse chased the cat"),
    ];

    let mut pair_labels: Vec<String> = Vec::new();
    let mut bow_sims: Vec<f64> = Vec::new();
    for (i, (sent_a, sent_b)) in pairs.iter().enumerate() {
        let tokens_a = tokenize(sent_a);
        let tokens_b = tokenize(sent_b);
        let bow_a = build_bow(&tokens_a);
        let bow_b = build_bow(&tokens_b);

        let mut vocab_set: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for t in &tokens_a { vocab_set.insert(t.clone()); }
        for t in &tokens_b { vocab_set.insert(t.clone()); }
        let vocab: Vec<String> = vocab_set.into_iter().collect();

        let vec_a = bow_vector(&tokens_a, &vocab);
        let vec_b = bow_vector(&tokens_b, &vocab);
        let sim = cosine_similarity(&vec_a, &vec_b);

        pair_labels.push(format!("Pair {}", i + 1));
        bow_sims.push((sim * 10000.0).round() / 10000.0);

        println!("\nPair {}:", i + 1);
        println!("  A: \"{}\"", sent_a);
        println!("  B: \"{}\"", sent_b);
        println!("  BoW A: {:?}", bow_a);
        println!("  BoW B: {:?}", bow_b);
        println!("  Vectors identical: {}", vec_a == vec_b);
        println!("  Cosine similarity: {:.4}", sim);
        if sim > 0.9999 {
            println!("  --> BoW CANNOT distinguish these sentences!");
        }
    }
    println!();

    // ============================================================
    println!("{}", "=".repeat(65));
    println!("PART 2: N-GRAMS AS A PARTIAL FIX");
    println!("{}", "=".repeat(65));

    let sent_a = "The dog bit the man";
    let sent_b = "The man bit the dog";
    let tokens_a = tokenize(sent_a);
    let tokens_b = tokenize(sent_b);

    let mut ngram_labels: Vec<String> = Vec::new();
    let mut ngram_sims: Vec<f64> = Vec::new();
    for n in 1..=3 {
        let label = match n { 1 => "Unigrams (BoW)", 2 => "Bigrams", _ => "Trigrams" };
        let ngrams_a = get_ngrams(&tokens_a, n);
        let ngrams_b = get_ngrams(&tokens_b, n);

        let mut ng_vocab_set: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for ng in &ngrams_a { ng_vocab_set.insert(ng.clone()); }
        for ng in &ngrams_b { ng_vocab_set.insert(ng.clone()); }
        let ng_vocab: Vec<String> = ng_vocab_set.into_iter().collect();

        let vec_a = ngram_vector(&ngrams_a, &ng_vocab);
        let vec_b = ngram_vector(&ngrams_b, &ng_vocab);
        let sim = cosine_similarity(&vec_a, &vec_b);

        ngram_labels.push(label.to_string());
        ngram_sims.push((sim * 10000.0).round() / 10000.0);

        let shared: Vec<&String> = ngrams_a.iter().filter(|ng| ngrams_b.contains(ng)).collect();
        let only_a: Vec<&String> = ngrams_a.iter().filter(|ng| !ngrams_b.contains(ng)).collect();
        let only_b: Vec<&String> = ngrams_b.iter().filter(|ng| !ngrams_a.contains(ng)).collect();

        println!("\n  {}:", label);
        println!("    A: {:?}", ngrams_a);
        println!("    B: {:?}", ngrams_b);
        println!("    Shared: {:?}", shared);
        println!("    Only in A: {:?}", only_a);
        println!("    Only in B: {:?}", only_b);
        println!("    Cosine similarity: {:.4}", sim);
        if sim > 0.9999 {
            println!("    --> Still identical! {} cannot help here.", label);
        } else {
            println!("    --> Different! {} capture some word order.", label);
        }
    }
    println!();

    // --- Visualization ---
    {
        let labels: Vec<&str> = ngram_labels.iter().map(|s| s.as_str()).collect();
        show_chart(&bar_chart(
            "BoW vs N-gram Similarity: 'The dog bit the man' vs 'The man bit the dog'",
            &labels, "Cosine Similarity", &ngram_sims, "#3b82f6",
        ));
    }
    {
        let labels: Vec<&str> = pair_labels.iter().map(|s| s.as_str()).collect();
        show_chart(&bar_chart(
            "BoW Similarity Across Word-Order Pairs (All ~1.0)",
            &labels, "BoW Cosine Similarity", &bow_sims, "#ef4444",
        ));
    }

    // ============================================================
    println!("{}", "=".repeat(65));
    println!("PART 3: QUANTIFYING INFORMATION LOSS");
    println!("{}", "=".repeat(65));
    println!();
    println!("How much meaning does word order carry?");
    println!("Let's take one sentence and count distinct meanings from reorderings.");
    println!();

    let base_tokens: Vec<&str> = vec!["only", "she", "told", "him", "that", "she", "loved", "him"];
    let sentence: String = base_tokens.join(" ");
    println!("Original: \"{}\"", sentence);
    println!();

    let reorderings: Vec<(Vec<&str>, &str)> = vec![
        (vec!["only", "she", "told", "him", "that", "she", "loved", "him"], "Only SHE told him (nobody else told him)"),
        (vec!["she", "only", "told", "him", "that", "she", "loved", "him"], "She ONLY told him (she didn't show it)"),
        (vec!["she", "told", "only", "him", "that", "she", "loved", "him"], "She told ONLY him (nobody else was told)"),
        (vec!["she", "told", "him", "only", "that", "she", "loved", "him"], "She told him only THAT (nothing else)"),
        (vec!["she", "told", "him", "that", "she", "loved", "only", "him"], "She loved ONLY him (nobody else)"),
        (vec!["she", "told", "him", "that", "only", "she", "loved", "him"], "Only SHE loved him (nobody else did)"),
    ];

    let base_bow = build_bow(&base_tokens.iter().map(|s| s.to_string()).collect::<Vec<_>>());
    println!("BoW for ALL of these: {:?}", base_bow);
    println!();

    for (tokens, meaning) in &reorderings {
        let this_bow = build_bow(&tokens.iter().map(|s| s.to_string()).collect::<Vec<_>>());
        let same = this_bow == base_bow;
        println!("  \"{}\"", tokens.join(" "));
        println!("    Meaning: {}", meaning);
        println!("    Same BoW: {}", same);
        println!();
    }

    println!("  6 different meanings, ALL with identical BoW vectors.");
    println!("  Word order carries enormous semantic information that BoW discards.");
    println!();

    // ============================================================
    println!("{}", "=".repeat(65));
    println!("PART 4: THE VOCABULARY EXPLOSION PROBLEM WITH N-GRAMS");
    println!("{}", "=".repeat(65));
    println!();

    let vocab_sizes: Vec<u64> = vec![100, 500, 1000, 5000, 10000, 50000];
    println!("  {:>12}  {:>12}  {:>14}  {:>16}", "Vocab Size", "Unigrams", "Bigrams", "Trigrams");
    println!("  {}  {}  {}  {}", "-".repeat(12), "-".repeat(12), "-".repeat(14), "-".repeat(16));

    for v in &vocab_sizes {
        let uni = *v;
        let bi = v * v;
        let tri = v * v * v;
        println!("  {:>12}  {:>12}  {:>14}  {:>16}", uni, uni, bi, tri);
    }

    println!();
    println!("  As n-gram order increases, possible features grow EXPONENTIALLY.");
    println!("  Most n-grams will never appear in any document -> extreme sparsity.");
    println!("  And n-grams still cannot capture dependencies beyond n words apart.");
    println!();

    // ============================================================
    println!("{}", "=".repeat(65));
    println!("PART 5: LONG-RANGE DEPENDENCIES DEFEAT N-GRAMS");
    println!("{}", "=".repeat(65));
    println!();

    let examples: Vec<(&str, &str, &str, usize, &str)> = vec![
        ("The cat sat on the mat", "cat", "sat", 1, "Subject and verb are adjacent"),
        ("The cat that I saw yesterday at the park sat on the mat", "cat", "sat", 8, "Subject and verb separated by a relative clause"),
        ("The cat which my neighbor who lives next door adopted last year sat on the mat", "cat", "sat", 11, "Subject and verb separated by nested relative clauses"),
    ];

    println!("How far apart can related words be?\n");

    for (sent, word1, word2, _distance, description) in &examples {
        let tokens = tokenize(sent);
        let idx1 = tokens.iter().position(|t| t == word1).unwrap_or(0);
        let idx2 = tokens.iter().position(|t| t == word2).unwrap_or(0);
        let actual_dist = if idx2 > idx1 { idx2 - idx1 } else { 0 };
        let needed_n = actual_dist + 1;

        println!("  \"{}\"", sent);
        println!("    '{}' at position {}, '{}' at position {}", word1, idx1, word2, idx2);
        println!("    Distance: {} words apart", actual_dist);
        println!("    N-gram needed to capture this: {}-gram", needed_n);
        println!("    Note: {}", description);

        let bigrams = get_ngrams(&tokens, 2);
        let trigrams = get_ngrams(&tokens, 3);
        let has_pair_bigram = bigrams.iter().any(|bg| bg.contains(word1) && bg.contains(word2));
        let has_pair_trigram = trigrams.iter().any(|tg| tg.contains(word1) && tg.contains(word2));
        println!("    Bigrams capture '{}'-'{}' link: {}", word1, word2, has_pair_bigram);
        println!("    Trigrams capture '{}'-'{}' link: {}", word1, word2, has_pair_trigram);
        println!();
    }

    println!("  As sentences get more complex, important relationships span");
    println!("  more words. No fixed n-gram size can handle all cases.");
    println!("  This is why we need models that process SEQUENCES, not bags.");
    println!();

    // ============================================================
    println!("{}", "=".repeat(65));
    println!("PART 6: NEGATION -- BOW'S WORST ENEMY");
    println!("{}", "=".repeat(65));
    println!();

    let sentiment_pairs: Vec<(&str, &str)> = vec![
        ("This movie is good", "This movie is not good"),
        ("I would recommend this film", "I would not recommend this film"),
        ("The acting was excellent and the plot was never boring", "The acting was never excellent and the plot was boring"),
    ];

    for (sent_a, sent_b) in &sentiment_pairs {
        let tokens_a = tokenize(sent_a);
        let tokens_b = tokenize(sent_b);
        let mut vocab_set: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for t in &tokens_a { vocab_set.insert(t.clone()); }
        for t in &tokens_b { vocab_set.insert(t.clone()); }
        let vocab: Vec<String> = vocab_set.into_iter().collect();
        let vec_a = bow_vector(&tokens_a, &vocab);
        let vec_b = bow_vector(&tokens_b, &vocab);
        let sim = cosine_similarity(&vec_a, &vec_b);

        println!("  A: \"{}\"", sent_a);
        println!("  B: \"{}\"", sent_b);
        println!("  Cosine similarity: {:.4}", sim);

        let set_a: std::collections::HashSet<&String> = tokens_a.iter().collect();
        let set_b: std::collections::HashSet<&String> = tokens_b.iter().collect();
        let only_a: Vec<&&String> = set_a.difference(&set_b).collect();
        let only_b: Vec<&&String> = set_b.difference(&set_a).collect();
        if !only_a.is_empty() { println!("  Only in A: {:?}", only_a); }
        if !only_b.is_empty() { println!("  Only in B: {:?}", only_b); }
        if only_a.is_empty() && only_b.is_empty() {
            println!("  --> Same words, different order: BoW is completely blind!");
        }
        println!();
    }

    println!("  Negation words like 'not' and 'never' flip the meaning of");
    println!("  entire phrases. BoW either ignores them (same words) or treats");
    println!("  'not' as just another word with a small IDF weight.");
    println!("  Understanding negation requires understanding SEQUENCE.");
}
```

---

## Key Takeaways

- **Word order is meaning, not noise.** Sentences with identical words in different orders can describe completely opposite events. BoW and TF-IDF are mathematically incapable of distinguishing them because they are permutation-invariant representations.
- **N-grams are a partial fix with exponential cost.** Bigrams and trigrams capture local word order and can distinguish some reorderings, but the vocabulary grows exponentially (V^n), most n-grams are vanishingly sparse, and long-range dependencies remain invisible.
- **Long-range dependencies are everywhere in natural language.** Subjects and verbs, pronouns and their referents, negation and the words being negated -- these relationships can span many words, far beyond what any practical n-gram can capture.
- **Context changes meaning, and models must account for it.** The failure of orderless representations motivates sequence models -- architectures that read text in order and maintain a running understanding of what has been said so far.
