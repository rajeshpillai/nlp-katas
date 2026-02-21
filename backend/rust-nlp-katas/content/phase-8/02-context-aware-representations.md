# Compare Context-Aware vs Context-Free Representations

> Phase 8 — Context & Sequence Modeling | Kata 8.2

---

## Concept & Intuition

### What problem are we solving?

In every representation we have built so far -- BoW, TF-IDF, even n-grams -- each word gets **one fixed representation** regardless of how it is used. The word "bank" always maps to the same vector component, whether the sentence is "I deposited money at the bank" or "I sat on the river bank." This is called a **context-free** representation: the vector for a word type is the same no matter where that word token appears.

But human language is deeply **polysemous** -- most common words have multiple meanings, and we resolve which meaning is intended by looking at the surrounding words (the **context**). "Bank" next to "money" and "deposited" means a financial institution. "Bank" next to "river" and "sat" means a slope of land. A representation that cannot distinguish these is fundamentally limited.

This kata builds two kinds of representations side by side: a context-free approach (one vector per word, regardless of surroundings) and a simple context-sensitive approach (a vector that changes based on neighboring words). By comparing them, we see exactly why context matters and why the field moved toward attention-based models.

### Why earlier methods fail

**BoW and TF-IDF assign one weight per word type across the entire vocabulary.** The word "bank" gets a single TF-IDF score in a given document. Even if "bank" appears three times with three different meanings, TF-IDF sees it as one feature. It cannot represent the fact that "bank" in sentence 5 means something different from "bank" in sentence 12.

**Static word embeddings (like early Word2Vec)** assign one dense vector per word type, learned from co-occurrence statistics. The vector for "bank" is a single point in space -- a weighted average of all the contexts "bank" ever appeared in. This means financial-bank and river-bank get mashed into one compromise vector that is not quite right for either meaning.

**The core limitation is treating words as types rather than tokens.** A word *type* is an entry in the dictionary (e.g., "bank"). A word *token* is a specific use of that word in a specific sentence. Context-free representations collapse all tokens of a type into one representation. Context-aware representations give each token its own representation based on its surroundings.

### Mental models

- **Context-free = dictionary definition.** A dictionary lists every meaning of "bank" but cannot tell you which one applies in a given sentence. You need to read the sentence to know.
- **Context-aware = reading comprehension.** When you read "I fished by the bank," your brain activates the river-bank meaning and suppresses the financial-bank meaning. The context window around the word resolves ambiguity in real time.
- **Window-based context averaging** is a simple approximation: represent each word token as the average of its own vector and its neighbors' vectors. Nearby words "color" the representation. This is crude but demonstrates the principle that attention mechanisms later refine.

### Visual explanations

```
CONTEXT-FREE vs CONTEXT-AWARE REPRESENTATIONS
==============================================

Context-free (one vector per word type):

  "bank" -> [0.3, 0.7, 0.1, 0.5]   (same vector everywhere)

  "I deposited money at the bank"    -> bank = [0.3, 0.7, 0.1, 0.5]
  "I sat on the river bank"          -> bank = [0.3, 0.7, 0.1, 0.5]
  "I need to bank on your support"   -> bank = [0.3, 0.7, 0.1, 0.5]

  All three uses get the SAME representation.
  The model cannot tell which meaning is intended.


Context-aware (different vector per word token):

  "I deposited money at the bank"
       context: [deposited, money, at, the]
       bank -> [0.8, 0.9, 0.1, 0.2]   (financial meaning)

  "I sat on the river bank"
       context: [on, the, river]
       bank -> [0.1, 0.2, 0.9, 0.8]   (geographical meaning)

  "I need to bank on your support"
       context: [need, to, on, your]
       bank -> [0.4, 0.3, 0.5, 0.6]   (reliance meaning)

  Each use gets a DIFFERENT representation based on neighbors.


HOW WINDOWED CONTEXT AVERAGING WORKS
=====================================

  Sentence: "I deposited money at the bank yesterday"
  Window size: 2 (look 2 words left and 2 words right)

  To represent "bank" (position 5):
    Window = [at, the, bank, yesterday]
              pos 3  pos 4  pos 5  pos 6

    context_vector("bank") = average of vectors for:
      vec("at") + vec("the") + vec("bank") + vec("yesterday")
      ------------------------------------------------
                           4

  This is crude but effective: "deposited", "money" nearby
  pull the "bank" vector toward financial meanings.
  "river", "sat" nearby would pull it toward geographical meanings.


WHY THIS MOTIVATES ATTENTION
==============================

  Window averaging treats ALL neighbors equally.
  But not all neighbors are equally relevant:

  "The old man who worked at the bank for thirty years retired"

  For "bank": "worked" and "years" are more relevant than "the" or "for"

  Attention mechanisms learn WHICH neighbors to pay attention to.
  They assign different weights to different context words.

  Windowed average: equal weight to all neighbors (crude)
  Attention:        learned weight per neighbor   (powerful)
```

---

## Hands-on Exploration

1. List five English words that have at least three distinct meanings (e.g., "bat", "spring", "crane", "light", "match"). For each word, write three sentences using different meanings. Then ask: what context words would a system need to see to disambiguate each meaning? How large a window would be required in each case?

2. Consider the word "cold" in these sentences: "I caught a cold," "The cold wind blew," "She gave me a cold stare." If you average the vectors of all neighboring words in a window of size 2, sketch out (in words) what each averaged vector would "look like" -- what topics or domains would it lean toward? Would this be enough to distinguish the three meanings?

3. Think about a failure case for windowed averaging: "The bank that I visited last Tuesday after my dentist appointment was closed." Here "bank" is far from the word "visited," and close to "appointment" and "closed." Would a window of size 2 pick up enough signal to know this is a financial bank? What about a window of size 5? What does this tell you about the limitations of fixed-size windows?

---

## Live Code

```rust
use std::collections::HashMap;

fn main() {
    // --- Context-Aware vs Context-Free Representations ---
    // Only Rust stdlib -- no external crates

    fn tokenize(text: &str) -> Vec<String> {
        let mut result = Vec::new();
        let mut current = String::new();
        for ch in text.to_lowercase().chars() {
            if ch.is_alphabetic() { current.push(ch); }
            else if !current.is_empty() { result.push(current.clone()); current.clear(); }
        }
        if !current.is_empty() { result.push(current); }
        result
    }

    fn dot_product(v1: &[f64], v2: &[f64]) -> f64 { v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum() }
    fn magnitude(v: &[f64]) -> f64 { v.iter().map(|x| x * x).sum::<f64>().sqrt() }
    fn cosine_similarity(v1: &[f64], v2: &[f64]) -> f64 {
        let d = dot_product(v1, v2); let m1 = magnitude(v1); let m2 = magnitude(v2);
        if m1 == 0.0 || m2 == 0.0 { 0.0 } else { d / (m1 * m2) }
    }
    fn vector_add(v1: &[f64], v2: &[f64]) -> Vec<f64> { v1.iter().zip(v2.iter()).map(|(a, b)| a + b).collect() }
    fn vector_scale(v: &[f64], s: f64) -> Vec<f64> { v.iter().map(|x| x * s).collect() }

    // ============================================================
    // Build a small word vector space using co-occurrence
    // ============================================================

    let corpus: Vec<&str> = vec![
        "I deposited money at the bank this morning",
        "The bank approved my loan application today",
        "She withdrew cash from the bank account",
        "The bank charged interest on the loan",
        "My savings at the bank grew over time",
        "The bank manager reviewed the financial records",
        "I sat on the river bank and watched the water",
        "The river bank was covered with tall grass",
        "We picnicked on the bank of the stream",
        "Fish jumped near the muddy bank of the river",
        "The bank of the lake was rocky and steep",
        "Wildflowers grew along the bank of the creek",
        "Money and finance are important topics",
        "The loan officer reviewed the application carefully",
        "The river flowed through the green valley",
        "Water rushed over rocks in the stream",
    ];

    println!("{}", "=".repeat(65));
    println!("CORPUS");
    println!("{}", "=".repeat(65));
    for (i, sent) in corpus.iter().enumerate() {
        println!("  [{:2}] {}", i, sent);
    }
    println!();

    let tokenized_corpus: Vec<Vec<String>> = corpus.iter().map(|s| tokenize(s)).collect();
    let all_tokens: Vec<String> = tokenized_corpus.iter().flat_map(|s| s.iter().cloned()).collect();
    let vocab: Vec<String> = {
        let set: std::collections::BTreeSet<String> = all_tokens.iter().cloned().collect();
        set.into_iter().collect()
    };
    let word_to_idx: HashMap<String, usize> = vocab.iter().enumerate().map(|(i, w)| (w.clone(), i)).collect();
    let vocab_size = vocab.len();
    println!("Vocabulary size: {} words", vocab_size);
    println!();

    // ============================================================
    // Build co-occurrence vectors (context-free word representations)
    // ============================================================

    let window: usize = 3;
    let mut co_occurrence: HashMap<String, Vec<f64>> = HashMap::new();
    for w in &vocab { co_occurrence.insert(w.clone(), vec![0.0; vocab_size]); }

    for sent_tokens in &tokenized_corpus {
        for (i, word) in sent_tokens.iter().enumerate() {
            let start = if i >= window { i - window } else { 0 };
            let end = (i + window + 1).min(sent_tokens.len());
            for j in start..end {
                if j != i {
                    let neighbor = &sent_tokens[j];
                    co_occurrence.get_mut(word).unwrap()[word_to_idx[neighbor]] += 1.0;
                }
            }
        }
    }

    println!("{}", "=".repeat(65));
    println!("PART 1: CONTEXT-FREE REPRESENTATION OF 'BANK'");
    println!("{}", "=".repeat(65));
    println!();
    println!("Context-free means: ONE vector for 'bank', no matter the sentence.");
    println!();

    let bank_vec = &co_occurrence["bank"];
    let mut bank_cooc: Vec<(&str, f64)> = vocab.iter()
        .enumerate().filter(|(i, _)| bank_vec[*i] > 0.0)
        .map(|(i, w)| (w.as_str(), bank_vec[i])).collect();
    bank_cooc.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("Top words co-occurring with 'bank':");
    for (word, count) in bank_cooc.iter().take(15) {
        let bar = "#".repeat(*count as usize);
        println!("  {:>12}: {:>2}  {}", word, *count as usize, bar);
    }

    println!();
    println!("Notice: both financial words (money, loan, deposited) and");
    println!("nature words (river, water, grass) appear. The single vector");
    println!("for 'bank' is a MIX of both meanings -- good for neither.");
    println!();

    // ============================================================
    println!("{}", "=".repeat(65));
    println!("PART 2: CONTEXT-FREE SIMILARITY (THE PROBLEM)");
    println!("{}", "=".repeat(65));
    println!();

    let financial_words: Vec<&str> = vec!["money", "loan", "savings", "interest", "financial"]
        .into_iter().filter(|w| co_occurrence.contains_key(*w)).collect();
    let nature_words: Vec<&str> = vec!["river", "water", "grass", "stream", "rocky"]
        .into_iter().filter(|w| co_occurrence.contains_key(*w)).collect();

    println!("Financial domain words: {:?}", financial_words);
    println!("Nature domain words:    {:?}", nature_words);
    println!();
    println!("Cosine similarity of 'bank' with each word (context-free):");
    println!();

    for (label, words) in &[("Financial", &financial_words), ("Nature", &nature_words)] {
        let mut sims: Vec<(&str, f64)> = Vec::new();
        for w in *words {
            let sim = cosine_similarity(&co_occurrence["bank"], &co_occurrence[*w]);
            sims.push((w, sim));
            println!("  bank <-> {:>12}: {:.4}", w, sim);
        }
        let avg_sim: f64 = sims.iter().map(|(_, s)| s).sum::<f64>() / sims.len() as f64;
        println!("  {:>20}: {:.4}", format!("Average {}", label), avg_sim);
        println!();
    }

    println!("The context-free 'bank' vector is somewhat similar to BOTH domains.");
    println!("It cannot commit to either meaning because it must represent all of them.");
    println!();

    // ============================================================
    // Build context-AWARE representations
    // ============================================================

    println!("{}", "=".repeat(65));
    println!("PART 3: CONTEXT-AWARE REPRESENTATION OF 'BANK'");
    println!("{}", "=".repeat(65));
    println!();
    println!("Context-aware means: average 'bank' with its neighbors in each sentence.");
    println!("Each occurrence of 'bank' gets a DIFFERENT vector.");
    println!();

    let ctx_window: usize = 2;

    let mut bank_sentences: Vec<(usize, Vec<String>, usize)> = Vec::new();
    for (idx, sent_tokens) in tokenized_corpus.iter().enumerate() {
        if let Some(bank_pos) = sent_tokens.iter().position(|t| t == "bank") {
            bank_sentences.push((idx, sent_tokens.clone(), bank_pos));
        }
    }

    let mut context_vectors: Vec<(usize, Vec<String>, Vec<f64>)> = Vec::new();

    for (sent_idx, sent_tokens, bank_pos) in &bank_sentences {
        let start = if *bank_pos >= ctx_window { bank_pos - ctx_window } else { 0 };
        let end = (bank_pos + ctx_window + 1).min(sent_tokens.len());
        let context_words: Vec<String> = (start..end).map(|j| sent_tokens[j].clone()).collect();

        let mut avg_vec = vec![0.0; vocab_size];
        for cw in &context_words {
            avg_vec = vector_add(&avg_vec, &co_occurrence[cw]);
        }
        avg_vec = vector_scale(&avg_vec, 1.0 / context_words.len() as f64);

        println!("  Sentence [{:2}]: \"{}\"", sent_idx, corpus[*sent_idx]);
        println!("    'bank' at position {}", bank_pos);
        println!("    Context window: {:?}", context_words);
        println!();

        context_vectors.push((*sent_idx, context_words, avg_vec));
    }

    // ============================================================
    println!("{}", "=".repeat(65));
    println!("PART 4: CONTEXT-AWARE SIMILARITY (THE IMPROVEMENT)");
    println!("{}", "=".repeat(65));
    println!();

    fn domain_centroid(word_list: &[&str], co_occurrence: &HashMap<String, Vec<f64>>, vocab_size: usize) -> Vec<f64> {
        let mut centroid = vec![0.0; vocab_size];
        let mut count = 0;
        for w in word_list {
            if let Some(vec) = co_occurrence.get(*w) {
                for (i, v) in vec.iter().enumerate() { centroid[i] += v; }
                count += 1;
            }
        }
        if count > 0 { centroid = centroid.iter().map(|x| x / count as f64).collect(); }
        centroid
    }

    let financial_centroid = domain_centroid(&financial_words, &co_occurrence, vocab_size);
    let nature_centroid = domain_centroid(&nature_words, &co_occurrence, vocab_size);

    let cf_fin = cosine_similarity(&co_occurrence["bank"], &financial_centroid);
    let cf_nat = cosine_similarity(&co_occurrence["bank"], &nature_centroid);

    println!("Context-FREE 'bank' (single vector for all uses):");
    println!("  Similarity to financial domain: {:.4}", cf_fin);
    println!("  Similarity to nature domain:    {:.4}", cf_nat);
    println!("  Difference:                     {:.4}", (cf_fin - cf_nat).abs());
    println!("  --> Cannot strongly commit to either meaning.");
    println!();

    println!("Context-AWARE 'bank' (different vector per occurrence):");
    println!();

    let mut all_results: Vec<bool> = Vec::new();

    for (sent_idx, context_words, ctx_vec) in &context_vectors {
        let sim_fin = cosine_similarity(ctx_vec, &financial_centroid);
        let sim_nat = cosine_similarity(ctx_vec, &nature_centroid);

        let is_financial = *sent_idx < 6;
        let expected = if is_financial { "financial" } else { "nature" };
        let predicted = if sim_fin > sim_nat { "financial" } else { "nature" };
        let correct = expected == predicted;
        let marker = if correct { "CORRECT" } else { "WRONG" };

        let sent_preview = if corpus[*sent_idx].len() > 50 {
            format!("{}...", &corpus[*sent_idx][..50])
        } else {
            corpus[*sent_idx].to_string()
        };

        println!("  [{:2}] \"{}\"", sent_idx, sent_preview);
        println!("      Context: {:?}", context_words);
        println!("      Sim to financial: {:.4}  |  Sim to nature: {:.4}", sim_fin, sim_nat);
        println!("      Expected: {:>10}  |  Predicted: {:>10}  [{}]", expected, predicted, marker);
        println!();

        all_results.push(correct);
    }

    let accuracy = all_results.iter().filter(|&&r| r).count();
    println!("Disambiguation accuracy: {}/{} ({:.0}%)", accuracy, all_results.len(), accuracy as f64 / all_results.len() as f64 * 100.0);
    println!();

    // ============================================================
    println!("{}", "=".repeat(65));
    println!("PART 5: QUANTIFYING THE IMPROVEMENT");
    println!("{}", "=".repeat(65));
    println!();

    let financial_sents: Vec<(f64, f64)> = context_vectors.iter()
        .filter(|(idx, _, _)| *idx < 6)
        .map(|(_, _, vec)| (cosine_similarity(vec, &financial_centroid), cosine_similarity(vec, &nature_centroid)))
        .collect();
    let nature_sents: Vec<(f64, f64)> = context_vectors.iter()
        .filter(|(idx, _, _)| *idx >= 6 && *idx < 12)
        .map(|(_, _, vec)| (cosine_similarity(vec, &financial_centroid), cosine_similarity(vec, &nature_centroid)))
        .collect();

    if !financial_sents.is_empty() {
        let avg_fin_sim: f64 = financial_sents.iter().map(|(sf, _)| sf).sum::<f64>() / financial_sents.len() as f64;
        let avg_fin_nat: f64 = financial_sents.iter().map(|(_, sn)| sn).sum::<f64>() / financial_sents.len() as f64;
        println!("Financial sentences (bank = money institution):");
        println!("  Avg similarity to financial domain: {:.4}", avg_fin_sim);
        println!("  Avg similarity to nature domain:    {:.4}", avg_fin_nat);
        println!("  Separation:                         {:+.4}", avg_fin_sim - avg_fin_nat);
        println!();
    }

    if !nature_sents.is_empty() {
        let avg_nat_fin: f64 = nature_sents.iter().map(|(sf, _)| sf).sum::<f64>() / nature_sents.len() as f64;
        let avg_nat_nat: f64 = nature_sents.iter().map(|(_, sn)| sn).sum::<f64>() / nature_sents.len() as f64;
        println!("Nature sentences (bank = river edge):");
        println!("  Avg similarity to financial domain: {:.4}", avg_nat_fin);
        println!("  Avg similarity to nature domain:    {:.4}", avg_nat_nat);
        println!("  Separation:                         {:+.4}", avg_nat_nat - avg_nat_fin);
        println!();
    }

    println!("Context-free: 'bank' is stuck between both meanings.");
    println!("Context-aware: each 'bank' leans toward its intended meaning.");
    println!();
    println!("This simple windowed averaging is crude -- it weights all neighbors");
    println!("equally and uses a fixed window size. Attention mechanisms improve");
    println!("on this by learning WHICH context words matter most for each token.");
    println!();

    // ============================================================
    println!("{}", "=".repeat(65));
    println!("PART 6: MORE POLYSEMOUS WORDS");
    println!("{}", "=".repeat(65));
    println!();

    let polysemy_examples: Vec<(&str, Vec<&str>, &str)> = vec![
        ("bat", vec![
            "The bat flew out of the cave at dusk",
            "He swung the bat and hit the baseball",
            "The cricket bat was made of willow wood",
        ], "animal vs sports equipment"),
        ("spring", vec![
            "Flowers bloom in the spring sunshine",
            "The spring in the mattress broke last night",
            "Water bubbled up from the natural spring",
        ], "season vs coil vs water source"),
        ("light", vec![
            "Turn on the light so I can read",
            "The bag was light enough to carry",
            "We will light the candles for dinner",
        ], "illumination vs weight vs ignite"),
    ];

    for (word, sentences, description) in &polysemy_examples {
        println!("  Word: '{}' ({})", word, description);
        println!();
        for sent in sentences {
            let tokens = tokenize(sent);
            if let Some(pos) = tokens.iter().position(|t| t == *word) {
                let start = if pos >= ctx_window { pos - ctx_window } else { 0 };
                let end = (pos + ctx_window + 1).min(tokens.len());
                let context: Vec<&str> = tokens[start..end].iter().map(|s| s.as_str()).collect();
                println!("    \"{}\"", sent);
                println!("      Context window: {:?}", context);
            } else {
                println!("    \"{}\"", sent);
                println!("      ('{}' not found after tokenization)", word);
            }
        }
        println!();
    }

    println!("Each occurrence of the same word has different neighbors.");
    println!("A context-aware system uses those neighbors to resolve meaning.");
    println!("A context-free system gives all occurrences the same vector.");
    println!();
    println!("This is precisely why the field moved from static word embeddings");
    println!("to contextual models -- and why attention was invented.");
}
```

---

## Key Takeaways

- **Context-free representations assign one vector per word type.** The word "bank" gets the same representation whether it means a financial institution or a river's edge. This forces the vector to be a compromise that is accurate for no single meaning.
- **Context-aware representations give each word token its own vector.** By incorporating information from surrounding words, the same word type can have different representations in different sentences, capturing polysemy and disambiguation.
- **Windowed context averaging is a simple but effective demonstration.** Averaging a word's vector with its neighbors' vectors is crude -- it weights all neighbors equally and ignores long-range context -- but it already improves disambiguation over context-free methods.
- **Attention mechanisms are the principled evolution of this idea.** Instead of averaging all neighbors equally, attention learns which context words are most relevant for each token. This is the key innovation that makes transformers powerful for contextual understanding.
