# Tokenize Text Using Word, Character, and Subword Methods

> Phase 5 — Tokenization (Deep Dive) | Kata 5.1

---

## Concept & Intuition

### What problem are we solving?

Every NLP model operates on **tokens**, not raw text. A token is the atomic unit that a model sees — the smallest piece of text it can reason about. The question "how do we split text into tokens?" is not a trivial preprocessing step. It is a **design decision that determines what the model can and cannot learn**.

There are three fundamental approaches to tokenization, each with radically different tradeoffs:

1. **Word tokenization**: Split on whitespace and punctuation. Each word is a token.
2. **Character tokenization**: Each character is a token.
3. **Subword tokenization**: Split words into meaningful pieces (prefixes, suffixes, roots).

The choice determines vocabulary size, sequence length, and how the model handles words it has never seen before (the **out-of-vocabulary** or OOV problem). There is no universally correct answer — only tradeoffs.

### Why earlier methods fail

In Phases 2-4, we casually split text with `.split()` and called each word a token. This worked for small, controlled corpora. But it breaks badly in the real world:

**Word tokenization** creates enormous vocabularies. English has over 170,000 words in current use, plus technical jargon, names, misspellings, compound words, and neologisms. A word-level tokenizer trained on news articles will encounter "COVID-19", "defunding", or "ChatGPT" and have no idea what to do. These become **OOV tokens** — black holes in the representation where information is lost.

**Character tokenization** solves OOV completely (every word is made of characters), but the sequences become extremely long. The sentence "The cat sat" is 3 word tokens but 11 character tokens (including spaces). Long sequences are harder to model — the model must learn to compose characters into meaningful units, a task that is computationally expensive and semantically fragile.

Neither extreme is ideal. We need something in between.

### Mental models

- **Word tokenization** = using whole bricks to build. Fast to assemble, but you need a warehouse of every possible brick shape. If you encounter a shape you do not have, you are stuck.
- **Character tokenization** = using individual grains of sand. You can build anything, but it takes forever and you lose the structure of bricks entirely.
- **Subword tokenization** = using a small set of modular pieces (prefixes, roots, suffixes). You can assemble most words from parts, handle new words by decomposition, and keep sequences manageable.

### Visual explanations

```
THE TOKENIZATION SPECTRUM
=========================

Input: "unhappiness is unbelievable"

WORD TOKENIZATION (split on spaces):
  ["unhappiness", "is", "unbelievable"]
  Tokens: 3          Vocabulary needed: 3 full words
  OOV risk: HIGH     If "unbelievable" not in vocab, it's lost

CHARACTER TOKENIZATION (each char):
  ["u","n","h","a","p","p","i","n","e","s","s"," ","i","s"," ",
   "u","n","b","e","l","i","e","v","a","b","l","e"]
  Tokens: 27         Vocabulary needed: ~26 letters + space
  OOV risk: ZERO     But sequence is 9x longer!

SUBWORD TOKENIZATION (meaningful pieces):
  ["un", "happi", "ness", "is", "un", "believ", "able"]
  Tokens: 7          Vocabulary needed: ~6 subword units
  OOV risk: LOW      "un" + new root + "able" still works!


THE TRADEOFF:
=============

  Vocab Size    |  Word >>> Subword >>> Character
  Seq Length    |  Word <<< Subword <<< Character
  OOV Risk      |  Word >>> Subword >>> Character
  Meaning/Token |  Word >>> Subword >>> Character

  <<<  means "less/shorter"
  >>>  means "more/longer"


OOV (OUT-OF-VOCABULARY) PROBLEM:
=================================

  Training vocabulary: {"the", "cat", "sat", "on", "mat", "dog", "ran"}

  New input: "The cat sprinted across the carpet"

  Word tokenizer sees:
    "the"      -> KNOWN
    "cat"      -> KNOWN
    "sprinted" -> ??? OOV -> replaced with [UNK]
    "across"   -> ??? OOV -> replaced with [UNK]
    "the"      -> KNOWN
    "carpet"   -> ??? OOV -> replaced with [UNK]

  Result: ["the", "cat", [UNK], [UNK], "the", [UNK]]
  Three words lost! The meaning is destroyed.

  Subword tokenizer sees:
    "sprinted" -> ["sprint", "ed"]     (known root + known suffix)
    "across"   -> ["a", "cross"]       (known pieces)
    "carpet"   -> ["car", "pet"]       (might be wrong meaning, but not lost)

  At least the model sees SOMETHING for every input.
```

---

## Hands-on Exploration

1. Take the sentence: "The researchers investigated unforgettable preprocessing techniques." Tokenize it by hand using word, character, and subword methods. For subword, try splitting words into recognizable prefixes, roots, and suffixes (e.g., "un-forgett-able", "pre-process-ing"). Count the number of tokens each method produces. Which method gives the model the most semantic information per token?

2. Imagine you trained a word-level tokenizer on a corpus of English news articles from 2019. Now you feed it a 2024 article containing "deepfake", "defederation", "multimodal", and "GPT-4o". How many of these are likely OOV? Now consider: if you had used a subword tokenizer with pieces like "deep", "fake", "de", "multi", "modal" — how many could be reconstructed? What does this tell you about the longevity of different tokenization strategies?

3. Consider a multilingual scenario with text in English, German, and Japanese. German has compound words like "Handschuhschneeballwerfer" (a single word meaning "someone who throws snowballs with gloves on"). Japanese has no spaces between words. Which tokenization method handles each language best? Why does character-level tokenization become more attractive for languages with complex morphology or no word boundaries?

---

## Live Code

```rust
use std::collections::BTreeSet;

fn main() {
    // --- Tokenization Methods: Word, Character, and Subword ---
    // Comparing three fundamental approaches to splitting text into tokens
    // Uses ONLY Rust standard library

    /// Split text into word tokens on whitespace and punctuation boundaries.
    fn word_tokenize(text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut current = String::new();
        let lower = text.to_lowercase();
        let chars: Vec<char> = lower.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            let ch = chars[i];
            if ch.is_alphabetic() {
                current.push(ch);
                // Handle contractions like "don't"
                if i + 1 < chars.len() && chars[i + 1] == '\'' && i + 2 < chars.len() && chars[i + 2].is_alphabetic() {
                    current.push(chars[i + 1]);
                    current.push(chars[i + 2]);
                    i += 2;
                    // collect remaining alpha
                    while i + 1 < chars.len() && chars[i + 1].is_alphabetic() {
                        i += 1;
                        current.push(chars[i]);
                    }
                }
            } else if ch.is_ascii_digit() {
                if !current.is_empty() {
                    tokens.push(current.clone());
                    current.clear();
                }
                let mut num = String::new();
                num.push(ch);
                while i + 1 < chars.len() && chars[i + 1].is_ascii_digit() {
                    i += 1;
                    num.push(chars[i]);
                }
                tokens.push(num);
            } else if !ch.is_whitespace() && !ch.is_alphabetic() && !ch.is_ascii_digit() {
                if !current.is_empty() {
                    tokens.push(current.clone());
                    current.clear();
                }
                // punctuation as token
                tokens.push(ch.to_string());
            } else {
                if !current.is_empty() {
                    tokens.push(current.clone());
                    current.clear();
                }
            }
            i += 1;
        }
        if !current.is_empty() {
            tokens.push(current);
        }
        tokens
    }

    /// Split text into individual characters (including spaces).
    fn char_tokenize(text: &str) -> Vec<String> {
        text.to_lowercase().chars().map(|c| c.to_string()).collect()
    }

    /// Split words into subword pieces using common English affixes.
    fn subword_tokenize(text: &str) -> Vec<String> {
        let prefixes = ["under", "multi", "over", "mis", "dis", "pre", "out", "non", "un", "re", "de"];
        let suffixes = ["tion", "sion", "ment", "ness", "able", "ible", "less", "ize", "ise",
                        "ful", "ous", "ive", "ing", "ity", "est", "ed", "er", "ly", "al"];

        // Extract words (alpha only) from lowercased text
        let mut words = Vec::new();
        let mut current = String::new();
        for ch in text.to_lowercase().chars() {
            if ch.is_alphabetic() {
                current.push(ch);
            } else if ch.is_ascii_digit() {
                if !current.is_empty() {
                    words.push(current.clone());
                    current.clear();
                }
                let mut num = String::new();
                num.push(ch);
                words.push(num);
            } else if !ch.is_whitespace() && ch != '\'' {
                if !current.is_empty() {
                    words.push(current.clone());
                    current.clear();
                }
                words.push(ch.to_string());
            } else {
                if !current.is_empty() {
                    words.push(current.clone());
                    current.clear();
                }
            }
        }
        if !current.is_empty() { words.push(current); }

        let mut tokens = Vec::new();
        for word in &words {
            if word.len() <= 3 {
                tokens.push(word.clone());
                continue;
            }

            let mut pieces = Vec::new();
            let mut remaining = word.as_str();

            // Try to strip a prefix
            let mut prefix_sorted: Vec<&&str> = prefixes.iter().collect();
            prefix_sorted.sort_by(|a, b| b.len().cmp(&a.len()));
            for prefix in &prefix_sorted {
                if remaining.starts_with(**prefix) && remaining.len() > prefix.len() + 2 {
                    pieces.push(prefix.to_string());
                    remaining = &remaining[prefix.len()..];
                    break;
                }
            }

            // Try to strip a suffix
            let mut suffix_sorted: Vec<&&str> = suffixes.iter().collect();
            suffix_sorted.sort_by(|a, b| b.len().cmp(&a.len()));
            let mut suffix_found: Option<String> = None;
            for suffix in &suffix_sorted {
                if remaining.ends_with(**suffix) && remaining.len() > suffix.len() + 1 {
                    suffix_found = Some(suffix.to_string());
                    remaining = &remaining[..remaining.len() - suffix.len()];
                    break;
                }
            }

            pieces.push(remaining.to_string());
            if let Some(s) = suffix_found {
                pieces.push(s);
            }

            tokens.extend(pieces);
        }

        tokens
    }

    // --- Corpus ---
    let sentences = vec![
        "The researchers investigated unforgettable preprocessing techniques",
        "Unhappiness is unbelievable and overwhelming",
        "The disconnected microprocessor was malfunctioning repeatedly",
        "She enjoys running swimming and cycling outdoors",
        "Antidisestablishmentarianism is a remarkably long word",
    ];

    println!("{}", "=".repeat(70));
    println!("TOKENIZATION METHODS COMPARISON");
    println!("{}", "=".repeat(70));

    for sent in &sentences {
        println!("\nSentence: \"{}\"", sent);
        println!("{}", "-".repeat(60));

        let w_tokens = word_tokenize(sent);
        let c_tokens = char_tokenize(sent);
        let s_tokens = subword_tokenize(sent);

        println!("  Word tokens     ({:>2}): {:?}", w_tokens.len(), w_tokens);
        let c_preview: String = c_tokens.iter().take(40).map(|s| s.as_str()).collect::<Vec<_>>().join("");
        println!("  Character tokens({:>2}): {}...", c_tokens.len(), c_preview);
        println!("  Subword tokens  ({:>2}): {:?}", s_tokens.len(), s_tokens);
    }

    // --- Vocabulary Size Analysis ---
    println!("\n{}", "=".repeat(70));
    println!("VOCABULARY SIZE ANALYSIS");
    println!("{}", "=".repeat(70));

    let mut all_word_tokens = Vec::new();
    let mut all_char_tokens = Vec::new();
    let mut all_subword_tokens = Vec::new();

    for sent in &sentences {
        all_word_tokens.extend(word_tokenize(sent));
        all_char_tokens.extend(char_tokenize(sent));
        all_subword_tokens.extend(subword_tokenize(sent));
    }

    let word_vocab: BTreeSet<&str> = all_word_tokens.iter().map(|s| s.as_str()).collect();
    let char_vocab: BTreeSet<&str> = all_char_tokens.iter().map(|s| s.as_str()).collect();
    let subword_vocab: BTreeSet<&str> = all_subword_tokens.iter().map(|s| s.as_str()).collect();

    let n_sent = sentences.len() as f64;
    println!("\n  {:<15} {:>12} {:>14} {:>14}", "Method", "Vocab Size", "Total Tokens", "Avg Tok/Sent");
    println!("  {}", "-".repeat(57));
    println!("  {:<15} {:>12} {:>14} {:>14.1}", "Word", word_vocab.len(), all_word_tokens.len(), all_word_tokens.len() as f64 / n_sent);
    println!("  {:<15} {:>12} {:>14} {:>14.1}", "Character", char_vocab.len(), all_char_tokens.len(), all_char_tokens.len() as f64 / n_sent);
    println!("  {:<15} {:>12} {:>14} {:>14.1}", "Subword", subword_vocab.len(), all_subword_tokens.len(), all_subword_tokens.len() as f64 / n_sent);

    let mut word_vocab_sorted: Vec<&str> = word_vocab.iter().copied().collect();
    word_vocab_sorted.sort();
    let mut char_vocab_sorted: Vec<&str> = char_vocab.iter().copied().collect();
    char_vocab_sorted.sort();
    let mut subword_vocab_sorted: Vec<&str> = subword_vocab.iter().copied().collect();
    subword_vocab_sorted.sort();

    println!("\n  Word vocabulary:    {:?}", word_vocab_sorted);
    println!("  Char vocabulary:    {:?}", char_vocab_sorted);
    println!("  Subword vocabulary: {:?}", subword_vocab_sorted);

    // --- Vocabulary Size Chart ---
    let methods = ["Word", "Character", "Subword"];
    let vocab_sizes = [word_vocab.len() as f64, char_vocab.len() as f64, subword_vocab.len() as f64];
    let avg_tokens = [
        all_word_tokens.len() as f64 / n_sent,
        all_char_tokens.len() as f64 / n_sent,
        all_subword_tokens.len() as f64 / n_sent,
    ];

    show_chart(&bar_chart(
        "Vocabulary Size by Tokenization Method",
        &methods,
        "Vocabulary Size",
        &vocab_sizes,
        "#3b82f6",
    ));

    let avg_rounded: Vec<f64> = avg_tokens.iter().map(|v| (v * 10.0).round() / 10.0).collect();
    show_chart(&bar_chart(
        "Average Tokens per Sentence by Method",
        &methods,
        "Avg Tokens/Sentence",
        &avg_rounded,
        "#10b981",
    ));

    // --- OOV Demonstration ---
    println!("\n{}", "=".repeat(70));
    println!("OUT-OF-VOCABULARY (OOV) DEMONSTRATION");
    println!("{}", "=".repeat(70));

    let training_vocab_word = word_vocab.clone();
    let training_vocab_subword = subword_vocab.clone();

    let unseen_sentences = vec![
        "The undiscovered preprocessing was disheartening",
        "Rethinking multitasking requires carefulness",
        "Overwhelming discouragement is preventable",
    ];

    println!("\nTraining vocabulary learned from corpus above.");
    println!("Now testing on UNSEEN text:\n");

    for sent in &unseen_sentences {
        println!("  Input: \"{}\"", sent);

        // Word-level OOV
        let w_tokens = word_tokenize(sent);
        let w_oov: Vec<&str> = w_tokens.iter().filter(|t| !training_vocab_word.contains(t.as_str())).map(|s| s.as_str()).collect();
        let w_known: Vec<&str> = w_tokens.iter().filter(|t| training_vocab_word.contains(t.as_str())).map(|s| s.as_str()).collect();
        let w_oov_rate = if w_tokens.is_empty() { 0.0 } else { w_oov.len() as f64 / w_tokens.len() as f64 * 100.0 };

        // Subword-level OOV
        let s_tokens = subword_tokenize(sent);
        let s_oov: Vec<&str> = s_tokens.iter().filter(|t| !training_vocab_subword.contains(t.as_str())).map(|s| s.as_str()).collect();
        let s_known: Vec<&str> = s_tokens.iter().filter(|t| training_vocab_subword.contains(t.as_str())).map(|s| s.as_str()).collect();
        let s_oov_rate = if s_tokens.is_empty() { 0.0 } else { s_oov.len() as f64 / s_tokens.len() as f64 * 100.0 };

        println!("    Word tokens:    {:?}", w_tokens);
        println!("      Known: {:?}", w_known);
        println!("      OOV:   {:?}  ({:.0}% OOV)", w_oov, w_oov_rate);
        println!("    Subword tokens: {:?}", s_tokens);
        println!("      Known: {:?}", s_known);
        println!("      OOV:   {:?}  ({:.0}% OOV)", s_oov, s_oov_rate);
        println!();
    }

    println!("  Character tokenization: 0% OOV (always)");
    println!("  It can represent ANY input, but at the cost of very long sequences.");

    // --- Tradeoff Summary ---
    println!("\n{}", "=".repeat(70));
    println!("TRADEOFF SUMMARY");
    println!("{}", "=".repeat(70));
    println!("
  Property            Word         Subword      Character
  ------------------- ------------ ------------ ------------
  Vocabulary size     LARGE        MEDIUM       TINY (~30)
  Sequence length     SHORT        MEDIUM       VERY LONG
  OOV risk            HIGH         LOW          NONE
  Meaning per token   HIGH         MEDIUM       LOW
  Handles new words?  NO           MOSTLY       ALWAYS
  Handles typos?      NO           PARTIALLY    YES

  KEY INSIGHT: Tokenization defines what a model can and cannot see.
  Word tokenization is blind to morphology and unseen words.
  Character tokenization sees everything but understands nothing.
  Subword tokenization balances visibility with meaning.
");
}
```

---

## Key Takeaways

- **Tokenization is a design decision, not just preprocessing.** The choice of tokenization method determines what information the model receives. Word tokens carry meaning but create OOV gaps. Character tokens cover everything but dilute meaning into long sequences. Subword tokens balance both concerns.
- **The OOV problem is word tokenization's fatal flaw.** Any word not in the training vocabulary becomes an unknown token, destroying information. In domains with evolving language (medicine, technology, social media), word-level tokenizers become obsolete quickly.
- **Character tokenization trades OOV safety for sequence length.** A model processing characters must learn to compose letters into meanings — a much harder task than working with pre-formed words. This makes training slower and requires more model capacity.
- **Subword tokenization is the foundation of modern NLP.** By decomposing words into reusable pieces (prefixes, roots, suffixes), subword methods achieve small vocabularies, manageable sequence lengths, and robust OOV handling. This is why every major language model today uses subword tokenization.
