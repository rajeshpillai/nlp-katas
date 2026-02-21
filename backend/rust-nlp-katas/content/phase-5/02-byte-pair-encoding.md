# Implement BPE from Scratch

> Phase 5 — Tokenization (Deep Dive) | Kata 5.2

---

## Concept & Intuition

### What problem are we solving?

In Kata 5.1, we saw that subword tokenization balances vocabulary size, sequence length, and OOV handling. But our rule-based subword tokenizer relied on a hand-crafted list of English affixes. This does not scale to other languages, domains, or the messy reality of real text.

**Byte Pair Encoding (BPE)** solves this by **learning** the subword vocabulary directly from a corpus. It starts with the smallest possible units (characters) and iteratively merges the most frequent pair of adjacent tokens into a new token. After enough merges, the vocabulary contains a mix of characters, common subwords, and frequent whole words — all discovered automatically from the data.

BPE was originally a data compression algorithm (Gage, 1994) and was adapted for NLP by Sennrich et al. (2016). It is now the backbone of tokenization in GPT, LLaMA, and most modern language models.

### Why earlier methods fail

The rule-based subword approach from Kata 5.1 has three critical problems:

1. **Language dependence**: English affixes like "un-", "-ing", "-tion" do not apply to Chinese, Arabic, or Finnish. A universal tokenizer cannot rely on linguistic rules for any single language.
2. **Domain dependence**: Medical text has subwords like "cardio-", "-itis", "-ectomy". Code has subwords like "get", "set", "http". No hand-crafted list covers all domains.
3. **Optimality**: Which affixes should be in the list? How long should they be? There is no principled way to decide. BPE answers this with a data-driven algorithm that finds the optimal vocabulary for a given corpus.

### Mental models

- **BPE as compression**: Imagine you are compressing text by finding the most common two-character sequence and replacing it with a single new symbol. Then finding the next most common pair (including the new symbol), and so on. After many rounds, common words are single symbols, rare words are broken into pieces, and characters are the fallback.
- **BPE as bottom-up tree building**: Start with individual characters. The most frequent pair forms a branch. Then the most frequent pair of branches forms a bigger branch. Eventually common words are single leaves while rare words are decomposed into subtrees.
- **The merge table as a recipe**: BPE produces a ranked list of merge operations. To tokenize new text, you apply these merges in order. This is deterministic and reproducible.

### Visual explanations

```
BPE ALGORITHM STEP BY STEP
============================

Corpus: "low lower lowest"

Step 0 — Start with characters (add end-of-word marker </w>):
  "l o w </w>"      (from "low")
  "l o w e r </w>"  (from "lower")
  "l o w e s t </w>" (from "lowest")

Step 1 — Count all adjacent pairs:
  (l, o)   = 3    <-- most frequent!
  (o, w)   = 3    <-- tied!
  (w, </w>) = 1
  (w, e)   = 2
  (e, r)   = 1
  (r, </w>) = 1
  (e, s)   = 1
  (s, t)   = 1
  (t, </w>) = 1

Step 2 — Merge most frequent pair: (l, o) -> "lo"
  "lo w </w>"
  "lo w e r </w>"
  "lo w e s t </w>"
  Merge table: [(l, o) -> lo]

Step 3 — Recount pairs and merge (lo, w) -> "low":
  "low </w>"
  "low e r </w>"
  "low e s t </w>"
  Merge table: [(l, o) -> lo, (lo, w) -> low]

Step 4 — Merge (low, e) -> "lowe":
  "low </w>"
  "lowe r </w>"
  "lowe s t </w>"
  Merge table: [..., (low, e) -> lowe]

...and so on until desired vocabulary size is reached.


HOW BPE ENCODES UNSEEN WORDS
==============================

Learned merges (from training):
  1. (l, o) -> lo
  2. (lo, w) -> low
  3. (low, e) -> lowe
  4. (lowe, r) -> lower
  5. (e, s) -> es
  6. (es, t) -> est

New word: "lowest"
  Start:   l o w e s t
  Apply 1: lo w e s t
  Apply 2: low e s t
  Apply 3: lowe s t
  Apply 5: lowe st    (no, "st" not a merge — try others)
           lowe s t   (no applicable merge left)
  Result:  ["lowe", "s", "t"]

  The model has never seen "lowest" as a whole word,
  but it can represent it as ["lowe", "s", "t"] — pieces it knows!

New word: "slower"
  Start:   s l o w e r
  Apply 1: s lo w e r
  Apply 2: s low e r
  Apply 3: s lowe r
  Apply 4: s lower
  Result:  ["s", "lower"]

  "slower" is decomposed into "s" + "lower" — meaningful pieces!


MERGE TABLE GROWTH
===================

  Merges   Vocab contains
  ------   --------------------------------------------------------
     0     {a, b, c, ..., z, </w>}              (just characters)
     5     {a, b, ..., z, </w>, th, he, in, er, an}
    20     {... + the, ing, tion, and, ...}      (common subwords)
   100     {... + the</w>, and</w>, that, ...}   (common words appear)
  1000     {... + most common words as single tokens}
 30000     typical GPT-2 vocabulary size
```

---

## Hands-on Exploration

1. Take the mini corpus: "ab ab ab bc bc abc". Write out all characters with end-of-word markers. Count all adjacent pairs by hand. Perform 3 merge operations. After each merge, recount pairs and verify the new most frequent pair. How does the vocabulary grow with each merge?

2. After training BPE on the corpus "low lower lowest", try encoding the word "newest". Walk through the merge table step by step. Which merges apply? Which characters remain unmerged? Compare this to encoding "lower" (a word that was in the training corpus). What is the difference in how many tokens each produces?

3. Consider the effect of corpus size on BPE. If you train BPE on 100 documents about cooking, the merges will favor food-related subwords. If you then tokenize a legal document, many legal terms will be broken into tiny pieces. What does this tell you about the relationship between BPE training data and downstream performance? Why do modern models train their BPE tokenizer on massive, diverse corpora?

---

## Live Code

```rust
use std::collections::HashMap;

fn main() {
    // --- Byte Pair Encoding (BPE) from Scratch ---
    // Complete implementation using only Rust standard library

    /// Tokenize corpus into words and count frequencies.
    fn get_word_freqs(corpus: &[&str]) -> HashMap<String, usize> {
        let mut word_freqs: HashMap<String, usize> = HashMap::new();
        for sentence in corpus {
            for word in sentence.to_lowercase().split_whitespace() {
                *word_freqs.entry(word.to_string()).or_insert(0) += 1;
            }
        }
        word_freqs
    }

    /// Convert a word string to a tuple of character symbols with end marker.
    fn word_to_symbols(word: &str) -> Vec<String> {
        let mut syms: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        syms.push("</w>".to_string());
        syms
    }

    /// Count frequencies of all adjacent symbol pairs across the vocabulary.
    fn get_pair_freqs(word_freqs: &HashMap<String, usize>, word_symbols: &HashMap<String, Vec<String>>) -> HashMap<(String, String), usize> {
        let mut pair_freqs: HashMap<(String, String), usize> = HashMap::new();
        for (word, freq) in word_freqs {
            if let Some(symbols) = word_symbols.get(word) {
                for i in 0..symbols.len().saturating_sub(1) {
                    let pair = (symbols[i].clone(), symbols[i + 1].clone());
                    *pair_freqs.entry(pair).or_insert(0) += freq;
                }
            }
        }
        pair_freqs
    }

    /// Apply a merge operation: replace all occurrences of pair with new_symbol.
    fn merge_pair(word_symbols: &HashMap<String, Vec<String>>, pair: &(String, String), new_symbol: &str) -> HashMap<String, Vec<String>> {
        let mut new_word_symbols = HashMap::new();
        for (word, symbols) in word_symbols {
            let mut new_syms = Vec::new();
            let mut i = 0;
            while i < symbols.len() {
                if i < symbols.len() - 1 && symbols[i] == pair.0 && symbols[i + 1] == pair.1 {
                    new_syms.push(new_symbol.to_string());
                    i += 2;
                } else {
                    new_syms.push(symbols[i].clone());
                    i += 1;
                }
            }
            new_word_symbols.insert(word.clone(), new_syms);
        }
        new_word_symbols
    }

    /// Train BPE: learn merge operations from corpus.
    fn train_bpe(corpus: &[&str], num_merges: usize) -> (Vec<(String, String)>, std::collections::BTreeSet<String>, HashMap<String, Vec<String>>) {
        let word_freqs = get_word_freqs(corpus);

        let mut word_symbols: HashMap<String, Vec<String>> = HashMap::new();
        for word in word_freqs.keys() {
            word_symbols.insert(word.clone(), word_to_symbols(word));
        }

        let mut vocab = std::collections::BTreeSet::new();
        for symbols in word_symbols.values() {
            for s in symbols { vocab.insert(s.clone()); }
        }

        let mut merges: Vec<(String, String)> = Vec::new();

        println!("{}", "=".repeat(60));
        println!("BPE TRAINING");
        println!("{}", "=".repeat(60));

        let mut wf_sorted: Vec<(&String, &usize)> = word_freqs.iter().collect();
        wf_sorted.sort_by_key(|(w, _)| w.clone());
        println!("\nCorpus word frequencies: {:?}", wf_sorted.iter().map(|(w, c)| format!("{}: {}", w, c)).collect::<Vec<_>>().join(", "));
        let mut vocab_sorted: Vec<&String> = vocab.iter().collect();
        vocab_sorted.sort();
        println!("Initial vocabulary ({}): {:?}", vocab.len(), vocab_sorted);

        println!("\nInitial word representations:");
        let mut ws_sorted: Vec<(&String, &Vec<String>)> = word_symbols.iter().collect();
        ws_sorted.sort_by_key(|(w, _)| w.clone());
        for (word, symbols) in &ws_sorted {
            println!("  {:>12} -> {}", word, symbols.join(" "));
        }

        println!("\nPerforming {} merges...\n", num_merges);

        for step in 0..num_merges {
            let pair_freqs = get_pair_freqs(&word_freqs, &word_symbols);

            if pair_freqs.is_empty() {
                println!("  No more pairs to merge at step {}.", step + 1);
                break;
            }

            // Find the most frequent pair (deterministic: sort by freq desc, then pair lexically)
            let mut pairs_vec: Vec<(&(String, String), &usize)> = pair_freqs.iter().collect();
            pairs_vec.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
            let best_pair = pairs_vec[0].0.clone();
            let best_freq = *pairs_vec[0].1;

            let new_symbol = format!("{}{}", best_pair.0, best_pair.1);

            word_symbols = merge_pair(&word_symbols, &best_pair, &new_symbol);
            vocab.insert(new_symbol.clone());
            merges.push(best_pair.clone());

            println!("  Merge {}: ({:>6}, {:<6}) -> {:<10} (frequency: {})",
                step + 1, best_pair.0, best_pair.1, new_symbol, best_freq);

            let mut ws_sorted: Vec<(&String, &Vec<String>)> = word_symbols.iter().collect();
            ws_sorted.sort_by_key(|(w, _)| w.clone());
            for (word, symbols) in &ws_sorted {
                println!("           {:>12} -> {}", word, symbols.join(" "));
            }
            println!();
        }

        let vocab_sorted2: Vec<&String> = vocab.iter().collect();
        println!("Final vocabulary ({} symbols): {:?}", vocab.len(), vocab_sorted2);
        (merges, vocab, word_symbols)
    }

    /// Encode a single word using learned BPE merges.
    fn encode_word(word: &str, merges: &[(String, String)]) -> Vec<String> {
        let mut symbols: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        symbols.push("</w>".to_string());

        for pair in merges {
            let mut i = 0;
            while i < symbols.len().saturating_sub(1) {
                if symbols[i] == pair.0 && symbols[i + 1] == pair.1 {
                    let merged = format!("{}{}", pair.0, pair.1);
                    symbols.splice(i..=i + 1, std::iter::once(merged));
                } else {
                    i += 1;
                }
            }
        }

        symbols
    }

    /// Encode full text using learned BPE merges.
    fn encode_text(text: &str, merges: &[(String, String)]) -> Vec<String> {
        let mut tokens = Vec::new();
        for word in text.to_lowercase().split_whitespace() {
            tokens.extend(encode_word(word, merges));
        }
        tokens
    }

    // --- Train BPE on a small corpus ---
    let corpus: Vec<&str> = vec![
        "low low low low low",
        "lower lower lower",
        "lowest lowest",
        "newer newer newer newer",
        "newest newest",
        "wider wider wider",
        "widest widest",
    ];

    let (merges, vocab, _final_symbols) = train_bpe(&corpus, 15);

    // --- Show the merge table ---
    println!("\n{}", "=".repeat(60));
    println!("LEARNED MERGE TABLE (recipe for tokenization)");
    println!("{}", "=".repeat(60));
    for (i, (a, b)) in merges.iter().enumerate() {
        println!("  {:>3}. {} + {} -> {}", i + 1, a, b, format!("{}{}", a, b));
    }

    // --- Encode training words ---
    println!("\n{}", "=".repeat(60));
    println!("ENCODING TRAINING WORDS");
    println!("{}", "=".repeat(60));

    let training_words = ["low", "lower", "lowest", "newer", "newest", "wider", "widest"];
    for word in &training_words {
        let tokens = encode_word(word, &merges);
        println!("  {:>10} -> {:?}", word, tokens);
    }

    println!("\n{}", "=".repeat(60));
    println!("ENCODING UNSEEN WORDS");
    println!("{}", "=".repeat(60));
    println!("(Words NOT in training corpus)\n");

    let unseen_words = ["new", "slow", "slower", "slowest", "highest", "tower", "flower"];
    for word in &unseen_words {
        let tokens = encode_word(word, &merges);
        let in_vocab = tokens.iter().all(|t| vocab.contains(t));
        let status = if in_vocab { "all known" } else { "has unknown pieces" };
        println!("  {:>10} -> {:?}  ({})", word, tokens, status);
    }

    // --- Full text encoding ---
    println!("\n{}", "=".repeat(60));
    println!("ENCODING FULL SENTENCES");
    println!("{}", "=".repeat(60));

    let test_sentences = [
        "lower is lower",
        "the newest tower",
        "slower and widest",
    ];

    for sent in &test_sentences {
        let tokens = encode_text(sent, &merges);
        println!("  \"{}\"", sent);
        println!("    -> {:?}", tokens);
        println!("    ({} tokens)", tokens.len());
        println!();
    }

    // --- Statistics ---
    println!("{}", "=".repeat(60));
    println!("BPE STATISTICS");
    println!("{}", "=".repeat(60));

    let initial_chars: std::collections::BTreeSet<char> = training_words.iter()
        .flat_map(|w| w.chars()).collect();
    let initial_vocab_size = initial_chars.len() + 1; // +1 for </w>

    println!("  Number of merges learned:  {}", merges.len());
    println!("  Final vocabulary size:     {}", vocab.len());
    println!("  Initial vocab (chars only): {}", initial_vocab_size);
    println!("  Compression ratio:         {} symbols cover all words", vocab.len());
    println!();
    println!("  KEY INSIGHT: BPE learns a vocabulary that balances");
    println!("  whole words (for common terms) with character pieces");
    println!("  (for rare terms). It needs no linguistic rules —");
    println!("  only corpus statistics.");
}
```

---

## Key Takeaways

- **BPE learns subword units from data, not from rules.** It starts with characters and iteratively merges the most frequent adjacent pair. This process discovers common prefixes, suffixes, and roots automatically, making it language-agnostic and domain-adaptive.
- **The merge table is a deterministic recipe.** Once trained, the ordered list of merge operations is applied to any new text in the same sequence. This guarantees reproducible tokenization and allows the model to share a consistent vocabulary between training and inference.
- **BPE handles unseen words by decomposition.** A word never encountered during training is broken into smaller known pieces — subwords or individual characters. The model always has some representation, unlike word-level tokenizers that produce a meaningless [UNK] token.
- **Vocabulary size is a hyperparameter you control.** The number of BPE merges directly determines vocabulary size. More merges produce larger vocabularies with more whole-word tokens (shorter sequences). Fewer merges produce smaller vocabularies with more character-level decomposition (longer sequences). Modern models typically use 30,000-50,000 merges.
