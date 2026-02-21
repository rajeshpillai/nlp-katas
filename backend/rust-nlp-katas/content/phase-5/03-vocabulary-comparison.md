# Compare Vocabulary Sizes and OOV Handling Across Methods

> Phase 5 — Tokenization (Deep Dive) | Kata 5.3

---

## Concept & Intuition

### What problem are we solving?

We have now seen three tokenization methods: word-level, character-level, and BPE. Each produces different vocabularies, different sequence lengths, and different OOV behavior. But we have only seen these differences on individual examples. This kata asks the quantitative question: **on a real corpus, how do these methods actually compare?**

We will measure four concrete metrics across all three methods:
1. **Vocabulary size** — how many unique tokens does the method need?
2. **Average tokens per document** — how long are the tokenized sequences?
3. **OOV word count** — when we encounter new text, how many tokens are unknown?
4. **Coverage** — what percentage of new text can the tokenizer represent?

These numbers matter because they directly affect model design. A model's embedding layer size equals its vocabulary size. Its computational cost scales with sequence length. Its accuracy depends on how well it handles words it has never seen.

### Why earlier methods fail

Neither extreme is satisfactory:

**Word tokenization** gives small, readable token counts but creates enormous vocabularies. A corpus of 1 million sentences might contain 200,000 unique words. Each word needs its own embedding vector. That is 200,000 rows in the embedding matrix, most of which are seen only once or twice during training. These rare words get poorly trained embeddings — garbage in, garbage out. Worse, any word not in those 200,000 is completely invisible to the model.

**Character tokenization** gives a tiny vocabulary (~30 symbols for English) but produces sequences 5-7 times longer than word tokenization. A 512-token context window that could hold a full paragraph at the word level now holds only a sentence or two at the character level. The model must spend its capacity composing characters into meaning rather than processing meaning directly.

**BPE finds the sweet spot.** With a vocabulary of 30,000-50,000 tokens, it keeps sequences only slightly longer than word-level, while handling OOV gracefully by decomposing unknown words into known subword pieces. But this balance depends on the number of merges — too few and it behaves like character tokenization, too many and it approaches word tokenization.

### Mental models

- **Vocabulary size vs. OOV is a dial, not a switch.** At one extreme (characters), vocabulary is minimal and OOV is zero. At the other (words), vocabulary is huge and OOV is high. BPE lets you turn the dial to any point in between by choosing the number of merges.
- **Sequence length vs. meaning density is the inverse tradeoff.** Shorter sequences pack more meaning per token but risk OOV. Longer sequences never miss a character but dilute meaning.
- **Coverage is the metric that matters most.** A tokenizer that represents 100% of new text with known tokens is strictly better than one with gaps — as long as the sequence lengths remain manageable.

### Visual explanations

```
THE FUNDAMENTAL TRADEOFF
=========================

                   Word         BPE          Character
                   ----         ---          ---------
  Vocab size:     |==========|  |======|     |==|
  Seq length:     |===|         |=====|      |=============|
  OOV rate:       |========|    |==|          |  (zero)
  Meaning/token:  |==========|  |=======|    |==|


QUANTITATIVE EXAMPLE (hypothetical corpus):
=============================================

  Metric                   Word       BPE-5000    BPE-30000   Character
  ----------------------   --------   ---------   ----------  ---------
  Vocabulary size          150,000     5,000       30,000         28
  Avg tokens/document         50        70           55          280
  OOV on new text            12%        1%          0.3%          0%
  Embedding matrix rows   150,000     5,000       30,000         28

  Notice: BPE-30000 is close to word-level in sequence length
          but close to character-level in OOV handling.
          This is why modern models use ~30K-50K BPE merges.


OOV CASCADING FAILURE
======================

  Corpus: news articles (training)
  New input: medical report

  Word tokenizer vocabulary: {"the", "market", "stocks", "economy", ...}

  Medical text: "The patient exhibited tachycardia and dyspnea"

  Word tokens: ["the", [UNK], [UNK], [UNK], "and", [UNK]]
                        ^       ^       ^             ^
                     60% of content words are LOST

  BPE tokens:  ["the", "patient", "exhibit", "ed", "ta", "chy",
                "card", "ia", "and", "dys", "pn", "ea"]
                All pieces are known! Meaning is preserved in fragments.

  Character:   ["t","h","e"," ","p","a","t","i","e","n","t",...]
                100% coverage, but 47 tokens instead of 6.


VOCABULARY SIZE AS A FUNCTION OF BPE MERGES
=============================================

  Merges     Vocab     Behavior
  ------     -----     --------------------------
       0        28     = Character tokenization
     100       128     Mostly characters + common pairs
    1000      1028     Short subwords dominate
    5000      5028     Common words appear as tokens
   30000     30028     Most words are single tokens
  100000    100028     ~ Word tokenization
```

---

## Hands-on Exploration

1. Take a paragraph from a technical domain you know well (medicine, law, programming). Tokenize it mentally using word, character, and subword methods. Count: how many word tokens are domain-specific jargon that a general-purpose word tokenizer would miss? How would BPE handle those terms? Write out the BPE decomposition for 3 jargon words.

2. Consider two BPE tokenizers: one trained with 1,000 merges and another with 50,000 merges. For the sentence "The unprecedented investigation revealed uncharacteristic behavior", predict which tokenizer produces more tokens. Which one is more likely to have "unprecedented" as a single token? Which one will definitely split it? What are the consequences for a language model using each tokenizer?

3. A tokenizer trained on English text is applied to French text. Word-level OOV will be catastrophic (most French words are unseen). Character-level will work fine (same alphabet). What about BPE? It will partially work because many letter combinations are shared (e.g., "tion", "ment", "pre"). But accent marks ("e" vs "e" with accent) might be separate tokens or might be OOV characters. What does this reveal about the hidden assumptions in tokenizer training?

---

## Live Code

```rust
use std::collections::{HashMap, BTreeSet};

fn main() {
    // --- Vocabulary Size and OOV Comparison Across Tokenization Methods ---
    // Comparing word, character, and BPE tokenization on the same corpus
    // Uses ONLY Rust standard library

    // ===== TOKENIZERS =====

    /// Word-level tokenization: split on non-alphanumeric boundaries.
    fn word_tokenize(text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut current = String::new();
        let chars: Vec<char> = text.to_lowercase().chars().collect();
        let mut i = 0;
        while i < chars.len() {
            let ch = chars[i];
            if ch.is_alphabetic() {
                current.push(ch);
                // Handle contractions
                if i + 1 < chars.len() && chars[i + 1] == '\'' && i + 2 < chars.len() && chars[i + 2].is_alphabetic() {
                    current.push(chars[i + 1]);
                    i += 1;
                    while i + 1 < chars.len() && chars[i + 1].is_alphabetic() {
                        i += 1;
                        current.push(chars[i]);
                    }
                }
            } else if ch.is_ascii_digit() {
                if !current.is_empty() { tokens.push(current.clone()); current.clear(); }
                let mut num = String::new();
                num.push(ch);
                while i + 1 < chars.len() && chars[i + 1].is_ascii_digit() {
                    i += 1; num.push(chars[i]);
                }
                tokens.push(num);
            } else {
                if !current.is_empty() { tokens.push(current.clone()); current.clear(); }
            }
            i += 1;
        }
        if !current.is_empty() { tokens.push(current); }
        tokens
    }

    /// Character-level tokenization: each character is a token.
    fn char_tokenize(text: &str) -> Vec<String> {
        text.to_lowercase().chars()
            .filter(|c| c.is_alphabetic() || *c == ' ')
            .map(|c| c.to_string())
            .collect()
    }

    // ===== BPE TOKENIZER =====

    struct BPETokenizer {
        merges: Vec<(String, String)>,
        vocab: BTreeSet<String>,
    }

    impl BPETokenizer {
        fn new() -> Self {
            BPETokenizer { merges: Vec::new(), vocab: BTreeSet::new() }
        }

        fn word_to_symbols(word: &str) -> Vec<String> {
            let mut syms: Vec<String> = word.chars().map(|c| c.to_string()).collect();
            syms.push("</w>".to_string());
            syms
        }

        fn train(&mut self, corpus: &[&str], num_merges: usize) {
            let mut word_freqs: HashMap<String, usize> = HashMap::new();
            for sentence in corpus {
                for word in sentence.to_lowercase().split_whitespace() {
                    let cleaned: String = word.chars().filter(|c| c.is_alphabetic()).collect();
                    if !cleaned.is_empty() {
                        *word_freqs.entry(cleaned).or_insert(0) += 1;
                    }
                }
            }

            let mut word_symbols: HashMap<String, Vec<String>> = HashMap::new();
            for word in word_freqs.keys() {
                word_symbols.insert(word.clone(), Self::word_to_symbols(word));
            }

            self.vocab.clear();
            for symbols in word_symbols.values() {
                for s in symbols { self.vocab.insert(s.clone()); }
            }

            self.merges.clear();
            for _ in 0..num_merges {
                let mut pair_freqs: HashMap<(String, String), usize> = HashMap::new();
                for (word, freq) in &word_freqs {
                    if let Some(symbols) = word_symbols.get(word) {
                        for i in 0..symbols.len().saturating_sub(1) {
                            let pair = (symbols[i].clone(), symbols[i + 1].clone());
                            *pair_freqs.entry(pair).or_insert(0) += freq;
                        }
                    }
                }

                if pair_freqs.is_empty() { break; }

                let mut pairs_vec: Vec<(&(String, String), &usize)> = pair_freqs.iter().collect();
                pairs_vec.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
                let best_pair = pairs_vec[0].0.clone();
                let new_symbol = format!("{}{}", best_pair.0, best_pair.1);

                let mut new_ws = HashMap::new();
                for (word, symbols) in &word_symbols {
                    let mut new_syms = Vec::new();
                    let mut i = 0;
                    while i < symbols.len() {
                        if i < symbols.len() - 1 && symbols[i] == best_pair.0 && symbols[i + 1] == best_pair.1 {
                            new_syms.push(new_symbol.clone());
                            i += 2;
                        } else {
                            new_syms.push(symbols[i].clone());
                            i += 1;
                        }
                    }
                    new_ws.insert(word.clone(), new_syms);
                }

                word_symbols = new_ws;
                self.vocab.insert(new_symbol);
                self.merges.push(best_pair);
            }
        }

        fn encode_word(&self, word: &str) -> Vec<String> {
            let mut symbols: Vec<String> = word.to_lowercase().chars().map(|c| c.to_string()).collect();
            symbols.push("</w>".to_string());
            for pair in &self.merges {
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

        fn tokenize(&self, text: &str) -> Vec<String> {
            let mut tokens = Vec::new();
            for word in text.to_lowercase().split_whitespace() {
                let cleaned: String = word.chars().filter(|c| c.is_alphabetic()).collect();
                if !cleaned.is_empty() {
                    tokens.extend(self.encode_word(&cleaned));
                }
            }
            tokens
        }
    }

    // ===== CORPUS =====

    let training_corpus: Vec<&str> = vec![
        "The researchers discovered that natural language processing requires careful preprocessing",
        "Machine learning models need large amounts of training data to perform well",
        "The preprocessing pipeline includes tokenization normalization and stopword removal",
        "Deep learning has transformed natural language understanding significantly",
        "Text classification and sentiment analysis are common natural language tasks",
        "The training process involves optimizing model parameters on labeled data",
        "Researchers have developed new methods for text representation and analysis",
        "Language models learn statistical patterns from large text corpora",
        "The natural language processing community continues to make significant advances",
        "Understanding text requires both syntactic and semantic analysis of language",
        "Feature extraction transforms raw text into numerical representations for models",
        "The researchers evaluated their model on multiple benchmark datasets",
        "Preprocessing steps can significantly impact downstream task performance",
        "Word embeddings capture semantic relationships between words in vector space",
        "The language model was trained on a diverse corpus of text documents",
    ];

    let test_corpus: Vec<&str> = vec![
        "Biomedical researchers investigated unprecedented pharmacological interactions",
        "The computational linguists developed morphological analyzers for agglutinative languages",
        "Unsupervised pretraining dramatically outperformed traditional feature engineering",
        "Multilingual tokenization presents unique challenges for non-segmented scripts",
        "The neurolinguistic study examined psycholinguistic processing of garden path sentences",
    ];

    println!("{}", "=".repeat(70));
    println!("CORPUS OVERVIEW");
    println!("{}", "=".repeat(70));
    println!("\n  Training corpus: {} sentences", training_corpus.len());
    println!("  Test corpus:     {} sentences (different domain)", test_corpus.len());
    println!("\n  Training sample: \"{}\"", training_corpus[0]);
    println!("  Test sample:     \"{}\"", test_corpus[0]);

    // ===== BUILD TOKENIZERS =====

    let mut word_vocab = BTreeSet::new();
    for sent in &training_corpus {
        for t in word_tokenize(sent) { word_vocab.insert(t); }
    }

    let mut char_vocab = BTreeSet::new();
    for sent in &training_corpus {
        for t in char_tokenize(sent) { char_vocab.insert(t); }
    }

    let mut bpe_small = BPETokenizer::new();
    bpe_small.train(&training_corpus, 30);

    let mut bpe_medium = BPETokenizer::new();
    bpe_medium.train(&training_corpus, 75);

    let mut bpe_large = BPETokenizer::new();
    bpe_large.train(&training_corpus, 150);

    // ===== METRIC COMPUTATION =====

    struct Metrics {
        name: String,
        vocab_size: usize,
        total_tokens: usize,
        avg_tokens: f64,
        oov_count: usize,
        oov_rate: f64,
        coverage: f64,
        oov_examples: Vec<String>,
    }

    fn compute_metrics(
        name: &str,
        tokenize_fn: &dyn Fn(&str) -> Vec<String>,
        vocab: &BTreeSet<String>,
        corpus: &[&str],
    ) -> Metrics {
        let mut total_tokens = 0;
        let mut total_oov = 0;
        let mut total_known = 0;
        let mut oov_tokens_set = BTreeSet::new();

        for sent in corpus {
            let tokens = tokenize_fn(sent);
            total_tokens += tokens.len();
            for t in &tokens {
                if vocab.contains(t) {
                    total_known += 1;
                } else {
                    total_oov += 1;
                    oov_tokens_set.insert(t.clone());
                }
            }
        }

        let avg_tokens = if corpus.is_empty() { 0.0 } else { total_tokens as f64 / corpus.len() as f64 };
        let total = total_known + total_oov;
        let oov_rate = if total > 0 { total_oov as f64 / total as f64 * 100.0 } else { 0.0 };
        let coverage = 100.0 - oov_rate;
        let oov_examples: Vec<String> = oov_tokens_set.into_iter().take(10).collect();

        Metrics {
            name: name.to_string(),
            vocab_size: vocab.len(),
            total_tokens,
            avg_tokens,
            oov_count: total_oov,
            oov_rate,
            coverage,
            oov_examples,
        }
    }

    // ===== TRAINING CORPUS METRICS =====

    println!("\n\n{}", "=".repeat(70));
    println!("TRAINING CORPUS — VOCABULARY ANALYSIS");
    println!("{}", "=".repeat(70));

    let word_tok_fn = |s: &str| word_tokenize(s);
    let char_tok_fn = |s: &str| char_tokenize(s);

    let train_metrics = vec![
        compute_metrics("Word", &word_tok_fn, &word_vocab, &training_corpus),
        compute_metrics("Character", &char_tok_fn, &char_vocab, &training_corpus),
        compute_metrics("BPE-30", &|s: &str| bpe_small.tokenize(s), &bpe_small.vocab, &training_corpus),
        compute_metrics("BPE-75", &|s: &str| bpe_medium.tokenize(s), &bpe_medium.vocab, &training_corpus),
        compute_metrics("BPE-150", &|s: &str| bpe_large.tokenize(s), &bpe_large.vocab, &training_corpus),
    ];

    println!("\n  {:<12} {:>7} {:>8} {:>10} {:>5} {:>7} {:>10}",
        "Method", "Vocab", "Tokens", "Avg/Sent", "OOV", "OOV%", "Coverage");
    println!("  {}", "-".repeat(63));
    for m in &train_metrics {
        println!("  {:<12} {:>7} {:>8} {:>10.1} {:>5} {:>6.1}% {:>9.1}%",
            m.name, m.vocab_size, m.total_tokens, m.avg_tokens, m.oov_count, m.oov_rate, m.coverage);
    }

    println!("\n  On training data, all methods have 0% OOV (by definition).");
    println!("  The real test is on UNSEEN text.");

    // --- Training Corpus Charts ---
    let train_names: Vec<&str> = train_metrics.iter().map(|m| m.name.as_str()).collect();
    let train_vocab_data: Vec<f64> = train_metrics.iter().map(|m| m.vocab_size as f64).collect();
    let train_avg_data: Vec<f64> = train_metrics.iter().map(|m| (m.avg_tokens * 10.0).round() / 10.0).collect();

    show_chart(&bar_chart(
        "Training Corpus — Vocabulary Size by Method",
        &train_names,
        "Vocab Size",
        &train_vocab_data,
        "#3b82f6",
    ));

    show_chart(&bar_chart(
        "Training Corpus — Avg Tokens per Sentence",
        &train_names,
        "Avg Tokens/Sentence",
        &train_avg_data,
        "#10b981",
    ));

    // ===== TEST CORPUS METRICS =====

    println!("\n\n{}", "=".repeat(70));
    println!("TEST CORPUS — OOV AND COVERAGE ANALYSIS");
    println!("{}", "=".repeat(70));
    println!("  (Vocabularies learned from training corpus, applied to new text)\n");

    let test_metrics = vec![
        compute_metrics("Word", &word_tok_fn, &word_vocab, &test_corpus),
        compute_metrics("Character", &char_tok_fn, &char_vocab, &test_corpus),
        compute_metrics("BPE-30", &|s: &str| bpe_small.tokenize(s), &bpe_small.vocab, &test_corpus),
        compute_metrics("BPE-75", &|s: &str| bpe_medium.tokenize(s), &bpe_medium.vocab, &test_corpus),
        compute_metrics("BPE-150", &|s: &str| bpe_large.tokenize(s), &bpe_large.vocab, &test_corpus),
    ];

    println!("  {:<12} {:>7} {:>8} {:>10} {:>5} {:>7} {:>10}",
        "Method", "Vocab", "Tokens", "Avg/Sent", "OOV", "OOV%", "Coverage");
    println!("  {}", "-".repeat(63));
    for m in &test_metrics {
        println!("  {:<12} {:>7} {:>8} {:>10.1} {:>5} {:>6.1}% {:>9.1}%",
            m.name, m.vocab_size, m.total_tokens, m.avg_tokens, m.oov_count, m.oov_rate, m.coverage);
    }

    // --- Test Corpus Charts ---
    let test_names: Vec<&str> = test_metrics.iter().map(|m| m.name.as_str()).collect();
    let test_avg_data: Vec<f64> = test_metrics.iter().map(|m| (m.avg_tokens * 10.0).round() / 10.0).collect();
    let test_oov_data: Vec<f64> = test_metrics.iter().map(|m| (m.oov_rate * 10.0).round() / 10.0).collect();
    let test_vocab_div10: Vec<f64> = test_metrics.iter().map(|m| (m.vocab_size as f64 / 10.0 * 10.0).round() / 10.0).collect();

    show_chart(&multi_bar_chart(
        "Test Corpus — Comparison Across Methods",
        &test_names,
        &[
            ("Vocab Size (/10)", &test_vocab_div10, "#3b82f6"),
            ("Avg Tokens/Sentence", &test_avg_data, "#10b981"),
            ("OOV Rate (%)", &test_oov_data, "#ef4444"),
        ],
    ));

    show_chart(&bar_chart(
        "Test Corpus — OOV Rate by Method",
        &test_names,
        "OOV Rate (%)",
        &test_oov_data,
        "#ef4444",
    ));

    // ===== OOV DETAILS =====

    println!("\n\n{}", "=".repeat(70));
    println!("OOV DETAILS — WHAT EACH METHOD MISSES");
    println!("{}", "=".repeat(70));

    for m in &test_metrics {
        if !m.oov_examples.is_empty() {
            let examples: Vec<&str> = m.oov_examples.iter().take(8).map(|s| s.as_str()).collect();
            println!("\n  {} ({} OOV tokens):", m.name, m.oov_count);
            println!("    Examples: {:?}", examples);
        } else {
            println!("\n  {}: No OOV tokens", m.name);
        }
    }

    // ===== DETAILED TOKENIZATION COMPARISON =====

    println!("\n\n{}", "=".repeat(70));
    println!("SIDE-BY-SIDE TOKENIZATION OF TEST SENTENCES");
    println!("{}", "=".repeat(70));

    for sent in &test_corpus {
        println!("\n  Input: \"{}\"", sent);
        println!("  {}", "-".repeat(60));

        let w_tok = word_tokenize(sent);
        let c_tok = char_tokenize(sent);
        let bpe_tok = bpe_medium.tokenize(sent);

        let w_oov = w_tok.iter().filter(|t| !word_vocab.contains(t.as_str())).count();
        let bpe_oov = bpe_tok.iter().filter(|t| !bpe_medium.vocab.contains(t.as_str())).count();

        let w_preview: Vec<&str> = w_tok.iter().take(12).map(|s| s.as_str()).collect();
        let bpe_preview: Vec<&str> = bpe_tok.iter().take(12).map(|s| s.as_str()).collect();

        println!("  Word  ({:>3} tokens, {} OOV): {:?}{}", w_tok.len(), w_oov, w_preview, if w_tok.len() > 12 { "..." } else { "" });
        println!("  BPE   ({:>3} tokens, {} OOV): {:?}{}", bpe_tok.len(), bpe_oov, bpe_preview, if bpe_tok.len() > 12 { "..." } else { "" });
        println!("  Char  ({:>3} tokens,  0 OOV): [too long to display]", c_tok.len());
    }

    // ===== FINAL SUMMARY =====

    println!("\n\n{}", "=".repeat(70));
    println!("SUMMARY: THE TOKENIZATION TRADEOFF TRIANGLE");
    println!("{}", "=".repeat(70));
    println!("
  Every tokenization method trades between three properties:

              SMALL VOCABULARY
                   /\\
                  /  \\
                 / BPE\\
                /______\\
               /        \\
      SHORT   /   IDEAL  \\   ZERO
     SEQUENCE/  (impossible\\  OOV
             \\   with fixed/
              \\  methods) /
               \\        /
                \\______/

  - Word tokenization:      short sequences, but huge vocab and high OOV
  - Character tokenization:  tiny vocab and zero OOV, but very long sequences
  - BPE tokenization:        balanced — moderate vocab, moderate length, low OOV

  BPE is not perfect — it is the best available compromise.
  The number of merges is the dial that moves you along the spectrum.

  KEY INSIGHT: Tokenization defines what a model can and cannot see.
  Choosing a tokenizer is choosing what information to preserve and
  what information to discard. There is no free lunch.
");
}
```

---

## Key Takeaways

- **Vocabulary size and OOV rate are inversely related.** Word tokenization has the largest vocabulary but the highest OOV rate on new text. Character tokenization has the smallest vocabulary and zero OOV. BPE sits in between, and the number of merges controls exactly where.
- **Sequence length determines computational cost.** Character tokenization produces sequences 5-7 times longer than word tokenization, which means models need more memory, more computation, and larger context windows to process the same text. BPE keeps sequences only slightly longer than word-level.
- **BPE achieves the best balance for practical NLP.** With a vocabulary of 30,000-50,000 tokens, BPE handles unseen words by decomposing them into known subword pieces while keeping sequences manageable. This is why GPT, BERT, LLaMA, and virtually every modern language model uses a variant of BPE.
- **The tokenizer must match the domain.** A tokenizer trained on news articles will fragment medical or legal text into tiny pieces, increasing sequence length and reducing the model's effective context. This is why tokenizer training data should be as diverse as the model's intended use case.
