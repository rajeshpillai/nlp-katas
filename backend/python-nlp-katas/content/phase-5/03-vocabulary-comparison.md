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

```python
# --- Vocabulary Size and OOV Comparison Across Tokenization Methods ---
# Comparing word, character, and BPE tokenization on the same corpus
# Uses ONLY Python standard library

import re
from collections import Counter


# ===== TOKENIZERS =====

def word_tokenize(text):
    """Word-level tokenization: split on non-alphanumeric boundaries."""
    return re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?|[0-9]+", text.lower())


def char_tokenize(text):
    """Character-level tokenization: each character is a token."""
    return [c for c in text.lower() if c.isalpha() or c == " "]


class BPETokenizer:
    """A minimal BPE tokenizer that learns from a corpus."""

    def __init__(self):
        self.merges = []
        self.vocab = set()

    def _word_to_symbols(self, word):
        return tuple(list(word) + ["</w>"])

    def train(self, corpus, num_merges):
        """Learn BPE merge operations from corpus."""
        word_freqs = Counter()
        for sentence in corpus:
            for word in sentence.lower().split():
                cleaned = re.sub(r"[^a-zA-Z]", "", word)
                if cleaned:
                    word_freqs[cleaned] += 1

        word_symbols = {}
        for word in word_freqs:
            word_symbols[word] = self._word_to_symbols(word)

        self.vocab = set()
        for symbols in word_symbols.values():
            self.vocab.update(symbols)

        self.merges = []
        for _ in range(num_merges):
            pair_freqs = Counter()
            for word, freq in word_freqs.items():
                symbols = word_symbols[word]
                for i in range(len(symbols) - 1):
                    pair_freqs[(symbols[i], symbols[i + 1])] += freq

            if not pair_freqs:
                break

            best_pair = max(pair_freqs, key=pair_freqs.get)
            new_symbol = best_pair[0] + best_pair[1]

            new_word_symbols = {}
            for word, symbols in word_symbols.items():
                new_syms = []
                i = 0
                while i < len(symbols):
                    if (i < len(symbols) - 1 and
                            symbols[i] == best_pair[0] and
                            symbols[i + 1] == best_pair[1]):
                        new_syms.append(new_symbol)
                        i += 2
                    else:
                        new_syms.append(symbols[i])
                        i += 1
                new_word_symbols[word] = tuple(new_syms)

            word_symbols = new_word_symbols
            self.vocab.add(new_symbol)
            self.merges.append(best_pair)

    def encode_word(self, word):
        """Encode a single word using learned merges."""
        symbols = list(word.lower()) + ["</w>"]
        for pair in self.merges:
            i = 0
            while i < len(symbols) - 1:
                if symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
                    symbols = symbols[:i] + [pair[0] + pair[1]] + symbols[i + 2:]
                else:
                    i += 1
        return symbols

    def tokenize(self, text):
        """Tokenize full text using learned BPE merges."""
        tokens = []
        for word in text.lower().split():
            cleaned = re.sub(r"[^a-zA-Z]", "", word)
            if cleaned:
                tokens.extend(self.encode_word(cleaned))
        return tokens


# ===== CORPUS =====

training_corpus = [
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
]

test_corpus = [
    "Biomedical researchers investigated unprecedented pharmacological interactions",
    "The computational linguists developed morphological analyzers for agglutinative languages",
    "Unsupervised pretraining dramatically outperformed traditional feature engineering",
    "Multilingual tokenization presents unique challenges for non-segmented scripts",
    "The neurolinguistic study examined psycholinguistic processing of garden path sentences",
]

print("=" * 70)
print("CORPUS OVERVIEW")
print("=" * 70)
print(f"\n  Training corpus: {len(training_corpus)} sentences")
print(f"  Test corpus:     {len(test_corpus)} sentences (different domain)")
print(f"\n  Training sample: \"{training_corpus[0]}\"")
print(f"  Test sample:     \"{test_corpus[0]}\"")


# ===== BUILD TOKENIZERS =====

# Word tokenizer vocabulary from training corpus
word_vocab = set()
for sent in training_corpus:
    word_vocab.update(word_tokenize(sent))

# Character tokenizer vocabulary from training corpus
char_vocab = set()
for sent in training_corpus:
    char_vocab.update(char_tokenize(sent))

# BPE tokenizers with different merge counts
bpe_small = BPETokenizer()
bpe_small.train(training_corpus, num_merges=30)

bpe_medium = BPETokenizer()
bpe_medium.train(training_corpus, num_merges=75)

bpe_large = BPETokenizer()
bpe_large.train(training_corpus, num_merges=150)


# ===== METRIC COMPUTATION =====

def compute_metrics(name, tokenize_fn, vocab, corpus, is_bpe=False):
    """Compute vocabulary size, avg tokens, and OOV metrics."""
    total_tokens = 0
    total_oov = 0
    total_known = 0
    all_tokens = []
    oov_tokens = []

    for sent in corpus:
        tokens = tokenize_fn(sent)
        total_tokens += len(tokens)
        for t in tokens:
            if t in vocab:
                total_known += 1
            else:
                total_oov += 1
                oov_tokens.append(t)
        all_tokens.extend(tokens)

    avg_tokens = total_tokens / len(corpus) if corpus else 0
    total = total_known + total_oov
    oov_rate = (total_oov / total * 100) if total > 0 else 0
    coverage = 100 - oov_rate

    return {
        "name": name,
        "vocab_size": len(vocab),
        "total_tokens": total_tokens,
        "avg_tokens": avg_tokens,
        "oov_count": total_oov,
        "oov_rate": oov_rate,
        "coverage": coverage,
        "oov_examples": list(set(oov_tokens))[:10],
    }


# ===== TRAINING CORPUS METRICS =====

print("\n\n" + "=" * 70)
print("TRAINING CORPUS — VOCABULARY ANALYSIS")
print("=" * 70)

train_metrics = [
    compute_metrics("Word", word_tokenize, word_vocab, training_corpus),
    compute_metrics("Character", char_tokenize, char_vocab, training_corpus),
    compute_metrics("BPE-30", bpe_small.tokenize, bpe_small.vocab, training_corpus),
    compute_metrics("BPE-75", bpe_medium.tokenize, bpe_medium.vocab, training_corpus),
    compute_metrics("BPE-150", bpe_large.tokenize, bpe_large.vocab, training_corpus),
]

print(f"\n  {'Method':<12} {'Vocab':>7} {'Tokens':>8} {'Avg/Sent':>10}"
      f" {'OOV':>5} {'OOV%':>7} {'Coverage':>10}")
print("  " + "-" * 63)
for m in train_metrics:
    print(f"  {m['name']:<12} {m['vocab_size']:>7} {m['total_tokens']:>8}"
          f" {m['avg_tokens']:>10.1f} {m['oov_count']:>5}"
          f" {m['oov_rate']:>6.1f}% {m['coverage']:>9.1f}%")

print("\n  On training data, all methods have 0% OOV (by definition).")
print("  The real test is on UNSEEN text.")

# --- Training Corpus Charts ---
train_names = [m["name"] for m in train_metrics]
train_vocab = [m["vocab_size"] for m in train_metrics]
train_avg = [round(m["avg_tokens"], 1) for m in train_metrics]

show_chart({
    "type": "bar",
    "title": "Training Corpus — Vocabulary Size by Method",
    "labels": train_names,
    "datasets": [{"label": "Vocab Size", "data": train_vocab, "color": "#3b82f6"}],
    "options": {"x_label": "Method", "y_label": "Vocabulary Size"}
})

show_chart({
    "type": "bar",
    "title": "Training Corpus — Avg Tokens per Sentence",
    "labels": train_names,
    "datasets": [{"label": "Avg Tokens/Sentence", "data": train_avg, "color": "#10b981"}],
    "options": {"x_label": "Method", "y_label": "Avg Tokens per Sentence"}
})


# ===== TEST CORPUS METRICS =====

print("\n\n" + "=" * 70)
print("TEST CORPUS — OOV AND COVERAGE ANALYSIS")
print("=" * 70)
print("  (Vocabularies learned from training corpus, applied to new text)\n")

test_metrics = [
    compute_metrics("Word", word_tokenize, word_vocab, test_corpus),
    compute_metrics("Character", char_tokenize, char_vocab, test_corpus),
    compute_metrics("BPE-30", bpe_small.tokenize, bpe_small.vocab, test_corpus),
    compute_metrics("BPE-75", bpe_medium.tokenize, bpe_medium.vocab, test_corpus),
    compute_metrics("BPE-150", bpe_large.tokenize, bpe_large.vocab, test_corpus),
]

print(f"  {'Method':<12} {'Vocab':>7} {'Tokens':>8} {'Avg/Sent':>10}"
      f" {'OOV':>5} {'OOV%':>7} {'Coverage':>10}")
print("  " + "-" * 63)
for m in test_metrics:
    print(f"  {m['name']:<12} {m['vocab_size']:>7} {m['total_tokens']:>8}"
          f" {m['avg_tokens']:>10.1f} {m['oov_count']:>5}"
          f" {m['oov_rate']:>6.1f}% {m['coverage']:>9.1f}%")


# --- Test Corpus Charts ---
test_names = [m["name"] for m in test_metrics]
test_vocab = [m["vocab_size"] for m in test_metrics]
test_avg = [round(m["avg_tokens"], 1) for m in test_metrics]
test_oov = [round(m["oov_rate"], 1) for m in test_metrics]

show_chart({
    "type": "bar",
    "title": "Test Corpus — Comparison Across Methods",
    "labels": test_names,
    "datasets": [
        {"label": "Vocab Size (÷10)", "data": [round(v / 10, 1) for v in test_vocab], "color": "#3b82f6"},
        {"label": "Avg Tokens/Sentence", "data": test_avg, "color": "#10b981"},
        {"label": "OOV Rate (%)", "data": test_oov, "color": "#ef4444"},
    ],
    "options": {"x_label": "Method", "y_label": "Value"}
})

show_chart({
    "type": "bar",
    "title": "Test Corpus — OOV Rate by Method",
    "labels": test_names,
    "datasets": [{"label": "OOV Rate (%)", "data": test_oov, "color": "#ef4444"}],
    "options": {"x_label": "Method", "y_label": "OOV Rate (%)"}
})


# ===== OOV DETAILS =====

print("\n\n" + "=" * 70)
print("OOV DETAILS — WHAT EACH METHOD MISSES")
print("=" * 70)

for m in test_metrics:
    if m["oov_examples"]:
        examples = m["oov_examples"][:8]
        print(f"\n  {m['name']} ({m['oov_count']} OOV tokens):")
        print(f"    Examples: {examples}")
    else:
        print(f"\n  {m['name']}: No OOV tokens")


# ===== DETAILED TOKENIZATION COMPARISON =====

print("\n\n" + "=" * 70)
print("SIDE-BY-SIDE TOKENIZATION OF TEST SENTENCES")
print("=" * 70)

for sent in test_corpus:
    print(f'\n  Input: "{sent}"')
    print("  " + "-" * 60)

    w_tok = word_tokenize(sent)
    c_tok = char_tokenize(sent)
    bpe_tok = bpe_medium.tokenize(sent)

    w_oov = sum(1 for t in w_tok if t not in word_vocab)
    bpe_oov = sum(1 for t in bpe_tok if t not in bpe_medium.vocab)

    print(f"  Word  ({len(w_tok):>3} tokens, {w_oov} OOV): {w_tok[:12]}{'...' if len(w_tok) > 12 else ''}")
    print(f"  BPE   ({len(bpe_tok):>3} tokens, {bpe_oov} OOV): {bpe_tok[:12]}{'...' if len(bpe_tok) > 12 else ''}")
    print(f"  Char  ({len(c_tok):>3} tokens,  0 OOV): [too long to display]")


# ===== FINAL SUMMARY =====

print("\n\n" + "=" * 70)
print("SUMMARY: THE TOKENIZATION TRADEOFF TRIANGLE")
print("=" * 70)
print("""
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
""")
```

---

## Key Takeaways

- **Vocabulary size and OOV rate are inversely related.** Word tokenization has the largest vocabulary but the highest OOV rate on new text. Character tokenization has the smallest vocabulary and zero OOV. BPE sits in between, and the number of merges controls exactly where.
- **Sequence length determines computational cost.** Character tokenization produces sequences 5-7 times longer than word tokenization, which means models need more memory, more computation, and larger context windows to process the same text. BPE keeps sequences only slightly longer than word-level.
- **BPE achieves the best balance for practical NLP.** With a vocabulary of 30,000-50,000 tokens, BPE handles unseen words by decomposing them into known subword pieces while keeping sequences manageable. This is why GPT, BERT, LLaMA, and virtually every modern language model uses a variant of BPE.
- **The tokenizer must match the domain.** A tokenizer trained on news articles will fragment medical or legal text into tiny pieces, increasing sequence length and reducing the model's effective context. This is why tokenizer training data should be as diverse as the model's intended use case.
