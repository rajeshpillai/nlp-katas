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

```python
# --- Byte Pair Encoding (BPE) from Scratch ---
# Complete implementation using only Python standard library

from collections import Counter, defaultdict


def get_word_freqs(corpus):
    """Tokenize corpus into words and count frequencies.
    Each word is represented as a tuple of characters plus an end-of-word marker.
    """
    word_freqs = Counter()
    for sentence in corpus:
        for word in sentence.lower().split():
            # Represent word as space-separated characters + end marker
            word_freqs[word] += 1
    return word_freqs


def word_to_symbols(word):
    """Convert a word string to a tuple of character symbols with end marker."""
    return tuple(list(word) + ["</w>"])


def get_pair_freqs(word_freqs, word_symbols):
    """Count frequencies of all adjacent symbol pairs across the vocabulary."""
    pair_freqs = Counter()
    for word, freq in word_freqs.items():
        symbols = word_symbols[word]
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            pair_freqs[pair] += freq
    return pair_freqs


def merge_pair(word_symbols, pair, new_symbol):
    """Apply a merge operation: replace all occurrences of pair with new_symbol."""
    new_word_symbols = {}
    for word, symbols in word_symbols.items():
        new_symbols = []
        i = 0
        while i < len(symbols):
            if (i < len(symbols) - 1 and
                    symbols[i] == pair[0] and symbols[i + 1] == pair[1]):
                new_symbols.append(new_symbol)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        new_word_symbols[word] = tuple(new_symbols)
    return new_word_symbols


def train_bpe(corpus, num_merges):
    """Train BPE: learn merge operations from corpus."""
    # Step 1: Get word frequencies
    word_freqs = get_word_freqs(corpus)

    # Step 2: Initialize each word as character sequence
    word_symbols = {}
    for word in word_freqs:
        word_symbols[word] = word_to_symbols(word)

    # Build initial vocabulary (all individual characters + end marker)
    vocab = set()
    for symbols in word_symbols.values():
        vocab.update(symbols)

    merges = []

    print("=" * 60)
    print("BPE TRAINING")
    print("=" * 60)
    print(f"\nCorpus word frequencies: {dict(word_freqs)}")
    print(f"Initial vocabulary ({len(vocab)}): {sorted(vocab)}")
    print(f"\nInitial word representations:")
    for word, symbols in sorted(word_symbols.items()):
        print(f"  {word:>12} -> {' '.join(symbols)}")

    print(f"\nPerforming {num_merges} merges...\n")

    for step in range(num_merges):
        # Count all adjacent pairs
        pair_freqs = get_pair_freqs(word_freqs, word_symbols)

        if not pair_freqs:
            print(f"  No more pairs to merge at step {step + 1}.")
            break

        # Find the most frequent pair
        best_pair = max(pair_freqs, key=pair_freqs.get)
        best_freq = pair_freqs[best_pair]

        # Create new symbol by concatenating the pair
        new_symbol = best_pair[0] + best_pair[1]

        # Apply the merge
        word_symbols = merge_pair(word_symbols, best_pair, new_symbol)
        vocab.add(new_symbol)
        merges.append(best_pair)

        print(f"  Merge {step + 1}: ({best_pair[0]:>6}, {best_pair[1]:<6})"
              f" -> {new_symbol:<10} (frequency: {best_freq})")

        # Show current state of word representations
        for word, symbols in sorted(word_symbols.items()):
            print(f"           {word:>12} -> {' '.join(symbols)}")
        print()

    print(f"Final vocabulary ({len(vocab)} symbols): {sorted(vocab)}")
    return merges, vocab, word_symbols


def encode_word(word, merges):
    """Encode a single word using learned BPE merges."""
    symbols = list(word) + ["</w>"]

    for pair in merges:
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
                symbols = symbols[:i] + [pair[0] + pair[1]] + symbols[i + 2:]
            else:
                i += 1

    return symbols


def encode_text(text, merges):
    """Encode full text using learned BPE merges."""
    tokens = []
    for word in text.lower().split():
        word_tokens = encode_word(word, merges)
        tokens.extend(word_tokens)
    return tokens


# --- Train BPE on a small corpus ---
corpus = [
    "low low low low low",
    "lower lower lower",
    "lowest lowest",
    "newer newer newer newer",
    "newest newest",
    "wider wider wider",
    "widest widest",
]

merges, vocab, final_symbols = train_bpe(corpus, num_merges=15)

# --- Show the merge table ---
print("\n" + "=" * 60)
print("LEARNED MERGE TABLE (recipe for tokenization)")
print("=" * 60)
for i, (a, b) in enumerate(merges, 1):
    print(f"  {i:>3}. {a} + {b} -> {a + b}")

# --- Encode new text ---
print("\n" + "=" * 60)
print("ENCODING TRAINING WORDS")
print("=" * 60)

training_words = ["low", "lower", "lowest", "newer", "newest", "wider", "widest"]
for word in training_words:
    tokens = encode_word(word, merges)
    print(f"  {word:>10} -> {tokens}")

print("\n" + "=" * 60)
print("ENCODING UNSEEN WORDS")
print("=" * 60)
print("(Words NOT in training corpus)\n")

unseen_words = ["new", "slow", "slower", "slowest", "highest", "tower", "flower"]
for word in unseen_words:
    tokens = encode_word(word, merges)
    in_vocab = all(t in vocab for t in tokens)
    status = "all known" if in_vocab else "has unknown pieces"
    print(f"  {word:>10} -> {tokens}  ({status})")

# --- Full text encoding ---
print("\n" + "=" * 60)
print("ENCODING FULL SENTENCES")
print("=" * 60)

test_sentences = [
    "lower is lower",
    "the newest tower",
    "slower and widest",
]

for sent in test_sentences:
    tokens = encode_text(sent, merges)
    print(f'  "{sent}"')
    print(f"    -> {tokens}")
    print(f"    ({len(tokens)} tokens)")
    print()

# --- Statistics ---
print("=" * 60)
print("BPE STATISTICS")
print("=" * 60)
print(f"  Number of merges learned:  {len(merges)}")
print(f"  Final vocabulary size:     {len(vocab)}")
print(f"  Initial vocab (chars only): {len(set(c for w in training_words for c in w) | {'</w>'})}")
print(f"  Compression ratio:         {len(vocab)} symbols cover all words")
print()
print("  KEY INSIGHT: BPE learns a vocabulary that balances")
print("  whole words (for common terms) with character pieces")
print("  (for rare terms). It needs no linguistic rules —")
print("  only corpus statistics.")
```

---

## Key Takeaways

- **BPE learns subword units from data, not from rules.** It starts with characters and iteratively merges the most frequent adjacent pair. This process discovers common prefixes, suffixes, and roots automatically, making it language-agnostic and domain-adaptive.
- **The merge table is a deterministic recipe.** Once trained, the ordered list of merge operations is applied to any new text in the same sequence. This guarantees reproducible tokenization and allows the model to share a consistent vocabulary between training and inference.
- **BPE handles unseen words by decomposition.** A word never encountered during training is broken into smaller known pieces — subwords or individual characters. The model always has some representation, unlike word-level tokenizers that produce a meaningless [UNK] token.
- **Vocabulary size is a hyperparameter you control.** The number of BPE merges directly determines vocabulary size. More merges produce larger vocabularies with more whole-word tokens (shorter sequences). Fewer merges produce smaller vocabularies with more character-level decomposition (longer sequences). Modern models typically use 30,000-50,000 merges.
