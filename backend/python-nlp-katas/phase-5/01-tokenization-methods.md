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

```python
# --- Tokenization Methods: Word, Character, and Subword ---
# Comparing three fundamental approaches to splitting text into tokens
# Uses ONLY Python standard library

import re
from collections import Counter


def word_tokenize(text):
    """Split text into word tokens on whitespace and punctuation boundaries."""
    return re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?|[0-9]+|[^\s\w]", text.lower())


def char_tokenize(text):
    """Split text into individual characters (including spaces)."""
    return list(text.lower())


def subword_tokenize(text, common_affixes=None):
    """Split words into subword pieces using common English affixes.

    This is a simplified, rule-based subword tokenizer to illustrate the concept.
    Real subword tokenizers (BPE, WordPiece) learn their vocabulary from data.
    """
    if common_affixes is None:
        common_affixes = {
            "prefixes": ["un", "re", "pre", "dis", "mis", "over", "under",
                         "out", "non", "multi", "de"],
            "suffixes": ["ing", "tion", "sion", "ment", "ness", "able", "ible",
                         "ful", "less", "ous", "ive", "ed", "er", "est",
                         "ly", "al", "ity", "ize", "ise"],
        }

    words = re.findall(r"[a-zA-Z]+|[0-9]+|[^\s\w]", text.lower())
    tokens = []

    for word in words:
        if len(word) <= 3:
            tokens.append(word)
            continue

        pieces = []
        remaining = word

        # Try to strip a prefix
        for prefix in sorted(common_affixes["prefixes"], key=len, reverse=True):
            if remaining.startswith(prefix) and len(remaining) > len(prefix) + 2:
                pieces.append(prefix)
                remaining = remaining[len(prefix):]
                break

        # Try to strip a suffix
        suffix_found = None
        for suffix in sorted(common_affixes["suffixes"], key=len, reverse=True):
            if remaining.endswith(suffix) and len(remaining) > len(suffix) + 1:
                suffix_found = suffix
                remaining = remaining[:-len(suffix)]
                break

        pieces.append(remaining)
        if suffix_found:
            pieces.append(suffix_found)

        tokens.extend(pieces)

    return tokens


# --- Corpus ---
sentences = [
    "The researchers investigated unforgettable preprocessing techniques",
    "Unhappiness is unbelievable and overwhelming",
    "The disconnected microprocessor was malfunctioning repeatedly",
    "She enjoys running swimming and cycling outdoors",
    "Antidisestablishmentarianism is a remarkably long word",
]

print("=" * 70)
print("TOKENIZATION METHODS COMPARISON")
print("=" * 70)

for sent in sentences:
    print(f'\nSentence: "{sent}"')
    print("-" * 60)

    w_tokens = word_tokenize(sent)
    c_tokens = char_tokenize(sent)
    s_tokens = subword_tokenize(sent)

    print(f"  Word tokens     ({len(w_tokens):>2}): {w_tokens}")
    print(f"  Character tokens({len(c_tokens):>2}): {''.join(c_tokens[:40])}...")
    print(f"  Subword tokens  ({len(s_tokens):>2}): {s_tokens}")

# --- Vocabulary Size Analysis ---
print("\n" + "=" * 70)
print("VOCABULARY SIZE ANALYSIS")
print("=" * 70)

all_word_tokens = []
all_char_tokens = []
all_subword_tokens = []

for sent in sentences:
    all_word_tokens.extend(word_tokenize(sent))
    all_char_tokens.extend(char_tokenize(sent))
    all_subword_tokens.extend(subword_tokenize(sent))

word_vocab = set(all_word_tokens)
char_vocab = set(all_char_tokens)
subword_vocab = set(all_subword_tokens)

print(f"\n  {'Method':<15} {'Vocab Size':>12} {'Total Tokens':>14} {'Avg Tok/Sent':>14}")
print("  " + "-" * 57)
print(f"  {'Word':<15} {len(word_vocab):>12} {len(all_word_tokens):>14}"
      f" {len(all_word_tokens)/len(sentences):>14.1f}")
print(f"  {'Character':<15} {len(char_vocab):>12} {len(all_char_tokens):>14}"
      f" {len(all_char_tokens)/len(sentences):>14.1f}")
print(f"  {'Subword':<15} {len(subword_vocab):>12} {len(all_subword_tokens):>14}"
      f" {len(all_subword_tokens)/len(sentences):>14.1f}")

print(f"\n  Word vocabulary:    {sorted(word_vocab)}")
print(f"  Char vocabulary:    {sorted(char_vocab)}")
print(f"  Subword vocabulary: {sorted(subword_vocab)}")

# --- Vocabulary Size Chart ---
methods = ["Word", "Character", "Subword"]
vocab_sizes = [len(word_vocab), len(char_vocab), len(subword_vocab)]
avg_tokens = [
    len(all_word_tokens) / len(sentences),
    len(all_char_tokens) / len(sentences),
    len(all_subword_tokens) / len(sentences),
]

show_chart({
    "type": "bar",
    "title": "Vocabulary Size by Tokenization Method",
    "labels": methods,
    "datasets": [{"label": "Vocabulary Size", "data": vocab_sizes, "color": "#3b82f6"}],
    "options": {"x_label": "Method", "y_label": "Unique Tokens"}
})

show_chart({
    "type": "bar",
    "title": "Average Tokens per Sentence by Method",
    "labels": methods,
    "datasets": [{"label": "Avg Tokens/Sentence", "data": [round(v, 1) for v in avg_tokens], "color": "#10b981"}],
    "options": {"x_label": "Method", "y_label": "Avg Tokens per Sentence"}
})

# --- OOV Demonstration ---
print("\n" + "=" * 70)
print("OUT-OF-VOCABULARY (OOV) DEMONSTRATION")
print("=" * 70)

# Training vocabulary from our corpus
training_vocab_word = word_vocab.copy()
training_vocab_subword = subword_vocab.copy()

# New unseen text
unseen_sentences = [
    "The undiscovered preprocessing was disheartening",
    "Rethinking multitasking requires carefulness",
    "Overwhelming discouragement is preventable",
]

print("\nTraining vocabulary learned from corpus above.")
print("Now testing on UNSEEN text:\n")

for sent in unseen_sentences:
    print(f'  Input: "{sent}"')

    # Word-level OOV
    w_tokens = word_tokenize(sent)
    w_oov = [t for t in w_tokens if t not in training_vocab_word]
    w_known = [t for t in w_tokens if t in training_vocab_word]
    w_oov_rate = len(w_oov) / len(w_tokens) * 100 if w_tokens else 0

    # Subword-level OOV
    s_tokens = subword_tokenize(sent)
    s_oov = [t for t in s_tokens if t not in training_vocab_subword]
    s_known = [t for t in s_tokens if t in training_vocab_subword]
    s_oov_rate = len(s_oov) / len(s_tokens) * 100 if s_tokens else 0

    print(f"    Word tokens:    {w_tokens}")
    print(f"      Known: {w_known}")
    print(f"      OOV:   {w_oov}  ({w_oov_rate:.0f}% OOV)")
    print(f"    Subword tokens: {s_tokens}")
    print(f"      Known: {s_known}")
    print(f"      OOV:   {s_oov}  ({s_oov_rate:.0f}% OOV)")
    print()

# Character tokenization never has OOV
print("  Character tokenization: 0% OOV (always)")
print("  It can represent ANY input, but at the cost of very long sequences.")

# --- Tradeoff Summary ---
print("\n" + "=" * 70)
print("TRADEOFF SUMMARY")
print("=" * 70)
print("""
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
""")
```

---

## Key Takeaways

- **Tokenization is a design decision, not just preprocessing.** The choice of tokenization method determines what information the model receives. Word tokens carry meaning but create OOV gaps. Character tokens cover everything but dilute meaning into long sequences. Subword tokens balance both concerns.
- **The OOV problem is word tokenization's fatal flaw.** Any word not in the training vocabulary becomes an unknown token, destroying information. In domains with evolving language (medicine, technology, social media), word-level tokenizers become obsolete quickly.
- **Character tokenization trades OOV safety for sequence length.** A model processing characters must learn to compose letters into meanings — a much harder task than working with pre-formed words. This makes training slower and requires more model capacity.
- **Subword tokenization is the foundation of modern NLP.** By decomposing words into reusable pieces (prefixes, roots, suffixes), subword methods achieve small vocabularies, manageable sequence lengths, and robust OOV handling. This is why every major language model today uses subword tokenization.
