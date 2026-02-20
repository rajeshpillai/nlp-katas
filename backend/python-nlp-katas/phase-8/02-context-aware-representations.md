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

```python
# --- Context-Aware vs Context-Free Representations ---
# Only Python stdlib -- no external packages

import math
import random
from collections import Counter

random.seed(42)


def tokenize(text):
    """Lowercase and split on non-alpha characters."""
    result = []
    current = []
    for ch in text.lower():
        if ch.isalpha():
            current.append(ch)
        else:
            if current:
                result.append("".join(current))
                current = []
    if current:
        result.append("".join(current))
    return result


def dot_product(v1, v2):
    """Dot product of two vectors."""
    return sum(a * b for a, b in zip(v1, v2))


def magnitude(v):
    """L2 norm of a vector."""
    return math.sqrt(sum(x * x for x in v))


def cosine_similarity(v1, v2):
    """Cosine similarity between two vectors."""
    d = dot_product(v1, v2)
    m1, m2 = magnitude(v1), magnitude(v2)
    if m1 == 0 or m2 == 0:
        return 0.0
    return d / (m1 * m2)


def vector_add(v1, v2):
    """Element-wise addition of two vectors."""
    return [a + b for a, b in zip(v1, v2)]


def vector_scale(v, s):
    """Multiply a vector by a scalar."""
    return [x * s for x in v]


# ============================================================
# Build a small word vector space using co-occurrence
# ============================================================

# A small corpus designed so that "bank" appears in two distinct contexts
corpus = [
    # Financial context
    "I deposited money at the bank this morning",
    "The bank approved my loan application today",
    "She withdrew cash from the bank account",
    "The bank charged interest on the loan",
    "My savings at the bank grew over time",
    "The bank manager reviewed the financial records",
    # River/nature context
    "I sat on the river bank and watched the water",
    "The river bank was covered with tall grass",
    "We picnicked on the bank of the stream",
    "Fish jumped near the muddy bank of the river",
    "The bank of the lake was rocky and steep",
    "Wildflowers grew along the bank of the creek",
    # Additional context sentences (no "bank")
    "Money and finance are important topics",
    "The loan officer reviewed the application carefully",
    "The river flowed through the green valley",
    "Water rushed over rocks in the stream",
]

print("=" * 65)
print("CORPUS")
print("=" * 65)
for i, sent in enumerate(corpus):
    print(f"  [{i:2d}] {sent}")
print()

# Tokenize all sentences
tokenized_corpus = [tokenize(sent) for sent in corpus]
all_tokens = [tok for sent in tokenized_corpus for tok in sent]
vocab = sorted(set(all_tokens))
word_to_idx = {w: i for i, w in enumerate(vocab)}
vocab_size = len(vocab)

print(f"Vocabulary size: {vocab_size} words")
print()

# ============================================================
# Build co-occurrence vectors (context-free word representations)
# ============================================================

# Each word gets a vector based on what words appear near it in the corpus
WINDOW = 3
DIM = vocab_size  # Use raw co-occurrence counts as vectors

co_occurrence = {}
for w in vocab:
    co_occurrence[w] = [0] * vocab_size

for sent_tokens in tokenized_corpus:
    for i, word in enumerate(sent_tokens):
        # Look at neighbors within the window
        start = max(0, i - WINDOW)
        end = min(len(sent_tokens), i + WINDOW + 1)
        for j in range(start, end):
            if j != i:
                neighbor = sent_tokens[j]
                co_occurrence[word][word_to_idx[neighbor]] += 1

# These co-occurrence vectors are our CONTEXT-FREE representations
# Each word type gets ONE vector regardless of usage

print("=" * 65)
print("PART 1: CONTEXT-FREE REPRESENTATION OF 'BANK'")
print("=" * 65)
print()
print("Context-free means: ONE vector for 'bank', no matter the sentence.")
print()

bank_vec = co_occurrence["bank"]
# Show top co-occurring words
bank_cooc = [(vocab[i], bank_vec[i]) for i in range(vocab_size) if bank_vec[i] > 0]
bank_cooc.sort(key=lambda x: -x[1])

print("Top words co-occurring with 'bank':")
for word, count in bank_cooc[:15]:
    bar = "#" * count
    print(f"  {word:>12s}: {count:>2d}  {bar}")

print()
print("Notice: both financial words (money, loan, deposited) and")
print("nature words (river, water, grass) appear. The single vector")
print("for 'bank' is a MIX of both meanings -- good for neither.")
print()

# ============================================================
# Show that context-free "bank" is equally similar to both domains
# ============================================================

print("=" * 65)
print("PART 2: CONTEXT-FREE SIMILARITY (THE PROBLEM)")
print("=" * 65)
print()

# Pick representative words from each domain
financial_words = ["money", "loan", "savings", "interest", "financial"]
nature_words = ["river", "water", "grass", "stream", "rocky"]

# Available words in our vocab
financial_available = [w for w in financial_words if w in co_occurrence]
nature_available = [w for w in nature_words if w in co_occurrence]

print("Financial domain words:", financial_available)
print("Nature domain words:   ", nature_available)
print()

print("Cosine similarity of 'bank' with each word (context-free):")
print()

for label, words in [("Financial", financial_available), ("Nature", nature_available)]:
    sims = []
    for w in words:
        sim = cosine_similarity(co_occurrence["bank"], co_occurrence[w])
        sims.append((w, sim))
        print(f"  bank <-> {w:>12s}: {sim:.4f}")
    avg_sim = sum(s for _, s in sims) / len(sims) if sims else 0
    print(f"  {'Average ' + label + ':':>20s} {avg_sim:.4f}")
    print()

print("The context-free 'bank' vector is somewhat similar to BOTH domains.")
print("It cannot commit to either meaning because it must represent all of them.")
print()

# ============================================================
# Build context-AWARE representations
# ============================================================

print("=" * 65)
print("PART 3: CONTEXT-AWARE REPRESENTATION OF 'BANK'")
print("=" * 65)
print()
print("Context-aware means: average 'bank' with its neighbors in each sentence.")
print("Each occurrence of 'bank' gets a DIFFERENT vector.")
print()

CTX_WINDOW = 2  # Look 2 words left and right

bank_sentences = []
for idx, sent_tokens in enumerate(tokenized_corpus):
    if "bank" in sent_tokens:
        bank_pos = sent_tokens.index("bank")
        bank_sentences.append((idx, sent_tokens, bank_pos))

context_vectors = []

for sent_idx, sent_tokens, bank_pos in bank_sentences:
    # Gather context window
    start = max(0, bank_pos - CTX_WINDOW)
    end = min(len(sent_tokens), bank_pos + CTX_WINDOW + 1)
    context_words = [sent_tokens[j] for j in range(start, end)]

    # Average the co-occurrence vectors of all words in the window
    avg_vec = [0.0] * vocab_size
    for cw in context_words:
        avg_vec = vector_add(avg_vec, co_occurrence[cw])
    avg_vec = vector_scale(avg_vec, 1.0 / len(context_words))

    context_vectors.append((sent_idx, context_words, avg_vec))

    print(f"  Sentence [{sent_idx:2d}]: \"{corpus[sent_idx]}\"")
    print(f"    'bank' at position {bank_pos}")
    print(f"    Context window: {context_words}")
    print()

# ============================================================
# Compare context-aware bank vectors to each domain
# ============================================================

print("=" * 65)
print("PART 4: CONTEXT-AWARE SIMILARITY (THE IMPROVEMENT)")
print("=" * 65)
print()

# Average financial and nature word vectors to create domain centroids
def domain_centroid(word_list):
    centroid = [0.0] * vocab_size
    count = 0
    for w in word_list:
        if w in co_occurrence:
            centroid = vector_add(centroid, co_occurrence[w])
            count += 1
    if count > 0:
        centroid = vector_scale(centroid, 1.0 / count)
    return centroid

financial_centroid = domain_centroid(financial_available)
nature_centroid = domain_centroid(nature_available)

# Context-free bank similarity
cf_fin = cosine_similarity(co_occurrence["bank"], financial_centroid)
cf_nat = cosine_similarity(co_occurrence["bank"], nature_centroid)

print("Context-FREE 'bank' (single vector for all uses):")
print(f"  Similarity to financial domain: {cf_fin:.4f}")
print(f"  Similarity to nature domain:    {cf_nat:.4f}")
print(f"  Difference:                     {abs(cf_fin - cf_nat):.4f}")
print(f"  --> Cannot strongly commit to either meaning.")
print()

print("Context-AWARE 'bank' (different vector per occurrence):")
print()

financial_sents = []
nature_sents = []

for sent_idx, context_words, ctx_vec in context_vectors:
    sim_fin = cosine_similarity(ctx_vec, financial_centroid)
    sim_nat = cosine_similarity(ctx_vec, nature_centroid)

    is_financial = sent_idx < 6  # First 6 sentences are financial
    expected = "financial" if is_financial else "nature"
    predicted = "financial" if sim_fin > sim_nat else "nature"
    correct = expected == predicted
    marker = "CORRECT" if correct else "WRONG"

    sent_preview = corpus[sent_idx]
    if len(sent_preview) > 50:
        sent_preview = sent_preview[:50] + "..."

    print(f"  [{sent_idx:2d}] \"{sent_preview}\"")
    print(f"      Context: {context_words}")
    print(f"      Sim to financial: {sim_fin:.4f}  |  Sim to nature: {sim_nat:.4f}")
    print(f"      Expected: {expected:>10s}  |  Predicted: {predicted:>10s}  [{marker}]")
    print()

    if is_financial:
        financial_sents.append((sim_fin, sim_nat))
    else:
        nature_sents.append((sim_fin, sim_nat))

# Compute accuracy
all_results = []
for sim_fin, sim_nat in financial_sents:
    all_results.append(sim_fin > sim_nat)
for sim_fin, sim_nat in nature_sents:
    all_results.append(sim_nat > sim_fin)

accuracy = sum(all_results) / len(all_results) if all_results else 0
print(f"Disambiguation accuracy: {sum(all_results)}/{len(all_results)} ({accuracy:.0%})")
print()

# ============================================================
# Show the improvement quantitatively
# ============================================================

print("=" * 65)
print("PART 5: QUANTIFYING THE IMPROVEMENT")
print("=" * 65)
print()

if financial_sents:
    avg_fin_sim = sum(sf for sf, _ in financial_sents) / len(financial_sents)
    avg_fin_nat = sum(sn for _, sn in financial_sents) / len(financial_sents)
    print(f"Financial sentences (bank = money institution):")
    print(f"  Avg similarity to financial domain: {avg_fin_sim:.4f}")
    print(f"  Avg similarity to nature domain:    {avg_fin_nat:.4f}")
    print(f"  Separation:                         {avg_fin_sim - avg_fin_nat:+.4f}")
    print()

if nature_sents:
    avg_nat_fin = sum(sf for sf, _ in nature_sents) / len(nature_sents)
    avg_nat_nat = sum(sn for _, sn in nature_sents) / len(nature_sents)
    print(f"Nature sentences (bank = river edge):")
    print(f"  Avg similarity to financial domain: {avg_nat_fin:.4f}")
    print(f"  Avg similarity to nature domain:    {avg_nat_nat:.4f}")
    print(f"  Separation:                         {avg_nat_nat - avg_nat_fin:+.4f}")
    print()

print("Context-free: 'bank' is stuck between both meanings.")
print("Context-aware: each 'bank' leans toward its intended meaning.")
print()
print("This simple windowed averaging is crude -- it weights all neighbors")
print("equally and uses a fixed window size. Attention mechanisms improve")
print("on this by learning WHICH context words matter most for each token.")
print()

# ============================================================
print("=" * 65)
print("PART 6: MORE POLYSEMOUS WORDS")
print("=" * 65)
print()

polysemy_examples = [
    ("bat",
     ["The bat flew out of the cave at dusk",
      "He swung the bat and hit the baseball",
      "The cricket bat was made of willow wood"],
     "animal vs sports equipment"),
    ("spring",
     ["Flowers bloom in the spring sunshine",
      "The spring in the mattress broke last night",
      "Water bubbled up from the natural spring"],
     "season vs coil vs water source"),
    ("light",
     ["Turn on the light so I can read",
      "The bag was light enough to carry",
      "We will light the candles for dinner"],
     "illumination vs weight vs ignite"),
]

for word, sentences, description in polysemy_examples:
    print(f"  Word: '{word}' ({description})")
    print()
    for sent in sentences:
        tokens = tokenize(sent)
        if word in tokens:
            pos = tokens.index(word)
            start = max(0, pos - CTX_WINDOW)
            end = min(len(tokens), pos + CTX_WINDOW + 1)
            context = tokens[start:end]
            print(f"    \"{sent}\"")
            print(f"      Context window: {context}")
        else:
            print(f"    \"{sent}\"")
            print(f"      ('{word}' not found after tokenization)")
    print()

print("Each occurrence of the same word has different neighbors.")
print("A context-aware system uses those neighbors to resolve meaning.")
print("A context-free system gives all occurrences the same vector.")
print()
print("This is precisely why the field moved from static word embeddings")
print("to contextual models -- and why attention was invented.")
```

---

## Key Takeaways

- **Context-free representations assign one vector per word type.** The word "bank" gets the same representation whether it means a financial institution or a river's edge. This forces the vector to be a compromise that is accurate for no single meaning.
- **Context-aware representations give each word token its own vector.** By incorporating information from surrounding words, the same word type can have different representations in different sentences, capturing polysemy and disambiguation.
- **Windowed context averaging is a simple but effective demonstration.** Averaging a word's vector with its neighbors' vectors is crude -- it weights all neighbors equally and ignores long-range context -- but it already improves disambiguation over context-free methods.
- **Attention mechanisms are the principled evolution of this idea.** Instead of averaging all neighbors equally, attention learns which context words are most relevant for each token. This is the key innovation that makes transformers powerful for contextual understanding.
