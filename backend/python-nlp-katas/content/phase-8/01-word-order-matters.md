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

```python
# --- Word Order Matters: BoW Failure Cases and N-gram Partial Fixes ---
# Only Python stdlib -- no external packages

import math
from collections import Counter


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


def build_bow(tokens):
    """Build a bag-of-words frequency dictionary."""
    return dict(Counter(tokens))


def bow_vector(tokens, vocab):
    """Build a count vector aligned to a vocabulary list."""
    counts = Counter(tokens)
    return [counts.get(w, 0) for w in vocab]


def cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors."""
    dot = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a * a for a in v1))
    mag2 = math.sqrt(sum(b * b for b in v2))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)


def get_ngrams(tokens, n):
    """Extract n-grams from a token list."""
    return ["_".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def ngram_vector(ngrams, vocab):
    """Build a count vector for n-grams aligned to a vocabulary list."""
    counts = Counter(ngrams)
    return [counts.get(ng, 0) for ng in vocab]


# ============================================================
print("=" * 65)
print("PART 1: SENTENCES WITH SAME WORDS, DIFFERENT MEANINGS")
print("=" * 65)

pairs = [
    ("The dog bit the man",
     "The man bit the dog"),
    ("She gave him the book",
     "He gave her the book"),
    ("The teacher failed the student",
     "The student failed the teacher"),
    ("Only I love you",
     "I love only you"),
    ("The cat chased the mouse",
     "The mouse chased the cat"),
]

pair_labels = []
bow_sims = []
for i, (sent_a, sent_b) in enumerate(pairs):
    tokens_a = tokenize(sent_a)
    tokens_b = tokenize(sent_b)

    bow_a = build_bow(tokens_a)
    bow_b = build_bow(tokens_b)

    vocab = sorted(set(tokens_a) | set(tokens_b))
    vec_a = bow_vector(tokens_a, vocab)
    vec_b = bow_vector(tokens_b, vocab)
    sim = cosine_similarity(vec_a, vec_b)

    pair_labels.append(f"Pair {i + 1}")
    bow_sims.append(round(sim, 4))

    print(f"\nPair {i + 1}:")
    print(f"  A: \"{sent_a}\"")
    print(f"  B: \"{sent_b}\"")
    print(f"  BoW A: {bow_a}")
    print(f"  BoW B: {bow_b}")
    print(f"  Vectors identical: {vec_a == vec_b}")
    print(f"  Cosine similarity: {sim:.4f}")
    if sim > 0.9999:
        print(f"  --> BoW CANNOT distinguish these sentences!")

print()

# ============================================================
print("=" * 65)
print("PART 2: N-GRAMS AS A PARTIAL FIX")
print("=" * 65)

sent_a = "The dog bit the man"
sent_b = "The man bit the dog"

tokens_a = tokenize(sent_a)
tokens_b = tokenize(sent_b)

ngram_labels = []
ngram_sims = []
for n in [1, 2, 3]:
    label = {1: "Unigrams (BoW)", 2: "Bigrams", 3: "Trigrams"}[n]
    ngrams_a = get_ngrams(tokens_a, n)
    ngrams_b = get_ngrams(tokens_b, n)

    ng_vocab = sorted(set(ngrams_a) | set(ngrams_b))
    vec_a = ngram_vector(ngrams_a, ng_vocab)
    vec_b = ngram_vector(ngrams_b, ng_vocab)
    sim = cosine_similarity(vec_a, vec_b)

    ngram_labels.append(label)
    ngram_sims.append(round(sim, 4))

    print(f"\n  {label}:")
    print(f"    A: {ngrams_a}")
    print(f"    B: {ngrams_b}")
    print(f"    Shared: {sorted(set(ngrams_a) & set(ngrams_b))}")
    print(f"    Only in A: {sorted(set(ngrams_a) - set(ngrams_b))}")
    print(f"    Only in B: {sorted(set(ngrams_b) - set(ngrams_a))}")
    print(f"    Cosine similarity: {sim:.4f}")
    if sim > 0.9999:
        print(f"    --> Still identical! {label} cannot help here.")
    else:
        print(f"    --> Different! {label} capture some word order.")

print()

# --- Visualization: BoW vs N-gram Similarity ---
show_chart({
    "type": "bar",
    "title": "BoW vs N-gram Similarity: 'The dog bit the man' vs 'The man bit the dog'",
    "labels": ngram_labels,
    "datasets": [{"label": "Cosine Similarity", "data": ngram_sims, "color": "#3b82f6"}],
    "options": {"x_label": "Representation", "y_label": "Cosine Similarity"}
})

# --- Visualization: All Pairs BoW Similarity (always ~1.0) ---
show_chart({
    "type": "bar",
    "title": "BoW Similarity Across Word-Order Pairs (All ~1.0)",
    "labels": pair_labels,
    "datasets": [{"label": "BoW Cosine Similarity", "data": bow_sims, "color": "#ef4444"}],
    "options": {"x_label": "Sentence Pair", "y_label": "Cosine Similarity"}
})

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(ngram_labels, ngram_sims, color=["#ef4444", "#f59e0b", "#10b981"])
    ax1.set_ylabel("Cosine Similarity")
    ax1.set_title("BoW vs N-gram Similarity")
    ax1.set_ylim(0, 1.15)
    for i, v in enumerate(ngram_sims):
        ax1.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)

    ax2.bar(pair_labels, bow_sims, color="#ef4444")
    ax2.set_ylabel("Cosine Similarity")
    ax2.set_title("BoW Similarity Across Pairs (Always ~1.0)")
    ax2.set_ylim(0, 1.15)

    plt.tight_layout()
    plt.savefig("bow_vs_ngram.png", dpi=100)
    plt.close()
    print("  [Saved matplotlib chart: bow_vs_ngram.png]")
except ImportError:
    pass

# ============================================================
print("=" * 65)
print("PART 3: QUANTIFYING INFORMATION LOSS")
print("=" * 65)
print()
print("How much meaning does word order carry?")
print("Let's take one sentence and count distinct meanings from reorderings.")
print()

# Using a carefully chosen sentence where reorderings create new meanings
base_tokens = ["only", "she", "told", "him", "that", "she", "loved", "him"]
sentence = " ".join(base_tokens)
print(f"Original: \"{sentence}\"")
print()

# Show specific reorderings that change meaning
reorderings = [
    (["only", "she", "told", "him", "that", "she", "loved", "him"],
     "Only SHE told him (nobody else told him)"),
    (["she", "only", "told", "him", "that", "she", "loved", "him"],
     "She ONLY told him (she didn't show it)"),
    (["she", "told", "only", "him", "that", "she", "loved", "him"],
     "She told ONLY him (nobody else was told)"),
    (["she", "told", "him", "only", "that", "she", "loved", "him"],
     "She told him only THAT (nothing else)"),
    (["she", "told", "him", "that", "she", "loved", "only", "him"],
     "She loved ONLY him (nobody else)"),
    (["she", "told", "him", "that", "only", "she", "loved", "him"],
     "Only SHE loved him (nobody else did)"),
]

base_bow = build_bow(base_tokens)
print(f"BoW for ALL of these: {dict(sorted(base_bow.items()))}")
print()

for tokens, meaning in reorderings:
    this_bow = build_bow(tokens)
    same = this_bow == base_bow
    print(f"  \"{' '.join(tokens)}\"")
    print(f"    Meaning: {meaning}")
    print(f"    Same BoW: {same}")
    print()

print(f"  6 different meanings, ALL with identical BoW vectors.")
print(f"  Word order carries enormous semantic information that BoW discards.")
print()

# ============================================================
print("=" * 65)
print("PART 4: THE VOCABULARY EXPLOSION PROBLEM WITH N-GRAMS")
print("=" * 65)
print()

# Simulate vocabulary growth
vocab_sizes = [100, 500, 1000, 5000, 10000, 50000]
print(f"  {'Vocab Size':>12}  {'Unigrams':>12}  {'Bigrams':>14}  {'Trigrams':>16}")
print(f"  {'-' * 12}  {'-' * 12}  {'-' * 14}  {'-' * 16}")

for v in vocab_sizes:
    uni = v
    bi = v * v
    tri = v * v * v
    print(f"  {v:>12,}  {uni:>12,}  {bi:>14,}  {tri:>16,}")

print()
print("  As n-gram order increases, possible features grow EXPONENTIALLY.")
print("  Most n-grams will never appear in any document -> extreme sparsity.")
print("  And n-grams still cannot capture dependencies beyond n words apart.")
print()

# ============================================================
print("=" * 65)
print("PART 5: LONG-RANGE DEPENDENCIES DEFEAT N-GRAMS")
print("=" * 65)
print()

# Show that important word relationships can span many words
examples = [
    ("The cat sat on the mat",
     "cat", "sat", 1,
     "Subject and verb are adjacent"),
    ("The cat that I saw yesterday at the park sat on the mat",
     "cat", "sat", 8,
     "Subject and verb separated by a relative clause"),
    ("The cat which my neighbor who lives next door adopted last year sat on the mat",
     "cat", "sat", 11,
     "Subject and verb separated by nested relative clauses"),
]

print("How far apart can related words be?\n")

for sent, word1, word2, distance, description in examples:
    tokens = tokenize(sent)
    idx1 = tokens.index(word1)
    idx2 = tokens.index(word2)
    actual_dist = idx2 - idx1

    # Check which n-gram would capture this
    needed_n = actual_dist + 1

    print(f"  \"{sent}\"")
    print(f"    '{word1}' at position {idx1}, '{word2}' at position {idx2}")
    print(f"    Distance: {actual_dist} words apart")
    print(f"    N-gram needed to capture this: {needed_n}-gram")
    print(f"    Note: {description}")

    # Show what bigrams and trigrams can see
    bigrams = get_ngrams(tokens, 2)
    trigrams = get_ngrams(tokens, 3)
    has_pair_bigram = any(word1 in bg and word2 in bg for bg in bigrams)
    has_pair_trigram = any(word1 in tg and word2 in tg for tg in trigrams)
    print(f"    Bigrams capture '{word1}'-'{word2}' link: {has_pair_bigram}")
    print(f"    Trigrams capture '{word1}'-'{word2}' link: {has_pair_trigram}")
    print()

print("  As sentences get more complex, important relationships span")
print("  more words. No fixed n-gram size can handle all cases.")
print("  This is why we need models that process SEQUENCES, not bags.")
print()

# ============================================================
print("=" * 65)
print("PART 6: NEGATION -- BOW'S WORST ENEMY")
print("=" * 65)
print()

sentiment_pairs = [
    ("This movie is good",
     "This movie is not good"),
    ("I would recommend this film",
     "I would not recommend this film"),
    ("The acting was excellent and the plot was never boring",
     "The acting was never excellent and the plot was boring"),
]

for sent_a, sent_b in sentiment_pairs:
    tokens_a = tokenize(sent_a)
    tokens_b = tokenize(sent_b)
    vocab = sorted(set(tokens_a) | set(tokens_b))
    vec_a = bow_vector(tokens_a, vocab)
    vec_b = bow_vector(tokens_b, vocab)
    sim = cosine_similarity(vec_a, vec_b)

    print(f"  A: \"{sent_a}\"")
    print(f"  B: \"{sent_b}\"")
    print(f"  Cosine similarity: {sim:.4f}")

    # Show what differs
    only_a = set(tokens_a) - set(tokens_b)
    only_b = set(tokens_b) - set(tokens_a)
    if only_a:
        print(f"  Only in A: {sorted(only_a)}")
    if only_b:
        print(f"  Only in B: {sorted(only_b)}")
    if not only_a and not only_b:
        print(f"  --> Same words, different order: BoW is completely blind!")
    print()

print("  Negation words like 'not' and 'never' flip the meaning of")
print("  entire phrases. BoW either ignores them (same words) or treats")
print("  'not' as just another word with a small IDF weight.")
print("  Understanding negation requires understanding SEQUENCE.")
```

---

## Key Takeaways

- **Word order is meaning, not noise.** Sentences with identical words in different orders can describe completely opposite events. BoW and TF-IDF are mathematically incapable of distinguishing them because they are permutation-invariant representations.
- **N-grams are a partial fix with exponential cost.** Bigrams and trigrams capture local word order and can distinguish some reorderings, but the vocabulary grows exponentially (V^n), most n-grams are vanishingly sparse, and long-range dependencies remain invisible.
- **Long-range dependencies are everywhere in natural language.** Subjects and verbs, pronouns and their referents, negation and the words being negated -- these relationships can span many words, far beyond what any practical n-gram can capture.
- **Context changes meaning, and models must account for it.** The failure of orderless representations motivates sequence models -- architectures that read text in order and maintain a running understanding of what has been said so far.
