# Compute TF-IDF Manually

> Phase 3 — TF-IDF | Kata 3.1

---

## Concept & Intuition

### What problem are we solving?

In Phase 2, we built Bag of Words (BoW) representations — raw counts of how often each word appears in a document. But raw counts treat all words equally. The word "the" appearing 10 times in a document does not make that document about "the." Meanwhile, the word "photosynthesis" appearing once is far more informative about the document's topic.

TF-IDF (Term Frequency — Inverse Document Frequency) solves this by asking two questions simultaneously:

1. **How often does this word appear in this document?** (Term Frequency)
2. **How rare is this word across all documents?** (Inverse Document Frequency)

Words that appear frequently in one document but rarely across the corpus get high scores. Words that appear everywhere get suppressed. This gives us a measure of **relevance**, not just presence.

### Why naive/earlier approaches fail

Raw word counts (BoW) have a fundamental flaw: **common words dominate the representation**.

Consider three documents about different topics. The words "the", "is", "and", "of" will have the highest counts in all three documents. When you compare these documents using raw counts, the similarity is driven by these shared function words, not by the topic-specific content words.

Stopword removal helps partially, but it requires a manually curated list and is language-specific. TF-IDF solves this more elegantly — it lets the corpus itself determine which words are informative.

### Mental models

- **TF-IDF as a signal-to-noise filter**: TF is the signal (this word matters in this document), IDF is the noise reduction (discount words that appear everywhere). Multiply them together and you get a cleaned-up importance score.
- **Rarity as information**: A word that appears in every document tells you nothing about any specific document. A word that appears in only one document tells you a lot. IDF quantifies this intuition using logarithms.
- **Three regimes of IDF**: A word in 1 of 100 documents has high IDF. A word in 50 of 100 has moderate IDF. A word in all 100 has IDF of zero — it carries no discriminative power.

### Visual explanations

```
THE TF-IDF FORMULA
==================

  TF-IDF(t, d) = TF(t, d) x IDF(t)

Where:
  TF(t, d)  = count of term t in document d / total terms in document d
  IDF(t)    = log( total number of documents / number of documents containing t )


EXAMPLE: Corpus of 3 documents
-------------------------------

  Doc 1: "the cat sat on the mat"
  Doc 2: "the dog sat on the rug"
  Doc 3: "the cat chased the dog"

  Word        Doc1  Doc2  Doc3  | Docs containing | IDF = log(3/n)
  --------    ----  ----  ----  | --------------- | --------------
  "the"        2     2     2   |        3         | log(3/3) = 0.00
  "cat"        1     0     1   |        2         | log(3/2) = 0.41
  "sat"        1     1     0   |        2         | log(3/2) = 0.41
  "on"         1     1     0   |        2         | log(3/2) = 0.41
  "mat"        1     0     0   |        1         | log(3/1) = 1.10
  "dog"        0     1     1   |        2         | log(3/2) = 0.41
  "rug"        0     1     0   |        1         | log(3/1) = 1.10
  "chased"     0     0     1   |        1         | log(3/1) = 1.10

  Notice: "the" gets IDF = 0 — it appears everywhere, so it's worthless.
          "mat", "rug", "chased" get high IDF — they're unique to one document.

TF-IDF FOR "cat" IN DOC 1:
  TF  = 1/6 = 0.167
  IDF = log(3/2) = 0.405
  TF-IDF = 0.167 x 0.405 = 0.068

TF-IDF FOR "the" IN DOC 1:
  TF  = 2/6 = 0.333
  IDF = log(3/3) = 0.000
  TF-IDF = 0.333 x 0.000 = 0.000   <-- suppressed despite high frequency!
```

```
WHY LOGARITHM IN IDF?
======================

Without log:    IDF("mat") = 3/1 = 3.0     IDF("cat") = 3/2 = 1.5
With log:       IDF("mat") = log(3/1) = 1.1  IDF("cat") = log(3/2) = 0.4

The logarithm compresses the scale. Without it, a word appearing in 1 of
1,000,000 documents would get IDF = 1,000,000 — an absurd weight. The log
keeps values manageable:

  log(1,000,000 / 1) = 13.8     (high, but not absurd)
  log(1,000,000 / 500,000) = 0.7  (moderate)
  log(1,000,000 / 1,000,000) = 0  (zero — as expected)

  Documents     Raw ratio     log(ratio)
  containing    (N/df)        ln(N/df)
  ----------    ---------     ----------
       1        1000000.0       13.82
      10         100000.0       11.51
     100          10000.0        9.21
    1000           1000.0        6.91
   10000            100.0        4.61
  100000             10.0        2.30
  500000              2.0        0.69
 1000000              1.0        0.00
```

---

## Hands-on Exploration

1. Take the three-document corpus shown above ("the cat sat on the mat", "the dog sat on the rug", "the cat chased the dog"). Compute by hand: TF for every word in Document 2, then IDF for every word in the vocabulary, then multiply to get TF-IDF. Which word has the highest TF-IDF in Document 2? Why does that make intuitive sense?

2. Create a fourth document that contains only common words: "the the the on on". Compute its TF-IDF vector. What happens? What does this tell you about documents composed entirely of frequent words?

3. Consider the word "cat" which appears in Doc 1 and Doc 3. Its IDF is the same regardless of which document we're computing TF-IDF for. Now imagine adding 10 more documents that all mention "cat". How would this change the IDF of "cat"? At what point does "cat" become as useless as "the" for distinguishing documents?

---

## Live Code

```python
# --- Computing TF-IDF from scratch using only Python stdlib ---
import math

# Our small corpus
corpus = [
    "the cat sat on the mat",
    "the dog sat on the rug",
    "the cat chased the dog",
]

# Step 1: Tokenize each document into words
documents = [doc.lower().split() for doc in corpus]

print("=== CORPUS ===\n")
for i, doc in enumerate(documents):
    print(f"  Doc {i}: {doc}")

# Step 2: Build vocabulary (all unique words across corpus)
vocabulary = sorted(set(word for doc in documents for word in doc))
print(f"\nVocabulary ({len(vocabulary)} words): {vocabulary}\n")

# Step 3: Compute Term Frequency (TF)
# TF(t, d) = count of t in d / total words in d

def compute_tf(document):
    """Compute term frequency for each word in a document."""
    word_count = len(document)
    tf = {}
    for word in document:
        tf[word] = tf.get(word, 0) + 1
    # Normalize by document length
    for word in tf:
        tf[word] = tf[word] / word_count
    return tf

print("=== TERM FREQUENCY (TF) ===\n")
tf_scores = []
for i, doc in enumerate(documents):
    tf = compute_tf(doc)
    tf_scores.append(tf)
    print(f"  Doc {i} ({len(doc)} words):")
    for word in vocabulary:
        score = tf.get(word, 0)
        if score > 0:
            count = sum(1 for w in doc if w == word)
            print(f"    {word:>10s}: count={count}, TF={score:.4f}")
    print()

# Step 4: Compute Inverse Document Frequency (IDF)
# IDF(t) = log(N / df(t))
# where N = total documents, df(t) = documents containing term t

def compute_idf(documents, vocabulary):
    """Compute IDF for each word in the vocabulary."""
    n = len(documents)
    idf = {}
    for word in vocabulary:
        # Count how many documents contain this word
        doc_freq = sum(1 for doc in documents if word in doc)
        idf[word] = math.log(n / doc_freq)
    return idf

idf_scores = compute_idf(documents, vocabulary)

print("=== INVERSE DOCUMENT FREQUENCY (IDF) ===\n")
print(f"  Total documents (N): {len(documents)}\n")
for word in vocabulary:
    doc_freq = sum(1 for doc in documents if word in doc)
    print(f"  {word:>10s}: appears in {doc_freq}/{len(documents)} docs,"
          f"  IDF = log({len(documents)}/{doc_freq}) = {idf_scores[word]:.4f}")

# Step 5: Compute TF-IDF = TF * IDF
print("\n=== TF-IDF SCORES ===\n")

tfidf_all = []
for i, doc in enumerate(documents):
    print(f"  Doc {i}: \"{corpus[i]}\"")
    tfidf = {}
    for word in vocabulary:
        tf_val = tf_scores[i].get(word, 0)
        idf_val = idf_scores[word]
        tfidf[word] = tf_val * idf_val
    tfidf_all.append(tfidf)

    # Show non-zero scores sorted by value (descending)
    nonzero = {w: s for w, s in tfidf.items() if s > 0}
    for word, score in sorted(nonzero.items(), key=lambda x: -x[1]):
        tf_val = tf_scores[i].get(word, 0)
        print(f"    {word:>10s}: TF={tf_val:.4f} x IDF={idf_scores[word]:.4f}"
              f" = TF-IDF={score:.4f}")

    # Show suppressed words
    zeros = [w for w, s in tfidf.items() if s == 0 and w in doc]
    if zeros:
        print(f"    {'[suppressed]':>10s}: {', '.join(zeros)}"
              f" (IDF=0, appear in all docs)")
    print()

    # Visualize top TF-IDF words for this document
    nonzero_sorted = sorted(
        [(w, s) for w, s in tfidf.items() if s > 0],
        key=lambda x: -x[1]
    )
    if nonzero_sorted:
        chart_labels = [w for w, s in nonzero_sorted]
        chart_data = [round(s, 4) for w, s in nonzero_sorted]
        colors = ["#3b82f6", "#10b981", "#f59e0b"]
        show_chart({
            "type": "bar",
            "title": f"TF-IDF Scores — Doc {i}: \"{corpus[i][:30]}...\"",
            "labels": chart_labels,
            "datasets": [{"label": "TF-IDF", "data": chart_data, "color": colors[i % 3]}],
            "options": {"x_label": "Word", "y_label": "TF-IDF Score"}
        })

# Step 6: Show the full document-term matrix
print("=== DOCUMENT-TERM MATRIX (TF-IDF) ===\n")

header = "         " + "".join(f"{w:>10s}" for w in vocabulary)
print(header)
print("         " + "-" * (10 * len(vocabulary)))
for i, tfidf in enumerate(tfidf_all):
    row = f"  Doc {i}  "
    for word in vocabulary:
        score = tfidf[word]
        row += f"{score:10.4f}"
    print(row)

print("\n=== KEY OBSERVATIONS ===\n")
print('  "the" has TF-IDF = 0 everywhere — it appears in ALL documents.')
print('  "mat", "rug", "chased" have the highest scores — they are UNIQUE.')
print('  TF-IDF automatically identifies what makes each document distinctive.')

# Visualize IDF values across the vocabulary
idf_sorted_pairs = sorted(idf_scores.items(), key=lambda x: -x[1])
show_chart({
    "type": "bar",
    "title": "IDF Values Across Vocabulary",
    "labels": [w for w, s in idf_sorted_pairs],
    "datasets": [{
        "label": "IDF",
        "data": [round(s, 4) for w, s in idf_sorted_pairs],
        "color": "#8b5cf6"
    }],
    "options": {"x_label": "Word", "y_label": "IDF Score"}
})
```

---

## Key Takeaways

- **TF-IDF combines two signals**: term frequency (local importance within a document) and inverse document frequency (global rarity across the corpus). Neither signal alone is sufficient.
- **Common words are automatically suppressed.** Words appearing in every document get IDF = 0, which zeroes out their TF-IDF score — no manual stopword list needed.
- **The logarithm in IDF is essential.** Without it, rare words would receive astronomically large weights that would dominate all other terms. The log compresses the scale to keep scores balanced.
- **TF-IDF is a relevance heuristic, not "understanding."** It measures statistical distinctiveness, not meaning. A word can have high TF-IDF simply because it is rare, even if it is not semantically important.
