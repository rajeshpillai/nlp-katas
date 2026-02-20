# Visualize Word Importance

> Phase 3 — TF-IDF | Kata 3.3

---

## Concept & Intuition

### What problem are we solving?

TF-IDF assigns a numerical score to every word in every document. But a table of numbers is hard to interpret. Which words does TF-IDF consider "important" for each document? Which words get suppressed? How do the importance rankings differ across documents in the same corpus?

Visualization makes the invisible visible. By plotting word importance as bar charts, we can immediately see the TF-IDF weighting at work: common words shrink to nothing, rare topic-specific words rise to the top, and each document gets a distinct "importance fingerprint."

This kata is about building intuition for what TF-IDF actually does — not through formulas, but through visual inspection of its output.

### Why naive/earlier approaches fail

Looking at raw TF-IDF numbers in a table is like reading a spreadsheet of pixel values instead of looking at an image. You can verify any individual number, but you cannot see the pattern. Humans are visual creatures — we spot trends, outliers, and clusters far faster in a chart than in a grid of decimals.

Without visualization:
- You might not notice that "the" gets a score of exactly 0 in every document
- You might miss that the top-ranked word in each document is the one that makes it unique
- You might not see how similar two documents' importance profiles are

### Mental models

- **TF-IDF as a spotlight**: Imagine each document on a dark stage. TF-IDF shines a spotlight on the words that make that document distinctive. Common words stay in shadow. Unique words glow brightly. The visualization shows you where the light falls.
- **Importance fingerprint**: Each document has a unique pattern of word importance scores. Like a fingerprint, this pattern identifies the document. Two documents about the same topic will have similar fingerprints. Visualization lets you compare fingerprints at a glance.
- **The long tail of language**: In any corpus, a few words appear everywhere (and get low TF-IDF), while most words appear rarely (and get high TF-IDF). Visualizing this distribution reveals the "long tail" of language — most of the vocabulary is rare, and rare words carry the most information.

### Visual explanations

```
WORD IMPORTANCE RANKING (what TF-IDF reveals)
==============================================

Document: "Solar energy powers millions of homes across the nation"

  Raw counts (BoW):                TF-IDF scores:
  ----------------                 ---------------
  the      ####  (common)          solar    ########  (rare, important)
  of       ###   (common)          energy   #######   (rare, important)
  solar    ##    (rare)            powers   ######    (rare, important)
  energy   ##    (rare)            homes    #####     (somewhat rare)
  powers   ##    (rare)            nation   #####     (somewhat rare)
  homes    ##    (rare)            millions ####      (somewhat rare)
  millions ##    (rare)            across   ##        (moderate)
  across   ##    (moderate)        of       #         (common, suppressed)
  nation   ##    (rare)            the               (ubiquitous, ZERO)

  BoW says "the" is the most          TF-IDF says "solar" and "energy"
  important word. That's useless.     are most important. That's useful.
```

```
COMPARING IMPORTANCE ACROSS DOCUMENTS
======================================

Doc 0 (about cooking):           Doc 1 (about astronomy):
  recipe   ########               telescope  #########
  flour    #######                 orbit      ########
  bake     ######                  planet     #######
  oven     #####                   galaxy     ######
  minutes  ####                    light      #####
  the                              the

Each document has its own "importance fingerprint."
The word "the" is invisible in both — TF-IDF killed it.
The top words immediately tell you what each document is about.
```

```
ASCII BAR CHART ANATOMY
========================

  word      |  score  |  bar
  ----------+---------+------------------------------
  recipe    |  0.231  |  ========================
  flour     |  0.198  |  ====================
  bake      |  0.154  |  ================
  ...       |  ...    |  ...
  the       |  0.000  |  (nothing — zero score)

  Bar length = score / max_score * bar_width
  This normalizes all bars relative to the top word.
```

---

## Hands-on Exploration

1. Take a document you care about (a favorite quote, a paragraph from a book, a news headline). Imagine placing it into a corpus of 100 random documents. Which words do you predict TF-IDF would rank highest? Which words would get suppressed? Write down your predictions, then check against the code output below.

2. Look at the output of the code and find two documents whose "importance fingerprints" overlap (they share high-scoring words). What does this tell you about their topical similarity? Now find two documents with completely different top words. How different are their topics?

3. Consider a failure case: a document containing a rare typo like "reciepe" instead of "recipe". TF-IDF will give "reciepe" a very high score because it appears in only one document. Is this word truly important? What does this reveal about the limitations of using rarity as a proxy for importance?

---

## Live Code

```python
# --- Visualizing word importance with TF-IDF (ASCII bar charts) ---
import math

# A small corpus with distinct topics
corpus = [
    "the chef prepared a delicious pasta recipe with fresh tomato sauce",
    "the astronomer observed a distant galaxy through the new telescope",
    "the chef baked fresh bread and prepared a tomato basil soup recipe",
    "the telescope captured images of a distant planet orbiting a star",
    "the recipe called for fresh herbs and a pinch of sea salt",
]

print("=== CORPUS ===\n")
for i, doc in enumerate(corpus):
    print(f"  Doc {i}: \"{doc}\"")
print()

# Tokenize
documents = [doc.lower().split() for doc in corpus]
vocabulary = sorted(set(word for doc in documents for word in doc))


# --- Compute TF-IDF ---

def compute_tf(doc):
    total = len(doc)
    counts = {}
    for word in doc:
        counts[word] = counts.get(word, 0) + 1
    return {word: count / total for word, count in counts.items()}

def compute_idf(documents, vocab):
    n = len(documents)
    idf = {}
    for word in vocab:
        doc_freq = sum(1 for doc in documents if word in doc)
        idf[word] = math.log(n / doc_freq) if doc_freq > 0 else 0
    return idf

idf = compute_idf(documents, vocabulary)


# --- ASCII bar chart function ---

def ascii_bar(value, max_value, width=40):
    """Create an ASCII bar of proportional length."""
    if max_value == 0:
        return ""
    length = int((value / max_value) * width)
    return "#" * length


# --- Visualize word importance for each document ---

print("=" * 70)
print("  WORD IMPORTANCE BY DOCUMENT (TF-IDF)")
print("=" * 70)

for i, doc in enumerate(documents):
    tf = compute_tf(doc)
    tfidf = {}
    for word in vocabulary:
        tfidf[word] = tf.get(word, 0) * idf[word]

    # Sort by TF-IDF score descending
    ranked = sorted(tfidf.items(), key=lambda x: -x[1])

    # Find max score for scaling bars
    max_score = ranked[0][1] if ranked[0][1] > 0 else 1

    print(f"\n  Doc {i}: \"{corpus[i]}\"")
    print(f"  {'-' * 64}")

    for word, score in ranked:
        # Only show words that appear in this document or have non-zero score
        if word in doc or score > 0:
            if word not in doc:
                continue
            bar = ascii_bar(score, max_score)
            if score == 0:
                print(f"    {word:>14s} | {score:.4f} |  (suppressed)")
            else:
                print(f"    {word:>14s} | {score:.4f} | {bar}")

    # Show suppressed words explicitly
    suppressed = [w for w in doc if tfidf.get(w, 0) == 0]
    if suppressed:
        unique_suppressed = sorted(set(suppressed))
        print(f"    {'':>14s} |        |")
        print(f"    {'[zero score]':>14s} | {', '.join(unique_suppressed)}")


# --- Compare top words across documents ---

print("\n" + "=" * 70)
print("  TOP 3 MOST IMPORTANT WORDS PER DOCUMENT")
print("=" * 70 + "\n")

for i, doc in enumerate(documents):
    tf = compute_tf(doc)
    tfidf = {word: tf.get(word, 0) * idf[word] for word in doc}
    top3 = sorted(tfidf.items(), key=lambda x: -x[1])[:3]
    words = ", ".join(f"\"{w}\" ({s:.3f})" for w, s in top3)
    print(f"  Doc {i}: {words}")

print()
print("  Notice how the top words immediately reveal each document's topic.")
print("  Documents 0, 2, 4 share cooking words. Documents 1, 3 share astronomy words.")


# --- Show the IDF spectrum ---

print("\n" + "=" * 70)
print("  IDF SPECTRUM: FROM COMMON TO RARE")
print("=" * 70 + "\n")

idf_sorted = sorted(idf.items(), key=lambda x: x[1])
max_idf = max(idf.values()) if idf.values() else 1

for word, score in idf_sorted:
    doc_freq = sum(1 for doc in documents if word in doc)
    bar = ascii_bar(score, max_idf, width=35)
    freq_label = f"{doc_freq}/{len(documents)} docs"
    print(f"  {word:>14s}  IDF={score:.3f}  ({freq_label})  {bar}")


# --- Show importance distribution within a single document ---

print("\n" + "=" * 70)
print("  IMPORTANCE DISTRIBUTION: DOC 0 (detailed)")
print("=" * 70 + "\n")

doc_idx = 0
doc = documents[doc_idx]
tf = compute_tf(doc)
tfidf = {}
for word in set(doc):
    tfidf[word] = tf[word] * idf[word]

ranked = sorted(tfidf.items(), key=lambda x: -x[1])
max_score = ranked[0][1] if ranked[0][1] > 0 else 1
total_tfidf = sum(tfidf.values())

print(f"  Document: \"{corpus[doc_idx]}\"")
print(f"  Total TF-IDF mass: {total_tfidf:.4f}\n")

cumulative = 0
for word, score in ranked:
    cumulative += score
    pct = (score / total_tfidf * 100) if total_tfidf > 0 else 0
    cum_pct = (cumulative / total_tfidf * 100) if total_tfidf > 0 else 0
    bar = ascii_bar(score, max_score, width=30)
    status = ""
    if score == 0:
        status = " <-- ZERO (appears everywhere)"
    print(f"  {word:>14s}  {score:.4f}  ({pct:5.1f}%)  cumul: {cum_pct:5.1f}%"
          f"  {bar}{status}")

print(f"\n  The top few words carry most of the TF-IDF \"mass\".")
print(f"  Common words contribute nothing. This is by design.")


# --- Side-by-side comparison of two documents ---

print("\n" + "=" * 70)
print("  SIDE-BY-SIDE: COOKING DOC vs ASTRONOMY DOC")
print("=" * 70 + "\n")

def get_tfidf_for_doc(doc_idx):
    doc = documents[doc_idx]
    tf = compute_tf(doc)
    return {word: tf[word] * idf[word] for word in set(doc)}

cooking = get_tfidf_for_doc(0)
astronomy = get_tfidf_for_doc(1)

cooking_ranked = sorted(cooking.items(), key=lambda x: -x[1])
astro_ranked = sorted(astronomy.items(), key=lambda x: -x[1])

max_len = max(len(cooking_ranked), len(astro_ranked))
print(f"  {'Doc 0 (cooking)':^32s}  |  {'Doc 1 (astronomy)':^32s}")
print(f"  {'-'*32}  |  {'-'*32}")

for idx in range(max_len):
    left = ""
    right = ""
    if idx < len(cooking_ranked):
        w, s = cooking_ranked[idx]
        bar = ascii_bar(s, cooking_ranked[0][1], width=14)
        left = f"{w:>10s} {s:.3f} {bar}"
    if idx < len(astro_ranked):
        w, s = astro_ranked[idx]
        bar = ascii_bar(s, astro_ranked[0][1], width=14)
        right = f"{w:>10s} {s:.3f} {bar}"
    print(f"  {left:<32s}  |  {right:<32s}")

print()
print("  Each document has a completely different importance profile.")
print("  The visualization makes topical differences immediately obvious.")
print("  TF-IDF creates a unique \"fingerprint\" for each document's content.")
```

---

## Key Takeaways

- **Visualization reveals what numbers hide.** A table of TF-IDF scores is hard to interpret, but an ASCII bar chart immediately shows which words matter and which are suppressed.
- **Each document has a unique importance fingerprint.** The top-ranked words in each document reveal its topic at a glance. Documents about similar topics share similar fingerprints.
- **Common words vanish under TF-IDF.** Words like "the" and "a" that appear in every document get a score of zero. The visualization makes this suppression obvious and dramatic.
- **TF-IDF is a relevance heuristic, not "understanding."** High TF-IDF does not mean a word is semantically important -- it means the word is statistically distinctive. Typos, jargon, and names all get high scores simply because they are rare, regardless of whether they carry real meaning.
