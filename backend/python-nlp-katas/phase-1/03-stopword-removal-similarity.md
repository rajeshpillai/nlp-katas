# Measure How Stopword Removal Changes Document Similarity

> Phase 1 — Text Preprocessing | Kata 1.3

---

## Concept & Intuition

### What problem are we solving?

Some words appear in almost every English sentence: "the", "is", "at", "on", "a", "and". These are called **stopwords**. They're grammatically necessary but often carry little meaning for tasks like search and classification.

Removing stopwords seems obvious — less noise, better results. But it's not that simple. Stopwords can carry meaning ("to be or not to be" loses everything if you remove stopwords), and removing them changes how similar two documents appear to be. This kata measures that change directly.

### Why naive approaches fail

- Removing all stopwords from "the president" gives "president" — fine
- Removing all stopwords from "not good" gives "good" — meaning is **inverted**
- Removing all stopwords from "to be or not to be" gives "" — the entire sentence is destroyed
- "The Who" (band name) becomes "Who" or nothing

Stopword lists are blunt instruments. The same word can be a stopword in one context and critical in another. "It" is usually meaningless, but in "It was the best of times" it's a stylistic choice; in "IT department" it's an acronym.

### Mental models

- **Signal-to-noise ratio**: Stopwords are high-frequency, low-information words. Removing them is like turning down the background noise — the signal (content words) becomes clearer. But turn down too much and you lose the music.
- **Document fingerprint**: Without stopwords, documents are reduced to their content words. Two documents about the same topic will have more overlapping words, making similarity easier to detect.
- **The 80/20 of language**: Roughly 50% of any English text is made up of about 100 common words. Removing them cuts the text in half while preserving most of the topical content.

### Visual explanations

```
Original:      "The cat sat on the mat and the dog sat on the rug"
Content words:  "cat sat mat dog sat rug"

Original:      "The cat is on the mat and the dog is on the rug"
Content words:  "cat mat dog rug"

Without stopwords, both sentences clearly share a topic (animals + furniture).
With stopwords, the shared words (the, on, the, and, the) dominate but add
no topical information.

Word frequency in a typical document:
  the  ████████████████████████████  (most frequent)
  is   ██████████████████
  a    █████████████████
  of   ████████████████
  and  ███████████████
  ---  (stopword boundary)  ---
  cat  ███
  sat  ██
  mat  ██
  dog  █
  rug  █

The top words tell you nothing about the topic.
The bottom words tell you everything.
```

---

## Hands-on Exploration

1. Take two paragraphs about the same topic — count overlapping words with and without stopwords
2. Find a sentence where removing stopwords changes or destroys the meaning
3. Compare the word frequency distribution before and after stopword removal

---

## Live Code

```python
# --- How stopword removal changes document similarity ---

import re
from collections import Counter

# A curated stopword list (subset of common English stopwords)
STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were',
    'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
    'will', 'would', 'could', 'should', 'may', 'might', 'shall', 'can',
    'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as',
    'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'it', 'its', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
    'she', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your',
    'his', 'our', 'their', 'what', 'which', 'who', 'when', 'where', 'how',
    'not', 'no', 'nor', 'so', 'if', 'then', 'than', 'too', 'very',
    'just', 'about', 'up', 'out', 'also', 'each', 'all', 'both', 'only',
}

def tokenize(text):
    """Split text into lowercase words."""
    return re.findall(r'[a-z]+', text.lower())

def remove_stopwords(words):
    """Filter out stopwords from a word list."""
    return [w for w in words if w not in STOPWORDS]

def word_overlap_similarity(words1, words2):
    """Compute Jaccard similarity: |intersection| / |union| of word sets."""
    set1, set2 = set(words1), set(words2)
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    intersection = set1 & set2
    union = set1 | set2
    return len(intersection) / len(union)


# === Document pairs to compare ===

doc_pairs = [
    (
        "The cat sat on the mat in the living room",
        "A cat was sitting on a mat in a room",
        "Same topic, different function words",
    ),
    (
        "The stock market crashed and investors lost millions of dollars",
        "The financial market experienced a crash and investors suffered large losses",
        "Same topic, partially overlapping vocabulary",
    ),
    (
        "The quick brown fox jumps over the lazy dog",
        "The weather forecast predicts rain for the entire week",
        "Completely different topics",
    ),
    (
        "The president announced the new policy at the conference",
        "The CEO announced the new strategy at the meeting",
        "Similar structure, different domain",
    ),
]

print("=== Stopword Removal and Document Similarity ===\n")

chart_labels = []
sims_with = []
sims_without = []

for doc1, doc2, description in doc_pairs:
    words1 = tokenize(doc1)
    words2 = tokenize(doc2)
    content1 = remove_stopwords(words1)
    content2 = remove_stopwords(words2)

    sim_with = word_overlap_similarity(words1, words2)
    sim_without = word_overlap_similarity(content1, content2)

    chart_labels.append(description)
    sims_with.append(round(sim_with, 3))
    sims_without.append(round(sim_without, 3))

    print(f"--- {description} ---")
    print(f"  Doc 1: '{doc1}'")
    print(f"  Doc 2: '{doc2}'")
    print(f"  With stopwords:    {len(words1)} + {len(words2)} words, "
          f"similarity = {sim_with:.3f}")
    print(f"  Without stopwords: {len(content1)} + {len(content2)} words, "
          f"similarity = {sim_without:.3f}")
    change = sim_without - sim_with
    direction = "increased" if change > 0 else "decreased" if change < 0 else "unchanged"
    print(f"  Change: {change:+.3f} ({direction})")
    print()

show_chart({
    "type": "bar",
    "title": "Similarity: With vs Without Stopwords",
    "labels": chart_labels,
    "datasets": [
        {"label": "With stopwords", "data": sims_with, "color": "#3b82f6"},
        {"label": "Without stopwords", "data": sims_without, "color": "#10b981"},
    ],
    "options": {"x_label": "Document Pair", "y_label": "Jaccard Similarity"}
})


# === Show what stopwords dominate ===

print("\n=== Word Frequency: Before and After ===\n")

text = """The president of the United States announced that the new
economic policy would be implemented by the end of the year. The policy
is expected to have a significant impact on the economy and the job
market. The president also stated that the government would provide
additional support for small businesses and workers who have been
affected by the recent economic downturn."""

words = tokenize(text)
content = remove_stopwords(words)

print(f"Total words: {len(words)}")
print(f"After removing stopwords: {len(content)}")
print(f"Stopwords removed: {len(words) - len(content)} "
      f"({(len(words) - len(content)) / len(words) * 100:.0f}%)")
print()

print("Top 10 words WITH stopwords:")
for word, count in Counter(words).most_common(10):
    bar = "█" * (count * 2)
    is_stop = " (stopword)" if word in STOPWORDS else ""
    print(f"  {word:<12} {count:>3}  {bar}{is_stop}")

print()
print("Top 10 words WITHOUT stopwords:")
for word, count in Counter(content).most_common(10):
    bar = "█" * (count * 2)
    print(f"  {word:<12} {count:>3}  {bar}")


# === Danger cases: when removal destroys meaning ===

print("\n\n=== When Stopword Removal Goes Wrong ===\n")

danger_cases = [
    "To be or not to be",
    "This is not good at all",
    "I will not go",
    "Let it be",
    "It is what it is",
    "The Who released a new album",
]

for text in danger_cases:
    words = tokenize(text)
    content = remove_stopwords(words)
    result = ' '.join(content) if content else "(empty)"
    print(f"  '{text}'")
    print(f"  -> '{result}'")
    print()

print("Stopword removal is not always safe.")
print("Negation words ('not', 'no') carry critical meaning.")
print("Some phrases are MADE of stopwords.")
```

---

## Key Takeaways

- **Stopword removal often improves similarity detection.** By removing high-frequency, low-information words, the remaining content words better represent the document's topic.
- **The effect is not uniform.** For similar documents, stopword removal typically increases measured similarity. For dissimilar documents, the effect varies.
- **Stopwords dominate text.** Roughly 40-60% of tokens in typical English text are stopwords. Removing them dramatically reduces the representation size.
- **Stopword removal can destroy meaning.** Negation words ("not", "no"), phrases made of function words ("to be or not to be"), and some proper nouns ("The Who") are casualties of blind stopword removal. Always consider your task before applying this step.
