# Compare Stemming vs Lemmatization Output

> Phase 1 — Text Preprocessing | Kata 1.2

---

## Concept & Intuition

### What problem are we solving?

Words come in many forms: "running", "runs", "ran", "runner". For many NLP tasks, we want to treat these as variations of the same concept. But how do we reduce words to a common root?

There are two approaches:
- **Stemming**: Chop off suffixes using rules. Fast, crude, often produces non-words.
- **Lemmatization**: Use vocabulary and grammar to find the dictionary form. Slower, accurate, always produces real words.

Understanding the difference — and when each is appropriate — is fundamental to NLP preprocessing.

### Why naive approaches fail

You might think: "just strip common endings like -ing, -ed, -s." But:

- "caring" stems to "car" (wrong — dropped too much)
- "university" and "universe" stem to the same root "univers" (wrong — different concepts)
- "better" should reduce to "good" (no suffix rule can do this)
- "was" should reduce to "be" (irregular — needs linguistic knowledge)

Stemming is a blunt instrument. Lemmatization is precise but requires more machinery. Neither is universally correct.

### Mental models

- **Stemming = hedge trimmer**: Cuts aggressively, fast, doesn't care about shape. Good enough for many jobs.
- **Lemmatization = sculptor**: Carefully carves to the correct form. Slower, but the result is a real word.
- **Normalization spectrum**: Raw word → Stem (approximate root) → Lemma (dictionary form). More normalization = less variation, but also less distinction.

### Visual explanations

```
Word: "running"

Stemming (rule-based):
  running → remove "ning"? No.
  running → remove "ing"  → "runn" → "run"
  Result: "run" (happens to be correct!)

Word: "caring"

Stemming (rule-based):
  caring → remove "ing" → "car"
  Result: "car" (WRONG — should be "care")

Lemmatization (vocabulary lookup):
  caring → verb → base form → "care"
  Result: "care" (correct)

Word: "better"

Stemming: "better" → remove "er" → "bett" (nonsense)
Lemmatization: "better" → adjective → base form → "good" (correct)
```

---

## Hands-on Exploration

1. Take a paragraph and stem every word — count how many non-words you get
2. Try the same paragraph with lemmatization — compare the results
3. Find 5 words where stemming and lemmatization give different results
4. Find a case where stemming accidentally merges two unrelated words into the same stem

---

## Live Code

```python
# --- Stemming vs Lemmatization: understanding the tradeoff ---

# We'll implement a simple suffix-stripping stemmer from scratch,
# then compare it with a more sophisticated approach.

import re

# === STEMMING: Rule-based suffix stripping ===

def simple_stem(word):
    """A basic suffix-stripping stemmer.

    Applies rules in order to reduce a word to its approximate root.
    This is a simplified version of the Porter Stemmer algorithm.
    """
    word = word.lower()

    # Rule set (order matters — most specific first)
    suffixes = [
        ('ational', 'ate'),    # relational -> relate
        ('tional', 'tion'),    # conditional -> condition
        ('iveness', 'ive'),    # effectiveness -> effective
        ('fulness', 'ful'),    # hopefulness -> hopeful
        ('ousli', 'ous'),      # dangerously -> dangerous
        ('ation', 'ate'),      # information -> inform... wait, "inforate"?
        ('ness', ''),          # happiness -> happi
        ('ment', ''),          # movement -> move (if long enough)
        ('ling', ''),          # darling -> dar (oops)
        ('ally', ''),          # finally -> fin (oops)
        ('ing', ''),           # running -> runn
        ('ies', 'i'),          # ponies -> poni
        ('ed', ''),            # jumped -> jump
        ('ly', ''),            # slowly -> slow
        ('er', ''),            # runner -> runn
        ('s', ''),             # cats -> cat
    ]

    for suffix, replacement in suffixes:
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[:-len(suffix)] + replacement
            break

    return word


# === LEMMATIZATION: Vocabulary-based normalization ===

# A real lemmatizer uses a dictionary + part-of-speech tags.
# Here we simulate one with a lookup table to show the concept.

LEMMA_TABLE = {
    # Verb forms
    'running': 'run', 'runs': 'run', 'ran': 'run',
    'swimming': 'swim', 'swims': 'swim', 'swam': 'swim',
    'caring': 'care', 'cares': 'care', 'cared': 'care',
    'having': 'have', 'has': 'have', 'had': 'have',
    'doing': 'do', 'does': 'do', 'did': 'do',
    'going': 'go', 'goes': 'go', 'went': 'go', 'gone': 'go',
    'was': 'be', 'were': 'be', 'been': 'be', 'being': 'be',
    'is': 'be', 'are': 'be', 'am': 'be',
    'said': 'say', 'says': 'say', 'saying': 'say',
    'took': 'take', 'takes': 'take', 'taking': 'take', 'taken': 'take',

    # Irregular adjectives
    'better': 'good', 'best': 'good',
    'worse': 'bad', 'worst': 'bad',
    'more': 'much', 'most': 'much',
    'larger': 'large', 'largest': 'large',

    # Noun plurals (irregular)
    'mice': 'mouse', 'geese': 'goose',
    'children': 'child', 'teeth': 'tooth',
    'feet': 'foot', 'men': 'man', 'women': 'woman',
    'people': 'person',

    # Regular forms (a real lemmatizer handles these with rules + dictionary)
    'cats': 'cat', 'dogs': 'dog',
    'jumping': 'jump', 'jumped': 'jump', 'jumps': 'jump',
    'happiness': 'happiness', 'sadly': 'sad',
    'studies': 'study', 'studied': 'study', 'studying': 'study',
    'universities': 'university',
}

def simple_lemmatize(word):
    """Look up the lemma (dictionary form) of a word."""
    return LEMMA_TABLE.get(word.lower(), word.lower())


# === COMPARISON ===

print("=== Stemming vs Lemmatization: Head to Head ===\n")

test_words = [
    'running', 'ran', 'runs',
    'caring', 'cares',
    'better', 'best',
    'studies', 'studying', 'studied',
    'mice', 'children', 'geese',
    'universities', 'universe',
    'was', 'were', 'been',
    'happiness', 'sadly',
    'jumping', 'jumped',
]

print(f"{'Word':<16} {'Stem':<16} {'Lemma':<16} {'Match?'}")
print("-" * 60)
for word in test_words:
    stem = simple_stem(word)
    lemma = simple_lemmatize(word)
    match = "yes" if stem == lemma else "NO"
    print(f"{word:<16} {stem:<16} {lemma:<16} {match}")

print()

# === WHERE STEMMING FAILS ===

print("=== Stemming Failures ===\n")

failure_cases = [
    ('caring', 'Should reduce to "care", not "car"'),
    ('university', 'Should NOT merge with "universe"'),
    ('better', 'Should reduce to "good" (irregular)'),
    ('was', 'Should reduce to "be" (irregular verb)'),
    ('mice', 'Should reduce to "mouse" (irregular plural)'),
]

for word, explanation in failure_cases:
    stem = simple_stem(word)
    lemma = simple_lemmatize(word)
    print(f"  Word: '{word}'")
    print(f"    Stem:  '{stem}'")
    print(f"    Lemma: '{lemma}'")
    print(f"    Issue: {explanation}")
    print()


# === PRACTICAL IMPACT: How does this affect text matching? ===

print("=== Impact on Document Matching ===\n")

doc1 = "The children were running and swimming in the pool"
doc2 = "A child runs and swims every day"

words1 = doc1.lower().split()
words2 = doc2.lower().split()

stems1 = set(simple_stem(w) for w in words1)
stems2 = set(simple_stem(w) for w in words2)

lemmas1 = set(simple_lemmatize(w) for w in words1)
lemmas2 = set(simple_lemmatize(w) for w in words2)

raw_overlap = set(words1) & set(words2)
stem_overlap = stems1 & stems2
lemma_overlap = lemmas1 & lemmas2

print(f"Doc 1: '{doc1}'")
print(f"Doc 2: '{doc2}'")
print()
print(f"  Raw word overlap:   {raw_overlap or '{none}'}")
print(f"  Stem overlap:       {stem_overlap}")
print(f"  Lemma overlap:      {lemma_overlap}")
print()
print("Lemmatization captures that 'children/child', 'were/be',")
print("'running/run', and 'swimming/swim' are the same concepts.")
```

---

## Key Takeaways

- **Stemming is fast but crude.** It chops suffixes using rules, often producing non-words ("univers", "car" from "caring"). It works well for search and information retrieval where approximate matching is acceptable.
- **Lemmatization is accurate but heavier.** It uses vocabulary and grammar knowledge to produce real dictionary forms. It's better when you need precise word meanings.
- **Neither is always right.** The choice depends on your task: search engines typically use stemming (speed matters, approximate is fine); question answering systems typically use lemmatization (precision matters).
- **Irregular forms break stemming.** English has hundreds of irregular verbs, plurals, and adjectives that no suffix rule can handle.
