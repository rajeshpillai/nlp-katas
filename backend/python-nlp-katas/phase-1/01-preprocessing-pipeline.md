# Apply a Preprocessing Pipeline to Raw Text

> Phase 1 — Text Preprocessing | Kata 1.1

---

## Concept & Intuition

### What problem are we solving?

Raw text is inconsistent. The same word appears as "Hello", "hello", "HELLO". Sentences carry punctuation that adds nothing to meaning for most tasks. Whitespace is erratic. Before text can become numbers (which is the entire point of NLP), it must be **normalized** — brought into a consistent, predictable form.

A preprocessing pipeline is a sequence of transformations applied to raw text. Each step makes a decision about what to keep and what to discard. These decisions are not neutral — they encode assumptions about what matters.

### Why naive approaches fail

You might think: "just lowercase everything and strip punctuation." But consider:

- Lowercasing "US" (country) makes it "us" (pronoun) — meaning is lost
- Removing punctuation from "Dr. Smith" gives "Dr Smith" — minor. But from "3.14" gives "314" — catastrophic
- Stripping whitespace from "New York" gives "NewYork" — the entity is destroyed

There is no universal "correct" pipeline. Every choice is a tradeoff between simplifying the text and preserving meaning.

### Mental models

- **Pipeline as a funnel**: Raw text goes in, normalized text comes out. Each step narrows the variation but also narrows the information.
- **Lossy compression**: Preprocessing is like compressing an image — you reduce size (variation) but you can't get the original back. Choose your compression carefully.
- **Order matters**: Lowercasing before tokenization produces different results than tokenizing first. Each step transforms the input for the next step.

### Visual explanations

```
Raw text:   "  The Quick Brown FOX jumped over the lazy dog!!! 😊  "
                |
         [strip whitespace]
                |
            "The Quick Brown FOX jumped over the lazy dog!!! 😊"
                |
            [lowercase]
                |
            "the quick brown fox jumped over the lazy dog!!! 😊"
                |
         [remove punctuation]
                |
            "the quick brown fox jumped over the lazy dog  😊"
                |
          [remove emoji]
                |
            "the quick brown fox jumped over the lazy dog"
                |
        [normalize whitespace]
                |
            "the quick brown fox jumped over the lazy dog"

Each step is a decision. Each decision loses something.
```

---

## Hands-on Exploration

1. Take a messy sentence (with mixed case, punctuation, extra spaces) and apply each preprocessing step one at a time — observe what changes at each stage
2. Find a sentence where lowercasing changes the meaning (hint: proper nouns, acronyms)
3. Find a sentence where removing punctuation changes the meaning (hint: "Let's eat, grandma" vs "Let's eat grandma")

---

## Live Code

```python
# --- Building a text preprocessing pipeline from scratch ---
import re

# Step 1: Individual preprocessing functions

def strip_whitespace(text):
    """Remove leading/trailing whitespace."""
    return text.strip()

def lowercase(text):
    """Convert to lowercase."""
    return text.lower()

def remove_punctuation(text):
    """Remove punctuation characters, keep letters, digits, whitespace."""
    return re.sub(r'[^\w\s]', '', text)

def normalize_whitespace(text):
    """Collapse multiple spaces/tabs/newlines into single space."""
    return ' '.join(text.split())

def remove_digits(text):
    """Remove all digit characters."""
    return re.sub(r'\d+', '', text)


# Step 2: Build a pipeline — a list of functions applied in order

def preprocess(text, steps):
    """Apply a sequence of preprocessing steps to text."""
    print(f"  Original:  '{text}'")
    for step in steps:
        text = step(text)
        print(f"  After {step.__name__:.<25s} '{text}'")
    print()
    return text


# Step 3: Apply to messy real-world text

samples = [
    "  The Quick Brown FOX jumped over the lazy dog!!!  ",
    "  Dr. Smith earned $3.14 million in Q3 2024.  ",
    "I can't believe it's not butter!!! #food @brand  ",
    "NEW YORK -- The U.S. economy grew 2.5% last quarter...",
    "   Hello,    World!!   How's   it   going???   ",
]

pipeline = [
    strip_whitespace,
    lowercase,
    remove_punctuation,
    normalize_whitespace,
]

print("=== Standard Pipeline ===\n")
for sample in samples:
    result = preprocess(sample, pipeline)


# Step 4: Compare different pipeline orders

print("\n=== Order Matters ===\n")

text = "The U.S. has 50 states."

print("Pipeline A: lowercase -> remove punctuation -> normalize whitespace")
a = preprocess(text, [lowercase, remove_punctuation, normalize_whitespace])

print("Pipeline B: remove punctuation -> lowercase -> normalize whitespace")
b = preprocess(text, [remove_punctuation, lowercase, normalize_whitespace])

print(f"Same result? {a == b}")
print()


# Step 5: Show what preprocessing destroys

print("\n=== What Gets Lost ===\n")

problem_cases = [
    ("US vs us", "The US economy is strong."),
    ("Acronyms", "I work at NASA."),
    ("Decimal numbers", "Pi is approximately 3.14159."),
    ("Meaningful punctuation", "Let's eat, grandma!"),
    ("Contractions", "I can't, won't, and shouldn't."),
]

for label, text in problem_cases:
    cleaned = preprocess(text, pipeline)
    print(f"  Issue ({label}):")
    print(f"    Before: '{text}'")
    print(f"    After:  '{cleaned}'")
    print()
```

---

## Key Takeaways

- **Preprocessing is not neutral.** Every step encodes an assumption about what matters in the text.
- **Order matters.** The sequence of transformations affects the final result.
- **There is no universal pipeline.** The right pipeline depends on your task — sentiment analysis, search, classification, and translation all need different preprocessing.
- **Preprocessing is lossy.** Once information is removed, it cannot be recovered. Err on the side of keeping information unless you're certain it's noise.
