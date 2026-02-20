# Rule-Based Named Entity Recognition

> Phase 6 — Named Entity Recognition (NER) | Kata 6.1

---

## Concept & Intuition

### What problem are we solving?

Text contains **named entities** — references to specific people, organizations, locations, dates, and other real-world objects. Extracting these entities from unstructured text is called Named Entity Recognition (NER). It turns prose into structured data.

Consider this sentence: "Priya Sharma joined Infosys in Bangalore last March." A human immediately spots three entities: a person, a company, and a city. But to a computer, these are just tokens in a string. NER is the task of finding where entities begin and end (the **boundary** problem) and what type they are (the **labeling** problem).

In this kata, we build NER using **rules** — handcrafted patterns that exploit regularities in how entities appear in text. Rules are the oldest and most transparent approach to NER, and understanding them is essential before moving to statistical methods.

### Why earlier methods fail

Bag of Words and TF-IDF treat every word as an independent feature. They can tell you that "Bangalore" appears in a document, but they cannot tell you:
- That "Bangalore" is a LOCATION
- That "New York" is a single entity spanning two tokens
- That "Apple" in "Apple released a new phone" is an ORGANIZATION, not a fruit

Earlier representations flatten text into word counts or weighted scores. NER requires understanding **which specific tokens** in a sentence refer to real-world entities, and that requires looking at the tokens themselves, their shape, their context, and their position.

### Mental models

- **Entity = span + label**: An entity is not just a word — it is a contiguous span of text (one or more tokens) assigned a category. "New York City" is one LOCATION entity spanning three tokens.
- **Gazetteer**: A lookup table of known entities. If you have a list of 10,000 city names, you can match them against text. Fast and precise, but incomplete — new entities appear constantly.
- **Pattern rules**: Entities follow surface patterns. Person names are often capitalized. Organizations often end with "Inc.", "Ltd.", "Corp." Locations often follow prepositions like "in" or "from."
- **The boundary problem**: Deciding where an entity starts and ends is harder than it looks. In "the University of California, Berkeley," is the entity "University of California" or "University of California, Berkeley"?

### Visual explanations

```
Input text:
  "Priya Sharma joined Infosys in Bangalore last March."

NER output:
   Priya Sharma   joined   Infosys   in   Bangalore   last   March
   [--- PERSON --]          [ORG  ]       [LOCATION]         [DATE]

How rules work:

  Rule 1: Capitalized word sequences -> candidate PERSON
    "Priya Sharma" -> two consecutive capitalized words -> PERSON candidate

  Rule 2: Gazetteer lookup -> ORGANIZATION
    "Infosys" -> found in company list -> ORGANIZATION

  Rule 3: Preposition + capitalized word -> LOCATION candidate
    "in Bangalore" -> "in" + capitalized word -> LOCATION candidate

  Rule 4: Month names -> DATE
    "March" -> found in month list -> DATE

Where rules break:

  "Paris Hilton went to Paris."
   [--- PERSON --]         [LOC]
   Rule 1 says: two capitalized words -> PERSON (correct for "Paris Hilton")
   Rule 3 says: "to" + capitalized word -> LOCATION (correct for "Paris")
   But Rule 1 alone would also match "Paris" as part of a person name.
   Without context, rules cannot disambiguate.
```

---

## Hands-on Exploration

1. Take the sentence "Dr. Amara Khan works at the World Health Organization in Geneva." Identify all entities by hand, marking the exact span and type of each. Notice how "World Health Organization" is one entity spanning three words — boundary detection is not trivial.

2. Write down a set of rules that would correctly extract PERSON, ORGANIZATION, and LOCATION from 5 news-like sentences of your choice. Now find a sentence where your rules produce a wrong answer. How many rules did it take before you hit a failure case?

3. Consider the word "Washington" in these three contexts: "Washington signed the treaty," "Washington approved the bill," "I flew to Washington." The same word is a PERSON, an ORGANIZATION (metonym for government), and a LOCATION. Can any fixed rule handle all three? What additional information would you need?

---

## Live Code

```python
import re
from collections import defaultdict

# ============================================================
# RULE-BASED NAMED ENTITY RECOGNITION
# ============================================================
# We build a NER system using only handcrafted rules:
#   - Gazetteers (lookup lists of known entities)
#   - Capitalization patterns
#   - Contextual clues (words that appear near entities)
#   - Regular expressions for dates
# ============================================================

# --- Gazetteers: lists of known entities ---

PERSON_TITLES = {"mr", "mrs", "ms", "dr", "prof", "sir", "lady", "captain", "judge"}

KNOWN_ORGS = {
    "infosys", "google", "microsoft", "amazon", "tesla", "apple",
    "united nations", "world health organization", "european union",
    "reserve bank of india", "supreme court", "nasa", "unesco",
    "red cross", "greenpeace", "amnesty international",
}

ORG_SUFFIXES = {"inc", "ltd", "corp", "llc", "co", "plc", "foundation", "institute", "association"}

KNOWN_LOCATIONS = {
    "mumbai", "delhi", "bangalore", "chennai", "kolkata", "hyderabad",
    "new york", "london", "paris", "tokyo", "berlin", "sydney",
    "california", "texas", "india", "china", "germany", "brazil",
    "geneva", "washington", "singapore", "toronto", "dubai",
}

LOCATION_PREPOSITIONS = {"in", "at", "from", "to", "near", "across", "through"}

MONTHS = {
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
}


def tokenize(text):
    """Split text into tokens, preserving their positions."""
    tokens = []
    for match in re.finditer(r"[A-Za-z0-9]+(?:\.[A-Za-z])?|[.,;:!?]", text):
        tokens.append({
            "text": match.group(),
            "start": match.start(),
            "end": match.end(),
        })
    return tokens


def find_entities(text):
    """Apply rule-based NER to text. Returns list of (span, label, rule)."""
    tokens = tokenize(text)
    entities = []
    used_indices = set()
    text_lower = text.lower()

    # --- Rule 1: Multi-word organization gazetteer lookup ---
    for org in sorted(KNOWN_ORGS, key=len, reverse=True):
        pattern = re.compile(re.escape(org), re.IGNORECASE)
        for m in pattern.finditer(text):
            entities.append({
                "text": m.group(),
                "start": m.start(),
                "end": m.end(),
                "label": "ORG",
                "rule": "org_gazetteer",
            })

    # --- Rule 2: Organization suffix patterns (e.g., "... Inc.", "... Foundation") ---
    for i, tok in enumerate(tokens):
        if tok["text"].lower().rstrip(".") in ORG_SUFFIXES:
            # Look back for capitalized words before the suffix
            span_tokens = [tok]
            j = i - 1
            while j >= 0 and tokens[j]["text"][0].isupper():
                span_tokens.insert(0, tokens[j])
                j -= 1
            if len(span_tokens) > 1:
                span_text = text[span_tokens[0]["start"]:span_tokens[-1]["end"]]
                entities.append({
                    "text": span_text,
                    "start": span_tokens[0]["start"],
                    "end": span_tokens[-1]["end"],
                    "label": "ORG",
                    "rule": "org_suffix",
                })

    # --- Rule 3: Title + capitalized words -> PERSON ---
    for i, tok in enumerate(tokens):
        if tok["text"].lower().rstrip(".") in PERSON_TITLES:
            # Collect capitalized words following the title
            span_tokens = [tok]
            j = i + 1
            while j < len(tokens) and tokens[j]["text"][0].isupper():
                span_tokens.append(tokens[j])
                j += 1
            if len(span_tokens) > 1:
                span_text = text[span_tokens[0]["start"]:span_tokens[-1]["end"]]
                entities.append({
                    "text": span_text,
                    "start": span_tokens[0]["start"],
                    "end": span_tokens[-1]["end"],
                    "label": "PERSON",
                    "rule": "title_pattern",
                })
                for k in range(i, j):
                    used_indices.add(k)

    # --- Rule 4: Location preposition + capitalized word(s) -> LOCATION ---
    for i, tok in enumerate(tokens):
        if tok["text"].lower() in LOCATION_PREPOSITIONS:
            span_tokens = []
            j = i + 1
            while j < len(tokens) and tokens[j]["text"][0].isupper():
                span_tokens.append(tokens[j])
                j += 1
            if span_tokens:
                candidate = " ".join(t["text"] for t in span_tokens).lower()
                if candidate in KNOWN_LOCATIONS or len(span_tokens) == 1:
                    span_text = text[span_tokens[0]["start"]:span_tokens[-1]["end"]]
                    entities.append({
                        "text": span_text,
                        "start": span_tokens[0]["start"],
                        "end": span_tokens[-1]["end"],
                        "label": "LOCATION",
                        "rule": "prep_location",
                    })

    # --- Rule 5: Known location gazetteer ---
    for loc in sorted(KNOWN_LOCATIONS, key=len, reverse=True):
        pattern = re.compile(r"\b" + re.escape(loc) + r"\b", re.IGNORECASE)
        for m in pattern.finditer(text):
            # Only add if not already covered by another entity at this position
            already_covered = any(
                e["start"] <= m.start() and e["end"] >= m.end()
                for e in entities
            )
            if not already_covered:
                entities.append({
                    "text": m.group(),
                    "start": m.start(),
                    "end": m.end(),
                    "label": "LOCATION",
                    "rule": "loc_gazetteer",
                })

    # --- Rule 6: Date patterns ---
    # Month names
    for month in MONTHS:
        pattern = re.compile(r"\b" + re.escape(month) + r"(?:\s+\d{1,2})?(?:\s*,?\s*\d{4})?\b", re.IGNORECASE)
        for m in pattern.finditer(text):
            entities.append({
                "text": m.group(),
                "start": m.start(),
                "end": m.end(),
                "label": "DATE",
                "rule": "month_pattern",
            })

    # Numeric date patterns (e.g., 2024-03-15, 15/03/2024)
    date_re = re.compile(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b|\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b")
    for m in date_re.finditer(text):
        entities.append({
            "text": m.group(),
            "start": m.start(),
            "end": m.end(),
            "label": "DATE",
            "rule": "numeric_date",
        })

    # --- Deduplicate: keep longest span at each position ---
    entities.sort(key=lambda e: (e["start"], -(e["end"] - e["start"])))
    filtered = []
    covered_until = -1
    for e in entities:
        if e["start"] >= covered_until:
            filtered.append(e)
            covered_until = e["end"]

    return filtered


def display_entities(text, entities):
    """Display entities with inline annotations."""
    if not entities:
        print(f"  Text: {text}")
        print("  Entities: (none found)")
        return

    print(f"  Text: {text}")
    for e in entities:
        print(f"    [{e['label']:>8}]  \"{e['text']}\"  (rule: {e['rule']})")


# ============================================================
# TEST ON SAMPLE SENTENCES
# ============================================================

test_sentences = [
    "Dr. Amara Khan works at the World Health Organization in Geneva.",
    "Infosys announced a new office in Bangalore on March 15, 2025.",
    "Prof. Raj Mehta presented at the United Nations summit in New York.",
    "Tesla CEO visited the Google campus in California last January.",
    "The Reserve Bank of India raised interest rates in Mumbai.",
]

print("=" * 65)
print("  RULE-BASED NER: SUCCESSFUL EXTRACTIONS")
print("=" * 65)
print()

for sent in test_sentences:
    entities = find_entities(sent)
    display_entities(sent, entities)
    print()


# ============================================================
# WHERE RULES FAIL: FALSE POSITIVES AND MISSED ENTITIES
# ============================================================

tricky_sentences = [
    # "May" is a month AND a verb — rule produces false positive DATE
    "May I go to Paris in May?",
    # "Apple" is a company but also a fruit — no title or suffix to help
    "Apple released a new product.",
    # "Jordan" is both a country and a common first/last name
    "Jordan visited Jordan last summer.",
    # No capitalization clues (lowercased text from social media)
    "just landed in mumbai, meeting priya tomorrow at infosys office",
    # Nested/ambiguous organization reference
    "The president of the United States addressed Congress in Washington.",
]

print("=" * 65)
print("  WHERE RULES FAIL")
print("=" * 65)
print()

expected_failures = [
    "Expected: LOCATION 'Paris', DATE 'May' (end). Problem: first 'May' is a verb, not a date.",
    "Expected: ORG 'Apple'. Problem: no title/suffix, not in org gazetteer as standalone capitalized word.",
    "Expected: PERSON 'Jordan' (first), LOCATION 'Jordan' (second). Problem: same word, two types.",
    "Expected: LOCATION 'mumbai', PERSON 'priya', ORG 'infosys'. Problem: no capitalization signals.",
    "Expected: ORG 'United States', ORG 'Congress', LOCATION 'Washington'. Problem: overlapping rules.",
]

for sent, note in zip(tricky_sentences, expected_failures):
    entities = find_entities(sent)
    display_entities(sent, entities)
    print(f"    NOTE: {note}")
    print()


# ============================================================
# RULE COVERAGE ANALYSIS
# ============================================================

print("=" * 65)
print("  RULE COVERAGE ANALYSIS")
print("=" * 65)
print()

all_sentences = test_sentences + tricky_sentences
rule_counts = defaultdict(int)
label_counts = defaultdict(int)
total_entities = 0

for sent in all_sentences:
    entities = find_entities(sent)
    for e in entities:
        rule_counts[e["rule"]] += 1
        label_counts[e["label"]] += 1
        total_entities += 1

print(f"  Total entities found: {total_entities}")
print()
print("  Entities by type:")
for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
    bar = "#" * count
    print(f"    {label:>8}: {count:>3}  {bar}")

print()
print("  Entities by rule:")
for rule, count in sorted(rule_counts.items(), key=lambda x: -x[1]):
    bar = "#" * count
    print(f"    {rule:>16}: {count:>3}  {bar}")

print()
print("  Key observations:")
print("    - Gazetteers provide high precision but miss unknown entities")
print("    - Pattern rules (title, suffix) catch novel entities but produce false positives")
print("    - Lowercased text defeats capitalization-based rules entirely")
print("    - Ambiguous words (May, Apple, Jordan) require context, not patterns")
print()
print("  Rule-based NER is transparent and fast,")
print("  but it is brittle — every new edge case requires a new rule.")
```

---

## Key Takeaways

- **NER is a boundary + labeling task.** Finding where an entity starts and ends is just as important as determining its type. "World Health Organization" is one entity, not three.
- **Rules exploit surface patterns** — capitalization, known word lists, contextual prepositions, and suffixes. They are transparent and require no training data, but they break on ambiguous words, missing capitalization, and novel entities.
- **Gazetteers are precise but incomplete.** A list of 10,000 cities will catch "Mumbai" but miss "Koramangala" (a neighborhood). Lists can never be exhaustive for a living language.
- **Ambiguity is the fundamental limit of rules.** "May" (month vs. verb), "Apple" (company vs. fruit), "Jordan" (person vs. country) — resolving these requires context that no fixed pattern can capture. This motivates statistical approaches.
