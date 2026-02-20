# Compare Structured vs Unstructured Text

> Phase 0 — Language & Text (Foundations) | Kata 0.3

---

## Concept & Intuition

### What problem are we solving?

Not all text is created equal. A CSV file with columns "name, age, city" is easy for a computer to parse — every value has a fixed position and type. A customer email saying "Hi, I'm 30 and I live in Mumbai" contains the same information, but extracting it requires **understanding language**.

NLP exists because most of the world's text is **unstructured** — it doesn't fit into neat rows and columns. Understanding the spectrum from structured to unstructured data is essential to knowing when you need NLP and when you don't.

### Why naive approaches fail

With structured data, you can write exact rules: "column 3 is the city." With unstructured text, there are no columns. The city could appear anywhere, in any format:
- "I live in Mumbai"
- "Mumbai-based professional"
- "currently residing in Mumbai, India"
- "from Bombay" (same city, old name)

Simple pattern matching breaks because natural language has infinite ways to express the same fact.

### Mental models

- **Structured data** = a filled form. Every field has a label and a slot. Easy to extract.
- **Unstructured data** = a conversation. Information is embedded in context. Hard to extract.
- **Semi-structured data** = a form with a "comments" box. Part of it is organized, part is free text.
- **The NLP spectrum**: The less structure your data has, the more NLP you need.

### Visual explanations

```
Structured (easy to parse):
  ┌──────────┬─────┬──────────┐
  │ Name     │ Age │ City     │
  ├──────────┼─────┼──────────┤
  │ Alice    │  30 │ Mumbai   │
  │ Bob      │  25 │ Delhi    │
  └──────────┴─────┴──────────┘

Semi-structured (partially parseable):
  {
    "name": "Alice",
    "bio": "30 year old developer from Mumbai who loves hiking"
  }
  → "name" is easy. Extracting age and city from "bio" requires NLP.

Unstructured (requires NLP):
  "Hi! I'm Alice. I moved to Mumbai three years ago after
   finishing my degree. I'm turning 31 next month."
  → Name, city, age are all embedded in prose.
```

---

## Hands-on Exploration

1. Take a restaurant review and try to extract the restaurant name, rating, and cuisine — notice how much harder it is than reading a CSV
2. Compare how many lines of code it takes to extract a name from a JSON object vs from a paragraph of text
3. Find an example of semi-structured data (like an HTML page) and identify which parts are structured and which are free text

---

## Live Code

```python
import json
import csv
import io

# --- Structured data: trivial to extract ---

csv_data = """name,age,city
Alice,30,Mumbai
Bob,25,Delhi
Charlie,35,Bangalore"""

print("--- Structured (CSV) ---")
reader = csv.DictReader(io.StringIO(csv_data))
for row in reader:
    print(f"  {row['name']} is {row['age']} years old, lives in {row['city']}")
print()

# JSON: also structured
json_data = [
    {"name": "Alice", "age": 30, "city": "Mumbai"},
    {"name": "Bob", "age": 25, "city": "Delhi"},
]

print("--- Structured (JSON) ---")
for person in json_data:
    print(f"  {person['name']} is {person['age']} years old, lives in {person['city']}")
print()


# --- Unstructured data: same information, but embedded in prose ---

unstructured_texts = [
    "Hi, I'm Alice! I'm 30 years old and I live in Mumbai.",
    "Bob here. 25, based in Delhi. Love cricket and coding.",
    "My name is Charlie, I'm a 35-year-old engineer from Bangalore.",
]

print("--- Unstructured (free text) ---")
for text in unstructured_texts:
    print(f"  Text: '{text}'")
    print(f"  How do we extract name, age, city from this?")
    print()


# --- Attempting extraction from unstructured text (naive) ---

import re

def naive_extract(text):
    """Try to extract name, age, city with simple patterns."""
    # Try to find age
    age_match = re.search(r'(\d{1,2})\s*(?:years?\s*old|,)', text)
    age = age_match.group(1) if age_match else "?"

    # Try to find city (hardcoded list — doesn't scale)
    cities = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata"]
    city = "?"
    for c in cities:
        if c.lower() in text.lower():
            city = c
            break

    # Name? Almost impossible without NLP
    name = "?"

    return {"name": name, "age": age, "city": city}

print("--- Naive extraction attempt ---")
for text in unstructured_texts:
    result = naive_extract(text)
    print(f"  Text: '{text[:50]}...'")
    print(f"  Extracted: {result}")
    print()


# --- Semi-structured: mix of both ---

semi_structured = [
    {
        "username": "alice_m",
        "verified": True,
        "bio": "30 year old developer from Mumbai. Love hiking and coffee.",
    },
    {
        "username": "bob_d",
        "verified": False,
        "bio": "Delhi boy, 25. Into cricket, coding, and street food.",
    },
]

print("--- Semi-structured ---")
for profile in semi_structured:
    print(f"  Username: {profile['username']} (structured)")
    print(f"  Verified: {profile['verified']} (structured)")
    print(f"  Bio: '{profile['bio']}' (unstructured — needs NLP)")
    print()


# --- The key comparison ---
print("--- Effort comparison ---")
print()
print("  Structured (CSV/JSON):")
print("    Code to extract city: row['city']")
print("    Lines of code: 1")
print("    Accuracy: 100%")
print()
print("  Unstructured (free text):")
print("    Code to extract city: regex, city lists, NLP models...")
print("    Lines of code: 10-100+")
print("    Accuracy: varies, never 100%")
print()
print("  This gap is why NLP exists.")
```

---

## Key Takeaways

- **Structured data has explicit labels** (column names, JSON keys). Extraction is trivial.
- **Unstructured text encodes information in prose.** Extracting it requires understanding language — this is the domain of NLP.
- **Semi-structured data is the most common** in practice — a JSON response with a free-text "description" field, an HTML page with navigation and articles.
- **NLP is expensive.** If your data is structured, use structured tools (SQL, pandas). Reach for NLP only when the information is embedded in natural language.
- **Language is symbolic, contextual, and lossy when digitized.** The same fact can be expressed in infinite ways, and converting it to a structured format always loses something.
