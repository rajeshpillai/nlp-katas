# Identify Noise in Real-World Text

> Phase 0 — Language & Text (Foundations) | Kata 0.2

---

## Concept & Intuition

### What problem are we solving?

Text in the real world is messy. It contains typos, inconsistent casing, HTML tags, encoding artifacts, extra whitespace, slang, abbreviations, and emoji. Before any NLP system can work with text, we need to understand **what noise looks like** and how it distorts meaning.

This kata is about seeing text the way a computer sees it — as a raw sequence of characters — and understanding how far that is from what a human reads.

### Why naive approaches fail

If you treat text as-is, you'll find that:
- "Hello", "hello", "HELLO", and "  hello  " are all different strings
- "don't" might appear as "don\u2019t", "don't", or "don&rsquo;t" depending on the source
- A product review scraped from a webpage might contain `<br>` tags, `&amp;` entities, and invisible Unicode characters

A system that doesn't account for noise will treat these as completely different inputs, even though a human reads them identically.

### Mental models

- **Signal vs noise**: The meaning you want is the signal. Everything else — formatting artifacts, encoding issues, typos — is noise.
- **The telephone game**: Every time text passes through a system (web scraping, copy-paste, database storage, API response), it can pick up noise.
- **Garbage in, garbage out**: If you feed noisy text to an NLP model, the model learns the noise, not the signal.

### Visual explanations

```
The same review, from different sources:

Source 1 (clean):
  "This product is great! Highly recommend it."

Source 2 (web scrape):
  "<p>This product is great!&nbsp;Highly recommend it.</p>\n\n"

Source 3 (social media):
  "this product is GREAT!!! highly reccomend it 😍😍"

Source 4 (encoding issue):
  "This product is great\ufffd Highly recommend it."

Same meaning. Four different strings. A computer sees four different inputs.
```

---

## Hands-on Exploration

1. Copy text from a webpage and paste it into a plain text editor — notice hidden characters
2. Compare the same message written formally vs as a text message
3. Try to write code that checks if two noisy strings "mean the same thing" — notice how many edge cases arise

---

## Live Code

```python
# --- Types of noise in real-world text ---

# 1. Casing inconsistency
texts_casing = ["Hello World", "hello world", "HELLO WORLD", "hElLo WoRlD"]
print("--- Casing ---")
for t in texts_casing:
    print(f"  '{t}' -> unique chars: {len(set(t))}")
print(f"  All equal? {len(set(texts_casing)) == 1}")
print(f"  All equal (lowered)? {len(set(t.lower() for t in texts_casing)) == 1}")
print()

# 2. Whitespace noise
texts_whitespace = [
    "hello world",
    "  hello   world  ",
    "hello\tworld",
    "hello\n\nworld",
]
print("--- Whitespace ---")
for t in texts_whitespace:
    print(f"  {repr(t)}")

# Simple normalization: split and rejoin
normalized = [" ".join(t.split()) for t in texts_whitespace]
print(f"  After normalization: {normalized}")
print(f"  All equal? {len(set(normalized)) == 1}")
print()

# 3. Encoding artifacts and special characters
texts_encoding = [
    "don't",           # straight apostrophe
    "don\u2019t",      # curly apostrophe (Unicode RIGHT SINGLE QUOTATION MARK)
    "don&rsquo;t",     # HTML entity
]
print("--- Encoding ---")
for t in texts_encoding:
    print(f"  '{t}' -> bytes: {[hex(b) for b in t.encode('utf-8')]}")
print()

# 4. HTML/markup artifacts
html_text = """<div class="review">
  <p>This product is <b>great</b>!&nbsp;Highly recommend&hellip;</p>
  <br/>
  <span>★★★★★</span>
</div>"""

print("--- HTML noise ---")
print(f"  Raw: {repr(html_text[:60])}...")
print()

# Strip HTML with basic approach (no libraries)
import re
def strip_html(text):
    """Remove HTML tags and decode common entities."""
    text = re.sub(r'<[^>]+>', ' ', text)      # remove tags
    text = text.replace('&nbsp;', ' ')         # non-breaking space
    text = text.replace('&hellip;', '...')     # ellipsis
    text = text.replace('&amp;', '&')          # ampersand
    text = ' '.join(text.split())              # normalize whitespace
    return text.strip()

cleaned = strip_html(html_text)
print(f"  Cleaned: '{cleaned}'")
print()

# 5. Social media noise
social_texts = [
    "this is AMAZINGGG!!! 😍😍😍 #bestproduct @company",
    "ngl this product slaps fr fr no cap 💯",
    "Soooo good!!!! Would def reccomend 2 everyone",
    "worst. product. ever. do NOT buy 🚫🚫🚫",
]

print("--- Social media text ---")
for t in social_texts:
    words = t.split()
    has_emoji = any(ord(c) > 127 for c in t)
    has_hashtag = any(w.startswith('#') for w in words)
    has_mention = any(w.startswith('@') for w in words)
    repeated_chars = bool(re.search(r'(.)\1{2,}', t))
    print(f"  '{t}'")
    print(f"    emoji: {has_emoji}, hashtag: {has_hashtag}, "
          f"mention: {has_mention}, repeated chars: {repeated_chars}")
print()

# 6. Putting it together: a noise detection function
def analyze_noise(text):
    """Detect common types of noise in text."""
    issues = []
    if text != text.strip():
        issues.append("leading/trailing whitespace")
    if '  ' in text or '\t' in text or '\n' in text:
        issues.append("irregular whitespace")
    if re.search(r'<[^>]+>', text):
        issues.append("HTML tags")
    if re.search(r'&\w+;', text):
        issues.append("HTML entities")
    if re.search(r'(.)\1{2,}', text):
        issues.append("repeated characters")
    if any(ord(c) > 127 for c in text):
        issues.append("non-ASCII characters")
    if text != text.lower() and text != text.upper():
        issues.append("mixed casing")
    return issues

test_inputs = [
    "Clean simple text here.",
    "  MESSY   text\twith\n\nproblems  ",
    "<p>HTML &amp; entities</p>",
    "sooooo gooood!!! 😍😍",
]

print("--- Noise analysis ---")
for text in test_inputs:
    issues = analyze_noise(text)
    label = "clean" if not issues else ", ".join(issues)
    print(f"  '{text}' -> {label}")
```

---

## Key Takeaways

- **Real-world text is noisy.** Casing, whitespace, encoding, HTML artifacts, and social media conventions all distort the raw signal.
- **Noise is not random** — it follows patterns (web scraping produces HTML noise, social media produces emoji and slang).
- **A computer sees strings, not meaning.** "Hello" and "hello" are as different to a computer as "Hello" and "Goodbye."
- **Noise detection comes before noise removal.** You must understand what noise exists before deciding how to handle it — some "noise" (like casing or emoji) might carry meaning you want to preserve.
