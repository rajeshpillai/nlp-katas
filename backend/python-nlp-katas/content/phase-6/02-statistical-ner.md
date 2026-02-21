# Simple ML-Based Named Entity Recognition

> Phase 6 — Named Entity Recognition (NER) | Kata 6.2

---

## Concept & Intuition

### What problem are we solving?

In Kata 6.1, we built NER with handcrafted rules. Rules are transparent but brittle — every new edge case demands a new rule, and ambiguous words like "May" or "Apple" cannot be resolved by patterns alone. The question becomes: can we **learn** to recognize entities from patterns in data, instead of writing each pattern by hand?

Statistical NER shifts the work from **writing rules** to **designing features**. Instead of saying "if a word is capitalized and follows 'Dr.', it is a PERSON," we say "extract the capitalization status, the previous word, the word shape, and the position — then let a scoring system decide." The features capture the same intuitions as rules, but they combine flexibly and can handle novel cases that no single rule anticipated.

This kata builds a feature-based NER system from scratch. We do not use any machine learning library. Instead, we use a simple weighted scoring approach that demonstrates the core idea: **features plus weights equal predictions**.

### Why earlier methods fail

Rule-based NER fails for three fundamental reasons:

1. **Rigid boundaries**: A rule either fires or it does not. There is no notion of "mostly looks like a person name." Statistical approaches assign confidence scores.
2. **No feature interaction**: Rules check conditions independently. But entity recognition often requires combining multiple weak signals — a word is capitalized AND follows a preposition AND is at the end of a sentence. Rules require explicit enumeration of every useful combination.
3. **No generalization**: A rule for "Dr. + Name" does not help with "CEO + Name" unless you add another rule. Feature-based systems generalize because the feature "preceded by a title-like word" covers both cases.

### Mental models

- **Features as lenses**: Each feature is a different way of looking at a token. Capitalization is one lens. Word shape is another. Context words are another. No single lens is sufficient, but together they form a clear picture.
- **Voting by features**: Think of each feature as a voter. "This word is capitalized" votes weakly for PERSON. "The previous word is 'in'" votes for LOCATION. "The word shape is Xxxxx" votes for a named entity of some kind. The entity type with the most weighted votes wins.
- **Context windows**: A token's identity depends on its neighbors. The word "Bank" means different things after "River" vs. after "Central." Feature-based NER captures this by including features from surrounding tokens.

### Visual explanations

```
Sentence: "Dr. Amara Khan works at Infosys in Bangalore."

Token-level features for "Khan":
  ┌─────────────────────────────────────────────┐
  │ Feature               │ Value               │
  ├─────────────────────────────────────────────┤
  │ is_capitalized        │ True                │
  │ is_all_caps           │ False               │
  │ word_shape            │ Xxxx                │
  │ word_length           │ 4                   │
  │ position_in_sentence  │ 0.28  (early)       │
  │ prev_word             │ "Amara"             │
  │ prev_is_capitalized   │ True                │
  │ prev_prev_word        │ "Dr."               │
  │ next_word             │ "works"             │
  │ next_is_capitalized   │ False               │
  │ has_title_prefix      │ True  (Dr. nearby)  │
  │ in_location_gazetteer │ False               │
  │ in_org_gazetteer      │ False               │
  └─────────────────────────────────────────────┘

Feature voting for "Khan":
  PERSON:    is_capitalized (+2) + prev_capitalized (+1) + has_title (+3) = +6
  LOCATION:  is_capitalized (+2) + in_loc_gazetteer (0)                  = +2
  ORG:       is_capitalized (+2) + in_org_gazetteer (0)                  = +2
  O (none):  next_is_lowercase (+1)                                      = +1

  Winner: PERSON (score 6)

Comparison with rules (Kata 6.1):
  Rule system: "Dr." + capitalized word -> PERSON  (binary: fires or not)
  Feature system: combines 10+ signals with weights (graded: confidence score)
```

---

## Hands-on Exploration

1. Pick the word "Washington" in three different sentences (a person, a location, a government reference). For each usage, list what features would differ between the three contexts. Which features are most discriminative? Notice that no single feature resolves the ambiguity — it takes a combination.

2. Design a feature set for detecting ORGANIZATION entities. Think about: what word shapes do company names have? What suffixes appear? What words typically come before or after organization names? Write at least 8 features.

3. Take two sentences where rule-based NER from Kata 6.1 failed (e.g., "Apple released a new product" or lowercased social media text). What features could a statistical system use that rules could not? Consider features like "the word after this one is a verb associated with companies" or "this word appears in a tech-news context."

---

## Live Code

```python
import re
from collections import defaultdict

# ============================================================
# FEATURE-BASED NER (NO EXTERNAL LIBRARIES)
# ============================================================
# Instead of writing rules, we extract features for each token
# and use weighted scoring to predict entity types.
#
# This demonstrates the core statistical NER idea:
#   features + weights = predictions
# ============================================================

# --- Gazetteers (reused from Kata 6.1 as features, not rules) ---

PERSON_TITLES = {"mr", "mrs", "ms", "dr", "prof", "sir", "captain", "judge", "ceo", "president"}

KNOWN_ORGS = {
    "infosys", "google", "microsoft", "amazon", "tesla", "apple",
    "united nations", "world health organization", "nasa", "unesco",
}

KNOWN_LOCATIONS = {
    "mumbai", "delhi", "bangalore", "chennai", "kolkata", "hyderabad",
    "new york", "london", "paris", "tokyo", "berlin", "sydney",
    "california", "texas", "india", "china", "geneva", "washington",
    "singapore", "toronto", "dubai", "jordan",
}

ORG_SUFFIXES = {"inc", "ltd", "corp", "llc", "co", "foundation", "institute"}

PERSON_NAME_INDICATORS = {"said", "told", "announced", "visited", "met", "joined", "presented"}

ORG_CONTEXT_WORDS = {"company", "firm", "corporation", "released", "announced", "acquired", "hired"}

LOCATION_CONTEXT_WORDS = {"in", "from", "to", "at", "near", "visited", "landed", "moved", "based"}

MONTHS = {"january", "february", "march", "april", "may", "june",
          "july", "august", "september", "october", "november", "december"}


def get_word_shape(word):
    """Convert a word to its shape pattern.
    Uppercase -> X, lowercase -> x, digit -> d, other -> itself.
    Consecutive same-type characters are collapsed."""
    shape = []
    for ch in word:
        if ch.isupper():
            s = "X"
        elif ch.islower():
            s = "x"
        elif ch.isdigit():
            s = "d"
        else:
            s = ch
        if not shape or shape[-1] != s:
            shape.append(s)
    return "".join(shape)


def tokenize(text):
    """Split text into word tokens."""
    tokens = []
    for match in re.finditer(r"[A-Za-z0-9]+(?:\.[A-Za-z])?\.?|[.,;:!?]", text):
        tokens.append(match.group())
    return tokens


def extract_features(tokens, index):
    """Extract features for the token at the given index."""
    word = tokens[index]
    word_lower = word.lower().rstrip(".")
    n = len(tokens)

    prev_word = tokens[index - 1] if index > 0 else "<START>"
    prev2_word = tokens[index - 2] if index > 1 else "<START>"
    next_word = tokens[index + 1] if index < n - 1 else "<END>"
    next2_word = tokens[index + 2] if index < n - 2 else "<END>"

    features = {}

    # --- Word-level features ---
    features["is_capitalized"] = word[0].isupper() if word else False
    features["is_all_caps"] = word.isupper() and len(word) > 1
    features["is_all_lower"] = word.islower()
    features["word_shape"] = get_word_shape(word)
    features["word_length"] = len(word)
    features["has_period"] = "." in word
    features["is_alpha"] = word.isalpha()
    features["position_ratio"] = round(index / max(n, 1), 2)
    features["is_first_token"] = index == 0

    # --- Gazetteer features ---
    features["in_person_titles"] = word_lower in PERSON_TITLES
    features["in_org_gazetteer"] = word_lower in KNOWN_ORGS
    features["in_loc_gazetteer"] = word_lower in KNOWN_LOCATIONS
    features["has_org_suffix"] = word_lower in ORG_SUFFIXES
    features["is_month"] = word_lower in MONTHS

    # --- Context features (previous word) ---
    prev_lower = prev_word.lower().rstrip(".")
    features["prev_is_capitalized"] = prev_word[0].isupper() if prev_word not in ("<START>",) else False
    features["prev_is_title"] = prev_lower in PERSON_TITLES
    features["prev_is_loc_prep"] = prev_lower in LOCATION_CONTEXT_WORDS
    features["prev_is_org_context"] = prev_lower in ORG_CONTEXT_WORDS

    # --- Context features (previous-previous word) ---
    prev2_lower = prev2_word.lower().rstrip(".")
    features["prev2_is_title"] = prev2_lower in PERSON_TITLES

    # --- Context features (next word) ---
    next_lower = next_word.lower().rstrip(".")
    features["next_is_capitalized"] = next_word[0].isupper() if next_word not in ("<END>",) else False
    features["next_is_person_verb"] = next_lower in PERSON_NAME_INDICATORS
    features["next_is_org_context"] = next_lower in ORG_CONTEXT_WORDS
    features["next_is_loc_prep"] = next_lower in LOCATION_CONTEXT_WORDS

    return features


# --- Weighted scoring: manually assigned weights ---
# In a real ML system, these weights would be learned from labeled data.
# Here we assign them by hand to demonstrate the concept.

WEIGHT_TABLE = {
    "PERSON": {
        "is_capitalized": 2.0,
        "is_all_caps": -0.5,
        "is_all_lower": -3.0,
        "prev_is_capitalized": 1.5,
        "prev_is_title": 4.0,
        "prev2_is_title": 2.5,
        "next_is_person_verb": 3.0,
        "next_is_capitalized": 1.0,
        "in_loc_gazetteer": -2.0,
        "in_org_gazetteer": -2.0,
        "is_month": -3.0,
        "is_first_token": -0.5,
    },
    "ORG": {
        "is_capitalized": 1.5,
        "is_all_caps": 2.0,
        "is_all_lower": -2.0,
        "in_org_gazetteer": 5.0,
        "has_org_suffix": 4.0,
        "next_is_org_context": 3.0,
        "prev_is_org_context": 2.0,
        "in_loc_gazetteer": -1.5,
        "prev_is_title": -1.0,
        "is_month": -3.0,
    },
    "LOCATION": {
        "is_capitalized": 1.5,
        "is_all_lower": -2.0,
        "in_loc_gazetteer": 5.0,
        "prev_is_loc_prep": 4.0,
        "next_is_loc_prep": 1.5,
        "in_org_gazetteer": -1.5,
        "prev_is_title": -2.0,
        "next_is_person_verb": -1.5,
        "is_month": -3.0,
    },
    "DATE": {
        "is_month": 6.0,
        "is_capitalized": -0.5,
        "in_loc_gazetteer": -2.0,
        "in_org_gazetteer": -2.0,
        "prev_is_title": -3.0,
    },
}

# Threshold: a token must score above this to be considered an entity
ENTITY_THRESHOLD = 3.0


def score_token(features, entity_type):
    """Compute weighted score for a token being a given entity type."""
    weights = WEIGHT_TABLE.get(entity_type, {})
    score = 0.0
    contributing = []
    for feat_name, feat_value in features.items():
        if feat_name in weights:
            if isinstance(feat_value, bool) and feat_value:
                score += weights[feat_name]
                contributing.append((feat_name, weights[feat_name]))
            elif isinstance(feat_value, (int, float)) and not isinstance(feat_value, bool) and feat_value > 0:
                score += weights[feat_name] * 0.5
                contributing.append((feat_name, weights[feat_name] * 0.5))
    return score, contributing


def predict_entities(text, verbose=False):
    """Predict entities using feature-based scoring."""
    tokens = tokenize(text)
    results = []

    for i, token in enumerate(tokens):
        # Skip punctuation
        if not token[0].isalpha():
            continue

        features = extract_features(tokens, i)

        # Score for each entity type
        scores = {}
        details = {}
        for etype in WEIGHT_TABLE:
            score, contrib = score_token(features, etype)
            scores[etype] = score
            details[etype] = contrib

        # Find best entity type
        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]

        if best_score >= ENTITY_THRESHOLD:
            results.append({
                "token": token,
                "label": best_type,
                "score": round(best_score, 1),
                "details": details[best_type],
                "all_scores": {k: round(v, 1) for k, v in scores.items()},
            })

    return results


def display_predictions(text, results, show_scores=False):
    """Display NER predictions."""
    print(f"  Text: {text}")
    if not results:
        print("  Entities: (none found)")
        return

    for r in results:
        score_info = f"  score={r['score']}" if show_scores else ""
        print(f"    [{r['label']:>8}]  \"{r['token']}\"{score_info}")

    if show_scores:
        print("    Score breakdown:")
        for r in results:
            top_features = sorted(r["details"], key=lambda x: -abs(x[1]))[:3]
            feat_str = ", ".join(f"{f}={w:+.1f}" for f, w in top_features)
            print(f"      \"{r['token']}\" -> {r['label']}: {feat_str}")


# ============================================================
# TEST: SENTENCES THAT WORK WELL
# ============================================================

test_sentences = [
    "Dr. Amara Khan works at Infosys in Bangalore.",
    "Prof. Raj Mehta presented at NASA in Washington.",
    "Tesla announced a new factory in Berlin on March 15.",
    "Ms. Priya Nair visited Google in California last June.",
    "The CEO joined Microsoft in Tokyo.",
]

print("=" * 65)
print("  FEATURE-BASED NER: PREDICTIONS WITH SCORING")
print("=" * 65)
print()

for sent in test_sentences:
    results = predict_entities(sent)
    display_predictions(sent, results, show_scores=True)
    print()


# ============================================================
# COMPARISON: FEATURE-BASED vs RULE-BASED ON TRICKY CASES
# ============================================================

tricky_sentences = [
    ("May I go to Paris in May?",
     "Rule-based: both 'May' tagged as DATE. Feature-based: first 'May' is sentence-initial + not preceded by prep = lower DATE score."),
    ("Jordan visited Jordan last summer.",
     "Rule-based: cannot distinguish. Feature-based: first 'Jordan' + verb 'visited' = PERSON signal; 'last summer' context for second."),
    ("Apple released a new product.",
     "Rule-based: misses 'Apple' (not in org list as standalone). Feature-based: 'released' is org-context = ORG signal."),
    ("Dr. Smith met Prof. Lee in Geneva.",
     "Both systems handle titles well, but feature-based also scores confidence."),
]

print("=" * 65)
print("  COMPARISON: FEATURE-BASED vs RULE-BASED")
print("=" * 65)
print()

for sent, comparison_note in tricky_sentences:
    results = predict_entities(sent)
    display_predictions(sent, results, show_scores=True)
    print(f"    COMPARISON: {comparison_note}")
    print()


# ============================================================
# FEATURE ANALYSIS: WHAT FEATURES MATTER MOST?
# ============================================================

print("=" * 65)
print("  FEATURE IMPORTANCE ANALYSIS")
print("=" * 65)
print()

feature_fires = defaultdict(lambda: defaultdict(int))
all_test = [s for s in test_sentences] + [s for s, _ in tricky_sentences]

for sent in all_test:
    tokens = tokenize(sent)
    for i, token in enumerate(tokens):
        if not token[0].isalpha():
            continue
        features = extract_features(tokens, i)
        for etype in WEIGHT_TABLE:
            _, contrib = score_token(features, etype)
            for feat_name, _ in contrib:
                feature_fires[etype][feat_name] += 1

print("  How often each feature contributes to each entity type:\n")
for etype in WEIGHT_TABLE:
    fires = feature_fires[etype]
    if not fires:
        continue
    sorted_feats = sorted(fires.items(), key=lambda x: -x[1])[:5]
    print(f"  {etype}:")
    for feat, count in sorted_feats:
        bar = "#" * count
        print(f"    {feat:<25} fires {count:>2}x  {bar}")
    print()

print("  Key differences from rule-based NER:")
print("    - Features combine multiple weak signals (no single feature decides)")
print("    - Context features (prev_word, next_word) help disambiguate")
print("    - Scores provide confidence, not just binary yes/no")
print("    - The same feature set handles all entity types (rules need separate logic)")
print()
print("  Limitation of this approach:")
print("    - Weights are hand-tuned (a real ML system learns them from data)")
print("    - Single-token entities only (multi-token spans need sequence labeling)")
print("    - Feature engineering still requires domain knowledge")
print("    - A real statistical NER uses BIO tagging and sequence models")
```

---

## Key Takeaways

- **Features replace rules.** Instead of writing "if capitalized and after a title, then PERSON," we extract features (is_capitalized, prev_is_title) and let weighted scoring combine them. This is more flexible and generalizes better to unseen text.
- **Context windows are powerful.** Looking at the words before and after a token provides signals that the token alone cannot. "Apple released" (ORG context) vs. "ate an apple" (not an entity) — the verb after the word changes everything.
- **Scores express confidence, not certainty.** A rule fires or it does not. A feature-based system produces a score, which lets you set thresholds, rank alternatives, and understand why a decision was made. This transparency is critical for error analysis.
- **Feature engineering is the bottleneck.** The system is only as good as the features you design. This motivates the move to neural methods (Phase 7+) that learn features automatically from data, instead of requiring human-designed feature templates.
