# NER Error Analysis

> Phase 6 — Named Entity Recognition (NER) | Kata 6.3

---

## Concept & Intuition

### What problem are we solving?

Building a NER system is only half the work. The other half is understanding **where it fails and why**. A system that correctly identifies 80% of entities sounds good until you discover it misses every organization name and mislabels locations as people. Aggregate accuracy hides critical patterns. Error analysis breaks open the black box.

In this kata, we take a NER system's predictions, compare them against gold-standard (human-annotated) labels, and systematically categorize every error. We compute precision, recall, and F1 per entity type. We classify errors into four categories: boundary errors, type confusion, missed entities, and false positives. This is how real NLP practitioners diagnose and improve their systems.

### Why earlier methods fail

Looking at overall accuracy tells you almost nothing actionable. If your system scores 75% F1, you do not know:
- Is it failing on PERSON, LOCATION, or ORG entities?
- Is it finding the right words but labeling them wrong (type confusion)?
- Is it finding the right labels but with wrong boundaries ("New" instead of "New York")?
- Is it hallucinating entities that do not exist (false positives)?
- Is it missing entities entirely (false negatives)?

Each of these failure modes has a different fix. Boundary errors suggest tokenization problems. Type confusion suggests feature overlap between categories. Missed entities suggest recall gaps. False positives suggest overly aggressive patterns. Without error analysis, you are debugging blindly.

### Mental models

- **Precision vs. recall tradeoff**: Precision asks "of the entities I found, how many are correct?" Recall asks "of the entities that exist, how many did I find?" A system that only labels entities it is very confident about has high precision but low recall. A system that labels everything as an entity has high recall but low precision.
- **F1 score**: The harmonic mean of precision and recall. It balances both: you cannot game F1 by optimizing only one side. F1 = 2 * (P * R) / (P + R).
- **Confusion matrix for NER**: A table showing how often each predicted type matches each gold type. Off-diagonal cells reveal systematic confusions (e.g., the system frequently labels LOCATION as ORG).
- **Error taxonomy**: Not all errors are equal. Boundary errors ("New" instead of "New York") are less severe than type errors ("New York" labeled as PERSON). A good error analysis distinguishes these.

### Visual explanations

```
Gold (human annotations):
  "Dr. Amara Khan works at the World Health Organization in Geneva."
        [-- PERSON --]       [------- ORG -------------]    [LOC -]

System predictions:
  "Dr. Amara Khan works at the World Health Organization in Geneva."
   [-- PERSON ---]             [--- ORG --]                 [LOC -]
                                (missed "World")

Error analysis:
  Entity 1: Gold=[Amara Khan, PERSON]  Pred=[Dr. Amara Khan, PERSON]
    -> BOUNDARY ERROR: prediction includes "Dr." which is not part of the name

  Entity 2: Gold=[World Health Organization, ORG]  Pred=[Health Organization, ORG]
    -> BOUNDARY ERROR: prediction missed "World"

  Entity 3: Gold=[Geneva, LOCATION]  Pred=[Geneva, LOCATION]
    -> CORRECT

Precision, Recall, F1 (per type):
  ┌──────────┬───────────┬────────┬──────┐
  │ Type     │ Precision │ Recall │  F1  │
  ├──────────┼───────────┼────────┼──────┤
  │ PERSON   │ 1/1=1.00  │ 1/1=1.00│ 1.00│  (boundary off, but span overlaps)
  │ ORG      │ 1/1=1.00  │ 1/1=1.00│ 1.00│  (boundary off, but detected)
  │ LOCATION │ 1/1=1.00  │ 1/1=1.00│ 1.00│  (exact match)
  └──────────┴───────────┴────────┴──────┘

  But with STRICT matching (exact boundaries required):
  │ PERSON   │ 0/1=0.00  │ 0/1=0.00│ 0.00│  (boundary mismatch)
  │ ORG      │ 0/1=0.00  │ 0/1=0.00│ 0.00│  (boundary mismatch)

  Matching strategy matters enormously for reported scores.
```

---

## Hands-on Exploration

1. Given these gold and predicted entities, compute precision, recall, and F1 by hand for each type:
   - Gold: [("Priya", PERSON), ("Infosys", ORG), ("Bangalore", LOCATION), ("March", DATE)]
   - Predicted: [("Priya", PERSON), ("Infosys", LOCATION), ("March", DATE), ("India", LOCATION)]
   - Which entity was missed? Which was mislabeled? Which was hallucinated?

2. Consider a system that never predicts ORG entities but perfectly predicts PERSON and LOCATION. What would its per-type F1 scores look like? What would the macro-average F1 be? Why is per-type analysis essential?

3. Think about two systems: System A has 90% precision and 50% recall. System B has 60% precision and 85% recall. Which is better for a medical NER task where missing an entity (a drug name) is dangerous? Which is better for a search engine where showing wrong entities annoys users? How does the application context change which metric matters?

---

## Live Code

```python
from collections import defaultdict

# ============================================================
# NER ERROR ANALYSIS
# ============================================================
# Given gold-standard annotations and system predictions,
# we compute per-type precision/recall/F1, build a confusion
# matrix, and categorize every error.
# ============================================================

# --- Gold-standard annotations and system predictions ---
# Format: list of (sentence, gold_entities, predicted_entities)
# Each entity: (text, start_char, end_char, label)

EVALUATION_DATA = [
    {
        "text": "Dr. Amara Khan joined Infosys in Bangalore last March.",
        "gold": [
            ("Amara Khan", 4, 14, "PERSON"),
            ("Infosys", 22, 29, "ORG"),
            ("Bangalore", 33, 42, "LOCATION"),
            ("March", 48, 53, "DATE"),
        ],
        "pred": [
            ("Dr. Amara Khan", 0, 14, "PERSON"),   # boundary error: includes "Dr."
            ("Infosys", 22, 29, "ORG"),             # correct
            ("Bangalore", 33, 42, "LOCATION"),      # correct
            ("March", 48, 53, "DATE"),              # correct
        ],
    },
    {
        "text": "Tesla CEO Elon Musk visited the United Nations in New York.",
        "gold": [
            ("Tesla", 0, 5, "ORG"),
            ("Elon Musk", 10, 19, "PERSON"),
            ("United Nations", 32, 46, "ORG"),
            ("New York", 50, 58, "LOCATION"),
        ],
        "pred": [
            ("Tesla", 0, 5, "ORG"),                # correct
            ("Elon Musk", 10, 19, "PERSON"),        # correct
            ("United Nations", 32, 46, "ORG"),      # correct
            ("New York", 50, 58, "PERSON"),         # type error: PERSON instead of LOCATION
        ],
    },
    {
        "text": "Prof. Lee presented at NASA headquarters in Washington on June 5.",
        "gold": [
            ("Lee", 6, 9, "PERSON"),
            ("NASA", 23, 27, "ORG"),
            ("Washington", 44, 54, "LOCATION"),
            ("June 5", 58, 64, "DATE"),
        ],
        "pred": [
            ("Prof. Lee", 0, 9, "PERSON"),          # boundary error
            ("NASA", 23, 27, "ORG"),                # correct
            ("Washington", 44, 54, "LOCATION"),     # correct
            # missed: June 5 (DATE)
        ],
    },
    {
        "text": "Apple announced record earnings in California.",
        "gold": [
            ("Apple", 0, 5, "ORG"),
            ("California", 35, 45, "LOCATION"),
        ],
        "pred": [
            # missed: Apple (ORG)
            ("California", 35, 45, "LOCATION"),     # correct
        ],
    },
    {
        "text": "May I visit Paris in May?",
        "gold": [
            ("Paris", 12, 17, "LOCATION"),
            ("May", 21, 24, "DATE"),
        ],
        "pred": [
            ("May", 0, 3, "DATE"),                  # false positive: "May" (verb) tagged as DATE
            ("Paris", 12, 17, "LOCATION"),           # correct
            ("May", 21, 24, "DATE"),                 # correct
        ],
    },
    {
        "text": "Jordan flew to Jordan for a meeting with the Red Cross.",
        "gold": [
            ("Jordan", 0, 6, "PERSON"),
            ("Jordan", 14, 20, "LOCATION"),
            ("Red Cross", 43, 52, "ORG"),
        ],
        "pred": [
            ("Jordan", 0, 6, "LOCATION"),           # type error: LOCATION instead of PERSON
            ("Jordan", 14, 20, "LOCATION"),          # correct
            # missed: Red Cross (ORG)
        ],
    },
    {
        "text": "Ms. Priya Nair works at Google in Tokyo.",
        "gold": [
            ("Priya Nair", 4, 14, "PERSON"),
            ("Google", 24, 30, "ORG"),
            ("Tokyo", 34, 39, "LOCATION"),
        ],
        "pred": [
            ("Priya Nair", 4, 14, "PERSON"),        # correct
            ("Google", 24, 30, "ORG"),               # correct
            ("Tokyo", 34, 39, "LOCATION"),           # correct
            ("Nair", 10, 14, "PERSON"),              # false positive: duplicate partial
        ],
    },
]


# ============================================================
# MATCHING: COMPARE GOLD TO PREDICTED ENTITIES
# ============================================================

def match_entities(gold_entities, pred_entities, strict=True):
    """Match gold entities to predicted entities.

    strict=True: exact span match required
    strict=False: any overlap counts as a match
    """
    matches = []         # (gold, pred, match_type)
    unmatched_gold = []  # missed entities
    unmatched_pred = []  # false positives

    pred_used = [False] * len(pred_entities)

    for g_text, g_start, g_end, g_label in gold_entities:
        best_match = None
        best_idx = -1

        for j, (p_text, p_start, p_end, p_label) in enumerate(pred_entities):
            if pred_used[j]:
                continue

            if strict:
                # Exact boundary match
                if g_start == p_start and g_end == p_end:
                    best_match = (p_text, p_start, p_end, p_label)
                    best_idx = j
                    break
            else:
                # Overlap match: any character overlap
                overlap = max(0, min(g_end, p_end) - max(g_start, p_start))
                if overlap > 0:
                    best_match = (p_text, p_start, p_end, p_label)
                    best_idx = j
                    break

        if best_match:
            pred_used[best_idx] = True
            if g_label == best_match[3]:
                match_type = "correct"
            else:
                match_type = "type_error"
            matches.append(((g_text, g_start, g_end, g_label), best_match, match_type))
        else:
            unmatched_gold.append((g_text, g_start, g_end, g_label))

    for j, (p_text, p_start, p_end, p_label) in enumerate(pred_entities):
        if not pred_used[j]:
            unmatched_pred.append((p_text, p_start, p_end, p_label))

    return matches, unmatched_gold, unmatched_pred


# ============================================================
# COMPUTE PRECISION, RECALL, F1 PER ENTITY TYPE
# ============================================================

def compute_metrics(evaluation_data, strict=True):
    """Compute per-type and overall P/R/F1."""
    type_tp = defaultdict(int)      # true positives per type
    type_fp = defaultdict(int)      # false positives per type
    type_fn = defaultdict(int)      # false negatives per type
    type_type_err = defaultdict(int)  # type confusion counts

    all_errors = []

    for item in evaluation_data:
        gold = item["gold"]
        pred = item["pred"]
        text = item["text"]

        matches, missed, false_pos = match_entities(gold, pred, strict=strict)

        for g_ent, p_ent, match_type in matches:
            if match_type == "correct":
                type_tp[g_ent[3]] += 1
            else:
                # Type error: gold says one type, pred says another
                type_fn[g_ent[3]] += 1
                type_fp[p_ent[3]] += 1
                type_type_err[(g_ent[3], p_ent[3])] += 1
                all_errors.append({
                    "text": text,
                    "category": "TYPE_CONFUSION",
                    "gold": f"{g_ent[0]} [{g_ent[3]}]",
                    "pred": f"{p_ent[0]} [{p_ent[3]}]",
                })

        for g_ent in missed:
            type_fn[g_ent[3]] += 1
            all_errors.append({
                "text": text,
                "category": "MISSED_ENTITY",
                "gold": f"{g_ent[0]} [{g_ent[3]}]",
                "pred": "(none)",
            })

        for p_ent in false_pos:
            type_fp[p_ent[3]] += 1
            all_errors.append({
                "text": text,
                "category": "FALSE_POSITIVE",
                "gold": "(none)",
                "pred": f"{p_ent[0]} [{p_ent[3]}]",
            })

    # Compute P/R/F1 per type
    all_types = sorted(set(list(type_tp.keys()) + list(type_fp.keys()) + list(type_fn.keys())))
    metrics = {}
    for t in all_types:
        tp = type_tp[t]
        fp = type_fp[t]
        fn = type_fn[t]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        metrics[t] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
        }

    return metrics, all_errors, type_type_err


# ============================================================
# RUN EVALUATION: STRICT MATCHING
# ============================================================

print("=" * 65)
print("  NER ERROR ANALYSIS: STRICT MATCHING")
print("  (exact span boundaries required)")
print("=" * 65)
print()

metrics_strict, errors_strict, confusion_strict = compute_metrics(EVALUATION_DATA, strict=True)

print("  Per-type metrics:")
print(f"  {'Type':<10} {'TP':>4} {'FP':>4} {'FN':>4} {'Prec':>7} {'Rec':>7} {'F1':>7}")
print("  " + "-" * 46)

total_tp = total_fp = total_fn = 0
f1_sum = 0
for t, m in sorted(metrics_strict.items()):
    print(f"  {t:<10} {m['tp']:>4} {m['fp']:>4} {m['fn']:>4} "
          f"{m['precision']:>7.2f} {m['recall']:>7.2f} {m['f1']:>7.2f}")
    total_tp += m["tp"]
    total_fp += m["fp"]
    total_fn += m["fn"]
    f1_sum += m["f1"]

micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0
macro_f1 = f1_sum / len(metrics_strict) if metrics_strict else 0

print("  " + "-" * 46)
print(f"  {'Micro':>10} {total_tp:>4} {total_fp:>4} {total_fn:>4} "
      f"{micro_p:>7.2f} {micro_r:>7.2f} {micro_f1:>7.2f}")
print(f"  {'Macro F1':>10} {'':>4} {'':>4} {'':>4} {'':>7} {'':>7} {macro_f1:>7.2f}")
print()

# --- Per-type P/R/F1 Chart (Strict) ---
strict_types = sorted(metrics_strict.keys())
strict_precision = [round(metrics_strict[t]["precision"], 2) for t in strict_types]
strict_recall = [round(metrics_strict[t]["recall"], 2) for t in strict_types]
strict_f1 = [round(metrics_strict[t]["f1"], 2) for t in strict_types]

show_chart({
    "type": "bar",
    "title": "Per-Entity-Type Metrics (Strict Matching)",
    "labels": strict_types,
    "datasets": [
        {"label": "Precision", "data": strict_precision, "color": "#3b82f6"},
        {"label": "Recall", "data": strict_recall, "color": "#10b981"},
        {"label": "F1", "data": strict_f1, "color": "#f59e0b"},
    ],
    "options": {"x_label": "Entity Type", "y_label": "Score"}
})


# ============================================================
# RUN EVALUATION: RELAXED MATCHING (overlap-based)
# ============================================================

print("=" * 65)
print("  NER ERROR ANALYSIS: RELAXED MATCHING")
print("  (any span overlap counts as match)")
print("=" * 65)
print()

metrics_relaxed, errors_relaxed, confusion_relaxed = compute_metrics(EVALUATION_DATA, strict=False)

print("  Per-type metrics:")
print(f"  {'Type':<10} {'TP':>4} {'FP':>4} {'FN':>4} {'Prec':>7} {'Rec':>7} {'F1':>7}")
print("  " + "-" * 46)

total_tp = total_fp = total_fn = 0
f1_sum = 0
for t, m in sorted(metrics_relaxed.items()):
    print(f"  {t:<10} {m['tp']:>4} {m['fp']:>4} {m['fn']:>4} "
          f"{m['precision']:>7.2f} {m['recall']:>7.2f} {m['f1']:>7.2f}")
    total_tp += m["tp"]
    total_fp += m["fp"]
    total_fn += m["fn"]
    f1_sum += m["f1"]

micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0
macro_f1 = f1_sum / len(metrics_relaxed) if metrics_relaxed else 0

print("  " + "-" * 46)
print(f"  {'Micro':>10} {total_tp:>4} {total_fp:>4} {total_fn:>4} "
      f"{micro_p:>7.2f} {micro_r:>7.2f} {micro_f1:>7.2f}")
print(f"  {'Macro F1':>10} {'':>4} {'':>4} {'':>4} {'':>7} {'':>7} {macro_f1:>7.2f}")
print()

print("  NOTE: Relaxed matching gives higher scores because boundary")
print("  errors (e.g., 'Dr. Amara Khan' vs 'Amara Khan') count as matches.")
print("  The gap between strict and relaxed reveals boundary error rate.")
print()

# --- Per-type P/R/F1 Chart (Relaxed) ---
relaxed_types = sorted(metrics_relaxed.keys())
relaxed_precision = [round(metrics_relaxed[t]["precision"], 2) for t in relaxed_types]
relaxed_recall = [round(metrics_relaxed[t]["recall"], 2) for t in relaxed_types]
relaxed_f1 = [round(metrics_relaxed[t]["f1"], 2) for t in relaxed_types]

show_chart({
    "type": "bar",
    "title": "Per-Entity-Type Metrics (Relaxed Matching)",
    "labels": relaxed_types,
    "datasets": [
        {"label": "Precision", "data": relaxed_precision, "color": "#3b82f6"},
        {"label": "Recall", "data": relaxed_recall, "color": "#10b981"},
        {"label": "F1", "data": relaxed_f1, "color": "#f59e0b"},
    ],
    "options": {"x_label": "Entity Type", "y_label": "Score"}
})


# ============================================================
# ERROR CATEGORIZATION
# ============================================================

print("=" * 65)
print("  ERROR CATEGORIZATION (strict matching)")
print("=" * 65)
print()

# Detect boundary errors by comparing strict vs relaxed
boundary_errors = []
for item in EVALUATION_DATA:
    gold = item["gold"]
    pred = item["pred"]
    text = item["text"]

    strict_matches, _, _ = match_entities(gold, pred, strict=True)
    relaxed_matches, _, _ = match_entities(gold, pred, strict=False)

    strict_pairs = {(g[0], g[3]) for g, p, mt in strict_matches if mt == "correct"}
    relaxed_pairs = {(g[0], g[3]) for g, p, mt in relaxed_matches if mt == "correct"}

    # Entities that match with relaxed but not strict = boundary errors
    for g, p, mt in relaxed_matches:
        if mt == "correct" and (g[0], g[3]) not in strict_pairs:
            if g[0] != p[0]:
                boundary_errors.append({
                    "text": text,
                    "gold_span": g[0],
                    "pred_span": p[0],
                    "label": g[3],
                })

# Count error categories
category_counts = defaultdict(int)
for e in errors_strict:
    category_counts[e["category"]] += 1

print("  Error counts by category:")
for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
    bar = "#" * (count * 3)
    print(f"    {cat:<20} {count:>3}  {bar}")
print()

if boundary_errors:
    print(f"  Boundary errors detected: {len(boundary_errors)}")
    for be in boundary_errors:
        print(f"    Gold: \"{be['gold_span']}\" [{be['label']}]  "
              f"Pred: \"{be['pred_span']}\" [{be['label']}]")
    print()

print("  Detailed error list (strict matching):")
print()
for e in errors_strict:
    print(f"    [{e['category']:<17}]  Gold: {e['gold']:<30}  Pred: {e['pred']}")
print()


# ============================================================
# CONFUSION MATRIX FOR ENTITY TYPES
# ============================================================

print("=" * 65)
print("  ENTITY TYPE CONFUSION MATRIX")
print("=" * 65)
print()

# Build confusion from relaxed matching (to capture type errors)
all_types = sorted(set(
    [g[3] for item in EVALUATION_DATA for g in item["gold"]] +
    [p[3] for item in EVALUATION_DATA for p in item["pred"]]
))

confusion = defaultdict(lambda: defaultdict(int))

for item in EVALUATION_DATA:
    matches, missed, false_pos = match_entities(item["gold"], item["pred"], strict=False)
    for g, p, mt in matches:
        confusion[g[3]][p[3]] += 1
    for g in missed:
        confusion[g[3]]["(MISSED)"] += 1
    for p in false_pos:
        confusion["(NONE)"][p[3]] += 1

col_labels = all_types + ["(MISSED)"]
print(f"  {'Gold \\ Pred':<12}", end="")
for col in col_labels:
    print(f"{col:>10}", end="")
print()
print("  " + "-" * (12 + 10 * len(col_labels)))

for row in all_types + ["(NONE)"]:
    print(f"  {row:<12}", end="")
    for col in col_labels:
        val = confusion[row].get(col, 0)
        if val > 0:
            cell = str(val)
        else:
            cell = "."
        print(f"{cell:>10}", end="")
    print()

print()
print("  Reading the matrix:")
print("    - Diagonal = correct predictions")
print("    - Off-diagonal = type confusions (gold type != predicted type)")
print("    - (MISSED) column = entities the system failed to detect")
print("    - (NONE) row = false positives (system predicted an entity where none exists)")
print()

# --- Confusion Matrix Heatmap ---
heatmap_rows = all_types + ["(NONE)"]
heatmap_cols = col_labels
heatmap_data = []
for row in heatmap_rows:
    row_data = []
    for col in heatmap_cols:
        row_data.append(confusion[row].get(col, 0))
    heatmap_data.append(row_data)

show_chart({
    "type": "heatmap",
    "title": "Entity Type Confusion Matrix",
    "x_labels": heatmap_cols,
    "y_labels": heatmap_rows,
    "data": heatmap_data,
    "color_scale": "blue",
})


# ============================================================
# ACTIONABLE INSIGHTS
# ============================================================

print("=" * 65)
print("  ACTIONABLE INSIGHTS")
print("=" * 65)
print()

# Find weakest entity type
worst_type = min(metrics_strict.items(), key=lambda x: x[1]["f1"])
best_type = max(metrics_strict.items(), key=lambda x: x[1]["f1"])

print(f"  Weakest entity type:  {worst_type[0]} (F1={worst_type[1]['f1']:.2f})")
print(f"  Strongest entity type: {best_type[0]} (F1={best_type[1]['f1']:.2f})")
print()

# Identify dominant error category
if category_counts:
    dominant_error = max(category_counts.items(), key=lambda x: x[1])
    print(f"  Most common error: {dominant_error[0]} ({dominant_error[1]} instances)")
    print()

    if dominant_error[0] == "MISSED_ENTITY":
        print("  Recommendation: Improve recall — add more patterns/features")
        print("  for detecting entities. Consider expanding gazetteers or")
        print("  lowering confidence thresholds.")
    elif dominant_error[0] == "FALSE_POSITIVE":
        print("  Recommendation: Improve precision — add negative constraints")
        print("  or raise confidence thresholds. Check for common false")
        print("  triggers (e.g., 'May' as verb vs month).")
    elif dominant_error[0] == "TYPE_CONFUSION":
        print("  Recommendation: Improve type discrimination — add features")
        print("  that distinguish between confused types. Check the confusion")
        print("  matrix to identify which types are being confused.")

print()

# Strict vs relaxed gap
strict_f1s = [m["f1"] for m in metrics_strict.values()]
relaxed_f1s = [m["f1"] for m in metrics_relaxed.values()]
strict_avg = sum(strict_f1s) / len(strict_f1s) if strict_f1s else 0
relaxed_avg = sum(relaxed_f1s) / len(relaxed_f1s) if relaxed_f1s else 0
gap = relaxed_avg - strict_avg

print(f"  Strict vs Relaxed F1 gap: {gap:.2f}")
if gap > 0.1:
    print("  Large gap indicates significant boundary errors.")
    print("  Recommendation: Review tokenization and span detection logic.")
elif gap > 0.0:
    print("  Small gap indicates some boundary errors exist but type")
    print("  detection is the bigger issue.")
else:
    print("  No gap: boundaries are accurate, focus on type prediction.")

print()
print("  Error analysis is not a one-time task. Every time you change")
print("  your NER system, re-run this analysis to see if your changes")
print("  helped the errors you targeted without creating new ones.")
```

---

## Key Takeaways

- **Aggregate scores hide critical failures.** A system with 75% overall F1 might have 95% F1 on PERSON and 30% F1 on ORG. Per-type analysis reveals where the system actually needs work, and without it you are optimizing blindly.
- **Error categories drive different fixes.** Boundary errors call for better tokenization. Type confusion calls for better discriminative features. Missed entities call for higher recall. False positives call for tighter constraints. Knowing which category dominates tells you exactly where to invest effort.
- **Matching strategy changes your numbers.** Strict matching (exact boundaries) gives lower scores than relaxed matching (any overlap). The gap between them quantifies your boundary error rate. Always report which matching strategy you used — a "90% F1" score means nothing without this context.
- **Error analysis is iterative.** Fix the dominant error category, re-evaluate, and the next dominant category will emerge. This cycle of analyze-fix-reanalyze is how real NLP systems improve incrementally, and it applies equally to rule-based, statistical, and neural NER.
