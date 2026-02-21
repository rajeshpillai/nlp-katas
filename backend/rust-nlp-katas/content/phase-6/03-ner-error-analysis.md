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
  +----------+-----------+--------+------+
  | Type     | Precision | Recall |  F1  |
  +----------+-----------+--------+------+
  | PERSON   | 1/1=1.00  | 1/1=1.00| 1.00|  (boundary off, but span overlaps)
  | ORG      | 1/1=1.00  | 1/1=1.00| 1.00|  (boundary off, but detected)
  | LOCATION | 1/1=1.00  | 1/1=1.00| 1.00|  (exact match)
  +----------+-----------+--------+------+

  But with STRICT matching (exact boundaries required):
  | PERSON   | 0/1=0.00  | 0/1=0.00| 0.00|  (boundary mismatch)
  | ORG      | 0/1=0.00  | 0/1=0.00| 0.00|  (boundary mismatch)

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

```rust
use std::collections::HashMap;

fn main() {
    // ============================================================
    // NER ERROR ANALYSIS
    // ============================================================
    // Given gold-standard annotations and system predictions,
    // we compute per-type precision/recall/F1, build a confusion
    // matrix, and categorize every error.
    // ============================================================

    // --- Gold-standard annotations and system predictions ---
    // Each entity: (text, start_char, end_char, label)

    #[derive(Clone, Debug)]
    struct EntitySpan {
        text: String,
        start: usize,
        end: usize,
        label: String,
    }

    struct EvalItem {
        text: String,
        gold: Vec<EntitySpan>,
        pred: Vec<EntitySpan>,
    }

    let evaluation_data: Vec<EvalItem> = vec![
        EvalItem {
            text: "Dr. Amara Khan joined Infosys in Bangalore last March.".to_string(),
            gold: vec![
                EntitySpan { text: "Amara Khan".into(), start: 4, end: 14, label: "PERSON".into() },
                EntitySpan { text: "Infosys".into(), start: 22, end: 29, label: "ORG".into() },
                EntitySpan { text: "Bangalore".into(), start: 33, end: 42, label: "LOCATION".into() },
                EntitySpan { text: "March".into(), start: 48, end: 53, label: "DATE".into() },
            ],
            pred: vec![
                EntitySpan { text: "Dr. Amara Khan".into(), start: 0, end: 14, label: "PERSON".into() },
                EntitySpan { text: "Infosys".into(), start: 22, end: 29, label: "ORG".into() },
                EntitySpan { text: "Bangalore".into(), start: 33, end: 42, label: "LOCATION".into() },
                EntitySpan { text: "March".into(), start: 48, end: 53, label: "DATE".into() },
            ],
        },
        EvalItem {
            text: "Tesla CEO Elon Musk visited the United Nations in New York.".to_string(),
            gold: vec![
                EntitySpan { text: "Tesla".into(), start: 0, end: 5, label: "ORG".into() },
                EntitySpan { text: "Elon Musk".into(), start: 10, end: 19, label: "PERSON".into() },
                EntitySpan { text: "United Nations".into(), start: 32, end: 46, label: "ORG".into() },
                EntitySpan { text: "New York".into(), start: 50, end: 58, label: "LOCATION".into() },
            ],
            pred: vec![
                EntitySpan { text: "Tesla".into(), start: 0, end: 5, label: "ORG".into() },
                EntitySpan { text: "Elon Musk".into(), start: 10, end: 19, label: "PERSON".into() },
                EntitySpan { text: "United Nations".into(), start: 32, end: 46, label: "ORG".into() },
                EntitySpan { text: "New York".into(), start: 50, end: 58, label: "PERSON".into() },
            ],
        },
        EvalItem {
            text: "Prof. Lee presented at NASA headquarters in Washington on June 5.".to_string(),
            gold: vec![
                EntitySpan { text: "Lee".into(), start: 6, end: 9, label: "PERSON".into() },
                EntitySpan { text: "NASA".into(), start: 23, end: 27, label: "ORG".into() },
                EntitySpan { text: "Washington".into(), start: 44, end: 54, label: "LOCATION".into() },
                EntitySpan { text: "June 5".into(), start: 58, end: 64, label: "DATE".into() },
            ],
            pred: vec![
                EntitySpan { text: "Prof. Lee".into(), start: 0, end: 9, label: "PERSON".into() },
                EntitySpan { text: "NASA".into(), start: 23, end: 27, label: "ORG".into() },
                EntitySpan { text: "Washington".into(), start: 44, end: 54, label: "LOCATION".into() },
                // missed: June 5 (DATE)
            ],
        },
        EvalItem {
            text: "Apple announced record earnings in California.".to_string(),
            gold: vec![
                EntitySpan { text: "Apple".into(), start: 0, end: 5, label: "ORG".into() },
                EntitySpan { text: "California".into(), start: 35, end: 45, label: "LOCATION".into() },
            ],
            pred: vec![
                // missed: Apple (ORG)
                EntitySpan { text: "California".into(), start: 35, end: 45, label: "LOCATION".into() },
            ],
        },
        EvalItem {
            text: "May I visit Paris in May?".to_string(),
            gold: vec![
                EntitySpan { text: "Paris".into(), start: 12, end: 17, label: "LOCATION".into() },
                EntitySpan { text: "May".into(), start: 21, end: 24, label: "DATE".into() },
            ],
            pred: vec![
                EntitySpan { text: "May".into(), start: 0, end: 3, label: "DATE".into() },
                EntitySpan { text: "Paris".into(), start: 12, end: 17, label: "LOCATION".into() },
                EntitySpan { text: "May".into(), start: 21, end: 24, label: "DATE".into() },
            ],
        },
        EvalItem {
            text: "Jordan flew to Jordan for a meeting with the Red Cross.".to_string(),
            gold: vec![
                EntitySpan { text: "Jordan".into(), start: 0, end: 6, label: "PERSON".into() },
                EntitySpan { text: "Jordan".into(), start: 14, end: 20, label: "LOCATION".into() },
                EntitySpan { text: "Red Cross".into(), start: 43, end: 52, label: "ORG".into() },
            ],
            pred: vec![
                EntitySpan { text: "Jordan".into(), start: 0, end: 6, label: "LOCATION".into() },
                EntitySpan { text: "Jordan".into(), start: 14, end: 20, label: "LOCATION".into() },
                // missed: Red Cross (ORG)
            ],
        },
        EvalItem {
            text: "Ms. Priya Nair works at Google in Tokyo.".to_string(),
            gold: vec![
                EntitySpan { text: "Priya Nair".into(), start: 4, end: 14, label: "PERSON".into() },
                EntitySpan { text: "Google".into(), start: 24, end: 30, label: "ORG".into() },
                EntitySpan { text: "Tokyo".into(), start: 34, end: 39, label: "LOCATION".into() },
            ],
            pred: vec![
                EntitySpan { text: "Priya Nair".into(), start: 4, end: 14, label: "PERSON".into() },
                EntitySpan { text: "Google".into(), start: 24, end: 30, label: "ORG".into() },
                EntitySpan { text: "Tokyo".into(), start: 34, end: 39, label: "LOCATION".into() },
                EntitySpan { text: "Nair".into(), start: 10, end: 14, label: "PERSON".into() },
            ],
        },
    ];

    // ============================================================
    // MATCHING: COMPARE GOLD TO PREDICTED ENTITIES
    // ============================================================

    #[derive(Clone)]
    struct MatchResult {
        gold: EntitySpan,
        pred: EntitySpan,
        match_type: String, // "correct" or "type_error"
    }

    fn match_entities(gold: &[EntitySpan], pred: &[EntitySpan], strict: bool) -> (Vec<MatchResult>, Vec<EntitySpan>, Vec<EntitySpan>) {
        let mut matches = Vec::new();
        let mut unmatched_gold = Vec::new();
        let mut pred_used = vec![false; pred.len()];

        for g in gold {
            let mut best_match: Option<(usize, &EntitySpan)> = None;
            for (j, p) in pred.iter().enumerate() {
                if pred_used[j] { continue; }
                if strict {
                    if g.start == p.start && g.end == p.end {
                        best_match = Some((j, p));
                        break;
                    }
                } else {
                    let overlap_start = g.start.max(p.start);
                    let overlap_end = g.end.min(p.end);
                    if overlap_end > overlap_start {
                        best_match = Some((j, p));
                        break;
                    }
                }
            }

            if let Some((j, p)) = best_match {
                pred_used[j] = true;
                let mt = if g.label == p.label { "correct" } else { "type_error" };
                matches.push(MatchResult { gold: g.clone(), pred: p.clone(), match_type: mt.to_string() });
            } else {
                unmatched_gold.push(g.clone());
            }
        }

        let unmatched_pred: Vec<EntitySpan> = pred.iter().enumerate()
            .filter(|(j, _)| !pred_used[*j])
            .map(|(_, p)| p.clone())
            .collect();

        (matches, unmatched_gold, unmatched_pred)
    }

    // ============================================================
    // COMPUTE PRECISION, RECALL, F1 PER ENTITY TYPE
    // ============================================================

    struct TypeMetrics {
        tp: usize,
        fp: usize,
        fn_count: usize,
        precision: f64,
        recall: f64,
        f1: f64,
    }

    struct ErrorEntry {
        text: String,
        category: String,
        gold: String,
        pred: String,
    }

    fn compute_metrics(data: &[EvalItem], strict: bool) -> (HashMap<String, TypeMetrics>, Vec<ErrorEntry>, HashMap<(String, String), usize>) {
        let mut type_tp: HashMap<String, usize> = HashMap::new();
        let mut type_fp: HashMap<String, usize> = HashMap::new();
        let mut type_fn: HashMap<String, usize> = HashMap::new();
        let mut type_type_err: HashMap<(String, String), usize> = HashMap::new();
        let mut all_errors = Vec::new();

        for item in data {
            let (matches, missed, false_pos) = match_entities(&item.gold, &item.pred, strict);

            for m in &matches {
                if m.match_type == "correct" {
                    *type_tp.entry(m.gold.label.clone()).or_insert(0) += 1;
                } else {
                    *type_fn.entry(m.gold.label.clone()).or_insert(0) += 1;
                    *type_fp.entry(m.pred.label.clone()).or_insert(0) += 1;
                    *type_type_err.entry((m.gold.label.clone(), m.pred.label.clone())).or_insert(0) += 1;
                    all_errors.push(ErrorEntry {
                        text: item.text.clone(),
                        category: "TYPE_CONFUSION".to_string(),
                        gold: format!("{} [{}]", m.gold.text, m.gold.label),
                        pred: format!("{} [{}]", m.pred.text, m.pred.label),
                    });
                }
            }

            for g in &missed {
                *type_fn.entry(g.label.clone()).or_insert(0) += 1;
                all_errors.push(ErrorEntry {
                    text: item.text.clone(),
                    category: "MISSED_ENTITY".to_string(),
                    gold: format!("{} [{}]", g.text, g.label),
                    pred: "(none)".to_string(),
                });
            }

            for p in &false_pos {
                *type_fp.entry(p.label.clone()).or_insert(0) += 1;
                all_errors.push(ErrorEntry {
                    text: item.text.clone(),
                    category: "FALSE_POSITIVE".to_string(),
                    gold: "(none)".to_string(),
                    pred: format!("{} [{}]", p.text, p.label),
                });
            }
        }

        let mut all_types = std::collections::BTreeSet::new();
        for k in type_tp.keys() { all_types.insert(k.clone()); }
        for k in type_fp.keys() { all_types.insert(k.clone()); }
        for k in type_fn.keys() { all_types.insert(k.clone()); }

        let mut metrics = HashMap::new();
        for t in &all_types {
            let tp = *type_tp.get(t).unwrap_or(&0);
            let fp = *type_fp.get(t).unwrap_or(&0);
            let fn_c = *type_fn.get(t).unwrap_or(&0);
            let precision = if tp + fp > 0 { tp as f64 / (tp + fp) as f64 } else { 0.0 };
            let recall = if tp + fn_c > 0 { tp as f64 / (tp + fn_c) as f64 } else { 0.0 };
            let f1 = if precision + recall > 0.0 { 2.0 * precision * recall / (precision + recall) } else { 0.0 };
            metrics.insert(t.clone(), TypeMetrics { tp, fp, fn_count: fn_c, precision, recall, f1 });
        }

        (metrics, all_errors, type_type_err)
    }

    // ============================================================
    // RUN EVALUATION: STRICT MATCHING
    // ============================================================

    println!("{}", "=".repeat(65));
    println!("  NER ERROR ANALYSIS: STRICT MATCHING");
    println!("  (exact span boundaries required)");
    println!("{}", "=".repeat(65));
    println!();

    let (metrics_strict, errors_strict, _confusion_strict) = compute_metrics(&evaluation_data, true);

    println!("  Per-type metrics:");
    println!("  {:<10} {:>4} {:>4} {:>4} {:>7} {:>7} {:>7}", "Type", "TP", "FP", "FN", "Prec", "Rec", "F1");
    println!("  {}", "-".repeat(46));

    let mut total_tp = 0;
    let mut total_fp = 0;
    let mut total_fn = 0;
    let mut f1_sum = 0.0;
    let mut sorted_types: Vec<&String> = metrics_strict.keys().collect();
    sorted_types.sort();
    for t in &sorted_types {
        let m = &metrics_strict[*t];
        println!("  {:<10} {:>4} {:>4} {:>4} {:>7.2} {:>7.2} {:>7.2}",
            t, m.tp, m.fp, m.fn_count, m.precision, m.recall, m.f1);
        total_tp += m.tp;
        total_fp += m.fp;
        total_fn += m.fn_count;
        f1_sum += m.f1;
    }

    let micro_p = if total_tp + total_fp > 0 { total_tp as f64 / (total_tp + total_fp) as f64 } else { 0.0 };
    let micro_r = if total_tp + total_fn > 0 { total_tp as f64 / (total_tp + total_fn) as f64 } else { 0.0 };
    let micro_f1 = if micro_p + micro_r > 0.0 { 2.0 * micro_p * micro_r / (micro_p + micro_r) } else { 0.0 };
    let macro_f1 = if !metrics_strict.is_empty() { f1_sum / metrics_strict.len() as f64 } else { 0.0 };

    println!("  {}", "-".repeat(46));
    println!("  {:<10} {:>4} {:>4} {:>4} {:>7.2} {:>7.2} {:>7.2}", "Micro", total_tp, total_fp, total_fn, micro_p, micro_r, micro_f1);
    println!("  {:<10} {:>4} {:>4} {:>4} {:>7} {:>7} {:>7.2}", "Macro F1", "", "", "", "", "", macro_f1);
    println!();

    // --- Per-type P/R/F1 Chart (Strict) ---
    let strict_types: Vec<&str> = sorted_types.iter().map(|s| s.as_str()).collect();
    let strict_precision: Vec<f64> = sorted_types.iter().map(|t| (metrics_strict[*t].precision * 100.0).round() / 100.0).collect();
    let strict_recall: Vec<f64> = sorted_types.iter().map(|t| (metrics_strict[*t].recall * 100.0).round() / 100.0).collect();
    let strict_f1: Vec<f64> = sorted_types.iter().map(|t| (metrics_strict[*t].f1 * 100.0).round() / 100.0).collect();

    show_chart(&multi_bar_chart(
        "Per-Entity-Type Metrics (Strict Matching)",
        &strict_types,
        &[
            ("Precision", &strict_precision, "#3b82f6"),
            ("Recall", &strict_recall, "#10b981"),
            ("F1", &strict_f1, "#f59e0b"),
        ],
    ));

    // ============================================================
    // RUN EVALUATION: RELAXED MATCHING
    // ============================================================

    println!("{}", "=".repeat(65));
    println!("  NER ERROR ANALYSIS: RELAXED MATCHING");
    println!("  (any span overlap counts as match)");
    println!("{}", "=".repeat(65));
    println!();

    let (metrics_relaxed, _errors_relaxed, _confusion_relaxed) = compute_metrics(&evaluation_data, false);

    println!("  Per-type metrics:");
    println!("  {:<10} {:>4} {:>4} {:>4} {:>7} {:>7} {:>7}", "Type", "TP", "FP", "FN", "Prec", "Rec", "F1");
    println!("  {}", "-".repeat(46));

    let mut total_tp = 0;
    let mut total_fp = 0;
    let mut total_fn = 0;
    let mut f1_sum = 0.0;
    let mut sorted_types_r: Vec<&String> = metrics_relaxed.keys().collect();
    sorted_types_r.sort();
    for t in &sorted_types_r {
        let m = &metrics_relaxed[*t];
        println!("  {:<10} {:>4} {:>4} {:>4} {:>7.2} {:>7.2} {:>7.2}",
            t, m.tp, m.fp, m.fn_count, m.precision, m.recall, m.f1);
        total_tp += m.tp;
        total_fp += m.fp;
        total_fn += m.fn_count;
        f1_sum += m.f1;
    }

    let micro_p = if total_tp + total_fp > 0 { total_tp as f64 / (total_tp + total_fp) as f64 } else { 0.0 };
    let micro_r = if total_tp + total_fn > 0 { total_tp as f64 / (total_tp + total_fn) as f64 } else { 0.0 };
    let micro_f1 = if micro_p + micro_r > 0.0 { 2.0 * micro_p * micro_r / (micro_p + micro_r) } else { 0.0 };
    let macro_f1 = if !metrics_relaxed.is_empty() { f1_sum / metrics_relaxed.len() as f64 } else { 0.0 };

    println!("  {}", "-".repeat(46));
    println!("  {:<10} {:>4} {:>4} {:>4} {:>7.2} {:>7.2} {:>7.2}", "Micro", total_tp, total_fp, total_fn, micro_p, micro_r, micro_f1);
    println!("  {:<10} {:>4} {:>4} {:>4} {:>7} {:>7} {:>7.2}", "Macro F1", "", "", "", "", "", macro_f1);
    println!();

    println!("  NOTE: Relaxed matching gives higher scores because boundary");
    println!("  errors (e.g., 'Dr. Amara Khan' vs 'Amara Khan') count as matches.");
    println!("  The gap between strict and relaxed reveals boundary error rate.");
    println!();

    // --- Per-type P/R/F1 Chart (Relaxed) ---
    let relaxed_types: Vec<&str> = sorted_types_r.iter().map(|s| s.as_str()).collect();
    let relaxed_precision: Vec<f64> = sorted_types_r.iter().map(|t| (metrics_relaxed[*t].precision * 100.0).round() / 100.0).collect();
    let relaxed_recall: Vec<f64> = sorted_types_r.iter().map(|t| (metrics_relaxed[*t].recall * 100.0).round() / 100.0).collect();
    let relaxed_f1: Vec<f64> = sorted_types_r.iter().map(|t| (metrics_relaxed[*t].f1 * 100.0).round() / 100.0).collect();

    show_chart(&multi_bar_chart(
        "Per-Entity-Type Metrics (Relaxed Matching)",
        &relaxed_types,
        &[
            ("Precision", &relaxed_precision, "#3b82f6"),
            ("Recall", &relaxed_recall, "#10b981"),
            ("F1", &relaxed_f1, "#f59e0b"),
        ],
    ));

    // ============================================================
    // ERROR CATEGORIZATION
    // ============================================================

    println!("{}", "=".repeat(65));
    println!("  ERROR CATEGORIZATION (strict matching)");
    println!("{}", "=".repeat(65));
    println!();

    // Detect boundary errors
    let mut boundary_errors = Vec::new();
    for item in &evaluation_data {
        let (strict_matches, _, _) = match_entities(&item.gold, &item.pred, true);
        let (relaxed_matches, _, _) = match_entities(&item.gold, &item.pred, false);

        let strict_pairs: std::collections::BTreeSet<(String, String)> = strict_matches.iter()
            .filter(|m| m.match_type == "correct")
            .map(|m| (m.gold.text.clone(), m.gold.label.clone()))
            .collect();

        for m in &relaxed_matches {
            if m.match_type == "correct" {
                let key = (m.gold.text.clone(), m.gold.label.clone());
                if !strict_pairs.contains(&key) && m.gold.text != m.pred.text {
                    boundary_errors.push((item.text.clone(), m.gold.text.clone(), m.pred.text.clone(), m.gold.label.clone()));
                }
            }
        }
    }

    // Count error categories
    let mut category_counts: HashMap<String, usize> = HashMap::new();
    for e in &errors_strict {
        *category_counts.entry(e.category.clone()).or_insert(0) += 1;
    }

    println!("  Error counts by category:");
    let mut cat_sorted: Vec<(&String, &usize)> = category_counts.iter().collect();
    cat_sorted.sort_by(|a, b| b.1.cmp(a.1));
    for (cat, count) in &cat_sorted {
        let bar: String = "#".repeat(**count * 3);
        println!("    {:<20} {:>3}  {}", cat, count, bar);
    }
    println!();

    if !boundary_errors.is_empty() {
        println!("  Boundary errors detected: {}", boundary_errors.len());
        for (_, gold_span, pred_span, label) in &boundary_errors {
            println!("    Gold: \"{}\" [{}]  Pred: \"{}\" [{}]", gold_span, label, pred_span, label);
        }
        println!();
    }

    println!("  Detailed error list (strict matching):");
    println!();
    for e in &errors_strict {
        println!("    [{:<17}]  Gold: {:<30}  Pred: {}", e.category, e.gold, e.pred);
    }
    println!();

    // ============================================================
    // CONFUSION MATRIX FOR ENTITY TYPES
    // ============================================================

    println!("{}", "=".repeat(65));
    println!("  ENTITY TYPE CONFUSION MATRIX");
    println!("{}", "=".repeat(65));
    println!();

    let mut all_types = std::collections::BTreeSet::new();
    for item in &evaluation_data {
        for g in &item.gold { all_types.insert(g.label.clone()); }
        for p in &item.pred { all_types.insert(p.label.clone()); }
    }
    let all_types_vec: Vec<String> = all_types.into_iter().collect();

    let mut confusion: HashMap<String, HashMap<String, usize>> = HashMap::new();
    for item in &evaluation_data {
        let (matches, missed, false_pos) = match_entities(&item.gold, &item.pred, false);
        for m in &matches {
            *confusion.entry(m.gold.label.clone()).or_insert_with(HashMap::new)
                .entry(m.pred.label.clone()).or_insert(0) += 1;
        }
        for g in &missed {
            *confusion.entry(g.label.clone()).or_insert_with(HashMap::new)
                .entry("(MISSED)".to_string()).or_insert(0) += 1;
        }
        for p in &false_pos {
            *confusion.entry("(NONE)".to_string()).or_insert_with(HashMap::new)
                .entry(p.label.clone()).or_insert(0) += 1;
        }
    }

    let mut col_labels: Vec<String> = all_types_vec.clone();
    col_labels.push("(MISSED)".to_string());
    let mut row_labels: Vec<String> = all_types_vec.clone();
    row_labels.push("(NONE)".to_string());

    print!("  {:<12}", "Gold \\ Pred");
    for col in &col_labels { print!("{:>10}", col); }
    println!();
    println!("  {}", "-".repeat(12 + 10 * col_labels.len()));

    let mut heatmap_data: Vec<Vec<f64>> = Vec::new();
    for row in &row_labels {
        print!("  {:<12}", row);
        let mut row_data = Vec::new();
        for col in &col_labels {
            let val = confusion.get(row).and_then(|r| r.get(col)).copied().unwrap_or(0);
            row_data.push(val as f64);
            if val > 0 {
                print!("{:>10}", val);
            } else {
                print!("{:>10}", ".");
            }
        }
        heatmap_data.push(row_data);
        println!();
    }

    println!();
    println!("  Reading the matrix:");
    println!("    - Diagonal = correct predictions");
    println!("    - Off-diagonal = type confusions (gold type != predicted type)");
    println!("    - (MISSED) column = entities the system failed to detect");
    println!("    - (NONE) row = false positives (system predicted an entity where none exists)");
    println!();

    let heatmap_row_refs: Vec<&str> = row_labels.iter().map(|s| s.as_str()).collect();
    let heatmap_col_refs: Vec<&str> = col_labels.iter().map(|s| s.as_str()).collect();
    show_chart(&heatmap_chart(
        "Entity Type Confusion Matrix",
        &heatmap_col_refs,
        &heatmap_row_refs,
        &heatmap_data,
    ));

    // ============================================================
    // ACTIONABLE INSIGHTS
    // ============================================================

    println!("{}", "=".repeat(65));
    println!("  ACTIONABLE INSIGHTS");
    println!("{}", "=".repeat(65));
    println!();

    let worst_type = sorted_types.iter().min_by(|a, b| {
        metrics_strict[**a].f1.partial_cmp(&metrics_strict[**b].f1).unwrap()
    });
    let best_type = sorted_types.iter().max_by(|a, b| {
        metrics_strict[**a].f1.partial_cmp(&metrics_strict[**b].f1).unwrap()
    });

    if let Some(w) = worst_type {
        println!("  Weakest entity type:  {} (F1={:.2})", w, metrics_strict[*w].f1);
    }
    if let Some(b) = best_type {
        println!("  Strongest entity type: {} (F1={:.2})", b, metrics_strict[*b].f1);
    }
    println!();

    if let Some(dominant_error) = cat_sorted.first() {
        println!("  Most common error: {} ({} instances)", dominant_error.0, dominant_error.1);
        println!();

        match dominant_error.0.as_str() {
            "MISSED_ENTITY" => {
                println!("  Recommendation: Improve recall — add more patterns/features");
                println!("  for detecting entities. Consider expanding gazetteers or");
                println!("  lowering confidence thresholds.");
            }
            "FALSE_POSITIVE" => {
                println!("  Recommendation: Improve precision — add negative constraints");
                println!("  or raise confidence thresholds. Check for common false");
                println!("  triggers (e.g., 'May' as verb vs month).");
            }
            "TYPE_CONFUSION" => {
                println!("  Recommendation: Improve type discrimination — add features");
                println!("  that distinguish between confused types. Check the confusion");
                println!("  matrix to identify which types are being confused.");
            }
            _ => {}
        }
    }
    println!();

    // Strict vs relaxed gap
    let strict_f1s: Vec<f64> = metrics_strict.values().map(|m| m.f1).collect();
    let relaxed_f1s: Vec<f64> = metrics_relaxed.values().map(|m| m.f1).collect();
    let strict_avg = if strict_f1s.is_empty() { 0.0 } else { strict_f1s.iter().sum::<f64>() / strict_f1s.len() as f64 };
    let relaxed_avg = if relaxed_f1s.is_empty() { 0.0 } else { relaxed_f1s.iter().sum::<f64>() / relaxed_f1s.len() as f64 };
    let gap = relaxed_avg - strict_avg;

    println!("  Strict vs Relaxed F1 gap: {:.2}", gap);
    if gap > 0.1 {
        println!("  Large gap indicates significant boundary errors.");
        println!("  Recommendation: Review tokenization and span detection logic.");
    } else if gap > 0.0 {
        println!("  Small gap indicates some boundary errors exist but type");
        println!("  detection is the bigger issue.");
    } else {
        println!("  No gap: boundaries are accurate, focus on type prediction.");
    }

    println!();
    println!("  Error analysis is not a one-time task. Every time you change");
    println!("  your NER system, re-run this analysis to see if your changes");
    println!("  helped the errors you targeted without creating new ones.");
}
```

---

## Key Takeaways

- **Aggregate scores hide critical failures.** A system with 75% overall F1 might have 95% F1 on PERSON and 30% F1 on ORG. Per-type analysis reveals where the system actually needs work, and without it you are optimizing blindly.
- **Error categories drive different fixes.** Boundary errors call for better tokenization. Type confusion calls for better discriminative features. Missed entities call for higher recall. False positives call for tighter constraints. Knowing which category dominates tells you exactly where to invest effort.
- **Matching strategy changes your numbers.** Strict matching (exact boundaries) gives lower scores than relaxed matching (any overlap). The gap between them quantifies your boundary error rate. Always report which matching strategy you used — a "90% F1" score means nothing without this context.
- **Error analysis is iterative.** Fix the dominant error category, re-evaluate, and the next dominant category will emerge. This cycle of analyze-fix-reanalyze is how real NLP systems improve incrementally, and it applies equally to rule-based, statistical, and neural NER.
