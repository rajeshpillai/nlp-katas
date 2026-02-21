# Visualize Embedding Spaces

> Phase 7 — Small Neural Text Models | Kata 7.2

---

## Concept & Intuition

### What problem are we solving?

In Kata 7.1, we built word embeddings --- dense vectors with 20 dimensions. We measured cosine similarity and saw that "cat" and "dog" are close while "cat" and "king" are far apart. But numbers alone do not build intuition. **We need to see the embedding space** to understand what these vectors actually look like.

Visualization answers critical questions: Do semantically related words actually cluster together? Are there visible groupings by category (animals, royalty, objects)? How does the structure of the embedding space reflect the structure of meaning in the corpus?

The problem is that our embeddings live in 20 dimensions, and we can only see 2. We need **dimensionality reduction** --- a way to project high-dimensional vectors onto a 2D plane while preserving the relative distances as much as possible.

### Why earlier methods fail

In Phase 2 (Kata 2.2), we visualized BoW vectors using bar charts --- one bar per vocabulary word. That worked because each dimension had a clear meaning (a specific word's count). But embedding dimensions have **no human-interpretable meaning**. Dimension 7 of an embedding is not "the cat dimension" or "the royalty dimension." It is a learned, abstract feature.

Bar charts of embedding values are meaningless. Seeing that "cat" has value 0.23 in dimension 3 and "dog" has value 0.19 in dimension 3 tells you nothing. What matters is the **relative positions** of words in the space --- which words are near each other and which are far apart. For that, we need scatter plots, not bar charts.

### Mental models

- **City map analogy**: Imagine embedding space as a city. Similar words live in the same neighborhood. Animals are in one district, royalty in another, furniture in a third. A 2D projection is like a flattened tourist map --- some distances get distorted, but the neighborhoods remain visible.
- **Shadow projection**: Imagine 20 flashlights shining on 20-dimensional word-objects from different angles. Each angle produces a different 2D shadow. Some shadows preserve the clustering structure well, others do not. PCA picks the angle that preserves the most variance --- the most informative shadow.
- **Lossy compression**: Going from 20D to 2D throws away information. Two words that are far apart in 20D might overlap in 2D, just like two buildings on opposite sides of a city can appear at the same point on a badly-angled photograph. The goal is to minimize this distortion.

### Visual explanations

```
FROM HIGH-DIMENSIONAL TO 2D
==============================

20D embedding space (cannot visualize directly):
  "cat"    = [0.23, -0.11, 0.85, 0.03, ..., 0.42]  (20 numbers)
  "dog"    = [0.19, -0.15, 0.78, 0.01, ..., 0.35]  (20 numbers)
  "king"   = [-0.41, 0.62, -0.08, 0.77, ..., -0.15] (20 numbers)

2D projection (can visualize):
  "cat"    = (2.3, 1.1)
  "dog"    = (2.1, 0.9)
  "king"   = (-1.8, -0.5)


WHAT A GOOD EMBEDDING VISUALIZATION LOOKS LIKE
================================================

        ^
        |          king *
        |        queen *
        |
        |    man *
        |  woman *
        |
   -----+----------------------------->
        |
        |             cat *
        |           dog *  kitten *
        |
        |      mat *
        |    rug *
        |

  Animals cluster together (bottom right).
  Royalty clusters together (top right).
  Objects cluster together (bottom left).
  People cluster in between.


WHAT A BAD PROJECTION LOOKS LIKE
==================================

        ^
        |    king *  cat *
        |         dog *
        |  queen *       mat *
        |      man *  rug *
        |    woman *
   -----+----------------------------->

  All words mixed together, no visible structure.
  This happens when you pick bad projection dimensions
  or when the embeddings themselves are poor quality.
```

---

## Hands-on Exploration

1. Run the code below and examine the ASCII scatter plot. Identify which words cluster together. Then mentally draw circles around each cluster and label them (e.g., "animals", "royalty", "objects"). Does the clustering match your intuition about word meaning? Are there any words placed in surprising positions?

2. The code uses the first two principal components for projection. Modify it to use dimensions 0 and 1, then dimensions 2 and 3, then dimensions 5 and 10 of the raw embedding vectors. How does the clustering quality change? This demonstrates that **not all dimensions carry equal information** --- PCA finds the most informative combination.

3. Add 5 new sentences to the corpus about a new topic (e.g., food: "the chef cooked a meal", "the cook prepared fresh soup"). Rebuild the embeddings and visualize again. Does a new cluster appear for food-related words? How many sentences do you need before the cluster becomes visible? This reveals the **minimum corpus size** needed for meaningful embeddings.

---

## Live Code

```rust
use std::collections::HashMap;

fn main() {
    // --- Visualize Word Embedding Spaces (Pure Rust) ---
    // Build embeddings from co-occurrence and project to 2D for ASCII plotting.
    // No external crates --- everything from scratch.

    let mut rng = Rng::new(42);

    // ============================================================
    // VECTOR MATH HELPERS
    // ============================================================

    fn dot_product(v1: &[f64], v2: &[f64]) -> f64 {
        v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum()
    }

    fn magnitude(v: &[f64]) -> f64 {
        v.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    fn cosine_similarity(v1: &[f64], v2: &[f64]) -> f64 {
        let d = dot_product(v1, v2);
        let m1 = magnitude(v1);
        let m2 = magnitude(v2);
        if m1 == 0.0 || m2 == 0.0 { return 0.0; }
        d / (m1 * m2)
    }

    fn vector_subtract(v1: &[f64], v2: &[f64]) -> Vec<f64> {
        v1.iter().zip(v2.iter()).map(|(a, b)| a - b).collect()
    }

    // ============================================================
    // CORPUS AND EMBEDDING CONSTRUCTION
    // ============================================================

    let corpus: Vec<&str> = vec![
        "the cat sat on the mat in the room",
        "the cat chased the mouse around the house",
        "a small cat slept on the soft mat",
        "the dog sat on the rug in the hall",
        "the dog chased the ball around the yard",
        "a big dog slept on the warm rug",
        "the king ruled the land with wisdom",
        "the queen ruled the kingdom with grace",
        "the king wore a golden crown",
        "the queen wore a silver crown",
        "the man walked to the market",
        "the woman walked to the store",
        "the boy played in the park",
        "the girl played in the garden",
        "the cat and the dog sat together",
        "the king and the queen ruled together",
        "the man and the woman walked together",
        "the boy and the girl played together",
    ];

    fn tokenize(text: &str) -> Vec<String> {
        let mut result = Vec::new();
        let mut current = String::new();
        for ch in text.to_lowercase().chars() {
            if ch.is_alphabetic() {
                current.push(ch);
            } else if !current.is_empty() {
                result.push(current.clone());
                current.clear();
            }
        }
        if !current.is_empty() { result.push(current); }
        result
    }

    let mut all_tokens: Vec<String> = Vec::new();
    for sentence in &corpus {
        all_tokens.extend(tokenize(sentence));
    }

    let mut vocab: Vec<String> = all_tokens.iter().cloned().collect::<std::collections::BTreeSet<_>>().into_iter().collect();
    vocab.sort();
    let word_to_idx: HashMap<String, usize> = vocab.iter().enumerate().map(|(i, w)| (w.clone(), i)).collect();
    let vocab_size = vocab.len();

    // Build co-occurrence matrix
    let window_size: usize = 2;
    let mut cooccurrence: HashMap<String, HashMap<String, usize>> = HashMap::new();
    for w in &vocab { cooccurrence.insert(w.clone(), HashMap::new()); }

    for sentence in &corpus {
        let tokens = tokenize(sentence);
        for (i, target) in tokens.iter().enumerate() {
            let start = if i >= window_size { i - window_size } else { 0 };
            let end = (i + window_size + 1).min(tokens.len());
            for j in start..end {
                if i != j {
                    *cooccurrence.get_mut(target).unwrap().entry(tokens[j].clone()).or_insert(0) += 1;
                }
            }
        }
    }

    // Random projection to get embeddings
    let embed_dim: usize = 20;
    let mut projection: Vec<Vec<f64>> = Vec::new();
    for _ in 0..vocab_size {
        let row: Vec<f64> = (0..embed_dim).map(|_| if rng.next_f64() < 0.5 { -1.0 } else { 1.0 }).collect();
        projection.push(row);
    }

    let mut embeddings: HashMap<String, Vec<f64>> = HashMap::new();
    for word in &vocab {
        let co = cooccurrence.get(word).unwrap();
        let full_vec: Vec<f64> = vocab.iter().map(|w| *co.get(w).unwrap_or(&0) as f64).collect();
        let mut embed: Vec<f64> = (0..embed_dim).map(|d| {
            (0..vocab_size).map(|i| full_vec[i] * projection[i][d]).sum()
        }).collect();
        let mag = magnitude(&embed);
        if mag > 0.0 { embed = embed.iter().map(|x| x / mag).collect(); }
        embeddings.insert(word.clone(), embed);
    }

    println!("{}", "=".repeat(60));
    println!("WORD EMBEDDINGS BUILT");
    println!("{}", "=".repeat(60));
    println!("  Corpus: {} sentences, {} unique words", corpus.len(), vocab_size);
    println!("  Embedding dimensions: {}", embed_dim);
    println!();

    // ============================================================
    // SIMPLE PCA-LIKE PROJECTION TO 2D
    // ============================================================

    let focus_words_raw: Vec<&str> = vec![
        "cat", "dog", "mouse", "ball",
        "king", "queen", "crown", "kingdom",
        "man", "woman", "boy", "girl",
        "mat", "rug", "room", "hall",
        "park", "garden", "yard", "house",
    ];
    let focus_words: Vec<&str> = focus_words_raw.into_iter().filter(|w| embeddings.contains_key(*w)).collect();

    // Step 1: Compute mean embedding
    let mut mean_vec = vec![0.0f64; embed_dim];
    for w in &focus_words {
        let e = &embeddings[*w];
        for d in 0..embed_dim { mean_vec[d] += e[d]; }
    }
    let n = focus_words.len() as f64;
    mean_vec = mean_vec.iter().map(|x| x / n).collect();

    // Step 2: Center the embeddings
    let centered: HashMap<&str, Vec<f64>> = focus_words.iter()
        .map(|w| (*w, vector_subtract(&embeddings[*w], &mean_vec)))
        .collect();

    // Step 3: Power iteration to find principal components
    fn power_iteration(data_vecs: &[Vec<f64>], dim: usize, rng: &mut Rng, num_iterations: usize) -> Vec<f64> {
        let mut v: Vec<f64> = (0..dim).map(|_| rng.gauss(0.0, 1.0)).collect();
        let mag_v: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        v = v.iter().map(|x| x / mag_v).collect();

        for _ in 0..num_iterations {
            let mut new_v = vec![0.0; dim];
            for vec in data_vecs {
                let proj: f64 = vec.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
                for d in 0..dim { new_v[d] += proj * vec[d]; }
            }
            let mag_new: f64 = new_v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if mag_new > 0.0 { v = new_v.iter().map(|x| x / mag_new).collect(); }
        }
        v
    }

    let data_list: Vec<Vec<f64>> = focus_words.iter().map(|w| centered[w].clone()).collect();

    // First principal component
    let pc1 = power_iteration(&data_list, embed_dim, &mut rng, 200);

    // Remove PC1 component from data to find PC2
    let deflated: Vec<Vec<f64>> = data_list.iter().map(|vec| {
        let proj: f64 = vec.iter().zip(pc1.iter()).map(|(a, b)| a * b).sum();
        (0..embed_dim).map(|d| vec[d] - proj * pc1[d]).collect()
    }).collect();

    let pc2 = power_iteration(&deflated, embed_dim, &mut rng, 200);

    // Project each word onto PC1 and PC2
    let mut coords_2d: HashMap<&str, (f64, f64)> = HashMap::new();
    for w in &focus_words {
        let x: f64 = centered[w].iter().zip(pc1.iter()).map(|(a, b)| a * b).sum();
        let y: f64 = centered[w].iter().zip(pc2.iter()).map(|(a, b)| a * b).sum();
        coords_2d.insert(w, (x, y));
    }

    println!("{}", "=".repeat(60));
    println!("2D PROJECTIONS (PCA)");
    println!("{}", "=".repeat(60));
    for w in &focus_words {
        let (x, y) = coords_2d[w];
        println!("  {:>10} -> ({:+.4}, {:+.4})", w, x, y);
    }
    println!();

    // ============================================================
    // ASCII SCATTER PLOT
    // ============================================================

    let categories: Vec<(&str, Vec<&str>)> = vec![
        ("animals", vec!["cat", "dog", "mouse"]),
        ("royalty", vec!["king", "queen", "crown", "kingdom"]),
        ("people", vec!["man", "woman", "boy", "girl"]),
        ("objects", vec!["mat", "rug", "ball"]),
        ("places", vec!["room", "hall", "park", "garden", "yard", "house"]),
    ];

    let cat_symbols: HashMap<&str, char> = vec![
        ("animals", 'A'), ("royalty", 'R'), ("people", 'P'),
        ("objects", 'O'), ("places", 'L'),
    ].into_iter().collect();

    let mut word_category: HashMap<&str, &str> = HashMap::new();
    for (cat, words) in &categories {
        for w in words {
            if coords_2d.contains_key(w) { word_category.insert(w, cat); }
        }
    }

    let plot_width: usize = 60;
    let plot_height: usize = 30;

    let xs: Vec<f64> = focus_words.iter().map(|w| coords_2d[w].0).collect();
    let ys: Vec<f64> = focus_words.iter().map(|w| coords_2d[w].1).collect();
    let mut x_min = xs.iter().cloned().fold(f64::INFINITY, f64::min);
    let mut x_max = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut y_min = ys.iter().cloned().fold(f64::INFINITY, f64::min);
    let mut y_max = ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let x_pad = (x_max - x_min) * 0.1 + 0.001;
    let y_pad = (y_max - y_min) * 0.1 + 0.001;
    x_min -= x_pad; x_max += x_pad;
    y_min -= y_pad; y_max += y_pad;

    fn to_grid(x: f64, y: f64, x_min: f64, x_max: f64, y_min: f64, y_max: f64, pw: usize, ph: usize) -> (usize, usize) {
        let gx = ((x - x_min) / (x_max - x_min) * (pw - 1) as f64) as usize;
        let gy = ((y - y_min) / (y_max - y_min) * (ph - 1) as f64) as usize;
        (gx.min(pw - 1), gy.min(ph - 1))
    }

    println!("{}", "=".repeat(60));
    println!("ASCII SCATTER PLOT OF EMBEDDING SPACE");
    println!("{}", "=".repeat(60));
    println!();
    println!("  Legend: A=animal  R=royalty  P=people  O=object  L=place");
    println!();

    let mut grid: Vec<Vec<char>> = vec![vec![' '; plot_width]; plot_height];
    let mut placed: HashMap<(usize, usize), Vec<&str>> = HashMap::new();

    for w in &focus_words {
        let (x, y) = coords_2d[w];
        let (gx, gy) = to_grid(x, y, x_min, x_max, y_min, y_max, plot_width, plot_height);
        let cat = word_category.get(w).unwrap_or(&"?");
        let symbol = cat_symbols.get(cat).copied().unwrap_or('?');
        let gy_flipped = plot_height - 1 - gy;
        let key = (gx, gy_flipped);
        if !placed.contains_key(&key) {
            grid[gy_flipped][gx] = symbol;
            placed.insert(key, vec![w]);
        } else {
            placed.get_mut(&key).unwrap().push(w);
        }
    }

    print!("  ");
    println!("{}", "-".repeat(plot_width + 2));
    for row in &grid {
        print!("  |");
        for ch in row { print!("{}", ch); }
        println!("|");
    }
    print!("  ");
    println!("{}", "-".repeat(plot_width + 2));
    println!();

    // Print word positions with labels
    println!("  Word positions:");
    for w in &focus_words {
        let (x, y) = coords_2d[w];
        let cat = word_category.get(w).unwrap_or(&"?");
        let symbol = cat_symbols.get(cat).copied().unwrap_or('?');
        println!("    [{}] {:>10}  ({:+.3}, {:+.3})  ({})", symbol, w, x, y, cat);
    }
    println!();

    // ============================================================
    // CLUSTER ANALYSIS
    // ============================================================

    println!("{}", "=".repeat(60));
    println!("CLUSTER ANALYSIS");
    println!("{}", "=".repeat(60));
    println!();

    for (cat_name, cat_words) in &categories {
        let present: Vec<&str> = cat_words.iter().filter(|w| coords_2d.contains_key(**w)).copied().collect();
        if present.len() < 2 { continue; }

        let mut total_sim = 0.0;
        let mut count = 0;
        for i in 0..present.len() {
            for j in (i + 1)..present.len() {
                let sim = cosine_similarity(&embeddings[present[i]], &embeddings[present[j]]);
                total_sim += sim;
                count += 1;
            }
        }
        let avg_sim = if count > 0 { total_sim / count as f64 } else { 0.0 };

        let cx: f64 = present.iter().map(|w| coords_2d[w].0).sum::<f64>() / present.len() as f64;
        let cy: f64 = present.iter().map(|w| coords_2d[w].1).sum::<f64>() / present.len() as f64;

        let words_str = present.join(", ");
        println!("  {:>10}: {}", cat_name.to_uppercase(), words_str);
        println!("             Avg cosine similarity: {:.4}", avg_sim);
        println!("             2D centroid: ({:+.3}, {:+.3})", cx, cy);
        println!();
    }

    // ============================================================
    // INTER-CLUSTER VS INTRA-CLUSTER DISTANCES
    // ============================================================

    println!("{}", "=".repeat(60));
    println!("INTRA-CLUSTER vs INTER-CLUSTER SIMILARITY");
    println!("{}", "=".repeat(60));
    println!();
    println!("  Good embeddings: intra-cluster similarity >> inter-cluster");
    println!();

    let cat_names: Vec<&str> = categories.iter()
        .filter(|(_, ws)| ws.iter().filter(|w| embeddings.contains_key(**w)).count() >= 2)
        .map(|(name, _)| *name).collect();

    println!("  INTRA-CLUSTER (within category):");
    for cat_name in &cat_names {
        let cat_words: &Vec<&str> = &categories.iter().find(|(n, _)| n == cat_name).unwrap().1;
        let present: Vec<&str> = cat_words.iter().filter(|w| embeddings.contains_key(**w)).copied().collect();
        let mut total_sim = 0.0;
        let mut count = 0;
        for i in 0..present.len() {
            for j in (i + 1)..present.len() {
                total_sim += cosine_similarity(&embeddings[present[i]], &embeddings[present[j]]);
                count += 1;
            }
        }
        let avg = if count > 0 { total_sim / count as f64 } else { 0.0 };
        let bar = "#".repeat((avg.max(0.0) * 30.0) as usize);
        println!("    {:>10}: {:+.4}  {}", cat_name, avg, bar);
    }
    println!();

    println!("  INTER-CLUSTER (between categories):");
    for i in 0..cat_names.len() {
        for j in (i + 1)..cat_names.len() {
            let ws1: Vec<&str> = categories.iter().find(|(n, _)| *n == cat_names[i]).unwrap().1.iter().filter(|w| embeddings.contains_key(**w)).copied().collect();
            let ws2: Vec<&str> = categories.iter().find(|(n, _)| *n == cat_names[j]).unwrap().1.iter().filter(|w| embeddings.contains_key(**w)).copied().collect();
            let mut total_sim = 0.0;
            let mut count = 0;
            for w1 in &ws1 {
                for w2 in &ws2 {
                    total_sim += cosine_similarity(&embeddings[*w1], &embeddings[*w2]);
                    count += 1;
                }
            }
            let avg = if count > 0 { total_sim / count as f64 } else { 0.0 };
            let bar = "#".repeat((avg.max(0.0) * 30.0) as usize);
            println!("    {:>10} <-> {:<10}: {:+.4}  {}", cat_names[i], cat_names[j], avg, bar);
        }
    }

    println!();
    println!("  If intra-cluster scores are consistently higher than");
    println!("  inter-cluster scores, the embedding space has meaningful");
    println!("  structure --- words are organized by semantic category.");

    // ============================================================
    // CORPUS QUALITY EXPERIMENT
    // ============================================================

    println!();
    println!("{}", "=".repeat(60));
    println!("EMBEDDING QUALITY DEPENDS ON CORPUS");
    println!("{}", "=".repeat(60));
    println!();
    println!("  Key insight: embeddings are only as good as the corpus");
    println!("  they are built from.");
    println!();
    println!("  Our corpus has:");

    let word_freq: HashMap<String, usize> = {
        let mut freq = HashMap::new();
        for t in &all_tokens { *freq.entry(t.clone()).or_insert(0) += 1; }
        freq
    };
    for w in &["cat", "dog", "king", "queen", "mat", "rug"] {
        let ws = w.to_string();
        if let Some(&f) = word_freq.get(&ws) {
            let co_count: usize = cooccurrence.get(&ws).map(|c| c.values().sum()).unwrap_or(0);
            println!("    '{}': {} occurrences, {} co-occurrence events", w, f, co_count);
        }
    }

    println!();
    println!("  With a larger corpus:");
    println!("    - More co-occurrence data -> more accurate vectors");
    println!("    - More diverse contexts -> richer representations");
    println!("    - Rare words get enough data to be placed correctly");
    println!();
    println!("  With our small corpus:");
    println!("    - Common words (the, on, in) dominate co-occurrence");
    println!("    - Rare words may be placed poorly");
    println!("    - Analogies may not work reliably");
    println!();
    println!("  This is why real embedding systems (Word2Vec, GloVe)");
    println!("  train on billions of words, not dozens of sentences.");
}
```

---

## Key Takeaways

- **Visualization reveals structure that numbers hide.** A table of cosine similarities tells you two words are close, but a scatter plot shows you the entire landscape --- clusters, outliers, and the overall geometry of the embedding space. Seeing is understanding.
- **Dimensionality reduction is lossy but essential.** Projecting 20 dimensions to 2 throws away information. Some close words may appear far apart, and some distant words may overlap. The goal is to preserve the most important structure, not all of it.
- **Semantic categories form visible clusters.** When embeddings are working well, words from the same category (animals, royalty, places) group together in the projected space. This clustering is not programmed --- it emerges from co-occurrence patterns in the corpus.
- **Embedding quality is bounded by corpus quality.** A small, narrow corpus produces noisy embeddings with weak clustering. A large, diverse corpus produces clean embeddings with sharp category boundaries. There is no substitute for sufficient data.
