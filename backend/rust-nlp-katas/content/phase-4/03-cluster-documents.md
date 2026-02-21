# Cluster Documents by Topic

> Phase 4 — Similarity & Classical NLP Tasks | Kata 4.3

---

## Concept & Intuition

### What problem are we solving?

We have a pile of unlabeled documents and we want to **automatically group them by topic**. Nobody told us what the topics are. Nobody labeled anything. We just want the algorithm to discover that some documents are about sports, others about technology, and others about cooking — purely from the text.

This is **unsupervised learning**: finding structure in data without human labels. Document clustering converts each document into a TF-IDF vector and then groups vectors that are close together in space. Documents pointing in similar directions (high cosine similarity) end up in the same cluster.

### Why naive/earlier approaches fail

**Manual categorization** does not scale. A human can sort 50 documents, but not 50,000.

**Simple keyword rules** ("if the document contains 'goal', it's about sports") break immediately. "Goal" appears in business documents ("quarterly goals"), self-help articles, and game design discussions. Hard-coded rules are brittle and miss the nuance that comes from looking at the full distribution of words.

**Raw word overlap** without vectorization gives no way to measure "distance" between documents. We need documents in a shared vector space so that "close" and "far" have geometric meaning. TF-IDF vectors in a high-dimensional space give us exactly that.

### Mental models

- **K-means clustering** works by: (1) pick K random centroids, (2) assign each document to the nearest centroid, (3) recompute centroids as the mean of assigned documents, (4) repeat until stable. It is simple, interpretable, and surprisingly effective.
- **Centroid** = the "average" vector of a cluster. It represents the typical direction of documents in that group.
- **Convergence** = when no document changes its cluster assignment between iterations. The algorithm has found a stable grouping.
- **K** = the number of clusters. You must choose this in advance (a real limitation). Too few clusters merge distinct topics; too many split coherent topics.

### Visual explanations

```
K-means in 2D (imagine TF-IDF vectors projected to 2 dimensions):

  Step 1: Random centroids         Step 2: Assign to nearest
      *                                * x x
                                        x
          x  x                     o  x  x
        x      x                     x      x
                     x                            x
     o     x              o           o     x       o
        x    x                           x    x

  * = centroid                     x = assigned to *
  o = centroid                     (each point goes to nearest centroid)
  x = documents

  Step 3: Recompute centroids      Step 4: Reassign (converged!)
        x x                              x x
      *  x                             *  x
        x  x                             x  x
          x      x                         x      x
                            o                           o
            x                                x
         x    x                           x    x

  * moved to center of its           No points changed cluster.
  assigned points. o did too.         Algorithm stops. Done!


  Why cosine-based clustering works for text:

    Cluster 1 (Sports)          Cluster 2 (Tech)
    +---------------------+    +----------------------+
    | "team won game"      |    | "code runs fast"     |
    | "scored a goal"      |    | "debug the app"      |
    | "final match"        |    | "server crashed"     |
    +---------------------+    +----------------------+

    These vectors point in                These vectors point in
    a "sports" direction                  a "tech" direction
    in TF-IDF space.                      in TF-IDF space.

    The algorithm discovers this grouping WITHOUT knowing
    what "sports" or "tech" means — purely from word patterns.
```

---

## Hands-on Exploration

1. Look at the 12 documents in the code below. Before running it, manually sort them into 3 groups by topic. Then run the clustering algorithm and compare your grouping with the algorithm's output. Where do they agree? Where do they disagree? Why might the algorithm group things differently than you?

2. Change K from 3 to 2 and observe which topics get merged together. Then try K=4 and see if a coherent fourth cluster emerges or if a good cluster gets split. This demonstrates the fundamental challenge of choosing K.

3. Run the algorithm multiple times (it uses random initialization). Notice that you might get different clusterings each run. This non-determinism is a real property of k-means. Think about why random starting positions lead to different outcomes and what strategies might help (hint: the code runs multiple times and picks the best).

---

## Live Code

```rust
use std::collections::{HashMap, BTreeSet};

fn main() {
    // --- Document Clustering with K-Means on TF-IDF Vectors ---
    // Pure Rust stdlib — no external dependencies

    fn tokenize(text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut current = String::new();
        for ch in text.to_lowercase().chars() {
            if ch.is_alphabetic() {
                current.push(ch);
            } else if !current.is_empty() {
                tokens.push(current.clone());
                current.clear();
            }
        }
        if !current.is_empty() { tokens.push(current); }
        tokens
    }

    let stopwords: BTreeSet<&str> = [
        "a", "an", "the", "is", "it", "in", "on", "of", "to", "and",
        "for", "with", "that", "this", "are", "was", "be", "as", "at",
        "by", "or", "from", "but", "not", "have", "has", "had", "do",
        "does", "did", "will", "can", "so", "if", "its", "we", "they",
        "their", "about", "been", "more", "when", "also", "up", "into",
    ].iter().copied().collect();

    let preprocess = |text: &str| -> Vec<String> {
        tokenize(text).into_iter().filter(|t| !stopwords.contains(t.as_str())).collect()
    };

    fn compute_tfidf(doc_tokens_list: &[Vec<String>]) -> (Vec<String>, Vec<Vec<f64>>) {
        let mut vocab_set = BTreeSet::new();
        for tokens in doc_tokens_list {
            for t in tokens { vocab_set.insert(t.clone()); }
        }
        let vocab: Vec<String> = vocab_set.into_iter().collect();
        let word_to_idx: HashMap<&str, usize> = vocab.iter().enumerate().map(|(i, w)| (w.as_str(), i)).collect();

        let n_docs = doc_tokens_list.len();
        let n_words = vocab.len();

        // Compute document frequency
        let mut df = vec![0usize; n_words];
        for tokens in doc_tokens_list {
            let seen: BTreeSet<&str> = tokens.iter().map(|s| s.as_str()).collect();
            for word in seen {
                if let Some(&idx) = word_to_idx.get(word) {
                    df[idx] += 1;
                }
            }
        }

        // Compute IDF
        let idf: Vec<f64> = df.iter().map(|&d| {
            if d > 0 { (n_docs as f64 / d as f64).ln() } else { 0.0 }
        }).collect();

        // Compute TF-IDF vectors
        let mut vectors = Vec::new();
        for tokens in doc_tokens_list {
            let mut tf: HashMap<&str, f64> = HashMap::new();
            for t in tokens { *tf.entry(t.as_str()).or_insert(0.0) += 1.0; }
            let total = if tokens.is_empty() { 1.0 } else { tokens.len() as f64 };
            let mut vec = vec![0.0f64; n_words];
            for (word, count) in &tf {
                if let Some(&idx) = word_to_idx.get(word) {
                    vec[idx] = (count / total) * idf[idx];
                }
            }
            vectors.push(vec);
        }

        (vocab, vectors)
    }

    fn cosine_similarity(v1: &[f64], v2: &[f64]) -> f64 {
        let dot: f64 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
        let m1 = v1.iter().map(|a| a * a).sum::<f64>().sqrt();
        let m2 = v2.iter().map(|b| b * b).sum::<f64>().sqrt();
        if m1 == 0.0 || m2 == 0.0 { return 0.0; }
        dot / (m1 * m2)
    }

    fn vector_add(v1: &[f64], v2: &[f64]) -> Vec<f64> {
        v1.iter().zip(v2.iter()).map(|(a, b)| a + b).collect()
    }

    fn vector_scale(v: &[f64], scalar: f64) -> Vec<f64> {
        v.iter().map(|x| x * scalar).collect()
    }

    fn kmeans_cosine(vectors: &[Vec<f64>], k: usize, max_iters: usize, n_runs: usize, rng: &mut Rng) -> (Vec<usize>, f64) {
        let n = vectors.len();
        let dim = vectors[0].len();
        let mut best_assignment: Vec<usize> = vec![0; n];
        let mut best_score: f64 = -1.0;

        for _run in 0..n_runs {
            // Pick k random documents as centroids (Fisher-Yates partial shuffle)
            let mut indices: Vec<usize> = (0..n).collect();
            for i in 0..k {
                let j = i + (rng.next_u64() as usize % (n - i));
                indices.swap(i, j);
            }
            let mut centroids: Vec<Vec<f64>> = (0..k).map(|i| vectors[indices[i]].clone()).collect();

            let mut assignment = vec![0usize; n];

            for _iteration in 0..max_iters {
                // Assign each document to the most similar centroid
                let mut new_assignment = vec![0usize; n];
                for i in 0..n {
                    let mut best_sim = -1.0f64;
                    let mut best_cluster = 0usize;
                    for c in 0..k {
                        let sim = cosine_similarity(&vectors[i], &centroids[c]);
                        if sim > best_sim {
                            best_sim = sim;
                            best_cluster = c;
                        }
                    }
                    new_assignment[i] = best_cluster;
                }

                if new_assignment == assignment { break; }
                assignment = new_assignment;

                // Recompute centroids
                let mut new_centroids: Vec<Vec<f64>> = (0..k).map(|_| vec![0.0; dim]).collect();
                let mut counts = vec![0usize; k];
                for i in 0..n {
                    let c = assignment[i];
                    new_centroids[c] = vector_add(&new_centroids[c], &vectors[i]);
                    counts[c] += 1;
                }
                for c in 0..k {
                    if counts[c] > 0 {
                        centroids[c] = vector_scale(&new_centroids[c], 1.0 / counts[c] as f64);
                    }
                }
            }

            // Score: average within-cluster cosine similarity
            let mut total_sim = 0.0;
            let mut count = 0;
            for c in 0..k {
                let centroid = {
                    let members: Vec<usize> = (0..n).filter(|&i| assignment[i] == c).collect();
                    if members.is_empty() { continue; }
                    let mut sum = vec![0.0; dim];
                    for &i in &members { sum = vector_add(&sum, &vectors[i]); }
                    vector_scale(&sum, 1.0 / members.len() as f64)
                };
                for i in 0..n {
                    if assignment[i] == c {
                        total_sim += cosine_similarity(&vectors[i], &centroid);
                        count += 1;
                    }
                }
            }
            let avg_sim = if count > 0 { total_sim / count as f64 } else { 0.0 };

            if avg_sim > best_score {
                best_score = avg_sim;
                best_assignment = assignment.clone();
            }
        }

        (best_assignment, best_score)
    }

    fn get_top_terms(cluster_doc_indices: &[usize], vocab: &[String], vectors: &[Vec<f64>], top_n: usize) -> Vec<(String, f64)> {
        let dim = vocab.len();
        let mut centroid = vec![0.0; dim];
        for &i in cluster_doc_indices {
            centroid = vector_add(&centroid, &vectors[i]);
        }
        if !cluster_doc_indices.is_empty() {
            centroid = vector_scale(&centroid, 1.0 / cluster_doc_indices.len() as f64);
        }
        let mut term_scores: Vec<(String, f64)> = (0..dim)
            .filter(|&i| centroid[i] > 0.0)
            .map(|i| (vocab[i].clone(), centroid[i]))
            .collect();
        term_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        term_scores.truncate(top_n);
        term_scores
    }

    // --- Our corpus: 12 documents, 3 implicit topics ---
    let documents: Vec<(&str, &str)> = vec![
        ("D01", "The team scored three goals in the championship match"),
        ("D02", "Players trained hard for the upcoming tournament season"),
        ("D03", "The coach announced the starting lineup for the final game"),
        ("D04", "Fans cheered as the winning goal was scored in extra time"),
        ("D05", "New programming language features improve developer productivity"),
        ("D06", "The software update fixed critical security vulnerabilities"),
        ("D07", "Cloud computing platforms scale applications automatically"),
        ("D08", "Developers debug code using integrated testing frameworks"),
        ("D09", "The chef prepared a traditional recipe with fresh ingredients"),
        ("D10", "Baking bread requires precise measurements of flour and yeast"),
        ("D11", "The restaurant menu featured seasonal dishes and local produce"),
        ("D12", "Cooking techniques vary across different regional cuisines"),
    ];

    // Preprocess and vectorize
    let names: Vec<&str> = documents.iter().map(|(n, _)| *n).collect();
    let texts: Vec<&str> = documents.iter().map(|(_, t)| *t).collect();
    let doc_tokens_list: Vec<Vec<String>> = texts.iter().map(|t| preprocess(t)).collect();
    let (vocab, vectors) = compute_tfidf(&doc_tokens_list);

    println!("{}", "=".repeat(60));
    println!("CORPUS");
    println!("{}", "=".repeat(60));
    for (name, text) in &documents {
        println!("  {}: {}", name, text);
    }
    println!("\nVocabulary size: {} terms", vocab.len());
    println!("Documents: {}", documents.len());
    println!();

    // --- Cluster into K=3 groups ---
    let k = 3;
    let mut rng = Rng::new(42);
    let (assignment, score) = kmeans_cosine(&vectors, k, 50, 5, &mut rng);

    println!("{}", "=".repeat(60));
    println!("CLUSTERING RESULTS (K={})", k);
    println!("{}", "=".repeat(60));
    println!("Best average within-cluster similarity: {:.4}", score);
    println!();

    let mut cluster_sizes = Vec::new();
    let mut cluster_top_terms_data = Vec::new();
    let cluster_colors = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"];

    for c in 0..k {
        let members: Vec<usize> = (0..names.len()).filter(|&i| assignment[i] == c).collect();
        let top_terms = get_top_terms(&members, &vocab, &vectors, 5);
        cluster_sizes.push(members.len() as f64);

        let term_str: String = top_terms.iter()
            .map(|(w, s)| format!("{} ({:.3})", w, s))
            .collect::<Vec<_>>().join(", ");
        println!("--- Cluster {} ({} docs) ---", c + 1, members.len());
        println!("  Top terms: {}", term_str);
        for &i in &members {
            let preview: String = texts[i].chars().take(65).collect();
            println!("    {}: {}...", names[i], preview);
        }
        println!();
        cluster_top_terms_data.push(top_terms);
    }

    // Visualize cluster sizes
    let cluster_labels: Vec<String> = (0..k).map(|c| format!("Cluster {}", c + 1)).collect();
    let cluster_label_refs: Vec<&str> = cluster_labels.iter().map(|s| s.as_str()).collect();
    show_chart(&bar_chart(
        "Cluster Sizes (Number of Documents per Cluster)",
        &cluster_label_refs,
        "Documents",
        &cluster_sizes,
        "#3b82f6",
    ));

    // Visualize top terms per cluster
    for c in 0..k {
        let terms = &cluster_top_terms_data[c];
        if !terms.is_empty() {
            let labels: Vec<&str> = terms.iter().map(|(w, _)| w.as_str()).collect();
            let data: Vec<f64> = terms.iter().map(|(_, s)| (*s * 10000.0).round() / 10000.0).collect();
            show_chart(&bar_chart(
                &format!("Cluster {} — Top TF-IDF Terms", c + 1),
                &labels,
                "TF-IDF Weight",
                &data,
                cluster_colors[c % cluster_colors.len()],
            ));
        }
    }

    // --- Show pairwise similarity matrix for insight ---
    println!("{}", "=".repeat(60));
    println!("WITHIN-CLUSTER vs BETWEEN-CLUSTER SIMILARITY");
    println!("{}", "=".repeat(60));

    let mut within_avgs = Vec::new();
    for c in 0..k {
        let members: Vec<usize> = (0..names.len()).filter(|&i| assignment[i] == c).collect();
        if members.len() < 2 {
            within_avgs.push(0.0);
            continue;
        }
        let mut sims = Vec::new();
        for a in 0..members.len() {
            for b in (a + 1)..members.len() {
                sims.push(cosine_similarity(&vectors[members[a]], &vectors[members[b]]));
            }
        }
        let avg = if sims.is_empty() { 0.0 } else { sims.iter().sum::<f64>() / sims.len() as f64 };
        within_avgs.push((avg * 10000.0).round() / 10000.0);
        println!("  Cluster {} avg within-similarity: {:.4}", c + 1, avg);
    }

    let mut between_sims = Vec::new();
    for i in 0..names.len() {
        for j in (i + 1)..names.len() {
            if assignment[i] != assignment[j] {
                between_sims.push(cosine_similarity(&vectors[i], &vectors[j]));
            }
        }
    }
    let avg_between = if between_sims.is_empty() { 0.0 } else { between_sims.iter().sum::<f64>() / between_sims.len() as f64 };
    println!("  Average between-cluster similarity:    {:.4}", avg_between);
    println!();
    println!("Good clustering: within-cluster similarity >> between-cluster similarity");
    println!();

    // Visualize within vs between
    let mut bar_labels: Vec<String> = (0..k).map(|c| format!("Cluster {}", c + 1)).collect();
    bar_labels.push("Between Clusters".to_string());
    let bar_label_refs: Vec<&str> = bar_labels.iter().map(|s| s.as_str()).collect();
    let mut bar_data = within_avgs.clone();
    bar_data.push((avg_between * 10000.0).round() / 10000.0);
    show_chart(&bar_chart(
        "Within-Cluster vs Between-Cluster Similarity",
        &bar_label_refs,
        "Avg Cosine Similarity",
        &bar_data,
        "#8b5cf6",
    ));

    // --- Show what happens with different K values ---
    println!("{}", "=".repeat(60));
    println!("EFFECT OF CHOOSING DIFFERENT K");
    println!("{}", "=".repeat(60));

    for kv in [2, 3, 4, 5] {
        let mut rng2 = Rng::new(42);
        let (asgn, sc) = kmeans_cosine(&vectors, kv, 50, 5, &mut rng2);
        let csizes: Vec<usize> = (0..kv).map(|c| asgn.iter().filter(|&&a| a == c).count()).collect();
        println!("  K={}: avg similarity={:.4}, cluster sizes={:?}", kv, sc, csizes);
    }

    println!();
    println!("Note: K=3 matches our 3 natural topics (sports, tech, cooking).");
    println!("K=2 forces two topics to merge. K=5 splits coherent topics apart.");
    println!();
    println!("Many NLP problems reduce to similarity in vector space.");
    println!("Clustering discovers structure without labels — purely from");
    println!("the geometric arrangement of TF-IDF document vectors.");
}
```

---

## Key Takeaways

- **Clustering is unsupervised similarity.** Documents that point in similar directions in TF-IDF space naturally group together, revealing latent topics without any human labels.
- **K-means is simple but powerful.** The iterate-until-stable loop of "assign to nearest centroid, recompute centroids" discovers meaningful document groups from raw text vectors.
- **Choosing K is a real problem.** Too few clusters merge distinct topics; too many fragment coherent ones. There is no single right answer — K encodes your assumption about how many topics exist.
- **Good clusters have high internal similarity and low external similarity.** This is the geometric definition of "these documents belong together." Many NLP problems reduce to similarity in vector space.
