use std::collections::HashMap;
use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;

use crate::models::kata::{KataInfo, KatasResponse};
use crate::AppState;

fn rust_nlp_katas() -> Vec<KataInfo> {
    let katas = vec![
        // Phase 0 — Language & Text (Foundations)
        ("explore-ambiguity", "Explore Ambiguity in Sentences", 0, 1),
        ("identify-noise", "Identify Noise in Real-World Text", 0, 2),
        ("structured-vs-unstructured", "Compare Structured vs Unstructured Text", 0, 3),
        // Phase 1 — Text Preprocessing
        ("preprocessing-pipeline", "Apply a Preprocessing Pipeline to Raw Text", 1, 1),
        ("stemming-vs-lemmatization", "Compare Stemming vs Lemmatization Output", 1, 2),
        ("stopword-removal-similarity", "Measure How Stopword Removal Changes Document Similarity", 1, 3),
        // Phase 2 — Bag of Words (BoW)
        ("build-bow", "Build Bag of Words from Scratch", 2, 1),
        ("visualize-document-vectors", "Visualize Document Vectors", 2, 2),
        ("compare-documents-bow", "Compare Documents Using Bag of Words", 2, 3),
        // Phase 3 — TF-IDF
        ("compute-tfidf", "Compute TF-IDF Manually", 3, 1),
        ("bow-vs-tfidf", "Compare Similarity Using BoW vs TF-IDF", 3, 2),
        ("visualize-word-importance", "Visualize Word Importance", 3, 3),
        // Phase 4 — Similarity & Classical NLP Tasks
        ("cosine-similarity", "Compute Cosine Similarity Between Document Pairs", 4, 1),
        ("text-search-engine", "Build a Simple Text Search Engine", 4, 2),
        ("cluster-documents", "Cluster Documents by Topic", 4, 3),
        // Phase 5 — Tokenization (Deep Dive)
        ("tokenization-methods", "Tokenize Text Using Word, Character, and Subword Methods", 5, 1),
        ("byte-pair-encoding", "Implement BPE from Scratch", 5, 2),
        ("vocabulary-comparison", "Compare Vocabulary Sizes and OOV Handling Across Methods", 5, 3),
        // Phase 6 — Named Entity Recognition (NER)
        ("rule-based-ner", "Rule-Based Named Entity Recognition", 6, 1),
        ("statistical-ner", "Simple ML-Based Named Entity Recognition", 6, 2),
        ("ner-error-analysis", "NER Error Analysis", 6, 3),
        // Phase 7 — Small Neural Text Models
        ("word-embeddings", "Train Small Embedding-Based Models", 7, 1),
        ("visualize-embeddings", "Visualize Embedding Spaces", 7, 2),
        ("neural-vs-tfidf", "Compare Neural vs TF-IDF Models", 7, 3),
        // Phase 8 — Context & Sequence Modeling
        ("word-order-matters", "Show How Word Order Changes Meaning", 8, 1),
        ("context-aware-representations", "Compare Context-Aware vs Context-Free Representations", 8, 2),
        ("sequence-modeling-challenges", "Demonstrate Sequence Modeling Challenges", 8, 3),
        // Phase 9 — Transformer Architecture (Core Concepts)
        ("attention-weights", "Visualize Attention Weights", 9, 1),
        ("tiny-transformer-block", "Build a Tiny Transformer Block", 9, 2),
        ("encoder-vs-decoder", "Compare Encoder-Only vs Decoder-Only Tasks", 9, 3),
        // Phase 10 — Modern NLP Pipelines (Awareness)
        ("pretraining-vs-finetuning", "Pretraining vs Fine-Tuning", 10, 1),
        ("encoder-decoder-models", "Encoder-Only vs Decoder-Only Models", 10, 2),
        ("where-llms-fit", "Where LLMs Fit in the NLP Stack", 10, 3),
    ];

    katas
        .into_iter()
        .map(|(id, title, phase, sequence)| KataInfo {
            id: id.into(),
            title: title.into(),
            phase,
            sequence,
            track_id: "rust-nlp".into(),
        })
        .collect()
}

fn phase_names() -> HashMap<String, String> {
    [
        ("0", "Language & Text (Foundations)"),
        ("1", "Text Preprocessing"),
        ("2", "Bag of Words (BoW)"),
        ("3", "TF-IDF"),
        ("4", "Similarity & Classical NLP Tasks"),
        ("5", "Tokenization (Deep Dive)"),
        ("6", "Named Entity Recognition (NER)"),
        ("7", "Small Neural Text Models"),
        ("8", "Context & Sequence Modeling"),
        ("9", "Transformer Architecture (Core Concepts)"),
        ("10", "Modern NLP Pipelines (Awareness)"),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect()
}

pub async fn list_katas(
    Path(track_id): Path<String>,
    State(_state): State<Arc<AppState>>,
) -> Response {
    if track_id != "rust-nlp" {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"detail": format!("Track '{}' not found", track_id)})),
        )
            .into_response();
    }

    Json(KatasResponse {
        katas: rust_nlp_katas(),
        phases: phase_names(),
    })
    .into_response()
}

pub async fn get_kata_content(
    Path((track_id, phase_id, kata_id)): Path<(String, u32, String)>,
    State(state): State<Arc<AppState>>,
) -> Response {
    if track_id != "rust-nlp" {
        return (
            StatusCode::NOT_FOUND,
            format!("Track '{}' not found", track_id),
        )
            .into_response();
    }

    let katas = rust_nlp_katas();
    let kata = katas
        .iter()
        .find(|k| k.id == kata_id && k.phase == phase_id);

    let kata = match kata {
        Some(k) => k,
        None => {
            return (
                StatusCode::NOT_FOUND,
                format!("Kata '{}' not found in phase {}", kata_id, phase_id),
            )
                .into_response();
        }
    };

    let filename = format!("{:02}-{}.md", kata.sequence, kata.id);
    let filepath = state
        .content_dir
        .join(format!("phase-{}", phase_id))
        .join(&filename);

    match tokio::fs::read_to_string(&filepath).await {
        Ok(content) => content.into_response(),
        Err(_) => (
            StatusCode::NOT_FOUND,
            format!("Content file not found: {}", filename),
        )
            .into_response(),
    }
}
