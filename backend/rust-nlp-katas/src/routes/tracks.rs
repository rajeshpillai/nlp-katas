use axum::Json;

use crate::models::kata::Track;

pub async fn list_tracks() -> Json<Vec<Track>> {
    Json(vec![
        Track {
            id: "python-nlp".into(),
            name: "Python NLP".into(),
            description: "Learn NLP from foundations to transformers using Python. \
                Build intuition for how text becomes numbers and why each technique exists."
                .into(),
            status: "active".into(),
        },
        Track {
            id: "rust-nlp".into(),
            name: "Rust NLP".into(),
            description: "Learn NLP from foundations to transformers using Rust. \
                Build intuition for how text becomes numbers with systems-level control."
                .into(),
            status: "active".into(),
        },
    ])
}
