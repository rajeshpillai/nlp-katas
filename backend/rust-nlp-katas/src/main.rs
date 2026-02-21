mod config;
mod models;
mod routes;
mod services;

use axum::routing::{get, post};
use axum::Router;
use std::path::PathBuf;
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};

#[derive(Clone)]
pub struct AppState {
    pub content_dir: PathBuf,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let content_dir = std::env::var("CONTENT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("content"));

    services::kata_loader::validate_katas(&content_dir);

    let state = Arc::new(AppState { content_dir });

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let api = Router::new()
        .route("/tracks", get(routes::tracks::list_tracks))
        .route("/tracks/{track_id}/katas", get(routes::katas::list_katas))
        .route(
            "/tracks/{track_id}/katas/{phase_id}/{kata_id}/content",
            get(routes::katas::get_kata_content),
        )
        .route("/execute", post(routes::execute::run_code))
        .route("/execute/{track_id}", post(routes::execute::run_code))
        .with_state(state);

    let app = Router::new()
        .route("/health", get(routes::health::health_check))
        .nest("/api", api)
        .layer(cors);

    let port = config::port();
    let addr = format!("0.0.0.0:{}", port);

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .unwrap_or_else(|_| panic!("Failed to bind to {}", addr));

    tracing::info!("Rust NLP Katas server running on http://localhost:{}", port);
    println!("Rust NLP Katas server running on http://localhost:{}", port);

    axum::serve(listener, app).await.unwrap();
}
