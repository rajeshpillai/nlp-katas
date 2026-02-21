use std::time::Duration;

pub fn port() -> u16 {
    std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8001)
}

pub fn compile_timeout() -> Duration {
    let secs = std::env::var("NLP_KATAS_COMPILE_TIMEOUT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(15);
    Duration::from_secs(secs)
}

pub fn run_timeout() -> Duration {
    let secs = std::env::var("NLP_KATAS_RUN_TIMEOUT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(30);
    Duration::from_secs(secs)
}
