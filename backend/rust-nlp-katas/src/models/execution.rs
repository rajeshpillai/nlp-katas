use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct ExecutionRequest {
    pub code: String,
    pub kata_id: String,
}

#[derive(Debug, Serialize)]
pub struct ExecutionResult {
    pub stdout: String,
    pub stderr: String,
    pub error: Option<String>,
    pub execution_time_ms: f64,
}

impl ExecutionResult {
    pub fn error(msg: String) -> Self {
        Self {
            stdout: String::new(),
            stderr: String::new(),
            error: Some(msg),
            execution_time_ms: 0.0,
        }
    }
}
