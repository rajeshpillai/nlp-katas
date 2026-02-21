use std::time::Instant;
use tempfile::TempDir;
use tokio::time::timeout;

use crate::config;
use crate::models::execution::ExecutionResult;

const VIZ_HELPERS: &str = include_str!("../../helpers/nlp_katas_viz.rs");

pub async fn execute_rust_code(code: &str) -> ExecutionResult {
    let start = Instant::now();

    let full_code = format!("{}\n\n{}", VIZ_HELPERS, code);

    let tmp_dir = match TempDir::new() {
        Ok(d) => d,
        Err(e) => return ExecutionResult::error(format!("Failed to create temp dir: {}", e)),
    };

    let source_path = tmp_dir.path().join("main.rs");
    let binary_path = tmp_dir.path().join("main");

    if let Err(e) = tokio::fs::write(&source_path, &full_code).await {
        return ExecutionResult::error(format!("Failed to write source: {}", e));
    }

    // Compile with rustc
    let compile_result = timeout(
        config::compile_timeout(),
        tokio::process::Command::new("rustc")
            .arg("--edition")
            .arg("2021")
            .arg(&source_path)
            .arg("-o")
            .arg(&binary_path)
            .output(),
    )
    .await;

    let compile_output = match compile_result {
        Ok(Ok(output)) => output,
        Ok(Err(e)) => return ExecutionResult::error(format!("Failed to run rustc: {}", e)),
        Err(_) => {
            return ExecutionResult {
                stdout: String::new(),
                stderr: String::new(),
                error: Some("Compilation timed out after 15 seconds.".into()),
                execution_time_ms: start.elapsed().as_millis() as f64,
            }
        }
    };

    if !compile_output.status.success() {
        let stderr = String::from_utf8_lossy(&compile_output.stderr).to_string();
        // Strip temp dir paths from error messages for cleaner output
        let cleaned_stderr = clean_error_paths(&stderr, tmp_dir.path().to_str().unwrap_or(""));
        return ExecutionResult {
            stdout: String::new(),
            stderr: cleaned_stderr,
            error: Some("Compilation failed.".into()),
            execution_time_ms: start.elapsed().as_millis() as f64,
        };
    }

    // Run the compiled binary
    let run_result = timeout(
        config::run_timeout(),
        tokio::process::Command::new(&binary_path).output(),
    )
    .await;

    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

    match run_result {
        Ok(Ok(output)) => ExecutionResult {
            stdout: String::from_utf8_lossy(&output.stdout).to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
            error: if output.status.success() {
                None
            } else {
                Some("Runtime error.".into())
            },
            execution_time_ms: (elapsed * 100.0).round() / 100.0,
        },
        Ok(Err(e)) => ExecutionResult::error(format!("Failed to run binary: {}", e)),
        Err(_) => ExecutionResult {
            stdout: String::new(),
            stderr: String::new(),
            error: Some("Execution timed out after 30 seconds.".into()),
            execution_time_ms: elapsed,
        },
    }
}

fn clean_error_paths(stderr: &str, tmp_path: &str) -> String {
    stderr.replace(&format!("{}/main.rs", tmp_path), "main.rs")
}
