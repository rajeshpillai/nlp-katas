use std::path::Path;

pub fn validate_katas(content_dir: &Path) {
    let mut found = 0;
    let mut missing = 0;

    for phase in 0..=10 {
        let phase_dir = content_dir.join(format!("phase-{}", phase));
        if !phase_dir.exists() {
            tracing::warn!("Missing phase directory: phase-{}", phase);
            missing += 3;
            continue;
        }

        if let Ok(entries) = std::fs::read_dir(&phase_dir) {
            let count = entries
                .filter_map(|e| e.ok())
                .filter(|e| {
                    e.path()
                        .extension()
                        .map_or(false, |ext| ext == "md")
                })
                .count();
            found += count;
            if count < 3 {
                tracing::warn!(
                    "phase-{} has {} kata(s), expected 3",
                    phase,
                    count
                );
                missing += 3 - count;
            }
        }
    }

    tracing::info!(
        "Kata validation: {} found, {} missing",
        found,
        missing
    );
}
