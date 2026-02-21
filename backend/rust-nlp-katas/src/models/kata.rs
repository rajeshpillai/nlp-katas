use serde::Serialize;
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize)]
pub struct Track {
    pub id: String,
    pub name: String,
    pub description: String,
    pub status: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct KataInfo {
    pub id: String,
    pub title: String,
    pub phase: u32,
    pub sequence: u32,
    pub track_id: String,
}

#[derive(Debug, Serialize)]
pub struct KatasResponse {
    pub katas: Vec<KataInfo>,
    pub phases: HashMap<String, String>,
}
