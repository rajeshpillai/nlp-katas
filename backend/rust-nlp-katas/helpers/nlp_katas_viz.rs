// --- NLP Katas Visualization Helpers ---
// This code is automatically prepended to kata code for rich output support.
// Do NOT add `fn main()` here — the kata code provides its own.

#[allow(dead_code)]
fn show_html(html: &str) {
    println!("<!--NLP_RICH:html-->\n{}\n<!--/NLP_RICH-->", html);
}

#[allow(dead_code)]
fn show_svg(svg: &str) {
    println!("<!--NLP_RICH:svg-->\n{}\n<!--/NLP_RICH-->", svg);
}

#[allow(dead_code)]
fn show_chart(json: &str) {
    println!("<!--NLP_RICH:chart-->\n{}\n<!--/NLP_RICH-->", json);
}

#[allow(dead_code)]
fn show_image_base64(data_uri: &str) {
    println!("<!--NLP_RICH:image-->\n{}\n<!--/NLP_RICH-->", data_uri);
}

// --- Base64 encoder (no external crates) ---
#[allow(dead_code)]
fn base64_encode(data: &[u8]) -> String {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut result = String::with_capacity((data.len() + 2) / 3 * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;
        result.push(CHARS[((triple >> 18) & 0x3F) as usize] as char);
        result.push(CHARS[((triple >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            result.push(CHARS[((triple >> 6) & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
        if chunk.len() > 2 {
            result.push(CHARS[(triple & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
    }
    result
}

// --- JSON string escaping (no external crates) ---
#[allow(dead_code)]
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

// --- Chart builder helpers ---

#[allow(dead_code)]
fn bar_chart(title: &str, labels: &[&str], dataset_label: &str, data: &[f64], color: &str) -> String {
    let labels_json = labels.iter().map(|l| format!("\"{}\"", json_escape(l))).collect::<Vec<_>>().join(",");
    let data_json = data.iter().map(|d| format!("{:.6}", d)).collect::<Vec<_>>().join(",");
    format!(
        r#"{{"type":"bar","title":"{}","labels":[{}],"datasets":[{{"label":"{}","data":[{}],"color":"{}"}}]}}"#,
        json_escape(title), labels_json, json_escape(dataset_label), data_json, color
    )
}

#[allow(dead_code)]
fn multi_bar_chart(title: &str, labels: &[&str], datasets: &[(&str, &[f64], &str)]) -> String {
    let labels_json = labels.iter().map(|l| format!("\"{}\"", json_escape(l))).collect::<Vec<_>>().join(",");
    let ds_json: Vec<String> = datasets.iter().map(|(label, data, color)| {
        let data_json = data.iter().map(|d| format!("{:.6}", d)).collect::<Vec<_>>().join(",");
        format!(r#"{{"label":"{}","data":[{}],"color":"{}"}}"#, json_escape(label), data_json, color)
    }).collect();
    format!(
        r#"{{"type":"bar","title":"{}","labels":[{}],"datasets":[{}]}}"#,
        json_escape(title), labels_json, ds_json.join(",")
    )
}

#[allow(dead_code)]
fn line_chart(title: &str, labels: &[&str], dataset_label: &str, data: &[f64], color: &str) -> String {
    let labels_json = labels.iter().map(|l| format!("\"{}\"", json_escape(l))).collect::<Vec<_>>().join(",");
    let data_json = data.iter().map(|d| format!("{:.6}", d)).collect::<Vec<_>>().join(",");
    format!(
        r#"{{"type":"line","title":"{}","labels":[{}],"datasets":[{{"label":"{}","data":[{}],"color":"{}"}}]}}"#,
        json_escape(title), labels_json, json_escape(dataset_label), data_json, color
    )
}

#[allow(dead_code)]
fn scatter_chart(title: &str, datasets: &[(&str, &[(f64, f64)], &str)]) -> String {
    let ds_json: Vec<String> = datasets.iter().map(|(label, points, color)| {
        let pts = points.iter().map(|(x, y)| format!(r#"{{"x":{:.6},"y":{:.6}}}"#, x, y)).collect::<Vec<_>>().join(",");
        format!(r#"{{"label":"{}","data":[{}],"color":"{}"}}"#, json_escape(label), pts, color)
    }).collect();
    format!(
        r#"{{"type":"scatter","title":"{}","datasets":[{}]}}"#,
        json_escape(title), ds_json.join(",")
    )
}

#[allow(dead_code)]
fn pie_chart(title: &str, labels: &[&str], data: &[f64]) -> String {
    let labels_json = labels.iter().map(|l| format!("\"{}\"", json_escape(l))).collect::<Vec<_>>().join(",");
    let data_json = data.iter().map(|d| format!("{:.6}", d)).collect::<Vec<_>>().join(",");
    format!(
        r#"{{"type":"pie","title":"{}","labels":[{}],"datasets":[{{"data":[{}]}}]}}"#,
        json_escape(title), labels_json, data_json
    )
}

#[allow(dead_code)]
fn heatmap_chart(title: &str, x_labels: &[&str], y_labels: &[&str], data: &[Vec<f64>]) -> String {
    let xl = x_labels.iter().map(|l| format!("\"{}\"", json_escape(l))).collect::<Vec<_>>().join(",");
    let yl = y_labels.iter().map(|l| format!("\"{}\"", json_escape(l))).collect::<Vec<_>>().join(",");
    let rows: Vec<String> = data.iter().map(|row| {
        let vals = row.iter().map(|v| format!("{:.6}", v)).collect::<Vec<_>>().join(",");
        format!("[{}]", vals)
    }).collect();
    format!(
        r#"{{"type":"heatmap","title":"{}","x_labels":[{}],"y_labels":[{}],"data":[{}]}}"#,
        json_escape(title), xl, yl, rows.join(",")
    )
}

// --- Simple deterministic PRNG (xorshift64) ---
#[allow(dead_code)]
struct Rng {
    state: u64,
}

#[allow(dead_code)]
impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: if seed == 0 { 1 } else { seed } }
    }

    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() as f64) / (u64::MAX as f64)
    }

    fn gauss(&mut self, mean: f64, std: f64) -> f64 {
        // Box-Muller transform
        let u1 = self.next_f64().max(1e-10);
        let u2 = self.next_f64();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        mean + std * z
    }
}
