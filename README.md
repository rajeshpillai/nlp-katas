# NLP Katas

Learn NLP as a representation and modeling discipline — from raw text to transformers.

## Quick Start

Start all three services (Python backend, Rust backend, frontend):

```bash
# Terminal 1 — Python backend (port 8000)
cd backend/python-nlp-katas
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Terminal 2 — Rust backend (port 8001)
cd backend/rust-nlp-katas
cargo run

# Terminal 3 — Frontend (port 3000)
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000` — both Python and Rust tracks will be available.

## Backends

### Python Track (port 8000)

```bash
cd backend/python-nlp-katas
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Executes user code via subprocess with Python 3. Includes matplotlib support and a viz helper module for rich output (charts, HTML, SVG, images).

### Rust Track (port 8001)

```bash
cd backend/rust-nlp-katas
cargo run
```

Compiles and runs user code via `rustc --edition 2021` in a temp directory sandbox. Visualization helpers (`show_chart`, `show_html`, `show_svg`, chart builders) are automatically prepended to user code — no external crates needed.

**Environment variables:**

| Variable | Default | Description |
|---|---|---|
| `PORT` | `8001` | Server port |
| `CONTENT_DIR` | `.` | Path to kata content directory |
| `COMPILE_TIMEOUT` | `15` | Rust compilation timeout (seconds) |
| `RUN_TIMEOUT` | `30` | Code execution timeout (seconds) |

### API Endpoints (both backends)

| Endpoint | Description |
|---|---|
| `GET /health` | Health check |
| `GET /api/tracks` | List available tracks |
| `GET /api/tracks/{track_id}/katas` | List all katas for a track |
| `GET /api/tracks/{track_id}/katas/{phase}/{kata_id}/content` | Get kata markdown |
| `POST /api/execute/{track_id}` | Execute code (`{"code": "...", "kata_id": "..."}`) |

## Frontend

```bash
cd frontend
npm install
npm run dev
```

SolidJS + Tailwind CSS v4 app on port 3000. The Vite dev server proxies API requests to the correct backend based on track:

- `/api/tracks/rust-nlp/*` and `/api/execute/rust-nlp` → port 8001
- All other `/api/*` requests → port 8000

## Project Structure

```
nlp-katas/
├── frontend/                        (SolidJS + Tailwind CSS v4)
│   ├── src/
│   │   ├── pages/                   (landing, track-page, kata-page)
│   │   ├── components/              (layout, markdown, kata-workspace)
│   │   └── lib/                     (api-client)
│   └── vite.config.ts               (proxy routing)
├── backend/
│   ├── python-nlp-katas/            (FastAPI, port 8000)
│   │   ├── main.py
│   │   ├── app/                     (routes, models, services)
│   │   └── content/phase-0/ through phase-10/
│   └── rust-nlp-katas/              (Axum, port 8001)
│       ├── src/                     (main, routes, models, services)
│       ├── helpers/nlp_katas_viz.rs (viz helpers prepended to user code)
│       └── content/phase-0/ through phase-10/
├── todo.md                          (global checklist)
└── CLAUDE.md                        (project instructions)
```

## Kata Tracks

Both tracks cover the same 33 katas across 11 phases — same NLP concepts, different language implementations:

| Phase | Topic | Katas |
|---|---|---|
| 0 | Language & Text (Foundations) | 3 |
| 1 | Text Preprocessing | 3 |
| 2 | Bag of Words (BoW) | 3 |
| 3 | TF-IDF | 3 |
| 4 | Similarity & Classical NLP Tasks | 3 |
| 5 | Tokenization (Deep Dive) | 3 |
| 6 | Named Entity Recognition (NER) | 3 |
| 7 | Small Neural Text Models | 3 |
| 8 | Context & Sequence Modeling | 3 |
| 9 | Transformer Architecture (Core Concepts) | 3 |
| 10 | Modern NLP Pipelines (Awareness) | 3 |
