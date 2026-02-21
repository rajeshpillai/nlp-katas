# NLP Katas — Global Checklist

## Tracks

### Python (`python-nlp-katas`)
- [x] Phase 0 — Language & Text (Foundations) — 3 katas
- [x] Phase 1 — Text Preprocessing — 3 katas
- [x] Phase 2 — Bag of Words (BoW) — 3 katas
- [x] Phase 3 — TF-IDF — 3 katas
- [x] Phase 4 — Similarity & Classical NLP Tasks — 3 katas
- [x] Phase 5 — Tokenization (Deep Dive) — 3 katas
- [x] Phase 6 — Named Entity Recognition (NER) — 3 katas
- [x] Phase 7 — Small Neural Text Models — 3 katas
- [x] Phase 8 — Context & Sequence Modeling — 3 katas
- [x] Phase 9 — Transformer Architecture (Core Concepts) — 3 katas
- [x] Phase 10 — Modern NLP Pipelines (Awareness) — 3 katas

### Rust (`rust-nlp-katas`)
- [x] Phase 0 — Language & Text (Foundations) — 3 katas
- [x] Phase 1 — Text Preprocessing — 3 katas
- [x] Phase 2 — Bag of Words (BoW) — 3 katas
- [x] Phase 3 — TF-IDF — 3 katas
- [x] Phase 4 — Similarity & Classical NLP Tasks — 3 katas
- [x] Phase 5 — Tokenization (Deep Dive) — 3 katas
- [x] Phase 6 — Named Entity Recognition (NER) — 3 katas
- [x] Phase 7 — Small Neural Text Models — 3 katas
- [x] Phase 8 — Context & Sequence Modeling — 3 katas
- [x] Phase 9 — Transformer Architecture (Core Concepts) — 3 katas
- [x] Phase 10 — Modern NLP Pipelines (Awareness) — 3 katas

## Frontend
- [x] SolidJS + Tailwind CSS v4 setup
- [x] Kata markdown renderer
- [x] CodeMirror 6 code editor with Python and Rust syntax highlighting
- [x] Resizable code/output split panels
- [x] Collapsible sidebar with hamburger menu
- [x] Dark/light theme toggle (sun/moon icons)
- [x] Panel maximize/restore
- [x] Rich output panel (HTML, SVG, Chart.js, base64 images)

## Backend
- [x] FastAPI with subprocess-based code execution (Python track, port 8000)
- [x] Axum with rustc-based code execution (Rust track, port 8001)
- [x] Kata content API (markdown serving)
- [x] Python viz helper module (show_chart, show_html, show_svg, plt.show patch)
- [x] Rust viz helper module (show_chart, show_html, show_svg, chart builders)
- [x] Matplotlib support (Python track)
- [x] Multi-track proxy routing (vite proxy → port 8000/8001 by track)

## Rich Visualizations
- [x] Chart.js bar charts, line charts, scatter plots
- [x] Heatmaps (attention weights, similarity matrices, confusion matrices)
- [x] Matplotlib inline images (embedding scatter plots, comparison charts)
- [x] 25 of 33 katas produce rich visual output

---

## App Improvement Recommendations

### High Priority
- [ ] **Save user code to localStorage** — persist edits per kata so learners don't lose work on navigation or refresh
- [ ] **Keyboard shortcut overlay** — show Ctrl+Enter, Tab, Ctrl+Z hints on first visit
- [ ] **Loading skeleton** — show content placeholders while kata markdown loads instead of "Loading..." text
- [ ] **Error line highlighting** — parse Python traceback line numbers and highlight the error line in the editor
- [ ] **Mobile responsive layout** — sidebar overlay mode and stacked code/output panels on small screens

### Medium Priority
- [ ] **Progress tracking** — mark katas as completed (localStorage), show progress per phase in sidebar
- [ ] **Kata navigation arrows** — prev/next buttons to move between katas without sidebar
- [ ] **Code diff on reset** — show what changed before resetting to starter code
- [ ] **Output panel auto-scroll** — scroll to bottom on new output, scroll to first error on failure
- [ ] **Search/filter katas** — search bar in sidebar to filter katas by title or phase
- [ ] **Chart dark mode sync** — ensure Chart.js/matplotlib plots use theme-appropriate backgrounds
- [ ] **Lazy-load Chart.js** — dynamic import to reduce initial bundle size (~65KB savings)

### Low Priority / Nice to Have
- [ ] **Export code** — download current kata code as `.py` file
- [ ] **Share kata results** — generate shareable link with code + output snapshot
- [ ] **Kata difficulty indicators** — show beginner/intermediate/advanced tags
- [ ] **Code execution history** — show previous runs and their outputs
- [ ] **Markdown TOC sidebar** — show concept section headings for quick navigation within a kata
- [ ] **Interactive experiments** — add editable text inputs in the concept tab (not just the code tab)
- [ ] **Landing page** — proper homepage with track selection, learning path overview, and getting started guide
- [x] **Rust track** — port katas to Rust with a Rust backend execution environment
- [ ] **Multi-language support** — allow switching between Python/Rust implementations of the same kata
- [ ] **User accounts** — optional login to sync progress across devices
