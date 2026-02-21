# NLP Katas

Teach NLP as a **representation and modeling discipline** — not a shortcut to LLMs.
Every technique — BoW, TF-IDF, NER, Transformers — must be explained through this lens.

---

## Project Structure

```
nlp-katas/
├── frontend/                         (shared SolidJS + Tailwind CSS v4 app)
│   ├── src/
│   │   ├── pages/                    (landing, track-page, kata-page)
│   │   ├── components/
│   │   │   ├── layout/               (main-layout, sidebar, theme-toggle)
│   │   │   ├── markdown-content/     (kata markdown renderer)
│   │   │   └── kata-workspace/       (code-panel, output-panel, resizable)
│   │   ├── context/                  (theme-context)
│   │   └── lib/                      (api-client)
│   ├── package.json
│   └── vite.config.ts                (port 3000, proxies /api to backend)
├── backend/
│   ├── python-nlp-katas/             (Python track: FastAPI backend + kata content)
│   │   ├── main.py                   (FastAPI entry, port 8000)
│   │   ├── requirements.txt
│   │   ├── app/                      (routes, models, services)
│   │   ├── content/phase-0/          (kata markdown files)
│   │   └── todo.md
│   └── rust-nlp-katas/               (Rust track: Axum backend + kata content)
│       ├── src/                      (main, routes, models, services)
│       ├── helpers/nlp_katas_viz.rs  (viz helpers prepended to user code)
│       ├── content/phase-0/          (kata markdown files)
│       └── todo.md
├── todo.md                           (global checklist)
└── CLAUDE.md
```

- Each language track lives under `backend/` with its own backend server + kata content
- Future tracks (e.g. `rust-nlp-katas`) will follow the same pattern under `backend/`
- The `frontend/` is shared across all tracks

## Coding Conventions

- All file/folder names in lowercase-hyphenated
- All katas are defined as markdown files in `phase-N/` directories
- Kata file naming: `{sequence:02d}-{kata-id}.md` (e.g. `01-explore-ambiguity.md`)
- Frontend: SolidJS + Tailwind CSS v4, TypeScript, no inline CSS (create CSS classes
  for UI elements/components), each component gets its own `.css` file
- Backend: FastAPI, Python, subprocess-based code execution with timeout
- Create a global `todo.md` that tracks overall features
- Create a `todo.md` for each track, and track language implementations

---

## Learning Sequence (MANDATORY ORDER)

You must follow this order **strictly**, even if it contradicts modern “LLM-first” tutorials.

## PHASE 0 — Language & Text (Foundations)

**Goal:** Understand the raw material

Teach:
- What is language vs text
- Characters, words, sentences, documents
- Ambiguity in language
- Noise in real-world text
- Why NLP is harder than vision

Katas:
- Explore ambiguity in sentences (same word, different meanings)
- Identify noise in real-world text samples
- Compare structured vs unstructured text

Key insight:
> Language is symbolic, contextual, and lossy when digitized.

---

## PHASE 1 — Text Preprocessing

**Goal:** Prepare text for computation

Teach:
- Lowercasing
- Punctuation handling
- Stopwords
- Stemming vs lemmatization
- Normalization tradeoffs

Katas:
- Apply preprocessing pipeline to raw text
- Compare stemming vs lemmatization output
- Measure how stopword removal changes document similarity

Key insight:
> Preprocessing decisions encode assumptions about meaning.

---

## PHASE 2 — Bag of Words (BoW)

**Goal:** First numerical representation of text

Teach:
- Vocabulary construction
- Document-term matrix
- Sparsity
- Word order loss

Katas:
- Build BoW from scratch
- Visualize document vectors
- Compare documents using BoW

Key insight:
> BoW ignores order but captures presence.

---

## PHASE 3 — TF-IDF

**Goal:** Weight words by importance

Teach:
- Term frequency
- Inverse document frequency
- Why rare words matter more
- TF-IDF vs raw counts

Katas:
- Compute TF-IDF manually
- Compare similarity using BoW vs TF-IDF
- Visualize word importance

Key insight:
> TF-IDF is a relevance heuristic, not “understanding”.

---

## PHASE 4 — Similarity & Classical NLP Tasks

**Goal:** Use representations for tasks

Teach:
- Cosine similarity
- Text search
- Document clustering
- Simple text classification

Katas:
- Compute cosine similarity between document pairs
- Build a simple text search engine
- Cluster documents by topic

Datasets:
- News articles
- Movie reviews
- Support tickets

Key insight:
> Many NLP problems reduce to similarity in vector space.

---

## PHASE 5 — Tokenization (Deep Dive)

**Goal:** Break text into model-friendly units

Teach:
- Word tokenization
- Character tokenization
- Subword tokenization
- Byte Pair Encoding (BPE)
- Why tokenization matters for models

Katas:
- Tokenize text using word, character, and subword methods
- Implement BPE from scratch
- Compare vocabulary sizes and OOV handling across methods

Key insight:
> Tokenization defines what a model can and cannot see.

---

## PHASE 6 — Named Entity Recognition (NER)

**Goal:** Structured information from text

Teach:
- What entities are
- Rule-based NER
- Statistical sequence labeling (high-level)
- Evaluation metrics (precision, recall, F1)

Katas:
- Rule-based NER
- Simple ML-based NER
- Error analysis

Key insight:
> NER is about boundaries and labels, not “understanding”.

---

## PHASE 7 — Small Neural Text Models

**Goal:** Neural representations of language

Teach:
- Word embeddings (intuition)
- Simple feedforward text classifiers
- Sequence models overview (RNN intuition)
- Why dense representations help

Katas:
- Train small embedding-based models
- Visualize embedding spaces
- Compare neural vs TF-IDF models

Key insight:
> Neural models learn representations, not rules.

---

## PHASE 8 — Context & Sequence Modeling

**Goal:** Meaning depends on context

Teach:
- Word order importance
- Limitations of BoW/TF-IDF
- RNNs and their issues (overview)
- Motivation for attention

Katas:
- Show how word order changes meaning (BoW failure cases)
- Compare context-aware vs context-free representations
- Demonstrate vanishing gradient problem in sequences (conceptual)

Key insight:
> Context changes meaning, and models must account for it.

---

## PHASE 9 — Transformer Architecture (Core Concepts)

**Goal:** Understand transformers, not just use them

Teach:
- Encoder vs decoder
- Self-attention intuition
- Positional encoding
- Parallelism advantage

Katas:
- Visualize attention weights
- Build a tiny transformer block
- Compare encoder-only vs decoder-only tasks

Key insight:
> Transformers replace recurrence with attention.

---

## PHASE 10 — Modern NLP Pipelines (Awareness)

**Goal:** Connect foundations to today’s models

Teach:
- Pretraining vs fine-tuning
- Encoder-only models (BERT-style)
- Decoder-only models (GPT-style)
- Where LLMs fit in the NLP stack

Rule:
- Do NOT hype
- Do NOT anthropomorphize models

Key insight:
> Modern NLP stands on decades of classical foundations.

---

## Kata Structure (MANDATORY)

Each NLP kata must include:

1. **Concept & Intuition**
   - What problem is being solved
   - Why earlier methods fail
   - Clear mental models

2. **Interactive Experiment**
   - Editable text input
   - Live vector/attention visualizations
   - Similarity and model output inspection

3. **Live Code**
   - Editable code
   - Reset to original
   - Save versions for logged-in users

---

## Teaching Rules (VERY IMPORTANT)

You must:
- Explain *why a technique exists*
- Show limitations explicitly
- Compare classical vs modern approaches
- Emphasize representation over algorithms

You must NOT:
- Jump straight to transformers
- Treat TF-IDF/BoW as obsolete
- Pretend models “understand language”
- Skip error analysis

---

## Success Criteria

This system is successful if learners can:
- Explain how text becomes numbers
- Choose BoW vs TF-IDF appropriately
- Understand tokenization tradeoffs
- Explain why transformers were invented
- Read modern NLP papers without fear

---

## Final Instruction

Teach NLP as a **representation and modeling discipline**, not a shortcut to LLMs.

When in doubt:
- Choose linguistic intuition over hype
- Choose foundations over shortcuts
- Choose understanding over scale

Proceed deliberately.  
Explain everything.  
Never assume.