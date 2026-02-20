from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse

router = APIRouter()

TRACK_ROOT = Path(__file__).resolve().parents[2]

CONTENT_DIRS = {
    "python-nlp": TRACK_ROOT,
}

PYTHON_NLP_KATAS: list[dict] = [
    # Phase 0 — Language & Text (Foundations)
    {"id": "explore-ambiguity", "title": "Explore Ambiguity in Sentences", "phase": 0, "sequence": 1, "track_id": "python-nlp"},
    {"id": "identify-noise", "title": "Identify Noise in Real-World Text", "phase": 0, "sequence": 2, "track_id": "python-nlp"},
    {"id": "structured-vs-unstructured", "title": "Compare Structured vs Unstructured Text", "phase": 0, "sequence": 3, "track_id": "python-nlp"},
    # Phase 1 — Text Preprocessing
    {"id": "preprocessing-pipeline", "title": "Apply a Preprocessing Pipeline to Raw Text", "phase": 1, "sequence": 1, "track_id": "python-nlp"},
    {"id": "stemming-vs-lemmatization", "title": "Compare Stemming vs Lemmatization Output", "phase": 1, "sequence": 2, "track_id": "python-nlp"},
    {"id": "stopword-removal-similarity", "title": "Measure How Stopword Removal Changes Document Similarity", "phase": 1, "sequence": 3, "track_id": "python-nlp"},
    # Phase 2 — Bag of Words (BoW)
    {"id": "build-bow", "title": "Build Bag of Words from Scratch", "phase": 2, "sequence": 1, "track_id": "python-nlp"},
    {"id": "visualize-document-vectors", "title": "Visualize Document Vectors", "phase": 2, "sequence": 2, "track_id": "python-nlp"},
    {"id": "compare-documents-bow", "title": "Compare Documents Using Bag of Words", "phase": 2, "sequence": 3, "track_id": "python-nlp"},
    # Phase 3 — TF-IDF
    {"id": "compute-tfidf", "title": "Compute TF-IDF Manually", "phase": 3, "sequence": 1, "track_id": "python-nlp"},
    {"id": "bow-vs-tfidf", "title": "Compare Similarity Using BoW vs TF-IDF", "phase": 3, "sequence": 2, "track_id": "python-nlp"},
    {"id": "visualize-word-importance", "title": "Visualize Word Importance", "phase": 3, "sequence": 3, "track_id": "python-nlp"},
    # Phase 4 — Similarity & Classical NLP Tasks
    {"id": "cosine-similarity", "title": "Compute Cosine Similarity Between Document Pairs", "phase": 4, "sequence": 1, "track_id": "python-nlp"},
    {"id": "text-search-engine", "title": "Build a Simple Text Search Engine", "phase": 4, "sequence": 2, "track_id": "python-nlp"},
    {"id": "cluster-documents", "title": "Cluster Documents by Topic", "phase": 4, "sequence": 3, "track_id": "python-nlp"},
    # Phase 5 — Tokenization (Deep Dive)
    {"id": "tokenization-methods", "title": "Tokenize Text Using Word, Character, and Subword Methods", "phase": 5, "sequence": 1, "track_id": "python-nlp"},
    {"id": "byte-pair-encoding", "title": "Implement BPE from Scratch", "phase": 5, "sequence": 2, "track_id": "python-nlp"},
    {"id": "vocabulary-comparison", "title": "Compare Vocabulary Sizes and OOV Handling Across Methods", "phase": 5, "sequence": 3, "track_id": "python-nlp"},
    # Phase 6 — Named Entity Recognition (NER)
    {"id": "rule-based-ner", "title": "Rule-Based Named Entity Recognition", "phase": 6, "sequence": 1, "track_id": "python-nlp"},
    {"id": "statistical-ner", "title": "Simple ML-Based Named Entity Recognition", "phase": 6, "sequence": 2, "track_id": "python-nlp"},
    {"id": "ner-error-analysis", "title": "NER Error Analysis", "phase": 6, "sequence": 3, "track_id": "python-nlp"},
    # Phase 7 — Small Neural Text Models
    {"id": "word-embeddings", "title": "Train Small Embedding-Based Models", "phase": 7, "sequence": 1, "track_id": "python-nlp"},
    {"id": "visualize-embeddings", "title": "Visualize Embedding Spaces", "phase": 7, "sequence": 2, "track_id": "python-nlp"},
    {"id": "neural-vs-tfidf", "title": "Compare Neural vs TF-IDF Models", "phase": 7, "sequence": 3, "track_id": "python-nlp"},
    # Phase 8 — Context & Sequence Modeling
    {"id": "word-order-matters", "title": "Show How Word Order Changes Meaning", "phase": 8, "sequence": 1, "track_id": "python-nlp"},
    {"id": "context-aware-representations", "title": "Compare Context-Aware vs Context-Free Representations", "phase": 8, "sequence": 2, "track_id": "python-nlp"},
    {"id": "sequence-modeling-challenges", "title": "Demonstrate Sequence Modeling Challenges", "phase": 8, "sequence": 3, "track_id": "python-nlp"},
    # Phase 9 — Transformer Architecture (Core Concepts)
    {"id": "attention-weights", "title": "Visualize Attention Weights", "phase": 9, "sequence": 1, "track_id": "python-nlp"},
    {"id": "tiny-transformer-block", "title": "Build a Tiny Transformer Block", "phase": 9, "sequence": 2, "track_id": "python-nlp"},
    {"id": "encoder-vs-decoder", "title": "Compare Encoder-Only vs Decoder-Only Tasks", "phase": 9, "sequence": 3, "track_id": "python-nlp"},
    # Phase 10 — Modern NLP Pipelines (Awareness)
    {"id": "pretraining-vs-finetuning", "title": "Pretraining vs Fine-Tuning", "phase": 10, "sequence": 1, "track_id": "python-nlp"},
    {"id": "encoder-decoder-models", "title": "Encoder-Only vs Decoder-Only Models", "phase": 10, "sequence": 2, "track_id": "python-nlp"},
    {"id": "where-llms-fit", "title": "Where LLMs Fit in the NLP Stack", "phase": 10, "sequence": 3, "track_id": "python-nlp"},
]

PHASE_NAMES = {
    0: "Language & Text (Foundations)",
    1: "Text Preprocessing",
    2: "Bag of Words (BoW)",
    3: "TF-IDF",
    4: "Similarity & Classical NLP Tasks",
    5: "Tokenization (Deep Dive)",
    6: "Named Entity Recognition (NER)",
    7: "Small Neural Text Models",
    8: "Context & Sequence Modeling",
    9: "Transformer Architecture (Core Concepts)",
    10: "Modern NLP Pipelines (Awareness)",
}

TRACK_KATAS = {
    "python-nlp": PYTHON_NLP_KATAS,
}


@router.get("/tracks/{track_id}/katas")
async def list_katas(track_id: str):
    if track_id not in TRACK_KATAS:
        raise HTTPException(status_code=404, detail=f"Track '{track_id}' not found")

    return {
        "katas": TRACK_KATAS[track_id],
        "phases": PHASE_NAMES,
    }


@router.get(
    "/tracks/{track_id}/katas/{phase_id}/{kata_id}/content",
    response_class=PlainTextResponse,
)
async def get_kata_content(track_id: str, phase_id: int, kata_id: str):
    if track_id not in CONTENT_DIRS:
        raise HTTPException(status_code=404, detail=f"Track '{track_id}' not found")

    katas = TRACK_KATAS.get(track_id, [])
    kata = next(
        (k for k in katas if k["id"] == kata_id and k["phase"] == phase_id),
        None,
    )
    if not kata:
        raise HTTPException(
            status_code=404,
            detail=f"Kata '{kata_id}' not found in phase {phase_id}",
        )

    content_dir = CONTENT_DIRS[track_id]
    seq = kata["sequence"]
    filename = f"{seq:02d}-{kata_id}.md"
    filepath = content_dir / f"phase-{phase_id}" / filename

    if not filepath.exists():
        raise HTTPException(
            status_code=404, detail=f"Content file not found: {filename}"
        )

    return filepath.read_text(encoding="utf-8")
