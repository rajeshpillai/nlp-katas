from fastapi import APIRouter

router = APIRouter()

TRACKS = [
    {
        "id": "python-nlp",
        "name": "Python NLP",
        "description": "Learn NLP from foundations to transformers using Python. "
        "Build intuition for how text becomes numbers and why each technique exists.",
        "status": "active",
    },
    {
        "id": "rust-nlp",
        "name": "Rust NLP",
        "description": "Learn NLP from foundations to transformers using Rust. "
        "Build intuition for how text becomes numbers with systems-level control.",
        "status": "active",
    },
]


@router.get("/tracks")
async def list_tracks():
    return TRACKS
