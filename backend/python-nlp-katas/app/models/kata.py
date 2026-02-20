from pydantic import BaseModel


class Kata(BaseModel):
    id: str
    title: str
    phase: int
    sequence: int
    track_id: str


class KataList(BaseModel):
    katas: list[Kata]
