from pydantic import BaseModel


class ExecutionRequest(BaseModel):
    code: str
    kata_id: str


class ExecutionResult(BaseModel):
    stdout: str = ""
    stderr: str = ""
    error: str | None = None
    execution_time_ms: float = 0.0
