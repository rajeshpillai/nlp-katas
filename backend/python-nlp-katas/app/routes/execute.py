from fastapi import APIRouter

from app.models.execution_result import ExecutionRequest, ExecutionResult
from app.services.sandbox import execute_code

router = APIRouter()


@router.post("/execute", response_model=ExecutionResult)
@router.post("/execute/{track_id}", response_model=ExecutionResult)
async def run_code(request: ExecutionRequest, track_id: str = "python-nlp"):
    return await execute_code(request.code, request.kata_id)
