import os
import subprocess
import time
from pathlib import Path

from app.config import settings
from app.models.execution_result import ExecutionResult

VIZ_HELPER_DIR = str(Path(__file__).parent)


async def execute_code(code: str, kata_id: str) -> ExecutionResult:
    # Prepend helper import so show_html/show_chart/plt.show work automatically
    full_code = "from nlp_katas_viz import *\n\n" + code

    env = os.environ.copy()
    existing_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = VIZ_HELPER_DIR + (":" + existing_path if existing_path else "")

    start = time.perf_counter()
    try:
        result = subprocess.run(
            ["python", "-c", full_code],
            capture_output=True,
            text=True,
            timeout=settings.sandbox_timeout_seconds,
            shell=False,
            env=env,
        )
        elapsed = (time.perf_counter() - start) * 1000
        return ExecutionResult(
            stdout=result.stdout,
            stderr=result.stderr,
            error=result.stderr if result.returncode != 0 else None,
            execution_time_ms=round(elapsed, 2),
        )
    except subprocess.TimeoutExpired:
        elapsed = (time.perf_counter() - start) * 1000
        return ExecutionResult(
            error=f"Execution timed out after {settings.sandbox_timeout_seconds} seconds. "
            "Your code took too long to run.",
            execution_time_ms=round(elapsed, 2),
        )
    except Exception as e:
        elapsed = (time.perf_counter() - start) * 1000
        return ExecutionResult(
            error=f"Execution failed: {str(e)}",
            execution_time_ms=round(elapsed, 2),
        )
