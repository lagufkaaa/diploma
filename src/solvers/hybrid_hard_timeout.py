import multiprocessing as mp
import traceback
from typing import Any, Dict, Optional, Tuple


def _solve_problem_worker(queue: mp.Queue, problem_kwargs: Dict[str, Any]) -> None:
    try:
        from solvers.model_hybrid import Problem as HybridProblem

        problem = HybridProblem(**problem_kwargs)
        result = problem.solve()
        queue.put({"ok": True, "result": result})
    except Exception:
        queue.put({"ok": False, "error": traceback.format_exc()})


def solve_hybrid_problem_with_hard_timeout(
    *,
    problem_kwargs: Dict[str, Any],
    timeout_sec: Optional[float],
) -> Tuple[dict, bool, Optional[str]]:
    """
    Solve HybridProblem in a separate process and hard-stop it by timeout.

    Returns:
      (result_dict, timed_out, error_text)
    """
    timeout = None if timeout_sec is None else max(0.0, float(timeout_sec))
    if timeout is None:
        from solvers.model_hybrid import Problem as HybridProblem

        problem = HybridProblem(**problem_kwargs)
        return problem.solve(), False, None

    queue: Optional[mp.Queue] = None
    try:
        ctx = mp.get_context("spawn")
        queue = ctx.Queue(maxsize=1)
        proc = ctx.Process(
            target=_solve_problem_worker,
            args=(queue, problem_kwargs),
            daemon=True,
        )
        proc.start()
        proc.join(timeout=timeout)

        if proc.is_alive():
            proc.terminate()
            proc.join()
            return {"status": "NOT_SOLVED", "objective_value": None}, True, None

        try:
            payload = queue.get_nowait()
        except Exception:
            return {"status": "NOT_SOLVED", "objective_value": None}, False, (
                "subprocess finished without payload"
            )

        if not isinstance(payload, dict):
            return {"status": "NOT_SOLVED", "objective_value": None}, False, (
                "invalid subprocess payload type"
            )

        if bool(payload.get("ok")):
            result = payload.get("result")
            if isinstance(result, dict):
                return result, False, None
            return {"status": "NOT_SOLVED", "objective_value": None}, False, (
                "invalid subprocess result payload"
            )

        return {"status": "NOT_SOLVED", "objective_value": None}, False, str(
            payload.get("error") or "subprocess failed"
        )
    except Exception:
        # Fallback for environments where multiprocessing is unavailable/restricted.
        try:
            from solvers.model_hybrid import Problem as HybridProblem

            problem = HybridProblem(**problem_kwargs)
            result = problem.solve()
            return result, False, "hard-timeout unavailable, used direct solve"
        except Exception:
            return {"status": "NOT_SOLVED", "objective_value": None}, False, traceback.format_exc()
    finally:
        if queue is not None:
            try:
                queue.close()
            except Exception:
                pass
