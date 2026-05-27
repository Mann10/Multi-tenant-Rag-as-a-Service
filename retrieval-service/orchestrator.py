from typing import Dict, Any
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langsmith import traceable

from .agents import validation_agent, retrieval_agent


@traceable(run_type="chain", name="rag_orchestrator")
def build_orchestrator():
    branch = RunnableBranch(
        (lambda s: s.get("status") == "validated", RunnableLambda(retrieval_agent)),
        (lambda s: s.get("status") == "needs_clarification", RunnableLambda(lambda s: s)),
        RunnableLambda(lambda s: {**s, "status": "failed", "error_message": "Unknown routing state"})
    )

    pipeline = (
        RunnableLambda(validation_agent)
        | branch
    )

    return pipeline


orchestrator = build_orchestrator()