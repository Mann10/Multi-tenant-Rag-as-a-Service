import json
from typing import Dict, Any
from langchain_core.runnables import RunnableLambda
from langsmith import traceable
import re

from .llm_client import validation_llm, answer_llm
from .vector_client import get_query_embeddings, query_pinecone
import json
from langchain_core.prompts import ChatPromptTemplate

from pydantic import BaseModel, Field
from typing import Literal

class ValidationResult(BaseModel):
    status: Literal["VALIDATED", "NEEDS_CLARIFICATION"]
    reason: str = Field(description="Brief explanation of the decision")
    clarification_question: str = Field(
        default="",
        description="Follow-up question if vague, else empty string"
    )

VALIDATION_SYSTEM_PROMPT = """You are a query validation agent for a RAG system.
Your job is to determine if a user query is sufficient to retrieve relevant documents.

A query is VALIDATED if it:
- Contains specific entities, names, dates, or clear subject matter
- Is not empty, too short (under 3 words), or purely conversational
- Has clear intent that can be matched against document chunks

A query is NEEDS_CLARIFICATION if it:
- Is ambiguous, overly broad, or lacks specifics
- Is just a greeting, a single word, or an unclear question
- Cannot be meaningfully matched to document embeddings

Always respond with a JSON object with keys:
- status: exactly "VALIDATED" or "NEEDS_CLARIFICATION"
- reason: string
- clarification_question: string (empty if VALIDATED)"""

def sanitize_query(query: str) -> str:
    query = re.sub(r'https?://\S+', '', query)
    return ' '.join(query.split()).strip()

structured_validation_llm = validation_llm.with_structured_output(
    ValidationResult,
    method="json_mode"
)

@traceable(run_type="chain", name="validation_agent")
async def validation_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    chat_history = state.get("chat_history", [])
    current_query = sanitize_query(state["current_query"])

    # ✅ Build prompt dynamically so chat_history is injected per call
    prompt = ChatPromptTemplate.from_messages([
        ("system", VALIDATION_SYSTEM_PROMPT),
        *chat_history,
        ("human", "{query}")
    ])

    # ✅ Build chain with structured output
    validation_chain = prompt | structured_validation_llm

    try:
        result: ValidationResult = await validation_chain.ainvoke({"query": current_query})
    except Exception as e:
        print(f"[Validation] Structured output failed: {e}")
        result = ValidationResult(
            status="VALIDATED",
            reason="Fallback: could not parse structured output.",
            clarification_question=""
        )

    state["status"] = result.status.lower()
    state["validation_reason"] = result.reason
    state["chat_history"] = chat_history + [
    ("human", current_query),   # ← append after sanitization
    ]
    if result.status == "NEEDS_CLARIFICATION":
        state["chat_history"] = chat_history + [
            ("assistant", result.clarification_question)
        ]
        state["clarification_question"] = result.clarification_question

    return state


RETRIEVAL_PROMPT = """You are a helpful assistant. Answer the user's question based ONLY on the provided context.

If the context does not contain enough information, say so clearly.

Context:
{context}

User Question: {query}

Provide a clear, accurate answer in markdown format. Cite your sources using [Source: filename, page X] where applicable.

Your Answer:"""


@traceable(run_type="chain", name="retrieval_agent")
async def retrieval_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    state["status"] = "retrieving"

    embeddings = await get_query_embeddings([state["current_query"]])
    query_vector = embeddings[0]

    chunks = await query_pinecone(
        state["tenant_id"],
        query_vector,
        top_k=5
    )
    state["chunks"] = chunks

    if not chunks:
        state["status"] = "completed"
        state["answer"] = "I could not find any relevant documents for your query."
        state["sources"] = []
        return state

    context = "\n\n---\n\n".join([
        f"[Source: {c['source']}, page {c['page']}]\n{c['text']}" 
        for c in chunks
    ])

    prompt = RETRIEVAL_PROMPT.format(context=context, query=state["current_query"])
    response = await answer_llm.ainvoke(prompt)

    state["status"] = "completed"
    state["answer"] = response.content.strip()
    state["sources"] = list(set(c["source"] for c in chunks))

    return state