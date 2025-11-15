from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import Optional
import time
from app.agents.retriever_agent import RetrieverAgent
from app.agents.reasoning_agent import ReasoningAgent
from app.agents.governance_agent import GovernanceAgent
from app.utils.logger import get_logger
from app.utils.memory import memory_store
from app.utils.status_tracker import status_registry

logger = get_logger("gateway", "logs/gateway.log")

app = FastAPI(title="Policy-Aware RAG")

retriever = RetrieverAgent()
reasoner = ReasoningAgent()
governor = GovernanceAgent()

# Initialize status trackers
gateway_tracker = status_registry.get_tracker("gateway")
retriever_tracker = status_registry.get_tracker("retriever")
reasoner_tracker = status_registry.get_tracker("reasoner")
governance_tracker = status_registry.get_tracker("governance")

# Record initial activity
gateway_tracker.record_activity(latency_ms=0.0)
retriever_tracker.record_activity(latency_ms=0.0)
reasoner_tracker.record_activity(latency_ms=0.0)
governance_tracker.record_activity(latency_ms=0.0)

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

@app.get("/health")
async def health_check():
    """Get summary health status for all agents."""
    gateway_tracker.record_activity(latency_ms=0.0)
    
    # Quick checks
    retriever_ok = retriever and retriever.index is not None
    try:
        # Lightweight ping check
        reasoner_ok = reasoner is not None
    except:
        reasoner_ok = False
    
    all_status = status_registry.get_all_status()
    
    # Minimal JSON response
    return {
        "status": "ok" if retriever_ok and reasoner_ok else "degraded",
        "agents": all_status
    }

@app.get("/health/{agent}")
async def health_agent(agent: str):
    """Get detailed health status for a specific agent."""
    gateway_tracker.record_activity(latency_ms=0.0)
    
    detailed = status_registry.get_agent_status(agent.lower())
    if not detailed:
        raise HTTPException(status_code=404, detail=f"Agent '{agent}' not found")
    
    return detailed

@app.post("/query")
async def query(req: QueryRequest, session_id: Optional[str] = Header(default="default")):
    query_start = time.time()
    q = req.query
    top_k = req.top_k or 5
    logger.info("Received query: %s [session=%s]", q, session_id)

    # 1) Retrieve
    retrieve_start = time.time()
    try:
        passages = retriever.retrieve(q, top_k=top_k)
        retrieve_latency = (time.time() - retrieve_start) * 1000
        retriever_tracker.record_activity(latency_ms=retrieve_latency, error=False)
        retriever_confidence = 0.0
        if passages:
            try:
                retriever_confidence = max((p.get("score", 0.0) for p in passages), default=0.0)
            except ValueError:
                retriever_confidence = 0.0
    except Exception as e:
        retrieve_latency = (time.time() - retrieve_start) * 1000
        retriever_tracker.record_activity(latency_ms=retrieve_latency, error=True)
        logger.error("Retriever error: %s", e)
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")

    # Memory context to reasoning
    previous_turns = memory_store.get(session_id)
    if previous_turns:
        history_text = "\n".join(
            [f"[MEMORY {i}] Q: {turn['query']} | A: {turn['answer']}" for i, turn in enumerate(previous_turns)]
        )
        # Append memory context to the query
        q = f"Previous context:\n{history_text}\n\nNew Query:\n{req.query}"

    # 2) Reason
    reason_start = time.time()
    try:
        reasoning_result = await reasoner.reason(q, passages)
        reason_latency = (time.time() - reason_start) * 1000
        reasoner_tracker.record_activity(latency_ms=reason_latency, error=False)
        answer = reasoning_result.get("answer", "")
        trace = reasoning_result.get("trace", [])
        confidence = float(reasoning_result.get("confidence", 0.0))
    except Exception as e:
        reason_latency = (time.time() - reason_start) * 1000
        reasoner_tracker.record_activity(latency_ms=reason_latency, error=True)
        logger.error("Reasoning error: %s", e)
        raise HTTPException(status_code=500, detail=f"Reasoning failed: {str(e)}")

    # 3) Govern
    govern_start = time.time()
    try:
        decision = governor.evaluate(
            answer,
            trace,
            confidence,
            retriever_confidence=retriever_confidence,
        )
        govern_latency = (time.time() - govern_start) * 1000
        governance_tracker.record_activity(latency_ms=govern_latency, error=False)
        final_answer = decision.get("redacted_answer", answer)
    except Exception as e:
        govern_latency = (time.time() - govern_start) * 1000
        governance_tracker.record_activity(latency_ms=govern_latency, error=True)
        logger.error("Governance error: %s", e)
        # Don't fail the request, just log
        final_answer = answer

    # 4) Save memory
    memory_store.add(session_id, req.query, final_answer, trace)

    # Record gateway activity
    total_latency = (time.time() - query_start) * 1000
    gateway_tracker.record_activity(latency_ms=total_latency, error=False)

    response = {
        "query": req.query,
        "answer": final_answer,
        "governance": {"approved": decision["approved"], "reason": decision["reason"]},
        "trace": trace,
        "retrieved": passages,
        "confidence": confidence,
        "session_id": session_id
    }
    return response

@app.get("/trace")
async def get_trace(session_id: Optional[str] = Header(default="default")):
    history = memory_store.get(session_id)
    if not history:
        raise HTTPException(status_code=404, detail="No trace found for this session")
    return {
        "session_id": session_id,
        "turns": history
    }

@app.delete("/memory/clear")
def clear_memory(session_id: Optional[str] = Header(default="default")):
    history = memory_store.get(session_id)
    if not history:
        raise HTTPException(status_code=404, detail="This session not found")
    memory_store.clear(session_id)
    return {"message": f"Memory cleared for session {session_id}"}