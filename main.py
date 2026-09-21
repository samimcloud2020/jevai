import os
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from agent_graph import build_agent_graph
from dotenv import load_dotenv
load_dotenv(override=True)

app = FastAPI()

agent = build_agent_graph()


class ChatRequest(BaseModel):
    user_id: str
    message: str
    goal: Optional[str] = None  # e.g. "Place an order for product X"
    history: Optional[List[Dict[str, str]]] = None  # [{role: "user"|"assistant", content: "..."}]


class ChatResponse(BaseModel):
    reply: str
    state_snapshot: Dict[str, Any]


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    # Build messages
    messages: List[HumanMessage | AIMessage] = []
    if req.history:
        for m in req.history:
            role = m.get("role", "")
            content = m.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                from langchain_core.messages import AIMessage
                messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=req.message))

    initial_state = {
        "messages": messages,
        "tool_calls": [],
        "tool_outputs": [],
        "last_decision": None,
        "goal": req.goal,
    }

    try:
        final_state = agent.invoke(initial_state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Last assistant message as reply
    msgs = final_state.get("messages", [])
    reply = ""
    for m in reversed(msgs):
        if hasattr(m, "content") and m.content:
            reply = m.content
            break

    # Strip internal fields for response
    state_snapshot = {
        "tool_outputs": final_state.get("tool_outputs", []),
        "goal": final_state.get("goal"),
    }

    return ChatResponse(reply=reply, state_snapshot=state_snapshot)
