from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel
from agent_graph import build_agent_graph

load_dotenv(override=True)

app = FastAPI()
agent = build_agent_graph()

class ChatRequest(BaseModel):
    user_id: str
    message: str
    goal: Optional[str] = None
    history: Optional[List[Dict[str, str]]] = None

class StateSnapshot(BaseModel):
    tool_outputs: List[Dict[str, Any]]
    goal: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    state_snapshot: StateSnapshot

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    messages = []
    
    if req.history:
        for m in req.history:
            role, content = m.get("role"), m.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=req.message))

    try:
        final_state = agent.invoke({
            "messages": messages,
            "goal": req.goal,
            "tool_outputs": [],
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Get the last AI message with non-empty content
    reply_msg = next(
        (m.content for m in reversed(final_state["messages"]) if isinstance(m, AIMessage) and m.content),
        ""
    )

    return ChatResponse(
        reply=str(reply_msg),
        state_snapshot=StateSnapshot(
            tool_outputs=final_state.get("tool_outputs", []),
            goal=final_state.get("goal")
        )
    )
