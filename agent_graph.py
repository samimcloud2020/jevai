from typing import Annotated, Dict, List, Any, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from jev_utils import guard_tool_call, review_completion

# ---------- State ----------

class AgentState(dict):
    messages: Annotated[List[BaseMessage], add_messages]
    tool_calls: List[Dict[str, Any]]
    tool_outputs: List[Dict[str, Any]]
    last_decision: Optional[Dict[str, Any]]
    goal: Optional[str]


# ---------- LLM ----------

llm = ChatOpenAI(model="gpt-4.1", temperature=0)  # or your preferred model


# ---------- Tools (stubs) ----------

def get_catalog_info(query: str) -> Dict[str, Any]:
    return {"result": f"Catalog info for: {query}"}


def add_to_cart(user_id: str, product_id: str, qty: int) -> Dict[str, Any]:
    return {"status": "ok", "user_id": user_id, "product_id": product_id, "qty": qty}


def checkout(user_id: str, cart_id: str, address: Dict[str, str]) -> Dict[str, Any]:
    return {
        "status": "ok",
        "order_id": "ord_12345",
        "user_id": user_id,
        "items": [{"product_id": "prod_777", "qty": 2}],
        "ship_to": address,
    }


def issue_refund(order_id: str, amount_usd: float, reason: str) -> Dict[str, Any]:
    return {
        "status": "ok",
        "refund_id": "ref_999",
        "order_id": order_id,
        "amount_usd": amount_usd,
    }


TOOLS = {
    "get_catalog_info": get_catalog_info,
    "add_to_cart": add_to_cart,
    "checkout": checkout,
    "issue_refund": issue_refund,
}

TOOL_DEFINITIONS = [
    {
        "name": "get_catalog_info",
        "description": "Get product/catalog information.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "User query about products"}
            },
            "required": ["query"],
        },
    },
    {
        "name": "add_to_cart",
        "description": "Add a product to the user's cart.",
        "parameters": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "product_id": {"type": "string"},
                "qty": {"type": "integer"},
            },
            "required": ["user_id", "product_id", "qty"],
        },
    },
    {
        "name": "checkout",
        "description": "Place an order from the user's cart.",
        "parameters": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "cart_id": {"type": "string"},
                "address": {
                    "type": "object",
                    "properties": {
                        "line1": {"type": "string"},
                        "city": {"type": "string"},
                        "state": {"type": "string"},
                        "country": {"type": "string"},
                        "pin": {"type": "string"},
                    },
                    "required": ["line1", "city", "state", "country", "pin"],
                },
            },
            "required": ["user_id", "cart_id", "address"],
        },
    },
    {
        "name": "issue_refund",
        "description": "Issue a refund for an order.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "amount_usd": {"type": "number"},
                "reason": {"type": "string"},
            },
            "required": ["order_id", "amount_usd", "reason"],
        },
    },
]

llm_with_tools = llm.bind_tools(TOOL_DEFINITIONS)


# ---------- Nodes ----------

def agent_node(state: AgentState) -> AgentState:
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {
        "messages": [response],
        "tool_calls": getattr(response, "tool_calls", []) or [],
    }


def tools_node(state: AgentState) -> AgentState:
    tool_calls = state.get("tool_calls", [])
    outputs = []
    tool_messages = []

    for tc in tool_calls:
        name = tc["name"]
        args = tc["args"]
        tool_call_id = tc["id"]
        tool_fn = TOOLS[name]

        # Decide if we need a Jev guard
        needs_guard = name in ("checkout", "issue_refund")

        if needs_guard:
            action_summary = f"{name} with args: {args}"

            if name == "checkout":
                policy = ["Payment is captured; order is created."]
                safeguards = ["Address validated; payment method on file."]
                side_effects = ["Charges customer", "Creates order record"]
                reversibility = "partially_reversible"
            elif name == "issue_refund":
                policy = ["Refunds above USD 500 require human approval."]
                safeguards = ["Order ownership and issue reason verified."]
                side_effects = ["Moves funds", "Changes order payment state"]
                reversibility = "partially_reversible"
            else:
                policy = []
                safeguards = []
                side_effects = []
                reversibility = "fully_reversible"

            decision = guard_tool_call(
                tool=name,
                action=action_summary,
                arguments_summary=[str(args)],
                side_effects=side_effects,
                safeguards=safeguards,
                policy=policy,
                reversibility=reversibility,
            )

            # Dev mode: only block on explicit "deny"
            if decision["decision"] == "deny":
                msg_text = (
                    f"I cannot perform `{name}` automatically. "
                    f"Decision: {decision['decision']}, confidence: {decision['confidence']:.2f}. "
                    f"Reason: {decision.get('guidance') or 'Policy/risk constraints.'}"
                )
                outputs.append({"status": "blocked_by_policy", "tool": name, "message": msg_text})
                tool_messages.append(
                    ToolMessage(content=msg_text, tool_call_id=tool_call_id)
                )
                continue

        # Execute tool
        result = tool_fn(**args)
        outputs.append({"status": "ok", "tool": name, "result": result})

        # Create a proper ToolMessage
        tool_messages.append(
            ToolMessage(content=f"Tool {name} result: {result}", tool_call_id=tool_call_id)
        )

    return {
        "messages": tool_messages,
        "tool_outputs": outputs,
        "last_decision": None,
    }


def completion_check_node(state: AgentState) -> AgentState:
    goal = state.get("goal")
    if not goal:
        return {}

    msgs = state.get("messages", [])
    state_summary = "\n".join(
        [f"{m.type}: {m.content}" for m in msgs[-6:] if hasattr(m, "content") and m.content]
    )

    result = review_completion(state_summary, goal)

    if result["status"] != "complete" or result["confidence"] < 0.5:
        msg = AIMessage(
            content=(
                "I'm not fully sure this task is complete yet. "
                f"Jev assessment: {result['status']}, confidence: {result['confidence']:.2f}. "
                "Do you want me to continue or is this good enough?"
            )
        )
        return {"messages": [msg]}

    return {}


# ---------- Graph ----------

def build_agent_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_node("completion_check", completion_check_node)

    graph.add_edge(START, "agent")

    def route_after_agent(state: AgentState):
        if state.get("tool_calls"):
            return "tools"
        return "completion_check"

    graph.add_conditional_edges("agent", route_after_agent)
    graph.add_edge("tools", "agent")
    graph.add_edge("completion_check", END)

    return graph.compile()
