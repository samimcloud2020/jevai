from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from jev_utils import guard_tool_call, review_completion

# --- Typed State ---

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    goal: Optional[str]
    tool_outputs: List[Dict[str, Any]]

# --- Tools Definitions & Logic ---

TOOL_POLICIES = {
    "checkout": {
        "policy": ["Payment captured upon order creation."],
        "side_effects": ["Charges customer"],
    },
    "issue_refund": {
        "policy": ["Refunds above $500 require approval."],
        "side_effects": ["Moves funds"],
    },
}

def get_catalog_info(query: str) -> Dict[str, Any]:
    return {"result": f"Catalog info for: {query}"}

def add_to_cart(user_id: str, product_id: str, qty: int) -> Dict[str, Any]:
    return {"status": "ok", "user_id": user_id, "product_id": product_id, "qty": qty}

def checkout(user_id: str, cart_id: str, address: Dict[str, str], **_) -> Dict[str, Any]:
    return {
        "status": "ok",
        "order_id": "ord_12345",
        "user_id": cart_id,
        "items": [{"product_id": "prod_777", "qty": 2}],
        "ship_to": address,
    }

def issue_refund(order_id: str, amount_usd: float, reason: str) -> Dict[str, Any]:
    return {"status": "ok", "refund_id": "ref_999", "order_id": order_id, "amount_usd": amount_usd}

TOOLS = {
    "get_catalog_info": get_catalog_info,
    "add_to_cart": add_to_cart,
    "checkout": checkout,
    "issue_refund": issue_refund,
}

TOOL_SCHEMAS = [
    {
        "name": "get_catalog_info",
        "description": "Get product/catalog information.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "add_to_cart",
        "description": "Add product to cart.",
        "parameters": {
            "type": "object",
            "properties": {"user_id": {"type": "string"}, "product_id": {"type": "string"}, "qty": {"type": "integer"}},
            "required": ["user_id", "product_id", "qty"],
        },
    },
    {
        "name": "checkout",
        "description": "Place an order.",
        "parameters": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "cart_id": {"type": "string"},
                "address": {
                    "type": "object",
                    "properties": {"line1": {"type": "string"}, "city": {"type": "string"}, "state": {"type": "string"}, "country": {"type": "string"}, "pin": {"type": "string"}},
                    "required": ["line1", "city", "state", "country", "pin"],
                },
            },
            "required": ["user_id", "cart_id", "address"],
        },
    },
    {
        "name": "issue_refund",
        "description": "Issue order refund.",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}, "amount_usd": {"type": "number"}, "reason": {"type": "string"}},
            "required": ["order_id", "amount_usd", "reason"],
        },
    },
]

llm = ChatOpenAI(model="gpt-4.1", temperature=0).bind_tools(TOOL_SCHEMAS)

# --- Graph Nodes ---

def agent_node(state: AgentState) -> Dict[str, Any]:
    sys_msg = SystemMessage(
        content=f"You are an e-commerce assistant. Target Goal: {state.get('goal') or 'N/A'}"
    )
    response = llm.invoke([sys_msg] + state["messages"])
    return {"messages": [response]}

def tools_node(state: AgentState) -> Dict[str, Any]:
    last_msg = state["messages"][-1]
    tool_messages = []
    tool_outputs = list(state.get("tool_outputs", []))

    for tc in getattr(last_msg, "tool_calls", []):
        name, args, call_id = tc["name"], tc["args"], tc["id"]
        
        # Guard check
        if name in TOOL_POLICIES:
            config = TOOL_POLICIES[name]
            guard = guard_tool_call(name, args, config["policy"], config["side_effects"])
            
            is_blocked = (
                guard["decision"] == "deny"
                or (name == "issue_refund" and args.get("amount_usd", 0) > 500)
            )
            
            if is_blocked:
                msg = f"Action `{name}` blocked by policy or high-risk evaluation."
                tool_messages.append(ToolMessage(content=msg, tool_call_id=call_id))
                tool_outputs.append({
                    "status": "blocked",
                    "tool": name,
                    "result": {"error": msg}
                })
                continue

        # Execute safe tool
        res = TOOLS[name](**args)
        tool_messages.append(ToolMessage(content=f"Result: {res}", tool_call_id=call_id))
        
        # Append structured tool execution record
        tool_outputs.append({
            "status": res.get("status", "ok"),
            "tool": name,
            "result": res
        })

    return {"messages": tool_messages, "tool_outputs": tool_outputs}

def completion_check_node(state: AgentState) -> Dict[str, Any]:
    # Pass-through node; does not pollute message history
    return {}

# --- Builder ---

def build_agent_graph():
    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tools_node)
    builder.add_node("completion_check", completion_check_node)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        lambda state: "tools" if getattr(state["messages"][-1], "tool_calls", None) else "completion_check",
    )
    builder.add_edge("tools", "agent")
    builder.add_edge("completion_check", END)

    return builder.compile()
