import os
from typing import Any, Dict, List
from langchain_typesafe import Choice, Score, TypeSafeClassifier

if not os.getenv("TYPESAFE_API_KEY"):
    raise RuntimeError("TYPESAFE_API_KEY not set")

classifier = TypeSafeClassifier()

def guard_tool_call(tool: str, args: Dict[str, Any], policy: List[str], side_effects: List[str]) -> Dict[str, Any]:
    questions = {
        "decision": Choice(
            instructions="Choose the safest next action for this tool call.",
            criteria={
                "allow": "Execute immediately.",
                "confirm": "Proceed after user confirmation.",
                "review": "Require human review.",
                "deny": "Do not execute.",
            },
        ),
        "risk": Score(
            instructions="Score the financial and policy risk.",
            criteria=["Low", "Moderate", "High", "Critical"],
        ),
    }
    
    res = classifier.invoke({
        "state": {"tool": tool, "arguments": args, "policy": policy, "side_effects": side_effects},
        "questions": questions,
    })
    
    return {
        "decision": res.choices["decision"].choice,
        "confidence": res.choices["decision"].confidence,
        "risk_score": res.scores["risk"].score,
    }

def review_completion(state_summary: str, goal: str) -> Dict[str, Any]:
    questions = {
        "complete": Choice(
            instructions="Is the stated goal fully achieved?",
            criteria={
                "complete": "Goal fully achieved.",
                "incomplete": "Goal not achieved.",
            },
        )
    }
    
    res = classifier.invoke({
        "state": {"state_summary": state_summary, "goal": goal},
        "questions": questions,
    })
    
    return {
        "status": res.choices["complete"].choice,
        "confidence": res.choices["complete"].confidence,
    }
