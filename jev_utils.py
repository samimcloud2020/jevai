import os
from typing import Any, Dict, List, Optional
from langchain_typesafe import Noul, Score, Choice, TypeSafeClassifier

TYPESAFE_API_KEY = os.getenv("TYPESAFE_API_KEY")
if not TYPESAFE_API_KEY:
    raise RuntimeError("TYPESAFE_API_KEY not set")

classifier = TypeSafeClassifier()  # uses jev-latest by default


def guard_tool_call(
    tool: str,
    action: str,
    arguments_summary: List[str],
    side_effects: List[str],
    safeguards: List[str],
    policy: List[str],
    reversibility: str = "partially_reversible",
) -> Dict[str, Any]:
    state = {
        "tool": tool,
        "action": action,
        "arguments_summary": arguments_summary,
        "side_effects": side_effects,
        "safeguards": safeguards,
        "policy": policy,
        "reversibility": reversibility,
    }

    questions = {
        "decision": Choice(
            instructions="Choose the safest next action for this tool call.",
            criteria={
                "allow": "Execute the tool immediately.",
                "confirm": "Proceed only after explicit user confirmation.",
                "review": "Require human review before executing.",
                "deny": "Do not execute the tool.",
            },
        ),
        "needs_human_review": Noul(
            instructions="Does this tool call require human review under the stated policy?"
        ),
        "risk": Score(
            instructions="Score the financial and policy risk of executing this tool.",
            criteria=["Low", "Moderate", "High", "Critical"],
        ),
    }

    response = classifier.invoke({"state": state, "questions": questions})

    # Inspect what's available if unsure:
    # score_ans = response.scores["risk"]
    # print(dir(score_ans))

    score_ans = response.scores["risk"]

    decision_data = {
        "decision": response.choices["decision"].choice,
        "confidence": response.choices["decision"].confidence,
        "probabilities": response.choices["decision"].probabilities,
        "needs_human_review": response.nouls["needs_human_review"].noul,
        "risk_score": score_ans.score,
        # Remove .distribution to avoid AttributeError
        "risk_confidence": getattr(score_ans, "confidence", None),
        "guidance": "",
    }
    return decision_data


def review_completion(state_text: str, goal_text: str) -> Dict[str, Any]:
    questions = {
        "complete": Choice(
            instructions="Is the stated goal fully achieved given this state?",
            criteria={
                "complete": "Goal is fully achieved; no further steps needed.",
                "verify_more": "Need more verification before declaring completion.",
                "incomplete": "Goal is not yet achieved; further actions required.",
            },
        ),
        "confidence": Noul(
            instructions="How confident are you that the current assessment is correct?"
        ),
    }

    response = classifier.invoke(
        {
            "state": {"state_summary": state_text, "goal": goal_text},
            "questions": questions,
        }
    )

    return {
        "status": response.choices["complete"].choice,
        "confidence": response.choices["complete"].confidence,
        "noul_confidence": response.nouls["confidence"].noul,
    }
