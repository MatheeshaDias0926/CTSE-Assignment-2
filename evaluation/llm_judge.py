"""
LLM-as-a-Judge Evaluation Script
===================================
Automated evaluation of Smart Factory MAS outputs using the same
local Ollama model. Scores the system's output on 5 criteria.

Usage:
    python -m evaluation.llm_judge
    python -m evaluation.llm_judge --report-path outputs/report.md
"""

import argparse
import json
import os
import sys
import glob
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from config import (
    EVAL_MODEL, OLLAMA_BASE_URL, EVAL_TEMPERATURE, OUTPUT_DIR,
)
from logger import get_logger

_logger = get_logger("LLM-Judge")

JUDGE_SYSTEM_PROMPT = """You are an impartial quality evaluator for industrial production reports.

You will receive a production optimization report and must score it on 5 criteria.

For EACH criterion, provide:
1. A score from 1 to 5 (1=poor, 5=excellent)
2. A brief justification (1-2 sentences)

CRITERIA:
1. FACTUAL_GROUNDING: Are all numbers and statistics traceable to source data? No invented figures?
2. ACTIONABLE_SPECIFICITY: Are recommendations concrete and implementable? Not vague?
3. HALLUCINATION_DETECTION: Does the output contain fabricated data or unsupported claims?
4. COMPLETENESS: Does it address production efficiency, bottlenecks, and optimization?
5. LOGICAL_COHERENCE: Is the reasoning chain sound? Do conclusions follow from evidence?

RESPOND ONLY in this exact JSON format:
{
  "scores": {
    "factual_grounding": {"score": <1-5>, "justification": "<text>"},
    "actionable_specificity": {"score": <1-5>, "justification": "<text>"},
    "hallucination_detection": {"score": <1-5>, "justification": "<text>"},
    "completeness": {"score": <1-5>, "justification": "<text>"},
    "logical_coherence": {"score": <1-5>, "justification": "<text>"}
  },
  "overall_score": <average of all scores>,
  "summary": "<2-3 sentence overall assessment>"
}

IMPORTANT: Output ONLY valid JSON. No extra text."""


def evaluate_report(report_content: str) -> dict:
    """
    Evaluate a report using the LLM-as-a-Judge approach.

    Args:
        report_content: The full text of the optimization report.

    Returns:
        Evaluation results dict with scores and justifications.
    """
    _logger.info("Starting LLM-as-a-Judge evaluation...")

    llm = ChatOllama(
        model=EVAL_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=EVAL_TEMPERATURE,
    )

    messages = [
        SystemMessage(content=JUDGE_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Evaluate the following production optimization report:\n\n"
            f"---BEGIN REPORT---\n{report_content[:4000]}\n---END REPORT---\n\n"
            f"Score this report on all 5 criteria. Respond ONLY with JSON."
        )),
    ]

    response = llm.invoke(messages)
    response_text = response.content.strip()

    _logger.info(f"Judge raw response length: {len(response_text)}")

    # Parse the JSON response
    evaluation = _parse_evaluation(response_text)
    return evaluation


def _parse_evaluation(text: str) -> dict:
    """Parse the JSON evaluation from the LLM response."""
    # Try direct JSON parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in the text
    import re
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Fallback: return raw text
    _logger.warning("Could not parse JSON from judge response.")
    return {
        "scores": {},
        "overall_score": 0,
        "summary": f"Parse error. Raw response: {text[:500]}",
        "parse_error": True,
    }


def find_latest_report() -> str:
    """Find the most recently generated report."""
    pattern = os.path.join(OUTPUT_DIR, "optimization_report_*.md")
    reports = glob.glob(pattern)
    if not reports:
        return ""
    return max(reports, key=os.path.getmtime)


def print_evaluation(evaluation: dict) -> None:
    """Pretty-print the evaluation results."""
    print("\n" + "═" * 60)
    print("  📋 LLM-AS-A-JUDGE EVALUATION RESULTS")
    print("═" * 60)

    scores = evaluation.get("scores", {})
    total = 0
    count = 0

    for criterion, data in scores.items():
        if isinstance(data, dict):
            score = data.get("score", "?")
            justification = data.get("justification", "N/A")
            bar = "█" * int(score) + "░" * (5 - int(score)) if isinstance(score, (int, float)) else "?"
            print(f"\n  {criterion.upper()}")
            print(f"    Score: [{bar}] {score}/5")
            print(f"    {justification}")
            if isinstance(score, (int, float)):
                total += score
                count += 1

    overall = evaluation.get("overall_score", total / max(count, 1))
    print(f"\n{'─' * 60}")
    print(f"  OVERALL SCORE: {overall:.1f}/5.0")
    print(f"{'─' * 60}")

    summary = evaluation.get("summary", "")
    if summary:
        print(f"\n  Summary: {summary}")

    print("═" * 60)


def save_evaluation(evaluation: dict, report_path: str) -> str:
    """Save evaluation results to a JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_path = os.path.join(
        OUTPUT_DIR, f"evaluation_{timestamp}.json"
    )
    output = {
        "evaluated_report": report_path,
        "evaluation_timestamp": datetime.now().isoformat(),
        "model_used": EVAL_MODEL,
        "results": evaluation,
    }
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\n💾 Evaluation saved to: {eval_path}")
    return eval_path


def main():
    """CLI entry point for the evaluation script."""
    parser = argparse.ArgumentParser(
        description="LLM-as-a-Judge evaluation for Smart Factory MAS"
    )
    parser.add_argument(
        "--report-path", "-r", default="",
        help="Path to the report to evaluate (defaults to latest).",
    )
    args = parser.parse_args()

    # Find the report
    report_path = args.report_path or find_latest_report()
    if not report_path or not os.path.exists(report_path):
        print("❌ No report found to evaluate.")
        print(f"   Run main.py first or specify --report-path")
        sys.exit(1)

    print(f"📄 Evaluating: {report_path}")

    with open(report_path, "r", encoding="utf-8") as f:
        report_content = f.read()

    evaluation = evaluate_report(report_content)
    print_evaluation(evaluation)
    save_evaluation(evaluation, report_path)


if __name__ == "__main__":
    main()
