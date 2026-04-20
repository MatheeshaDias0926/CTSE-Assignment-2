"""
Optimization Strategist Agent
===============================
Worker agent that formulates actionable improvement plans based on
bottleneck analysis and writes the final optimisation report.
"""

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.prebuilt import ToolNode

from config import OLLAMA_MODEL, OLLAMA_BASE_URL, TEMPERATURE
from logger import get_logger, log_agent_action
from prompts.system_prompts import OPTIMIZATION_STRATEGIST_SYSTEM_PROMPT
from state.global_state import FactoryState
from tools.report_tools import write_optimization_report

_logger = get_logger("OptimizationStrategist")

# Tools available to this agent
OPTIMIZATION_TOOLS = [write_optimization_report]


def _get_llm() -> ChatOllama:
    """Instantiate the Ollama LLM with report tools bound."""
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=TEMPERATURE,
    )
    return llm.bind_tools(OPTIMIZATION_TOOLS)


def optimization_strategist_node(state: FactoryState) -> dict[str, Any]:
    """
    Optimization Strategist worker node.

    Reviews bottleneck findings and formulates an actionable improvement plan.
    Writes the final report using the write_optimization_report tool.

    Args:
        state: The current global FactoryState.

    Returns:
        Updated state with optimization_plan, report_path,
        completed_tasks, agent_trace, and messages.
    """
    log_agent_action(_logger, "OptimizationStrategist", "started", {})

    llm_with_tools = _get_llm()
    trace = list(state.get("agent_trace", []))
    completed = list(state.get("completed_tasks", []))
    bottleneck_findings = state.get("bottleneck_findings", "No findings available.")
    ptp_analysis = state.get("ptp_analysis", {})

    # ── Step 1: Generate the optimisation plan via LLM ──────────────────
    messages = [
        SystemMessage(content=OPTIMIZATION_STRATEGIST_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Based on the following bottleneck analysis, create a detailed "
            f"optimization plan for the automotive production facility.\n\n"
            f"BOTTLENECK FINDINGS:\n{bottleneck_findings[:3000]}\n\n"
            f"Structure your plan with:\n"
            f"1. Executive Summary\n"
            f"2. Critical Issues\n"
            f"3. Recommended Actions (numbered, specific)\n"
            f"4. Resource Requirements\n"
            f"5. Implementation Timeline\n\n"
            f"After creating the plan, use the write_optimization_report tool "
            f"to save it."
        )),
    ]

    response = llm_with_tools.invoke(messages)
    optimization_plan = response.content.strip()

    log_agent_action(
        _logger, "OptimizationStrategist", "plan_generated",
        {"plan_length": len(optimization_plan)},
    )

    # ── Step 2: Write the report (direct tool call as fallback) ─────────
    # Determine priority from bottleneck findings
    priority = _determine_priority(bottleneck_findings)

    report_result = write_optimization_report.invoke({
        "title": "Automotive Production Line Optimization Report",
        "content": optimization_plan,
        "priority": priority,
    })

    # Extract report path from tool result
    report_path = ""
    if "saved successfully to:" in report_result:
        report_path = report_result.split("saved successfully to:")[-1].strip()

    log_agent_action(
        _logger, "OptimizationStrategist", "report_written",
        {"report_path": report_path, "priority": priority},
    )

    completed.append("optimization")

    trace_entry = log_agent_action(
        _logger, "OptimizationStrategist", "completed",
        {"priority": priority, "report_path": report_path},
    )

    return {
        "optimization_plan": optimization_plan,
        "report_path": report_path,
        "completed_tasks": completed,
        "agent_trace": trace + [trace_entry],
        "messages": [
            AIMessage(content=(
                f"[OptimizationStrategist] Plan generated (Priority: {priority}).\n"
                f"Report saved to: {report_path}\n\n"
                f"Optimization Plan:\n{optimization_plan}"
            ))
        ],
    }


# ── Helpers ─────────────────────────────────────────────────────────────────

def _determine_priority(findings: str) -> str:
    """
    Determine report priority based on bottleneck findings text.

    Args:
        findings: The bottleneck analysis text to scan.

    Returns:
        Priority string: 'CRITICAL', 'HIGH', 'MEDIUM', or 'LOW'.
    """
    findings_lower = findings.lower()
    if "critical" in findings_lower:
        return "CRITICAL"
    elif "warning" in findings_lower:
        return "HIGH"
    elif "on_target" in findings_lower or "on target" in findings_lower:
        return "MEDIUM"
    return "MEDIUM"
