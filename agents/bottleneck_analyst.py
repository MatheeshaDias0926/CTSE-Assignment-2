"""
Bottleneck Analyst Agent
=========================
Worker agent that analyses PTP efficiency, checks inventory levels,
and identifies production bottlenecks.
"""

import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.prebuilt import ToolNode

from config import OLLAMA_MODEL, OLLAMA_BASE_URL, TEMPERATURE
from logger import get_logger, log_agent_action
from prompts.system_prompts import BOTTLENECK_ANALYST_SYSTEM_PROMPT
from state.global_state import FactoryState
from tools.production_tools import calculate_ptp_efficiency
from tools.inventory_tools import query_inventory_db

_logger = get_logger("BottleneckAnalyst")

# Tools available to this agent
ANALYSIS_TOOLS = [calculate_ptp_efficiency, query_inventory_db]


def _get_llm() -> ChatOllama:
    """Instantiate the Ollama LLM with analysis tools bound."""
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=TEMPERATURE,
    )
    return llm.bind_tools(ANALYSIS_TOOLS)


def bottleneck_analyst_node(state: FactoryState) -> dict[str, Any]:
    """
    Bottleneck Analyst worker node.

    Uses production data from state to:
    1. Calculate PTP efficiency for production lines.
    2. Query inventory for low-stock materials.
    3. Synthesise bottleneck findings.

    Args:
        state: The current global FactoryState.

    Returns:
        Updated state with ptp_analysis, inventory_status,
        bottleneck_findings, completed_tasks, agent_trace, and messages.
    """
    log_agent_action(_logger, "BottleneckAnalyst", "started", {})

    llm_with_tools = _get_llm()
    trace = list(state.get("agent_trace", []))
    completed = list(state.get("completed_tasks", []))
    production_data = state.get("production_data", {})

    # ── Step 1: Compute PTP efficiency from production data ─────────────
    ptp_results = _compute_ptp_from_data(production_data)
    log_agent_action(
        _logger, "BottleneckAnalyst", "ptp_computed",
        {"lines_analysed": len(ptp_results)},
    )

    # ── Step 2: Query inventory for low stock ───────────────────────────
    inventory_output = query_inventory_db.invoke({
        "query_type": "low_stock",
        "material_name": "",
    })
    try:
        inventory_status = json.loads(inventory_output)
    except (json.JSONDecodeError, TypeError):
        inventory_status = {"raw_output": inventory_output}

    log_agent_action(
        _logger, "BottleneckAnalyst", "inventory_checked",
        {"low_stock_count": inventory_status.get("result_count", "unknown")},
    )

    # ── Step 3: Ask LLM to synthesise findings ──────────────────────────
    ptp_summary = json.dumps(ptp_results, indent=2)
    inv_summary = json.dumps(inventory_status, indent=2, default=str)

    messages = [
        SystemMessage(content=BOTTLENECK_ANALYST_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Here are the PTP efficiency results for each production line:\n"
            f"{ptp_summary[:2500]}\n\n"
            f"Here are the low-stock inventory items:\n"
            f"{inv_summary[:1500]}\n\n"
            "Based on this data, identify the top bottlenecks and efficiency gaps. "
            "Structure your response with: 1) PTP Analysis, 2) Inventory Alerts, "
            "3) Bottleneck Summary with root cause hypotheses."
        )),
    ]

    response = llm_with_tools.invoke(messages)
    bottleneck_findings = response.content.strip()

    completed.append("bottleneck_analysis")

    trace_entry = log_agent_action(
        _logger, "BottleneckAnalyst", "completed",
        {"findings_length": len(bottleneck_findings)},
    )

    return {
        "ptp_analysis": {"line_results": ptp_results},
        "inventory_status": inventory_status,
        "bottleneck_findings": bottleneck_findings,
        "completed_tasks": completed,
        "agent_trace": trace + [trace_entry],
        "messages": [
            AIMessage(content=f"[BottleneckAnalyst] Findings:\n{bottleneck_findings}")
        ],
    }


# ── Helpers ─────────────────────────────────────────────────────────────────

def _compute_ptp_from_data(production_data: dict) -> list[dict]:
    """
    Extract line-level planned vs actual figures from the production data
    and compute PTP efficiency for each line.

    Args:
        production_data: The parsed production data dict from the Data
                         Retrieval Agent.

    Returns:
        A list of PTP result dicts, one per production line.
    """
    results = []
    sample_data = production_data.get("sample_data", [])

    if not sample_data:
        # Fallback: use numeric stats if sample data isn't available
        stats = production_data.get("numeric_stats", {})
        planned_stats = stats.get("planned_units", {})
        actual_stats = stats.get("actual_units", {})
        if planned_stats and actual_stats:
            ptp_output = calculate_ptp_efficiency.invoke({
                "planned": planned_stats.get("mean", 100),
                "actual": actual_stats.get("mean", 80),
            })
            try:
                results.append(json.loads(ptp_output))
            except json.JSONDecodeError:
                results.append({"raw": ptp_output})
        return results

    # Aggregate by line_id
    line_data: dict[str, dict] = {}
    for row in sample_data:
        line_id = str(row.get("line_id", "unknown"))
        if line_id not in line_data:
            line_data[line_id] = {"planned": 0, "actual": 0, "count": 0}
        line_data[line_id]["planned"] += float(row.get("planned_units", 0))
        line_data[line_id]["actual"] += float(row.get("actual_units", 0))
        line_data[line_id]["count"] += 1

    for line_id, data in line_data.items():
        if data["planned"] > 0:
            ptp_output = calculate_ptp_efficiency.invoke({
                "planned": data["planned"],
                "actual": data["actual"],
            })
            try:
                result = json.loads(ptp_output)
                result["line_id"] = line_id
                result["records_analysed"] = data["count"]
                results.append(result)
            except json.JSONDecodeError:
                results.append({
                    "line_id": line_id,
                    "raw": ptp_output,
                })

    return results
