"""
Data Retrieval Agent
=====================
Worker agent responsible for reading local CSV production logs
and summarising the data for downstream agents.

Note: Uses direct tool invocation (not bind_tools) because llama3:8b
does not support native tool calling.
"""

import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from config import OLLAMA_MODEL, OLLAMA_BASE_URL, TEMPERATURE
from logger import get_logger, log_agent_action
from prompts.system_prompts import DATA_RETRIEVAL_SYSTEM_PROMPT
from state.global_state import FactoryState
from tools.production_tools import read_production_data

_logger = get_logger("DataRetrievalAgent")


def _get_llm() -> ChatOllama:
    """Instantiate the Ollama LLM (no tool binding — llama3:8b unsupported)."""
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=TEMPERATURE,
    )


def data_retrieval_node(state: FactoryState) -> dict[str, Any]:
    """
    Data Retrieval worker node.

    Directly invokes the read_production_data tool, then asks the LLM
    to summarise the results.

    Args:
        state: The current global FactoryState.

    Returns:
        Updated state with production_data, completed_tasks,
        agent_trace, and messages.
    """
    log_agent_action(_logger, "DataRetrievalAgent", "started", {})

    llm = _get_llm()
    trace = list(state.get("agent_trace", []))
    completed = list(state.get("completed_tasks", []))

    # ── Step 1: Directly call the tool ──────────────────────────────────
    log_agent_action(_logger, "DataRetrievalAgent", "calling_read_production_data", {})
    tool_output = read_production_data.invoke({})

    production_data = {}
    try:
        production_data = json.loads(tool_output)
    except (json.JSONDecodeError, TypeError):
        production_data = {"raw_output": tool_output}

    log_agent_action(
        _logger, "DataRetrievalAgent", "data_retrieved",
        {"rows_found": production_data.get("total_rows", "unknown")},
    )

    # ── Step 2: Ask LLM to summarise the data ──────────────────────────
    messages = [
        SystemMessage(content=DATA_RETRIEVAL_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Here is the production data retrieved from the CSV file:\n"
            f"{tool_output[:3000]}\n\n"
            f"Summarise this data concisely and factually."
        )),
    ]

    summary_response = llm.invoke(messages)
    data_summary = summary_response.content.strip()

    completed.append("data_retrieval")

    trace_entry = log_agent_action(
        _logger, "DataRetrievalAgent", "completed",
        {"rows_found": production_data.get("total_rows", "unknown")},
    )

    return {
        "production_data": production_data,
        "completed_tasks": completed,
        "agent_trace": trace + [trace_entry],
        "messages": [
            AIMessage(content=f"[DataRetrievalAgent] Data Summary:\n{data_summary}")
        ],
    }
