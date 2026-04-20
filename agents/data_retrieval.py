"""
Data Retrieval Agent
=====================
Worker agent responsible for reading local CSV production logs
and summarising the data for downstream agents.
"""

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.prebuilt import ToolNode

from config import OLLAMA_MODEL, OLLAMA_BASE_URL, TEMPERATURE
from logger import get_logger, log_agent_action
from prompts.system_prompts import DATA_RETRIEVAL_SYSTEM_PROMPT
from state.global_state import FactoryState
from tools.production_tools import read_production_data

_logger = get_logger("DataRetrievalAgent")

# Tools available to this agent
DATA_TOOLS = [read_production_data]


def _get_llm() -> ChatOllama:
    """Instantiate the Ollama LLM with data tools bound."""
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=TEMPERATURE,
    )
    return llm.bind_tools(DATA_TOOLS)


def data_retrieval_node(state: FactoryState) -> dict[str, Any]:
    """
    Data Retrieval worker node.

    Invokes the LLM with tool bindings to read the production CSV,
    then executes any tool calls and feeds results back to the LLM
    for summarisation.

    Args:
        state: The current global FactoryState.

    Returns:
        Updated state with production_data, completed_tasks,
        agent_trace, and messages.
    """
    log_agent_action(_logger, "DataRetrievalAgent", "started", {})

    llm_with_tools = _get_llm()
    tool_node = ToolNode(DATA_TOOLS)
    trace = list(state.get("agent_trace", []))
    completed = list(state.get("completed_tasks", []))

    # ── Step 1: Ask the LLM to use the tool ─────────────────────────────
    messages = [
        SystemMessage(content=DATA_RETRIEVAL_SYSTEM_PROMPT),
        HumanMessage(content=(
            "Read the production data from the default CSV file. "
            "Use the read_production_data tool, then summarise the data."
        )),
    ]

    response = llm_with_tools.invoke(messages)
    messages.append(response)

    # ── Step 2: Execute tool calls if present ───────────────────────────
    production_data = {}
    data_summary = ""

    if response.tool_calls:
        log_agent_action(
            _logger, "DataRetrievalAgent", "tool_calls_detected",
            [tc["name"] for tc in response.tool_calls],
        )

        # Execute tools via ToolNode
        tool_result_state = tool_node.invoke({"messages": messages})
        tool_messages = tool_result_state["messages"]
        messages.extend(tool_messages)

        # Get tool output
        for tm in tool_messages:
            if hasattr(tm, "content"):
                try:
                    import json
                    production_data = json.loads(tm.content)
                except (json.JSONDecodeError, TypeError):
                    production_data = {"raw_output": tm.content}

        # Ask LLM to summarise the tool output
        messages.append(
            HumanMessage(content="Now summarise the production data you received from the tool. Be concise and factual.")
        )
        summary_response = llm_with_tools.invoke(messages)
        data_summary = summary_response.content.strip()
    else:
        # Fallback: directly call the tool if LLM didn't use tool calling
        log_agent_action(
            _logger, "DataRetrievalAgent", "fallback_direct_tool_call", {}
        )
        import json
        tool_output = read_production_data.invoke({})
        try:
            production_data = json.loads(tool_output)
        except (json.JSONDecodeError, TypeError):
            production_data = {"raw_output": tool_output}

        # Ask LLM to summarise
        messages.append(HumanMessage(content=f"Here is the production data:\n{tool_output[:3000]}\n\nSummarise this data concisely."))
        summary_response = llm_with_tools.invoke(messages)
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
