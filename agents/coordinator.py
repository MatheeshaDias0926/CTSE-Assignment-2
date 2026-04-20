"""
Production Coordinator Agent
==============================
Supervisor node that decomposes user queries into sub-tasks,
routes them to worker agents, and synthesises the final response.
"""

import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from config import OLLAMA_MODEL, OLLAMA_BASE_URL, TEMPERATURE, MAX_COORDINATOR_ITERATIONS
from logger import get_logger, log_agent_action
from prompts.system_prompts import COORDINATOR_SYSTEM_PROMPT, SYNTHESIS_PROMPT
from state.global_state import FactoryState

_logger = get_logger("Coordinator")


def _get_llm() -> ChatOllama:
    """Instantiate the Ollama LLM for the coordinator."""
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=TEMPERATURE,
    )


def coordinator_node(state: FactoryState) -> dict[str, Any]:
    """
    Coordinator node for the LangGraph workflow.

    On first invocation (empty task_queue and no completed tasks), it
    decomposes the user query into a task list. On subsequent invocations,
    it pops the next task and routes accordingly.

    Args:
        state: The current global FactoryState.

    Returns:
        Updated state fields: current_task, task_queue, agent_trace,
        iteration_count, and messages.
    """
    iteration = state.get("iteration_count", 0) + 1
    task_queue = list(state.get("task_queue", []))
    completed = list(state.get("completed_tasks", []))
    trace = list(state.get("agent_trace", []))

    log_agent_action(
        _logger, "Coordinator", "invoked",
        {"iteration": iteration, "queue": task_queue, "completed": completed},
    )

    # ── Safety guard ────────────────────────────────────────────────────
    if iteration > MAX_COORDINATOR_ITERATIONS:
        _logger.warning("Max iterations reached — forcing completion.")
        return {
            "current_task": "DONE",
            "task_queue": [],
            "iteration_count": iteration,
            "agent_trace": trace + [
                log_agent_action(_logger, "Coordinator", "max_iterations_reached", {})
            ],
        }

    # ── First invocation: decompose query into tasks ────────────────────
    if not task_queue and not completed:
        llm = _get_llm()
        messages = [
            SystemMessage(content=COORDINATOR_SYSTEM_PROMPT),
            HumanMessage(content=(
                "Decompose this request into sub-tasks. "
                "Reply ONLY with a JSON list of task names.\n\n"
                f"User request: {_extract_user_query(state)}"
            )),
        ]

        response = llm.invoke(messages)
        response_text = response.content.strip()

        log_agent_action(_logger, "Coordinator", "decomposition_response", response_text)

        # Parse task list — fallback to default sequence if parsing fails
        task_queue = _parse_task_list(response_text)

        if not task_queue:
            task_queue = ["data_retrieval", "bottleneck_analysis", "optimization"]
            _logger.info("Using default task sequence (LLM parse failed).")

        next_task = task_queue.pop(0)
        trace_entry = log_agent_action(
            _logger, "Coordinator", "task_decomposed",
            {"tasks": [next_task] + task_queue, "routing_to": next_task},
        )

        return {
            "current_task": next_task,
            "task_queue": task_queue,
            "completed_tasks": [],
            "iteration_count": iteration,
            "agent_trace": trace + [trace_entry],
            "messages": [AIMessage(content=f"[Coordinator] Task decomposed. Starting with: {next_task}")],
        }

    # ── Subsequent invocations: pop next task or finish ─────────────────
    if task_queue:
        next_task = task_queue.pop(0)
        trace_entry = log_agent_action(
            _logger, "Coordinator", "routing_next_task",
            {"next_task": next_task, "remaining": task_queue},
        )
        return {
            "current_task": next_task,
            "task_queue": task_queue,
            "iteration_count": iteration,
            "agent_trace": trace + [trace_entry],
            "messages": [AIMessage(content=f"[Coordinator] Routing to: {next_task}")],
        }

    # ── All tasks done — mark for synthesis ─────────────────────────────
    trace_entry = log_agent_action(
        _logger, "Coordinator", "all_tasks_complete",
        {"completed": completed},
    )
    return {
        "current_task": "DONE",
        "task_queue": [],
        "iteration_count": iteration,
        "agent_trace": trace + [trace_entry],
        "messages": [AIMessage(content="[Coordinator] All worker tasks completed. Synthesising…")],
    }


def synthesizer_node(state: FactoryState) -> dict[str, Any]:
    """
    Final synthesis node — combines all worker outputs into a cohesive summary.

    Args:
        state: The current global FactoryState.

    Returns:
        Updated state with final_report and messages.
    """
    log_agent_action(_logger, "Coordinator", "synthesising_final_report", {})

    llm = _get_llm()

    # Gather worker outputs
    prod_data = state.get("production_data", {})
    prod_summary = json.dumps(prod_data, indent=2, default=str) if prod_data else "No data retrieved."

    bottleneck = state.get("bottleneck_findings", "No bottleneck analysis available.")
    optimization = state.get("optimization_plan", "No optimization plan available.")
    report_path = state.get("report_path", "No report generated.")

    synthesis_prompt = SYNTHESIS_PROMPT.format(
        production_summary=prod_summary[:2000],  # Truncate for context window
        bottleneck_findings=bottleneck[:2000],
        optimization_plan=optimization[:2000],
        report_path=report_path,
    )

    messages = [
        SystemMessage(content="You are the Production Coordinator. Synthesise the final report."),
        HumanMessage(content=synthesis_prompt),
    ]

    response = llm.invoke(messages)
    final_report = response.content.strip()

    trace_entry = log_agent_action(
        _logger, "Coordinator", "synthesis_complete",
        {"report_length": len(final_report)},
    )
    trace = list(state.get("agent_trace", []))

    return {
        "final_report": final_report,
        "agent_trace": trace + [trace_entry],
        "messages": [AIMessage(content=f"[Coordinator] Final Report:\n\n{final_report}")],
    }


# ── Routing Functions ───────────────────────────────────────────────────────

def route_to_worker(state: FactoryState) -> str:
    """
    Conditional edge function — routes to the appropriate worker node
    based on ``current_task``, or to the synthesiser if all tasks are done.

    Returns:
        Node name: 'data_retrieval', 'bottleneck_analyst',
        'optimization_strategist', or 'synthesizer'.
    """
    current_task = state.get("current_task", "DONE")

    routing_map = {
        "data_retrieval": "data_retrieval",
        "bottleneck_analysis": "bottleneck_analyst",
        "optimization": "optimization_strategist",
        "DONE": "synthesizer",
    }

    destination = routing_map.get(current_task, "synthesizer")
    log_agent_action(
        _logger, "Coordinator", "routing",
        {"current_task": current_task, "destination": destination},
    )
    return destination


# ── Helpers ─────────────────────────────────────────────────────────────────

def _extract_user_query(state: FactoryState) -> str:
    """Pull the original user query from the message history."""
    messages = state.get("messages", [])
    for msg in messages:
        if isinstance(msg, HumanMessage):
            return msg.content
    return "Analyse the production data, identify bottlenecks, and suggest optimizations."


def _parse_task_list(text: str) -> list[str]:
    """
    Attempt to extract a JSON list of task names from the LLM response.
    Falls back to an empty list if parsing fails.
    """
    # Try direct JSON parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            valid = [t for t in parsed if t in ("data_retrieval", "bottleneck_analysis", "optimization")]
            return valid if valid else []
    except json.JSONDecodeError:
        pass

    # Try to find JSON array in the text
    import re
    match = re.search(r'\[.*?\]', text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                valid = [t for t in parsed if t in ("data_retrieval", "bottleneck_analysis", "optimization")]
                return valid if valid else []
        except json.JSONDecodeError:
            pass

    return []
