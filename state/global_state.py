"""
Global State Definition for Smart Factory MAS
===============================================
TypedDict-based state ensures type-safe context passing between all agents
without data loss. LangGraph uses this schema to manage the shared state
across the entire coordinator-worker workflow.
"""

from typing import Annotated, Sequence
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class FactoryState(TypedDict):
    """
    Global state shared across all agents in the Smart Factory MAS.

    Attributes
    ----------
    messages : Sequence[BaseMessage]
        Full conversation history. Uses LangGraph's ``add_messages`` reducer
        so new messages are *appended* rather than overwriting.
    current_task : str
        The sub-task currently being executed (set by the Coordinator).
    task_queue : list[str]
        Ordered list of remaining sub-tasks the Coordinator must dispatch.
    completed_tasks : list[str]
        Sub-tasks that have been completed by worker agents.
    production_data : dict
        Raw / summarised data retrieved from the production CSV log.
    ptp_analysis : dict
        Plan-to-Performance efficiency calculations and classifications.
    inventory_status : dict
        Results from the SQLite inventory database query.
    bottleneck_findings : str
        Free-text summary produced by the Bottleneck Analyst.
    optimization_plan : str
        Actionable improvement plan from the Optimization Strategist.
    final_report : str
        The synthesised final report content (Markdown).
    report_path : str
        Filesystem path where the report was saved.
    agent_trace : list[dict]
        Observability trace – every agent action / tool call is appended here.
    iteration_count : int
        Safety counter for the coordinator loop.
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]
    current_task: str
    task_queue: list[str]
    completed_tasks: list[str]
    production_data: dict
    ptp_analysis: dict
    inventory_status: dict
    bottleneck_findings: str
    optimization_plan: str
    final_report: str
    report_path: str
    agent_trace: list[dict]
    iteration_count: int
