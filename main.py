"""
Smart Factory MAS — Main LangGraph Workflow
=============================================
Coordinator-Worker orchestration for automotive production analysis.

Usage:
    python main.py
    python main.py --query "Analyse LINE-03 performance"
"""

import argparse
import sys
import time

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END

from config import RECURSION_LIMIT
from logger import get_logger, log_agent_action
from state.global_state import FactoryState
from agents.coordinator import (
    coordinator_node, synthesizer_node, route_to_worker,
)
from agents.data_retrieval import data_retrieval_node
from agents.bottleneck_analyst import bottleneck_analyst_node
from agents.optimization_strategist import optimization_strategist_node

_logger = get_logger("Main")


def build_graph():
    """Build and compile the LangGraph StateGraph."""
    graph = StateGraph(FactoryState)

    graph.add_node("coordinator", coordinator_node)
    graph.add_node("data_retrieval", data_retrieval_node)
    graph.add_node("bottleneck_analyst", bottleneck_analyst_node)
    graph.add_node("optimization_strategist", optimization_strategist_node)
    graph.add_node("synthesizer", synthesizer_node)

    graph.add_edge(START, "coordinator")
    graph.add_conditional_edges(
        "coordinator",
        route_to_worker,
        {
            "data_retrieval": "data_retrieval",
            "bottleneck_analyst": "bottleneck_analyst",
            "optimization_strategist": "optimization_strategist",
            "synthesizer": "synthesizer",
        },
    )
    graph.add_edge("data_retrieval", "coordinator")
    graph.add_edge("bottleneck_analyst", "coordinator")
    graph.add_edge("optimization_strategist", "coordinator")
    graph.add_edge("synthesizer", END)

    _logger.info("LangGraph workflow built successfully.")
    return graph.compile()


DEFAULT_QUERY = (
    "Analyse the automotive production data for all production lines. "
    "Identify efficiency bottlenecks using PTP metrics, "
    "check raw material inventory for shortages, and provide a "
    "comprehensive optimization plan with actionable recommendations."
)


def run_workflow(query: str) -> dict:
    """Execute the full MAS workflow."""
    _logger.info("=" * 70)
    _logger.info("  SMART FACTORY MAS — WORKFLOW STARTED")
    _logger.info(f"  Query: {query}")
    _logger.info("=" * 70)

    app = build_graph()

    initial_state: FactoryState = {
        "messages": [HumanMessage(content=query)],
        "current_task": "",
        "task_queue": [],
        "completed_tasks": [],
        "production_data": {},
        "ptp_analysis": {},
        "inventory_status": {},
        "bottleneck_findings": "",
        "optimization_plan": "",
        "final_report": "",
        "report_path": "",
        "agent_trace": [],
        "iteration_count": 0,
    }

    start_time = time.time()
    icons = {
        "coordinator": "🎯", "data_retrieval": "📂",
        "bottleneck_analyst": "🔍", "optimization_strategist": "💡",
        "synthesizer": "📊",
    }

    print("\n" + "═" * 70)
    print("  🏭 SMART FACTORY MULTI-AGENT SYSTEM")
    print("  🤖 LLM: llama3:8b via Ollama (100% Local)")
    print("  📋 Query:", query[:80])
    print("═" * 70 + "\n")

    final_state = None
    try:
        for step_num, event in enumerate(
            app.stream(initial_state, {"recursion_limit": RECURSION_LIMIT}),
            start=1,
        ):
            for node_name, output in event.items():
                icon = icons.get(node_name, "⚙️")
                print(f"  {icon} Step {step_num}: {node_name}")

                if node_name == "coordinator":
                    task = output.get("current_task", "")
                    if task:
                        print(f"     ↳ Current task: {task}")
                elif node_name == "data_retrieval":
                    rows = output.get("production_data", {}).get("total_rows", "?")
                    print(f"     ↳ Retrieved {rows} production records")
                elif node_name == "bottleneck_analyst":
                    f_len = len(output.get("bottleneck_findings", ""))
                    print(f"     ↳ Analysis complete ({f_len} chars)")
                elif node_name == "optimization_strategist":
                    path = output.get("report_path", "")
                    print(f"     ↳ Report saved: {path}")

                final_state = output
    except Exception as e:
        _logger.error(f"Workflow failed: {e}")
        print(f"\n❌ ERROR: {e}")
        raise

    elapsed = time.time() - start_time

    print("\n" + "═" * 70)
    print("  ✅ WORKFLOW COMPLETE")
    print(f"  ⏱  Time: {elapsed:.1f} seconds")
    print("═" * 70)

    if final_state:
        report = final_state.get("final_report", "")
        if report:
            print("\n" + "─" * 70)
            print("  📊 FINAL EXECUTIVE SUMMARY")
            print("─" * 70)
            print(report)
        path = final_state.get("report_path", "")
        if path:
            print(f"\n📄 Full report saved to: {path}")
        trace = final_state.get("agent_trace", [])
        print(f"\n🔍 Agent trace: {len(trace)} actions logged")

    _logger.info(f"Workflow completed in {elapsed:.1f}s")
    return final_state


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Smart Factory Production & Efficiency Optimizer"
    )
    parser.add_argument("--query", "-q", default=DEFAULT_QUERY)
    args = parser.parse_args()

    try:
        run_workflow(args.query)
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
