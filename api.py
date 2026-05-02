"""
Smart Factory MAS — FastAPI Backend
=====================================
Serves the web dashboard and provides SSE streaming for real-time
agent progress during MAS workflow execution.

Usage:
    python api.py
    Open http://localhost:8000
"""

import asyncio
import json
import os
import glob
import time
import threading
from queue import Queue, Empty
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from config import OUTPUT_DIR, PRODUCTION_CSV_PATH, INVENTORY_DB_PATH

app = FastAPI(title="Smart Factory MAS Dashboard")

# Serve frontend static files
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


# ── Serve the dashboard ────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the main dashboard HTML."""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


# ── SSE Streaming Workflow ─────────────────────────────────────────────────
@app.get("/api/run/stream")
async def stream_workflow(query: str):
    """
    SSE endpoint that runs the MAS workflow and streams progress events.
    Each agent step is sent as an SSE event in real-time.
    """
    event_queue: Queue = Queue()

    def run_mas_in_thread():
        """Run the MAS workflow in a background thread, pushing events to the queue."""
        try:
            from langchain_core.messages import HumanMessage
            from langgraph.graph import StateGraph, START, END
            from config import RECURSION_LIMIT
            from state.global_state import FactoryState
            from agents.coordinator import (
                coordinator_node, synthesizer_node, route_to_worker,
            )
            from agents.data_retrieval import data_retrieval_node
            from agents.bottleneck_analyst import bottleneck_analyst_node
            from agents.optimization_strategist import optimization_strategist_node

            # Build graph
            graph = StateGraph(FactoryState)
            graph.add_node("coordinator", coordinator_node)
            graph.add_node("data_retrieval", data_retrieval_node)
            graph.add_node("bottleneck_analyst", bottleneck_analyst_node)
            graph.add_node("optimization_strategist", optimization_strategist_node)
            graph.add_node("synthesizer", synthesizer_node)
            graph.add_edge(START, "coordinator")
            graph.add_conditional_edges("coordinator", route_to_worker, {
                "data_retrieval": "data_retrieval",
                "bottleneck_analyst": "bottleneck_analyst",
                "optimization_strategist": "optimization_strategist",
                "synthesizer": "synthesizer",
            })
            graph.add_edge("data_retrieval", "coordinator")
            graph.add_edge("bottleneck_analyst", "coordinator")
            graph.add_edge("optimization_strategist", "coordinator")
            graph.add_edge("synthesizer", END)
            workflow = graph.compile()

            initial_state = {
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
            final_state = None

            for step_num, event in enumerate(
                workflow.stream(initial_state, {"recursion_limit": RECURSION_LIMIT}),
                start=1,
            ):
                for node_name, output in event.items():
                    detail = ""
                    extra = {}

                    if node_name == "coordinator":
                        task = output.get("current_task", "")
                        completed = output.get("completed_tasks", [])
                        detail = f"Current task: {task}" if task else "Routing..."
                        extra = {"current_task": task, "completed_tasks": completed}
                    elif node_name == "data_retrieval":
                        rows = output.get("production_data", {}).get("total_rows", "?")
                        detail = f"Retrieved {rows} production records"
                        extra = {"total_rows": rows}
                    elif node_name == "bottleneck_analyst":
                        ptp = output.get("ptp_analysis", {})
                        inv = output.get("inventory_status", {})
                        findings_len = len(output.get("bottleneck_findings", ""))
                        detail = f"Analyzed {len(ptp.get('line_results', []))} lines, {inv.get('result_count', '?')} low-stock items"
                        extra = {
                            "ptp_results": ptp.get("line_results", []),
                            "inventory": inv,
                            "findings": output.get("bottleneck_findings", ""),
                        }
                    elif node_name == "optimization_strategist":
                        path = output.get("report_path", "")
                        detail = f"Report saved"
                        extra = {
                            "report_path": path,
                            "optimization_plan": output.get("optimization_plan", ""),
                        }
                    elif node_name == "synthesizer":
                        detail = "Final synthesis complete"
                        extra = {"final_report": output.get("final_report", "")}

                    event_queue.put({
                        "event": "step",
                        "data": {
                            "step": step_num,
                            "agent": node_name,
                            "detail": detail,
                            "elapsed": round(time.time() - start_time, 1),
                            **extra,
                        },
                    })
                    final_state = output

            elapsed = round(time.time() - start_time, 1)
            event_queue.put({
                "event": "complete",
                "data": {
                    "final_report": final_state.get("final_report", "") if final_state else "",
                    "report_path": final_state.get("report_path", "") if final_state else "",
                    "elapsed": elapsed,
                    "trace_count": len(final_state.get("agent_trace", [])) if final_state else 0,
                },
            })

        except Exception as e:
            event_queue.put({
                "event": "error",
                "data": {"message": str(e)},
            })
        finally:
            event_queue.put(None)  # Sentinel to stop SSE

    # Start workflow in background thread
    thread = threading.Thread(target=run_mas_in_thread, daemon=True)
    thread.start()

    # SSE generator
    async def event_generator():
        while True:
            try:
                item = event_queue.get(timeout=0.5)
            except Empty:
                # Send keepalive
                yield {"event": "ping", "data": "{}"}
                continue

            if item is None:
                break

            yield {
                "event": item["event"],
                "data": json.dumps(item["data"], default=str),
            }

    return EventSourceResponse(event_generator())


# ── Reports API ────────────────────────────────────────────────────────────
@app.get("/api/reports")
async def list_reports():
    """List all generated optimization reports."""
    pattern = os.path.join(OUTPUT_DIR, "optimization_report_*.md")
    reports = glob.glob(pattern)
    reports.sort(key=os.path.getmtime, reverse=True)
    return JSONResponse([
        {
            "filename": os.path.basename(r),
            "size": os.path.getsize(r),
            "modified": os.path.getmtime(r),
        }
        for r in reports
    ])


@app.get("/api/reports/{filename}")
async def get_report(filename: str):
    """Return the content of a specific report."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(filepath):
        return JSONResponse({"error": "Not found"}, status_code=404)
    with open(filepath, "r", encoding="utf-8") as f:
        return JSONResponse({"filename": filename, "content": f.read()})


# ── Evaluation API ─────────────────────────────────────────────────────────
@app.post("/api/evaluate")
async def run_evaluation():
    """Run LLM-as-a-Judge evaluation on the latest report."""
    event_queue: Queue = Queue()

    def eval_in_thread():
        try:
            from evaluation.llm_judge import evaluate_report, find_latest_report, save_evaluation
            report_path = find_latest_report()
            if not report_path:
                event_queue.put({"error": "No report found"})
                return
            with open(report_path, "r", encoding="utf-8") as f:
                content = f.read()
            result = evaluate_report(content)
            save_evaluation(result, report_path)

            # Map the scores to the frontend format
            scores = result.get("scores", {})
            criteria = {}
            for key, val in scores.items():
                if isinstance(val, dict):
                    criteria[key] = {
                        "score": val.get("score", 0),
                        "justification": val.get("justification", ""),
                    }

            event_queue.put({
                "evaluation": {
                    "criteria": criteria,
                    "overall_score": result.get("overall_score", 0),
                    "summary": result.get("summary", ""),
                },
                "report": os.path.basename(report_path),
            })
        except Exception as e:
            event_queue.put({"error": str(e)})

    thread = threading.Thread(target=eval_in_thread, daemon=True)
    thread.start()
    thread.join(timeout=120)

    try:
        result = event_queue.get(timeout=1)
        if "error" in result:
            return JSONResponse(result, status_code=500)
        return JSONResponse(result)
    except Empty:
        return JSONResponse({"error": "Evaluation timed out"}, status_code=504)


# ── Data Summary API ───────────────────────────────────────────────────────
@app.get("/api/data/summary")
async def data_summary():
    """Return a quick summary of the production data."""
    import pandas as pd
    try:
        df = pd.read_csv(PRODUCTION_CSV_PATH)
        lines = df["line_id"].unique().tolist()
        return JSONResponse({
            "total_records": len(df),
            "production_lines": lines,
            "date_range": {"start": df["date"].min(), "end": df["date"].max()},
            "avg_planned": round(df["planned_units"].mean(), 1),
            "avg_actual": round(df["actual_units"].mean(), 1),
            "avg_defects": round(df["defect_count"].mean(), 1),
            "total_downtime": int(df["downtime_minutes"].sum()),
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── Health Check ───────────────────────────────────────────────────────────
@app.get("/api/health")
async def health_check():
    """Check if Ollama is reachable."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get("http://localhost:11434/api/tags")
            models = resp.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            return JSONResponse({"status": "ok", "models": model_names})
    except Exception:
        return JSONResponse({"status": "unreachable"}, status_code=503)


# ── Entry Point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("\n🏭 Smart Factory MAS — Web Dashboard")
    print("   Open http://localhost:8000 in your browser\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
