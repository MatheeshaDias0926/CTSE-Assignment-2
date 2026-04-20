# Smart Factory Production & Efficiency Optimizer

## Multi-Agent System (MAS) — Automotive Manufacturing

A locally-hosted Multi-Agent System that analyses automotive production metrics
and provides optimisation strategies. Built with **LangGraph** and **llama3:8b via Ollama**.

> **Zero external API calls.** Everything runs 100% locally on your machine.

---

## Architecture

```
START → Coordinator → Data Retrieval Agent → Coordinator
                    → Bottleneck Analyst   → Coordinator
                    → Optimization Strategist → Coordinator
                    → Synthesizer → END
```

### Agents

| Agent | Role | Tools |
|-------|------|-------|
| **Production Coordinator** | Decomposes queries, routes tasks, synthesises results | — |
| **Data Retrieval Agent** | Reads production CSV data | `read_production_data` |
| **Bottleneck Analyst** | PTP efficiency analysis + inventory checks | `calculate_ptp_efficiency`, `query_inventory_db` |
| **Optimization Strategist** | Creates improvement plans + writes reports | `write_optimization_report` |

### Custom Tools

| Tool | Description |
|------|-------------|
| `read_production_data` | Reads local CSV files, returns JSON summary with statistics |
| `calculate_ptp_efficiency` | Computes Plan-to-Performance ratio with classification |
| `query_inventory_db` | Queries SQLite database for raw material inventory |
| `write_optimization_report` | Saves Markdown reports to `outputs/` directory |

---

## Prerequisites

- **Python 3.10+**
- **Ollama** installed and running (`ollama serve`)
- **llama3:8b** model pulled (`ollama pull llama3:8b`)

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Sample Data

```bash
python generate_sample_data.py
```

This creates:
- `data/production_log.csv` — 100 automotive production records
- `data/inventory.db` — SQLite database with 15 raw materials

### 3. Run the MAS

```bash
python main.py
```

Or with a custom query:

```bash
python main.py --query "Analyse LINE-03 performance and suggest improvements"
```

### 4. Run Evaluation (Optional)

```bash
python -m evaluation.llm_judge
```

---

## Project Structure

```
├── main.py                        # LangGraph workflow entry point
├── config.py                      # Central configuration
├── logger.py                      # Centralized logging system
├── generate_sample_data.py        # Sample data generator
├── requirements.txt               # Python dependencies
├── agents/
│   ├── coordinator.py             # Production Coordinator
│   ├── data_retrieval.py          # Data Retrieval Agent
│   ├── bottleneck_analyst.py      # Bottleneck Analyst
│   └── optimization_strategist.py # Optimization Strategist
├── tools/
│   ├── production_tools.py        # CSV reader + PTP calculator
│   ├── inventory_tools.py         # SQLite inventory queries
│   └── report_tools.py           # Report writer
├── state/
│   └── global_state.py           # TypedDict global state
├── prompts/
│   └── system_prompts.py         # SLM-optimized system prompts
├── evaluation/
│   └── llm_judge.py              # LLM-as-a-Judge evaluator
├── data/                          # Generated sample data
├── outputs/                       # Generated reports
└── logs/                          # Execution logs
```

---

## Observability

All agent actions and tool calls are logged to:
- **Console** — color-coded real-time output
- **Log file** — `logs/mas_execution_<timestamp>.log`

Each log entry includes: timestamp, agent name, action type, and details.

---

## Technical Stack

| Component | Technology |
|-----------|-----------|
| LLM Engine | llama3:8b via Ollama |
| Orchestrator | LangGraph (StateGraph) |
| State Management | Python TypedDict |
| Data Processing | Pandas |
| Database | SQLite3 |
| Logging | Python logging module |
