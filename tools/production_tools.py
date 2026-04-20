"""
Production Data Tools
======================
Tools for reading local CSV production logs and calculating
Plan-to-Performance (PTP) efficiency metrics.

All tools use strict type hints and comprehensive docstrings
so the LLM can understand when and how to invoke them.
"""

import json
import os
from typing import Optional

import pandas as pd
from langchain_core.tools import tool

from config import PRODUCTION_CSV_PATH
from logger import get_logger, log_tool_call

_logger = get_logger("ProductionTools")


@tool
def read_production_data(file_path: Optional[str] = None) -> str:
    """Read production data from a local CSV file and return a JSON summary.

    This tool reads an automotive production log CSV file and returns:
    - The first 20 rows of data as records
    - Column names and data types
    - Basic statistics (count, mean, min, max) for numeric columns
    - Total number of rows in the dataset

    Args:
        file_path: Absolute or relative path to the CSV file.
                   If not provided, uses the default production_log.csv.

    Returns:
        A JSON string containing the data summary. Returns an error
        message string if the file cannot be read.
    """
    path = file_path if file_path else PRODUCTION_CSV_PATH

    try:
        if not os.path.exists(path):
            error_msg = f"ERROR: File not found at '{path}'"
            log_tool_call(_logger, "read_production_data", {"file_path": path}, error_msg)
            return error_msg

        df = pd.read_csv(path)

        # Build summary
        summary = {
            "total_rows": len(df),
            "columns": list(df.columns),
            "column_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "sample_data": json.loads(df.head(20).to_json(orient="records")),
            "numeric_stats": {},
        }

        # Compute statistics for numeric columns
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
        for col in numeric_cols:
            summary["numeric_stats"][col] = {
                "count": int(df[col].count()),
                "mean": round(float(df[col].mean()), 2),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "std": round(float(df[col].std()), 2),
            }

        result = json.dumps(summary, indent=2)
        log_tool_call(
            _logger,
            "read_production_data",
            {"file_path": path},
            f"Successfully read {len(df)} rows from {os.path.basename(path)}",
        )
        return result

    except pd.errors.EmptyDataError:
        error_msg = f"ERROR: CSV file at '{path}' is empty."
        log_tool_call(_logger, "read_production_data", {"file_path": path}, error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"ERROR: Failed to read CSV — {type(e).__name__}: {e}"
        log_tool_call(_logger, "read_production_data", {"file_path": path}, error_msg)
        return error_msg


@tool
def calculate_ptp_efficiency(
    planned: float,
    actual: float,
    target_threshold: float = 85.0,
) -> str:
    """Calculate Plan-to-Performance (PTP) efficiency for a production line.

    Computes the ratio of actual output to planned output, expressed as
    a percentage, and classifies the result into a severity category.

    Classification thresholds:
        - CRITICAL  : PTP < 70%
        - WARNING   : 70% <= PTP < 85%
        - ON_TARGET : 85% <= PTP < 95%
        - EXCEEDING : PTP >= 95%

    Args:
        planned:          The planned/target number of units.
        actual:           The actual number of units produced.
        target_threshold: Custom threshold for 'ON_TARGET' lower bound
                          (default 85.0).

    Returns:
        A JSON string with keys: planned, actual, ptp_percentage,
        classification, gap_units, and recommendation.
    """
    try:
        if planned <= 0:
            error_msg = "ERROR: 'planned' must be a positive number."
            log_tool_call(
                _logger,
                "calculate_ptp_efficiency",
                {"planned": planned, "actual": actual},
                error_msg,
            )
            return error_msg

        ptp_pct = round((actual / planned) * 100, 2)
        gap = round(planned - actual, 2)

        # Classify
        if ptp_pct < 70:
            classification = "CRITICAL"
            recommendation = (
                "Immediate investigation required. Production is severely "
                "under-performing. Check for equipment failures, staffing "
                "shortages, or raw-material stock-outs."
            )
        elif ptp_pct < target_threshold:
            classification = "WARNING"
            recommendation = (
                "Performance is below target. Review shift schedules, "
                "machine downtime logs, and quality rejection rates."
            )
        elif ptp_pct < 95:
            classification = "ON_TARGET"
            recommendation = (
                "Production is within acceptable range. Continue monitoring "
                "and look for incremental improvements."
            )
        else:
            classification = "EXCEEDING"
            recommendation = (
                "Excellent performance. Document best practices and consider "
                "raising planned targets for future periods."
            )

        result = {
            "planned_units": planned,
            "actual_units": actual,
            "ptp_percentage": ptp_pct,
            "classification": classification,
            "gap_units": gap,
            "target_threshold": target_threshold,
            "recommendation": recommendation,
        }

        result_str = json.dumps(result, indent=2)
        log_tool_call(
            _logger,
            "calculate_ptp_efficiency",
            {"planned": planned, "actual": actual, "target_threshold": target_threshold},
            f"PTP={ptp_pct}% [{classification}]",
        )
        return result_str

    except Exception as e:
        error_msg = f"ERROR: PTP calculation failed — {type(e).__name__}: {e}"
        log_tool_call(
            _logger,
            "calculate_ptp_efficiency",
            {"planned": planned, "actual": actual},
            error_msg,
        )
        return error_msg
