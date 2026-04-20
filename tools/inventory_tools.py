"""
Inventory Database Tool
========================
Simulates local SQL/SQLite queries against a raw-materials inventory
database for the automotive production facility.
"""

import json
import os
import sqlite3
from typing import Literal

from langchain_core.tools import tool

from config import INVENTORY_DB_PATH
from logger import get_logger, log_tool_call

_logger = get_logger("InventoryTools")


@tool
def query_inventory_db(
    query_type: str,
    material_name: str = "",
) -> str:
    """Query the local SQLite inventory database for raw-material information.

    Connects to the local inventory.db and retrieves information about
    automotive raw materials used in the production facility.

    Supported query types:
        - "stock_level"  : Get current stock for a specific material.
        - "low_stock"    : List all materials below their minimum threshold.
        - "supplier_info": Get supplier details for a specific material.
        - "all"          : Return the full inventory table.

    Args:
        query_type:    One of "stock_level", "low_stock", "supplier_info",
                       or "all".
        material_name: Name of the material to query (required for
                       "stock_level" and "supplier_info"; ignored for
                       "low_stock" and "all").

    Returns:
        A JSON string with the query results, or an error message if the
        query fails.
    """
    try:
        if not os.path.exists(INVENTORY_DB_PATH):
            error_msg = f"ERROR: Inventory database not found at '{INVENTORY_DB_PATH}'"
            log_tool_call(
                _logger, "query_inventory_db",
                {"query_type": query_type, "material_name": material_name},
                error_msg,
            )
            return error_msg

        conn = sqlite3.connect(INVENTORY_DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if query_type == "stock_level":
            if not material_name:
                return "ERROR: 'material_name' is required for query_type 'stock_level'."
            cursor.execute(
                "SELECT * FROM raw_materials WHERE LOWER(name) LIKE LOWER(?)",
                (f"%{material_name}%",),
            )

        elif query_type == "low_stock":
            cursor.execute(
                "SELECT * FROM raw_materials WHERE stock_qty < min_threshold"
            )

        elif query_type == "supplier_info":
            if not material_name:
                return "ERROR: 'material_name' is required for query_type 'supplier_info'."
            cursor.execute(
                "SELECT name, supplier, lead_time_days, last_restock "
                "FROM raw_materials WHERE LOWER(name) LIKE LOWER(?)",
                (f"%{material_name}%",),
            )

        elif query_type == "all":
            cursor.execute("SELECT * FROM raw_materials")

        else:
            conn.close()
            return (
                f"ERROR: Unknown query_type '{query_type}'. "
                "Use 'stock_level', 'low_stock', 'supplier_info', or 'all'."
            )

        rows = cursor.fetchall()
        conn.close()

        results = [dict(row) for row in rows]
        output = {
            "query_type": query_type,
            "material_filter": material_name or "N/A",
            "result_count": len(results),
            "results": results,
        }

        result_str = json.dumps(output, indent=2, default=str)
        log_tool_call(
            _logger,
            "query_inventory_db",
            {"query_type": query_type, "material_name": material_name},
            f"Returned {len(results)} rows",
        )
        return result_str

    except sqlite3.Error as e:
        error_msg = f"ERROR: SQLite query failed — {e}"
        log_tool_call(
            _logger, "query_inventory_db",
            {"query_type": query_type, "material_name": material_name},
            error_msg,
        )
        return error_msg
    except Exception as e:
        error_msg = f"ERROR: Inventory query failed — {type(e).__name__}: {e}"
        log_tool_call(
            _logger, "query_inventory_db",
            {"query_type": query_type, "material_name": material_name},
            error_msg,
        )
        return error_msg
