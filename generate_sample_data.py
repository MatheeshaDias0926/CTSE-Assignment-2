"""
Sample Data Generator — Automotive Production
================================================
Creates realistic synthetic data for the Smart Factory MAS:
  1. production_log.csv — 100 rows of automotive production records
  2. inventory.db       — SQLite database with 15 raw materials

Run this script once before executing main.py.

Usage:
    python generate_sample_data.py
"""

import csv
import os
import random
import sqlite3
from datetime import datetime, timedelta

from config import PRODUCTION_CSV_PATH, INVENTORY_DB_PATH, DATA_DIR


def generate_production_csv() -> str:
    """
    Generate an automotive production log CSV with 100 rows.

    Columns:
        date, line_id, product, planned_units, actual_units,
        defect_count, downtime_minutes, shift, operator_id

    Includes deliberate bottleneck patterns:
        - LINE-03 has consistently low actual vs planned (simulating equipment issues)
        - Night shifts have higher defect rates
        - Some days have elevated downtime (simulating maintenance windows)

    Returns:
        Path to the generated CSV file.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    # Automotive products and production lines
    products = [
        "Engine Block V6",
        "Transmission Assembly",
        "Brake Rotor Set",
        "Suspension Strut",
        "Exhaust Manifold",
        "Steering Rack",
        "Cylinder Head",
        "Drive Shaft",
    ]

    lines = ["LINE-01", "LINE-02", "LINE-03", "LINE-04", "LINE-05"]
    shifts = ["Morning", "Afternoon", "Night"]
    operators = [f"OP-{i:03d}" for i in range(1, 16)]

    # Start date
    start_date = datetime(2025, 1, 6)

    rows = []
    for i in range(100):
        date = start_date + timedelta(days=i // 5)  # ~5 records per day
        line = lines[i % len(lines)]
        product = random.choice(products)
        shift = shifts[i % len(shifts)]
        operator = random.choice(operators)

        # Base planned units
        planned = random.randint(180, 300)

        # Actual units — introduce bottleneck patterns
        if line == "LINE-03":
            # Bottleneck line: consistently 55-72% of planned
            actual = int(planned * random.uniform(0.55, 0.72))
        elif line == "LINE-05" and shift == "Night":
            # Night shift on LINE-05: occasionally problematic
            actual = int(planned * random.uniform(0.65, 0.85))
        elif shift == "Night":
            # Night shifts generally slightly lower
            actual = int(planned * random.uniform(0.78, 0.92))
        else:
            # Normal performance
            actual = int(planned * random.uniform(0.85, 1.02))

        # Defect count — higher on bottleneck line and night shifts
        if line == "LINE-03":
            defects = random.randint(8, 25)
        elif shift == "Night":
            defects = random.randint(3, 15)
        else:
            defects = random.randint(0, 8)

        # Downtime — inject maintenance windows
        if i % 20 == 0:
            downtime = random.randint(90, 180)  # Scheduled maintenance
        elif line == "LINE-03":
            downtime = random.randint(30, 120)  # Frequent breakdowns
        else:
            downtime = random.randint(0, 45)

        rows.append({
            "date": date.strftime("%Y-%m-%d"),
            "line_id": line,
            "product": product,
            "planned_units": planned,
            "actual_units": actual,
            "defect_count": defects,
            "downtime_minutes": downtime,
            "shift": shift,
            "operator_id": operator,
        })

    # Write CSV
    fieldnames = [
        "date", "line_id", "product", "planned_units", "actual_units",
        "defect_count", "downtime_minutes", "shift", "operator_id",
    ]
    with open(PRODUCTION_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Generated {len(rows)} production records → {PRODUCTION_CSV_PATH}")
    return PRODUCTION_CSV_PATH


def generate_inventory_db() -> str:
    """
    Generate a SQLite inventory database with 15 automotive raw materials.

    Table: raw_materials
        material_id  INTEGER PRIMARY KEY
        name         TEXT
        stock_qty    REAL
        min_threshold REAL
        unit         TEXT
        supplier     TEXT
        lead_time_days INTEGER
        last_restock TEXT (ISO date)

    Includes materials below minimum threshold to simulate stock-outs.

    Returns:
        Path to the generated SQLite database.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    # Remove existing DB to start fresh
    if os.path.exists(INVENTORY_DB_PATH):
        os.remove(INVENTORY_DB_PATH)

    conn = sqlite3.connect(INVENTORY_DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS raw_materials (
            material_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            name          TEXT NOT NULL,
            stock_qty     REAL NOT NULL,
            min_threshold REAL NOT NULL,
            unit          TEXT NOT NULL,
            supplier      TEXT NOT NULL,
            lead_time_days INTEGER NOT NULL,
            last_restock  TEXT NOT NULL
        )
    """)

    # Automotive raw materials — some intentionally below threshold
    materials = [
        ("Aluminium Alloy Sheet (6061)",    1200.0,  500.0, "kg",    "MetalCorp Asia",         14, "2025-01-10"),
        ("High-Carbon Steel Rod",            350.0,  400.0, "kg",    "SteelWorks International", 7, "2025-01-05"),  # LOW
        ("Brake Pad Compound",               180.0,  200.0, "kg",    "FrictionTech Ltd",        10, "2025-01-08"),  # LOW
        ("Cast Iron Ingot",                 2500.0, 1000.0, "kg",    "FoundryMax Inc",          21, "2024-12-28"),
        ("Synthetic Rubber Strip",            75.0,  100.0, "meters","PolyFlex Materials",       5, "2025-01-12"),  # LOW
        ("Copper Wiring Harness",            800.0,  300.0, "meters","WireWorld Electronics",    3, "2025-01-14"),
        ("Tempered Glass Panel",             450.0,  150.0, "units", "GlassTech Premium",       12, "2025-01-06"),
        ("Titanium Bolt Set (M10)",           50.0,   80.0, "boxes", "FastenerPro Japan",        8, "2025-01-03"),  # LOW
        ("Engine Gasket Kit",               1100.0,  400.0, "units", "SealRight Manufacturing", 10, "2025-01-11"),
        ("Hydraulic Fluid ISO 46",          3000.0, 1500.0, "liters","LubeMax Industries",       4, "2025-01-13"),
        ("Carbon Fibre Sheet",               120.0,  200.0, "sq_m", "CompositeTech Korea",     18, "2024-12-20"),  # LOW
        ("Paint Primer (Epoxy)",             600.0,  250.0, "liters","CoatingsPro Germany",      6, "2025-01-09"),
        ("Catalytic Converter Core",          40.0,   60.0, "units", "EmissionTech LLC",        25, "2024-12-15"),  # LOW
        ("Polyurethane Foam Block",          900.0,  350.0, "kg",    "FoamCraft Industries",     7, "2025-01-07"),
        ("Stainless Steel Tube (304)",       280.0,  300.0, "meters","TubeWorks Australia",     11, "2025-01-02"),  # LOW
    ]

    cursor.executemany(
        "INSERT INTO raw_materials "
        "(name, stock_qty, min_threshold, unit, supplier, lead_time_days, last_restock) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        materials,
    )

    conn.commit()
    conn.close()

    low_stock_count = sum(1 for m in materials if m[1] < m[2])
    print(f"✅ Generated {len(materials)} materials ({low_stock_count} below threshold) → {INVENTORY_DB_PATH}")
    return INVENTORY_DB_PATH


def main() -> None:
    """Generate all sample data for the Smart Factory MAS."""
    print("=" * 60)
    print("  Smart Factory MAS — Sample Data Generator")
    print("  Industry: Automotive Manufacturing")
    print("=" * 60)
    print()

    csv_path = generate_production_csv()
    db_path = generate_inventory_db()

    print()
    print("─" * 60)
    print("  All sample data generated successfully!")
    print(f"  CSV: {csv_path}")
    print(f"  DB:  {db_path}")
    print("─" * 60)


if __name__ == "__main__":
    main()
