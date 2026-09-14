"""Generate deterministic synthetic asset master + maintenance history for RailBlock Phase 2.0."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 26027
HISTORY_START = pd.Timestamp("2025-01-01")
HISTORY_END = pd.Timestamp("2026-01-04")
REFERENCE_DATE = pd.Timestamp("2026-01-05")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic"
TASKS_FILE = SYNTHETIC_DIR / "maintenance_tasks.csv"
ASSET_MASTER_FILE = SYNTHETIC_DIR / "asset_master.csv"
HISTORY_FILE = SYNTHETIC_DIR / "maintenance_history.csv"


def generate_asset_master(tasks_df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    required = {"asset_id", "department", "asset_type", "section_id", "criticality"}
    missing = required - set(tasks_df.columns)
    if missing:
        raise ValueError(f"maintenance_tasks.csv is missing required columns: {sorted(missing)}")
    assets = (tasks_df[list(required)].drop_duplicates("asset_id").sort_values("asset_id").reset_index(drop=True))
    rows = []
    ranges = {"critical": (2012, 2020), "high": (2010, 2021), "medium": (2008, 2022), "low": (2010, 2023)}
    for _, a in assets.iterrows():
        low, high = ranges.get(str(a.criticality), (2010, 2022))
        year = int(rng.integers(low, high + 1))
        max_back = max(60, min(365, 45 + (REFERENCE_DATE.year - year) * 12))
        last_major = REFERENCE_DATE - pd.Timedelta(days=int(rng.integers(30, max_back + 1)))
        rows.append({
            "asset_id": a.asset_id, "department": a.department, "asset_type": a.asset_type,
            "section_id": a.section_id, "criticality": a.criticality,
            "installation_year": year, "last_major_maintenance": last_major.normalize(),
            "age_years": REFERENCE_DATE.year - year,
        })
    return pd.DataFrame(rows)


def generate_maintenance_history(asset_master: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows, counter = [], 1
    interval_ranges = {"critical": (25, 75), "high": (35, 100), "medium": (50, 150), "low": (70, 210)}
    defect_p = {"critical": .70, "high": .52, "medium": .34, "low": .20}
    failure_p = {"critical": .16, "high": .10, "medium": .06, "low": .03}
    for _, a in asset_master.iterrows():
        c = str(a.criticality)
        date = HISTORY_START + pd.Timedelta(days=int(rng.integers(0, 45)))
        while date <= HISTORY_END:
            defect = int(rng.random() < defect_p.get(c, .30))
            failure = int(defect and rng.random() < failure_p.get(c, .05))
            if failure:
                mtype, severity, downtime = "emergency_repair", "critical", int(rng.choice([180, 240, 300, 360]))
            elif defect:
                mtype = str(rng.choice(["repair", "corrective_maintenance", "inspection"]))
                severity = str(rng.choice(["medium", "high"], p=[.65, .35]))
                downtime = int(rng.choice([60, 90, 120, 150, 180, 240]))
            else:
                mtype, severity = str(rng.choice(["inspection", "preventive_maintenance"])), "low"
                downtime = int(rng.choice([30, 45, 60, 90]))
            rows.append({
                "history_id": f"H{counter:05d}", "asset_id": a.asset_id, "section_id": a.section_id,
                "department": a.department, "maintenance_date": date.normalize(),
                "maintenance_type": mtype, "defect_found": defect, "failure_occurred": failure,
                "downtime_min": downtime, "severity": severity,
            })
            counter += 1
            low, high = interval_ranges.get(c, (50, 150))
            date += pd.Timedelta(days=int(rng.integers(low, high + 1)))
    return pd.DataFrame(rows).sort_values(["maintenance_date", "asset_id", "history_id"]).reset_index(drop=True)


def validate_outputs(assets: pd.DataFrame, history: pd.DataFrame) -> None:
    if assets["asset_id"].duplicated().any(): raise ValueError("Duplicate asset_id")
    if history["history_id"].duplicated().any(): raise ValueError("Duplicate history_id")
    if not set(history.asset_id) <= set(assets.asset_id): raise ValueError("Unknown asset in history")
    dates = pd.to_datetime(history.maintenance_date)
    if not dates.between(HISTORY_START, HISTORY_END).all(): raise ValueError("History date outside range")
    if (history.downtime_min <= 0).any(): raise ValueError("Non-positive downtime")
    if not set(history.severity) <= {"low", "medium", "high", "critical"}: raise ValueError("Invalid severity")


def main() -> None:
    if not TASKS_FILE.exists():
        raise FileNotFoundError(f"Could not find {TASKS_FILE}")
    rng = np.random.default_rng(SEED)
    tasks = pd.read_csv(TASKS_FILE)
    assets = generate_asset_master(tasks, rng)
    history = generate_maintenance_history(assets, rng)
    if history.empty: raise ValueError("Generated history is empty")
    validate_outputs(assets, history)
    ASSET_MASTER_FILE.parent.mkdir(parents=True, exist_ok=True)
    assets.to_csv(ASSET_MASTER_FILE, index=False)
    history.to_csv(HISTORY_FILE, index=False)
    print("RailBlock Phase 2.0 synthetic data generated")
    print(f"Assets              : {len(assets)}")
    print(f"Historical events   : {len(history)}")
    print(f"History period      : {HISTORY_START.date()} to {HISTORY_END.date()}")
    print(f"Asset master        : {ASSET_MASTER_FILE}")
    print(f"Maintenance history : {HISTORY_FILE}")

if __name__ == "__main__": main()