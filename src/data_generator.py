import numpy as np
import pandas as pd

FACILITIES = [
    {"facility_id": "HOSP_A", "name": "Hospital A - Bhopal Referral Hospital", "type": "Hospital", "cold_chain": True, "safety_days": 7},
    {"facility_id": "HOSP_B", "name": "Hospital B - Hamidia Wing Bhopal", "type": "Hospital", "cold_chain": True, "safety_days": 7},
    {"facility_id": "HOSP_C", "name": "Hospital C - Indore Metro Care", "type": "Hospital", "cold_chain": True, "safety_days": 6},
    {"facility_id": "HOSP_D", "name": "Hospital D - Jabalpur District Clinic", "type": "Hospital", "cold_chain": False, "safety_days": 5},
    {"facility_id": "WH_01",  "name": "Central Medical Warehouse - Bhopal", "type": "Warehouse", "cold_chain": True, "safety_days": 10},
]

MEDICINES = [
    {"name": "Ceftriaxone 1g Injection", "cold_chain": True, "criticality": "Critical", "base_daily": 70, "unit_cost": 85.0},
    {"name": "Human Insulin 40IU Vial", "cold_chain": True, "criticality": "Critical", "base_daily": 45, "unit_cost": 210.0},
    {"name": "Paracetamol 500mg Tablet", "cold_chain": False, "criticality": "Essential", "base_daily": 350, "unit_cost": 2.5},
    {"name": "Amoxicillin 500mg Capsule", "cold_chain": False, "criticality": "Essential", "base_daily": 180, "unit_cost": 6.0},
    {"name": "Ringer Lactate 500ml IV", "cold_chain": False, "criticality": "Critical", "base_daily": 120, "unit_cost": 48.0},
]

TRANSIT_MATRIX = {
    ("WH_01", "HOSP_A"): (0.5, 4.0),
    ("HOSP_B", "HOSP_A"): (0.5, 5.5),
    ("HOSP_C", "HOSP_A"): (1.5, 14.0),
    ("HOSP_D", "HOSP_A"): (2.0, 18.0),
    ("WH_01", "HOSP_B"): (0.5, 4.5),
    ("WH_01", "HOSP_C"): (1.5, 13.5),
    ("WH_01", "HOSP_D"): (2.0, 17.5),
    ("HOSP_A", "HOSP_B"): (0.5, 5.5),
    ("HOSP_B", "HOSP_C"): (1.5, 14.0),
    ("HOSP_C", "HOSP_D"): (2.5, 21.0),
}

def generate_network_data(seed: int = 42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end="2026-11-16", periods=120, freq="D")

    # 1. Historical 120-day consumption time-series
    demand_rows = []
    for fac in FACILITIES:
        scale = 1.0 if fac["type"] == "Hospital" else 0.3
        for med in MEDICINES:
            weekly_wave = np.sin(np.arange(120) * 2 * np.pi / 7.0) * 0.15
            noise = rng.normal(0, 0.12, size=120)
            daily_qty = np.maximum(5, np.round(med["base_daily"] * scale * (1 + weekly_wave + noise))).astype(int)
            for d, q in zip(dates, daily_qty):
                demand_rows.append({
                    "date": d,
                    "facility_id": fac["facility_id"],
                    "facility_name": fac["name"],
                    "medicine": med["name"],
                    "consumption": int(q),
                })
    demand_df = pd.DataFrame(demand_rows)

    # 2. Batch-level FEFO inventory (with intentional disruption scenario on Ceftriaxone at HOSP_A)
    batch_rows = []
    b_idx = 100
    for fac in FACILITIES:
        for med in MEDICINES:
            b_idx += 1
            if fac["facility_id"] == "HOSP_A" and med["name"] == "Ceftriaxone 1g Injection":
                # Simulated 7-day supplier delay leaves only 220 units (~3.1 days cover)
                batch_rows.append({
                    "batch_id": f"B-{b_idx}", "facility_id": fac["facility_id"], "facility_name": fac["name"],
                    "medicine": med["name"], "qty": 220, "days_to_expiry": 95,
                    "cold_chain": med["cold_chain"], "criticality": med["criticality"]
                })
            elif fac["facility_id"] in ("HOSP_B", "WH_01") and med["name"] == "Ceftriaxone 1g Injection":
                # Healthy surplus + one short-expiry batch (20 days) that FEFO solver must reject for transfer
                batch_rows.append({
                    "batch_id": f"B-{b_idx}A", "facility_id": fac["facility_id"], "facility_name": fac["name"],
                    "medicine": med["name"], "qty": 1150, "days_to_expiry": 140,
                    "cold_chain": med["cold_chain"], "criticality": med["criticality"]
                })
                batch_rows.append({
                    "batch_id": f"B-{b_idx}B", "facility_id": fac["facility_id"], "facility_name": fac["name"],
                    "medicine": med["name"], "qty": 180, "days_to_expiry": 20,
                    "cold_chain": med["cold_chain"], "criticality": med["criticality"]
                })
            else:
                qty = int(med["base_daily"] * rng.uniform(8, 18))
                exp = int(rng.integers(45, 240))
                batch_rows.append({
                    "batch_id": f"B-{b_idx}", "facility_id": fac["facility_id"], "facility_name": fac["name"],
                    "medicine": med["name"], "qty": qty, "days_to_expiry": exp,
                    "cold_chain": med["cold_chain"], "criticality": med["criticality"]
                })
    batches_df = pd.DataFrame(batch_rows)
    return demand_df, batches_df
