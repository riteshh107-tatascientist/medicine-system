import pandas as pd
from ortools.linear_solver import pywraplp
from src.data_generator import FACILITIES, TRANSIT_MATRIX

def run_fefo_redistribution(batches_df: pd.DataFrame, demand_df: pd.DataFrame, min_shelf_life_on_arrival: int = 30):
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if not solver:
        raise RuntimeError("SCIP solver unavailable in OR-Tools installation.")

    fac_meta = {f["facility_id"]: f for f in FACILITIES}
    avg_demand = (
        demand_df.groupby(["facility_id", "medicine"])["consumption"]
        .mean()
        .to_dict()
    )

    medicines = batches_df["medicine"].unique().tolist()
    fac_ids = [f["facility_id"] for f in FACILITIES]

    transfer_vars = {}
    trip_vars = {}
    unmet_vars = {}

    for med in medicines:
        med_batches = batches_df[batches_df["medicine"] == med]
        req_cold = bool(med_batches["cold_chain"].iloc[0])

        # Compute transferable surplus & shortage deficit per facility
        surplus = {}
        deficit = {}
        days_cover = {}

        for fid in fac_ids:
            fb = med_batches[med_batches["facility_id"] == fid]
            total_usable = int(fb[fb["days_to_expiry"] > 0]["qty"].sum())
            d_rate = float(avg_demand.get((fid, med), 50.0))
            cover = total_usable / max(d_rate, 1.0)
            days_cover[fid] = cover

            safety_units = int(round(d_rate * fac_meta[fid]["safety_days"]))
            # FEFO rule: only batches with >= min_shelf_life_on_arrival + transit buffer can be donated
            eligible_donor_qty = int(fb[fb["days_to_expiry"] >= (min_shelf_life_on_arrival + 3)]["qty"].sum())
            surplus[fid] = max(0, min(eligible_donor_qty, total_usable - safety_units))

            target_units = int(round(d_rate * 10.0))  # 10-day resilience target
            deficit[fid] = max(0, target_units - total_usable) if cover < 5.0 else 0

        for dst in fac_ids:
            if deficit[dst] <= 0:
                continue
            unmet_vars[(med, dst)] = solver.NumVar(0, deficit[dst], f"unmet_{med}_{dst}")
            incoming = []
            for src in fac_ids:
                if src == dst or surplus[src] <= 0:
                    continue
                if (src, dst) not in TRANSIT_MATRIX:
                    continue
                eta, unit_cost = TRANSIT_MATRIX[(src, dst)]
                # Hard rule 1: cold-chain compatibility
                if req_cold and (not fac_meta[src]["cold_chain"] or not fac_meta[dst]["cold_chain"]):
                    continue
                # Hard rule 2: shipment must arrive BEFORE destination stockout day
                if eta >= days_cover[dst]:
                    continue

                max_q = min(surplus[src], deficit[dst])
                x = solver.IntVar(0, max_q, f"x_{med}_{src}_{dst}")
                y = solver.BoolVar(f"trip_{med}_{src}_{dst}")
                solver.Add(x <= max_q * y)
                transfer_vars[(med, src, dst)] = (x, unit_cost, eta)
                trip_vars[(med, src, dst)] = y
                incoming.append(x)

            solver.Add(sum(incoming) + unmet_vars[(med, dst)] >= deficit[dst])

        # Donor capacity constraint across all destinations
        for src in fac_ids:
            outgoing = [transfer_vars[(med, src, d)][0] for d in fac_ids if (med, src, d) in transfer_vars]
            if outgoing:
                solver.Add(sum(outgoing) <= surplus[src])

    # Objective: Minimize transport + fixed dispatch cost + high penalty for unmet critical demand
    obj = solver.Objective()
    for key, (x, unit_cost, _) in transfer_vars.items():
        obj.SetCoefficient(x, unit_cost)
    for key, y in trip_vars.items():
        obj.SetCoefficient(y, 450.0)  # Fixed refrigerated/courier trip fee (INR)
    for key, u in unmet_vars.items():
        obj.SetCoefficient(u, 5000.0)  # Heavy penalty for leaving a hospital in stockout
    obj.SetMinimization()

    status = solver.Solve()
    status_map = {pywraplp.Solver.OPTIMAL: "OPTIMAL", pywraplp.Solver.FEASIBLE: "FEASIBLE"}

    actions = []
    for (med, src, dst), (x, unit_cost, eta) in transfer_vars.items():
        qty = int(round(x.solution_value()))
        if qty > 0:
            actions.append({
                "Action ID": f"REC-{len(actions)+101}",
                "Medicine": med,
                "From (Donor)": fac_meta[src]["name"],
                "To (Deficit)": fac_meta[dst]["name"],
                "Transfer Qty": qty,
                "ETA (Days)": eta,
                "Est. Cost (INR)": round(qty * unit_cost + 450.0, 2),
                "FEFO Guard": f">= {min_shelf_life_on_arrival}d Shelf Life Verified",
                "Governance": "PENDING HUMAN APPROVAL",
            })

    return status_map.get(status, "NO_FEASIBLE_PLAN"), pd.DataFrame(actions)
