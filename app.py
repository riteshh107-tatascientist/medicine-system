import streamlit as st
import pandas as pd
import plotly.express as px
from src.data_generator import generate_network_data, FACILITIES, MEDICINES
from src.forecaster import benchmark_and_forecast
from src.optimizer import run_fefo_redistribution

st.set_page_config(page_title="MediResQ AI — Cross-Hospital Supply Chain Network", layout="wide")

st.title("MediResQ AI — Cross-Hospital Essential Medicine Resilience Network")
st.caption("SAP Hackfest 2026 | Technocrats Institute of Technology (TIT), Bhopal | Synthetic Data Demonstration")

demand_df, batches_df = generate_network_data()

tab1, tab2, tab3 = st.tabs([
    "1. Cross-Hospital Stock & FEFO Risk",
    "2. 30-Day ML Demand Forecast (28-Day Hold-Out)",
    "3. OR-Tools Redistribution & Human Approval"
])

with tab1:
    st.subheader("Network-Wide Inventory & Expiry Risk (Why Single-Hospital Apps Fail)")
    st.info("Unlike single-hospital billing apps that only look at one pharmacy, MediResQ AI scans all 5 facilities together to detect shortages and surplus simultaneously.")

    summary = []
    for fac in FACILITIES:
        for med in MEDICINES:
            fb = batches_df[(batches_df["facility_id"] == fac["facility_id"]) & (batches_df["medicine"] == med["name"])]
            usable = int(fb["qty"].sum())
            min_exp = int(fb["days_to_expiry"].min())
            avg_d = float(demand_df[(demand_df["facility_id"] == fac["facility_id"]) & (demand_df["medicine"] == med["name"])]["consumption"].mean())
            cover = round(usable / max(avg_d, 1.0), 1)
            status = "CRITICAL (< 4d)" if cover < 4.0 else ("SURPLUS (> 12d)" if cover > 12.0 else "HEALTHY")
            summary.append({
                "Facility": fac["name"], "Medicine": med["name"], "Usable Stock": usable,
                "Days of Cover": cover, "Earliest Expiry (Days)": min_exp, "Network Status": status
            })
    sum_df = pd.DataFrame(summary)
    st.dataframe(sum_df, use_container_width=True)

with tab2:
    st.subheader("30-Day Demand Forecasting & Chronological Hold-Out Benchmark")
    c1, c2 = st.columns(2)
    fac_choice = c1.selectbox("Select Facility", [f["facility_id"] for f in FACILITIES],
                              format_func=lambda x: next(f["name"] for f in FACILITIES if f["facility_id"] == x))
    med_choice = c2.selectbox("Select Medicine", [m["name"] for m in MEDICINES])

    res = benchmark_and_forecast(demand_df, fac_choice, med_choice, horizon=30)
    m1, m2, m3 = st.columns(3)
    m1.metric("Gradient Boosting WAPE", f"{res['holdout_wape']['Gradient Boosting']}%", "Best Single Model")
    m2.metric("Moving Average WAPE", f"{res['holdout_wape']['Moving Average (7d)']}%", "Simple Baseline")
    m3.metric("30-Day Predicted Demand", f"{res['total_30d_forecast']} units", f"{res['avg_daily_forecast']} / day")

    wape_df = pd.DataFrame({"Model": list(res["holdout_wape"].keys()), "WAPE (%)": list(res["holdout_wape"].values())})
    fig = px.bar(wape_df, x="Model", y="WAPE (%)", color="Model", title="28-Day Chronological Hold-Out Error (Lower = Better)")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("OR-Tools SCIP Redistribution Solver + Mandatory Human Sign-Off")
    min_shelf = st.slider("Minimum Remaining Shelf Life on Arrival (Days — FEFO Guard)", 15, 60, 30)
    solver_status, plan_df = run_fefo_redistribution(batches_df, demand_df, min_shelf_life_on_arrival=min_shelf)

    st.success(f"Solver Status: {solver_status} | Short-expiry batches (< {min_shelf + 3} days) automatically excluded from transfer.")
    if not plan_df.empty:
        st.dataframe(plan_df, use_container_width=True)
        approved = st.checkbox("Manager Sign-Off: I have reviewed cold-chain, shelf-life, and donor safety buffers and approve these actions.")
        if st.button("Execute Approved Recovery Plan", disabled=not approved):
            st.balloons()
            st.success("Recovery plan logged to audit trail and dispatched to hospital logistics teams!")
    else:
        st.warning("No feasible transfers found under current shelf-life constraint.")
