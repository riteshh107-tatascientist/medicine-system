# MediResQ AI — Cross-Hospital Essential Medicine Supply Chain Resilience Network

**SAP Hackfest 2026 | Technocrats Institute of Technology (TIT), Bhopal**  
**Team:** Ritesh Kumar Singh (Leader), Rohit Basor, Maneesh Kushwaha, Chitranshu Bhikonde

## Why This Project Is Unique (Not a Single-Hospital App)
Standard hospital inventory apps only count stock inside **one isolated pharmacy** and alert staff *after* medicines run out. **MediResQ AI** connects multiple hospitals and central warehouses into a unified predictive resilience network:
1. **30-Day ML Demand Forecasting** evaluated on a strict 28-day chronological hold-out (no future data leakage).
2. **FEFO (First-Expired, First-Out) Batch & Expiry Engine** ensuring short-shelf-life batches are never transferred across facilities.
3. **Google OR-Tools SCIP Mixed-Integer Programming (MIP) Solver** computing cost-optimal, cold-chain-safe transfers before stockout occurs.
4. **Human-in-the-Loop Governance** requiring mandatory manager approval before any action executes.

## Quick Start
```bash
pip install -r requirements.txt
streamlit run app.py
```
