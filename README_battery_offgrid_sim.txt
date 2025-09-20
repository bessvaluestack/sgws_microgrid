Battery Off-Grid Simulation — Quick Start

1) Prepare inputs
   - PV CSV (either simple two-column: timestamp,pv_kw) OR a PVWatts export CSV.
   - Load CSV (timestamp,load_kw). Timestamp can be any pandas-parsable format.

2) Use from Python
   from pathlib import Path
   import pandas as pd
   from battery_offgrid_sim import read_pv_timeseries, read_load_timeseries, BatteryConfig, simulate_offgrid_dispatch

   pv = read_pv_timeseries(Path("pv.csv"))
   ld = read_load_timeseries(Path("load.csv"))
   cfg = BatteryConfig(capacity_kwh=800, power_kw=200, soc_min_frac=0.1, soc_init_frac=0.5, eta_c=0.96, eta_d=0.96)
   ts, kpis = simulate_offgrid_dispatch(pv["pv_kw"], ld["load_kw"], cfg)
   ts.to_csv("offgrid.timeseries.csv")
   pd.Series(kpis).to_json("offgrid.kpis.json", indent=2)

3) Notes
   - The simulator charges from PV surplus and discharges to meet deficits.
   - Power and SoC limits and round-trip efficiency are enforced (separate eta_c / eta_d).
   - Any remaining deficit after discharging is recorded as unmet load (loss of load).
   - Excess PV after charging is recorded as curtailed.
   - KPIs include ENS (kWh), Loss-of-Load Hours, PV utilization, and estimated cycle count.
