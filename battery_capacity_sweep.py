from pathlib import Path
import pandas as pd
from battery_offgrid_sim import read_pv_timeseries, read_load_timeseries, BatteryConfig, simulate_offgrid_dispatch

def run_capacity_sweep(
    pv_path: str,
    load_path: str,
    capacity_start: float,
    capacity_end: float,
    capacity_step: float,
    power_kw: float,
    out_file: str = "capacity_comparison.csv",
):
    pv = read_pv_timeseries(Path(pv_path))
    ld = read_load_timeseries(Path(load_path))

    results = []

    for cap in range(int(capacity_start), int(capacity_end) + 1, int(capacity_step)):
        cfg = BatteryConfig(
            capacity_kwh=float(cap),
            power_kw=power_kw,
            soc_min_frac=0.1,
            soc_init_frac=0.5,
            eta_c=0.96,
            eta_d=0.96,
        )
        ts, kpis = simulate_offgrid_dispatch(pv["pv_kw"], ld["load_kw"], cfg)
        kpis["capacity_kwh"] = cap
        kpis["power_kw"] = power_kw
        results.append(kpis)

    df = pd.DataFrame(results)
    df.to_csv(out_file, index=False)
    print(f"Saved comparison results to {out_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sweep battery capacities and record KPIs.")
    parser.add_argument("--pv", required=True, help="Path to PV CSV (simple or PVWatts)")
    parser.add_argument("--load", required=True, help="Path to load CSV")
    parser.add_argument("--capacity-start", type=float, required=True)
    parser.add_argument("--capacity-end", type=float, required=True)
    parser.add_argument("--capacity-step", type=float, required=True)
    parser.add_argument("--power-kw", type=float, required=True)
    parser.add_argument("--out-file", default="capacity_comparison.csv")

    args = parser.parse_args()

    run_capacity_sweep(
        pv_path=args.pv,
        load_path=args.load,
        capacity_start=args.capacity_start,
        capacity_end=args.capacity_end,
        capacity_step=args.capacity_step,
        power_kw=args.power_kw,
        out_file=args.out_file,
    )
