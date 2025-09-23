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
    fixed_load_kw: float = 0.0,
    resample_to: str | None = None,
    interp: str = "linear",
    out_file: str = "capacity_comparison.csv",
):
    pv = read_pv_timeseries(Path(pv_path))
    ld = read_load_timeseries(Path(load_path))

    results = []
    # Tiny diagnostic: report detected native steps
    def infer_step_hours(idx):
        if len(idx) < 2: return None
        import numpy as np
        dt = float(np.median(np.diff(idx.view("int64")))) / 1e9 / 3600.0
        return dt
    print(f"[diag] PV native step (h): {infer_step_hours(pv.index)}; Load native step (h): {infer_step_hours(ld.index)}")
    print(f"[diag] Forcing resample_to={resample_to!r}, interp={interp!r}")

    for cap in range(int(capacity_start), int(capacity_end) + 1, int(capacity_step)):
        cfg = BatteryConfig(
            capacity_kwh=float(cap),
            power_kw=power_kw,
            soc_min_frac=0.1,
            soc_init_frac=0.5,
            eta_c=0.96,
            eta_d=0.96,
        )
        kpis = simulate_offgrid_dispatch(
            pv["pv_kw"], ld["load_kw"], cfg,
            fixed_load_kw=fixed_load_kw,
            resample_to=resample_to,
            interp=interp,
        )
        kpis["capacity_kwh"] = cap
        kpis["power_kw"] = power_kw
        results.append(kpis)

    df = pd.DataFrame(results)
    df.to_csv(out_file, index=False)
    print(f"Saved comparison results to {out_file}")
    print("[diag] Columns:", list(df.columns)[:12], "...")
    print(df[["capacity_kwh","total_load_kwh","total_pv_kwh","pv_curtailed_kwh","pv_used_kwh","diag_target_freq"]].head())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sweep battery capacities and record KPIs.")
    parser.add_argument("--pv", required=True, help="Path to PV CSV (simple or PVWatts)")
    parser.add_argument("--load", required=True, help="Path to load CSV")
    parser.add_argument("--capacity-start", type=float, required=True)
    parser.add_argument("--capacity-end", type=float, required=True)
    parser.add_argument("--capacity-step", type=float, required=True)
    parser.add_argument("--power-kw", type=float, required=True)
    parser.add_argument("--fixed-load-kw", type=float, default=0.0, help="Optional 24/7 fixed load (kW)")
    parser.add_argument("--resample", default=None, help="Force pandas freq (e.g., '30min' or '1H'). If omitted, auto-infer finer step.")
    parser.add_argument("--interp", default="linear", choices=["linear","ffill"], help="PV interpolation when upsampling.")
    parser.add_argument("--out-file", default="capacity_comparison.csv")

    args = parser.parse_args()

    run_capacity_sweep(
        pv_path=args.pv,
        load_path=args.load,
        capacity_start=args.capacity_start,
        capacity_end=args.capacity_end,
        capacity_step=args.capacity_step,
        power_kw=args.power_kw,
        fixed_load_kw=args.fixed_load_kw,
        resample_to=args.resample,
        interp=args.interp,
        out_file=args.out_file,
    )
