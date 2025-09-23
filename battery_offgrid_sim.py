# battery_offgrid_sim.py
from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import csv
import re

@dataclass
class BatteryConfig:
    capacity_kwh: float
    power_kw: float
    charge_power_kw: Optional[float] = None
    discharge_power_kw: Optional[float] = None
    eta_c: float = 0.96
    eta_d: float = 0.96
    soc_min_frac: float = 0.10
    soc_init_frac: float = 0.50
    def normalized(self) -> "BatteryConfig":
        charge_p = self.charge_power_kw if self.charge_power_kw is not None else self.power_kw
        discharge_p = self.discharge_power_kw if self.discharge_power_kw is not None else self.power_kw
        return BatteryConfig(
            capacity_kwh=float(self.capacity_kwh),
            power_kw=float(self.power_kw),
            charge_power_kw=float(charge_p),
            discharge_power_kw=float(discharge_p),
            eta_c=float(np.clip(self.eta_c, 1e-6, 1.0)),
            eta_d=float(np.clip(self.eta_d, 1e-6, 1.0)),
            soc_min_frac=float(np.clip(self.soc_min_frac, 0.0, 0.99)),
            soc_init_frac=float(np.clip(self.soc_init_frac, self.soc_min_frac, 0.999)),
        )

def _infer_datetime_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    cols = {c.lower(): c for c in df.columns}
    if "timestamp" in cols or "datetime" in cols:
        col = cols.get("timestamp", cols.get("datetime"))
        return pd.to_datetime(df[col], errors="raise", utc=False)
    if all(k in cols for k in ["month", "day", "hour"]):
        year = 2024
        ts = pd.to_datetime(dict(year=year, month=df[cols["month"]], day=df[cols["day"]], hour=df[cols["hour"]]), errors="raise")
        return pd.DatetimeIndex(ts)
    raise ValueError("Provide 'timestamp' or PVWatts Month/Day/Hour columns.")

def _normalize_col(name: str) -> str:
    # Lowercase, strip quotes/whitespace/BOM, remove non-alnum characters
    if name is None:
        return ""
    s = name.replace("\ufeff", "").strip().strip('"').strip("'").lower()
    s = re.sub(r"[^a-z0-9]+", "", s)  # keep only letters/numbers
    return s

def _find_header_row_and_fields(path: Path):
    """Return (header_row_index, header_fields_as_list) for PVWatts, else (None, None)."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            s = line.strip()
            # tolerant detection (any quoting/spacing)
            if ("month" in s.lower()) and ("day" in s.lower()) and ("hour" in s.lower()):
                # Parse the actual header fields with csv module to preserve exact names
                reader = csv.reader([line])
                fields = next(reader, [])
                return i, fields
    return None, None

def read_pv_timeseries(path: Path) -> pd.DataFrame:
    """
    Robust PV reader for:
      - PVWatts exports with ragged preamble
      - Simple CSVs (timestamp + pv column)

    Returns index=timestamp, column 'pv_kw' (kW).
    """
    header_row, header_fields = _find_header_row_and_fields(path)

    if header_row is not None:
        # 1) First attempt: read using header=header_row
        try:
            df = pd.read_csv(
                path,
                header=header_row,
                engine="python",
                quotechar='"',
                skip_blank_lines=True,
            )
        except Exception:
            df = None

        # 2) Normalize columns and try to locate Month/Day/Hour + AC/DC fields
        def locate_cols(df_try: pd.DataFrame):
            norm = {_normalize_col(c): c for c in df_try.columns}
            # map keys we need
            month_col = norm.get("month")
            day_col   = norm.get("day")
            hour_col  = norm.get("hour")

            ac_cand = None
            dc_cand = None
            for k, orig in norm.items():
                if k.startswith("acsystemoutput") or k.endswith("acsystemoutputw"):
                    ac_cand = orig
                if k.startswith("dcarrayoutput") or k.endswith("dcarrayoutputw"):
                    dc_cand = dc_cand or orig
            return month_col, day_col, hour_col, ac_cand, dc_cand

        if df is not None:
            mcol, dcol, hcol, acc, dcc = locate_cols(df)
            if mcol and dcol and hcol:
                year = 2024
                idx = pd.to_datetime(
                    dict(year=year, month=df[mcol], day=df[dcol], hour=df[hcol]),
                    errors="raise",
                )
                if acc is not None:
                    pv_kw = pd.to_numeric(df[acc], errors="coerce").fillna(0.0) / 1000.0
                elif dcc is not None:
                    pv_kw = pd.to_numeric(df[dcc], errors="coerce").fillna(0.0) / 1000.0
                else:
                    raise ValueError("PVWatts file missing AC/DC output columns after normalization.")
                out = pd.DataFrame({"pv_kw": pv_kw.values}, index=pd.DatetimeIndex(idx)).sort_index()
                out.index.name = "timestamp"
                return out
            # If we got here, we found the header row but pandas didn't align columns as expected.

        # 3) Fallback: re-read with header=None and explicit names from the *actual* header line
        #    This avoids pandas trying to “helpfully” shift columns when the header row has odd quoting.
        names = [c.strip() for c in header_fields]
        df2 = pd.read_csv(
            path,
            header=None,
            names=names,
            skiprows=header_row + 1,
            engine="python",
            quotechar='"',
            skip_blank_lines=True,
        )

        # Normalize and locate columns again
        # Some PVWatts exports will repeat the header within data; drop rows where Month/Day/Hour aren't numeric
        # to be extra safe.
        def is_numeric_series(s):
            try:
                pd.to_numeric(s)
                return True
            except Exception:
                return False

        # Keep rows where Month/Day/Hour look numeric after normalization/mapping
        norm2 = {_normalize_col(c): c for c in df2.columns}
        mcol = norm2.get("month"); dcol = norm2.get("day"); hcol = norm2.get("hour")
        if not (mcol and dcol and hcol):
            raise ValueError(
                f"PVWatts header row found, but Month/Day/Hour columns are missing. "
                f"Columns seen (normalized→original): {norm2}"
            )

        # filter out non-numeric header repeats if any
        df2 = df2[pd.to_numeric(df2[mcol], errors="coerce").notna() &
                  pd.to_numeric(df2[dcol], errors="coerce").notna() &
                  pd.to_numeric(df2[hcol], errors="coerce").notna()]

        year = 2024
        idx = pd.to_datetime(
            dict(year=year, month=pd.to_numeric(df2[mcol]),
                 day=pd.to_numeric(df2[dcol]), hour=pd.to_numeric(df2[hcol])),
            errors="raise",
        )

        # Prefer AC, else DC
        acc = None; dcc = None
        for k, orig in norm2.items():
            if k.startswith("acsystemoutput") or k.endswith("acsystemoutputw"):
                acc = orig
            if k.startswith("dcarrayoutput") or k.endswith("dcarrayoutputw"):
                dcc = dcc or orig

        if acc is not None:
            pv_kw = pd.to_numeric(df2[acc], errors="coerce").fillna(0.0) / 1000.0
        elif dcc is not None:
            pv_kw = pd.to_numeric(df2[dcc], errors="coerce").fillna(0.0) / 1000.0
        else:
            raise ValueError(
                f"PVWatts file missing AC/DC output columns. Available columns: {list(df2.columns)}"
            )

        out = pd.DataFrame({"pv_kw": pv_kw.values}, index=pd.DatetimeIndex(idx)).sort_index()
        out.index.name = "timestamp"
        return out

    # ---------- Simple CSV branch ----------
    try:
        df = pd.read_csv(path, engine="python")
    except Exception as e:
        raise ValueError(
            f"Failed to parse CSV. If this is a PVWatts export, it must contain a header line with Month,Day,Hour. "
            f"Original error: {e}"
        ) from e

    idx = _infer_datetime_index(df)
    cand = [c for c in df.columns if c.lower() in ("pv_kw", "pv", "ac_kw", "power_kw", "power")]
    if not cand:
        if df.shape[1] >= 2:
            cand = [df.columns[1]]
        else:
            raise ValueError("Simple PV CSV must include a PV power column, e.g. 'pv_kw'.")
    pv_col = cand[0]
    pv_kw = pd.to_numeric(df[pv_col], errors="coerce").fillna(0.0)
    out = pd.DataFrame({"pv_kw": pv_kw.values}, index=pd.DatetimeIndex(idx)).sort_index()
    out.index.name = "timestamp"
    return out

def read_load_timeseries(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    idx = _infer_datetime_index(df)
    cand = [c for c in df.columns if c.lower() in ("load_kw","load","kw","power_kw","demand_kw", "total_load_kw")]
    if not cand:
        if df.shape[1] >= 2:
            cand = [df.columns[1]]
        else:
            raise ValueError("Load CSV must include a load column.")
    load_col = cand[0]
    load_kw = pd.to_numeric(df[load_col], errors="coerce").fillna(0.0)
    return pd.DataFrame({"load_kw": load_kw.values}, index=idx).sort_index()

def _time_step_hours(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        raise ValueError("Need at least 2 timestamps.")
    dt = np.median(np.diff(index.view(np.int64)))  # ns
    return float(dt / 1e9 / 3600.0)

def _max_contiguous_true(arr_bool: np.ndarray) -> int:
    if len(arr_bool) == 0: return 0
    x = np.array(arr_bool, dtype=int)
    d = np.diff(np.concatenate(([0], x, [0])))
    starts = np.where(d == 1)[0]; ends = np.where(d == -1)[0]
    if len(starts) == 0: return 0
    return int((ends - starts).max())

def add_fixed_load(load_kw: pd.Series, fixed_load_kw: float) -> pd.Series:
    """Add a fixed 24/7/365 load to the existing load timeseries.

    Args:
        load_kw: Original load timeseries
        fixed_load_kw: Fixed load to add (kW)

    Returns:
        Load timeseries with fixed load added
    """
    return load_kw + fixed_load_kw



def simulate_offgrid_dispatch(
    pv_kw: pd.Series,
    load_kw: pd.Series,
    cfg: BatteryConfig,
    fixed_load_kw: float = 0.0,
    resample_to: str | None = None,     # pandas freq string to force (e.g., "30min"); if None, infer finer step
    interp: str = "linear"              # "linear" or "ffill" for PV interpolation when upsampling
) -> Tuple[pd.DataFrame, dict]:
    """
    Off-grid dispatch with automatic time alignment + diagnostics.
    - If resample_to is None: detect finer step and resample both series to it.
    - If resample_to is given (e.g., "30min" or "1H"), resample both series to that explicit grid.
    - PV interpolation: "linear" (default) or "ffill" (step-wise)
    Diagnostics are added to KPIs to verify alignment.
    """
    cfg = cfg.normalized()

    # Determine time coverage
    start = min(pv_kw.index.min(), load_kw.index.min())
    end   = max(pv_kw.index.max(), load_kw.index.max())

    def _infer_dt_seconds_safe(idx: pd.DatetimeIndex) -> float | None:
        if len(idx) < 2:
            return None
        dt_ns = float(np.median(np.diff(idx.view("int64"))))
        return dt_ns / 1e9

    pv_dt = _infer_dt_seconds_safe(pv_kw.index)
    ld_dt = _infer_dt_seconds_safe(load_kw.index)

    # Choose target frequency
    if resample_to is None:
        candidate = [x for x in [pv_dt, ld_dt] if x is not None and x > 0]
        target_sec = min(candidate) if candidate else 3600.0
        sec_rounded = int(round(target_sec))
        if sec_rounded % 3600 == 0:
            freq = f"{sec_rounded // 3600}H"
        elif sec_rounded % 60 == 0:
            freq = f"{sec_rounded // 60}min"
        else:
            freq = f"{sec_rounded}S"
    else:
        freq = resample_to

    idx = pd.date_range(start=start, end=end, freq=freq)

    # Pre-diagnostics (means at original grids)
    pv_mean_before = float(pv_kw.mean()) if len(pv_kw) else float("nan")
    ld_mean_before = float(load_kw.mean()) if len(load_kw) else float("nan")

    # PV alignment
    pv_aligned = pv_kw.reindex(idx)
    if interp == "ffill":
        pv_aligned = pv_aligned.ffill().bfill()
    else:
        pv_aligned = pv_aligned.interpolate(limit_direction="both")
    pv_aligned = pv_aligned.fillna(0.0).clip(lower=0.0)

    # Load alignment
    ld_aligned = load_kw.reindex(idx).fillna(0.0).clip(lower=0.0)

    # Fixed load
    ld_aligned = ld_aligned + float(fixed_load_kw)

    # Step for energy
    if len(idx) < 2:
        raise ValueError("Index after alignment has <2 points.")
    dt_hours = (idx[1] - idx[0]).total_seconds() / 3600.0

    # Post-diagnostics (means at target grid)
    pv_mean_after = float(pv_aligned.mean())
    ld_mean_after = float(ld_aligned.mean())

    # Sim
    n = len(idx)
    soc_kwh = np.zeros(n); charge_kw = np.zeros(n); discharge_kw = np.zeros(n)
    curtailed_kw = np.zeros(n); unmet_kw = np.zeros(n)
    soc_min = cfg.soc_min_frac * cfg.capacity_kwh
    soc_max = cfg.capacity_kwh
    soc = cfg.soc_init_frac * cfg.capacity_kwh

    for t in range(n):
        pv_t = max(float(pv_aligned.iloc[t]), 0.0)
        load_t = max(float(ld_aligned.iloc[t]), 0.0)
        surplus = pv_t - load_t
        if surplus > 0:
            cap_room_kwh = soc_max - soc
            max_charge_from_room_kw = cap_room_kwh / (cfg.eta_c * dt_hours) if dt_hours > 0 else 0.0
            p_charge = min(surplus, cfg.charge_power_kw, max_charge_from_room_kw)
            soc += p_charge * cfg.eta_c * dt_hours
            charge_kw[t] = p_charge
            curtailed_kw[t] = max(surplus - p_charge, 0.0)
            unmet_kw[t] = 0.0
        else:
            deficit = -surplus
            energy_above_min_kwh = max(soc - soc_min, 0.0)
            max_discharge_from_soc_kw = (energy_above_min_kwh * cfg.eta_d / dt_hours) if dt_hours > 0 else 0.0
            p_dis = min(deficit, cfg.discharge_power_kw, max_discharge_from_soc_kw)
            soc -= (p_dis * dt_hours) / cfg.eta_d
            discharge_kw[t] = p_dis
            unmet_kw[t] = max(deficit - p_dis, 0.0)
            curtailed_kw[t] = 0.0
        soc = float(np.clip(soc, soc_min, soc_max))
        soc_kwh[t] = soc

    # KPIs
    pv = pv_aligned
    ld = ld_aligned
    load_energy = float((ld * dt_hours).sum())
    pv_energy = float((pv * dt_hours).sum())
    curtail_energy = float((pd.Series(curtailed_kw, index=idx) * dt_hours).sum())
    ens_kwh = float((pd.Series(unmet_kw, index=idx) * dt_hours).sum())
    thru_charge = float((pd.Series(charge_kw, index=idx) * dt_hours).sum())
    thru_discharge = float((pd.Series(discharge_kw, index=idx) * dt_hours).sum())
    effective_throughput = min(thru_charge * cfg.eta_c, thru_discharge / cfg.eta_d)
    cycles = effective_throughput / (2.0 * cfg.capacity_kwh) if cfg.capacity_kwh > 0 else np.nan

    kpis = {
        # Core
        #"time_step_hours": dt_hours,
        "total_load_kwh": np.round(load_energy, 2),
        "total_pv_kwh": np.round(pv_energy,2),
        "pv_curtailed_kwh": np.round(curtail_energy,2),
        "pv_used_kwh": np.round(pv_energy - curtail_energy,2),
        "energy_not_served_kwh": np.round(ens_kwh, 2),
        "self_sufficiency_percent": np.round((1.0 - ens_kwh / load_energy),2) * 100.0 if load_energy > 0 else np.nan,
        "pv_utilization_percent": np.round(((pv_energy - curtail_energy) / pv_energy),2) * 100.0 if pv_energy > 0 else np.nan,
        #"avg_soc_fraction": float(np.mean(soc_kwh) / cfg.capacity_kwh) if cfg.capacity_kwh > 0 else np.nan,
        #"min_soc_fraction": float(np.min(soc_kwh) / cfg.capacity_kwh) if cfg.capacity_kwh > 0 else np.nan,
        #"max_soc_fraction": float(np.max(soc_kwh) / cfg.capacity_kwh) if cfg.capacity_kwh > 0 else np.nan,
        "throughput_charge_kwh": np.round(thru_charge, 2),
        "throughput_discharge_kwh": np.round(thru_discharge, 2),
        "estimated_full_cycles": np.round(cycles,2),
        "loss_of_load_hours": np.round(int((pd.Series(unmet_kw, index=idx) > 1e-9).sum()),2),
        "max_contiguous_lol_hours": np.round(int(_max_contiguous_true((pd.Series(unmet_kw, index=idx) > 1e-9).values) * dt_hours),2),

        # Diagnostics
        "diag_pv_points": int(len(pv_kw)),
        "diag_load_points": int(len(load_kw)),
        "diag_pv_step_seconds": float(pv_dt) if pv_dt is not None else None,
        "diag_load_step_seconds": float(ld_dt) if ld_dt is not None else None,
        "diag_target_freq": str(freq),
        "diag_target_step_hours": float(dt_hours),
        "diag_pv_mean_before": pv_mean_before,
        "diag_pv_mean_after": pv_mean_after,
        "diag_load_mean_before": ld_mean_before,
        "diag_load_mean_after": ld_mean_after,
        "diag_interp": interp,
    }
    return kpis
    
    '''
    ts = pd.DataFrame({
        "pv_kw": pv.values,
        "load_kw": ld.values,
        "net_load_kw": (ld - pv).values,
        "charge_kw": charge_kw,
        "discharge_kw": discharge_kw,
        "curtailed_kw": curtailed_kw,
        "unmet_kw": unmet_kw,
        "soc_kwh": soc_kwh,
        "soc_frac": soc_kwh / cfg.capacity_kwh if cfg.capacity_kwh > 0 else np.nan,
    }, index=idx)
    ts.index.name = "timestamp"
    return ts, kpis
    '''
    
