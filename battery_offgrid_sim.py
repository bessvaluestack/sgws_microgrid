# battery_offgrid_sim.py
from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

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
        year = 2020
        ts = pd.to_datetime(dict(year=year, month=df[cols["month"]], day=df[cols["day"]], hour=df[cols["hour"]]), errors="raise")
        return pd.DatetimeIndex(ts)
    raise ValueError("Provide 'timestamp' or PVWatts Month/Day/Hour columns.")

def read_pv_timeseries(path: Path) -> pd.DataFrame:
    df_raw = pd.read_csv(path)
    is_pvwatts = "pvwatts" in " ".join(df_raw.columns.astype(str)).lower() or any(
        c.lower() in ("month","day","hour") for c in df_raw.columns
    )
    if is_pvwatts:
        if not {"Month","Day","Hour"}.issubset(set(df_raw.columns)):
            rows = pd.read_csv(path, header=None).astype(str)
            header_row = None
            for i in range(min(len(rows),100)):
                row_vals = [v.strip('"') for v in rows.iloc[i].tolist()]
                if {"Month","Day","Hour"}.issubset(set(row_vals)):
                    header_row = i; break
            if header_row is None:
                raise ValueError("PVWatts header row not found.")
            df = pd.read_csv(path, header=header_row)
        else:
            df = df_raw.copy()
        df.columns = [c.strip().strip('"') for c in df.columns]
        idx = _infer_datetime_index(df)
        ac_cols = [c for c in df.columns if c.lower().startswith("ac system output")]
        dc_cols = [c for c in df.columns if c.lower().startswith("dc array output")]
        if ac_cols:
            pv_kw = df[ac_cols[0]].astype(float) / 1000.0
        elif dc_cols:
            pv_kw = df[dc_cols[0]].astype(float) / 1000.0
        else:
            raise ValueError("PVWatts file missing AC/DC output columns.")
        return pd.DataFrame({"pv_kw": pv_kw.values}, index=idx).sort_index()
    else:
        df = df_raw.copy()
        idx = _infer_datetime_index(df)
        cand = [c for c in df.columns if c.lower() in ("pv_kw","pv","ac_kw","power_kw","power")]
        if not cand:
            if df.shape[1] >= 2:
                cand = [df.columns[1]]
            else:
                raise ValueError("Simple PV CSV must have power column (e.g., pv_kw).")
        pv_col = cand[0]
        pv_kw = pd.to_numeric(df[pv_col], errors="coerce").fillna(0.0)
        return pd.DataFrame({"pv_kw": pv_kw.values}, index=idx).sort_index()

def read_load_timeseries(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    idx = _infer_datetime_index(df)
    cand = [c for c in df.columns if c.lower() in ("load_kw","load","kw","power_kw","demand_kw")]
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

def simulate_offgrid_dispatch(pv_kw: pd.Series, load_kw: pd.Series, cfg: BatteryConfig) -> Tuple[pd.DataFrame, dict]:
    cfg = cfg.normalized()
    idx = pv_kw.index.union(load_kw.index).unique().sort_values()
    pv = pv_kw.reindex(idx).fillna(0.0).astype(float)
    ld = load_kw.reindex(idx).fillna(0.0).astype(float)
    dt_h = _time_step_hours(idx)
    n = len(idx)
    soc_kwh = np.zeros(n); charge_kw = np.zeros(n); discharge_kw = np.zeros(n); curtailed_kw = np.zeros(n); unmet_kw = np.zeros(n)
    soc_min = cfg.soc_min_frac * cfg.capacity_kwh; soc_max = cfg.capacity_kwh; soc = cfg.soc_init_frac * cfg.capacity_kwh
    for t in range(n):
        pv_t = max(pv.iloc[t], 0.0); load_t = max(ld.iloc[t], 0.0)
        surplus = pv_t - load_t
        if surplus > 0:
            cap_room_kwh = soc_max - soc
            max_charge_from_room_kw = cap_room_kwh / (cfg.eta_c * dt_h) if dt_h > 0 else 0.0
            p_charge = min(surplus, cfg.charge_power_kw, max_charge_from_room_kw)
            soc += p_charge * cfg.eta_c * dt_h
            charge_kw[t] = p_charge; curtailed_kw[t] = max(surplus - p_charge, 0.0); unmet_kw[t] = 0.0
        else:
            deficit = -surplus
            energy_above_min_kwh = max(soc - soc_min, 0.0)
            max_discharge_from_soc_kw = energy_above_min_kwh * cfg.eta_d / dt_h if dt_h > 0 else 0.0
            p_dis = min(deficit, cfg.discharge_power_kw, max_discharge_from_soc_kw)
            soc -= (p_dis * dt_h) / cfg.eta_d
            discharge_kw[t] = p_dis; unmet_kw[t] = max(deficit - p_dis, 0.0); curtailed_kw[t] = 0.0
        soc = float(np.clip(soc, soc_min, soc_max)); soc_kwh[t] = soc
    load_energy = float((ld * dt_h).sum()); pv_energy = float((pv * dt_h).sum())
    curtail_energy = float((pd.Series(curtailed_kw, index=idx) * dt_h).sum())
    ens_kwh = float((pd.Series(unmet_kw, index=idx) * dt_h).sum())
    thru_charge = float((pd.Series(charge_kw, index=idx) * dt_h).sum())
    thru_discharge = float((pd.Series(discharge_kw, index=idx) * dt_h).sum())
    effective_throughput = min(thru_charge * cfg.eta_c, thru_discharge / cfg.eta_d)
    cycles = effective_throughput / (2.0 * cfg.capacity_kwh) if cfg.capacity_kwh > 0 else np.nan
    kpis = {
        "time_step_hours": dt_h,
        "total_load_kwh": load_energy,
        "total_pv_kwh": pv_energy,
        "pv_curtailed_kwh": curtail_energy,
        "pv_used_kwh": pv_energy - curtail_energy,
        "energy_not_served_kwh": ens_kwh,
        "self_sufficiency_percent": (1.0 - ens_kwh / load_energy) * 100.0 if load_energy > 0 else np.nan,
        "pv_utilization_percent": ((pv_energy - curtail_energy) / pv_energy) * 100.0 if pv_energy > 0 else np.nan,
        "avg_soc_fraction": float(np.mean(soc_kwh) / cfg.capacity_kwh) if cfg.capacity_kwh > 0 else np.nan,
        "min_soc_fraction": float(np.min(soc_kwh) / cfg.capacity_kwh) if cfg.capacity_kwh > 0 else np.nan,
        "max_soc_fraction": float(np.max(soc_kwh) / cfg.capacity_kwh) if cfg.capacity_kwh > 0 else np.nan,
        "throughput_charge_kwh": thru_charge,
        "throughput_discharge_kwh": thru_discharge,
        "estimated_full_cycles": cycles,
        "loss_of_load_hours": int((pd.Series(unmet_kw, index=idx) > 1e-9).sum()),
        "max_contiguous_lol_hours": int(_max_contiguous_true((pd.Series(unmet_kw, index=idx) > 1e-9).values) * dt_h),
    }
    ts = pd.DataFrame({
        "pv_kw": pv.reindex(idx).values,
        "load_kw": ld.reindex(idx).values,
        "net_load_kw": (ld - pv).reindex(idx).values,
        "charge_kw": charge_kw,
        "discharge_kw": discharge_kw,
        "curtailed_kw": curtailed_kw,
        "unmet_kw": unmet_kw,
        "soc_kwh": soc_kwh,
        "soc_frac": soc_kwh / cfg.capacity_kwh if cfg.capacity_kwh > 0 else np.nan,
    }, index=idx)
    ts.index.name = "timestamp"
    return ts, kpis
