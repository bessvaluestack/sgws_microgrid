#!/bin/zsh

# Battery Capacity Sweep Runner
# Runs battery capacity sweeps for different PV systems and load scenarios

set -e  # Exit on any error

echo "Starting battery capacity sweep analysis..."
echo "$(date): Beginning runs"

# 259kW PV System (Large) - High Load
echo "Running: 259kW PV + High Load A1..."
python battery_capacity_sweep.py --pv pvwatts_hourly_fl_259kw_a1.csv \
  --load ev_load_high_load_2024_a1.csv \
  --capacity-start 50 \
  --capacity-end 2500 \
  --capacity-step 50 \
  --power-kw 60 \
  --fixed-load-kw 2 \
  --resample 30min \
  --interp linear \
  --out-file comparison_high_large_pv_a1.csv

echo "Running: 259kW PV + High Load A2..."
python battery_capacity_sweep.py --pv pvwatts_hourly_fl_259kw_a2.csv \
  --load ev_load_high_load_2024_a2.csv \
  --capacity-start 50 \
  --capacity-end 2500 \
  --capacity-step 50 \
  --power-kw 30 \
  --fixed-load-kw 1 \
  --resample 30min \
  --interp linear \
  --out-file comparison_high_large_pv_a2.csv

# 259kW PV System (Large) - Medium Load
echo "Running: 259kW PV + Medium Load A1..."
python battery_capacity_sweep.py --pv pvwatts_hourly_fl_259kw_a1.csv \
  --load ev_load_medium_load_2024_a1.csv \
  --capacity-start 50 \
  --capacity-end 2500 \
  --capacity-step 50 \
  --power-kw 60 \
  --fixed-load-kw 2 \
  --resample 30min \
  --interp linear \
  --out-file comparison_med_large_pv_a1.csv

echo "Running: 259kW PV + Medium Load A2..."
python battery_capacity_sweep.py --pv pvwatts_hourly_fl_259kw_a2.csv \
  --load ev_load_medium_load_2024_a2.csv \
  --capacity-start 50 \
  --capacity-end 2500 \
  --capacity-step 50 \
  --power-kw 30 \
  --fixed-load-kw 1 \
  --resample 30min \
  --interp linear \
  --out-file comparison_med_large_pv_a2.csv

# 259kW PV System (Large) - Low Load
echo "Running: 259kW PV + Low Load A1..."
python battery_capacity_sweep.py --pv pvwatts_hourly_fl_259kw_a1.csv \
  --load ev_load_low_load_2024_a1.csv \
  --capacity-start 50 \
  --capacity-end 2500 \
  --capacity-step 50 \
  --power-kw 60 \
  --fixed-load-kw 2 \
  --resample 30min \
  --interp linear \
  --out-file comparison_low_large_pv_a1.csv

echo "Running: 259kW PV + Low Load A2..."
python battery_capacity_sweep.py --pv pvwatts_hourly_fl_259kw_a2.csv \
  --load ev_load_low_load_2024_a2.csv \
  --capacity-start 50 \
  --capacity-end 2500 \
  --capacity-step 50 \
  --power-kw 30 \
  --fixed-load-kw 1 \
  --resample 30min \
  --interp linear \
  --out-file comparison_low_large_pv_a2.csv

# 194kW PV System (Medium) - High Load
echo "Running: 194kW PV + High Load A1..."
python battery_capacity_sweep.py --pv pvwatts_hourly_fl_194kw_a1.csv \
  --load ev_load_high_load_2024_a1.csv \
  --capacity-start 50 \
  --capacity-end 2500 \
  --capacity-step 50 \
  --power-kw 60 \
  --fixed-load-kw 2 \
  --resample 30min \
  --interp linear \
  --out-file comparison_high_medium_pv_a1.csv

echo "Running: 194kW PV + High Load A2..."
python battery_capacity_sweep.py --pv pvwatts_hourly_fl_194kw_a2.csv \
  --load ev_load_high_load_2024_a2.csv \
  --capacity-start 50 \
  --capacity-end 2500 \
  --capacity-step 50 \
  --power-kw 30 \
  --fixed-load-kw 1 \
  --resample 30min \
  --interp linear \
  --out-file comparison_high_medium_pv_a2.csv

# 194kW PV System (Medium) - Medium Load
echo "Running: 194kW PV + Medium Load A1..."
python battery_capacity_sweep.py --pv pvwatts_hourly_fl_194kw_a1.csv \
  --load ev_load_medium_load_2024_a1.csv \
  --capacity-start 50 \
  --capacity-end 2500 \
  --capacity-step 50 \
  --power-kw 60 \
  --fixed-load-kw 2 \
  --resample 30min \
  --interp linear \
  --out-file comparison_med_medium_pv_a1.csv

echo "Running: 194kW PV + Medium Load A2..."
python battery_capacity_sweep.py --pv pvwatts_hourly_fl_194kw_a2.csv \
  --load ev_load_medium_load_2024_a2.csv \
  --capacity-start 50 \
  --capacity-end 2500 \
  --capacity-step 50 \
  --power-kw 30 \
  --fixed-load-kw 1 \
  --resample 30min \
  --interp linear \
  --out-file comparison_med_medium_pv_a2.csv

# 194kW PV System (Medium) - Low Load
echo "Running: 194kW PV + Low Load A1..."
python battery_capacity_sweep.py --pv pvwatts_hourly_fl_194kw_a1.csv \
  --load ev_load_low_load_2024_a1.csv \
  --capacity-start 50 \
  --capacity-end 2500 \
  --capacity-step 50 \
  --power-kw 60 \
  --fixed-load-kw 2 \
  --resample 30min \
  --interp linear \
  --out-file comparison_low_medium_pv_a1.csv

echo "Running: 194kW PV + Low Load A2..."
python battery_capacity_sweep.py --pv pvwatts_hourly_fl_194kw_a2.csv \
  --load ev_load_low_load_2024_a2.csv \
  --capacity-start 50 \
  --capacity-end 2500 \
  --capacity-step 50 \
  --power-kw 30 \
  --fixed-load-kw 1 \
  --resample 30min \
  --interp linear \
  --out-file comparison_low_medium_pv_a2.csv

echo "$(date): All battery capacity sweeps completed successfully!"
echo "Output files generated:"
echo "  - comparison_med_large_pv_a1.csv"
echo "  - comparison_med_large_pv_a2.csv"
echo "  - comparison_low_large_pv_a1.csv"
echo "  - comparison_low_large_pv_a2.csv"
echo "  - comparison_high_medium_pv_a1.csv"
echo "  - comparison_high_medium_pv_a2.csv"
echo "  - comparison_med_medium_pv_a1.csv"
echo "  - comparison_med_medium_pv_a2.csv"
echo "  - comparison_low_medium_pv_a1.csv"
echo "  - comparison_low_medium_pv_a2.csv"