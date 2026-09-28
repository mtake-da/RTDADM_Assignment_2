# AITU Smart Classroom Energy Stream — Assignment 2

Synthetic educational data. Pandas/NumPy pipeline for one classroom (C1.3.101), 10:00–10:11.

## Files
- `classroom_stream.py` — cleaning, rolling stats, feature engineering, statistics (source of truth)
- `Assignment2_AITU_Classroom.ipynb` — same pipeline, notebook form, with the two required plots
- `cleaned_classroom_data.csv` — cleaned dataset (interpolated Temp_C/CO2_ppm, forward-filled Occupancy)
- `plot_rolling_temp_co2.png`, `plot_occupancy_vs_power.png` — the two required plots
- `Assignment2_Student_Technical_Report.pdf` — completed report template

## Run
```bash
pip install pandas numpy matplotlib
python3 classroom_stream.py      # prints tables, writes cleaned_classroom_data.csv
python3 make_plots.py            # writes the two PNG plots
```

## Key results
- Average temperature 22.48°C, median 22.55°C, range 21.5–23.3°C
- Mean CO2 789.17 ppm, max 950 ppm
- Peak power 5.7 kW at 10:11
- Occupancy–power correlation: 0.99 (very strong positive)
- All 12 readings pass the comfort flag (CO2 ≤ 1000 ppm, 20–24°C)
