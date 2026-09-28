"""
classroom_stream.py
Assignment 2 - Real-Time Data Processing with Pandas and NumPy
AITU Smart Classroom Energy Stream (synthetic educational data).
"""

import numpy as np
import pandas as pd

RAW_DATA = [
    {"Time": "10:00", "Room": "C1.3.101", "Temp_C": 21.5, "CO2_ppm": 620, "Occupancy": 18, "Power_kW": 3.2},
    {"Time": "10:01", "Room": "C1.3.101", "Temp_C": 21.7, "CO2_ppm": 650, "Occupancy": 20, "Power_kW": 3.4},
    {"Time": "10:02", "Room": "C1.3.101", "Temp_C": None, "CO2_ppm": 690, "Occupancy": 22, "Power_kW": 3.5},
    {"Time": "10:03", "Room": "C1.3.101", "Temp_C": 22.1, "CO2_ppm": 720, "Occupancy": 24, "Power_kW": 3.8},
    {"Time": "10:04", "Room": "C1.3.101", "Temp_C": 22.3, "CO2_ppm": None, "Occupancy": 25, "Power_kW": 4.0},
    {"Time": "10:05", "Room": "C1.3.101", "Temp_C": 22.5, "CO2_ppm": 780, "Occupancy": 27, "Power_kW": 4.2},
    {"Time": "10:06", "Room": "C1.3.101", "Temp_C": 22.6, "CO2_ppm": 810, "Occupancy": 29, "Power_kW": 4.4},
    {"Time": "10:07", "Room": "C1.3.101", "Temp_C": 22.7, "CO2_ppm": 830, "Occupancy": None, "Power_kW": 4.6},
    {"Time": "10:08", "Room": "C1.3.101", "Temp_C": 22.9, "CO2_ppm": 850, "Occupancy": 31, "Power_kW": 4.8},
    {"Time": "10:09", "Room": "C1.3.101", "Temp_C": 23.1, "CO2_ppm": 900, "Occupancy": 33, "Power_kW": 5.1},
    {"Time": "10:10", "Room": "C1.3.101", "Temp_C": 23.2, "CO2_ppm": 920, "Occupancy": 35, "Power_kW": 5.4},
    {"Time": "10:11", "Room": "C1.3.101", "Temp_C": 23.3, "CO2_ppm": 950, "Occupancy": 37, "Power_kW": 5.7},
]


def load_raw() -> pd.DataFrame:
    df = pd.DataFrame(RAW_DATA)
    df["Time"] = pd.to_datetime(df["Time"], format="%H:%M")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Temp_C"] = out["Temp_C"].interpolate(method="linear")
    out["CO2_ppm"] = out["CO2_ppm"].interpolate(method="linear")
    out["Occupancy"] = out["Occupancy"].ffill()
    out["Occupancy"] = out["Occupancy"].astype(int)
    return out


def missing_value_report(raw: pd.DataFrame, cleaned: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, method in [("Temp_C", "linear interpolation"), ("CO2_ppm", "linear interpolation"), ("Occupancy", "forward fill")]:
        mask = raw[col].isna()
        for idx in raw[mask].index:
            rows.append({
                "Time": raw.loc[idx, "Time"].strftime("%H:%M"),
                "Variable": col,
                "Method": method,
                "Value_after_cleaning": cleaned.loc[idx, col],
            })
    return pd.DataFrame(rows)


def rolling_stats(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    out = df.copy()
    out["Temp_roll_mean"] = out["Temp_C"].rolling(window).mean()
    out["CO2_roll_mean"] = out["CO2_ppm"].rolling(window).mean()
    out["Power_roll_mean"] = out["Power_kW"].rolling(window).mean()
    return out


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["occupancy_density"] = out["Occupancy"] / 40
    out["energy_per_student"] = out["Power_kW"] / out["Occupancy"]
    out["comfort_flag"] = (out["CO2_ppm"] <= 1000) & out["Temp_C"].between(20, 24)
    return out


def statistical_summary(df: pd.DataFrame) -> dict:
    corr = np.corrcoef(df["Occupancy"], df["Power_kW"])[0, 1]
    busiest_power_row = df.loc[df["Power_kW"].idxmax()]
    return {
        "temp_mean": round(df["Temp_C"].mean(), 2),
        "temp_median": round(df["Temp_C"].median(), 2),
        "temp_min": df["Temp_C"].min(),
        "temp_max": df["Temp_C"].max(),
        "co2_mean": round(df["CO2_ppm"].mean(), 2),
        "co2_max": df["CO2_ppm"].max(),
        "peak_power_time": busiest_power_row["Time"].strftime("%H:%M"),
        "peak_power_kw": busiest_power_row["Power_kW"],
        "occupancy_power_corr": round(corr, 4),
    }


if __name__ == "__main__":
    raw = load_raw()
    cleaned = clean(raw)
    print(missing_value_report(raw, cleaned).to_string(index=False))
    print()
    rolled = rolling_stats(cleaned)
    print(rolled[["Time", "Temp_roll_mean", "CO2_roll_mean", "Power_roll_mean"]].to_string(index=False))
    print()
    featured = engineer_features(cleaned)
    print(featured[["Time", "occupancy_density", "energy_per_student", "comfort_flag"]].to_string(index=False))
    print()
    print(statistical_summary(cleaned))
    export = cleaned.copy()
    export["Time"] = export["Time"].dt.strftime("%H:%M")
    export.to_csv("cleaned_classroom_data.csv", index=False)
