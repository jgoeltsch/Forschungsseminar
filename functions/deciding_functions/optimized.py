import pandas as pd
import numpy as np
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpBinary, PULP_CBC_CMD, LpStatus

def optimize_energy_flow(df: pd.DataFrame,
                         battery_capacity: float,
                         initial_battery: float,
                         charging_rate: float,
                         discharge_rate: float,
                         export_price_factor: float,
                         forecast_horizon: int,
                         stepsize: int):
    """
    Sliding-Window-Optimierung für den gesamten Datensatz.
    """
    window = pd.Timedelta(hours=forecast_horizon)
    step = pd.Timedelta(hours=stepsize)
    start_time = df["datetime"].min()
    end_time = df["datetime"].max()
    results_opt = []
    current_start = start_time
    current_initial_battery = initial_battery

    while current_start < end_time:
        current_end = current_start + window
        df_window = df[(df["datetime"] >= current_start) & (df["datetime"] < current_end)].copy()
        if len(df_window) == 0:
            break

        # Einzelnes Fenster optimieren (ohne Sliding-Parameter)
        result_df_opt = _single_window_optimization(
            df_window,
            battery_capacity,
            current_initial_battery,
            charging_rate,
            discharge_rate,
            export_price_factor
        )

        # Letztes Fenster komplett übernehmen, sonst nur stepsize
        if current_end >= end_time:
            results_opt.append(result_df_opt)
            break
        else:
            results_opt.append(result_df_opt.iloc[:stepsize])

        # SOC für das nächste Fenster übernehmen
        current_initial_battery = result_df_opt["SOC_MWh"].iloc[stepsize - 1]
        current_start += step

    # Ergebnisse zu DataFrame zusammenfassen
    return pd.concat(results_opt, ignore_index=True)

def _single_window_optimization(df, battery_capacity, initial_battery, charging_rate, discharge_rate, export_price_factor):
    # Zeitraster [h]
    if "datetime" in df.columns:
        t = pd.to_datetime(df["datetime"])
        dt = np.diff(t.values).astype("timedelta64[m]").astype(float) / 60.0
        if len(dt) == 0: dt = np.array([1.0])
        dt = np.append(dt, dt[-1])
    else:
        dt = np.ones(len(df))

    T  = range(len(df))
    EE = df["total_energy_production"].to_numpy()
    D  = df["energy_demand"].to_numpy()
    P  = df["spotprice"].to_numpy()

    M_ch  = charging_rate  * dt
    M_dis = discharge_rate * dt

    m = LpProblem("EnergyCostMin", LpMinimize)

    u   = {i: LpVariable(f"ee_to_load_{i}", lowBound=0)           for i in T}
    cEE = {i: LpVariable(f"ee_to_batt_{i}", lowBound=0)           for i in T}
    exp = {i: LpVariable(f"ee_export_{i}", lowBound=0)            for i in T}
    gL  = {i: LpVariable(f"grid_to_load_{i}", lowBound=0)         for i in T}
    gB  = {i: LpVariable(f"grid_to_batt_{i}", lowBound=0)         for i in T}
    d   = {i: LpVariable(f"batt_discharge_{i}", lowBound=0)       for i in T}
    soc = {i: LpVariable(f"soc_{i}", lowBound=0, upBound=battery_capacity) for i in T}
    y   = {i: LpVariable(f"charge_mode_{i}", cat=LpBinary)        for i in T}

    for i in T:
        m += u[i] + cEE[i] + exp[i] == EE[i]
        m += u[i] + d[i] + gL[i] == D[i]
        m += cEE[i] + gB[i] <= M_ch[i]
        m += d[i] <= M_dis[i]
        m += cEE[i] + gB[i] <= M_ch[i] * y[i]
        m += d[i] <= M_dis[i] * (1 - y[i])
        m += soc[i] == (initial_battery if i==0 else soc[i-1]) + (cEE[i] + gB[i] - d[i])

    m += lpSum([ P[i]*(gL[i] + gB[i]) - export_price_factor * P[i] * exp[i] for i in T ])
    m.solve(PULP_CBC_CMD(msg=False))

    out = pd.DataFrame({
        "datetime": df["datetime"] if "datetime" in df.columns else np.arange(len(df)),
        "spotprice_EUR_per_MWh": P,
        "EE_total_MWh": EE,
        "demand_MWh": D,
        "ee_to_load_MWh": [u[i].value() for i in T],
        "ee_to_batt_MWh": [cEE[i].value() for i in T],
        "ee_export_MWh": [exp[i].value() for i in T],
        "grid_to_load_MWh": [gL[i].value() for i in T],
        "grid_to_batt_MWh": [gB[i].value() for i in T],
        "batt_discharge_MWh": [d[i].value() for i in T],
        "SOC_MWh": [soc[i].value() for i in T],
        "charge_mode_binary": [y[i].value() for i in T],
        "solver_status": LpStatus[m.status]
    })

    return out