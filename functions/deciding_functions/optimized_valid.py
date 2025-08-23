import pandas as pd
import numpy as np
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpBinary, PULP_CBC_CMD, LpStatus

def optimize_energy_flow_valid(df_forecast: pd.DataFrame,
                         df_real: pd.DataFrame,
                         battery_capacity: float,
                         initial_battery: float,
                         charging_rate: float,
                         discharge_rate: float,
                         export_price_factor: float,
                         forecast_horizon: int,
                         stepsize: int):
    """
    Sliding-Window-Optimierung: Optimierung mit Forecast-Daten, Berechnung der Flüsse mit Real-Daten.
    Erwartet: df_forecast, df_real als separate DataFrames.
    """
    window = pd.Timedelta(hours=forecast_horizon)
    step = pd.Timedelta(hours=stepsize)
    start_time = df_forecast["datetime"].min()
    end_time = df_forecast["datetime"].max()
    results_opt = []
    current_start = start_time
    current_initial_battery = initial_battery

    while current_start < end_time:
        current_end = current_start + window
        df_window_forecast = df_forecast[(df_forecast["datetime"] >= current_start) & (df_forecast["datetime"] < current_end)].copy()
        df_window_real = df_real[(df_real["datetime"] >= current_start) & (df_real["datetime"] < current_end)].copy()
        if len(df_window_forecast) == 0 or len(df_window_real) == 0:
            break

        # Optimierung mit Forecast-Daten
        result_df_opt = _single_window_optimization(
            df_window_forecast,
            battery_capacity,
            current_initial_battery,
            charging_rate,
            discharge_rate,
            export_price_factor
        )

        # Steuerungsvorgaben (z.B. Ladevorgaben) auf Real-Daten anwenden
        # Annahme: Reihenfolge und Länge stimmen überein
        n = min(len(result_df_opt), len(df_window_real))
        real_datetimes = df_window_real["datetime"].values[:n]
        real_demand = df_window_real["energy_demand"].values[:n]
        real_ee = df_window_real["total_energy_production"].values[:n]
        real_spot = df_window_real["spotprice"].values[:n]

        # Steuerungsvorgaben aus Optimierung
        ee_to_load = result_df_opt["ee_to_load_MWh"].values[:n]
        ee_to_batt = result_df_opt["ee_to_batt_MWh"].values[:n]
        ee_export = result_df_opt["ee_export_MWh"].values[:n]
        grid_to_load = result_df_opt["grid_to_load_MWh"].values[:n]
        grid_to_batt = result_df_opt["grid_to_batt_MWh"].values[:n]
        batt_discharge = result_df_opt["batt_discharge_MWh"].values[:n]
        charge_mode = result_df_opt["charge_mode_binary"].values[:n]

        # SOC und Flüsse mit Realwerten berechnen
        soc_real = np.zeros(n)
        soc_real[0] = current_initial_battery + ee_to_batt[0] + grid_to_batt[0] - batt_discharge[0]
        for i in range(1, n):
            soc_real[i] = soc_real[i-1] + ee_to_batt[i] + grid_to_batt[i] - batt_discharge[i]
            soc_real[i] = np.clip(soc_real[i], 0, battery_capacity)

        out = pd.DataFrame({
            "datetime": real_datetimes,
            "spotprice_EUR_per_MWh": real_spot,
            "EE_total_MWh": real_ee,
            "demand_MWh": real_demand,
            "ee_to_load_MWh": ee_to_load,
            "ee_to_batt_MWh": ee_to_batt,
            "ee_export_MWh": ee_export,
            "grid_to_load_MWh": grid_to_load,
            "grid_to_batt_MWh": grid_to_batt,
            "batt_discharge_MWh": batt_discharge,
            "SOC_MWh": soc_real,
            "charge_mode_binary": charge_mode,
            "solver_status": result_df_opt["solver_status"].values[0]
        })

        # Letztes Fenster komplett übernehmen, sonst nur stepsize
        if current_end >= end_time:
            results_opt.append(out)
            break
        else:
            results_opt.append(out.iloc[:stepsize])

        # SOC für das nächste Fenster übernehmen
        current_initial_battery = soc_real[stepsize - 1]
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