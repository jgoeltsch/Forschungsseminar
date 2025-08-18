import pandas as pd
import numpy as np

def rule_based_energy_flow(df: pd.DataFrame,
                           battery_capacity: float,
                           initial_battery: float,
                           charging_rate: float,
                           discharge_rate: float,
                           export_price_factor: float,
                           price_quantile: float = 0.30):
    df = df.copy()

    # Zeitraster
    t = pd.to_datetime(df["datetime"])
    dt = np.diff(t.values).astype("timedelta64[m]").astype(float) / 60.0
    if len(dt) == 0: dt = np.array([1.0])
    dt = np.append(dt, dt[-1])

    # Preise in €/MWh
    price = df["spotprice"].to_numpy()
    low_price_thresh = np.quantile(price, price_quantile)

    soc = initial_battery
    rows = []

    for i, row in df.reset_index(drop=True).iterrows():
        demand = float(row["energy_demand"])
        gen    = float(row["total_energy_production"])
        p      = price[i]
        ch_lim = charging_rate  * dt[i]
        dis_lim= discharge_rate * dt[i]

        # EE -> Last
        ee_to_load = min(demand, gen)
        residual   = demand - ee_to_load
        ee_excess  = gen - ee_to_load

        # EE -> Batterie
        cap_room     = max(battery_capacity - soc, 0.0)
        ee_ch_cap    = min(cap_room, ch_lim)
        ee_to_batt   = min(max(ee_excess, 0.0), ee_ch_cap)
        soc         += ee_to_batt
        ee_excess   -= ee_to_batt
        ch_left      = ee_ch_cap - ee_to_batt

        # Netz -> Batterie bei niedrigen Preisen
        grid_to_batt = 0.0
        if ch_left > 0 and p <= low_price_thresh:
            grid_to_batt = ch_left
            soc += grid_to_batt

        # Batterie -> Last nur wenn nicht geladen wurde
        batt_discharge = 0.0
        if ee_to_batt == 0.0 and grid_to_batt == 0.0:
            batt_discharge = min(soc, dis_lim, residual)
            soc -= batt_discharge
            residual -= batt_discharge

        # Netz -> Last
        grid_to_load = max(residual, 0.0)

        rows.append({
            "datetime": row["datetime"],
            "spotprice_EUR_per_MWh": p,
            "EE_total_MWh": gen,
            "demand_MWh": demand,
            "ee_to_load_MWh": ee_to_load,
            "ee_to_batt_MWh": ee_to_batt,
            "ee_export_MWh": max(ee_excess, 0.0),
            "grid_to_load_MWh": grid_to_load,
            "grid_to_batt_MWh": grid_to_batt,
            "batt_discharge_MWh": batt_discharge,
            "SOC_MWh": soc,
            "charge_mode_binary": 1 if (ee_to_batt + grid_to_batt) > 0 else 0
        })

    out = pd.DataFrame(rows)

    return out
