import pandas as pd

def calculate_kpis_wo_spotprice(df: pd.DataFrame, df_spot: pd.DataFrame, export_price_factor: float):
    grid_energy   = float((df["grid_to_load_MWh"] + df["grid_to_batt_MWh"]).sum())
    grid_cost     = float(((df["grid_to_load_MWh"] + df["grid_to_batt_MWh"]) * df_spot["spotprice_real"]).sum())
    export_energy = float(df["ee_export_MWh"].sum())
    export_rev    = float((export_price_factor * df["ee_export_MWh"] * df_spot["spotprice_real"]).sum())
    charge_total  = float((df["ee_to_batt_MWh"] + df["grid_to_batt_MWh"]).sum())
    discharge_tot = float(df["batt_discharge_MWh"].sum())
    net_cost      = grid_cost - export_rev

    report = {
        "Netto Stromkosten": net_cost,
        "Netzstromkosten": grid_cost,
        "Einspeisevergütung": export_rev,
        "Netzbezug": grid_energy,
        "Einspeisung": export_energy,
        "Batterieladung": charge_total,
        "Batterieentladung": discharge_tot
    }
    return report