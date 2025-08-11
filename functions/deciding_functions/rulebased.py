import pandas as pd

def rule_based_energy_flow(df, df_spot, battery_capacity, initial_battery):
    """
    df: DataFrame mit Spalten [datetime, energy_demand, total_energy_production]
    df_spot: DataFrame mit Spalten [datetime, spotprice] (lokale, naive Zeit)
    battery_capacity: MWh
    initial_battery: MWh
    """
    # Spotpreise über datetime an den Haupt-Datensatz hängen (Left Join)
    df = df.copy()
    df = df.merge(df_spot[["datetime", "spotprice"]], on="datetime", how="left")

    battery_state = initial_battery
    results = []

    for _, row in df.iterrows():
        demand = float(row["energy_demand"])                 # Strombedarf [MWh] in dieser Stunde
        generation = float(row["total_energy_production"])   # Stromerzeugung [MWh] in dieser Stunde

        # 1. Eigennutzung: erzeugten Strom direkt für den Bedarf nutzen
        used_generation = min(demand, generation)

        # 2. Restbedarf / Überschuss berechnen
        residual_demand = demand - used_generation             # Netzbezug, falls positiv [MWh]
        excess_generation = generation - used_generation       # Einspeisung, falls positiv [MWh]

        # 3. Akku laden mit überschüssiger Energie
        battery_charge = min(battery_capacity - battery_state, max(excess_generation, 0.0))
        battery_state += battery_charge
        excess_generation -= battery_charge

        # 4. Akku entladen, um Restbedarf zu decken
        battery_discharge = min(battery_state, max(residual_demand, 0.0))
        battery_state -= battery_discharge
        residual_demand -= battery_discharge

        results.append({
            "datetime": row["datetime"],
            "grid_buy": max(residual_demand, 0.0),            # Strom aus Netz [MWh]
            "grid_feed_in": max(excess_generation, 0.0),      # Überschuss ins Netz [MWh]
            "battery_charge": battery_charge,                 # Akkuladung in dieser Stunde [MWh]
            "battery_discharge": battery_discharge,           # Akkuentladung in dieser Stunde [MWh]
            "battery_state": battery_state,                   # aktueller Akkustand [MWh]
            "spotprice": row["spotprice"]                     # Strompreis in ct/kWh
        })

    result_df_rule = pd.DataFrame(results)

    # Kostenberechnung:
    # grid_buy ist in MWh → Umrechnung in kWh (*1000)
    # spotprice ist in ct/kWh → Umrechnung in € (/100)
    result_df_rule["grid_buy_cost_eur"] = (
        result_df_rule["grid_buy"] * 1000.0 * result_df_rule["spotprice"] / 100.0
    )

    # Summe der Gesamtkosten aus Netzbezug in Euro
    total_grid_cost_eur = result_df_rule["grid_buy_cost_eur"].sum()

    # Originaldaten wieder mit den Berechnungsergebnissen zusammenführen
    df = df.reset_index(drop=True)
    result_df_rule = pd.concat([df.drop(columns=["spotprice"]),
                                result_df_rule.drop(columns=["datetime"])], axis=1)

    return result_df_rule, float(total_grid_cost_eur)
