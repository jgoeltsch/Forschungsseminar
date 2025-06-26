import pandas as pd

def rule_based_energy_flow(df, battery_capacity, initial_battery):
    battery_state = initial_battery
    results = []

    for _, row in df.iterrows():

        # 0. Demand und generation ziehen
        demand = row["energy_demand"]
        generation = row["total_energy_production"]

        # 1. Eigennutzung
        used_generation = min(demand, generation)

        # 2. Restbedarf und Überschuss berechnen
        residual_demand = demand - used_generation
        excess_generation = generation - used_generation

        # 3. Akku laden mit Überschuss
        battery_charge = min(battery_capacity - battery_state, excess_generation)
        battery_state += battery_charge
        excess_generation -= battery_charge

        # 4. Akku entladen
        battery_discharge = min(battery_state, residual_demand)
        battery_state -= battery_discharge
        residual_demand -= battery_discharge

        results.append({
            "datetime": row["datetime"],
            "grid_buy": residual_demand,
            "grid_feed_in": excess_generation,
            "battery_charge": battery_charge,
            "battery_discharge": battery_discharge,
            "battery_state": battery_state
        })

    result_df_rule = pd.DataFrame(results)
    df = df.reset_index(drop=True)
    result_df_rule = pd.concat([df, result_df_rule.drop(columns=["datetime"])], axis=1)
    return result_df_rule