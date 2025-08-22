import pandas as pd

def reinforcement_learning_test(
    model,
    df_test,
    env_class,
    env_kwargs,
    initial_battery
):
    results = []
    window_size = env_kwargs.get('window_size', 48)
    env = env_class(
        df_test,
        **{**env_kwargs, 'initial_battery': initial_battery}
    )
    obs, info = env.reset()
    terminated = False
    idx = 0
    n = len(df_test)
    # Schleife: für alle Zeitschritte, auch wenn window_size == n
    while not terminated and idx < n:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        row = df_test.iloc[idx]
        results.append({
            "datetime": row["datetime"],
            "spotprice_EUR_per_MWh": row["spotprice"],
            "EE_total_MWh": row["total_energy_production"],
            "demand_MWh": row["energy_demand"],
            "ee_to_load_MWh": info["ee_to_load"],
            "ee_to_batt_MWh": info["ee_to_batt"],
            "ee_export_MWh": info["ee_export"],
            "grid_to_load_MWh": info["grid_to_load"],
            "grid_to_batt_MWh": info["grid_to_batt"],
            "batt_discharge_MWh": info["batt_discharge"],
            "SOC_MWh": info["SOC"],
            "charge_mode_binary": 1 if (info["ee_to_batt"] + info["grid_to_batt"]) > 0 else 0,
            "solver_status": "RL"
        })
        idx += 1
    return pd.DataFrame(results)