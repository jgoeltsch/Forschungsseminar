import pandas as pd

def get_window_indices(df, window_days, step_hours):
    window_size = window_days * 24  # Annahme: stündliche Daten
    indices = []
    start = 0
    while start + window_size <= len(df):
        end = start + window_size
        indices.append((start, end))
        start += step_hours
    if indices and indices[-1][1] < len(df):
        indices.append((len(df) - window_size, len(df)))
    return indices

def reinforcement_learning_test(
    model,
    df_test,
    env_class,
    env_kwargs,
    window_days,
    step_hours,
    initial_battery
):
    
    test_indices = get_window_indices(df_test, window_days, step_hours)
    soc_test = initial_battery
    results = []

    num_windows = len(test_indices)
    for i, test_idx in enumerate(test_indices):
        test_start, test_end = test_idx
        env = env_class(
            df_test.iloc[test_start:test_end],
            **{**env_kwargs, 'initial_battery': soc_test}
        )
        obs, info = env.reset()
        terminated = False
        truncated = False
        window_rows = []
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            idx = env.idx - 1
            if idx < (test_end - test_start):
                row = df_test.iloc[test_start + idx]
                window_rows.append({
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

        # SOC für nächstes Testfenster übernehmen
        if window_rows:
            soc_test = window_rows[-1]["SOC_MWh"]

        # Ergebnisaggregation: nur die ersten step_hours übernehmen, außer letztes Fenster (alles)
        if i < num_windows - 1:
            if window_rows:
                results.extend(window_rows[:step_hours])
        else:
            results.extend(window_rows)

    return pd.DataFrame(results)