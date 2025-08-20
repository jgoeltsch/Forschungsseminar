import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import DQN
from gymnasium import spaces

# Hilfsfunktion: Sliding-Window-Indices
def get_window_indices(df, window_days, step_hours):
    window_size = window_days * 24  # Annahme: stündliche Daten
    indices = []
    start = 0
    while start + window_size <= len(df):
        end = start + window_size
        indices.append((start, end))
        start += step_hours
    # Letztes Fenster ggf. anhängen, falls nicht exakt aufgeht
    if indices and indices[-1][1] < len(df):
        indices.append((len(df) - window_size, len(df)))
    return indices


class EMSenv(gym.Env):
    """
    Action Mapping:
    0: Alles aus EE decken, Überschuss exportieren
    1: EE deckt Last, Überschuss in Batterie
    2: EE deckt Last, Netz lädt Batterie
    3: Netz deckt Last, Batterie laden
    4: Batterie entlädt zur Deckung der Last
    5: Netz deckt Last, keine Batterieaktion
    """
    def __init__(self, df, battery_capacity, initial_battery, charging_rate, discharge_rate, export_price_factor):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.battery_capacity = battery_capacity
        self.charging_rate = charging_rate
        self.discharge_rate = discharge_rate
        self.export_price_factor = export_price_factor
        self.initial_battery = initial_battery
        self.action_space = spaces.Discrete(6)  # 6 kombinierte Aktionen
        # state: [soc, demand, gen, price]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0]), 
            high=np.array([battery_capacity, 100, 100, 1000]), 
            dtype=np.float32
        )
        self.reset()

    def reset(self, *, seed=None, options=None):
        self.idx = 0
        self.soc = self.initial_battery
        obs = self._get_obs()
        info = {}
        return obs, info

    def _get_obs(self):
        row = self.df.iloc[self.idx]
        return np.array([
            self.soc,
            row["energy_demand"],
            row["total_energy_production"],
            row["spotprice"]
        ], dtype=np.float32)

    def step(self, action):
        row = self.df.iloc[self.idx]
        demand = float(row["energy_demand"])
        gen = float(row["total_energy_production"])
        price = float(row["spotprice"])
        reward = 0

        # Default Flüsse
        grid_to_load = 0
        grid_to_batt = 0
        ee_to_load = 0
        ee_to_batt = 0
        ee_export = 0
        batt_discharge = 0

        # Action-Logik
        if action == 0:
            # EE deckt Last, Überschuss exportieren
            ee_to_load = min(gen, demand)
            ee_export = max(gen - demand, 0)
            grid_to_load = max(demand - gen, 0)
        elif action == 1:
            # EE deckt Last, Überschuss in Batterie
            ee_to_load = min(gen, demand)
            rest = max(gen - demand, 0)
            charge = min(self.charging_rate, self.battery_capacity - self.soc, rest)
            ee_to_batt = charge
            self.soc += charge
            ee_export = max(rest - charge, 0)
            grid_to_load = max(demand - gen, 0)
        elif action == 2:
            # EE deckt Last, Netz lädt Batterie
            ee_to_load = min(gen, demand)
            grid_to_load = max(demand - gen, 0)
            charge = min(self.charging_rate, self.battery_capacity - self.soc)
            grid_to_batt = charge
            self.soc += charge
        elif action == 3:
            # Netz deckt Last, Batterie laden
            grid_to_load = demand
            charge = min(self.charging_rate, self.battery_capacity - self.soc)
            grid_to_batt = charge
            self.soc += charge
        elif action == 4:
            # Batterie entlädt zur Deckung der Last
            discharge = min(self.discharge_rate, self.soc, demand)
            batt_discharge = discharge
            self.soc -= discharge
            grid_to_load = max(demand - discharge, 0)
        elif action == 5:
            # Netz deckt Last, keine Batterieaktion
            grid_to_load = demand

        # Reward: negative Kosten (Kosten minimieren)
        cost = (
            grid_to_load * price
            + grid_to_batt * price
            - self.export_price_factor * ee_export * price
        )
        reward = -cost

        self.idx += 1
        terminated = self.idx >= len(self.df)
        truncated = False
        obs = self._get_obs() if not terminated else np.zeros(4, dtype=np.float32)
        info = {
            "grid_to_load": grid_to_load,
            "grid_to_batt": grid_to_batt,
            "ee_to_load": ee_to_load,
            "ee_to_batt": ee_to_batt,
            "ee_export": ee_export,
            "batt_discharge": batt_discharge,
            "SOC": self.soc
        }
        return obs, reward, terminated, truncated, info


# Online-Learning mit Sliding Window
def reinforcement_learning_train(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    battery_capacity: float,
    initial_battery: float,
    charging_rate: float,
    discharge_rate: float,
    export_price_factor: float,
    window_days: int,
    step_hours: int,
    total_timesteps: int
):
    
    # Fenster-Indices
    train_indices = get_window_indices(df_train, window_days, step_hours)
    
    # Modell initialisieren mit erstem Trainingsfenster
    env_train = EMSenv(
        df_train.iloc[train_indices[0][0]:train_indices[0][1]],
        battery_capacity, initial_battery, charging_rate, discharge_rate, export_price_factor
    )
    model = DQN("MlpPolicy", env_train, verbose=1)

    # TRAININGPHASE
    soc_train = initial_battery
    for i, train_idx in enumerate(train_indices):
        train_start, train_end = train_idx
        env_train = EMSenv(
            df_train.iloc[train_start:train_end],
            battery_capacity, soc_train, charging_rate, discharge_rate, export_price_factor
        )
        model.set_env(env_train)
        model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False)
        # SOC am Ende des Trainingsfensters für nächstes Fenster übernehmen
        obs, info = env_train.reset()
        terminated = False
        truncated = False
        last_soc = soc_train
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env_train.step(action)
            last_soc = info["SOC"]
        soc_train = last_soc

    # Modell speichern
    model.save("model/rl_model_trained.zip")

    # Kein Testteil mehr in diesem Skript. Nur Training und Speichern des Modells.
    return model