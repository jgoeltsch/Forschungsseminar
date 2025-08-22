import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import DQN
from gymnasium import spaces
from datetime import datetime

class EMSenv(gym.Env):
    """
    Action Mapping:
    0: 1) EE->Load 2) EE->Batterie 3) EE->Grid
    1: 1) EE->Load 2) EE->Batterie 3) Grid->Batterie
    2: 1) EE->Load 2) EE->Batterie
    3: 1) EE->Load 2) Batterie->Load 3) Grid->Load
    4: 1) EE->Load 2) Grid->Load
    5: 1) EE->Load 2) Grid->Load 3) Grid->Batterie
    """
    def __init__(self, df, battery_capacity, initial_battery, charging_rate, discharge_rate, export_price_factor, window_size=48):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.battery_capacity = battery_capacity
        self.charging_rate = charging_rate
        self.discharge_rate = discharge_rate
        self.export_price_factor = export_price_factor
        self.initial_battery = initial_battery
        self.window_size = window_size
        self.action_space = spaces.Discrete(6)  # 6 kombinierte Aktionen
        # state: [soc, demand, gen, price] für window_size Zeitschritte
        self.observation_space = spaces.Box(
            low=np.tile(np.array([0, 0, 0, 0]), self.window_size),
            high=np.tile(np.array([battery_capacity, 100, 100, 1000]), self.window_size),
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
        # Fenster von window_size Zeitschritten ab aktuellem Index
        window = []
        for i in range(self.window_size):
            idx = self.idx + i
            if idx < len(self.df):
                row = self.df.iloc[idx]
                window.append([
                    self.soc if i == 0 else np.nan,  # nur für t=0 ist SOC bekannt
                    row["energy_demand"],
                    row["total_energy_production"],
                    row["spotprice"]
                ])
            else:
                window.append([0, 0, 0, 0])  # Padding am Ende
        return np.array(window).flatten().astype(np.float32)

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

        # Action-Logik nach neuem Mapping
        if action == 0:
            # 0: 1) EE->Load 2) EE->Batterie 3) EE->Grid
            ee_to_load = min(gen, demand)
            rest = max(gen - demand, 0)
            charge = min(self.charging_rate, self.battery_capacity - self.soc, rest)
            ee_to_batt = charge
            self.soc += charge
            ee_export = max(rest - charge, 0)
            grid_to_load = max(demand - gen, 0)
        elif action == 1:
            # 1: 1) EE->Load 2) EE->Batterie 3) Grid->Batterie
            ee_to_load = min(gen, demand)
            rest = max(gen - demand, 0)
            charge_ee = min(self.charging_rate, self.battery_capacity - self.soc, rest)
            ee_to_batt = charge_ee
            self.soc += charge_ee
            # Grid->Batterie (zusätzlich, falls noch Kapazität)
            rest_batt = self.battery_capacity - self.soc
            charge_grid = min(self.charging_rate - charge_ee, rest_batt) if self.charging_rate > charge_ee else 0
            grid_to_batt = charge_grid
            self.soc += charge_grid
            grid_to_load = max(demand - gen, 0)
        elif action == 2:
            # 2: 1) EE->Load 2) EE->Batterie
            ee_to_load = min(gen, demand)
            rest = max(gen - demand, 0)
            charge = min(self.charging_rate, self.battery_capacity - self.soc, rest)
            ee_to_batt = charge
            self.soc += charge
            grid_to_load = max(demand - gen, 0)
        elif action == 3:
            # 3: 1) EE->Load 2) Batterie->Load 3) Grid->Load
            ee_to_load = min(gen, demand)
            rest_demand = max(demand - gen, 0)
            discharge = min(self.discharge_rate, self.soc, rest_demand)
            batt_discharge = discharge
            self.soc -= discharge
            grid_to_load = max(rest_demand - discharge, 0)
        elif action == 4:
            # 4: 1) EE->Load 2) Grid->Load
            ee_to_load = min(gen, demand)
            grid_to_load = max(demand - gen, 0)
        elif action == 5:
            # 5: 1) EE->Load 2) Grid->Load 3) Grid->Batterie
            ee_to_load = min(gen, demand)
            grid_to_load = max(demand - gen, 0)
            # Grid->Batterie (zusätzlich)
            charge = min(self.charging_rate, self.battery_capacity - self.soc)
            grid_to_batt = charge
            self.soc += charge

        # Reward: negative Kosten (Kosten minimieren)
        cost = (
            grid_to_load * price
            + grid_to_batt * price
            - self.export_price_factor * ee_export * price
        )
        reward = -cost

        self.idx += 1
        terminated = self.idx > (len(self.df) - self.window_size)
        truncated = False
        obs = self._get_obs() if not terminated else np.zeros(self.window_size * 4, dtype=np.float32)
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


def reinforcement_learning_train(
    df_train: pd.DataFrame,
    battery_capacity: float,
    initial_battery: float,
    charging_rate: float,
    discharge_rate: float,
    export_price_factor: float,
    total_timesteps: int,
    window_size: int = 48
):
    env_train = EMSenv(
        df_train,
        battery_capacity, initial_battery, charging_rate, discharge_rate, export_price_factor,
        window_size=window_size
    )
    policy_kwargs = dict(
        net_arch=[256, 256, 256],
        activation_fn = __import__('torch').nn.ReLU
    )
    model = DQN(
        "MlpPolicy",
        env_train,
        buffer_size=100_000,
        batch_size=64,
        learning_rate=1e-4,
        gamma=0.99,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        exploration_fraction=50_000/total_timesteps if total_timesteps > 0 else 1.0,
        target_update_interval=1000,
        policy_kwargs=policy_kwargs,
        verbose=1
    )
    model.learn(total_timesteps=total_timesteps)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H_%M")
    model_path = f"model/trained_model_{timestamp}.zip"
    model.save(model_path)
    return model