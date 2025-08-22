# drqn_ems.py
import numpy as np
import pandas as pd
from datetime import datetime
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
import torch.optim as optim
import random
from typing import Tuple, Dict, List

# -----------------------------
# ENV: unverändert nutzbar
# -----------------------------
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
    metadata = {"render_modes": []}

    def __init__(self, df, battery_capacity, initial_battery, charging_rate, discharge_rate, export_price_factor):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.battery_capacity = float(battery_capacity)
        self.charging_rate = float(charging_rate)
        self.discharge_rate = float(discharge_rate)
        self.export_price_factor = float(export_price_factor)
        self.initial_battery = float(initial_battery)

        self.action_space = spaces.Discrete(6)
        # state: [soc, demand, gen, price]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0], dtype=np.float32),
            high=np.array([self.battery_capacity, 1e3, 1e3, 1e4], dtype=np.float32),
            dtype=np.float32,
        )
        self.idx = 0
        self.soc = self.initial_battery

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.idx = 0
        self.soc = self.initial_battery
        return self._get_obs(), {}

    def _get_obs(self):
        row = self.df.iloc[self.idx]
        return np.array(
            [self.soc, row["energy_demand"], row["total_energy_production"], row["spotprice"]],
            dtype=np.float32,
        )

    def step(self, action: int):
        row = self.df.iloc[self.idx]
        demand = float(row["energy_demand"])
        gen = float(row["total_energy_production"])
        price = float(row["spotprice"])

        grid_to_load = 0.0
        grid_to_batt = 0.0
        ee_to_load = 0.0
        ee_to_batt = 0.0
        ee_export = 0.0
        batt_discharge = 0.0

        if action == 0:
            ee_to_load = min(gen, demand)
            rest = max(gen - demand, 0.0)
            charge = min(self.charging_rate, self.battery_capacity - self.soc, rest)
            ee_to_batt = charge
            self.soc += charge
            ee_export = max(rest - charge, 0.0)
            grid_to_load = max(demand - gen, 0.0)

        elif action == 1:
            ee_to_load = min(gen, demand)
            rest = max(gen - demand, 0.0)
            charge_ee = min(self.charging_rate, self.battery_capacity - self.soc, rest)
            ee_to_batt = charge_ee
            self.soc += charge_ee
            rest_batt = self.battery_capacity - self.soc
            charge_grid = min(max(self.charging_rate - charge_ee, 0.0), rest_batt)
            grid_to_batt = charge_grid
            self.soc += charge_grid
            grid_to_load = max(demand - gen, 0.0)

        elif action == 2:
            ee_to_load = min(gen, demand)
            rest = max(gen - demand, 0.0)
            charge = min(self.charging_rate, self.battery_capacity - self.soc, rest)
            ee_to_batt = charge
            self.soc += charge
            grid_to_load = max(demand - gen, 0.0)

        elif action == 3:
            ee_to_load = min(gen, demand)
            rest_demand = max(demand - gen, 0.0)
            discharge = min(self.discharge_rate, self.soc, rest_demand)
            batt_discharge = discharge
            self.soc -= discharge
            grid_to_load = max(rest_demand - discharge, 0.0)

        elif action == 4:
            ee_to_load = min(gen, demand)
            grid_to_load = max(demand - gen, 0.0)

        elif action == 5:
            ee_to_load = min(gen, demand)
            grid_to_load = max(demand - gen, 0.0)
            charge = min(self.charging_rate, self.battery_capacity - self.soc)
            grid_to_batt = charge
            self.soc += charge

        cost = (grid_to_load + grid_to_batt) * price - self.export_price_factor * ee_export * price
        reward = -float(cost)

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
            "SOC": self.soc,
        }
        return obs, reward, terminated, truncated, info


# -----------------------------
# DRQN (Dueling Double DQN mit GRU)
# -----------------------------
class DuelingGRUQ(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_size: int = 256):
        super().__init__()
        self.fc_in = nn.Linear(obs_dim, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        # Dueling Heads
        self.adv = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, act_dim))
        self.val = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1))

    def forward(self, x: torch.Tensor, h: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, obs_dim]
        z = torch.relu(self.fc_in(x))
        out, h_next = self.gru(z, h)  # out: [B, T, H]
        A = self.adv(out)             # [B, T, A]
        V = self.val(out)             # [B, T, 1]
        Q = V + (A - A.mean(dim=-1, keepdim=True))
        return Q, h_next

    def init_hidden(self, batch_size: int = 1, device: str = "cpu"):
        return torch.zeros(1, batch_size, self.gru.hidden_size, device=device)


class SequenceReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.acts = np.zeros((capacity,), dtype=np.int64)
        self.rews = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.ep_id = np.zeros((capacity,), dtype=np.int64)
        self.ptr = 0
        self.size = 0
        self.cur_ep = 0

    def add(self, o, a, r, no, done):
        i = self.ptr
        self.obs[i] = o
        self.acts[i] = a
        self.rews[i] = r
        self.next_obs[i] = no
        self.dones[i] = float(done)
        self.ep_id[i] = self.cur_ep
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        if done:
            self.cur_ep += 1

    def sample_sequences(self, batch_size: int, seq_len: int) -> Dict[str, torch.Tensor]:
        # wähle Startindizes, die Seq innerhalb einer Episode halten
        starts = []
        trials = 0
        while len(starts) < batch_size and trials < batch_size * 20:
            s = random.randint(0, self.size - seq_len - 1)
            if np.all(self.ep_id[s : s + seq_len] == self.ep_id[s]):
                starts.append(s)
            trials += 1
        if len(starts) < batch_size:
            return None

        def stack(arr, dtype):
            batch = np.stack([arr[s : s + seq_len] for s in starts], axis=0)
            return torch.as_tensor(batch, dtype=dtype)

        return {
            "obs": stack(self.obs, torch.float32),             # [B, T, obs]
            "acts": stack(self.acts, torch.long),              # [B, T]
            "rews": stack(self.rews, torch.float32),           # [B, T]
            "next_obs": stack(self.next_obs, torch.float32),   # [B, T, obs]
            "dones": stack(self.dones, torch.float32),         # [B, T]
        }


class DRQNAgent:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_size: int = 256,
        gamma: float = 0.99,
        lr: float = 1e-4,
        buffer_size: int = 1_000_000,
        seq_len: int = 96,
        burn_in: int = 24,
        device: str = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.seq_len = seq_len
        self.burn_in = burn_in
        self.q = DuelingGRUQ(obs_dim, act_dim, hidden_size).to(self.device)
        self.q_target = DuelingGRUQ(obs_dim, act_dim, hidden_size).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.optim = optim.Adam(self.q.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss(reduction="none")
        self.replay = SequenceReplayBuffer(buffer_size, obs_dim)
        self.act_dim = act_dim
        self.train_steps = 0

    @torch.no_grad()
    def act(self, obs: np.ndarray, h: torch.Tensor, eps: float) -> Tuple[int, torch.Tensor]:
        if random.random() < eps:
            a = random.randint(0, self.act_dim - 1)
            # GRU-State bleibt unverändert bei zufälliger Aktion
            return a, h
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).view(1, 1, -1)
        q_seq, h_next = self.q(o, h)
        a = int(torch.argmax(q_seq[0, -1]).item())
        return a, h_next

    def update(self, batch: Dict[str, torch.Tensor]):
        obs = batch["obs"].to(self.device)           # [B, T, obs]
        acts = batch["acts"].to(self.device)         # [B, T]
        rews = batch["rews"].to(self.device)         # [B, T]
        next_obs = batch["next_obs"].to(self.device) # [B, T, obs]
        dones = batch["dones"].to(self.device)       # [B, T]

        B, T, _ = obs.shape
        assert T == self.seq_len

        h0 = self.q.init_hidden(batch_size=B, device=self.device)
        # Vorwärmen
        with torch.no_grad():
            _, h_after_burn = self.q(obs[:, : self.burn_in], h0)

        q_seq, _ = self.q(obs[:, self.burn_in :], h_after_burn)             # [B, T-B, A]
        q_taken = q_seq.gather(-1, acts[:, self.burn_in :].unsqueeze(-1)).squeeze(-1)  # [B, T-B]

        with torch.no_grad():
            # Double DQN: Online wählt, Target bewertet
            q_next_online, _ = self.q(next_obs[:, self.burn_in :], h_after_burn)
            a_next = torch.argmax(q_next_online, dim=-1)  # [B, T-B]
            q_next_target, _ = self.q_target(next_obs[:, self.burn_in :], h_after_burn)
            q_next = q_next_target.gather(-1, a_next.unsqueeze(-1)).squeeze(-1)  # [B, T-B]
            targets = rews[:, self.burn_in :] + self.gamma * (1.0 - dones[:, self.burn_in :]) * q_next

        loss = self.loss_fn(q_taken, targets).mean()
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
        self.optim.step()
        self.train_steps += 1
        return float(loss.item())

    def soft_update(self, tau: float = 0.005):
        with torch.no_grad():
            for p, pt in zip(self.q.parameters(), self.q_target.parameters()):
                pt.data.mul_(1 - tau).add_(tau * p.data)

    def save(self, path: str):
        torch.save({"q": self.q.state_dict(), "q_target": self.q_target.state_dict()}, path)

    def load(self, path: str, map_location=None):
        state = torch.load(path, map_location=map_location or self.device)
        self.q.load_state_dict(state["q"])
        self.q_target.load_state_dict(state["q_target"])


# -----------------------------
# TRAINING
# -----------------------------
def reinforcement_learning_train(
    df_train: pd.DataFrame,
    battery_capacity: float,
    initial_battery: float,
    charging_rate: float,
    discharge_rate: float,
    export_price_factor: float,
    total_timesteps: int,
    seq_len: int = 96,
    burn_in: int = 24,
    batch_size: int = 32,
    target_update_period: int = 2000,
    epsilon_start: float = 1.0,
    epsilon_final: float = 0.05,
    epsilon_decay_steps: int = 500_000,
):
    env = EMSenv(
        df_train,
        battery_capacity=battery_capacity,
        initial_battery=initial_battery,
        charging_rate=charging_rate,
        discharge_rate=discharge_rate,
        export_price_factor=export_price_factor,
    )

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    agent = DRQNAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_size=256,
        gamma=0.99,
        lr=1e-4,
        buffer_size=1_000_000,
        seq_len=seq_len,
        burn_in=burn_in,
    )

    eps = epsilon_start
    eps_decay = (epsilon_start - epsilon_final) / max(1, epsilon_decay_steps)

    # Sammeln + Lernen
    obs, _ = env.reset()
    h = agent.q.init_hidden(batch_size=1, device=agent.device)
    steps = 0
    losses = []

    while steps < total_timesteps:
        action, h = agent.act(obs, h, eps)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.replay.add(obs, action, reward, next_obs, done)

        obs = next_obs
        steps += 1

        if eps > epsilon_final:
            eps -= eps_decay

        # Episode zu Ende → neue Episode
        if done:
            obs, _ = env.reset()
            h = agent.q.init_hidden(batch_size=1, device=agent.device)

        # Updates sobald genug Daten
        if agent.replay.size > (seq_len * 10):
            batch = agent.replay.sample_sequences(batch_size, seq_len)
            if batch is not None:
                loss = agent.update(batch)
                losses.append(loss)

        # Target-Update (soft)
        if agent.train_steps % target_update_period == 0 and agent.train_steps > 0:
            agent.soft_update(tau=0.005)

    # Speichern
    timestamp = datetime.now().strftime("%Y-%m-%d_%H_%M")
    model_path = f"model/trained_drqn_{timestamp}.pt"
    agent.save(model_path)
    return agent


# -----------------------------
# TESTING mit optionalem Pre-Context
# -----------------------------
def get_window_indices(df, forecast_horizon, stepsize):
    window_size = forecast_horizon
    indices = []
    start = 0
    while start + window_size <= len(df):
        end = start + window_size
        indices.append((start, end))
        start += stepsize
    if indices and indices[-1][1] < len(df):
        indices.append((len(df) - window_size, len(df)))
    return indices


@torch.no_grad()
def reinforcement_learning_test(
    model: DRQNAgent,
    df_test: pd.DataFrame,
    env_class,
    env_kwargs: dict,
    forecast_horizon: int,
    stepsize: int,
    initial_battery: float,
    precontext_hours: int = 0,   # optionales Warmup für Hidden-State
):
    test_indices = get_window_indices(df_test, forecast_horizon, stepsize)
    soc_test = float(initial_battery)
    results = []

    num_windows = len(test_indices)
    device = model.device

    for i, (test_start, test_end) in enumerate(test_indices):
        env = env_class(
            df_test.iloc[test_start:test_end],
            **{**env_kwargs, "initial_battery": soc_test},
        )
        obs, _ = env.reset()
        h = model.q.init_hidden(batch_size=1, device=device)

        # Optional: Hidden-State mit Pre-Context „aufwärmen“ (nur Beobachtungen, keine Zustandsänderung)
        if precontext_hours > 0:
            pc_start = max(0, test_start - precontext_hours)
            pc_df = df_test.iloc[pc_start:test_start]
            if len(pc_df) > 0:
                soc_dummy = soc_test
                pc_obs_seq = []
                for _, r in pc_df.iterrows():
                    pc_obs_seq.append([soc_dummy, r["energy_demand"], r["total_energy_production"], r["spotprice"]])
                pc_obs_seq = torch.as_tensor(pc_obs_seq, dtype=torch.float32, device=device).view(1, -1, 4)
                _, h = model.q(pc_obs_seq, h)

        terminated = False
        truncated = False
        window_rows = []
        while not (terminated or truncated):
            # Greedy Aktion
            o_t = torch.as_tensor(obs, dtype=torch.float32, device=device).view(1, 1, -1)
            q_seq, h = model.q(o_t, h)
            action = int(torch.argmax(q_seq[0, -1]).item())

            obs, reward, terminated, truncated, info = env.step(action)
            idx = env.idx - 1
            if idx < (test_end - test_start):
                row = df_test.iloc[test_start + idx]
                window_rows.append(
                    {
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
                        "solver_status": "RL_DRQN",
                    }
                )

        if window_rows:
            soc_test = window_rows[-1]["SOC_MWh"]

        if i < num_windows - 1:
            if window_rows:
                results.extend(window_rows[:stepsize])
        else:
            results.extend(window_rows)

    return pd.DataFrame(results)
