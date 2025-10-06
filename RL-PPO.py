import os
import random
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt

from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent, IrrigationManagement

warnings.filterwarnings('ignore')
os.environ['DEVELOPMENT'] = 'True'
torch.set_num_threads(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42
def set_seeds(seed: int = SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seeds(SEED)

IS_COLAB = os.path.isdir("/content")
OUTPUT_DIR = os.path.abspath(os.getenv("OUTPUT_DIR", "/content" if IS_COLAB else os.getcwd()))
os.makedirs(OUTPUT_DIR, exist_ok=True)

def _abs_and_prepare(path_like: str):
    path_abs = os.path.abspath(path_like)
    os.makedirs(os.path.dirname(path_abs) or ".", exist_ok=True)
    return path_abs

print(f"[INFO] Dispositivo   : {device}")
print(f"[INFO] CWD           : {os.getcwd()}")
print(f"[INFO] OUTPUT_DIR    : {OUTPUT_DIR}")
print(f"[INFO] Existe OUTPUT?: {os.path.isdir(OUTPUT_DIR)}")

def make_custom_soil_and_crop():
    custom_soil = Soil('custom', cn=46, rew=7)
    custom_soil.add_layer(thickness=0.1, thWP=0.268, thFC=0.434, thS=0.620, Ksat=47.2, penetrability=100)
    custom_soil.add_layer(thickness=0.1, thWP=0.279, thFC=0.439, thS=0.618, Ksat=14.3, penetrability=100)
    custom_soil.add_layer(thickness=0.1, thWP=0.301, thFC=0.480, thS=0.618, Ksat=9.1, penetrability=100)
    custom_soil.add_layer(thickness=0.1, thWP=0.293, thFC=0.500, thS=0.611, Ksat=8.0, penetrability=100)
    custom_crop = Crop(
        'custom',
        CropType=1,
        PlantMethod=0,
        CalendarType=1,
        SwitchGDD=0,
        planting_date='05/01',
        harvest_date='05/30',
        EmergenceCD=8,
        MaxRootingCD=20,
        SenescenceCD=29,
        MaturityCD=30,
        HIstart=6,
        FloweringCD=0,
        YldFormCD=7,
        Tbase=10.0,
        Tupp=30.0,
        Zmin=0.10,
        Zmax=0.30,
        fshape_r=15,
        SxTopQ=0.06,
        SxBotQ=0.012,
        CCx=0.85,
        CGC_CD=0.9,
        CDC_CD=0.08,
        Kcb=1.10,
        WP=1700,
        WPy=100,
        fsink=1,
        HI0=85,
        fshape_w1=2.5,
        fshape_w2=3.0,
        fshape_w3=3.0,
        fshape_w4=1.0,
        p_up1=0.4,
        p_up2=0.50,
        p_up3=0.85,
        p_up4=0.90,
        Tmin_up=8,
        Tmax_up=40,
        Aer=5,
        LagAer=3,
        PlantPop=160000,
        SeedSize=15,
        p_lo1=0.20,
        p_lo2=0.25,
        p_lo3=0.30,
        p_lo4=0.45,
        Determinant=1,
        HIstartCD=15,
        YldWC=90,
    )
    initWC = InitialWaterContent(wc_type='Pct', value=[40])
    return custom_soil, custom_crop, initWC

def _drop_feb29(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mask_29feb = (df['Date'].dt.month == 2) & (df['Date'].dt.day == 29)
    if mask_29feb.any():
        n = int(mask_29feb.sum())
        print(f"[FIX] Removendo {n} linha(s) de 29/02 do clima para compatibilidade com AquaCrop.")
        df = df.loc[~mask_29feb].reset_index(drop=True)
    return df

def read_excel_decimal_comma(path):
    df = pd.read_excel(path, dtype=str)
    expected = ['MinTemp','MaxTemp','Precipitation','ReferenceET','Date']
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes no Excel: {missing}. Esperado: {expected}")
    for col in ['MinTemp','MaxTemp','Precipitation','ReferenceET']:
        df[col] = (df[col].str.replace(',', '.', regex=False).str.replace(' ', '', regex=False).astype(float))
    df['Date'] = pd.to_datetime(df['Date'], errors='raise')
    df = df.sort_values('Date').reset_index(drop=True)
    return _drop_feb29(df)

def _to_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    if not np.issubdtype(df2['Date'].dtype, np.datetime64):
        df2['Date'] = pd.to_datetime(df2['Date'], errors='raise')
    df2['Date'] = df2['Date'].dt.floor('D')
    return df2

class RunningNorm:
    def __init__(self, dim: int, eps: float = 1e-6):
        self.dim = dim; self.eps = eps
        self.mean = np.zeros(dim, dtype=np.float64)
        self.M2   = np.ones(dim, dtype=np.float64)
        self.count = 1.0
    def update(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        assert x.shape[-1] == self.dim
        self.count += 1.0
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2
    def std(self):
        var = self.M2 / max(self.count - 1.0, 1.0)
        return np.sqrt(np.maximum(var, self.eps))
    def normalize(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return (x - self.mean) / (self.std() + self.eps)
    def state_dict(self):
        return {'dim': self.dim,'eps': self.eps,'mean': self.mean,'M2': self.M2,'count': self.count}
    def load_state_dict(self, state):
        assert state['dim'] == self.dim
        self.eps = float(state['eps'])
        self.mean = np.array(state['mean'], dtype=np.float64)
        self.M2   = np.array(state['M2'],   dtype=np.float64)
        self.count = float(state['count'])

class AquaCropRealEnv:
    def __init__(self,
                 climate_data,
                 is_path=True,
                 max_irrigation=600.0,
                 culture_name='custom',
                 cost_per_mm=0.003,
                 stress_w=0.8,
                 deep_perc_w=0.05,
                 rain_w=0.02,
                 custom_soil=None,
                 custom_crop=None,
                 init_wc=None):
        if is_path:
            df = read_excel_decimal_comma(climate_data)
        else:
            df = climate_data.copy()
            req = {'MinTemp','MaxTemp','Precipitation','ReferenceET','Date'}
            if not req.issubset(df.columns):
                raise ValueError(f"O DataFrame deve conter as colunas: {req}")
            for col in ['MinTemp','MaxTemp','Precipitation','ReferenceET']:
                df[col] = pd.to_numeric(df[col], errors='raise')
            if not np.issubdtype(df['Date'].dtype, np.datetime64):
                df['Date'] = pd.to_datetime(df['Date'], errors='raise')
            df = df.sort_values('Date').reset_index(drop=True)
            df = _drop_feb29(df)
        self.df_climate = df
        self.total_days = len(self.df_climate)
        self.dates = self.df_climate['Date'].tolist()
        self.sim_start = self.dates[0].strftime('%Y/%m/%d')
        self.sim_end_all = self.dates[-1].strftime('%Y/%m/%d')
        print(f"Período de simulação: {self.sim_start} a {self.sim_end_all} | Dias: {self.total_days}")
        if (custom_soil is None) or (custom_crop is None) or (init_wc is None):
            self.custom_soil, self.crop, self.fc = make_custom_soil_and_crop()
        else:
            self.custom_soil, self.crop, self.fc = custom_soil, custom_crop, init_wc
        self.max_irrigation = float(max_irrigation)
        self.action_space = [0.0, 25.0]
        self.observation_space_shape = 18
        self.cost_per_mm = float(cost_per_mm)
        self.stress_w = float(stress_w)
        self.deep_perc_w = float(deep_perc_w)
        self.rain_w = float(rain_w)
        self.current_day = 0
        self.irrigation_schedule = []
        self.last_metrics = {'Tr': 0.0, 'TrPot': 1e-6, 'ratio': 1.0, 'CC': 0.0, 'Dp': 0.0}

    def _span_until(self, day_idx):
        if self.total_days == 0:
            raise RuntimeError("Clima vazio.")
        end_idx_sim = max(1, day_idx) if day_idx == 0 and self.total_days >= 2 else day_idx
        end_idx_sim = min(end_idx_sim, self.total_days - 1)
        n_rows = end_idx_sim + 1
        weather_sub = self.df_climate.iloc[:n_rows].copy()
        depth = np.zeros(n_rows, dtype=float)
        k = min(len(self.irrigation_schedule), n_rows)
        if k > 0:
            depth[:k] = np.array(self.irrigation_schedule[:k], dtype=float)
        schedule_df = pd.DataFrame({
            'Date': self.df_climate['Date'].iloc[:n_rows].to_list(),
            'Depth': depth
        })
        row_idx_to_read = day_idx
        return weather_sub, schedule_df, end_idx_sim, row_idx_to_read

    def _run_aquacrop_partial(self, end_idx_sim, weather_sub, schedule_df):
        weather_sub = _to_timestamp(weather_sub)
        schedule_df = _to_timestamp(schedule_df)
        sim_end = weather_sub['Date'].iloc[-1].strftime('%Y/%m/%d')
        model = AquaCropModel(
            sim_start_time=self.sim_start,
            sim_end_time=sim_end,
            weather_df=weather_sub,
            soil=self.custom_soil,
            crop=self.crop,
            irrigation_management=IrrigationManagement(irrigation_method=3, Schedule=schedule_df),
            initial_water_content=self.fc
        )
        model.run_model(till_termination=True)
        return model._outputs

    def reset(self):
        self.current_day = 0
        self.irrigation_schedule = []
        self.last_metrics = {'Tr': 0.0, 'TrPot': 1e-6, 'ratio': 1.0, 'CC': 0.0, 'Dp': 0.0}
        return self._get_state()

    def _get_state(self):
        if self.current_day >= self.total_days:
            return np.zeros(self.observation_space_shape, dtype=np.float32)
        row = self.df_climate.iloc[self.current_day]
        day = row['Date'].day; month = row['Date'].month
        start7 = max(0, self.current_day - 7); end = self.current_day + 1
        means = self.df_climate.iloc[start7:end][['Precipitation','ReferenceET']].mean().values
        sums  = self.df_climate.iloc[0:end][['Precipitation','ReferenceET']].sum().values
        if self.current_day > 0:
            lag1 = self.df_climate.iloc[self.current_day-1][['Precipitation','ReferenceET']].values
        else:
            lag1 = np.array([0.0, 0.0])
        ratio = float(self.last_metrics.get('ratio', 1.0))
        cc    = float(self.last_metrics.get('CC', 0.0))
        dp    = float(self.last_metrics.get('Dp', 0.0))
        dap = float(self.current_day)
        irr_cum = float(np.sum(self.irrigation_schedule)) if self.irrigation_schedule else 0.0
        remaining_irr = float(max(0.0, self.max_irrigation - irr_cum))
        f = min(max(self.current_day / max(self.total_days-1, 1), 0.0), 1.0)
        if f < 0.15: gs = 0
        elif f < 0.5: gs = 1
        elif f < 0.85: gs = 2
        else: gs = 3
        gs_1h = np.zeros(4, dtype=np.float32); gs_1h[gs] = 1.0
        obs = np.array([
            float(day), float(month), ratio, dap, irr_cum,
            cc, dp, remaining_irr,
            float(means[0]), float(means[1]), float(sums[0]), float(sums[1]),
            float(lag1[0]), float(lag1[1]),
            gs_1h[0], gs_1h[1], gs_1h[2], gs_1h[3],
        ], dtype=np.float32)
        return obs

    def _daily_reward_real(self, action_mm, last_metrics):
        reward = - self.cost_per_mm * float(action_mm)
        ratio = float(last_metrics.get('ratio', 1.0))
        stress = max(0.0, 1.0 - ratio)
        reward -= self.stress_w * stress
        if ratio >= 0.9 and float(action_mm) <= 0.01:
            reward += 0.04
        P_today = 0.0
        if self.current_day < self.total_days:
            P_today = float(self.df_climate.iloc[self.current_day]['Precipitation'])
        rain_factor = P_today / (P_today + 5.0)
        reward -= self.rain_w * float(action_mm) * rain_factor
        dp = float(last_metrics.get('Dp', 0.0))
        reward -= self.deep_perc_w * (dp / 10.0)
        return reward

    def step(self, action):
        action = float(np.clip(action, self.action_space[0], self.action_space[1]))
        irr_cum = float(np.sum(self.irrigation_schedule)) if self.irrigation_schedule else 0.0
        remaining = max(0.0, self.max_irrigation - irr_cum)
        action = float(min(action, remaining))
        self.irrigation_schedule.append(action)
        try:
            weather_sub, schedule_df, end_idx_sim, row_idx = self._span_until(self.current_day)
            outputs = self._run_aquacrop_partial(end_idx_sim, weather_sub, schedule_df)
            wf = outputs.water_flux
            cg = getattr(outputs, 'crop_growth', None)
            if wf is not None and len(wf) > row_idx:
                row_wf = wf.iloc[row_idx]
                Tr = float(row_wf['Tr']) if 'Tr' in row_wf else 0.0
                TrPot = float(row_wf['TrPot']) if 'TrPot' in row_wf else 1e-6
                Dp = float(row_wf['Dp']) if 'Dp' in row_wf else 0.0
                ratio = Tr / (TrPot + 1e-8)
            else:
                Tr, TrPot, Dp, ratio = 0.0, 1e-6, 0.0, 0.8
            CC = 0.0
            if cg is not None and len(cg) > row_idx and 'CC' in cg.columns:
                CC = float(cg.iloc[row_idx]['CC'])
            self.last_metrics = {'Tr': Tr, 'TrPot': TrPot, 'ratio': ratio, 'CC': CC, 'Dp': Dp}
        except Exception:
            self.last_metrics = {'Tr': 0.0, 'TrPot': 1e-6, 'ratio': 0.8, 'CC': 0.0, 'Dp': 0.0}
        daily_reward = self._daily_reward_real(action, self.last_metrics)
        self.current_day += 1
        done = self.current_day >= self.total_days
        if done:
            final_reward = self._final_reward_real_fullseason()
            total_reward = float(daily_reward + final_reward)
            next_state = np.zeros(self.observation_space_shape, dtype=np.float32)
        else:
            total_reward = float(daily_reward)
            next_state = self._get_state()
        return next_state, total_reward, bool(done), {}

    def _final_reward_real_fullseason(self):
        try:
            depth = np.zeros(self.total_days, dtype=float)
            n = min(self.total_days, len(self.irrigation_schedule))
            if n > 0:
                depth[:n] = np.array(self.irrigation_schedule[:n], dtype=float)
            schedule_df = pd.DataFrame({'Date': self.df_climate['Date'], 'Depth': depth})
            schedule_df = _to_timestamp(schedule_df)
            weather_df = _to_timestamp(self.df_climate)
            model = AquaCropModel(
                sim_start_time=self.sim_start,
                sim_end_time=self.sim_end_all,
                weather_df=weather_df,
                soil=self.custom_soil,
                crop=self.crop,
                irrigation_management=IrrigationManagement(irrigation_method=3, Schedule=schedule_df),
                initial_water_content=self.fc
            )
            model.run_model(till_termination=True)
            out = model._outputs
            dry_yield = np.nan
            seasonal_irr = float(np.nansum(depth))
            stress_mean = 0.0
            wf = getattr(out, 'water_flux', None)
            if wf is not None and len(wf) > 0:
                rat = (wf['Tr'].to_numpy(dtype=float)) / (wf['TrPot'].to_numpy(dtype=float) + 1e-8)
                stress_mean = float(np.mean(np.maximum(0.0, 1.0 - rat)))
            fs = getattr(out, 'final_stats', None)
            if fs is not None and len(fs) > 0 and 'Dry yield (tonne/ha)' in fs.columns:
                dry_yield = float(fs['Dry yield (tonne/ha)'].values[0])
            if not np.isfinite(dry_yield):
                dry_yield = max(0.0, 6.0 * (1.0 - stress_mean))
            wy, wi, ws = 2.0, 0.5, 0.5
            yield_norm = 1.0 / (1.0 + np.exp(-0.12 * (dry_yield - 3.0)))
            water_use = float(seasonal_irr / max(self.max_irrigation, 1.0))
            final_reward = wy * yield_norm - wi * water_use - ws * stress_mean
            return float(final_reward)
        except Exception:
            return -0.1

class ActorNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, 1)
        self.std_head = nn.Linear(hidden_dim, 1)
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0)
        nn.init.orthogonal_(self.mean.weight, gain=0.01)
        nn.init.constant_(self.mean.bias, -2.0)
        nn.init.orthogonal_(self.std_head.weight, gain=0.01)
        nn.init.constant_(self.std_head.bias, -1.2)
    def forward(self, state):
        x = self.net(state)
        mean = 12.5 + 12.5 * torch.tanh(self.mean(x) / 2.0)
        log_std = torch.clamp(self.std_head(x), -1.6, -0.2)
        std = torch.exp(log_std)
        return mean, std

class CriticNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0)
    def forward(self, state):
        return self.net(state).squeeze(-1)

class PPO:
    def __init__(self, env,
                 learning_rate=2e-4,
                 batch_size=32,
                 gamma=0.99,
                 gae_lambda=0.95,
                 clip_range=0.15,
                 ent_coef=0.05,
                 vf_coef=0.5,
                 max_grad_norm=0.5,
                 n_epochs=4,
                 target_kl=0.05,
                 min_kl_epochs=2,
                 min_kl_batches=2,
                 verbose=1):
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.min_kl_epochs = min_kl_epochs
        self.min_kl_batches = min_kl_batches
        self.verbose = verbose
        self.actor = ActorNetwork(env.observation_space_shape).to(device)
        self.critic = CriticNetwork(env.observation_space_shape).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate, eps=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate, eps=1e-5)
        self.states, self.actions, self.rewards, self.dones, self.log_probs, self.values = [], [], [], [], [], []
        self.obs_rms = RunningNorm(env.observation_space_shape)

    def _normalize_state(self, s_np: np.ndarray) -> np.ndarray:
        self.obs_rms.update(s_np)
        return self.obs_rms.normalize(s_np).astype(np.float32)

    def select_action(self, state_np):
        state_np_n = self._normalize_state(state_np)
        st = torch.from_numpy(state_np_n).unsqueeze(0).to(device)
        with torch.no_grad():
            mean, std = self.actor(st)
            dist = Normal(mean, std)
            a = dist.sample()
            lp = dist.log_prob(a).sum(dim=-1)
            v = self.critic(st)
        return float(a.item()), float(lp.item()), float(v.item()), state_np_n

    def store_transition(self, state_n, action, reward, done, log_prob, value):
        self.states.append(state_n)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def compute_gae(self):
        advantages = []
        gae = 0.0
        for t in reversed(range(len(self.rewards))):
            next_non_terminal = 1.0 - float(self.dones[t])
            next_value = self.values[t] if t == len(self.rewards)-1 else self.values[t+1]
            delta = self.rewards[t] + self.gamma * next_value * next_non_terminal - self.values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages.insert(0, gae)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
        values_t  = torch.tensor(self.values, dtype=torch.float32, device=device)
        returns   = advantages + values_t
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def _sync_buffers(self):
        lens = [len(self.states), len(self.actions), len(self.rewards), len(self.dones), len(self.log_probs), len(self.values)]
        m = min(lens)
        if len(set(lens)) != 1 and self.verbose:
            print(f"[WARN] Buffers desalinhados {lens}, truncando para {m}")
        self.states   = self.states[:m]
        self.actions  = self.actions[:m]
        self.rewards  = self.rewards[:m]
        self.dones    = self.dones[:m]
        self.log_probs= self.log_probs[:m]
        self.values   = self.values[:m]
        return m

    def update(self):
        N = self._sync_buffers()
        if N < self.batch_size:
            self._clear_buffers(); return
        N_eff = (N // self.batch_size) * self.batch_size
        if N_eff < self.batch_size:
            self._clear_buffers(); return
        if N_eff != N and self.verbose:
            print(f"[INFO] Usando N_eff={N_eff} de N={N} (descartando cauda)")
        old_states = torch.from_numpy(np.array(self.states[:N_eff], dtype=np.float32)).to(device)
        old_actions = torch.from_numpy(np.array(self.actions[:N_eff], dtype=np.float32)).unsqueeze(-1).to(device)
        old_log_probs = torch.from_numpy(np.array(self.log_probs[:N_eff], dtype=np.float32)).to(device)
        advantages, returns = self.compute_gae()
        advantages = advantages[:N_eff]; returns = returns[:N_eff]
        idxs = np.arange(N_eff)
        for epoch in range(self.n_epochs):
            np.random.shuffle(idxs)
            approx_kl_list = []; batches = 0
            for start in range(0, N_eff, self.batch_size):
                batch_idx = idxs[start:start+self.batch_size]
                bs = old_states[batch_idx]
                ba = old_actions[batch_idx]
                bold_lp = old_log_probs[batch_idx]
                badv = advantages[batch_idx]
                bret = returns[batch_idx]
                values = self.critic(bs)
                critic_loss = 0.5 * (values - bret).pow(2).mean()
                means, stds = self.actor(bs)
                dist = Normal(means, stds)
                new_log_probs = dist.log_prob(ba).sum(dim=-1)
                ratio = (new_log_probs - bold_lp).exp()
                surr1 = ratio * badv
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * badv
                actor_loss = -torch.min(surr1, surr2).mean()
                entropy = dist.entropy().mean()
                total_loss = actor_loss + self.vf_coef * critic_loss - self.ent_coef * entropy
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                with torch.no_grad():
                    approx_kl = (bold_lp - new_log_probs).mean().clamp(min=0.0).item()
                    approx_kl_list.append(approx_kl)
                batches += 1
            if (self.target_kl is not None and epoch + 1 >= self.min_kl_epochs and batches >= self.min_kl_batches):
                mean_kl = float(np.mean(approx_kl_list)) if len(approx_kl_list)>0 else 0.0
                if mean_kl > self.target_kl and self.verbose:
                    print(f"[PPO] Early stop por KL: {mean_kl:.4f} > {self.target_kl:.4f} (epoch {epoch+1})")
                    break
        self._clear_buffers()

    def _clear_buffers(self):
        self.states, self.actions, self.rewards, self.dones, self.log_probs, self.values = [], [], [], [], [], []

    def train_fast(self, num_episodes, train_df_full: pd.DataFrame, episode_days=64,
                   log_every=100, save_every=500):
        rewards_history = []
        best_reward = -np.inf
        def sample_slice_raw(df, days):
            total = len(df)
            if days >= total:
                s = 0; e = total
            else:
                s = random.randint(0, total - days)
                e = s + days
            return df.iloc[s:e].reset_index(drop=True)
        def sample_slice_same_year(df, days, max_tries=20):
            for _ in range(max_tries):
                sl = sample_slice_raw(df, days)
                years = pd.to_datetime(sl['Date']).dt.year.values
                if years[0] == years[-1]:
                    return sl
            return sample_slice_raw(df, days)
        def is_slice_ok(sl_df) -> bool:
            try:
                req_cols = ['Date','Precipitation','ReferenceET']
                if any(c not in sl_df.columns for c in req_cols): return False
                if sl_df['Date'].isna().any(): return False
                if sl_df[['Precipitation','ReferenceET']].isna().any().any(): return False
                if (sl_df[['Precipitation','ReferenceET']] < 0).any().any(): return False
                if not sl_df['Date'].is_monotonic_increasing: return False
                soil, crop, wc = make_custom_soil_and_crop()
                env_probe = AquaCropRealEnv(sl_df, is_path=False, max_irrigation=600.0,
                                            culture_name='custom',
                                            custom_soil=soil, custom_crop=crop, init_wc=wc)
                idx = 1 if env_probe.total_days > 1 else 0
                wsub, ssch, eidx, ridx = env_probe._span_until(idx)
                out = env_probe._run_aquacrop_partial(eidx, wsub, ssch)
                wf = getattr(out, 'water_flux', None)
                if wf is None or len(wf) == 0: return False
                tr = wf['Tr'].to_numpy(dtype=float)
                trp = wf['TrPot'].to_numpy(dtype=float) + 1e-8
                ratio = tr / trp
                if not np.isfinite(ratio).all(): return False
                if (ratio < -0.05).any() or (ratio > 1.2).any(): return False
                return True
            except Exception:
                return False
        for ep in range(1, int(num_episodes)+1):
            tries, MAX_TRIES = 0, 12
            while True:
                df_slice = sample_slice_same_year(train_df_full, episode_days)
                if is_slice_ok(df_slice):
                    break
                tries += 1
                if tries >= MAX_TRIES:
                    alt_days = min(len(train_df_full), episode_days + 16)
                    df_slice = sample_slice_raw(train_df_full, alt_days)
                    _ = is_slice_ok(df_slice)
                    break
            soil, crop, wc = make_custom_soil_and_crop()
            self.env = AquaCropRealEnv(df_slice, is_path=False, max_irrigation=600.0,
                                       culture_name='custom',
                                       custom_soil=soil, custom_crop=crop, init_wc=wc)
            state = self.env.reset()
            ep_reward = 0.0; done = False
            while not done:
                action, logp, value, state_n = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.store_transition(state_n, action, reward, done, logp, value)
                ep_reward += reward
                state = next_state
            self.update()
            rewards_history.append(ep_reward)
            if ep_reward > best_reward:
                best_reward = ep_reward
                obs_state = self.obs_rms.state_dict()
                obs_norm_safe = {
                    'dim': int(obs_state['dim']),
                    'eps': float(obs_state['eps']),
                    'mean': np.asarray(obs_state['mean'], dtype=np.float64).tolist(),
                    'M2':   np.asarray(obs_state['M2'],   dtype=np.float64).tolist(),
                    'count': float(obs_state['count']),
                }
                ckpt_path = _abs_and_prepare(os.path.join(OUTPUT_DIR, 'best_model.pth'))
                torch.save({
                    'actor': self.actor.state_dict(),
                    'critic': self.critic.state_dict(),
                    'reward': float(ep_reward),
                    'episode': int(ep),
                    'obs_norm': obs_norm_safe,
                }, ckpt_path)
            if (ep % log_every) == 0:
                avg100 = np.mean(rewards_history[-100:]) if len(rewards_history) >= 1 else ep_reward
                print(f"[TRAIN] Ep {ep}/{num_episodes} | Reward: {ep_reward:.2f} | Avg100: {avg100:.2f} | Best: {best_reward:.2f}")
            if (ep % save_every) == 0:
                path = _abs_and_prepare(os.path.join(OUTPUT_DIR, f"model_ep{ep}.pth"))
                torch.save({'actor': self.actor.state_dict(),
                            'critic': self.critic.state_dict(),
                            'obs_norm': self.obs_rms.state_dict()}, path)
        hist_csv = _abs_and_prepare(os.path.join(OUTPUT_DIR, "rewards_history.csv"))
        pd.DataFrame({'episode': np.arange(1, len(rewards_history)+1),
                      'reward': rewards_history}).to_csv(hist_csv, index=False)
        print(f"[OK] Histórico salvo em: {hist_csv}")
        return rewards_history

def plot_training_results(rewards_history, window=100, save_path=None):
    if save_path is None:
        save_path = _abs_and_prepare(os.path.join(OUTPUT_DIR, "training_curve.png"))
    plt.figure(figsize=(12,5))
    if len(rewards_history) > 0:
        xs = np.arange(1, len(rewards_history)+1)
        plt.plot(xs, rewards_history, alpha=0.35, label='Recompensa/Episódio')
        if len(rewards_history) >= 2:
            mv = [np.mean(rewards_history[max(0, i-window):i+1]) for i in range(len(rewards_history))]
            plt.plot(xs, mv, linewidth=2, label=f'Média Móvel ({window})')
        best_idx = int(np.argmax(rewards_history))
        plt.scatter(xs[best_idx], rewards_history[best_idx], s=80, label=f'Melhor={rewards_history[best_idx]:.2f}')
    plt.title('Treinamento — Recompensa por Episódio (AquaCrop real em cada passo)')
    plt.xlabel('Episódio'); plt.ylabel('Recompensa'); plt.grid(True); plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=140)
    print(f"[OK] Curva de treinamento salva em: {save_path}")

def actor_deterministic_action(actor, state_np, obs_norm: RunningNorm):
    state_np_n = obs_norm.normalize(state_np).astype(np.float32)
    with torch.no_grad():
        st = torch.from_numpy(state_np_n).unsqueeze(0).to(device)
        mean, std = actor(st)
        action = mean.squeeze(0).squeeze(-1).item()
    return float(np.clip(action, 0.0, 25.0))

def _load_checkpoint_compat(model_ckpt_path):
    try:
        ckpt = torch.load(model_ckpt_path, map_location=device)
        return ckpt, "safe"
    except Exception as e:
        print(f"[load] Falha com weights_only=True: {e}\nTentando weights_only=False (use apenas com fonte confiável).")
        ckpt = torch.load(model_ckpt_path, map_location=device, weights_only=False)
        return ckpt, "unsafe"

def evaluate_fullseason(env_eval: AquaCropRealEnv, actions_mm: np.ndarray):
    try:
        total_days = env_eval.total_days
        depth = np.zeros(total_days, dtype=float)
        n = min(total_days, len(actions_mm))
        if n > 0:
            depth[:n] = np.array(actions_mm[:n], dtype=float)
        schedule_df = pd.DataFrame({'Date': env_eval.df_climate['Date'], 'Depth': depth})
        schedule_df = _to_timestamp(schedule_df)
        weather_df = _to_timestamp(env_eval.df_climate)
        model = AquaCropModel(
            sim_start_time=env_eval.sim_start,
            sim_end_time=env_eval.sim_end_all,
            weather_df=weather_df,
            soil=env_eval.custom_soil,
            crop=env_eval.crop,
            irrigation_management=IrrigationManagement(irrigation_method=3, Schedule=schedule_df),
            initial_water_content=env_eval.fc
        )
        model.run_model(till_termination=True)
        out = model._outputs
        dry_yield = np.nan
        seasonal_irr = float(np.nansum(depth))
        stress_mean = 0.0
        wf = getattr(out, 'water_flux', None)
        if wf is not None and len(wf) > 0:
            rat = (wf['Tr'].to_numpy(dtype=float)) / (wf['TrPot'].to_numpy(dtype=float) + 1e-8)
            stress_mean = float(np.mean(np.maximum(0.0, 1.0 - rat)))
        fs = getattr(out, 'final_stats', None)
        if fs is not None and len(fs) > 0 and 'Dry yield (tonne/ha)' in fs.columns:
            dry_yield = float(fs['Dry yield (tonne/ha)'].values[0])
        if not np.isfinite(dry_yield):
            dry_yield = max(0.0, 6.0 * (1.0 - stress_mean))
        return {'dry_yield_t_ha': float(dry_yield),
                'seasonal_irrigation_mm': float(seasonal_irr),
                'stress_mean': float(stress_mean)}
    except Exception as e:
        print(f"[eval] Erro na avaliação de temporada: {e}")
        return {'dry_yield_t_ha': float('nan'),
                'seasonal_irrigation_mm': float('nan'),
                'stress_mean': float('nan')}

def recommend_and_evaluate(new_file_path,
                           model_ckpt_path=None,
                           save_csv_path=None,
                           save_excel_path=None,
                           max_irrigation=600.0,
                           culture_name='custom'):
    if model_ckpt_path is None:
        model_ckpt_path = os.path.join(OUTPUT_DIR, "best_model.pth")
    model_ckpt_path = _abs_and_prepare(model_ckpt_path)
    df_new = read_excel_decimal_comma(new_file_path)
    soil, crop, wc = make_custom_soil_and_crop()
    env_new = AquaCropRealEnv(df_new, is_path=False, max_irrigation=max_irrigation,
                              culture_name=culture_name,
                              custom_soil=soil, custom_crop=crop, init_wc=wc)
    checkpoint, _ = _load_checkpoint_compat(model_ckpt_path)
    temp_actor = ActorNetwork(env_new.observation_space_shape).to(device)
    temp_actor.load_state_dict(checkpoint['actor']); temp_actor.eval()
    obs_norm = RunningNorm(env_new.observation_space_shape)
    obs_state = checkpoint.get('obs_norm', None)
    if obs_state is None:
        print("[WARN] checkpoint sem obs_norm; ações podem ficar deslocadas.")
    else:
        if isinstance(obs_state.get('mean', None), list):
            obs_norm.load_state_dict({
                'dim': obs_state['dim'], 'eps': obs_state['eps'],
                'mean': np.array(obs_state['mean'], dtype=np.float64),
                'M2':   np.array(obs_state['M2'],   dtype=np.float64),
                'count': obs_state['count'],
            })
        else:
            obs_norm.load_state_dict(obs_state)
    rec_dates, rec_actions = [], []
    state = env_new.reset(); done = False
    while not done:
        action = actor_deterministic_action(temp_actor, state, obs_norm)
        next_state, _, done, _ = env_new.step(action)
        rec_dates.append(env_new.df_climate.iloc[env_new.current_day-1]['Date'])
        rec_actions.append(action)
        state = next_state
    rec_df = pd.DataFrame({'Date': rec_dates, 'RecommendedIrrigation_mm': rec_actions})
    base = os.path.splitext(os.path.basename(new_file_path))[0]
    if save_csv_path is None:
        save_csv_path = os.path.join(OUTPUT_DIR, f"recomendacoes_{base}.csv")
    if save_excel_path is None:
        save_excel_path = os.path.join(OUTPUT_DIR, f"recomendacoes_{base}.xlsx")
    save_csv_path = _abs_and_prepare(save_csv_path); save_excel_path = _abs_and_prepare(save_excel_path)
    rec_df.to_csv(save_csv_path, index=False); rec_df.to_excel(save_excel_path, index=False)
    print(f"[OK] {base}: CSV -> {save_csv_path}")
    print(f"[OK] {base}: XLSX -> {save_excel_path}")
    stats = evaluate_fullseason(env_new, np.array(rec_actions, dtype=float))
    print(f"[METRICS] {base}: Dry yield (t/ha) = {stats['dry_yield_t_ha']:.4f} | "
          f"Seasonal irrigation (mm) = {stats['seasonal_irrigation_mm']:.1f} | "
          f"Stress médio = {stats['stress_mean']:.4f}")
    return rec_df, stats

def main():
    try:
        DATA_PATH = "Dados_DV_ultimos2anos.xlsx"
        EPISODES = 15000
        EPISODE_DAYS = 64
        LOG_EVERY = 100
        SAVE_EVERY = 500
        EVAL_AFTER_TRAIN = True
        train_df_full = read_excel_decimal_comma(DATA_PATH)
        init_slice = train_df_full.iloc[:EPISODE_DAYS].reset_index(drop=True)
        soil, crop, wc = make_custom_soil_and_crop()
        env = AquaCropRealEnv(init_slice, is_path=False, max_irrigation=600.0,
                              culture_name='custom',
                              custom_soil=soil, custom_crop=crop, init_wc=wc)
        ppo = PPO(env,
                  learning_rate=2e-4,
                  batch_size=32,
                  gamma=0.99,
                  gae_lambda=0.95,
                  clip_range=0.15,
                  ent_coef=0.05,
                  vf_coef=0.5,
                  max_grad_norm=0.5,
                  n_epochs=4,
                  target_kl=0.05,
                  min_kl_epochs=2,
                  min_kl_batches=2,
                  verbose=1)
        print(f"\n[TRAIN] Iniciando treino: {EPISODES} episódios, {EPISODE_DAYS} dias/episódio...")
        rewards_history = ppo.train_fast(EPISODES, train_df_full, episode_days=EPISODE_DAYS,
                                         log_every=LOG_EVERY, save_every=SAVE_EVERY)
        plot_path = _abs_and_prepare(os.path.join(OUTPUT_DIR, "training_curve.png"))
        plot_training_results(rewards_history, window=100, save_path=plot_path)
        if EVAL_AFTER_TRAIN:
            print("\n[POST] Gerando recomendações determinísticas e avaliando por estação...")
            arquivos_estacoes = [
                "/content/dados_inverno100.xlsx",
                "/content/dados_outono100.xlsx",
                "/content/dados_primavera100.xlsx",
                "/content/dados_verao100.xlsx",
            ]
            relatorio = []
            for path_estacao in arquivos_estacoes:
                if not os.path.isfile(path_estacao):
                    print(f"[WARN] Arquivo não encontrado: {path_estacao} — pulando.")
                    continue
                _, stats = recommend_and_evaluate(
                    new_file_path=path_estacao,
                    model_ckpt_path=os.path.join(OUTPUT_DIR, "best_model.pth"),
                    max_irrigation=600.0,
                    culture_name='custom'
                )
                relatorio.append({'arquivo': os.path.basename(path_estacao), **stats})
            if len(relatorio) > 0:
                resumo_df = pd.DataFrame(relatorio)
                resumo_csv  = _abs_and_prepare(os.path.join(OUTPUT_DIR, "resumo_estacoes.csv"))
                resumo_xlsx = _abs_and_prepare(os.path.join(OUTPUT_DIR, "resumo_estacoes.xlsx"))
                resumo_df.to_csv(resumo_csv, index=False); resumo_df.to_excel(resumo_xlsx, index=False)
                print(f"[OK] Resumo salvo em: {resumo_csv}")
                print(f"[OK] Resumo (XLSX) salvo em: {resumo_xlsx}")
    except Exception as e:
        print(f"Erro na execução: {e}")

if __name__ == "__main__":
    main()
