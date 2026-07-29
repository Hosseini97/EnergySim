import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# GPU enabled automatically by JAX

import jax
import jax.numpy as jnp
import flax.linen as nn
import equinox as eqx
import numpy as np
import optax
from typing import Tuple, Dict, Any

from energysim.sim.simulator import JAXSimulator
from energysim.core.data.dataset import SimulationDataset
from energysim.core.shared.data_structs import (
    BatteryConfig, RewardConfig, HeatPumpConfig, AirConditionerConfig,
    ThermalStorageConfig, PVConfig, SystemActions,
)
from energysim.rl.helpers import extract_obs
import energysim.utils.objectives as obj
from build_my_house import create_2_room_house
import sample_data_generator

EXAMPLES_DIR = Path(__file__).resolve().parent
CSV_PATH = EXAMPLES_DIR / "sample_data.csv"
DT_SECONDS = 900
DEFAULT_STEPS = 672          # 7 days at 15-min resolution
PRICE_WEIGHT = 10.0           # reward scaling factor; divide raw reward sums by this to get real EUR
FORECAST_HORIZON = 96         # 24h price forecast in the observation

# ==========================================
# 1. Environment: single-agent, battery-only, price-forecast observation
# ==========================================
# This is the finalized design validated against MPC/RBC baselines: full price
# forecast in the observation, physical SOC-feasibility clipping on the action,
# and an action-smoothness penalty to curb full-power charge/discharge chattering.

class EnvState(eqx.Module):
    sim_state: Any
    time_idx: jnp.ndarray
    step_count: jnp.ndarray
    prev_action: jnp.ndarray

class BatteryPriceForecastEnv:
    """A single-agent JAX environment wrapping JAXSimulator for battery-only SAC control."""

    def __init__(self, simulator, dataset, max_steps=DEFAULT_STEPS, is_eval=False, action_smoothness_weight=0.0):
        self.sim = simulator
        self.data_len = len(dataset)
        self.max_steps = min(max_steps, self.data_len)
        self.is_eval = is_eval
        # Penalizes |action_t - action_{t-1}|^2 to discourage chattering between
        # full-charge/full-discharge on consecutive steps. 0.0 = no penalty (pure economic reward).
        self.action_smoothness_weight = action_smoothness_weight
        self.n_rooms = len(simulator.configs[0].room_air_indices)
        self.room_indices = jnp.array(simulator.configs[0].room_air_indices)
        self.max_battery_power_w = simulator.battery.config.max_power_w

        def stack_pytree(list_of_trees):
            return jax.tree.map(lambda *args: jnp.stack(args), *list_of_trees)

        history_exo_list = [dataset[i] for i in range(len(dataset))]
        self.all_exo = stack_pytree(history_exo_list)

        # n_rooms + 5 (base obs) + 1 (current price) + FORECAST_HORIZON (price forecast)
        self.obs_dim = self.n_rooms + 5 + 1 + FORECAST_HORIZON
        self.action_dim = 1

    def _get_obs_with_forecast(self, sim_state, t: jnp.ndarray) -> jnp.ndarray:
        exo = jax.tree.map(lambda x: x[t], self.all_exo)
        base_obs = extract_obs(sim_state, exo, self.room_indices)

        max_start_idx = self.data_len - FORECAST_HORIZON
        start_idx = jnp.minimum(t, max_start_idx)

        price_forecast = jax.lax.dynamic_slice_in_dim(self.all_exo.price, start_idx, FORECAST_HORIZON)

        # Safe preprocessing inside the environment
        current_price = price_forecast[0]
        relative_price_forecast = (price_forecast - current_price) / 0.50
        scaled_current_price = jnp.array([current_price / 0.50])

        return jnp.concatenate([base_obs, scaled_current_price, relative_price_forecast])

    def reset(self, key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, EnvState]:
        sim_state = self.sim.reset()
        max_start_idx = max(1, self.data_len - self.max_steps)

        start_idx = jax.lax.select(
            self.is_eval,
            jnp.array(0, dtype=jnp.int32),
            jax.random.randint(key, shape=(), minval=0, maxval=max_start_idx)
        )

        env_state = EnvState(
            sim_state=sim_state,
            time_idx=start_idx,
            step_count=jnp.array(0, dtype=jnp.int32),
            prev_action=jnp.zeros((self.action_dim,), dtype=jnp.float32),
        )
        obs = self._get_obs_with_forecast(sim_state, start_idx)
        return obs, env_state

    def step(self, key: jax.random.PRNGKey, env_state: EnvState, action: jnp.ndarray) -> Tuple[jnp.ndarray, EnvState, jnp.ndarray, jnp.ndarray, Dict]:
        del key
        t = jnp.minimum(env_state.time_idx, self.data_len - 1)
        exo_current = jax.tree.map(lambda x: x[t], self.all_exo)
        curr_sim_state = env_state.sim_state

        clipped_action = jnp.reshape(jnp.clip(action, -1.0, 1.0), (self.action_dim,))
        bat_w = jnp.squeeze(clipped_action) * self.max_battery_power_w
        capacity_ws = curr_sim_state.battery.config.capacity_kwh * 3600000.0
        dt = curr_sim_state.dt_seconds
        current_soc = curr_sim_state.state.battery.soc

        max_charge_w = (1.0 - current_soc) * capacity_ws / dt
        max_discharge_w = -(current_soc * capacity_ws) / dt
        bat_w = jnp.clip(bat_w, max_discharge_w, max_charge_w)

        sys_actions = SystemActions(
            battery_power_w=bat_w,
            heat_pump_power_w=jnp.zeros(self.n_rooms, dtype=jnp.float32),
            ac_power_w=jnp.zeros(self.n_rooms, dtype=jnp.float32),
            storage_discharge_w=jnp.zeros(self.n_rooms, dtype=jnp.float32)
        )

        next_sim_state, outputs = curr_sim_state.step(sys_actions, exo_current)
        cost = obj.f_stage_cost(
            current_state=curr_sim_state.state, next_state=next_sim_state.state,
            actions=sys_actions, outputs=outputs, exogenous=exo_current,
            configs=curr_sim_state.configs, dt_seconds=curr_sim_state.dt_seconds,
        )
        smoothness_penalty = self.action_smoothness_weight * jnp.sum(
            jnp.square(clipped_action - env_state.prev_action)
        )
        reward = -cost - smoothness_penalty
        next_t = env_state.time_idx + 1
        next_env_state = EnvState(
            sim_state=next_sim_state,
            time_idx=next_t,
            step_count=env_state.step_count + 1,
            prev_action=clipped_action,
        )

        done = next_env_state.step_count >= self.max_steps
        next_obs_t = jnp.minimum(next_t, self.data_len - 1)
        next_obs = self._get_obs_with_forecast(next_sim_state, next_obs_t)

        info = {"cost": cost, "bat_power_w": bat_w, "soc": next_sim_state.battery.soc}
        return next_obs, next_env_state, reward, done, info

# ==========================================
# 2. SAC networks (orthogonal init, twin critics with optional LayerNorm)
# ==========================================

HIDDEN_INIT = nn.initializers.orthogonal(jnp.sqrt(2.0))
OUTPUT_INIT = nn.initializers.orthogonal(0.01)

class Actor(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, obs):
        x = nn.relu(nn.Dense(256, kernel_init=HIDDEN_INIT)(obs))
        x = nn.relu(nn.Dense(256, kernel_init=HIDDEN_INIT)(x))
        mean = nn.Dense(self.action_dim, kernel_init=OUTPUT_INIT, bias_init=nn.initializers.zeros)(x)
        log_std = nn.Dense(self.action_dim, kernel_init=HIDDEN_INIT)(x)
        return mean, jnp.clip(log_std, -5.0, 2.0)

class TwinQ(nn.Module):
    use_layer_norm: bool = True

    @nn.compact
    def __call__(self, obs, action):
        x = jnp.concatenate([obs, action], axis=-1)
        def head(y):
            y = nn.Dense(256, kernel_init=HIDDEN_INIT, use_bias=not self.use_layer_norm)(y)
            if self.use_layer_norm:
                y = nn.LayerNorm()(y)
            y = nn.relu(y)
            y = nn.Dense(256, kernel_init=HIDDEN_INIT, use_bias=not self.use_layer_norm)(y)
            if self.use_layer_norm:
                y = nn.LayerNorm()(y)
            y = nn.relu(y)
            return nn.Dense(1, kernel_init=HIDDEN_INIT)(y).squeeze(-1)
        return head(x), head(x)

def sample_action(actor, params, obs, key):
    mean, log_std = actor.apply(params, obs)
    std = jnp.exp(log_std)
    noise = jax.random.normal(key, mean.shape)
    pre_tanh = mean + noise * std
    action = jnp.tanh(pre_tanh)
    gaussian_logp = -0.5 * (((pre_tanh - mean) / (std + 1e-8)) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi))
    logp = jnp.sum(gaussian_logp - jnp.log(1.0 - action**2 + 1e-6), axis=-1)
    return action, logp

def soft_update(target, source, tau):
    return jax.tree.map(lambda t, s: (1.0 - tau) * t + tau * s, target, source)

# ==========================================
# 3. SAC update step (twin-Q, auto-tuned entropy coefficient)
# ==========================================

def make_update(actor, qnet, actor_opt, q_opt, alpha_opt, target_entropy, gamma=0.995, tau=0.001, min_alpha=0.01):
    min_log_alpha = jnp.log(min_alpha)

    @jax.jit
    def update(actor_params, q_params, q_target_params, log_alpha, actor_state, q_state, alpha_state, batch, key):
        obs, actions, rewards, next_obs, dones = batch
        key, next_key, actor_key = jax.random.split(key, 3)
        alpha = jnp.exp(log_alpha)

        def q_loss_fn(params):
            next_actions, next_logp = sample_action(actor, actor_params, next_obs, next_key)
            tq1, tq2 = qnet.apply(q_target_params, next_obs, next_actions)
            target_q = rewards + gamma * (1.0 - dones) * (jnp.minimum(tq1, tq2) - alpha * next_logp)
            q1, q2 = qnet.apply(params, obs, actions)
            return jnp.mean((q1 - target_q) ** 2 + (q2 - target_q) ** 2)

        q_loss, q_grads = jax.value_and_grad(q_loss_fn)(q_params)
        q_updates, q_state = q_opt.update(q_grads, q_state, q_params)
        q_params = optax.apply_updates(q_params, q_updates)

        def actor_loss_fn(params):
            sampled_actions, logp = sample_action(actor, params, obs, actor_key)
            q1, q2 = qnet.apply(q_params, obs, sampled_actions)
            loss = jnp.mean(alpha * logp - jnp.minimum(q1, q2))
            return loss, logp

        (actor_loss, actor_logp), actor_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(actor_params)
        actor_updates, actor_state = actor_opt.update(actor_grads, actor_state, actor_params)
        actor_params = optax.apply_updates(actor_params, actor_updates)

        # Auto-tuned entropy coefficient: adjusts alpha so mean policy entropy tracks target_entropy
        def alpha_loss_fn(log_alpha_param):
            return -jnp.mean(log_alpha_param * (jax.lax.stop_gradient(actor_logp) + target_entropy))

        alpha_loss, alpha_grad = jax.value_and_grad(alpha_loss_fn)(log_alpha)
        alpha_updates, alpha_state = alpha_opt.update(alpha_grad, alpha_state, log_alpha)
        log_alpha = optax.apply_updates(log_alpha, alpha_updates)
        log_alpha = jnp.maximum(log_alpha, min_log_alpha)  # floor so entropy can't collapse to 0

        q_target_params = soft_update(q_target_params, q_params, tau)
        return (actor_params, q_params, q_target_params, log_alpha,
                actor_state, q_state, alpha_state, key, q_loss, actor_loss, alpha_loss)

    return update

class ReplayBuffer:
    def __init__(self, size, obs_dim, action_dim):
        self.size = size
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((size,), dtype=np.float32)
        self.next_obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.dones = np.zeros((size,), dtype=np.float32)
        self.pos = 0
        self.full = False

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.pos] = np.asarray(obs, dtype=np.float32)
        self.actions[self.pos] = np.asarray(action, dtype=np.float32)
        self.rewards[self.pos] = float(reward)
        self.next_obs[self.pos] = np.asarray(next_obs, dtype=np.float32)
        self.dones[self.pos] = float(done)
        self.pos = (self.pos + 1) % self.size
        self.full = self.full or self.pos == 0

    def __len__(self):
        return self.size if self.full else self.pos

    def sample(self, batch_size):
        idx = np.random.randint(0, len(self), size=batch_size)
        return (
            jnp.asarray(self.obs[idx]),
            jnp.asarray(self.actions[idx]),
            jnp.asarray(self.rewards[idx]),
            jnp.asarray(self.next_obs[idx]),
            jnp.asarray(self.dones[idx]),
        )

def get_chunked_rollout(actor, env, rollout_steps=1000, start_steps=20_000):
    def step_fn(carry, _):
        current_obs, current_env_state, current_key, actor_params, global_step = carry
        current_key, step_key, action_key, reset_key, uniform_key = jax.random.split(current_key, 5)

        is_random = global_step < start_steps
        random_action = jax.random.uniform(uniform_key, (env.action_dim,), minval=-1.0, maxval=1.0)

        mean, log_std = actor.apply(actor_params, current_obs[None, :])
        std = jnp.exp(log_std)
        noise = jax.random.normal(action_key, mean.shape)
        actor_action = jnp.tanh(mean + noise * std)[0]

        action = jax.lax.select(is_random, random_action, actor_action)

        next_obs, next_env_state, reward, done, _ = env.step(step_key, current_env_state, action)

        reset_obs, reset_env_state = env.reset(reset_key)
        effective_next_obs = jax.lax.select(done, reset_obs, next_obs)
        effective_next_env_state = jax.tree.map(lambda x, y: jax.lax.select(done, x, y), reset_env_state, next_env_state)

        transition = (current_obs, action, reward, effective_next_obs, done)
        return (effective_next_obs, effective_next_env_state, current_key, actor_params, global_step + 1), transition

    return jax.jit(lambda o, s, k, p, g: jax.lax.scan(step_fn, (o, s, k, p, g), None, length=rollout_steps))

# ==========================================
# 4. Simulator / dataset setup
# ==========================================

def load_dataset() -> SimulationDataset:
    if not CSV_PATH.exists():
        # sample_data.csv is git-ignored; generate it from the underlying real
        # load/PV/price data (same source 03_train_ppo.py uses) if not present.
        sample_data_generator.FILE_NAME = str(CSV_PATH)
        sample_data_generator.create_sample_data(n_days=14)
    return SimulationDataset(str(CSV_PATH), DT_SECONDS)

def build_simulator(initial_battery_soc: float = 0.0) -> JAXSimulator:
    return JAXSimulator(
        dt_seconds=DT_SECONDS,
        t_config=create_2_room_house(),
        r_config=RewardConfig(price_weight=PRICE_WEIGHT, comfort_weight=50.0),
        b_config=BatteryConfig(),
        hp_config=HeatPumpConfig(),
        ac_config=AirConditionerConfig(),
        ts_config=ThermalStorageConfig(),
        pv_config=PVConfig(model_type="passthrough"),
        initial_battery_soc=initial_battery_soc,
    )

# ==========================================
# 5. Training loop
# ==========================================
# Defaults below are the finalized configuration chosen after comparing 7 variants
# (baseline, smoothness penalties at 0.02/0.2, LayerNorm alone, LayerNorm+penalty,
# a fixed-alpha/no-autotune variant, and a higher entropy floor of 0.05): LayerNorm on
# the critic + a moderate action-smoothness penalty gave the best trade-off between
# curbing full-power charge/discharge chattering and keeping the arbitrage return.

def train_sac(total_steps=300_000, batch_size=256, chunk_size=1000, start_steps=20_000,
              updates_per_env_step=4, action_smoothness_weight=0.05, use_q_layer_norm=True,
              min_alpha=0.01):
    dataset = load_dataset()
    sim = build_simulator(initial_battery_soc=0.0)
    env = BatteryPriceForecastEnv(simulator=sim, dataset=dataset, max_steps=DEFAULT_STEPS,
                                   action_smoothness_weight=action_smoothness_weight)

    rng = jax.random.PRNGKey(42)
    rng, reset_key, actor_key, q_key = jax.random.split(rng, 4)
    obs, env_state = env.reset(reset_key)
    obs_dim = obs.shape[0]
    action_dim = env.action_dim

    actor = Actor(action_dim)
    qnet = TwinQ(use_layer_norm=use_q_layer_norm)
    actor_params = actor.init(actor_key, jnp.zeros((1, obs_dim), dtype=jnp.float32))
    q_params = qnet.init(q_key, jnp.zeros((1, obs_dim), dtype=jnp.float32), jnp.zeros((1, action_dim), dtype=jnp.float32))
    q_target_params = jax.tree.map(lambda x: x.copy(), q_params)

    # Auto-tuned entropy coefficient: start at alpha=1.0 (log_alpha=0), let it anneal toward target_entropy=-action_dim
    log_alpha = jnp.array(0.0, dtype=jnp.float32)
    target_entropy = -float(action_dim)

    # Gradient clipping (global-norm 10.0) chained before Adam
    actor_opt = optax.chain(optax.clip_by_global_norm(10.0), optax.adam(3e-4))
    q_opt = optax.chain(optax.clip_by_global_norm(10.0), optax.adam(3e-4))
    alpha_opt = optax.adam(3e-4)
    actor_state = actor_opt.init(actor_params)
    q_state = q_opt.init(q_params)
    alpha_state = alpha_opt.init(log_alpha)

    update = make_update(actor, qnet, actor_opt, q_opt, alpha_opt, target_entropy, gamma=0.995, tau=0.001, min_alpha=min_alpha)
    buffer = ReplayBuffer(100_000, obs_dim, action_dim)
    compiled_rollout = get_chunked_rollout(actor, env, chunk_size, start_steps)

    print("Starting SAC training...")
    global_step = 0

    for chunk in range(total_steps // chunk_size):
        (obs, env_state, rng, actor_params, global_step), transitions = compiled_rollout(obs, env_state, rng, actor_params, global_step)
        obs_b, action_b, reward_b, next_obs_b, done_b = transitions

        # Push chunk transitions into replay buffer
        for i in range(chunk_size):
            buffer.add(
                np.asarray(obs_b[i]),
                np.asarray(action_b[i]),
                float(reward_b[i]),
                np.asarray(next_obs_b[i]),
                float(done_b[i])
            )

        # Perform online updates corresponding to this chunk (update-to-data ratio = updates_per_env_step)
        if len(buffer) >= batch_size:
            for _ in range(chunk_size * updates_per_env_step):
                rng, update_key = jax.random.split(rng)
                batch = buffer.sample(batch_size)
                (actor_params, q_params, q_target_params, log_alpha,
                 actor_state, q_state, alpha_state, rng, q_loss, actor_loss, alpha_loss) = update(
                    actor_params, q_params, q_target_params, log_alpha,
                    actor_state, q_state, alpha_state, batch, update_key
                )

        current_step = (chunk + 1) * chunk_size
        if current_step % 10000 == 0:
            print(f"Completed step={current_step}/{total_steps} | q_loss={float(q_loss):.6f} "
                  f"actor_loss={float(actor_loss):.6f} alpha={float(jnp.exp(log_alpha)):.4f}")

    eval_env = BatteryPriceForecastEnv(simulator=sim, dataset=dataset, max_steps=DEFAULT_STEPS, is_eval=True)
    return actor, actor_params, eval_env

# ==========================================
# 6. Evaluation
# ==========================================

def evaluate(actor, actor_params, env, output_path=None):
    obs, env_state = env.reset(jax.random.PRNGKey(999))
    rewards = []
    soc_track = []
    prices_track = []
    dataset = load_dataset()

    for t in range(env.max_steps):
        mean, _ = actor.apply(actor_params, jnp.asarray(obs)[None, :])
        action = jnp.tanh(mean[0])
        obs, env_state, reward, done, info = env.step(jax.random.PRNGKey(0), env_state, action)

        rewards.append(float(reward))
        soc_track.append(float(info["soc"]))
        prices_track.append(float(dataset[t].price))

        if bool(done):
            break

    output_path = output_path or (EXAMPLES_DIR / "sac_eval_results.npz")
    np.savez(output_path, rewards=rewards, soc=soc_track, prices=prices_track)
    print(f"SAC evaluation return: {np.sum(rewards):.6f} raw ({np.sum(rewards) / PRICE_WEIGHT:.3f} EUR)")

if __name__ == "__main__":
    actor, params, env = train_sac()
    evaluate(actor, params, env)
