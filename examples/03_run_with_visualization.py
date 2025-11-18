# examples/03_run_with_visualization.py
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np

from energysim.sim.simulator import JAXSimulator
from energysim.core.data.dataset import SimulationDataset
from energysim.core.shared.data_structs import (
    BatteryConfig, RewardConfig, HeatPumpConfig, AirConditionerConfig,
    ThermalStorageConfig, SolarConfig, SystemActions
)
from energysim.control.baselines import PIDThermostat, TimeOfUseBattery, CompositeBaseline
from energysim.analysis.renderer import Renderer
import sample_data_generator
from build_my_house import create_2_room_house

def stack_pytree(list_of_trees):
    """Helper: Stacks a list of Pytrees (e.g. States) into a single Pytree with an added time dimension."""
    return jax.tree.map(lambda *args: jnp.stack(args), *list_of_trees)

def run_viz():
    print("--- 1. Setup Simulation ---")
    # Create data
    sample_data_generator.create_sample_data(n_days=3)
    dt = sample_data_generator.DT_SECONDS
    dataset = SimulationDataset(sample_data_generator.FILE_NAME, dt)
    
    # Create House Config
    t_config = create_2_room_house()
    n_rooms = len(t_config.room_air_indices)

    # Create Simulator
    sim = JAXSimulator(
        dt_seconds=dt,
        t_config=t_config,
        r_config=RewardConfig(),
        b_config=BatteryConfig(capacity_kwh=13.5, max_power_kw=5.0),
        hp_config=HeatPumpConfig(model_type="ramping", max_electrical_power_w=4000.0),
        ac_config=AirConditionerConfig(model_type="ramping", max_electrical_power_w=4000.0),
        ts_config=ThermalStorageConfig(volume_m3=0.5, n_nodes=5), # Stratified tank
        s_config=SolarConfig(model_type="simple", panel_area_m2=25.0)
    )

    # Setup Controller: Mix of PID (for HVAC) and Time-of-Use (for Battery)
    pid = PIDThermostat(t_config, sim.heat_pump.config, sim.ac.config, kp=4000.0, ki=10.0)
    tou = TimeOfUseBattery(sim.battery.config, price_low_threshold=0.15, price_high_threshold=0.30)
    controller = CompositeBaseline(pid, tou)

    # Initialize Renderer
    renderer = Renderer(sim)

    print("--- 2. Running Simulation Loop ---")
    state = sim.reset().state
    
    # Prepare broadcasted zero-actions for initial step
    prev_action = SystemActions(
        battery_power_w=jnp.array(0.0),
        heat_pump_power_w=jnp.zeros(n_rooms),
        ac_power_w=jnp.zeros(n_rooms),
        storage_discharge_w=jnp.zeros(n_rooms)
    )

    # History collectors
    history_states = []
    history_actions = []
    history_exo = []
    history_costs = []

    # Define Split Factors for scalar->vector mapping
    # 60% of solar/internal gains go to Room 0, 40% to Room 1
    split = jnp.array([0.6, 0.4])

    for i in range(len(dataset)):
        # 1. Get Exogenous Data & Broadcast scalars to room vectors
        raw_exo = dataset[i]
        exo = eqx.tree_at(
            lambda e: (e.solar_gains_w, e.occupancy_gains_w, e.device_gains_w),
            raw_exo,
            (raw_exo.solar_gains_w * split, raw_exo.occupancy_gains_w * split, jnp.zeros(n_rooms))
        )

        # 2. Get Action from Controller
        action, controller = controller(state, exo, dt)

        # 3. Step Simulator
        new_sim, cost = sim.step(action, prev_action, exo)
        
        # 4. Record
        history_states.append(state)
        history_actions.append(action)
        history_exo.append(exo)
        history_costs.append(cost)

        # 5. Optional: Real-time render in terminal (every 24 steps / 6 hours)
        if i % 24 == 0:
            renderer.render_step(i, state, action, float(cost), exo)

        # Update loop variables
        state = new_sim.state
        prev_action = action

    print("--- 3. Generating Plots ---")
    # Stack lists into time-series arrays
    ts_states = stack_pytree(history_states)
    ts_actions = stack_pytree(history_actions)
    ts_exo = stack_pytree(history_exo)
    ts_costs = jnp.array(history_costs)

    # Generate the Static Analysis Plot
    renderer.plot_trajectory(
        states=ts_states,
        actions=ts_actions,
        exogenous=ts_exo,
        costs=ts_costs,
        save_path="simulation_dashboard.png",
        show=True
    )

if __name__ == "__main__":
    run_viz()