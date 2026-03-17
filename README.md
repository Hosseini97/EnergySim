# EnergySim

**EnergySim** is a JAX-native Home Energy Management System (HEMS) simulation framework designed for high-performance control optimization and reinforcement learning.

By leveraging JAX, the framework provides a fully differentiable physics engine, allowing for gradient-based Model Predictive Control (MPC) and massive parallelization on GPU/TPU hardware.

## Performance

EnergySim is built for extreme throughput, designed to bridge the gap between high-fidelity physics and the data requirements of modern RL.

* **Single Environment:** ~200,000 steps per second.
* **Vectorized Environment (16k parallel envs):** ~200,000,000 steps per second.

---

## What It Is

EnergySim simulates the energy dynamics of a residential building. It treats the entire house and its sub-components as a single functional computation graph.

### Physics-Based Modeling

The simulator includes high-fidelity, differentiable models for:

* **Thermal Dynamics:** Multi-room RC-network models with dynamic infiltration and inter-zone coupling.
* **Electrical Storage:** Battery models featuring efficiency losses and state-of-health (SOH) degradation.
* **HVAC Systems:** Heat pumps and air conditioners with variable Coefficient of Performance (COP) based on ambient conditions and thermal lag.
* **Thermal Storage:** Stratified water tank models (1D) and 3D finite-volume grid models.
* **Renewables:** Geometric PV models considering panel tilt, azimuth, and temperature coefficients.
* **Stochastic Behavior:** Models for EV charging, occupancy heat gains, and household appliances.

### Control and RL Integration

Because the framework is written in JAX/Equinox, it natively supports:

* **Gradient-Based MPC:** Formulate and solve optimal control problems using automatic differentiation through the physics engine.
* **Vectorized RL:** A memory-optimized `VectorizedEnergyEnv` that allows thousands of independent simulations to run in parallel on a single GPU.
* **Differentiable Forecasting:** Forecasters with "cones of uncertainty" for testing controller robustness under noise.

---

## Installation

```bash
git clone https://github.com/Hosseini97/EnergySim.git
cd EnergySim
pip install -e .

```

## Examples

You can find implementation examples in the `examples/` directory:

* `01_run_simple_simulation.py`: Basic setup and forward simulation.
* `02_run_mpc.py`: Solving optimal control using the JAX-native MPC solver.
* `03_train_ppo.py`: Training a PPO agent in a vectorized environment.