# Action-Adaptive Continual Learning (AACL) for Multi-Device HEMS
**Branch:** `feature/aacl-multi-device`
**Objective:** Expand the AACL framework from a battery-only simulation to a multi-device environment (Battery + Heat Pump) utilizing real-world datasets, directly addressing reviewer feedback for publication.

## Phase 1: Environment Definition (Markov Decision Process)

To demonstrate that the AACL architecture successfully decouples high-level economic policy from low-level physical actuation, the environment has been expanded to manage joint combinatorial actions under real-world weather and pricing constraints.

### 1. State Space ($\mathcal{S}$)
The observation space is expanded to include environmental factors necessary for multi-device operation, specifically external weather conditions affecting the heat pump's Coefficient of Performance (COP). 

Let $\mathcal{S}_t$ be the state vector at time step $t$:

| Feature Category | Variable | Description |
| :--- | :--- | :--- |
| **Temporal** | $H_t, D_t$ | Hour of the day and Day of the week (sine/cosine encoded). |
| **Economics** | $P_t$ | Current dynamic electricity price (real-world wholesale/TOU). |
| **Weather** | $T_{out, t}$ | Outdoor temperature (Celsius). |
| | $G_t$ | Solar Irradiance ($W/m^2$). |
| **Physical Systems**| $SoC_t$ | Battery State of Charge ($0.0 \dots 1.0$). |
| | $T_{in, t}$ | Indoor temperature of the house (Celsius). |
| **Load Profiles** | $L_{base, t}$ | Uncontrollable base household load ($kW$). |

### 2. Action Space ($\mathcal{A}$)
To align with realistic hardware actuation and maintain the discrete action space praised in the initial manuscript, the control mechanisms are discretized into specific wattage tiers. 

* **Battery Action ($a_{batt}$):** 5 Discrete States
  * $\{-7.0\text{kW}, -3.5\text{kW}, 0\text{kW}, 3.5\text{kW}, 7.0\text{kW}\}$ 
  * *(Negative = Discharge, Positive = Charge)*
* **Heat Pump Action ($a_{hp}$):** 4 Discrete States
  * $\{0\text{kW}, 1.0\text{kW}, 2.0\text{kW}, 3.0\text{kW}\}$
  * *(Off, Low, Medium, High compressor power)*

**Joint Action Space:**
The AACL `Decoder` network will map the latent "economic policy" to a single flattened integer representing the combinatorial product of both device states.
* Total Action Dimension = $Size(a_{batt}) \times Size(a_{hp})$ 
* $5 \times 4 = \textbf{20 possible joint actions.}$


## Phase 2: Reward Function Formulation ($R_t$)

The core challenge of the multi-device HEMS is balancing economic efficiency with user comfort. Because the PPO agent seeks to *maximize* cumulative reward, costs and discomfort are formulated as negative penalties.

### 1. Net Grid Energy ($E_{grid, t}$)
First, we calculate the total energy drawn from (or fed into) the grid at time $t$. 
$$E_{grid, t} = L_{base, t} + E_{hp}(a_{hp, t}) + E_{batt}(a_{batt, t}) - E_{pv}(G_t)$$
* $L_{base, t}$: Base uncontrollable load.
* $E_{hp}$: Energy consumed by the heat pump based on its current action.
* $E_{batt}$: Energy charged (positive) or discharged (negative) by the battery.
* $E_{pv}$: Solar generation based on irradiance $G_t$.

### 2. Economic Cost Component ($R_{cost, t}$)
The economic penalty is the net energy multiplied by the dynamic electricity price $P_t$.
$$R_{cost, t} = - (E_{grid, t} \times P_t)$$
*(Note: If $E_{grid, t}$ is negative, the agent is selling power back to the grid, resulting in a positive economic reward).*

### 3. Thermal Comfort Penalty ($R_{comfort, t}$)
To prevent the agent from freezing the house to save money, we apply a penalty based on the squared deviation from a target temperature ($T_{target}$).
$$R_{comfort, t} = - \lambda_{comfort} \times (T_{in, t} - T_{target})^2$$
* $\lambda_{comfort}$: A hyperparameter weight that scales the importance of comfort relative to cost.
* Squaring the error ensures that small deviations (e.g., $0.5^\circ\text{C}$) are lightly penalized, but large deviations (e.g., $3^\circ\text{C}$) are heavily punished.

### 4. Total Reward ($R_t$)
The final reward signal passed to the agent is the sum of the economic and comfort components:
$$R_t = R_{cost, t} + R_{comfort, t}$$

By adjusting $\lambda_{comfort}$, we can train agents with different preference profiles (e.g., an "Eco-mode" agent vs. a "Comfort-first" agent), further demonstrating the flexibility of the learned economic policy.

## Phase 3: Environment Implementation
* Created `hems_env.py` containing `HEMSMultiDeviceEnv`.
* Inherits from `gym.Env` and strictly implements `UnifiedEnvProtocol` to maintain compatibility with the legacy AACL PyTorch training loops. 
* Next milestone: Implement the step() physics for Battery and Heat Pump state transitions.

## Phase 4: Physics and Step Implementation
The `step()` function evaluates the 20-dimensional joint action space, calculating physical state transitions and the associated rewards.

**Key Mathematical Models Introduced:**
1. **Action Decoding:** The combinatorial integer action $a \in [0, 19]$ is factored using modular arithmetic ($a_{batt} = a \text{ // } 4$, $a_{hp} = a \text{ \% } 4$) to derive specific wattage commands.
2. **Boundary Enforcement:** The battery simulation explicitly caps charging and discharging logic to ensure $SoC \in [0.0, 1.0]$, preventing non-physical states.
3. **Dynamic Efficiency (COP):** The heat pump's Coefficient of Performance is modeled as a function of the external temperature ($T_{out}$). This ensures that heating the house during cold nights is realistically more energy-intensive than during warmer periods.
4. **Thermal Leakage Model:** Indoor temperature updates use a first-order thermal mass approximation: $T_{new} = T_{old} - \alpha(T_{old} - T_{out}) + \beta(Q_{heat})$, where $\alpha$ represents building insulation quality.