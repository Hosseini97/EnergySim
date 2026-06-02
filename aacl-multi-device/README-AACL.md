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