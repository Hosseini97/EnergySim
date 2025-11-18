import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import jax
import jax.numpy as jnp
from typing import Optional, Dict, Any
import datetime

# Assumed available as per instructions
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich.align import Align
from rich.bar import Bar

from ..sim.simulator import JAXSimulator
from ..core.shared.data_structs import SystemState, SystemActions, ExogenousData

class Renderer:
    """
    Top-Tier Analysis & Visualization Engine.

    Modes:
    1. static_plot(): Generates engineering-grade Matplotlib figures for trajectory analysis.
    2. render_step(): A rich-text terminal dashboard for real-time monitoring.
    """

    def __init__(self, simulator: JAXSimulator):
        self.sim = simulator
        self.t_config = simulator.thermal.config
        self.b_config = simulator.battery.config
        self.hp_config = simulator.heat_pump.config
        self.s_config = simulator.solar.config
        
        self.dt = simulator.dt_seconds
        self.console = Console()

        # --- Plotting Style ---
        self.colors = {
            'temp_room': '#d32f2f',     # Red 700
            'temp_amb': '#1976d2',      # Blue 700
            'temp_st_hot': '#e64a19',   # Deep Orange
            'temp_st_cold': '#ffcc80',  # Orange 200
            'power_solar': '#fbc02d',   # Yellow 700
            'power_base': '#757575',    # Grey 600
            'power_hvac': '#5e35b1',    # Deep Purple
            'power_bat': '#43a047',     # Green 600 (Charge)
            'soc': '#8e24aa',           # Purple 600
            'cost': '#c62828'           # Red 800
        }

    def _to_cpu(self, jax_array: Any) -> np.ndarray:
        """Helper to safely move JAX arrays to Numpy/CPU for plotting."""
        if hasattr(jax_array, 'shape'):
            return np.array(jax_array)
        return jax_array

    def plot_trajectory(
        self, 
        states: SystemState, 
        actions: SystemActions, 
        exogenous: ExogenousData, 
        costs: Optional[jax.Array] = None,
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        Generates a 4-Panel Engineering Analysis Plot.
        
        Panel 1: Thermal Dynamics (Room vs Ambient vs Constraints)
        Panel 2: Power Balance (Stacked Sources vs Sinks)
        Panel 3: Storage State (Battery SoC & Thermal Stratification)
        Panel 4: Cost Accumulation vs Price Signal
        """
        # 1. Prepare Data
        time_steps = len(exogenous.ambient_temp)
        hours = np.arange(time_steps) * (self.dt / 3600.0)
        
        # Unpack and flatten spatial dimensions where necessary
        T_room = self._to_cpu(states.thermal.T_vector)[:, self.t_config.room_air_indices]
        T_room_mean = np.mean(T_room, axis=1)
        T_room_min = np.min(T_room, axis=1)
        T_room_max = np.max(T_room, axis=1)
        
        T_amb = self._to_cpu(exogenous.ambient_temp)
        
        # Reconstruct Solar (Proxy calculation for visualization)
        # P_pv = Irradiance * Area * Eff
        irr = self._to_cpu(exogenous.solar_irradiance_w_m2)
        solar_proxy_kw = (irr * self.s_config.panel_area_m2 * self.s_config.efficiency) / 1000.0
        
        # Loads (kW)
        base_load_kw = self._to_cpu(exogenous.base_load_w) / 1000.0
        hp_kw = np.sum(self._to_cpu(actions.heat_pump_power_w), axis=1) / 1000.0
        ac_kw = np.sum(self._to_cpu(actions.ac_power_w), axis=1) / 1000.0
        bat_kw = self._to_cpu(actions.battery_power_w) / 1000.0 # +Charge, -Discharge
        
        total_load_kw = base_load_kw + hp_kw + ac_kw
        
        # Net Grid Calculation
        # Balance: Grid + Solar + Bat_Discharge = Load + Bat_Charge
        # Grid = Load + Bat_Charge - Solar - Bat_Discharge
        # Grid = Load + Bat_Power - Solar
        net_grid_kw = total_load_kw + bat_kw - solar_proxy_kw

        # Storage Temps
        T_storage = self._to_cpu(states.storage.temperatures_c) # (Time, Nodes)
        st_mean = np.mean(T_storage, axis=1)
        st_top = T_storage[:, 0]   # Hot
        st_btm = T_storage[:, -1]  # Cold

        # --- PLOTTING ---
        fig, axes = plt.subplots(4, 1, figsize=(12, 18), sharex=True)
        plt.subplots_adjust(hspace=0.1)

        # Panel 1: Thermal
        ax = axes[0]
        ax.set_ylabel("Temp (°C)", fontweight='bold')
        ax.set_title("Thermal Comfort & Envelope Response", loc='left', fontweight='bold')
        
        # Comfort Band
        setp = self.t_config.setpoint
        band = self.t_config.comfort_band
        ax.axhline(setp, color='k', linestyle='--', alpha=0.3, label='Setpoint')
        ax.fill_between(hours, setp - band, setp + band, color='green', alpha=0.1, label='Comfort Band')
        
        ax.plot(hours, T_amb, color=self.colors['temp_amb'], linestyle=':', label='Ambient')
        ax.plot(hours, T_room_mean, color=self.colors['temp_room'], linewidth=2, label='Room Mean')
        ax.fill_between(hours, T_room_min, T_room_max, color=self.colors['temp_room'], alpha=0.2)
        ax.legend(loc='upper right', fontsize='small', ncol=3)
        ax.grid(True, alpha=0.2)

        # Panel 2: Power Balance (Stacked)
        ax = axes[1]
        ax.set_ylabel("Power (kW)", fontweight='bold')
        ax.set_title("Electrical Power Balance", loc='left', fontweight='bold')
        
        # Stack Sinks (Consumption)
        ax.stackplot(hours, base_load_kw, hp_kw + ac_kw, 
                     labels=['Base Load', 'HVAC'], 
                     colors=[self.colors['power_base'], self.colors['power_hvac']], alpha=0.7)
        
        # Plot Sources (Solar) inverted for contrast, or line
        ax.plot(hours, solar_proxy_kw, color=self.colors['power_solar'], linewidth=2, label='Solar Gen')
        
        # Plot Battery Activity
        # Fill charge/discharge
        ax.fill_between(hours, 0, bat_kw, where=(bat_kw > 0), color=self.colors['power_bat'], alpha=0.5, label='Bat Charge')
        ax.fill_between(hours, 0, bat_kw, where=(bat_kw < 0), color='red', alpha=0.5, label='Bat Discharge')
        
        # Net Grid Line
        ax.plot(hours, net_grid_kw, color='black', linewidth=1.5, linestyle='--', label='Net Grid')
        
        ax.legend(loc='upper right', fontsize='small', ncol=3)
        ax.grid(True, alpha=0.2)

        # Panel 3: Storage State
        ax = axes[2]
        ax.set_ylabel("State", fontweight='bold')
        ax.set_title("Storage Dynamics (Battery & Thermal)", loc='left', fontweight='bold')
        
        # Battery SoC
        ax.plot(hours, self._to_cpu(states.battery.soc), color=self.colors['soc'], linewidth=2, label='Battery SoC (0-1)')
        
        # Thermal Storage Stratification (Twin Axis for Temp)
        ax2 = ax.twinx()
        ax2.set_ylabel("Tank Temp (°C)", color=self.colors['temp_st_hot'])
        ax2.plot(hours, st_top, color=self.colors['temp_st_hot'], linestyle='-', linewidth=1, label='Tank Top')
        ax2.plot(hours, st_btm, color=self.colors['temp_st_cold'], linestyle='-', linewidth=1, label='Tank Bottom')
        ax2.plot(hours, st_mean, color=self.colors['temp_st_hot'], linestyle='--', alpha=0.5, label='Tank Mean')
        ax2.tick_params(axis='y', labelcolor=self.colors['temp_st_hot'])
        
        # Combine legends
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc='upper right', fontsize='small', ncol=2)
        ax.grid(True, alpha=0.2)

        # Panel 4: Economics
        ax = axes[3]
        ax.set_ylabel("Accumulated Cost (€)", fontweight='bold')
        ax.set_xlabel("Simulation Time (Hours)", fontweight='bold')
        ax.set_title("Economic Performance", loc='left', fontweight='bold')
        
        if costs is not None:
            cum_cost = np.cumsum(self._to_cpu(costs))
            ax.plot(hours, cum_cost, color=self.colors['cost'], linewidth=2, label='Cumulative Cost')
            ax.fill_between(hours, 0, cum_cost, color=self.colors['cost'], alpha=0.1)
            
            # Price Signal Twin Axis
            ax3 = ax.twinx()
            price = self._to_cpu(exogenous.price)
            ax3.plot(hours, price, color='gray', linestyle=':', label='Grid Price')
            ax3.set_ylabel("Price (€/kWh)", color='gray')
            ax3.tick_params(axis='y', labelcolor='gray')
        
        ax.grid(True, alpha=0.2)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[Renderer] Trajectory plot saved to {save_path}")
        
        if show:
            plt.show()
        plt.close()

    def render_step(
        self, 
        step_idx: int, 
        state: SystemState, 
        action: SystemActions, 
        cost: float, 
        exo: ExogenousData
    ):
        """
        Displays a Rich dashboard frame in the terminal.
        Best used within a `Live` context manager in the training loop, 
        but works as a standalone print for debugging.
        """
        # 1. Extract Data (Scalars)
        # Thermal
        t_vec = self._to_cpu(state.thermal.T_vector)
        rooms = t_vec[self.t_config.room_air_indices]
        avg_t = float(np.mean(rooms))
        amb_t = float(self._to_cpu(exo.ambient_temp))
        
        # Constraints check
        sp = self.t_config.setpoint
        band = self.t_config.comfort_band
        is_comf = (sp - band) <= avg_t <= (sp + band)
        
        # Storage
        st_vec = self._to_cpu(state.storage.temperatures_c)
        st_soc = float(np.clip((np.mean(st_vec) - 30)/(60-30), 0, 1)) # Normalized approx
        bat_soc = float(self._to_cpu(state.battery.soc))
        
        # Power
        hp_w = float(np.sum(self._to_cpu(action.heat_pump_power_w)))
        ac_w = float(np.sum(self._to_cpu(action.ac_power_w)))
        bat_w = float(self._to_cpu(action.battery_power_w))
        base_load = float(self._to_cpu(exo.base_load_w))
        
        # Net Grid approx
        irr = float(self._to_cpu(exo.solar_irradiance_w_m2))
        solar_w = irr * self.s_config.panel_area_m2 * self.s_config.efficiency
        net_grid = base_load + hp_w + ac_w + bat_w - solar_w
        
        # 2. Build Layout
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body", ratio=1)
        )
        layout["body"].split_row(
            Layout(name="thermal"),
            Layout(name="electrical"),
            Layout(name="financial")
        )

        # Header
        layout["header"].update(
            Panel(
                Align.center(f"[bold white]EnergySim Step {step_idx:05d}[/bold white] | " 
                             f"Time: {(step_idx*self.dt)/3600:.1f}h"),
                style="on blue"
            )
        )

        # Thermal Panel
        t_color = "green" if is_comf else "red"
        t_icon = "✅" if is_comf else "⚠️"
        
        therm_table = Table.grid(padding=1)
        therm_table.add_column(style="bold")
        therm_table.add_column(justify="right")
        therm_table.add_row("Room Avg:", f"[{t_color}]{avg_t:.2f}°C {t_icon}[/]")
        therm_table.add_row("Setpoint:", f"{sp:.1f} ± {band:.1f}°C")
        therm_table.add_row("Ambient:", f"[blue]{amb_t:.2f}°C[/]")
        therm_table.add_row("Tank Temp:", f"{float(np.mean(st_vec)):.1f}°C")
        
        layout["thermal"].update(Panel(therm_table, title="[red]Thermal State[/]", border_style="red"))

        # Electrical Panel
        elec_table = Table.grid(padding=1)
        elec_table.add_column(style="bold")
        elec_table.add_column(justify="right")
        
        elec_table.add_row("Net Grid:", f"{net_grid/1000:.2f} kW")
        elec_table.add_row("Base Load:", f"{base_load/1000:.2f} kW")
        elec_table.add_row("HVAC:", f"{(hp_w + ac_w)/1000:.2f} kW")
        elec_table.add_row("Solar:", f"[yellow]{solar_w/1000:.2f} kW[/]")
        
        # Battery Bar
        bat_color = "green" if bat_soc > 0.2 else "red"
        bat_bar = Bar(size=10, begin=0, end=100, value=bat_soc*100, color=bat_color)
        elec_table.add_row("Bat SoC:", bat_bar)
        elec_table.add_row("Bat Flow:", f"{bat_w/1000:+.2f} kW")

        layout["electrical"].update(Panel(elec_table, title="[yellow]Electrical[/]", border_style="yellow"))

        # Financial Panel
        fin_table = Table.grid(padding=1)
        fin_table.add_column(style="bold")
        fin_table.add_column(justify="right")
        
        fin_table.add_row("Price:", f"{float(exo.price):.3f} €/kWh")
        fin_table.add_row("Step Cost:", f"{cost:.4f} €")
        
        # Status badges
        status = Text()
        status.append("\nSystem Status:\n", style="dim")
        if bat_soc < 0.05: status.append("BAT CRITICAL ", style="bold red on white")
        elif bat_soc > 0.95: status.append("BAT FULL ", style="bold green on white")
        else: status.append("BAT OK ", style="green")
        
        layout["financial"].update(Panel(Align.center(fin_table), title="[green]Economics[/]", border_style="green"))

        self.console.print(layout)