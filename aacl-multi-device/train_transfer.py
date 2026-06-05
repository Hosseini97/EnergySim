# train_transfer.py
import torch
from aim import Run

# Using the Masking Agent as our first Transfer Baseline
from aacl.agents.mask_agents import MaskModulationAgent
from aacl.configs.agent_configs import MaskAgentConfig
from hems_env import HEMSMultiDeviceEnv, HEMSConfig

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting Transfer Learning Experiment on: {device}")

    # 1. Initialize Aim and Environment
    aim_run = Run(experiment="Phase7_Transfer_Masking")
    hems_config = HEMSConfig(max_steps=96)
    env = HEMSMultiDeviceEnv(hems_config, device)

    # 2. Define the Hardware Topologies (Tasks)
    # House A (Task 0): All 20 joint actions are available.
    task_0_actions = list(range(20))
    
    # House B (Task 1): Smaller inverter. Battery restricted to [-3.5kW, 0, 3.5kW].
    # We remove batt_idx 0 (-7.0kW) and batt_idx 4 (7.0kW).
    # Allowed batt_idx = {1, 2, 3}. Joint action = (batt_idx * 4) + hp_idx
    task_1_actions = [a for a in range(20) if (a // 4) in [1, 2, 3]]
    
    tasks = [
        {"id": 0, "name": "House A (Full Capacity)", "actions": task_0_actions},
        {"id": 1, "name": "House B (Restricted Inverter)", "actions": task_1_actions}
    ]

    # 3. Initialize Agent Configuration
    config = MaskAgentConfig()
    config.rl_loop.rl_lr = 3e-4

    # 4. Initialize the Agent
    agent = MaskModulationAgent(
        env=env,
        state_dim=7,
        initial_action_dim=20,
        config=config,
        aim_run=aim_run
    )

    # 5. Run the Continual Learning Loop
    global_step = 0
    for task in tasks:
        print(f"\n{'='*50}")
        print(f"🏠 Starting {task['name']}")
        print(f"Allowed Actions: {task['actions']}")
        print(f"{'='*50}")
        
        # Tell the environment which hardware is available
        env.set_action_space(task['actions'])
        
        # Tell the agent to adapt its output layer/mask to the new hardware
        agent.adapt_to_new_task(task['actions'])
        
        # Train on this specific house for 500 episodes
        rl_buffer, global_step = agent.train_rl_stage(global_step=global_step)
        
        print(f"✅ Finished training on {task['name']}.")

if __name__ == "__main__":
    main()