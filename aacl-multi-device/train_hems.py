# train_hems.py
import torch
from aim import Run

# Import your existing agent and configs
from aacl.agents.baseline_agents import PPOBaselineAgent
# (Adjust this import path if your configs are located elsewhere)
from aacl.configs.agent_configs import BaseAgentConfig, ConfigManager 

# Import the new environment we just built
from hems_env import HEMSMultiDeviceEnv, HEMSConfig

def main():
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting training on device: {device}")

    # 2. Initialize the Aim Tracker (Standard for this repository)
    aim_run = Run(experiment="Phase6_MultiDevice_Baseline")

    # 3. Initialize the New Environment
    hems_config = HEMSConfig(
        max_steps=96,  # 15-minute intervals for 24 hours
        target_temp_c=22.0,
        lambda_comfort=2.0
    )
    env = HEMSMultiDeviceEnv(hems_config, device)

    # 4. Load the PPO Configuration
    # Assuming you have a default way to load configs in your repo. 
    # You may need to replace this with your actual config loading method.
    try:
        agent_config = ConfigManager.get_default_base_config()
    except AttributeError:
        # Fallback if ConfigManager is different in your local repo
        print("Warning: Using blank BaseAgentConfig, adjust if necessary.")
        agent_config = BaseAgentConfig()

    # 5. Instantiate the Baseline Agent
    # State_dim = 7, Action_dim = 20 (Joint Action Space)
    agent = PPOBaselineAgent(
        env=env,
        state_dim=7,
        action_dim=20, 
        config=agent_config,
        aim_run=aim_run
    )

    # 6. Run the Training Loop
    print("📈 Starting RL Training Stage...")
    global_step = 0
    
    # We will train for 500 episodes/iterations as a quick smoke test
    num_episodes = 500 
    
    for episode in range(num_episodes):
        # train_rl_stage handles the rollout collection and PPO update internally
        rl_buffer, global_step = agent.train_rl_stage(global_step=global_step)
        
        if episode % 10 == 0:
            print(f"✅ Completed Episode {episode}/{num_episodes} | Global Step: {global_step}")

    print("🎉 Training Complete! Check your Aim UI for the loss and reward curves.")

if __name__ == "__main__":
    main()