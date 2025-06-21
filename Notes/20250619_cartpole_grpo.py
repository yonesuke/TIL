import gymnasium as gym
import torch
import numpy as np
from typing import NamedTuple
import wandb

PROJECT_NAME = "GRPO_CartPole"
ENV_NAME = "CartPole-v1"
MAX_STEPS = 1000

class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int, n_units: int = 32, n_layers: int = 4, activation="gelu"):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(torch.nn.Linear(n_inputs if i == 0 else n_units, n_units))
            if activation == "gelu":
                self.layers.append(torch.nn.GELU(approximate='tanh'))
            elif activation == "relu":
                self.layers.append(torch.nn.ReLU())
        self.layers.append(torch.nn.Linear(n_units, n_outputs))
        
    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x

class Trajectory(NamedTuple):
    observations: np.ndarray
    actions: np.ndarray
    log_probs: np.ndarray
    rewards: np.ndarray

def collect_trajectory(env: gym.Env, policy: torch.nn.Module, max_steps: int = MAX_STEPS) -> Trajectory:
    observation, _ = env.reset()
    observations, log_probs, actions, rewards = [], [], [], []
    done = False
    
    while not done:
        observations.append(observation)
        # get action from policy
        observation_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0) # shape (1, 4)
        logit = policy(observation_tensor) # shape (1, 2)
        prob = torch.softmax(logit, dim=-1) # shape (1, 2)
        action = torch.multinomial(prob, num_samples=1).item() # int
        log_prob = torch.log(prob[0, action]) # type: ignore # shape ()
        actions.append(action)
        log_probs.append(log_prob.item())
        # take action
        observation, reward, done, _, _ = env.step(action)
        rewards.append(reward)
        # exit if max steps reached
        if len(observations) >= max_steps:
            break
    
    trajectory = Trajectory(
        observations=np.array(observations, dtype=np.float32),
        actions=np.array(actions, dtype=np.int64),
        log_probs=np.array(log_probs, dtype=np.float32),
        rewards=np.array(rewards, dtype=np.float32)
    )
    return trajectory

def calc_advatages_GRPO(trajectories: list[Trajectory]) -> np.ndarray:
    rewards = np.array([t.rewards.sum() for t in trajectories])
    mean_reward = rewards.mean()
    std_reward = rewards.std() + 1e-8  # avoid division by zero
    advantages = (rewards - mean_reward) / std_reward
    return advantages

def update_policy_GRPO(
    trajectories: list[Trajectory],
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    num_iterations: int = 20,
    epsilon: float = 0.2,
) -> None:
    advantages = calc_advatages_GRPO(trajectories)
    for _ in range(num_iterations):
        loss = torch.tensor(0.0, dtype=torch.float32)
        for trajectory, advantage in zip(trajectories, advantages):
            loss_trajectory = torch.tensor(0.0, dtype=torch.float32)
            for t in range(len(trajectory.actions)):
                observation_tensor = torch.tensor(trajectory.observations[t], dtype=torch.float32).unsqueeze(0) # shape (1, 4)
                new_logit = policy(observation_tensor) # shape (1, 2)
                new_prob = torch.softmax(new_logit, dim=-1) # shape (1, 2)
                new_log_prob = torch.log(new_prob[0, trajectory.actions[t]]) # shape ()
                old_log_prob = trajectory.log_probs[t] # shape ()
                ratio = torch.exp(new_log_prob - old_log_prob)
                clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
                loss_trajectory += -clipped_ratio * advantage
            loss += loss_trajectory / len(trajectory.actions)
        loss /= len(trajectories)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def main():
    env = gym.make(ENV_NAME)
    policy = MultiLayerPerceptron(4, 2, 64, 1, "gelu")
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    group_size = 5
    num_episodes_per_epoch = 100

    wandb.init(project=PROJECT_NAME)
    num_trials = 0
    goal_avg_reward = 400.0
    num_episodes = 0
    max_trials = 100
    while True:
        num_trials += 1
        
        for _ in range(num_episodes_per_epoch // group_size):
            trajectorires, episode_rewards = [], []
            with torch.no_grad():
                for _ in range(group_size):
                    trajectory = collect_trajectory(env, policy)
                    trajectorires.append(trajectory)
                    episode_rewards.append(trajectory.rewards.sum())
            num_episodes += num_episodes_per_epoch // group_size
            wandb.log({"average_rewards": np.mean(episode_rewards), "num_episodes": num_episodes})
            print(f">> Trial {num_trials}, Average Reward: {np.mean(episode_rewards):.2f}")
            update_policy_GRPO(trajectorires, policy, optimizer)
        
        avg_reward = np.mean(episode_rewards) # type: ignore
        print(f"[Finished Trial {num_trials}] Average Reward: {avg_reward:.2f}")
        if avg_reward >= goal_avg_reward:
            print(f"Goal reached in {num_trials} trials!")
            break
        if num_trials >= max_trials:
            print("Max trials reached without achieving goal.")
            break
    env.close()
    wandb.finish()

if __name__ == "__main__":
    main()
