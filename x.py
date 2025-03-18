import gymnasium as gym
import mo_gymnasium
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from collections import deque

torch.manual_seed(0)
np.random.seed(0)

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, reward_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, reward_dim)
        )


    def act(self, state):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.item(), action_logprob

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        return action_logprobs, state_values, dist_entropy

# PPO algorithm
class PPO:
    def __init__(self, state_dim, action_dim, reward_dim, lr=0.0003, gamma=0.99, eps_clip=0.1, K_epochs=20):
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.policy = ActorCritic(state_dim, action_dim, reward_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(state_dim, action_dim, reward_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state)
            action, action_logprob = self.policy_old.act(state)
            return action, action_logprob

    def update(self, memory, scalarized_weights):
        memory_tensor = memory.to_tensor()
        old_states = memory_tensor['states'].detach()
        old_actions = memory_tensor['actions'].detach()
        old_logprobs = memory_tensor['logprobs'].detach()
        rewards = []
        discounted_reward = torch.zeros(memory.rewards[0].size(-1))
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = torch.zeros_like(discounted_reward)
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        rewards = torch.stack(rewards)
        
        # Convert deque objects to lists before stacking
        old_states = torch.stack(list(memory.states)).detach()
        old_actions = torch.stack(list(memory.actions)).detach()
        old_logprobs = torch.stack(list(memory.logprobs)).detach()

        weighted_rewards = torch.matmul(rewards, torch.FloatTensor(scalarized_weights))
        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = weighted_rewards - torch.matmul(state_values, torch.FloatTensor(scalarized_weights))
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            critic_loss = 0
            for i in range(state_values.shape[1]):
                critic_loss += self.MseLoss(state_values[:, i], rewards[:, i])
            loss = -torch.min(surr1, surr2) + 0.5 * critic_loss - 0.01 * dist_entropy
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            self.policy_old.load_state_dict(self.policy.state_dict())

class Memory:
    def __init__(self):
        self.actions = deque()
        self.states = deque()
        self.logprobs = deque()
        self.rewards = deque()
        self.is_terminals = deque()

    def clear_memory(self):
        self.actions.clear()
        self.states.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.is_terminals.clear()

    def to_tensor(self):
        return {
            'states': torch.stack(list(self.states)),
            'actions': torch.stack(list(self.actions)),
            'logprobs': torch.stack(list(self.logprobs)),
            'rewards': torch.stack(list(self.rewards)),
            'is_terminals': torch.tensor(list(self.is_terminals))
        }

def train(load_existing=False):
    env_id = "mo-lunar-lander-v3"
    env = mo_gymnasium.make(env_id)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    reward_dim = 4
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Reward dimension: {reward_dim}")
    scalarized_weights = [0.6, 0.2, 0.1, 0.1]
    ppo = PPO(state_dim, action_dim, reward_dim, lr=0.0003, gamma=0.98, eps_clip=0.2, K_epochs=10)
    
    if load_existing and os.path.exists('./mo_lunar_lander_ppo.pth'):
        print("Loading existing model...")
        ppo.policy.load_state_dict(torch.load('./mo_lunar_lander_ppo.pth'))
        ppo.policy_old.load_state_dict(torch.load('./mo_lunar_lander_ppo.pth'))
        return evaluate_and_record(20)  # Run evaluation directly
    
    memory = Memory()
    max_episodes = 5000 #maceps
    max_timesteps = 1500
    update_timestep = 1000
    log_interval = 20
    running_reward = 0
    avg_length = 0
    timestep = 0
    episode_rewards = []
    episode_lengths = []
    
    for i_episode in range(1, max_episodes+1):
        state, _ = env.reset()
        episode_reward = np.zeros(reward_dim)
        has_landed = False
        
        for t in range(max_timesteps):
            timestep += 1
            action, logprob = ppo.select_action(state)
            next_state, reward_vector, terminated, truncated, _ = env.step(action)
            
            # Check if the lander has landed
            if not has_landed and terminated:
                has_landed = True
                terminated = True  # Terminate the episode immediately after landing
            
            done = terminated or truncated
            
            memory.states.append(torch.FloatTensor(state))
            memory.actions.append(torch.tensor(action))
            memory.logprobs.append(logprob)
            memory.rewards.append(torch.FloatTensor(reward_vector))
            memory.is_terminals.append(done)
            
            state = next_state
            
            episode_reward += reward_vector
            
            if timestep % update_timestep == 0:
                ppo.update(memory, scalarized_weights)
                memory.clear_memory()
                timestep = 0
            
            if done:
                break
        
        weighted_reward = np.dot(episode_reward, scalarized_weights)
        
        running_reward = 0.05 * weighted_reward + 0.95 * running_reward
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(t+1)
        
        avg_length += 0.05 * (t+1 - avg_length)
        
        if i_episode % log_interval == 0:
            print(f'Episode {i_episode}\t Avg length: {avg_length:.2f}\t Running Reward: {running_reward:.2f}')
            print(f'Episode Reward Components: Landing={episode_reward[0]:.2f}, Shaping={episode_reward[1]:.2f}, Main Fuel={episode_reward[2]:.2f}, Side Fuel={episode_reward[3]:.2f}')
        
        if i_episode > 100:
            last_100_landing_rewards = [r[0] for r in episode_rewards[-100:]]
            success_rate = sum(1 for r in last_100_landing_rewards if r > -50) / 100
            
            if i_episode % log_interval == 0:
                print(f"Last 100 episodes landing success rate: {success_rate:.2f}")
            
            if success_rate > 0.7:
                print("Solved!")
                torch.save(ppo.policy.state_dict(), './mo_lunar_lander_ppo.pth')
                break
    
    torch.save(ppo.policy.state_dict(), './mo_lunar_lander_ppo.pth')
    
    env.close()
    
    return episode_rewards, episode_lengths

def evaluate(num_episodes=10, render=True):
    env = mo_gymnasium.make("mo-lunar-lander-v3", render_mode="human" if render else None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    reward_dim = 4 
    model = ActorCritic(state_dim, action_dim, reward_dim)
    model.load_state_dict(torch.load('./mo_lunar_lander_ppo.pth'))
    model.eval()
    all_rewards = []
    
    for i in range(num_episodes):
        state, _ = env.reset()
        episode_reward = np.zeros(reward_dim)
        done = False
        
        while not done:
            state_tensor = torch.FloatTensor(state)
            with torch.no_grad():
                action_probs = model.actor(state_tensor)
                dist = Categorical(action_probs)
                action = dist.sample().item()
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                state = next_state
                episode_reward += reward
            
            if done:
                break
        
        all_rewards.append(episode_reward)
        
        print(f"Episode {i+1} Reward: Landing={episode_reward[0]:.2f}, Shaping={episode_reward[1]:.2f}, Main Fuel={episode_reward[2]:.2f}, Side Fuel={episode_reward[3]:.2f}")
    
    env.close()
    
    return all_rewards

def evaluate_and_record(num_episodes=10, video_folder="videos"):
    import gymnasium.wrappers
    env = mo_gymnasium.make("mo-lunar-lander-v3", render_mode="rgb_array")
    if not os.path.exists(video_folder):
        os.makedirs(video_folder)
    env = gymnasium.wrappers.RecordVideo(
        env,
        video_folder,
        episode_trigger=lambda x: True 
    )
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    reward_dim = 4
    model = ActorCritic(state_dim, action_dim, reward_dim)
    model.load_state_dict(torch.load('./mo_lunar_lander_ppo.pth'))
    model.eval()
    all_rewards = []
    
    for i in range(num_episodes):
        state, _ = env.reset()
        episode_reward = np.zeros(reward_dim)
        done = False
        
        while not done:
            state_tensor = torch.FloatTensor(state)
            with torch.no_grad():
                action_probs = model.actor(state_tensor)
                dist = Categorical(action_probs)
                action = dist.sample().item()
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                state = next_state
                episode_reward += reward
            
            if done:
                break
        
        all_rewards.append(episode_reward)
        
        print(f"Episode {i+1} Reward: Landing={episode_reward[0]:.2f}, Shaping={episode_reward[1]:.2f}, Main Fuel={episode_reward[2]:.2f}, Side Fuel={episode_reward[3]:.2f}")
    
    env.close()
    
    print(f"Videos saved to {video_folder} directory")
    
    return all_rewards

def plot_results(episode_rewards, episode_lengths):
    reward_array = np.array(episode_rewards)
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(reward_array[:, 0])
    plt.title('Landing Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.subplot(2, 2, 2)
    plt.plot(reward_array[:, 1])
    plt.title('Shaping Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.subplot(2, 2, 3)
    plt.plot(reward_array[:, 2])
    plt.title('Main Engine Fuel Cost')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.subplot(2, 2, 4)
    plt.plot(reward_array[:, 3])
    plt.title('Side Engine Fuel Cost')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.tight_layout()
    plt.savefig('mo_lunar_lander_rewards.png')
    plt.figure(figsize=(10, 5))
    plt.plot(episode_lengths)
    plt.title('Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Length')
    plt.savefig('mo_lunar_lander_lengths.png')
    plt.close()

if __name__ == '__main__':
    import os
    if os.path.exists('./mo_lunar_lander_ppo.pth'):
        print("Existing model found. Skipping training and running evaluation...")
        evaluate_rewards = train(load_existing=True)
        avg_rewards = np.mean(evaluate_rewards, axis=0)
        print(f"\nAverage Evaluation Rewards: Landing={avg_rewards[0]:.2f}, Shaping={avg_rewards[1]:.2f}, Main Fuel={avg_rewards[2]:.2f}, Side Fuel={avg_rewards[3]:.2f}")
    else:
        print("No existing model found. Starting training...")
        episode_rewards, episode_lengths = train()
        plot_results(episode_rewards, episode_lengths)
        print("\nEvaluating trained agent...")
        evaluate_rewards = evaluate_and_record(10)
        avg_rewards = np.mean(evaluate_rewards, axis=0)
        print(f"\nAverage Evaluation Rewards: Landing={avg_rewards[0]:.2f}, Shaping={avg_rewards[1]:.2f}, Main Fuel={avg_rewards[2]:.2f}, Side Fuel={avg_rewards[3]:.2f}")