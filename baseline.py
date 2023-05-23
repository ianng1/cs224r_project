import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random

# Define the environment
board_size = 10

class PredatorPreyEnvironment:
    def __init__(self):
        self.state_size = 4  # Example: 4-dimensional state
        self.action_size = 8  # Example: 4 actions (up, down, left, right)
        self.max_steps = 500  # Maximum number of steps per episode
        self.current_step = 0
        self.prey_position = [0, 0]  # Example: Prey's initial position
        self.predator_positions = [[2, 2], [3, 3]]  # Example: Initial positions of two predators

    def reset(self):
        self.current_step = 0
        self.prey_position = [0, 0]  # Example: Prey's initial position
        self.predator_positions = [[2, 2], [3, 3]]  # Example: Initial positions of two predators
        state = self._get_state()
        return state

    def step(self, actions):
        self.current_step += 1

        # Define movement directions for both predator and prey
        directions = {
            0: np.array([-1, 0]),   # Up
            1: np.array([1, 0]),    # Down
            2: np.array([0, -1]),   # Left
            3: np.array([0, 1]),    # Right
            4: np.array([-1, -1]),  # Up-Left
            5: np.array([-1, 1]),   # Up-Right
            6: np.array([1, -1]),   # Down-Left
            7: np.array([1, 1]),    # Down-Right
        }

        predator_direction = directions[action]
        prey_direction = self._get_prey_direction()

        # Update predator position based on the action
        self.predator_position += predator_direction
        self.predator_position = np.clip(self.predator_position, 0, board_size - 1)

        # Update prey position based on distance-based movement
        prey_speed = 2  # Number of grid cells the prey can move in a single time step
        self.prey_position += prey_speed * prey_direction
        self.prey_position = np.clip(self.prey_position, 0, board_size - 1)

        # Calculate rewards
        if np.array_equal(self.predator_position, self.prey_position):
            reward = 1.0  # Predator caught the prey
            done = True
        elif self.current_step >= self.max_steps:
            reward = 0.0  # Maximum steps reached
            done = True
        else:
            reward = -0.1  # Default reward
            done = False

        state = self._get_state()
        return state, reward, done, {}

    def _get_state(self):
        # Example: Concatenate predator and prey positions as the state
        state = np.concatenate((self.predator_positions[0], self.predator_positions[1], self.prey_position))
        return state

# Define the MADDPG agent

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class MADDPGAgent:
    def __init__(self, state_size, action_size, discount_factor=0.99, tau=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.discount_factor = discount_factor
        self.tau = tau

        # Create local and target actor and critic networks
        self.local_actor = Actor(state_size, action_size)
        self.target_actor = Actor(state_size, action_size)
        self.local_critic = Critic(state_size, action_size)
        self.target_critic = Critic(state_size, action_size)

        # Initialize target networks with local network weights
        self.target_actor.load_state_dict(self.local_actor.state_dict())
        self.target_critic.load_state_dict(self.local_critic.state_dict())

        # Set up optimizers
        self.actor_optimizer = optim.Adam(self.local_actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.local_critic.parameters(), lr=0.001)

    def get_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action = self.local_actor(state).squeeze(0).numpy()
        return action

    def train(self, states, actions, rewards, next_states, dones, other_agents, other_actions):
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Update critic
        next_actions = torch.cat([agent.target_actor(next_state) for agent, next_state in zip(other_agents, next_states)], dim=1)
        target_values = rewards + self.discount_factor * (1 - dones) * self.target_critic(next_states, next_actions)
        predicted_values = self.local_critic(states, actions)
        critic_loss = F.mse_loss(predicted_values, target_values)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        pred_actions = torch.cat([agent.local_actor(state) for agent, state in zip(other_agents, states)], dim=1)
        actor_loss = -self.local_critic(states, pred_actions).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        for target_param, local_param in zip(self.target_actor.parameters(), self.local_actor.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
        for target_param, local_param in zip(self.target_critic.parameters(), self.local_critic.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

# Define the training loop

def train_agents(env, agents, num_episodes):
    for episode in range(num_episodes):
        states = env.reset()
        done = False
        total_rewards = [0.0] * len(agents)

        while not done:
            actions = []
            for i, agent in enumerate(agents):
                action = agent.get_action(states)
                actions.append(action)

            next_states, rewards, done = env.step(actions)

            for i, agent in enumerate(agents):
                agent.train(states, actions[i], rewards[i], next_states, done, agents, actions)

                total_rewards[i] += rewards[i]

            states = next_states

        print(f"Episode: {episode + 1}, Total Rewards: {total_rewards}")

# Create the environment and agents

env = PredatorPreyEnvironment()
agent1 = MADDPGAgent(env.state_size, env.action_size)
agent2 = MADDPGAgent(env.state_size, env.action_size)
agents = [agent1, agent2]

# Train the agents

num_episodes = 1000
train_agents(env, agents, num_episodes)
