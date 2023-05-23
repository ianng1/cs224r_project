import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Define the environment

class PredatorPreyEnvironment:
    def __init__(self):
        self.state_size = 4  # Example: 4-dimensional state
        self.action_size = 8  # Example: 8 actions (up, down, left, right, diagonals)
        self.max_steps = 100  # Maximum number of steps per episode
        self.current_step = 0
        self.prey_position = [0, 0]  # Example: Prey's initial position
        self.predator_position = [2, 2]  # Example: Initial position of the predator

    def reset(self):
        self.current_step = 0
        self.prey_position = [0, 0]  # Example: Prey's initial position
        self.predator_position = [2, 2]  # Example: Initial position of the predator
        state = self._get_state()
        return state

    def step(self, action):
        self.current_step += 1

        # Update predator position based on the action
        if action == 0:  # Up
            self.predator_position[0] -= 1
        elif action == 1:  # Down
            self.predator_position[0] += 1
        elif action == 2:  # Left
            self.predator_position[1] -= 1
        elif action == 3:  # Right
            self.predator_position[1] += 1
        elif action == 4: 
            self.predator_position[0] += 1
            self.predator_position[1] += 1
        elif action == 5:
            self.predator_position[0] -= 1
            self.predator_position[1] += 1
        elif action == 6:
            self.predator_position[0] += 1
            self.predator_position[1] -= 1
        elif action == 8:
            self.predator_position[0] -= 1
            self.predator_position[1] += 1

        # Clip position within the boundaries of the environment
        self.predator_position = np.clip(self.predator_position, 0, 3)

        # Update prey position randomly
        self.prey_position[0] += np.random.randint(-1, 2)
        self.prey_position[1] += np.random.randint(-1, 2)
        self.prey_position = np.clip(self.prey_position, 0, 3)

        # Calculate rewards
        if self.predator_position == self.prey_position:
            reward = 1.0  # Predator caught the prey
            done = True
        elif self.current_step >= self.max_steps:
            reward = 0.0  # Maximum steps reached
            done = True
        else:
            reward = -0.1  # Default reward
            done = False

        state = self._get_state()
        return state, reward, done

    def _get_state(self):
        # Example: Concatenate predator and prey positions as the state
        state = np.concatenate((self.predator_position, self.prey_position))
        return state

# Define the Q-Network

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the DQN agent

class DQNAgent:
    def __init__(self, state_size, action_size, discount_factor=0.99, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate

        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.99  # Decay rate for exploration rate
        self.epsilon_min = 0.01  # Minimum exploration rate

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = QNetwork(state_size, action_size).to(self.device)
        self.target_model = QNetwork(state_size, action_size).to(self.device)
        self.update_target_model()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)  # Exploration
        else:
            return torch.argmax(q_values, dim=1).item()  # Exploitation

    def train(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(self.device)
        action = torch.tensor(action, dtype=torch.long).unsqueeze(0).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(0).to(self.device)
        done = torch.tensor(done, dtype=torch.float32).unsqueeze(0).to(self.device)

        q_values = self.model(state).gather(1, action)
        next_q_values = self.target_model(next_state).max(1)[0].unsqueeze(1)
        target = reward + self.discount_factor * next_q_values * (1 - done)

        loss = self.criterion(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Define the training loop

def train_agent(env, agent, num_episodes):
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.train(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state

        print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

# Create the environment and agent

env = PredatorPreyEnvironment()
agent = DQNAgent(env.state_size, env.action_size)

# Train the agent

num_episodes = 1000
train_agent(env, agent, num_episodes)
