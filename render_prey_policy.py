import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import csv
import matplotlib.pyplot as plt
# Define the PredatorPreyEnvironment Gym environment

class PredatorPreyEnvironment(gym.Env):
    def __init__(self):
        super(PredatorPreyEnvironment, self).__init__()
        self.state_size = 4  # Example: 4-dimensional state
        self.action_size = 8  # Example: 8 actions (up, down, left, right, up-left, up-right, down-left, down-right)
        self.max_steps = 100  # Maximum number of steps per episode
        self.current_step = 0
        self.prey_position = np.array([np.random.randint(0, 9), np.random.randint(0, 9)])  # Example: Prey's initial position
        self.predator_position = np.array([np.random.randint(0, 9), np.random.randint(0, 9)])  # Example: Initial position of the predator

    def reset(self):
        self.current_step = 0
        self.prey_position = np.array([np.random.randint(0, 9), np.random.randint(0, 9)])
        #self.prey_position = np.array([2, 2])  # Example: Prey's initial position
        self.predator_position = np.array([np.random.randint(0, 9), np.random.randint(0, 9)])  # Example: Initial position of the predator
        state = self._get_state()
        return state

    def step(self, action):
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

        prey_direction = directions[action]
        predator_direction = self._get_predator_direction()

        # Update predator position based on the action
        prey_speed = 3
        if (self.current_step % prey_speed == 0):
            self.predator_position += predator_direction
            self.predator_position = np.clip(self.predator_position, 0, 10)
        if np.array_equal(self.predator_position, self.prey_position):
            reward = -10.0  # Predator caught the prey
            done = True
        else:
            # Update prey position based on distance-based movement
             # Number of grid cells the prey can move in a single time step
            self.prey_position += prey_direction
            self.prey_position = np.clip(self.prey_position, 0, 10)

            # Calculate rewards
            if np.array_equal(self.predator_position, self.prey_position):
                reward = -10.0  # Predator caught the prey
                done = True
            elif self.current_step >= self.max_steps:
                reward = 10.0  # Maximum steps reached
                done = True
            else:
                reward = 1  # Default reward
                done = False

        state = self._get_state()
        return state, reward, done, {}

    def _get_state(self):
        # Example: Concatenate predator and prey positions as the state
        state = np.concatenate((self.predator_position, self.prey_position))
        return state

    def _get_predator_direction(self):
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
        new_locations = [(self.predator_position + directions[i]) for i in range(8)]
        new_locations = [np.clip(loc, 0, 10) for loc in new_locations]
        distances = [np.linalg.norm(x - np.array([self.prey_position[0], self.prey_position[1]])) for x in new_locations]
        best = np.argmin(np.array(distances))
        return directions[best]

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

        #print()
        #q_values = self.model(state).gather(1, action)
        q_values = self.model(state)[0][action]
        next_q_values = self.target_model(next_state).max(1)[0].unsqueeze(1)
        target = reward + self.discount_factor * next_q_values * (1 - done)

        loss = self.criterion(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self, location):
        torch.save(self.model.state_dict(), location)
# Define the training loop

def train_agent(env, agent, num_episodes):
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.train(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state

        print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

# Create the environment and agent

env = PredatorPreyEnvironment()
agent = DQNAgent(env.state_size, env.action_size)

ckpt = torch.load('models/prey_policy.pt')
agent.model.load_state_dict(ckpt['model_state_dict'])
agent.optimizer.load_state_dict(ckpt['optimizer_state_dict'])

state = env.reset()
done = False
total_reward = 0.0

with open('prey_train_movements.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=",")
    while not done:
        writer.writerow([state[0], state[1], state[2], state[3]])
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.train(state, action, reward, next_state, done)
        total_reward += reward
        state = next_state
        





plt.ion()
with open('prey_train_movements.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=",")
    for row in reader:
        ar = np.ones((11, 11, 3))
        ar[int(row[0]), int(row[1])] = np.array([255, 0, 0])
        ar[int(row[2]), int(row[3])] = np.array([0, 0, 255])
        plt.imshow(ar)
        plt.show()
        plt.pause(0.02)