import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import csv
# Define the PredatorPreyEnvironment Gym environment
board_size = 10

class PredatorPreyEnvironment(gym.Env):
    def __init__(self):
        super(PredatorPreyEnvironment, self).__init__()
        self.state_size = 4  # Example: 4-dimensional state 
        self.action_size = 8  # Example: 8 actions (up, down, left, right, up-left, up-right, down-left, down-right)
        self.max_steps = 100  # Maximum number of steps per episode
        self.current_step = 0
        self.prey_position = np.array([0, 0])  # Example: Prey's initial position
        #self.predator_position = np.array([2, 2])  # Example: Initial position of the predator
        self.predator_position = np.random.randint(0, board_size, 2)
        
        self.prey_agent = DQNAgent(self.state_size, self.action_size)
        ckpt = torch.load('models/prey_policy.pt')
        self.prey_agent.model.load_state_dict(ckpt['model_state_dict'])
    
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

        predator_direction = directions[action]
        #prey_direction = self._get_prey_direction()

        # Update predator position based on the action
        self.predator_position += predator_direction
        self.predator_position = np.clip(self.predator_position, 0, board_size - 1)

        # Update prey position based on distance-based movement
        prey_speed = 2  # Number of grid cells the prey can move in a single time step
        for _ in range(prey_speed):
            state = self._get_state()
            prey_action = self.prey_agent.get_action(state)
            self.prey_position += directions[prey_action]
            self.prey_position = np.clip(self.prey_position, 0, board_size - 1)
        

        # Calculate rewards
        if np.array_equal(self.predator_position, self.prey_position):
            reward = 50.0 - (0.5 * self.current_step) # Predator caught the prey
            done = True
        elif self.current_step >= self.max_steps:
            reward = -50.0  # Maximum steps reached
            done = True
        else:
            #print(0.01 * np.sqrt(max(0, 100 - np.linalg.norm(self.predator_position - self.prey_position)**2))) 
            reward = (0.03 * (10 - np.linalg.norm(self.predator_position - self.prey_position)))#(0.05 * np.sqrt(max(0, 100 - np.linalg.norm(self.predator_position - self.prey_position)**2))) # Default reward
            done = False

        state = self._get_state()
        return state, reward, done, {}

    def _get_state(self):
        # Example: Concatenate predator and prey positions as the state
        state = np.concatenate((self.predator_position, self.prey_position))
        return state

    # not pretrained policy
    def _get_prey_direction(self):
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
        new_locations = [(self.prey_position + directions[i]) for i in range(8)]
        new_locations = [np.clip(loc, 0, board_size - 1) for loc in new_locations]
        distances = [np.linalg.norm(x - np.array([self.predator_position[0], self.predator_position[1]])) for x in new_locations]
        best = np.argmax(np.array(distances))
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

        q_values = self.model(state).gather(1, action.unsqueeze(0))
        next_q_values = self.target_model(next_state).max(1)[0].unsqueeze(1)
        target = reward + self.discount_factor * next_q_values * (1 - done)

        loss = self.criterion(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, location):
        torch.save({'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()}, location)
# Define the training loop

allRewards = []
allSteps = []
def train_agent(env, agent, num_episodes):
    reward_graph = 0
    steps_graph = 0
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        iter = 0
        while not done:
            iter += 1
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.train(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state
        reward_graph += total_reward
        steps_graph += iter

        print(f"Episode: {episode + 1}, Steps till Capture: {iter}, Total Reward: {total_reward}")
        if (episode % 100 == 99):
            
            allRewards.append(reward_graph/100)
            allSteps.append(steps_graph/100)
            reward_graph = 0
            steps_graph = 0
# Create the environment and agent

env = PredatorPreyEnvironment()
agent = DQNAgent(env.state_size, env.action_size)

# Train the agent

num_episodes = 50000
train_agent(env, agent, num_episodes)
agent.save_model("models/one_predator_policy_smart_prey.pt")

with open("graphs/single_pred_perf.csv", 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=",")
    for x in range(len(allRewards)):
        writer.writerow([allRewards[x], allSteps[x]])