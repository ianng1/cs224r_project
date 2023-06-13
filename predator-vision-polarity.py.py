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
        self.state_size = 6  # Example: 4-dimensional state
        self.action_size = 8  # Example: 8 actions (up, down, left, right)
        self.max_steps = 100  # Maximum number of steps per episode
        self.current_step = 0
        self.prey_position = np.random.randint(0, board_size, 2)  # Example: Prey's initial position
        self.predator_positions = [np.random.randint(0, board_size, 2), np.random.randint(0, board_size, 2)]  # Example: Initial positions of two predators

        self.prey_agent = DQNAgent(4, self.action_size)
        ckpt = torch.load('./models/prey_policy.pt')
        self.prey_agent.model.load_state_dict(ckpt['model_state_dict'])

    def reset(self):
        self.current_step = 0
        self.prey_position = [0, 0]  # Example: Prey's initial position
        self.predator_positions = [[2, 2], [3, 3]]  # Example: Initial positions of two predators
        state = self._get_state()
        return state

    def step(self, action):
        self.current_step += 1

        action1 = int(torch.argmax(action[0]))
        action2 = int(torch.argmax(action[1]))

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

        #prey_direction = self._get_prey_direction()
        
        # Update predator position based on the action
        self.predator_positions[0] += directions[action1]
        self.predator_positions[0] = np.clip(self.predator_positions[0], 0, board_size - 1)
        self.predator_positions[1] += directions[action2]
        self.predator_positions[1] = np.clip(self.predator_positions[1], 0, board_size - 1)

        # Update prey position based on distance-based movement
        prey_speed = 2  # Number of grid cells the prey can move in a single time step
        for i in range(prey_speed):
            state = self._get_state()
            prey_action = self.prey_agent.get_action(state)
            self.prey_position += directions[prey_action]
            self.prey_position = np.clip(self.prey_position, 0, board_size - 1)

        # Calculate rewards
        if np.array_equal(self.predator_positions[0], self.prey_position) or np.array_equal(self.predator_positions[1], self.prey_position):
            reward = 50.0  - (0.5 * self.current_step) # Predator caught the prey
            done = True
        elif self.current_step >= self.max_steps:
            reward = -50.0  # Maximum steps reached
            done = True
        else:
            reward = 0.03 * (10 - np.linalg.norm(np.mean([self.predator_positions[0], self.predator_positions[1]]) - self.prey_position))  # Distance between predator and prey
            done = False

        state = self._get_state()
        return state, reward, done, {}

    def _get_state_closer(self):
        # Example: Concatenate predator and prey positions as the state
        closer_predator_position = None
        dist_pred1 = np.linalg.norm(np.array(self.predator_positions[0]) - np.array(self.prey_position))
        dist_pred2 = np.linalg.norm(np.array(self.predator_positions[1]) - np.array(self.prey_position))
        closer_predator_position = self.predator_positions[np.argmin([dist_pred1, dist_pred2])]
        
        state = np.concatenate((closer_predator_position, self.prey_position))
        return state
    
    def _get_state(self):
        # Example: Concatenate predator and prey positions as the state
        state = np.concatenate((self. predator_positions[0],
                                self.predator_positions[1], self.prey_position))
        return state

vision_dict = {0: 1, 1: 5}

# Define the MADDPG agent
def within_vision_box(agent_location, prey_location, box_size = 5):
    box_radius = (box_size - 1) // 2 
    diff_x = abs(agent_location[0] - prey_location[0])
    diff_y = abs(agent_location[1] - prey_location[1])
    if (diff_x <= box_radius and diff_y <= box_radius):
      return True
    return False

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        #self.fc1 = nn.Linear(state_size, 32)
        self.fc1 = nn.Linear(100, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, action_size)
    
    def forward(self, state, id):
        #print("in target actor", state.shape)
        #partial observability
        """
        flat_state = state.squeeze(0) if len(state.shape) > 1 else state

        #agent_location = np.array([flat_state[0].item(), flat_state[1].item()] if (id == 0) else [flat_state[2].item(), flat_state[3].item()])
        #prey_location = np.array([flat_state[4].item(), flat_state[5].item()])

        locs = [int(s) for s in flat_state]
        agent_location = locs[:2] if (id == 0) else locs[2:4]
        other_location = locs[2:4] if (id == 0) else locs[:2]
        prey_location = locs[4:]
        
        input_state = np.zeros(board_size * board_size)
        input_state[board_size * agent_location[0] + agent_location[1]] = 1
        if self.within_vision_box(agent_location, prey_location) == False:
          input_state[board_size * prey_location[0] + prey_location[1]] = -1
        if self.within_vision_box(agent_location, other_location) == False:
          input_state[board_size * other_location[0] + other_location[1]] = 1

        board = torch.tensor(input_state, dtype = torch.float32)
        """

        flat_state = state.squeeze(0) if len(state.shape) > 1 else state
        locs = [int(s) for s in flat_state]
        agent_location = locs[:2] if (id == 0) else locs[2:4]
        other_location = locs[2:4] if (id == 0) else locs[:2]
        prey_location = locs[4:]
   
        input_state = np.zeros(board_size * board_size)
        input_state[board_size * agent_location[0] + agent_location[1]] = 1
        if within_vision_box(agent_location, prey_location, vision_dict[0]) == True or within_vision_box(other_location, prey_location, vision_dict[1]) == True:
          input_state[board_size * prey_location[0] + prey_location[1]] = -1
        input_state[board_size * other_location[0] + other_location[1]] = 1

        board = torch.tensor(input_state, dtype = torch.float32)
        x = F.relu(self.fc1(board))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))

        x = x.unsqueeze(0) if len(state.shape) > 1 else x
        
        #print(state.shape, input_state.shape, x.shape)
        return x

class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(100 + action_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        
    def forward(self, state, action):

        locs = [int(s) for s in state[0]]
       
        agent_location, other_location, prey_location = locs[:2], locs[2:4], locs[4:]
          
        input_state = np.zeros(board_size * board_size)
        input_state[board_size * agent_location[0] + agent_location[1]] = 1
        if within_vision_box(agent_location, prey_location, vision_dict[0]) == True or within_vision_box(other_location, prey_location, vision_dict[1]) == True:
          input_state[board_size * prey_location[0] + prey_location[1]] = -1
        input_state[board_size * other_location[0] + other_location[1]] = 1

        board_state = torch.tensor(input_state, dtype = torch.float32).unsqueeze(0)
        
        if (action.shape[0] == 2):
          board_state = torch.cat((board_state, board_state), dim = 0)

        x = torch.cat((board_state, action), dim=1)
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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    def get_action(self, state, id):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action = self.local_actor(state, id).squeeze(0)
        return action

    def train(self, states, actions, rewards, next_states, dones, other_agents, other_actions):
        states = torch.tensor(states, dtype=torch.float32).unsqueeze(0).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).unsqueeze(0).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(0).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(0).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(0).to(self.device)

        next_actions = torch.stack([agents[0].target_actor(next_states[0], id = 0), agents[1].target_actor(next_states[0], id = 1)], dim=0)
        next_states = torch.stack([next_states[0], next_states[0]], dim=0)
        target_values = rewards + self.discount_factor * (1 - dones) * self.target_critic(next_states, next_actions)
        predicted_values = self.local_critic(states, actions)

        critic_loss = F.mse_loss(predicted_values, target_values)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        states = torch.stack([states[0], states[0]], dim=0)
        pred_actions = torch.stack([agent.local_actor(state, id) for id, (agent, state) in enumerate(zip(other_agents, states))], dim=0)
        actor_loss = -self.local_critic(states, pred_actions).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        for target_param, local_param in zip(self.target_actor.parameters(), self.local_actor.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
        for target_param, local_param in zip(self.target_critic.parameters(), self.local_critic.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
    
    def save_model(self, location):
        torch.save({'model_state_dict': self.local_actor.state_dict(), 'optimizer_state_dict': self.actor_optimizer.state_dict()}, location)

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, action_size)

    def forward(self, state):

        states = [int(s) for s in state[0]]
        prey_position = [states[4], states[5]]
        predator_positions = [[states[0], states[1]], [states[2], states[3]]]

        closer_predator_position = None
        dist_pred1 = np.linalg.norm(np.array(predator_positions[0]) - np.array(prey_position))
        dist_pred2 = np.linalg.norm(np.array(predator_positions[1]) - np.array(prey_position))
        closer_predator_position = predator_positions[np.argmin([dist_pred1, dist_pred2])]
        
        L = np.concatenate((closer_predator_position, prey_position))
        x = torch.tensor(L, dtype = torch.float32).unsqueeze(0)
        
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
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
        reward = torch.tensor(reward, dtype=torch.float32).to(self.device)
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
all_rewards = []
all_steps = []

every_reward = []
every_step = []

event_log = [] #get asymmetry results
def train_agents(env, agents, num_episodes):
    reward_graph = 0
    steps_graph = 0
    for episode in range(num_episodes):
        states = env.reset()
        done = False
        total_rewards = [0.0] * len(agents)
        iter = 0

      
        event_log.append([])
        event_log[-1].append(states)
        while not done:

            iter += 1
            actions = []
            for i, agent in enumerate(agents):
                action = agent.get_action(states, id = i)
                actions.append(action)
            next_states, rewards, done, _ = env.step(actions)

            for i, agent in enumerate(agents):
                agent.train(states, actions[i], rewards, next_states, done, agents, actions)

                total_rewards[i] += rewards

            states = next_states

            event_log[-1].append(states)

        reward_graph += total_rewards[0]
        steps_graph += iter

        every_reward.append(total_rewards[0])
        every_step.append(iter)

        print(f"Episode: {episode + 1}, Moves until Capture: {iter}, Total Rewards: {total_rewards}")

        if (episode % 10 == 9):
            all_rewards.append(reward_graph/10)
            all_steps.append(steps_graph/10)
            reward_graph = 0
            steps_graph = 0

# Create the environment and agents

env = PredatorPreyEnvironment()
agent1 = MADDPGAgent(env.state_size, env.action_size)
agent2 = MADDPGAgent(env.state_size, env.action_size)
agents = [agent1, agent2]

# Train the agents

num_episodes = 1000
train_agents(env, agents, num_episodes)
