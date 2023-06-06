import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import gym

"""
Define cooperative navigation environment.
"""
class CooperativeNavigationEnv:
    def __init__(self, n_agents, n_landmarks):
        self.n_agents = n_agents
        self.n_landmarks = n_landmarks
        self.grid_size = 10

        self.agent_pos = np.zeros((self.n_agents, 2), dtype=np.int32)
        self.landmark_pos = np.zeros((self.n_landmarks, 2), dtype=np.int32)

        self.reset()

    def reset(self):
        self.agent_pos = self.generate_unique_positions(self.n_agents)
        self.landmark_pos = self.generate_unique_positions(self.n_landmarks)

        return self.get_observation()

    def step(self, actions):
        obs_next_n = []
        rewards_n = []
        dones_n = []

        for agent_id, action in enumerate(actions):
            self.move_agent(agent_id, action)  # Move the agent based on the action

            # Update observation, rewards, and done status for the agent
            obs_next_n.append(self.get_agent_observation(agent_id))
            rewards_n.append(self.calculate_agent_reward(agent_id))
            dones_n.append(self.check_collision(agent_id) or self.check_all_agents_on_landmark())

        return obs_next_n, rewards_n, dones_n, {}

    def move_agent(self, agent_id, action):
        action = np.argmax(action)
        if action == 0:  # Move Up
            self.agent_pos[agent_id][1] = max(self.agent_pos[agent_id][1] - 1, 0)
        elif action == 1:  # Move Down
            self.agent_pos[agent_id][1] = min(self.agent_pos[agent_id][1] + 1, self.grid_size - 1)
        elif action == 2:  # Move Left
            self.agent_pos[agent_id][0] = max(self.agent_pos[agent_id][0] - 1, 0)
        elif action == 3:  # Move Right
            self.agent_pos[agent_id][0] = min(self.agent_pos[agent_id][0] + 1, self.grid_size - 1)
        else:
            raise ValueError("Invalid action.")

    def check_collision(self, agent_id):
        for i in range(self.n_agents):
            if i != agent_id and np.array_equal(self.agent_pos[i], self.agent_pos[agent_id]):
                return True
        return False

    def check_all_agents_on_landmark(self):
        for agent_pos in self.agent_pos:
            if not np.any((agent_pos == self.landmark_pos).all(axis=1)):
                return False
        return True

    def calculate_agent_reward(self, agent_id):
        if (self.agent_pos[agent_id] == self.landmark_pos[agent_id]).all():
            return 1.0
        else:
            return 0.0

    def get_observation(self):
        obs_n = []
        for agent_id in range(self.n_agents):
            obs_n.append(self.get_agent_observation(agent_id))
        return obs_n

    def get_agent_observation(self, agent_id):
        observation = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        observation[self.agent_pos[agent_id][0], self.agent_pos[agent_id][1]] = 1.0

        for landmark_pos in self.landmark_pos:
            observation[landmark_pos[0], landmark_pos[1]] = 0.5

        for i in range(self.n_agents):
            if i != agent_id:
                observation[self.agent_pos[i][0], self.agent_pos[i][1]] = -0.5

        return observation

    @staticmethod
    def generate_unique_positions(num_positions):
        positions = set()
        while len(positions) < num_positions:
            position = np.random.randint(0, 10, size=2)
            positions.add(tuple(position))
        return np.array(list(positions))

"""
Define MADDPG agent.
"""
class MADDPGAgent:
    def __init__(self, state_dim, action_dim, hidden_dim, discount, tau, lr):
        self.actor = Actor(state_dim, action_dim, hidden_dim)
        self.critic = Critic(state_dim, action_dim, hidden_dim)

        self.target_actor = Actor(state_dim, action_dim, hidden_dim)
        self.target_critic = Critic(state_dim, action_dim, hidden_dim)

        self.discount = discount
        self.tau = tau

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.hard_update(self.target_actor, self.actor)
        self.hard_update(self.target_critic, self.critic)

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).squeeze(0).detach().numpy()
        return action

    def update(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state).flatten()
        action = torch.FloatTensor(action)
        #reward = torch.FloatTensor(reward)
        next_state = torch.FloatTensor(next_state).flatten()
        done = torch.FloatTensor(done)

        target_action = self.target_actor(next_state)
        target_critic_output = self.target_critic(next_state, target_action)
        target_value = reward + self.discount * target_critic_output * (1 - done.unsqueeze(1))

        current_value = self.critic(state, action)
        critic_loss = F.mse_loss(current_value, target_value)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        pred_action = self.actor(state)
        actor_loss = -self.critic(state, pred_action).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.target_actor, self.actor)
        self.soft_update(self.target_critic, self.critic)

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def hard_update(self, target, source):
        target.load_state_dict(source.state_dict())


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        #x = torch.flatten(x, start_dim=0)  # Flatten the input state
        x = torch.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        x = torch.cat((x, a), dim=-1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MADDPG:
    def __init__(self, n_agents, state_dim, action_dim, hidden_dim, discount, tau, lr):
        self.n_agents = n_agents
        self.agents = [MADDPGAgent(state_dim, action_dim, hidden_dim, discount, tau, lr) for i in range(self.n_agents)]

    def get_action(self, agent_id, state):
        return self.agents[agent_id].get_action(state.flatten())

    def update(self, agent_id, state, action, reward, next_state, done):
        self.agents[agent_id].update(state, action, reward, next_state, done)


"""
Define training process.
"""
def config():
    n_agents = 2
    state_dim = 100
    action_dim = 4
    hidden_dim = 128
    discount = 0.99
    tau = 0.01
    lr = 0.001

    maddpg = MADDPG(n_agents, state_dim, action_dim, hidden_dim, discount, tau, lr)
    return maddpg

def run():
    env = CooperativeNavigationEnv(n_agents=2, n_landmarks=2)
    maddpg = config()

    n_episodes = 1000
    max_steps = 100
    gamma = 0.99
    tau = 0.01

    for episode in range(n_episodes):
        obs = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            actions = [maddpg.get_action(agent_id, obs[agent_id]) for agent_id in range(env.n_agents)]
            next_obs, rewards, done, _ = env.step(actions)

            for agent_id in range(env.n_agents):
                maddpg.update(agent_id, obs[agent_id], actions[agent_id], rewards[agent_id], next_obs[agent_id], done)

            obs = next_obs
            episode_reward += np.sum(rewards)

            if all(done):
                break

        print(f"Episode {episode + 1}: Total Reward = {episode_reward}")

    print("Training complete.")

    obs = env.reset()
    done = [False] * env.n_agents

    while not all(done):
        actions = [maddpg.get_action(agent_id, obs[agent_id]) for agent_id in range(env.n_agents)]
        obs, _, done = env.step(actions)
        env.render()

run()