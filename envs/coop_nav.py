import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import gym

class CooperativeNavigationEnv(gym.Env):
    def __init__(self, n_agents, n_landmarks):
        self.n_agents = n_agents
        self.n_landmarks = n_landmarks

        # Define action and observation spaces
        self.action_space = [gym.spaces.Discrete(5) for _ in range(self.n_agents)]
        self.observation_space = [gym.spaces.Box(low=-1, high=1, shape=(10, 10)) for _ in range(self.n_agents)]

        # Define initial positions for agents and landmarks
        self.agent_pos = np.zeros((self.n_agents, 2))
        self.landmark_pos = np.zeros((self.n_landmarks, 2))
        self.target_landmark_idx = None

        self.steps = 0
        self.max_steps = 100

    def reset(self):
        self.steps = 0
        self.agent_pos = np.zeros((self.n_agents, 2))
        self.landmark_pos = np.random.randint(0, 10, size=(self.n_landmarks, 2))
        self.target_landmark_idx = np.random.randint(0, self.n_landmarks)

        obs_n = []
        for _ in range(self.n_agents):
            obs_n.append(self.get_observation(_))

        return obs_n

    def step(self, actions):
        self.steps += 1
        rewards_n = np.zeros(self.n_agents)
        dones_n = np.zeros(self.n_agents, dtype=bool)

        for i in range(self.n_agents):
            self.move_agent(i, actions[i])

            # Check for collision with other agents
            if self.check_collision(i):
                dones_n[i] = True
                break

            # Check if agent occupies the target landmark
            if self.agent_pos[i][0] == self.landmark_pos[self.target_landmark_idx][0] and \
                    self.agent_pos[i][1] == self.landmark_pos[self.target_landmark_idx][1]:
                rewards_n[i] = 1.0
                dones_n[i] = True
                break

        if self.steps >= self.max_steps:
            dones_n.fill(True)

        obs_n = []
        for i in range(self.n_agents):
            obs_n.append(self.get_observation(i))

        return obs_n, rewards_n, dones_n, {}

    def move_agent(self, agent_id, action):
        # Define agent's movement based on the action
        if action == 0:  # Move Up
            self.agent_pos[agent_id][1] = min(self.agent_pos[agent_id][1] + 1, 9)
        elif action == 1:  # Move Down
            self.agent_pos[agent_id][1] = max(self.agent_pos[agent_id][1] - 1, 0)
        elif action == 2:  # Move Left
            self.agent_pos[agent_id][0] = max(self.agent_pos[agent_id][0] - 1, 0)
        elif action == 3:  # Move Right
            self.agent_pos[agent_id][0] = min(self.agent_pos[agent_id][0] + 1, 9)
        # else action == 4:  # No movement

    def check_collision(self, agent_id):
        # Check if the agent's position overlaps with any other agent's position
        for i in range(self.n_agents):
            if i != agent_id and np.array_equal(self.agent_pos[agent_id], self.agent_pos[i]):
                return True
        return False

    def get_observation(self, agent_id):
        # Construct a grid representation of the environment for a given agent
        obs = np.zeros((10, 10))
        obs[self.agent_pos[agent_id][1]][self.agent_pos[agent_id][0]] = 1
        obs[self.landmark_pos[self.target_landmark_idx][1]][self.landmark_pos[self.target_landmark_idx][0]] = 2
        return obs

# Define the MADDPG algorithm
class MADDPG:
    def __init__(self, n_agents, obs_shape_n, action_shape_n):
        self.n_agents = n_agents
        self.actors = [Actor(obs_shape_n[i], action_shape_n[i]) for i in range(self.n_agents)]
        self.critics = [Critic(obs_shape_n, action_shape_n, self.n_agents) for _ in range(self.n_agents)]
        self.target_actors = [Actor(obs_shape_n[i], action_shape_n[i]) for i in range(self.n_agents)]
        self.target_critics = [Critic(obs_shape_n, action_shape_n, self.n_agents) for _ in range(self.n_agents)]

        self.actor_optimizer = [optim.Adam(actor.parameters(), lr=0.001) for actor in self.actors]
        self.critic_optimizer = [optim.Adam(critic.parameters(), lr=0.001) for critic in self.critics]

        self.buffer = ReplayBuffer(buffer_size)

        self.gamma = 0.99
        self.tau = 0.01

    def step(self, obs_n):
        actions = []
        for i, actor in enumerate(self.actors):
            action = actor.get_action(obs_n[i])
            actions.append(action)

        return actions

    def update(self, batch_size):
        if len(self.buffer) < batch_size:
            return

        transitions = self.buffer.sample(batch_size)
        obs_n, actions_n, rewards_n, obs_next_n, dones_n = zip(*transitions)

        obs_n = np.array(obs_n)
        actions_n = np.array(actions_n)
        rewards_n = np.array(rewards_n)
        obs_next_n = np.array(obs_next_n)
        dones_n = np.array(dones_n)

        obs_n = torch.FloatTensor(obs_n)
        actions_n = torch.FloatTensor(actions_n)
        rewards_n = torch.FloatTensor(rewards_n)
        obs_next_n = torch.FloatTensor(obs_next_n)
        dones_n = torch.FloatTensor(dones_n)

        # Update critics
        for agent_id in range(self.n_agents):
            self.critic_optimizer[agent_id].zero_grad()

            # Compute Q targets
            target_actions_n = []
            for i, target_actor in enumerate(self.target_actors):
                if i == agent_id:
                    target_actions_n.append(target_actor.get_action(obs_next_n[:, i, :]))
                else:
                    target_actions_n.append(target_actor.get_action(obs_n[:, i, :]))

            target_actions_n = torch.stack(target_actions_n, dim=1)
            target_q_next = self.target_critics[agent_id](obs_next_n, target_actions_n)
            target_q = rewards_n[:, agent_id] + self.gamma * (1 - dones_n[:, agent_id]) * target_q_next.detach()

            # Compute current Q values
            q = self.critics[agent_id](obs_n, actions_n)

            # Update critic
            critic_loss = nn.MSELoss()(q, target_q)
            critic_loss.backward()
            self.critic_optimizer[agent_id].step()

        # Update actors
        for agent_id in range(self.n_agents):
            self.actor_optimizer[agent_id].zero_grad()

            # Compute actor loss
            actions_n_pred = []
            for i, actor in enumerate(self.actors):
                if i == agent_id:
                    actions_n_pred.append(actor.get_action(obs_n[:, i, :]))
                else:
                    actions_n_pred.append(actor.get_action(obs_n[:, i, :]).detach())

            actions_n_pred = torch.stack(actions_n_pred, dim=1)
            actor_loss = -self.critics[agent_id](obs_n, actions_n_pred).mean()

            # Update actor
            actor_loss.backward()
            self.actor_optimizer[agent_id].step()

        # Update target networks
        for target_actor, actor in zip(self.target_actors, self.actors):
            self.soft_update(target_actor, actor, self.tau)

        for target_critic, critic in zip(self.target_critics, self.critics):
            self.soft_update(target_critic, critic, self.tau)

    def soft_update(self, target_net, net, tau):
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


# Define the actor network
class Actor(nn.Module):
    def __init__(self, obs_shape, action_shape):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(obs_shape, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_shape)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, obs):
        x = self.relu(self.fc1(obs))
        x = self.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))
        return x

    def get_action(self, obs):
        obs = torch.FloatTensor(obs)
        action = self.forward(obs)
        return action.detach().numpy()


# Define the critic network
class Critic(nn.Module):
    def __init__(self, obs_shape_n, action_shape_n, n_agents):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(np.sum(obs_shape_n) + np.sum(action_shape_n), 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

        self.relu = nn.ReLU()

    def forward(self, obs_n, actions_n):
        x = torch.cat([obs_n.view(obs_n.size(0), -1), actions_n.view(actions_n.size(0), -1)], dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Define the replay buffer
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []

    def push(self, obs_n, actions_n, rewards_n, obs_next_n, dones_n):
        transition = (obs_n, actions_n, rewards_n, obs_next_n, dones_n)
        self.buffer.append(transition)

        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return list(zip(*batch))

    def __len__(self):
        return len(self.buffer)


# Define the run function
def run(config):
    n_agents = 2
    n_landmarks = n_agents
    env = CooperativeNavigationEnv(n_agents, n_landmarks)  # Adjust the number of agents and landmarks as needed

    obs_shape_n = [env.observation_space[i].shape[0] for i in range(env.n_agents)]
    action_shape_n = [env.action_space[i].n for i in range(env.n_agents)]

    maddpg = MADDPG(n_agents, obs_shape_n, action_shape_n)
    replay_buffer = ReplayBuffer(config.buffer_size)

    obs_n = env.reset()

    for episode in range(config.n_episodes):
        episode_rewards = np.zeros(env.n_agents)
        for step in range(config.episode_length):
            actions = maddpg.step(obs_n)

            obs_n_next, rewards_n, dones_n, _ = env.step(actions)

            replay_buffer.push(obs_n, actions, rewards_n, obs_n_next, dones_n)
            obs_n = obs_n_next

            if len(replay_buffer) > config.batch_size:
                maddpg.update(config.batch_size)

            episode_rewards += rewards_n

            if np.all(dones_n):
                break

        print("Episode {}: Total Reward = {:.2f}".format(episode, np.sum(episode_rewards)))


# Define the configuration parameters
class Config:
    def __init__(self):
        self.n_episodes = 1000
        self.episode_length = 100
        self.buffer_size = 100000
        self.batch_size = 128
        self.updates_per_step = 1


# Create an instance of the configuration
config = Config()

# Run the MADDPG algorithm
run(config)
