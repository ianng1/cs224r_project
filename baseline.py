import numpy as np
import tensorflow as tf

# Define the environment

class PredatorPreyEnvironment:
    def __init__(self):
        self.state_size = 4  # Example: 4-dimensional state
        self.action_size = 4  # Example: 4 actions (up, down, left, right)
        self.max_steps = 100  # Maximum number of steps per episode
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

        # Update predator positions based on the actions
        for i, action in enumerate(actions):
            if action == 0:  # Up
                self.predator_positions[i][0] -= 1
            elif action == 1:  # Down
                self.predator_positions[i][0] += 1
            elif action == 2:  # Left
                self.predator_positions[i][1] -= 1
            elif action == 3:  # Right
                self.predator_positions[i][1] += 1

            # Clip positions within the boundaries of the environment
            self.predator_positions[i] = np.clip(self.predator_positions[i], 0, 3)

        # Update prey position randomly
        self.prey_position[0] += np.random.randint(-1, 2)
        self.prey_position[1] += np.random.randint(-1, 2)
        self.prey_position = np.clip(self.prey_position, 0, 3)

        # Calculate rewards
        rewards = []
        done = False
        for i in range(len(actions)):
            if self.predator_positions[i] == self.prey_position:
                rewards.append(1.0)  # Predator caught the prey
                done = True
            elif self.current_step >= self.max_steps:
                rewards.append(0.0)  # Maximum steps reached
                done = True
            else:
                rewards.append(-0.1)  # Default reward

        state = self._get_state()
        return state, rewards, done

    def _get_state(self):
        # Example: Concatenate predator and prey positions as the state
        state = np.concatenate((self.predator_positions[0], self.predator_positions[1], self.prey_position))
        return state

# Define the MADDPG agent

class MADDPGAgent:
    def __init__(self, state_size, action_size, discount_factor=0.99, tau=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.discount_factor = discount_factor
        self.tau = tau

        # Create local and target actor and critic networks
        self.local_actor = self._build_actor_model()
        self.target_actor = self._build_actor_model()
        self.local_critic = self._build_critic_model()
        self.target_critic = self._build_critic_model()

        # Initialize target networks with local network weights
        self.update_target_networks(tau=1.0)

    def _build_actor_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=(self.state_size,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='tanh')
        ])
        return model

    def _build_critic_model(self):
        state_input = tf.keras.layers.Input(shape=(self.state_size,))
        action_input = tf.keras.layers.Input(shape=(self.action_size,))
        x = tf.keras.layers.Concatenate()([state_input, action_input])
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dense(1)(x)
        model = tf.keras.models.Model(inputs=[state_input, action_input], outputs=x)
        return model

    def update_target_networks(self, tau=None):
        if tau is None:
            tau = self.tau
        for local_weight, target_weight in zip(self.local_actor.weights, self.target_actor.weights):
            target_weight.assign(tau * local_weight + (1.0 - tau) * target_weight)
        for local_weight, target_weight in zip(self.local_critic.weights, self.target_critic.weights):
            target_weight.assign(tau * local_weight + (1.0 - tau) * target_weight)

    def get_action(self, state):
        state = np.reshape(state, [1, self.state_size])
        action = self.local_actor.predict(state)[0]
        return action

    def train(self, states, actions, rewards, next_states, dones, other_agents, other_actions):
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        # Update critic
        next_actions = [agent.target_actor.predict(np.reshape(next_state, (1, -1)))[0] for agent, next_state in zip(other_agents, next_states)]
        next_actions = np.array(next_actions)
        target_q_values = rewards[:, np.newaxis] + self.discount_factor * self.target_critic.predict([next_states, next_actions])
        self.local_critic.train_on_batch([states, actions], target_q_values)

        # Update actor
        with tf.GradientTape() as tape:
            pred_actions = [agent.local_actor.predict(np.reshape(state, (1, -1)))[0] for agent, state in zip(other_agents, states)]
            pred_actions = np.array(pred_actions)
            actor_loss = -tf.reduce_mean(self.local_critic([states, pred_actions]))
        actor_gradients = tape.gradient(actor_loss, self.local_actor.trainable_variables)
        self.local_actor.optimizer.apply_gradients(zip(actor_gradients, self.local_actor.trainable_variables))

        # Update target networks
        self.update_target_networks()

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