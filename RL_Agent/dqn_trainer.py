# RL_Agent/dqn_trainer.py

import tensorflow as tf
import numpy as np
import random
from collections import deque
import os

# Simple DQN agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state[np.newaxis], verbose=0)
        return np.argmax(act_values[0])  # best action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state[np.newaxis], verbose=0)[0])
            target_f = self.model.predict(state[np.newaxis], verbose=0)
            target_f[0][action] = target
            self.model.fit(state[np.newaxis], target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Mock environment interaction (replace this with your actual cloud environment logic)
def simulate_environment():
    state_size = 4
    action_size = 2
    agent = DQNAgent(state_size, action_size)
    episodes = 50

    for e in range(episodes):
        state = np.random.rand(state_size)
        for time in range(100):
            action = agent.act(state)
            next_state = np.random.rand(state_size)
            reward = np.random.rand()
            done = time == 99
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print(f"Episode {e+1}/{episodes} finished.")
                break
        if len(agent.memory) > 32:
            agent.replay(32)

    # Save model
    os.makedirs("RL_Agent/models", exist_ok=True)
    agent.model.save("RL_Agent/models/dqn_model.h5")
    print(" DQN model saved at RL_Agent/models/dqn_model.h5")

if __name__ == "__main__":
    simulate_environment()
