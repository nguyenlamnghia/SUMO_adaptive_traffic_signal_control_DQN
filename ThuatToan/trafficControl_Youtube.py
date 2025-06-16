# Step 1: Add modules to provide access to specific libraries and functions
import os
import sys
import traci
import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Step 2: Establish path to SUMO (SUMO_HOME)
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# Step 3: Add Traci module to provide access to specific libraries and functions
import traci

# Step 4: Define Sumo configuration
Sumo_config = [
    'sumo-gui',
    '-c', './DATN/datn.sumocfg',
    '--step-length', '0.05',
    '--delay', '1000',
    '--lateral-resolution', '0.1'
]

# Step 5: Open connection between SUMO and Traci
traci.start(Sumo_config)

# Step 6: Define simulation parameters
state_size = 4         # Number of inputs: [queue_length_N, S, E, W]
action_size = 4        # Actions: [green for N-S, E-W, left-turns, all-red]
gamma = 0.95           # Discount rate
epsilon = 1.0          # Exploration rate
epsilon_min = 0.01
epsilon_decay = 0.995
learning_rate = 0.001
memory = deque(maxlen=2000)

# Build DQN model
def build_model():
    model = Sequential()
    model.add(Dense(24, input_dim=state_size, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(action_size, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
    return model

model = build_model()

# Get state from SUMO simulation
def get_state():
    lanes = ['PVB_J0_J3_0', 'PVB_J6_J3_0', 'HQC_J3_J4_0', 'HQC_J2_J3_0']
    state = [traci.lane.getLastStepVehicleNumber(l) for l in lanes]
    return np.array(state)

# Choose action
def act(state):
    if np.random.rand() <= epsilon:
        return random.randrange(action_size)
    act_values = model.predict(np.expand_dims(state, axis=0), verbose=0)
    return np.argmax(act_values[0])

# Replay experience
def replay(batch_size):
    minibatch = random.sample(memory, batch_size)
    for state, action, reward, next_state, done in minibatch:
        target = reward
        if not done:
            target = reward + gamma * np.amax(model.predict(np.expand_dims(next_state, axis=0), verbose=0)[0])
        target_f = model.predict(np.expand_dims(state, axis=0), verbose=0)
        target_f[0][action] = target
        model.fit(np.expand_dims(state, axis=0), target_f, epochs=1, verbose=0)

# Apply action to SUMO simulation
def apply_action(action):
    if action == 0:  # Green for N-S
        traci.trafficlight.setPhase("clusterJ12_J2_J3_J6_#2more", 0)
    elif action == 1:  # Green for E-W
        traci.trafficlight.setPhase("clusterJ12_J2_J3_J6_#2more", 1)
    elif action == 2:  # Left-turns
        traci.trafficlight.setPhase("clusterJ12_J2_J3_J6_#2more", 2)
    elif action == 3:  # All-red
        traci.trafficlight.setPhase("clusterJ12_J2_J3_J6_#2more", 3)

# Step 7: Start simulation
episodes = 10
max_steps = 1000

for e in range(episodes):
    traci.load(Sumo_config[1:])  # Reload simulation for each episode
    state = get_state()
    for step in range(max_steps):
        state = get_state()
        action = act(state)
        apply_action(action)

        traci.simulationStep()

        # Choose and apply action
        
        # (Giả định action tương ứng với điều khiển tín hiệu ở đây - bạn cần triển khai)
        # traci.trafficlight.setPhase("junction_id", action)

        # Observe reward and next state
        next_state = get_state()
        reward = -sum(next_state)  # Minimize total vehicle waiting
        done = traci.simulation.getMinExpectedNumber() == 0

        # Store experience
        memory.append((state, action, reward, next_state, done))
        state = next_state

        # Train model
        if len(memory) > 32:
            replay(32)

        # Giảm epsilon để giảm exploration dần
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        if done:
            break

# Step 9: Close connection
traci.close()
