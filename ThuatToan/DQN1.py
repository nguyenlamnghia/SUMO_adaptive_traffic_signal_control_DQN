import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import deque

# Step 2: Establish path to SUMO (SUMO_HOME)
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# Step 3: Add Traci module
import traci

# Step 4: Define Sumo configuration
Sumo_config = [
    'sumo',
    '-c', './DATN/datn.sumocfg',
    '--step-length', '0.10',
    '--delay', '1000',
    '--lateral-resolution', '0.1'
]

# Step 5: Open connection between SUMO and Traci
traci.start(Sumo_config)

# Step 6: Define Variables
q_PVB_J6_J3_0 = 0
q_PVB_J6_J3_1 = 0
q_PVB_J6_J3_2 = 0
q_PVB_J0_J3_0 = 0
q_PVB_J0_J3_1 = 0
q_PVB_J0_J3_2 = 0
q_HQC_J2_J3_0 = 0
q_HQC_J4_J3_0 = 0
current_phase = 0

# ---- Reinforcement Learning Hyperparameters ----
TOTAL_STEPS = 10000
ALPHA = 0.001           # Learning rate for Adam optimizer
GAMMA = 0.9             # Discount factor
EPSILON = 1.0           # Initial exploration rate
EPSILON_MIN = 0.01      # Minimum exploration rate
EPSILON_DECAY = 0.995   # Epsilon decay rate
ACTIONS = [0, 1]        # Action space: 0 = keep phase, 1 = switch phase
BATCH_SIZE = 32         # Mini-batch size for training
MEMORY_SIZE = 2000      # Replay buffer size
TARGET_UPDATE = 100     # Steps between target network updates

# ---- Additional Stability Parameters ----
MIN_GREEN_STEPS = 100
last_switch_step = -MIN_GREEN_STEPS

# ---- Replay Buffer ----
memory = deque(maxlen=MEMORY_SIZE)

# Step 7: Define Functions
def build_model(state_size, action_size):
    """
    Build a feedforward neural network for Q-value approximation.
    """
    model = keras.Sequential()
    model.add(layers.Input(shape=(state_size,)))
    model.add(layers.Dense(24, activation='relu'))
    model.add(layers.Dense(24, activation='relu'))
    model.add(layers.Dense(action_size, activation='linear'))
    model.compile(
        loss='mse',
        optimizer=keras.optimizers.Adam(learning_rate=ALPHA)
    )
    return model

def to_array(state_tuple):
    """
    Convert state tuple to NumPy array for neural network input.
    """
    return np.array(state_tuple, dtype=np.float32).reshape((1, -1))

# Create main and target DQN models
state_size = 9
action_size = len(ACTIONS)
dqn_model = build_model(state_size, action_size)
target_model = build_model(state_size, action_size)
target_model.set_weights(dqn_model.get_weights())  # Initialize target network

def get_reward(state):
    """
    Reward function: Negative of total queue length.
    """
    total_queue = sum(state[:-1])  # Exclude current_phase
    return -float(total_queue)

def get_state():
    global q_PVB_J6_J3_0, q_PVB_J6_J3_1, q_PVB_J6_J3_2, q_PVB_J0_J3_0, q_PVB_J0_J3_1, q_PVB_J0_J3_2, q_HQC_J2_J3_0, q_HQC_J4_J3_0, current_phase
    
    detector_PVB_J6_J3_0 = "PVB_J6_J3_0"
    detector_PVB_J6_J3_1 = "PVB_J6_J3_1"
    detector_PVB_J6_J3_2 = "PVB_J6_J3_2"
    detector_PVB_J0_J3_0 = "PVB_J0_J3_0"
    detector_PVB_J0_J3_1 = "PVB_J0_J3_1"
    detector_PVB_J0_J3_2 = "PVB_J0_J3_2"
    detector_HQC_J2_J3_0 = "HQC_J2_J3_0"
    detector_HQC_J4_J3_0 = "HQC_J4_J3_0"
    traffic_light_id = "clusterJ12_J2_J3_J6_#2more"
    
    q_PVB_J6_J3_0 = get_lane_occupancy(detector_PVB_J6_J3_0)
    q_PVB_J6_J3_1 = get_lane_occupancy(detector_PVB_J6_J3_1)
    q_PVB_J6_J3_2 = get_lane_occupancy(detector_PVB_J6_J3_2)
    q_PVB_J0_J3_0 = get_lane_occupancy(detector_PVB_J0_J3_0)
    q_PVB_J0_J3_1 = get_lane_occupancy(detector_PVB_J0_J3_1)
    q_PVB_J0_J3_2 = get_lane_occupancy(detector_PVB_J0_J3_2)
    q_HQC_J2_J3_0 = get_lane_occupancy(detector_HQC_J2_J3_0)
    q_HQC_J4_J3_0 = get_lane_occupancy(detector_HQC_J4_J3_0)
    current_phase = get_current_phase(traffic_light_id)
    
    return (q_PVB_J6_J3_0, q_PVB_J6_J3_1, q_PVB_J6_J3_2, q_PVB_J0_J3_0, q_PVB_J0_J3_1, q_PVB_J0_J3_2, q_HQC_J2_J3_0, q_HQC_J4_J3_0, current_phase)

def apply_action(action, tls_id="clusterJ12_J2_J3_J6_#2more"):
    """
    Execute the chosen action, respecting minimum green time.
    """
    global last_switch_step
    if action == 0:
        return
    elif action == 1:
        if current_simulation_step - last_switch_step >= MIN_GREEN_STEPS:
            program = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            num_phases = len(program.phases)
            next_phase = (get_current_phase(tls_id) + 1) % num_phases
            traci.trafficlight.setPhase(tls_id, next_phase)
            last_switch_step = current_simulation_step

def remember(state, action, reward, next_state, done):
    """
    Store experience in replay buffer.
    """
    memory.append((state, action, reward, next_state, done))

def replay():
    """
    Train DQN model using a mini-batch from replay buffer.
    """
    if len(memory) < BATCH_SIZE:
        return
    
    minibatch = random.sample(memory, BATCH_SIZE)
    states = np.zeros((BATCH_SIZE, state_size))
    targets = np.zeros((BATCH_SIZE, action_size))
    
    for i, (state, action, reward, next_state, done) in enumerate(minibatch):
        state_array = to_array(state)
        next_state_array = to_array(next_state)
        target = dqn_model.predict(state_array, verbose=0)[0]
        
        if done:
            target[action] = reward
        else:
            target_Q = target_model.predict(next_state_array, verbose=0)[0]
            target[action] = reward + GAMMA * np.max(target_Q)
        
        states[i] = state_array
        targets[i] = target
    
    dqn_model.fit(states, targets, epochs=1, verbose=0)

def get_action_from_policy(state):
    """
    Epsilon-greedy policy with decaying epsilon.
    """
    global EPSILON
    if random.random() < EPSILON:
        return random.choice(ACTIONS)
    else:
        state_array = to_array(state)
        Q_values = dqn_model.predict(state_array, verbose=0)[0]
        return int(np.argmax(Q_values))

def get_lane_occupancy(detector_id):
    return traci.lanearea.getLastStepOccupancy(detector_id)

def get_current_phase(tls_id):
    return traci.trafficlight.getPhase(tls_id)

# Step 8: Fully Online Continuous Learning Loop
step_history = []
reward_history = []
queue_history = []
cumulative_reward = 0.0

print("\n=== Starting Fully Online Continuous Learning (DQN) ===")
for step in range(TOTAL_STEPS):
    current_simulation_step = step
    
    state = get_state()
    action = get_action_from_policy(state)
    apply_action(action)
    
    traci.simulationStep()
    
    new_state = get_state()
    reward = get_reward(new_state)
    cumulative_reward += reward
    
    # Store experience (done flag is False unless simulation ends)
    done = step == TOTAL_STEPS - 1
    remember(state, action, reward, new_state, done)
    
    # Train on mini-batch
    replay()
    
    # Decay epsilon
    EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)
    
    # Update target network periodically
    if step % TARGET_UPDATE == 0:
        target_model.set_weights(dqn_model.get_weights())
    
    # Record data every step
    if step % 1 == 0:
        Q_values = dqn_model.predict(to_array(state), verbose=0)[0]
        print(f"Step {step}, Current_State: {state}, Action: {action}, New_State: {new_state}, Reward: {reward:.2f}, Cumulative Reward: {cumulative_reward:.2f}, Q-values: {Q_values}, Epsilon: {EPSILON:.4f}")
        step_history.append(step)
        reward_history.append(cumulative_reward)
        queue_history.append(sum(new_state[:-1]))

# Step 9: Close connection
traci.close()

# Print model summary
print("\nOnline Training completed.")
print("DQN Model Summary:")
dqn_model.summary()

# Visualization
plt.figure(figsize=(10, 6))
plt.plot(step_history, reward_history, marker='o', linestyle='-', label="Cumulative Reward")
plt.xlabel("Simulation Step")
plt.ylabel("Cumulative Reward")
plt.title("DQN Training: Cumulative Reward over Steps")
plt.legend()
plt.grid(True)
plt.savefig('cumulative_reward.png')

plt.figure(figsize=(10, 6))
plt.plot(step_history, queue_history, marker='o', linestyle='-', label="Total Queue Length")
plt.xlabel("Simulation Step")
plt.ylabel("Total Queue Length")
plt.title("DQN Training: Queue Length over Steps")
plt.legend()
plt.grid(True)
plt.savefig('queue_length.png')