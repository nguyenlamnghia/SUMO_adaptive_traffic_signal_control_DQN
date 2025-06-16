# ==========================================================
# Step 1: Import Modules
# ==========================================================
import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import deque

# ==========================================================
# Step 2: Setup SUMO Path
# ==========================================================
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# ==========================================================
# Step 3: Import TraCI
# ==========================================================
import traci

# ==========================================================
# Step 4: SUMO Configuration
# ==========================================================
Sumo_config = [
    'sumo',
    '-c', './DATN/datn.sumocfg',
    '--step-length', '0.10',
    # '--delay', '1000',
    '--lateral-resolution', '0.1'
]

# ==========================================================
# Step 6: Define Constants & Parameters
# ==========================================================
ACTIONS = [0, 1, 2, 3]
GREEN_TIMES = [10, 20, 30, 40] # Duration of green phase for each action

ALPHA = 0.1 # Learning rate
GAMMA = 0.9 # Discount factor
EPSILON = 1.0 # Exploration rate
EPSILON_MIN = 0.01 # Minimum exploration rate
EPSILON_DECAY = 0.995 # Decay rate for exploration
BATCH_SIZE = 32 # Size of the minibatch for training
REPLAY_BUFFER_SIZE = 10000 # Size of the replay buffer
TARGET_UPDATE_FREQ = 10 # Frequency of target network updates
RECORD_DATA_FREQ = 1 # Frequency of data recording

TLS_ID = "clusterJ12_J2_J3_J6_#2more" # Traffic light ID for the junction
YELLOW_TIME = 3 # Duration of yellow phase in seconds
RED_TIME = 1 # Duration of red phase in seconds

EPOCHS = 2 # Number of training epochs
STEP_LENGTH = 0.1 # Length of each simulation step in seconds
SEC_TO_STEP = int(1 / STEP_LENGTH) # Convert seconds to simulation steps
TIME_SIMULATION = 1000 # Total simulation time in seconds
STEP_SIMULATION = TIME_SIMULATION * SEC_TO_STEP # Total number of simulation steps

state_size = 9 # Number of detectors + current phase
action_size = len(ACTIONS) # Number of actions (traffic light phases)
replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE) # Initialize replay buffer

# Define Detectors
detectors = [
    "PVB_J6_J3_0", "PVB_J6_J3_1", "PVB_J6_J3_2",
    "PVB_J0_J3_0", "PVB_J0_J3_1", "PVB_J0_J3_2",
    "HQC_J2_J3_0", "HQC_J4_J3_0"
]

# ==========================================================
# Step 7: Define Functions
# ==========================================================
class QuadDQN:
    def __init__(self):
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = keras.Sequential([
            layers.Dense(24, input_dim=state_size, activation='relu'),
            layers.Dense(24, activation='relu'),
            layers.Dense(action_size, activation='linear')
        ])
        model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=keras.optimizers.Adam(learning_rate=0.001))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state):
        if np.random.rand() <= EPSILON:
            return random.randrange(action_size)
        q_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)[0]
        return np.argmax(q_values)

    def train(self):
        if len(replay_buffer) < BATCH_SIZE:
            return
        minibatch = random.sample(replay_buffer, BATCH_SIZE)
        states, actions, rewards, next_states, dones = map(np.array, zip(*minibatch))

        targets = self.model.predict(states, verbose=0)
        target_next = self.target_model.predict(next_states, verbose=0)

        for i in range(BATCH_SIZE):
            targets[i][actions[i]] = rewards[i] if dones[i] else rewards[i] + GAMMA * np.max(target_next[i])

        return self.model.fit(states, targets, epochs=1, verbose=0)

    def save(self, filename):
        self.model.save(filename)

    @staticmethod
    def load(filename):
        dqn = QuadDQN()
        dqn.model = keras.models.load_model(filename)
        dqn.target_model = keras.models.load_model(filename)
        return dqn

def get_state():
    queue_values = [get_lane_occupancy(d) for d in detectors]
    current_phase = get_current_phase(TLS_ID)
    return np.array(queue_values + [current_phase])

# def get_reward(before, after):
#     # return before - after

def get_reward(before_avg_waiting_time, after_avg_waiting_time):
    return before_avg_waiting_time - after_avg_waiting_time

def get_total_waiting_time(_):
    return sum(traci.lane.getWaitingTime(d) for d in detectors)

def get_total_vehicle_in_lane():
    return sum(traci.lane.getLastStepVehicleNumber(d) for d in detectors)

def get_avg_waiting_time():
    total_vehicle_in_lane = get_total_vehicle_in_lane()
    if total_vehicle_in_lane == 0:
        return 0
    total_waiting_time = get_total_waiting_time("J3")
    return total_waiting_time / total_vehicle_in_lane if total_vehicle_in_lane > 0 else 0

def get_lane_occupancy(detector_id):
    return traci.lanearea.getLastStepOccupancy(detector_id)

def get_current_phase(tls_id):
    return traci.trafficlight.getPhase(tls_id)

def apply_action(current_phase, green_time):
    traci.trafficlight.setPhase(TLS_ID, current_phase)
    for _ in range(green_time * SEC_TO_STEP):
        traci.simulationStep()

    traci.trafficlight.setPhase(TLS_ID, current_phase + 1)
    for _ in range(YELLOW_TIME * SEC_TO_STEP):
        traci.simulationStep()

    traci.trafficlight.setPhase(TLS_ID, current_phase + 2)
    for _ in range(RED_TIME * SEC_TO_STEP):
        traci.simulationStep()

    next_phase = (current_phase + 3) % 9
    traci.trafficlight.setPhase(TLS_ID, next_phase)

def get_total_vehicle_in_junction(junction_id):
    lanes = [l for l in traci.lane.getIDList() if l.startswith(f":{junction_id}_")]
    return [veh for lane in lanes for veh in traci.lane.getLastStepVehicleIDs(lane)]

# ==========================================================
# Step 8: Main RL Loop
# ==========================================================
step_history = []
reward_history = []
waiting_time_change_history = []
waiting_time_history = []


loss_history = []
# step = 0
# cumulative_reward = 0.0

try:
    dqn = QuadDQN.load("quad_dqn.keras")
    print("Loaded existing DQN model.")
except Exception as e:
    print(f"Failed to load model: {e}, training new model.")
    dqn = QuadDQN()

print("\n=== Starting Fully Online Continuous Learning (DQN) ===")

for i in range(EPOCHS):
    print(f"\nEpoch {i + 1}/{EPOCHS}")
    step = 0
    cumulative_reward = 0.0
    
    traci.start(Sumo_config)

    while step < STEP_SIMULATION:
        # before_waiting_time = get_total_waiting_time("J3")
        before_avg_waiting_time = get_avg_waiting_time()

        state = get_state()
        action = dqn.get_action(state)

        current_phase = get_current_phase(TLS_ID)
        green_time = GREEN_TIMES[action]
        apply_action(current_phase, green_time)

        # after_waiting_time = get_total_waiting_time("J3")
        after_avg_waiting_time = get_avg_waiting_time()

        new_state = get_state()
        reward = get_reward(before_avg_waiting_time, after_avg_waiting_time)
        step += (green_time + YELLOW_TIME + RED_TIME) * SEC_TO_STEP

        vehicles_in_junction = get_total_vehicle_in_junction("J3")
        if len(vehicles_in_junction) > 20:
            for vehID in vehicles_in_junction:
                try:
                    traci.vehicle.remove(vehID)
                except Exception:
                    pass
            reward -= 10

        cumulative_reward += reward
        done = step >= STEP_SIMULATION
        replay_buffer.append((state, action, reward, new_state, done))
        result = dqn.train()
        if result is not None:
            loss_history.append(result.history['loss'][0])
            print(f"Loss: {result.history['loss'][0]:.4f}")


        if (step // ((green_time + YELLOW_TIME + RED_TIME) * SEC_TO_STEP)) % TARGET_UPDATE_FREQ == 0:
            dqn.update_target_model()

        EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)

        print(f"Step: {step}, Action: {action}, Reward: {reward:.2f}, Epsilon: {EPSILON:.2f}, Cumulative: {cumulative_reward:.2f}")

        if (step // ((green_time + YELLOW_TIME + RED_TIME) * SEC_TO_STEP)) % RECORD_DATA_FREQ == 0:
            step_history.append(step)
            reward_history.append(cumulative_reward)
            waiting_time_change_history.append(reward)
            # waiting_time_history.append(after_waiting_time)
    
    traci.close()

dqn.save("quad_dqn.keras")

# ==========================================================
# Step 9: Close SUMO and Visualize
# ==========================================================

# ----------------------------------------------------------
# Visualization
# ----------------------------------------------------------
def plot_graph(x, y, title, ylabel):
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o', linestyle='-')
    plt.title(title)
    plt.xlabel("Simulation Step")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()
    # save the plot
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")

plot_graph(step_history, reward_history, "Cumulative Reward over Steps", "Cumulative Reward")
plot_graph(step_history, waiting_time_change_history, "Change in Waiting Time per Cycle", "Change in Waiting Time (s)")
plot_graph(step_history, waiting_time_history, "Waiting Time per Cycle", "Waiting Time (s)")
