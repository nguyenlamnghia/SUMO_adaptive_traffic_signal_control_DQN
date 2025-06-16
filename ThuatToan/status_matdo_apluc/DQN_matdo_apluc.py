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
from scipy.ndimage import uniform_filter1d  # For moving average

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
    'sumo-gui',
    '-c', './DATN/datn.sumocfg',
    '--step-length', '0.10',
    '--lateral-resolution', '0.1'
]

# ==========================================================
# Step 6: Define Constants & Parameters
# ==========================================================
ACTIONS = [0, 1, 2, 3, 4]
GREEN_TIMES = [15, 25, 35, 45, 55]

ALPHA = 0.1 # Learning rate
GAMMA = 0.9 # Discount factor
EPSILON = 1.0 # Exploration rate
EPSILON_MIN = 0.01 # Minimum exploration rate
EPSILON_DECAY = 0.995 # Decay rate for exploration
BATCH_SIZE = 32 # Batch size for training
# BATCH_SIZE = 8  # Giảm kích thước batch để tránh lỗi OOM trên GPU
REPLAY_BUFFER_SIZE = 10000
TARGET_UPDATE_FREQ = 10
RECORD_DATA_FREQ = 1

TLS_ID = "clusterJ12_J2_J3_J6_#2more"
YELLOW_TIME = 3
RED_TIME = 1

EPOCHS = 5  # Tăng số epoch để minh họa
STEP_LENGTH = 0.1
SEC_TO_STEP = int(1 / STEP_LENGTH)
TIME_SIMULATION = 7200
STEP_SIMULATION = TIME_SIMULATION * SEC_TO_STEP

state_size = 9
action_size = len(ACTIONS)
replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

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

def get_reward(after_total_vehicle, before_total_vehicle):
    # return before_avg_waiting_time - after_avg_waiting_time
    return -(before_total_vehicle - after_total_vehicle)

def get_total_waiting_time(_):
    return sum(traci.lane.getWaitingTime(d) for d in detectors)

def get_total_vehicle_in_lane():
    return sum(traci.lane.getLastStepVehicleNumber(d) for d in detectors)

def get_avg_waiting_time():
    total_vehicle_in_lane = get_total_vehicle_in_lane()
    if total_vehicle_in_lane == 0:
        return 0
    total_waiting_time = get_total_waiting_time("J3")
    return total_waiting_time / total_vehicle_in_lane

def get_lane_occupancy(detector_id):
    return traci.lanearea.getLastStepOccupancy(detector_id)

def get_current_phase(tls_id):
    return traci.trafficlight.getPhase(tls_id)

def apply_action(current_phase, green_time):
    traci.trafficlight.setPhase(TLS_ID, current_phase)
    traci.trafficlight.setPhaseDuration(TLS_ID, green_time * SEC_TO_STEP)
    for _ in range(green_time * SEC_TO_STEP):
        traci.simulationStep()

    traci.trafficlight.setPhase(TLS_ID, current_phase + 1)
    traci.trafficlight.setPhaseDuration(TLS_ID, YELLOW_TIME * SEC_TO_STEP)
    for _ in range(YELLOW_TIME * SEC_TO_STEP):
        traci.simulationStep()

    traci.trafficlight.setPhase(TLS_ID, current_phase + 2)
    traci.trafficlight.setPhaseDuration(TLS_ID, RED_TIME * SEC_TO_STEP)
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
# Lưu dữ liệu cho từng epoch
epoch_data = {i: {
    'cycle_count': [],
    'reward': [],
    'waiting_time_change': [],
    'waiting_time': [],
    'loss': [],
    'epsilon': [],
    'action': [],
    'queue_length': []
} for i in range(EPOCHS)}

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
    cycle_count = 0
    cumulative_reward = 0.0
    
    traci.start(Sumo_config)

    while step < STEP_SIMULATION:
        before_avg_waiting_time = get_avg_waiting_time()
        before_total_vehicle = get_total_vehicle_in_lane()
        state = get_state()
        action = dqn.get_action(state)

        current_phase = get_current_phase(TLS_ID)
        green_time = GREEN_TIMES[action]
        apply_action(current_phase, green_time)

        after_avg_waiting_time = get_avg_waiting_time()
        after_total_vehicle = get_total_vehicle_in_lane()
        new_state = get_state()
        reward = get_reward(before_total_vehicle, after_total_vehicle)
        step += (green_time + YELLOW_TIME + RED_TIME) * SEC_TO_STEP
        cycle_count += 1  # Tăng số chu kỳ

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
            epoch_data[i]['loss'].append(result.history['loss'][0])
            print(f"Loss: {result.history['loss'][0]:.4f}")

        if cycle_count % TARGET_UPDATE_FREQ == 0:
            dqn.update_target_model()

        EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)

        print(f"Step: {step}, Phase: {current_phase}, Cycle: {cycle_count}, Action: {action}, Reward: {reward:.2f}, Epsilon: {EPSILON:.2f}, Cumulative: {cumulative_reward:.2f}")

        if cycle_count % RECORD_DATA_FREQ == 0:
            epoch_data[i]['cycle_count'].append(cycle_count)
            epoch_data[i]['reward'].append(cumulative_reward)
            epoch_data[i]['waiting_time_change'].append(reward)
            epoch_data[i]['waiting_time'].append(after_avg_waiting_time)
            epoch_data[i]['epsilon'].append(EPSILON)
            epoch_data[i]['action'].append(action)
            epoch_data[i]['queue_length'].append(get_total_vehicle_in_lane())
    
    traci.close()

dqn.save("quad_dqn.keras")

# ==========================================================
# Step 9: Close SUMO and Visualize
# ==========================================================

# Visualization function with moving average
def plot_graph(x, y, title, ylabel, filename, moving_average=False, window_size=10, label="Data"):
    plt.plot(x, y, marker='o', linestyle='-', label=label)
    if moving_average:
        y_smooth = uniform_filter1d(y, size=window_size)
        plt.plot(x, y_smooth, linestyle='--', label=f"Moving Average (window={window_size})", linewidth=2)

# Plot Action Distribution
def plot_action_distribution(actions, title, filename):
    plt.figure(figsize=(10, 6))
    plt.hist(actions, bins=range(len(ACTIONS) + 1), align='left', rwidth=0.8, density=True)
    plt.xticks(range(len(ACTIONS)), [f"Action {i} ({GREEN_TIMES[i]}s)" for i in range(len(ACTIONS))])
    plt.title(title)
    plt.xlabel("Action")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(filename)
    plt.show()

# Plot graphs for each epoch
for i in range(EPOCHS):
    data = epoch_data[i]
    cycle_count = data['cycle_count']
    
    # Cumulative Reward
    plt.figure(figsize=(10, 6))
    plot_graph(cycle_count, data['reward'], f"Cumulative Reward over Cycles (Epoch {i+1})", 
               "Cumulative Reward", f"cumulative_reward_epoch_{i+1}.png", moving_average=True, window_size=10, label=f"Epoch {i+1}")
    plt.title(f"Cumulative Reward over Cycles (Epoch {i+1})")
    plt.xlabel("Cycle")
    plt.ylabel("Cumulative Reward")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"cumulative_reward_epoch_{i+1}.png")
    plt.show()

    # Average Waiting Time
    plt.figure(figsize=(10, 6))
    plot_graph(cycle_count, data['waiting_time'], f"Average Waiting Time over Cycles (Epoch {i+1})", 
               "Average Waiting Time (s)", f"average_waiting_time_epoch_{i+1}.png", label=f"Epoch {i+1}")
    plt.title(f"Average Waiting Time over Cycles (Epoch {i+1})")
    plt.xlabel("Cycle")
    plt.ylabel("Average Waiting Time (s)")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"average_waiting_time_epoch_{i+1}.png")
    plt.show()

    # Training Loss
    # plt.figure(figsize=(10, 6))
    # plot_graph(cycle_count, data['loss'][:len(cycle_count)], f"Training Loss over Cycles (Epoch {i+1})", 
    #            "Loss", f"training_loss_epoch_{i+1}.png", label=f"Epoch {i+1}")
    # plt.title(f"Training Loss over Cycles (Epoch {i+1})")
    # plt.xlabel("Cycle")
    # plt.ylabel("Loss")
    # plt.grid(True)
    # plt.legend()
    # plt.savefig(f"training_loss_epoch_{i+1}.png")
    # plt.show()

    # Epsilon
    plt.figure(figsize=(10, 6))
    plot_graph(cycle_count, data['epsilon'], f"Epsilon over Cycles (Epoch {i+1})", 
               "Epsilon", f"epsilon_epoch_{i+1}.png", label=f"Epoch {i+1}")
    plt.title(f"Epsilon over Cycles (Epoch {i+1})")
    plt.xlabel("Cycle")
    plt.ylabel("Epsilon")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"epsilon_epoch_{i+1}.png")
    plt.show()

    # Queue Length
    plt.figure(figsize=(10, 6))
    plot_graph(cycle_count, data['queue_length'], f"Queue Length over Cycles (Epoch {i+1})", 
               "Number of Vehicles", f"queue_length_epoch_{i+1}.png", label=f"Epoch {i+1}")
    plt.title(f"Queue Length over Cycles (Epoch {i+1})")
    plt.xlabel("Cycle")
    plt.ylabel("Number of Vehicles")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"queue_length_epoch_{i+1}.png")
    plt.show()

    # Action Distribution
    plot_action_distribution(data['action'], f"Action Distribution (Epoch {i+1})", 
                            f"action_distribution_epoch_{i+1}.png")

# Plot aggregated graphs to compare epochs
plt.figure(figsize=(12, 8))
for i in range(EPOCHS):
    data = epoch_data[i]
    cycle_count = data['cycle_count']
    plt.plot(cycle_count, data['reward'], label=f"Epoch {i+1}", alpha=0.7)
plt.title("Cumulative Reward Comparison Across Epochs")
plt.xlabel("Cycle")
plt.ylabel("Cumulative Reward")
plt.grid(True)
plt.legend()
plt.savefig("cumulative_reward_comparison.png")
plt.show()

plt.figure(figsize=(12, 8))
for i in range(EPOCHS):
    data = epoch_data[i]
    cycle_count = data['cycle_count']
    plt.plot(cycle_count, data['waiting_time'], label=f"Epoch {i+1}", alpha=0.7)
plt.title("Average Waiting Time Comparison Across Epochs")
plt.xlabel("Cycle")
plt.ylabel("Average Waiting Time (s)")
plt.grid(True)
plt.legend()
plt.savefig("average_waiting_time_comparison.png")
plt.show()

# Plot average metrics per epoch
avg_rewards = [np.mean(epoch_data[i]['reward']) for i in range(EPOCHS)]
avg_waiting_times = [np.mean(epoch_data[i]['waiting_time']) for i in range(EPOCHS)]

plt.figure(figsize=(10, 6))
plt.plot(range(1, EPOCHS + 1), avg_rewards, marker='o', linestyle='-', label="Average Reward")
plt.title("Average Reward per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Average Reward")
plt.grid(True)
plt.legend()
plt.savefig("average_reward_per_epoch.png")
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(range(1, EPOCHS + 1), avg_waiting_times, marker='o', linestyle='-', label="Average Waiting Time")
plt.title("Average Waiting Time per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Average Waiting Time (s)")
plt.grid(True)
plt.legend()
plt.savefig("average_waiting_time_per_epoch.png")
plt.show()