# Step 1: Add modules to provide access to specific libraries and functions
import os  # Module provides functions to handle file paths, directories, environment variables
import sys  # Module provides access to Python-specific system parameters and functions
import random
import numpy as np
import matplotlib.pyplot as plt  # Visualization

# Step 1.1: (Additional) Imports for Deep Q-Learning
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

# Step 3: Add Traci module to provide access to specific libraries and functions
import traci  # Static network information (such as reading and analyzing network files)

# Step 4: Define Sumo configuration
Sumo_config = [
    'sumo-gui',
    '-c', './DATN/datn.sumocfg',
    '--step-length', '0.10',
    '--delay', '1000',
    '--lateral-resolution', '0.1'
]

# Step 5: Open connection between SUMO and Traci
traci.start(Sumo_config)
#traci.gui.setSchema("View #0", "real world")

# -------------------------
# Step 6: Define Variables
# -------------------------

# Current phase of the traffic light
current_phase = 0

# ---- Reinforcement Learning Hyperparameters ----
ALPHA = 0.1            # Learning rate (α) between[0, 1]    #If α = 1, you fully replace the old Q-value with the newly computed estimate.
                                                            #If α = 0, you ignore the new estimate and never update the Q-value.
GAMMA = 0.9            # Discount factor (γ) between[0, 1]  #If γ = 0, the agent only cares about the reward at the current step (no future rewards).
                                                            #If γ = 1, the agent cares equally about current and future rewards, looking at long-term gains.

ACTIONS = [0, 1, 2, 3]       # The discrete action space (0 = 10s, 1 = 20s, 2 = 30s, 3 = 40s)
GREEN_TIMES = [10 , 20, 30, 40]  # The green times for each action

EPSILON = 1.0  # Initial exploration rate
EPSILON_MIN = 0.01 # Minimum exploration rate
EPSILON_DECAY = 0.995  # Decay rate per step

BATCH_SIZE = 32 # Batch size for training
REPLAY_BUFFER_SIZE = 10000 # Size of the replay buffer
TARGET_UPDATE_FREQ = 10  # Update target network every 10 cycles
RECORD_DATA_FREQ = 1  # Record data every 3 cycles

TLS_ID = "clusterJ12_J2_J3_J6_#2more" # Traffic light ID
YELLOW_TIME = 3  # Yellow light duration (seconds)
RED_TIME = 1  # Red light duration (seconds)

STEP_LENGTH = 0.1 # Step length (seconds)
SEC_TO_STEP = int(1/STEP_LENGTH) # Seconds to simulation steps

TIME_SIMULATION = 1000 # Total time for simulation (seconds)
STEP_SIMULATION = TIME_SIMULATION*SEC_TO_STEP    # Total number of simulation steps

replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

# -------------------------
# Step 7: Define Functions
# -------------------------
# Xây dựng mạng Quad-DQN
class QuadDQN:
    def __init__(self):
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(24, input_dim=state_size, activation='relu'),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(action_size, activation='linear')
        ])
        model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state):
        if np.random.rand() <= EPSILON:
            return random.randrange(action_size)
        state = np.expand_dims(state, axis=0) 
        q_values = self.model.predict(state, verbose=0)[0]
        return np.argmax(q_values[:action_size])

    def train(self):
        if len(replay_buffer) < BATCH_SIZE:
            return
        minibatch = random.sample(replay_buffer, BATCH_SIZE)
        states = np.array([experience[0] for experience in minibatch], dtype=np.float32)
        actions = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch], dtype=np.float32)
        next_states = np.array([experience[3] for experience in minibatch], dtype=np.float32)
        dones = np.array([experience[4] for experience in minibatch], dtype=np.float32)

        targets = self.model.predict(states, verbose=0)
        target_next = self.target_model.predict(next_states, verbose=0)

        # Update Q-values
        for i in range(BATCH_SIZE):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + GAMMA * np.max(target_next[i])

        self.model.fit(states, targets, epochs=1, verbose=0)
    
    def save(self, filename):
        self.model.save(filename)

    @staticmethod
    def load(filename):
        dqn = QuadDQN()
        dqn.model = tf.keras.models.load_model(filename)
        dqn.target_model = tf.keras.models.load_model(filename)
        return dqn

    # Create the DQN model
state_size = 9   # (PVB_J6_J3_0, PVB_J6_J3_1, PVB_J6_J3_2, PVB_J0_J3_0, PVB_J0_J3_1, PVB_J0_J3_2, HQC_J2_J3_0, HQC_J4_J3_0, current_phase)
action_size = len(ACTIONS)
dqn = QuadDQN()
try:
    dqn = QuadDQN.load("quad_dqn.keras")
    print("Loaded existing DQN model from quad_dqn.h5")
except Exception as e:
    print(f"Failed to load DQN model: {e}")
    print("No existing DQN model found, starting fresh training.")
    pass
detectors = [
    "PVB_J6_J3_0",
    "PVB_J6_J3_1",
    "PVB_J6_J3_2",
    "PVB_J0_J3_0",
    "PVB_J0_J3_1",
    "PVB_J0_J3_2",
    "HQC_J2_J3_0",
    "HQC_J4_J3_0"
]

def get_reward(before_waiting_time, after_waiting_time): #2. Constraint 2 
    """
    Simple reward function:
    Negative of total queue length to encourage shorter queues.
    """
    """Hàm thưởng: Hiệu thời gian chờ trước và sau khi áp dụng action"""
    return before_waiting_time - after_waiting_time


def get_total_waiting_time(junction_id):
    """Tính tổng thời gian chờ của tất cả xe tại các hướng"""
    total_waiting_time = 0.0
    for detector in detectors:
        waiting_time = traci.lane.getWaitingTime(detector)
        print(f"Waiting time for {detector}: {waiting_time}")
        total_waiting_time += waiting_time
    return total_waiting_time

def get_state():  #3&4. Constraint 3 & 4

    # Detector IDs for PVB_J6_J3
    detector_PVB_J6_J3_0 = "PVB_J6_J3_0"
    detector_PVB_J6_J3_1 = "PVB_J6_J3_1"
    detector_PVB_J6_J3_2 = "PVB_J6_J3_2"

    # Detector IDs for PVB_J6_J3
    detector_PVB_J0_J3_0 = "PVB_J0_J3_0"
    detector_PVB_J0_J3_1 = "PVB_J0_J3_1"
    detector_PVB_J0_J3_2 = "PVB_J0_J3_2"

    # Detector IDs for HQC_J2_J3
    detector_HQC_J2_J3_0 = "HQC_J2_J3_0"

    # Detector IDs for HQC_J4_J3
    detector_HQC_J4_J3_0 = "HQC_J4_J3_0"

    # Traffic light ID
    traffic_light_id = "clusterJ12_J2_J3_J6_#2more"
    
    # Get queue lengths from each detector
    q_PVB_J6_J3_0 = get_lane_occupancy(detector_PVB_J6_J3_0)
    q_PVB_J6_J3_1 = get_lane_occupancy(detector_PVB_J6_J3_1)
    q_PVB_J6_J3_2 = get_lane_occupancy(detector_PVB_J6_J3_2)

    q_PVB_J0_J3_0 = get_lane_occupancy(detector_PVB_J0_J3_0)
    q_PVB_J0_J3_1 = get_lane_occupancy(detector_PVB_J0_J3_1)
    q_PVB_J0_J3_2 = get_lane_occupancy(detector_PVB_J0_J3_2)

    q_HQC_J2_J3_0 = get_lane_occupancy(detector_HQC_J2_J3_0)
    q_HQC_J4_J3_0 = get_lane_occupancy(detector_HQC_J4_J3_0)
    # Update global variables with current queue lengths

    # Get current phase index
    current_phase = get_current_phase(traffic_light_id)

    return np.array([q_PVB_J6_J3_0, q_PVB_J6_J3_1, q_PVB_J6_J3_2, q_PVB_J0_J3_0, q_PVB_J0_J3_1, q_PVB_J0_J3_2, q_HQC_J2_J3_0, q_HQC_J4_J3_0, current_phase])
def apply_action(current_phase, green_time): #5. Constraint 5

    # current phase
    traci.trafficlight.setPhase(TLS_ID, current_phase)
    traci.trafficlight.setPhaseDuration(TLS_ID, green_time)
    for _ in range(green_time*SEC_TO_STEP):
        traci.simulationStep()

    # yellow phase
    traci.trafficlight.setPhase(TLS_ID, current_phase + 1)
    traci.trafficlight.setPhaseDuration(TLS_ID, YELLOW_TIME)
    for _ in range(YELLOW_TIME*SEC_TO_STEP):
        traci.simulationStep()

    # all red phase
    traci.trafficlight.setPhase(TLS_ID, current_phase + 2)
    traci.trafficlight.setPhaseDuration(TLS_ID, RED_TIME)
    for _ in range(RED_TIME*SEC_TO_STEP):
        traci.simulationStep()

    # Switch to next phase
    next_phase = (current_phase + 3) % 9
    traci.trafficlight.setPhase(TLS_ID, next_phase)


def get_lane_occupancy(detector_id): #8.Constraint 8
    return traci.lanearea.getLastStepOccupancy(detector_id)

def get_current_phase(tls_id): #8.Constraint 8
    return traci.trafficlight.getPhase(tls_id)

def get_total_vehicle_in_junction(junction_id):
    internal_lanes = [l for l in traci.lane.getIDList() if l.startswith(f":{junction_id}_")]
    vehicles_in_junction = []
    for lane in internal_lanes:
        vehicles_in_junction.extend(traci.lane.getLastStepVehicleIDs(lane))
    return vehicles_in_junction

# -------------------------
# Step 8: Fully Online Continuous Learning Loop
# -------------------------

# Lists to record data for plotting
step_history = []
reward_history = []
waiting_time_change_history = []
waiting_time_history = []
step = 0
cumulative_reward = 0.0

print("\n=== Starting Fully Online Continuous Learning (DQN) ===")
while step < STEP_SIMULATION:

    # Set all red phase if congestion detected
    # current_phase = get_current_phase(TLS_ID)
    # while len(get_total_vehicle_in_junction("J3")) > 20:
    #     print("Switching to all-red phase due to congestion")
    #     traci.trafficlight.setPhase(TLS_ID, 2)  # 4: Toàn đỏ
    #     traci.trafficlight.setPhaseDuration(TLS_ID, 5)
    #     for _ in range(5*SEC_TO_STEP):
    #         traci.simulationStep()
    #     step += 5*SEC_TO_STEP
    # traci.trafficlight.setPhase(TLS_ID, current_phase)

    before_waiting_time = get_total_waiting_time("J3")
    # get state and action
    state = get_state()
    action = dqn.get_action(state)

    # get current phase, and green time
    current_phase = get_current_phase(TLS_ID)
    green_time = GREEN_TIMES[action]

    # apply action
    apply_action(current_phase, green_time=green_time)

    after_waiting_time = get_total_waiting_time("J3")
    step += green_time*SEC_TO_STEP + YELLOW_TIME*SEC_TO_STEP + RED_TIME*SEC_TO_STEP

    # get new state and reward
    new_state = get_state()
    reward = get_reward(before_waiting_time, after_waiting_time)

    # Check if congestion detected
    vehicles_in_junction = get_total_vehicle_in_junction("J3")
    print(len(vehicles_in_junction))
    if len(vehicles_in_junction) > 20:
        # delete vehicles from junction
        for vehID in vehicles_in_junction:
            try:
                traci.vehicle.remove(vehID)
                print(f"Đã xóa xe {vehID} khỏi junction")
            except Exception as e:
                print(f"Lỗi khi xóa xe {vehID}: {e}")
        # Penalize for congestion
        reward -= 10

    # get total reward
    cumulative_reward += reward

    # is done
    done = step >= STEP_SIMULATION - 1

    # add to replay buffer
    replay_buffer.append((state, action, reward, new_state, done))

    # train
    dqn.train()

    # Cập nhật target model mỗi 10 pha
    if (step // (green_time*SEC_TO_STEP + YELLOW_TIME*SEC_TO_STEP + RED_TIME*SEC_TO_STEP)) % TARGET_UPDATE_FREQ == 0:
        dqn.update_target_model()

    # Decay epsilon
    EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)
    print(f"Step {step}, Current_State: {state}, Action: {action}, New_State: {new_state}, Epsilon: {EPSILON:.2f}, Reward: {reward:.2f}, Cumulative Reward: {cumulative_reward:.2f}")

    # Record data every RECORD_DATA_FREQ cycles
    if step // int((green_time + YELLOW_TIME + RED_TIME) * SEC_TO_STEP) % RECORD_DATA_FREQ == 0:
        step_history.append(step)
        reward_history.append(cumulative_reward)
        waiting_time_change_history.append(reward)  # Lưu hiệu thời gian chờ
        waiting_time_history.append(after_waiting_time)
dqn.save("quad_dqn.keras")


# -------------------------
# Step 9: Close connection between SUMO and Traci
# -------------------------
traci.close()

# ~~~ Print final model summary (replacing Q-table info) ~~~
# print("\nOnline Training completed.")
# print("DQN Model Summary:")
# dqn_model.summary()

# -------------------------
# Visualization of Results
# -------------------------

# Plot Cumulative Reward over Simulation Steps

# Vẽ biểu đồ
plt.figure(figsize=(10, 6))
plt.plot(step_history, reward_history, marker='o', linestyle='-', label="Cumulative Reward")
plt.xlabel("Simulation Step")
plt.ylabel("Cumulative Reward")
plt.title("RL Training (DQN): Cumulative Reward over Steps")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(step_history, waiting_time_change_history, marker='o', linestyle='-', label="Change in Waiting Time")
plt.xlabel("Simulation Step")
plt.ylabel("Change in Waiting Time (s)")
plt.title("RL Training (DQN): Change in Waiting Time per Cycle")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(step_history, waiting_time_history, marker='o', linestyle='-', label="Waiting Time")
plt.xlabel("Simulation Step")
plt.ylabel("Waiting Time (s)")
plt.title("RL Training (DQN): Waiting Time per Cycle")
plt.legend()
plt.grid(True)
plt.show()
