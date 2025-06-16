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
GREEN_TIMES = [10, 20, 30, 40]  # The green times for each action

EPSILON = 1.0  # Initial exploration rate
EPSILON_MIN = 0.01 # Minimum exploration rate
EPSILON_DECAY = 0.995  # Decay rate per step

BATCH_SIZE = 32 # Batch size for training
REPLAY_BUFFER_SIZE = 10000 # Size of the replay buffer
TARGET_UPDATE_FREQ = 10  # Update target network every 10 cycles
RECORD_DATA_FREQ = 3  # Record data every 3 cycles

TLS_ID = "clusterJ12_J2_J3_J6_#2more" # Traffic light ID
YELLOW_TIME = 3  # Yellow light duration (seconds)
RED_TIME = 1  # Red light duration (seconds)

STEP_LENGTH = 0.1 # Step length (seconds)
SEC_TO_STEP = int(1/STEP_LENGTH) # Seconds to simulation steps

TIME_SIMULATION = 10000 # Total time for simulation (seconds)
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
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
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

    # Create the DQN model
state_size = 9   # (PVB_J6_J3_0, PVB_J6_J3_1, PVB_J6_J3_2, PVB_J0_J3_0, PVB_J0_J3_1, PVB_J0_J3_2, HQC_J2_J3_0, HQC_J4_J3_0, current_phase)
action_size = len(ACTIONS)
dqn = QuadDQN()


def get_reward(state): #2. Constraint 2 
    """
    Simple reward function:
    Negative of total queue length to encourage shorter queues.
    """
    total_queue = sum(state[:-1])  # Exclude the current_phase element
    reward = -float(total_queue)
    return reward

def get_state(vehicles_passed):
    """
    State chính là số xe đi qua trong chu kỳ vừa rồi của từng detector (làn)
    """
    state = []
    for det in detectors:
        state.append(vehicles_passed[det])
    # Thêm current_phase để mạng DQN biết pha hiện tại
    current_phase = get_current_phase(TLS_ID)
    state.append(current_phase)
    return np.array(state, dtype=np.float32)

# Danh sách các detector (làn) trong giao lộ
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
def apply_action(current_phase, green_time):
    # Trước khi chạy, reset bộ đếm xe đi qua cho từng detector
    vehicles_passed = {detector: 0 for detector in detectors}
    previous_vehicles = {detector: set(traci.lanearea.getLastStepVehicleIDs(detector)) for detector in detectors}
    
    total_steps = int((green_time + YELLOW_TIME + RED_TIME) * SEC_TO_STEP)
    
    # Thiết lập pha đèn hiện tại (pha xanh)
    traci.trafficlight.setPhase(TLS_ID, current_phase)
    traci.trafficlight.setPhaseDuration(TLS_ID, green_time)
    
    # Bước mô phỏng trong pha xanh
    for _ in range(int(green_time * SEC_TO_STEP)):
        traci.simulationStep()
        # Cập nhật xe đi qua cho từng detector
        for det in detectors:
            current_ids = set(traci.lanearea.getLastStepVehicleIDs(det))
            # Xe mới đi vào detector là những xe có trong current_ids nhưng không có trong previous_vehicles
            new_vehicles = current_ids - previous_vehicles[det]
            vehicles_passed[det] += len(new_vehicles)
            previous_vehicles[det] = current_ids

    # Pha vàng
    traci.trafficlight.setPhase(TLS_ID, current_phase + 1)
    traci.trafficlight.setPhaseDuration(TLS_ID, YELLOW_TIME)
    for _ in range(int(YELLOW_TIME * SEC_TO_STEP)):
        traci.simulationStep()
        for det in detectors:
            current_ids = set(traci.lanearea.getLastStepVehicleIDs(det))
            new_vehicles = current_ids - previous_vehicles[det]
            vehicles_passed[det] += len(new_vehicles)
            previous_vehicles[det] = current_ids

    # Pha đỏ toàn phần
    traci.trafficlight.setPhase(TLS_ID, current_phase + 2)
    traci.trafficlight.setPhaseDuration(TLS_ID, RED_TIME)
    for _ in range(int(RED_TIME * SEC_TO_STEP)):
        traci.simulationStep()
        for det in detectors:
            current_ids = set(traci.lanearea.getLastStepVehicleIDs(det))
            new_vehicles = current_ids - previous_vehicles[det]
            vehicles_passed[det] += len(new_vehicles)
            previous_vehicles[det] = current_ids

    # Chuyển pha tiếp theo (có thể điều chỉnh tùy kịch bản)
    next_phase = (current_phase + 3) % 8  # Giả sử có 8 pha (4 pha chính + 4 pha vàng/đỏ)
    traci.trafficlight.setPhase(TLS_ID, next_phase)

    # Trả về dict số xe đi qua từng detector trong chu kỳ
    return vehicles_passed


def get_queue_length(detector_id): #8.Constraint 8
    return traci.lanearea.getLastStepVehicleNumber(detector_id)

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
queue_history = []
step = 0
cumulative_reward = 0.0
reward = 0.0

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

    # get state and action
    state = get_state()
    action = dqn.get_action(state)

    # get current phase, and green time
    current_phase = get_current_phase(TLS_ID)
    print("Mật độ chiếm dụng",traci.lanearea.getLastStepOccupancy("HQC_J2_J3_0"))
    print("Tốc độ trung bình", traci.lanearea.getLastStepMeanSpeed("HQC_J2_J3_0"))
    print("Số lượng phương tiện đang dừng lại", traci.lanearea.getLastStepHaltingNumber("HQC_J2_J3_0"))
    green_time = GREEN_TIMES[action]

    # apply action
    apply_action(current_phase, green_time=green_time)
    step += green_time*SEC_TO_STEP + YELLOW_TIME*SEC_TO_STEP + RED_TIME*SEC_TO_STEP

    # get new state and reward
    new_state = get_state()
    reward = get_reward(new_state)

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
    if step // (green_time*SEC_TO_STEP + YELLOW_TIME*SEC_TO_STEP + RED_TIME*SEC_TO_STEP) % RECORD_DATA_FREQ == 0:
        # updated_q_vals = dqn_model.predict(to_array(state), verbose=0)[0]
        # print(f"Step {step}, Current_State: {state}, Action: {action}, New_State: {new_state}, Reward: {reward:.2f}, Cumulative Reward: {cumulative_reward:.2f}, Q-values(current_state): {updated_q_vals}")
        step_history.append(step)
        reward_history.append(cumulative_reward)
        queue_history.append(sum(new_state[:-1]))  # sum of queue lengths



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
plt.figure(figsize=(10, 6))
plt.plot(step_history, reward_history, marker='o', linestyle='-', label="Cumulative Reward")
plt.xlabel("Simulation Step")
plt.ylabel("Cumulative Reward")
plt.title("RL Training (DQN): Cumulative Reward over Steps")
plt.legend()
plt.grid(True)
plt.show()

# Plot Total Queue Length over Simulation Steps
plt.figure(figsize=(10, 6))
plt.plot(step_history, queue_history, marker='o', linestyle='-', label="Total Queue Length")
plt.xlabel("Simulation Step")
plt.ylabel("Total Queue Length")
plt.title("RL Training (DQN): Queue Length over Steps")
plt.legend()
plt.grid(True)
plt.show()