
import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import deque
import traci

# Thiết lập SUMO_HOME
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# Cấu hình SUMO
Sumo_config = [
    'sumo',
    '-c', './DATN/datn.sumocfg',
    '--step-length', '0.10',
    # '--delay', '1000',
    '--lateral-resolution', '0.1'
]

# Bắt đầu kết nối với SUMO
traci.start(Sumo_config)

# Các tham số
current_phase = 0
ALPHA = 0.1
GAMMA = 0.9
ACTIONS = [0, 1, 2, 3]
GREEN_TIMES = [15, 25, 35, 45]
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 10000
TARGET_UPDATE_FREQ = 10
RECORD_DATA_FREQ = 3
TLS_ID = "clusterJ12_J2_J3_J6_#2more"
YELLOW_TIME = 3
RED_TIME = 1
STEP_LENGTH = 0.1
SEC_TO_STEP = int(1/STEP_LENGTH)
TIME_SIMULATION = 7200
STEP_SIMULATION = TIME_SIMULATION * SEC_TO_STEP
replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

# Danh sách detector
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

        for i in range(BATCH_SIZE):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + GAMMA * np.max(target_next[i])

        self.model.fit(states, targets, epochs=1, verbose=0)

state_size = 9  # 8 detectors + current_phase
action_size = len(ACTIONS)
dqn = QuadDQN()

def get_total_waiting_time(junction_id):
    """Tính tổng thời gian chờ của tất cả xe tại các hướng"""
    total_waiting_time = 0.0
    for detector in detectors:
        waiting_time = traci.lane.getWaitingTime(detector)
        total_waiting_time += waiting_time
    return total_waiting_time


def get_reward(before_waiting_time, after_waiting_time):
    """Hàm thưởng: Hiệu thời gian chờ trước và sau khi áp dụng action"""
    return before_waiting_time - after_waiting_time

def get_state(vehicles_passed):
    """State là số xe đi qua từng detector trong chu kỳ và pha hiện tại"""
    state = []
    for det in detectors:
        state.append(vehicles_passed.get(det, 0))
    state.append(get_current_phase(TLS_ID))
    return np.array(state, dtype=np.float32)

def apply_action(current_phase, green_time):
    """Áp dụng action và đếm số xe đi qua trong toàn bộ chu kỳ (xanh + vàng + đỏ)"""
    vehicles_passed = {detector: 0 for detector in detectors}
    previous_vehicles = {detector: set(traci.lanearea.getLastStepVehicleIDs(detector)) for detector in detectors}
    
    total_steps = int((green_time + YELLOW_TIME + RED_TIME) * SEC_TO_STEP)
    
    # Pha xanh
    traci.trafficlight.setPhase(TLS_ID, current_phase)
    traci.trafficlight.setPhaseDuration(TLS_ID, green_time)
    for _ in range(int(green_time * SEC_TO_STEP)):
        traci.simulationStep()
        for det in detectors:
            current_ids = set(traci.lanearea.getLastStepVehicleIDs(det))
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

    # Pha đỏ
    traci.trafficlight.setPhase(TLS_ID, current_phase + 2)
    traci.trafficlight.setPhaseDuration(TLS_ID, RED_TIME)
    for _ in range(int(RED_TIME * SEC_TO_STEP)):
        traci.simulationStep()
        for det in detectors:
            current_ids = set(traci.lanearea.getLastStepVehicleIDs(det))
            new_vehicles = current_ids - previous_vehicles[det]
            vehicles_passed[det] += len(new_vehicles)
            previous_vehicles[det] = current_ids

    # Chuyển sang pha tiếp theo
    next_phase = (current_phase + 3) % 9
    traci.trafficlight.setPhase(TLS_ID, next_phase)

    # In thông tin gỡ lỗi
    print(f"Vehicles passed in cycle: {vehicles_passed}")

    return vehicles_passed

def get_queue_length(detector_id):
    return traci.lanearea.getLastStepVehicleNumber(detector_id)

def get_current_phase(tls_id):
    return traci.trafficlight.getPhase(tls_id)

def get_total_vehicle_in_junction(junction_id):
    internal_lanes = [l for l in traci.lane.getIDList() if l.startswith(f":{junction_id}_")]
    vehicles_in_junction = []
    for lane in internal_lanes:
        vehicles_in_junction.extend(traci.lane.getLastStepVehicleIDs(lane))
    return vehicles_in_junction

# Vòng lặp học liên tục
step_history = []
reward_history = []
waiting_time_change_history = []
waiting_time_history = []
step = 0
cumulative_reward = 0.0
previous_vehicles_passed = {detector: 0 for detector in detectors}  # Lưu trạng thái trước đó

print("\n=== Starting Fully Online Continuous Learning (DQN) ===")
while step < STEP_SIMULATION:
    # Lấy thời gian chờ trước khi áp dụng action
    before_waiting_time = get_total_waiting_time("J3")
    
    # Lấy state từ vehicles_passed của chu kỳ trước (để new_state của chu kỳ trước thành current_state)
    state = get_state(previous_vehicles_passed)
    action = dqn.get_action(state)

    # Lấy pha hiện tại và thời gian xanh
    current_phase = get_current_phase(TLS_ID)
    green_time = GREEN_TIMES[action]

    # Áp dụng action và đếm số xe đi qua
    vehicles_passed = apply_action(current_phase, green_time)
    step += int((green_time + YELLOW_TIME + RED_TIME) * SEC_TO_STEP)

    # Lấy state mới và thời gian chờ sau action
    new_state = get_state(vehicles_passed)
    after_waiting_time = get_total_waiting_time("J3")

    # Tính thưởng
    reward = get_reward(before_waiting_time, after_waiting_time)
    cumulative_reward += reward

    # Kiểm tra tắc nghẽn
    vehicles_in_junction = get_total_vehicle_in_junction("J3")
    if len(vehicles_in_junction) > 20:
        for veh_id in vehicles_in_junction:
            try:
                traci.vehicle.remove(veh_id)
                print(f"Đã xóa xe {veh_id} khỏi junction")
            except Exception as e:
                print(f"Lỗi khi xóa xe {veh_id}: {e}")
        reward -= 10

    # Kiểm tra kết thúc
    done = step >= STEP_SIMULATION - 1

    # Lưu vào replay buffer
    replay_buffer.append((state, action, reward, new_state, done))

    # Huấn luyện
    dqn.train()

    # Cập nhật target model
    if (step // int((green_time + YELLOW_TIME + RED_TIME) * SEC_TO_STEP)) % TARGET_UPDATE_FREQ == 0:
        dqn.update_target_model()

    # Giảm epsilon
    EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)

    # Cập nhật previous_vehicles_passed để sử dụng cho state của chu kỳ tiếp theo
    previous_vehicles_passed = vehicles_passed

    print(f"Step {step}, Current_State: {state}, Action: {action}, New_State: {new_state}, Epsilon: {EPSILON:.2f}, Reward: {reward:.2f}, Cumulative Reward: {cumulative_reward:.2f}")

    # Ghi dữ liệu
    if step // int((green_time + YELLOW_TIME + RED_TIME) * SEC_TO_STEP) % RECORD_DATA_FREQ == 0:
        step_history.append(step)
        reward_history.append(cumulative_reward)
        waiting_time_change_history.append(reward)  # Lưu hiệu thời gian chờ
        waiting_time_history.append(after_waiting_time)

# Đóng kết nối
traci.close()

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
