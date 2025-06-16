
import traci
import numpy as np
import tensorflow as tf
from collections import deque
import random
import os

# Thiết lập tham số
state_size = 11  # [density_north, density_south, density_east, density_west, waiting_time_north, waiting_time_south, waiting_time_east, waiting_time_west, total_waiting_time, total_density, phase]
action_size = 3  # 0: 10s, 1: 20s, 2: 30s
green_times = [10, 20, 30]
yellow_time = 3
red_time = 1
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
learning_rate = 0.001
batch_size = 32
memory = deque(maxlen=2000)

# Hệ số phần thưởng
k1 = 10  # Hiệu tổng thời gian chờ
k2 = 600  # Hiệu tổng mật độ
k3 = 3  # Tổng số xe còn lại

# Cấu hình làn (có thể chỉnh sửa)
lane_config = {
    "north": ["PVB_J0_J3_0", "PVB_J0_J3_1", "PVB_J0_J3_2"],  # Có thể thêm lane: ["PVB_J0_J3_0", "PVB_J0_J3_1"]
    "south": ["PVB_J6_J3_0", "PVB_J6_J3_1", "PVB_J6_J3_2"],
    "east": ["HQC_J3_J4_0"],
    "west": ["HQC_J2_J3_0"]
}

# ID đèn giao thông (có thể chỉnh sửa)
traffic_light_id = "clusterJ12_J2_J3_J6_#2more"

# Lưu kích thước loại xe
vehicle_sizes = {}

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
            tf.keras.layers.Dense(action_size + 1, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state):
        if np.random.rand() <= epsilon:
            return random.randrange(action_size)
        q_values = self.model.predict(state, verbose=0)[0]
        return np.argmax(q_values[:action_size])

    def train(self):
        if len(memory) < batch_size:
            return
        minibatch = random.sample(memory, batch_size)
        states = np.zeros((batch_size, state_size))
        next_states = np.zeros((batch_size, state_size))
        actions, rewards, dones = [], [], []

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            states[i] = state
            next_states[i] = next_state
            actions.append(action)
            rewards.append(reward)
            dones.append(done)

        targets = self.model.predict(states, verbose=0)
        target_next = self.target_model.predict(next_states, verbose=0)

        for i in range(batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                state_value = target_next[i][-1]
                max_action_value = np.max(target_next[i][:action_size])
                targets[i][actions[i]] = rewards[i] + gamma * (max_action_value + state_value)

        self.model.fit(states, targets, epochs=1, verbose=0)

    def save(self, filename):
        self.model.save(filename)

    @staticmethod
    def load(filename):
        dqn = QuadDQN()
        dqn.model = tf.keras.models.load_model(filename)
        dqn.target_model = tf.keras.models.load_model(filename)
        return dqn

# Lấy kích thước xe từ SUMO
def get_vehicle_sizes():
    global vehicle_sizes
    vehicle_types = ["motorcycle", "passenger", "bus", "truck"]
    for vtype in vehicle_types:
        # Giả định có ít nhất một xe của loại này trong mô phỏng
        for veh_id in traci.vehicle.getIDList():
            if traci.vehicle.getTypeID(veh_id) == vtype:
                length = traci.vehicle.getLength(veh_id)
                width = traci.vehicle.getWidth(veh_id)
                vehicle_sizes[vtype] = {"area": length * width}
                break
        if vtype not in vehicle_sizes:
            # Giá trị mặc định nếu không tìm thấy xe
            vehicle_sizes[vtype] = {"area": {"motorcycle": 3.6, "car": 14.95, "bus": 36, "truck": 22.5}.get(vtype, 10)}

# Tính mật độ và số xe cho một làn
def calculate_lane_metrics(lane_id):
    queue = traci.lane.getLastStepVehicleNumber(lane_id)
    waiting_time = traci.lane.getWaitingTime(lane_id) / max(1, queue) if queue > 0 else 0
    veh_ids = traci.lane.getLastStepVehicleIDs(lane_id)
    total_area = 0
    for veh_id in veh_ids:
        vtype = traci.vehicle.getTypeID(veh_id)
        if vtype not in vehicle_sizes:
            vehicle_sizes[vtype] = {"area": traci.vehicle.getLength(veh_id) * traci.vehicle.getWidth(veh_id)}
        total_area += vehicle_sizes[vtype]["area"]
    lane_area = traci.lane.getLength(lane_id) * traci.lane.getWidth(lane_id)
    density = total_area / lane_area if lane_area > 0 else 0
    return density, queue, waiting_time

# Hàm lấy state
def get_state():
    densities = [0] * 4
    queue = [0] * 4
    waiting_time = [0] * 4
    
    for i, direction in enumerate(["north", "south", "east", "west"]):
        lanes = lane_config[direction]
        lane_densities = []
        lane_queues = []
        lane_waiting_times = []
        for lane_id in lanes:
            d, q, wt = calculate_lane_metrics(lane_id)
            lane_densities.append(d)
            lane_queues.append(q)
            lane_waiting_times.append(wt)
        densities[i] = sum(lane_densities) / len(lanes) if lanes else 0
        queue[i] = sum(lane_queues)
        waiting_time[i] = sum(lane_waiting_times) / len(lanes) if lanes else 0
    
    total_density = sum(densities) / 4
    total_waiting_time = sum(waiting_time) / 4
    phase = 0 if traci.trafficlight.getPhase(traffic_light_id) == 0 else 1
    
    return np.array([
        densities[0], densities[1], densities[2], densities[3],
        waiting_time[0], waiting_time[1], waiting_time[2], waiting_time[3],
        total_density, total_waiting_time,
        phase
    ]), sum(queue), total_density, total_waiting_time

# Hàm tính phần thưởng
def calculate_reward(old_state, new_state, old_queue, new_queue, green_time, is_north_south):
    delta_total_waiting_time = new_state[9] - old_state[9]
    delta_total_density = new_state[8] - old_state[8]
    total_queue_remaining = new_queue
    
    reward = -k1 * delta_total_waiting_time - k2 * delta_total_density - k3 * total_queue_remaining
    
    # queue = new_state[0] + new_state[1] if is_north_south else new_state[2] + new_state[3]
    # if queue > 15 and green_time < 20:
    #     reward -= 10
    # elif queue < 5 and green_time > 20:
    #     reward -= 5
    
    return reward

# Hàm chạy một pha tín hiệu
def run_phase(phase, green_time):
    traci.trafficlight.setPhase(traffic_light_id, phase)
    traci.trafficlight.setPhaseDuration(traffic_light_id, green_time)
    for _ in range(green_time):
        traci.simulationStep()
    traci.trafficlight.setPhase(traffic_light_id, phase + 1)  # 1: Bắc-Nam vàng, 3: Đông-Tây vàng
    traci.trafficlight.setPhaseDuration(traffic_light_id, yellow_time)
    for _ in range(yellow_time):
        traci.simulationStep()
    traci.trafficlight.setPhase(traffic_light_id, phase + 2)  # 4: Toàn đỏ
    traci.trafficlight.setPhaseDuration(traffic_light_id, red_time)
    for _ in range(red_time):
        traci.simulationStep()

# Chạy mô phỏng
def run_simulation(sumo_config="./DATN/datn.sumocfg"):
    global epsilon
    sumo_cmd = ["sumo-gui", "-c", sumo_config]
    traci.start(sumo_cmd)
    step = 0
    total_reward = 0
    max_steps = 10000
    is_north_south = True

    dqn = QuadDQN()
    try:
        dqn = QuadDQN.load("quad_dqn.h5")
    except:
        pass

    get_vehicle_sizes()  # Khởi tạo kích thước xe

    while step < max_steps:
        state, old_queue, old_total_density, old_total_waiting_time = get_state()
        state = np.reshape(state, [1, state_size])
        
        action = dqn.get_action(state)
        green_time = green_times[action]
        phase = 0 if is_north_south else 3

        run_phase(phase, green_time)
        step += green_time + yellow_time + red_time

        new_state, new_queue, new_total_density, new_total_waiting_time = get_state()
        new_state = np.reshape(new_state, [1, state_size])
        reward = calculate_reward(state[0], new_state[0], old_queue, new_queue, green_time, is_north_south)
        total_reward += reward

        done = step >= max_steps - 1
        memory.append((state, action, reward, new_state, done))

        dqn.train()

        if (step // (green_time + yellow_time + red_time)) % 10 == 0:
            dqn.update_target_model()

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        is_north_south = not is_north_south

        if True:
            print(f"Step: {step}, Phase: {'North-South' if is_north_south else 'East-West'}, "
                  f"Green Time: {green_time}s, Reward: {reward}, Total Reward: {total_reward}, "
                  f"Epsilon: {epsilon}, Queue: {new_queue}, Waiting Time: {new_total_waiting_time*4}")

    dqn.save("quad_dqn.h5")
    traci.close()
    print(f"Simulation ended. Final Total Reward: {total_reward}")

if __name__ == "__main__":
    run_simulation()