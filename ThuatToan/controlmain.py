import traci
import numpy as np
import tensorflow as tf
from collections import deque
import random
import os

# Thiết lập tham số
state_size = 6  # [queue_north, queue_south, queue_east, queue_west, waiting_time, phase]
action_size = 3  # 0: 10s, 1: 20s, 2: 30s (thời gian xanh)
green_times = [10, 20, 30]  # Thời gian xanh
yellow_time = 3  # Thời gian vàng
red_time = 100  # Thời gian đỏ
gamma = 0.95  # Hệ số chiết khấu
epsilon = 1.0  # Tỷ lệ khám phá
epsilon_min = 0.01
epsilon_decay = 0.995
learning_rate = 0.001
batch_size = 32
memory = deque(maxlen=2000)

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
            tf.keras.layers.Dense(action_size + 1, activation='linear')  # +1 cho giá trị trạng thái
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

# Khởi tạo Quad-DQN
dqn = QuadDQN()

# Hàm lấy state từ SUMO
def get_state():
    queue_north = traci.edge.getLastStepVehicleNumber("PVB_J0_J3")
    queue_south = traci.edge.getLastStepVehicleNumber("PVB_J6_J3")
    queue_east = traci.edge.getLastStepVehicleNumber("HQC_J3_J4")
    queue_west = traci.edge.getLastStepVehicleNumber("HQC_J2_J3")
    waiting_time = (traci.edge.getWaitingTime("PVB_J0_J3") +
                    traci.edge.getWaitingTime("PVB_J6_J3") +
                    traci.edge.getWaitingTime("HQC_J3_J4") +
                    traci.edge.getWaitingTime("HQC_J2_J3")) / 4
    phase = 0 if traci.trafficlight.getPhase("clusterJ12_J2_J3_J6_#2more") == 0 else 1
    return np.array([queue_north, queue_south, queue_east, queue_west, waiting_time, phase])

# Hàm tính phần thưởng
def calculate_reward(old_waiting_time, new_waiting_time, queue_length, green_time, is_north_south):
    reward = -(new_waiting_time - old_waiting_time) * 10
    queue = queue_north + queue_south if is_north_south else queue_east + queue_west
    if queue > 15 and green_time < 20:
        reward -= 10
    elif queue < 5 and green_time > 20:
        reward -= 5
    return reward

# Hàm chạy một pha tín hiệu
def run_phase(phase, green_time):
    traci.trafficlight.setPhase("clusterJ12_J2_J3_J6_#2more", phase)
    traci.trafficlight.setPhaseDuration("clusterJ12_J2_J3_J6_#2more", green_time)
    for _ in range(green_time):
        traci.simulationStep()
    traci.trafficlight.setPhase("clusterJ12_J2_J3_J6_#2more", phase + 1)  # 1: Bắc-Nam vàng, 3: Đông-Tây vàng
    traci.trafficlight.setPhaseDuration("clusterJ12_J2_J3_J6_#2more", yellow_time)
    for _ in range(yellow_time):
        traci.simulationStep()
    traci.trafficlight.setPhase("clusterJ12_J2_J3_J6_#2more", phase + 2)  # 4: Toàn đỏ
    traci.trafficlight.setPhaseDuration("clusterJ12_J2_J3_J6_#2more", red_time)
    for _ in range(red_time):
        traci.simulationStep()

# Chạy mô phỏng
def run_simulation():
    global epsilon
    sumo_cmd = ["sumo-gui", "-c", "./DATN/datn.sumocfg"]
    traci.start(sumo_cmd)
    step = 0
    total_reward = 0
    max_steps = 10000  # 1 giờ mô phỏng
    is_north_south = True  # Bắt đầu với Bắc-Nam

    while step < max_steps:
        # Lấy state hiện tại
        global queue_north, queue_south, queue_east, queue_west
        state = get_state()
        state = np.reshape(state, [1, state_size])
        old_waiting_time = state[0][4]
        queue_north, queue_south, queue_east, queue_west = state[0][:4]

        # Chọn hành động (thời gian xanh)
        action = dqn.get_action(state)
        green_time = green_times[action]
        phase = 0 if is_north_south else 3  # Bắc-Nam hoặc Đông-Tây

        # Chạy pha tín hiệu
        run_phase(phase, green_time)
        step += green_time + yellow_time + red_time

        # Lấy state mới và tính phần thưởng
        new_state = get_state()
        new_state = np.reshape(new_state, [1, state_size])
        new_waiting_time = new_state[0][4]
        reward = calculate_reward(old_waiting_time, new_waiting_time,
                                queue_north + queue_south + queue_east + queue_west,
                                green_time, is_north_south)
        total_reward += reward

        # Lưu kinh nghiệm
        done = step >= max_steps - 1
        memory.append((state, action, reward, new_state, done))

        # Huấn luyện Quad-DQN
        dqn.train()

        # Cập nhật target model mỗi 10 pha
        if (step // (green_time + yellow_time + red_time)) % 10 == 0:
            dqn.update_target_model()

        # Giảm epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        # Luân phiên hướng
        is_north_south = not is_north_south

        # if step % 100 == 0:
        if True:
            print(f"Step: {step}, Phase: {'North-South' if is_north_south else 'East-West'}, "
                  f"Green Time: {green_time}s, Reward: {reward}, Total Reward: {total_reward}, Epsilon: {epsilon}"
                  f", Queue: {queue_north + queue_south + queue_east + queue_west}"
                  f", Waiting Time: {new_waiting_time}")

    traci.close()
    print(f"Simulation ended. Final Total Reward: {total_reward}")

if __name__ == "__main__":
    run_simulation()