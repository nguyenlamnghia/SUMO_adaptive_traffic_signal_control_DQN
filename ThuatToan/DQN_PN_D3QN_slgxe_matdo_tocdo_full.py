

import os
import sys
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, Lambda, Add, Subtract, Layer
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import traci
from collections import deque


class NoisyDense(Layer):
    def __init__(self, units, sigma_init=0.5):
        super().__init__()
        self.units = units
        self.sigma_init = sigma_init

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.mu_w = self.add_weight(name="mu_w", shape=[input_dim, self.units],
                                    initializer=keras.initializers.RandomUniform(-1/np.sqrt(input_dim), 1/np.sqrt(input_dim)))
        self.sigma_w = self.add_weight(name="sigma_w", shape=[input_dim, self.units],
                                    initializer=keras.initializers.Constant(self.sigma_init / np.sqrt(input_dim)))
        self.mu_b = self.add_weight(name="mu_b", shape=[self.units], initializer='zeros')
        self.sigma_b = self.add_weight(name="sigma_b", shape=[self.units],
                                    initializer=keras.initializers.Constant(self.sigma_init / np.sqrt(input_dim)))


    def call(self, inputs):
        epsilon_in = tf.random.normal([inputs.shape[-1], 1])
        epsilon_out = tf.random.normal([1, self.units])
        epsilon_w = epsilon_in * epsilon_out
        epsilon_b = tf.squeeze(epsilon_out)
        noisy_w = self.mu_w + self.sigma_w * epsilon_w
        noisy_b = self.mu_b + self.sigma_b * epsilon_b
        return tf.matmul(inputs, noisy_w) + noisy_b

class PN_D3QN:
    def __init__(self):
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        inputs = keras.Input(shape=(state_size,))
        x = NoisyDense(128)(inputs)
        from tensorflow.keras.layers import Activation
        x = Activation('relu')(x)

        x = NoisyDense(128)(x)
        from tensorflow.keras.layers import Activation
        x = Activation('relu')(x)

        value = NoisyDense(1)(x)
        advantage = NoisyDense(action_size)(x)
        mean_advantage = Lambda(lambda a: tf.reduce_mean(a, axis=1, keepdims=True))(advantage)
        q_values = Add()([value, Subtract()([advantage, mean_advantage])])
        model = keras.Model(inputs=inputs, outputs=q_values)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=ALPHA), loss='mse')
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

        target_q = self.target_model.predict(next_states, verbose=0)
        next_actions = np.argmax(self.model.predict(next_states, verbose=0), axis=1)
        targets = self.model.predict(states, verbose=0)

        for i in range(BATCH_SIZE):
            targets[i][actions[i]] = rewards[i] if dones[i] else rewards[i] + GAMMA * target_q[i][next_actions[i]]

        return self.model.fit(states, targets, epochs=1, verbose=0)

    def save(self, filename):
        self.model.save(filename)

    @staticmethod
    def load(filename):
        dqn = PN_D3QN()
        dqn.model = keras.models.load_model(filename, custom_objects={"NoisyDense": NoisyDense})
        dqn.target_model = keras.models.load_model(filename, custom_objects={"NoisyDense": NoisyDense})
        return dqn


# ==========================================================
# Step 1: Import Modules
# ==========================================================



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
EPSILON_MIN = 0.1 # Minimum exploration rate
EPSILON_DECAY = 0.998 # Decay rate for exploration
BATCH_SIZE = 32 # Batch size for training
# BATCH_SIZE = 8  # Giảm kích thước batch để tránh lỗi OOM trên GPU
REPLAY_BUFFER_SIZE = 10000
TARGET_UPDATE_FREQ = 10
RECORD_DATA_FREQ = 1

TLS_ID = "clusterJ12_J2_J3_J6_#2more"
YELLOW_TIME = 3
RED_TIME = 1

EPOCHS = 10  # Tăng số epoch để minh họa
STEP_LENGTH = 0.1
SEC_TO_STEP = int(1 / STEP_LENGTH)
TIME_SIMULATION = 7200
STEP_SIMULATION = TIME_SIMULATION * SEC_TO_STEP

state_size = 16
action_size = len(ACTIONS)
replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

detectors = [
    "PVB_J6_J3_0", "PVB_J6_J3_1", "PVB_J6_J3_2",
    "PVB_J0_J3_0", "PVB_J0_J3_1", "PVB_J0_J3_2",
    "HQC_J2_J3_0", "HQC_J4_J3_0"
]

table_convert_mean_vehicle = {
    "motorcycle": 0.3,
    "passenger": 1,
    "bus": 2,
    "truck": 2
}

# ==========================================================
# Step 7: Define Functions
# ==========================================================


def get_state(incoming_vehicles_count, action=None):
    current_phase = get_current_phase(TLS_ID)
    occupancy_values = [get_lane_occupancy(d) for d in detectors]
    mean_speech_values = [get_lane_mean_speech(d) for d in detectors]
    number_of_vehicle_values = [convert_mean_vehicle(get_vehicles_in_lane_area(d)) for d in detectors]
    if action is not None:
        incoming_vehicle_values = [incoming_vehicles_count[d]/(GREEN_TIMES[action]+YELLOW_TIME+RED_TIME) for d in detectors]
        print(f"Phase time: {(GREEN_TIMES[action]+YELLOW_TIME+RED_TIME)} seconds")
    else:
        # set array all incoming vehicle values to 0 if action is None
        incoming_vehicle_values = [0 for d in detectors]
    print(f"OCUPANCY: {occupancy_values}")
    print(f"MEAN SPEECH: {mean_speech_values}")
    print(f"NUMBER OF VEHICLES: {number_of_vehicle_values}")
    print(f"INCOMING VEHICLES: {incoming_vehicle_values}")
    print(f"CURRENT PHASE: {current_phase}")
    state = np.array(convert_status_lane_to_array(occupancy_values) + convert_status_lane_to_array(number_of_vehicle_values) + convert_status_lane_to_array(incoming_vehicle_values) + [current_phase])
    print(f"STATE: {state}")
    return state

def get_reward(before_vehicle_ids, after_vehicle_ids):
    return 10

def get_lane_mean_speech(detector_id):
    return traci.lanearea.getLastStepMeanSpeed(detector_id)

def get_total_waiting_time(_):
    return sum(traci.lane.getWaitingTime(d) for d in detectors)

def get_vehicles_in_lane_area(detector_id):
    return traci.lanearea.getLastStepVehicleIDs(detector_id)

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

def convert_mean_vehicle(vehicles):
    vehicle_classes = [traci.vehicle.getVehicleClass(veh_id) for veh_id in vehicles]
    if vehicle_classes == []:
        return 0
    return sum(table_convert_mean_vehicle[veh_class] for veh_class in vehicle_classes)

def convert_status_lane_to_array(status_lane):
    # result_array = [status_lane[0]+status_lane[1]+(status_lane[2]+status_lane[5])/2, (status_lane[2]+status_lane[5])/2, status_lane[3]+status_lane[4], status_lane[6]+status_lane[7]]
    result_array = [status_lane[0]+status_lane[1], status_lane[3]+status_lane[4], status_lane[2]+status_lane[5], status_lane[6], status_lane[7]]
    return result_array

def apply_action(current_phase, green_time):
    # count = 0

    count_throughput = 0
    counted_vehicles = []

    previous_vehicles = {detector: set(traci.lanearea.getLastStepVehicleIDs(detector)) for detector in detectors}
    incoming_vehicles_count = {detector: 0 for detector in detectors}

    traci.trafficlight.setPhase(TLS_ID, current_phase)
    traci.trafficlight.setPhaseDuration(TLS_ID, green_time * SEC_TO_STEP)
    for _ in range(green_time * SEC_TO_STEP):
        current_vehicles_in_junction = set(get_total_vehicle_in_junction("J3"))

        # Đếm phương tiện mới đến cho mỗi detector
        for detector in detectors:
            current_vehicles = set(traci.lanearea.getLastStepVehicleIDs(detector))
            new_vehicles = current_vehicles - previous_vehicles[detector]
            # incoming_vehicles_count[detector] += convert_mean_vehicle(new_vehicles)

            # Kiểm tra xem đã đếm chưa chưa đếm thì đếm
            for veh_id in new_vehicles:
                if veh_id not in counted_vehicles:
                    counted_vehicles.append(veh_id)
                    incoming_vehicles_count[detector] += convert_mean_vehicle(set(new_vehicles))

            previous_vehicles[detector] = current_vehicles

        traci.simulationStep()

        before_vehicles_in_junction = set(get_total_vehicle_in_junction("J3"))
        count_throughput += convert_mean_vehicle(before_vehicles_in_junction - current_vehicles_in_junction)

    # count += get_throughput("J3", green_time * SEC_TO_STEP)

    traci.trafficlight.setPhase(TLS_ID, current_phase + 1)
    traci.trafficlight.setPhaseDuration(TLS_ID, YELLOW_TIME * SEC_TO_STEP)
    for _ in range(YELLOW_TIME * SEC_TO_STEP):
        current_vehicles_in_junction = set(get_total_vehicle_in_junction("J3"))

        # Đếm phương tiện mới đến cho mỗi detector
        for detector in detectors:
            current_vehicles = set(traci.lanearea.getLastStepVehicleIDs(detector))
            new_vehicles = current_vehicles - previous_vehicles[detector]

            # Kiểm tra xem đã đếm chưa chưa đếm thì đếm
            for veh_id in new_vehicles:
                if veh_id not in counted_vehicles:
                    counted_vehicles.append(veh_id)
                    incoming_vehicles_count[detector] += convert_mean_vehicle(set(new_vehicles))

            previous_vehicles[detector] = current_vehicles
        
        traci.simulationStep()

        before_vehicles_in_junction = set(get_total_vehicle_in_junction("J3"))
        count_throughput += convert_mean_vehicle(before_vehicles_in_junction - current_vehicles_in_junction)

    # count += get_throughput("J3", YELLOW_TIME * SEC_TO_STEP)

    traci.trafficlight.setPhase(TLS_ID, current_phase + 2)
    traci.trafficlight.setPhaseDuration(TLS_ID, RED_TIME * SEC_TO_STEP)
    for _ in range(RED_TIME * SEC_TO_STEP):
        current_vehicles_in_junction = set(get_total_vehicle_in_junction("J3"))

        # Đếm phương tiện mới đến cho mỗi detector
        for detector in detectors:
            current_vehicles = set(traci.lanearea.getLastStepVehicleIDs(detector))
            new_vehicles = current_vehicles - previous_vehicles[detector]
            # incoming_vehicles_count[detector] += convert_mean_vehicle(new_vehicles)

            # Kiểm tra xem đã đếm chưa chưa đếm thì đếm
            for veh_id in new_vehicles:
                if veh_id not in counted_vehicles:
                    counted_vehicles.append(veh_id)
                    incoming_vehicles_count[detector] += convert_mean_vehicle(set(new_vehicles))

            previous_vehicles[detector] = current_vehicles

        traci.simulationStep()

        before_vehicles_in_junction = set(get_total_vehicle_in_junction("J3"))
        count_throughput += convert_mean_vehicle(before_vehicles_in_junction - current_vehicles_in_junction)

    # count += get_throughput("J3", RED_TIME * SEC_TO_STEP)
    # print(f"Throughput during phase {current_phase}: {count} vehicles")

    next_phase = (current_phase + 3) % 9
    traci.trafficlight.setPhase(TLS_ID, next_phase)
    return count_throughput, incoming_vehicles_count

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
    'cumulative_reward': [],
    'waiting_time': [],
    'waiting_time_per_vehicle': [],
    'loss': [],
    'epsilon': [],
    'action': [],
    'queue_length': [],
    'throughput': []
} for i in range(EPOCHS)}

try:
    dqn = PN_D3QN.load("quad_dqn.keras")
    print("Loaded existing DQN model.")
except Exception as e:
    print(f"Failed to load model: {e}, training new model.")
    dqn = PN_D3QN()

print("\n=== Starting Fully Online Continuous Learning (DQN) ===")

for i in range(EPOCHS):
    print(f"\nEpoch {i + 1}/{EPOCHS}")
    step = 0
    cycle_count = 0
    cumulative_reward = 0.0

    incoming_vehicles_count = {'PVB_J6_J3_0': 0, 'PVB_J6_J3_1': 0, 'PVB_J6_J3_2': 0, 'PVB_J0_J3_0': 0, 'PVB_J0_J3_1': 0, 'PVB_J0_J3_2': 0, 'HQC_J2_J3_0': 0, 'HQC_J4_J3_0': 0}
    
    traci.start(Sumo_config)
    action = None

    while step < STEP_SIMULATION:
        before_avg_waiting_time = get_avg_waiting_time()
        before_waiting_time = get_total_waiting_time("J3")
        before_total_vehicle = get_total_vehicle_in_lane()
        before_vehicle_ids = get_total_vehicle_in_junction("J3")

        before_total_vehicle_in_junction = convert_mean_vehicle(before_vehicle_ids)

        print("_______________________________________")
        print("BEFORE:")
        print(f"Waiting Time: {before_waiting_time:.2f} s")
        state = get_state(incoming_vehicles_count,action)
        action = dqn.get_action(state)

        current_phase = get_current_phase(TLS_ID)
        green_time = GREEN_TIMES[action]

        throughput, incoming_vehicles_count = apply_action(current_phase, green_time)

        print("THROUGHPUT:", throughput)
        print("INCOMING:", incoming_vehicles_count)
        
        # if current_phase == 0:
        #     reward = 
        # elif current_phase == 3:
        #     reward = throughput/(green_time + YELLOW_TIME + RED_TIME)
        # else:
        #     reward = throughput/(green_time + YELLOW_TIME + RED_TIME)/0.5

        after_avg_waiting_time = get_avg_waiting_time()
        after_waiting_time = get_total_waiting_time("J3")
        after_total_vehicle = get_total_vehicle_in_lane()
        after_vehicle_ids = get_total_vehicle_in_junction("J3")
        after_total_vehicle_in_junction = convert_mean_vehicle(after_vehicle_ids)

        reward = -(after_total_vehicle_in_junction - before_total_vehicle_in_junction)

        print("AFTER:")
        print(f"Waiting Time: {after_waiting_time:.2f} s")
        new_state = get_state(incoming_vehicles_count,action)
        # reward = get_reward(before_vehicle_ids, after_vehicle_ids)

        # reward = (throughput - sum([incoming_vehicles_count[d] for d in detectors]))

        step += (green_time + YELLOW_TIME + RED_TIME) * SEC_TO_STEP
        cycle_count += 1  # Tăng số chu kỳ

        vehicles_in_junction = get_total_vehicle_in_junction("J3")
        if len(vehicles_in_junction) > 30:
            for vehID in vehicles_in_junction:
                try:
                    traci.vehicle.remove(vehID)
                except Exception:
                    pass
            reward -= 1

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

        print(f"Epoch: {i + 1}, Step: {step}, Phase: {current_phase}, Cycle: {cycle_count}, Action: {action}, Reward: {reward:.2f}, Epsilon: {EPSILON:.2f}, Cumulative: {cumulative_reward:.2f}")

        if cycle_count % RECORD_DATA_FREQ == 0:
            epoch_data[i]['cycle_count'].append(cycle_count)
            epoch_data[i]['reward'].append(reward)
            epoch_data[i]['cumulative_reward'].append(cumulative_reward)
            # epoch_data[i]['loss'].append(result.history['loss'][0] if result is not None else 0)
            if result is not None:
                epoch_data[i]['loss'].append(result.history['loss'][0])
            epoch_data[i]['waiting_time'].append(after_waiting_time)
            epoch_data[i]['waiting_time_per_vehicle'].append(after_avg_waiting_time)
            epoch_data[i]['epsilon'].append(EPSILON)
            epoch_data[i]['action'].append(action)
            epoch_data[i]['queue_length'].append(get_total_vehicle_in_lane())
            epoch_data[i]['throughput'].append(throughput)

    # Save configuration and model after each epoch for training continuity and analysis
    print(f"Epoch {i + 1} completed. Total steps: {step}, Cumulative reward: {cumulative_reward:.2f}")
    with open(f"epoch_{i + 1}_config.txt", "w") as f:
        f.write(f"Epoch: {i + 1}\n")
        f.write(f"Epsilon: {EPSILON:.2f}\n")
        f.write(f"Average Loss: {np.mean(epoch_data[i]['loss']):.4f}\n")
        f.write(f"Average Queue Length: {np.mean(epoch_data[i]['queue_length']):.2f}\n")
        f.write(f"Average Waiting Time: {np.mean(epoch_data[i]['waiting_time']):.2f} s\n")
        f.write(f"Average Waiting Time per Vehicle: {np.mean(epoch_data[i]['waiting_time_per_vehicle']):.2f} s\n")
        f.write(f"Average Reward: {np.mean(epoch_data[i]['reward']):.2f}\n")
        f.write(f"Cumulative Reward: {np.mean(epoch_data[i]['cumulative_reward']):.2f}\n")
        f.write(f"Average Throughput: {np.mean(epoch_data[i]['throughput']):.2f}\n")

        # Total action distribution
        action_counts = np.bincount(epoch_data[i]['action'], minlength=len(ACTIONS))
        for j, count in enumerate(action_counts):
            f.write(f"Action {j} ({GREEN_TIMES[j]}s): {count}\n")

    print("Saving model...")
    # Save the model after each epoch
    dqn.save("quad_dqn.keras")
    
    
    traci.close()

# dqn.save("quad_dqn.keras")

# ==========================================================
# Step 9: Close SUMO and Visualize
# ==========================================================

# Compute per-epoch averages
avg_loss_per_epoch = []
avg_queue_length_per_epoch = []
avg_total_waiting_time_per_epoch = []
avg_waiting_time_per_vehicle_per_epoch = []
avg_reward_per_epoch = []
avg_cumulative_reward_per_epoch = []
avg_throughput_per_epoch = []

for i in range(EPOCHS):
    data = epoch_data[i]
    avg_loss_per_epoch.append(np.mean(data['loss']) if data['loss'] else 0)
    avg_queue_length_per_epoch.append(np.mean(data['queue_length']))
    avg_total_waiting_time_per_epoch.append(np.mean(data['waiting_time']))
    avg_waiting_time_per_vehicle_per_epoch.append(np.mean(data['waiting_time_per_vehicle']))
    avg_reward_per_epoch.append(np.mean(data['reward']))
    avg_cumulative_reward_per_epoch.append(np.mean(data['cumulative_reward']))
    avg_throughput_per_epoch.append(np.mean(data['throughput']))

# Plot Average Loss per Epoch
plt.figure(figsize=(8, 5))
plt.plot(range(1, EPOCHS + 1), avg_loss_per_epoch, marker='o', label="Average Loss")
plt.title("Average Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.savefig("avg_loss_per_epoch.png")
# plt.show()

# Plot Average Queue Length per Epoch
plt.figure(figsize=(8, 5))
plt.plot(range(1, EPOCHS + 1), avg_queue_length_per_epoch, marker='o', label="Average Queue Length")
plt.title("Average Queue Length per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Number of Vehicles")
plt.grid(True)
plt.legend()
plt.savefig("avg_queue_length_per_epoch.png")
# plt.show()

# Plot Total Waiting Time Change per Epoch
plt.figure(figsize=(8, 5))
plt.plot(range(1, EPOCHS + 1), avg_total_waiting_time_per_epoch, marker='o', label="Total Waiting Time Change")
plt.title("Total Waiting Time Change per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Δ Total Waiting Time (s)")
plt.grid(True)
plt.legend()
plt.savefig("avg_total_waiting_time_change_per_epoch.png")
# plt.show()

# Plot Avg Waiting Time per Vehicle per Epoch
plt.figure(figsize=(8, 5))
plt.plot(range(1, EPOCHS + 1), avg_waiting_time_per_vehicle_per_epoch, marker='o', label="Avg Waiting Time per Vehicle")
plt.title("Average Waiting Time per Vehicle per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Avg Waiting Time (s)")
plt.grid(True)
plt.legend()
plt.savefig("avg_waiting_time_per_vehicle_per_epoch.png")
# plt.show()

# Plot Average Reward per Epoch
plt.figure(figsize=(8, 5))
plt.plot(range(1, EPOCHS + 1), avg_reward_per_epoch, marker='o', label="Average Reward per Epoch")
plt.title("Average Reward per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Reward")
plt.grid(True)
plt.legend()
plt.savefig("avg_reward_per_epoch.png")
# plt.show()

# Plot Cumulative Reward per Epoch
plt.figure(figsize=(8, 5))
plt.plot(range(1, EPOCHS + 1), avg_cumulative_reward_per_epoch, marker='o', label="Cumulative Reward per Epoch")
plt.title("Cumulative Reward per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Reward")
plt.grid(True)
plt.legend()
plt.savefig("cumulative_reward_per_epoch.png")
# plt.show()

# Plot Throughput per Epoch
plt.figure(figsize=(8, 5))
plt.plot(range(1, EPOCHS + 1), avg_throughput_per_epoch, marker='o', label="Average Throughput per Epoch")
plt.title("Average Throughput per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Throughput")
plt.grid(True)
plt.legend()
plt.savefig("avg_throughput_per_epoch.png")
# plt.show()

# Plot Action Distribution per Epoch
def plot_action_distribution(actions, title, filename):
    plt.figure(figsize=(8, 5))
    plt.hist(actions, bins=range(len(ACTIONS) + 1), align='left', rwidth=0.8, density=True)
    plt.xticks(range(len(ACTIONS)), [f"A{i} ({GREEN_TIMES[i]}s)" for i in range(len(ACTIONS))])
    plt.title(title)
    plt.xlabel("Action")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(filename)
    # plt.show()

for i in range(EPOCHS):
    plot_action_distribution(
        epoch_data[i]['action'],
        f"Action Frequency (Epoch {i+1})",
        f"action_distribution_epoch_{i+1}.png"
    )