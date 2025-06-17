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
    'sumo', "--no-warnings",
    '-c', './Sumo/v1/datn.sumocfg',
    '--step-length', '0.10',
    '--lateral-resolution', '0.1'
]

# ==========================================================
# Step 6: Define Constants & Parameters
# ==========================================================
ACTIONS = [0, 1]  # 0: Keep current phase, 1: Switch to next phase
MIN_GREEN_TIME = 15  # Minimum green time before making decisions
DECISION_INTERVAL = 5  # Make decision every 5 seconds

ALPHA = 0.1 # Learning rate
GAMMA = 0.9 # Discount factor
EPSILON = 1.0 # Exploration rate
EPSILON_MIN = 0.1 # Minimum exploration rate
EPSILON_DECAY = 0.9998 # Decay rate for exploration
BATCH_SIZE = 32 # Batch size for training
REPLAY_BUFFER_SIZE = 10000
TARGET_UPDATE_FREQ = 10
RECORD_DATA_FREQ = 1

TLS_ID = "clusterJ12_J2_J3_J6_#2more"
YELLOW_TIME = 3
RED_TIME = 1

EPOCHS = 20  # Tăng số epoch để minh họa
STEP_LENGTH = 0.1
SEC_TO_STEP = int(1 / STEP_LENGTH)
TIME_SIMULATION = 7200
STEP_SIMULATION = TIME_SIMULATION * SEC_TO_STEP

state_size = 18
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

        # Double DQN
        targets = self.model.predict(states, verbose=0)
        # Predict Q-values từ online model để chọn hành động tốt nhất cho next_state
        next_actions = np.argmax(self.model.predict(next_states, verbose=0), axis=1)
        # Predict Q-values từ target model để lấy giá trị Q của action đã chọn ở trên
        target_next = self.target_model.predict(next_states, verbose=0)

        for i in range(BATCH_SIZE):
            targets[i][actions[i]] = (
                rewards[i] if dones[i] else rewards[i] + GAMMA * target_next[i][next_actions[i]]
            )
        q_values = self.model.fit(states, targets, epochs=1, verbose=0)
        return q_values

    def save(self, filename):
        self.model.save(filename)

    @staticmethod
    def load(filename):
        dqn = QuadDQN()
        dqn.model = keras.models.load_model(filename)
        dqn.target_model = keras.models.load_model(filename)
        return dqn

def get_state(time_in_current_phase=0):
    current_phase = get_current_phase(TLS_ID)
    occupancy_values = [get_lane_occupancy(d) for d in detectors]
    mean_speech_values = [get_lane_mean_speech(d) for d in detectors]
    number_of_vehicle_values = [convert_mean_vehicle(get_vehicles_in_lane_area(d)) for d in detectors]
    stop_vehicle_values = [convert_mean_vehicle(get_vehicles_stop_in_lane_area(d)) for d in detectors]
    
    # Add time in current phase to state
    state = np.array(number_of_vehicle_values + mean_speech_values + [time_in_current_phase] + [current_phase])
    print(f"STATE: {state}")
    return state

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

def get_vehicles_stop_in_lane_area(detector_id):
    """
    Get the number of vehicles that are stopped in the lane area.
    A vehicle is considered stopped if its speed is 0.
    """
    vehicle_ids = traci.lanearea.getLastStepVehicleIDs(detector_id)
    stopped_vehicles = [veh_id for veh_id in vehicle_ids if traci.vehicle.getSpeed(veh_id) < 0.1]  # Assuming speed < 0.1 m/s is stopped
    # Convert to mean vehicle count
    return stopped_vehicles

def classify_vehicles(vehicles):
    """
    Classify vehicles into different categories based on their types.
    """
    vehicle_classes = [traci.vehicle.getVehicleClass(veh_id) for veh_id in vehicles]
    return vehicle_classes

def get_total_vehicle_in_junction(junction_id):
    lanes = [l for l in traci.lane.getIDList() if l.startswith(f":{junction_id}_")]
    return [veh for lane in lanes for veh in traci.lane.getLastStepVehicleIDs(lane)]

def run_simulation_for_time(duration_seconds):
    """Run simulation for specified duration and count throughput"""
    count_throughput = 0
    vehicles_through = {
        "motorcycle": 0,
        "passenger": 0,
        "bus": 0,
        "truck": 0
    }
    
    for _ in range(duration_seconds * SEC_TO_STEP):
        current_vehicles_in_junction = set(get_total_vehicle_in_junction("J3"))
        traci.simulationStep()
        after_vehicles_in_junction = set(get_total_vehicle_in_junction("J3"))
        
        # Count classified vehicles that have passed through the junction
        list_vehicles_through = current_vehicles_in_junction - after_vehicles_in_junction
        classified_vehicles = classify_vehicles(list_vehicles_through)
        for vehicle in classified_vehicles:
            if vehicle in vehicles_through:
                vehicles_through[vehicle] += 1
        
        current_phase = get_current_phase(TLS_ID)
        # remove vehicles turn right on phase 3 or 6 from list of vehicles in junction
        if current_phase in [3, 6]:
            current_vehicles_in_junction -= set(traci.lane.getLastStepVehicleIDs(":J3_0_0"))
            current_vehicles_in_junction -= set(traci.lane.getLastStepVehicleIDs(":J3_8_0"))

        count_throughput += convert_mean_vehicle(current_vehicles_in_junction - after_vehicles_in_junction)


    # print(f"Count throughput: {count_throughput}, Vehicles through: {vehicles_through}")
    # print(f"Count throughput: {count_throughput}, Vehicles through: {vehicles_through}")
    if current_phase == 0:
        count_throughput = count_throughput/2.69
    elif current_phase == 3:
        count_throughput = count_throughput/0.71
    elif current_phase == 6:
        count_throughput = count_throughput/0.63
    else:
        count_throughput = 0
    # print(f"Adjusted Count throughput: {count_throughput}")

    # print(f"Adjusted Count throughput: {count_throughput}")
    
    return count_throughput, vehicles_through

def apply_action_keep(duration_seconds):
    """Keep current phase for specified duration"""
    traci.trafficlight.setPhase(TLS_ID, current_phase)
    traci.trafficlight.setPhaseDuration(TLS_ID, duration_seconds * SEC_TO_STEP)
    return run_simulation_for_time(duration_seconds)

def apply_action_switch(current_phase):
    """Switch to next phase"""
    # First run 5 seconds of current phase
    throughput_current, vehicles_current = run_simulation_for_time(5)
    
    # Yellow phase
    traci.trafficlight.setPhase(TLS_ID, current_phase + 1)
    traci.trafficlight.setPhaseDuration(TLS_ID, YELLOW_TIME * SEC_TO_STEP)
    throughput_yellow, vehicles_yellow = run_simulation_for_time(YELLOW_TIME)
    
    # Red phase
    traci.trafficlight.setPhase(TLS_ID, current_phase + 2)
    traci.trafficlight.setPhaseDuration(TLS_ID, RED_TIME * SEC_TO_STEP)
    throughput_red, vehicles_red = run_simulation_for_time(RED_TIME)

    # Next phase (10 seconds of green)
    next_phase = (current_phase + 3) % 9
    traci.trafficlight.setPhase(TLS_ID, next_phase)
    traci.trafficlight.setPhaseDuration(TLS_ID, 10 * SEC_TO_STEP)
    throughput_next, vehicles_next = run_simulation_for_time(10)
    
    # Combine all throughputs
    total_throughput = throughput_current + throughput_yellow + throughput_red + throughput_next

    # Combine all vehicles
    total_vehicles = {}
    for key in vehicles_current.keys():
        total_vehicles[key] = vehicles_current[key] + vehicles_yellow[key] + vehicles_red[key] + vehicles_next[key]
    
    return total_throughput, total_vehicles

def calculate_reward(action, throughput, duration):
    """Calculate reward based on action and throughput"""
    return throughput / duration

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
    'throughput': [],
    'vehicles_through': {
        "motorcycle": 0,
        "passenger": 0,
        "bus": 0,
        "truck": 0
    }
} for i in range(EPOCHS)}

try:
    dqn = QuadDQN.load("quad_dqn_main_2.keras")
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
    count_jam = 0

    vehicles_through = {
        "motorcycle": 0,
        "passenger": 0,
        "bus": 0,
        "truck": 0
    }
    
    traci.start(Sumo_config)
    
    # Initialize phase timing
    time_in_current_phase = 0
    current_reward = 0.0
    new_reward = 0.0
    
    while step < STEP_SIMULATION:
        current_phase = get_current_phase(TLS_ID)
        
        # Only make decisions if we've been in current phase for at least MIN_GREEN_TIME
        if time_in_current_phase >= MIN_GREEN_TIME and time_in_current_phase % DECISION_INTERVAL == 0:
            print("_______________________________________")
            print(f"Decision point: Phase {current_phase}, Time in phase: {time_in_current_phase}s")
            
            state = get_state(time_in_current_phase)
            action = dqn.get_action(state)
            
            print(f"Action chosen: {action} ({'Keep' if action == 0 else 'Switch'})")
            
            if action == 0:  # Keep current phase
                throughput, classified_vehicles = apply_action_keep(DECISION_INTERVAL)
                new_reward = calculate_reward(action, throughput, DECISION_INTERVAL)


                time_in_current_phase += DECISION_INTERVAL
                step += DECISION_INTERVAL * SEC_TO_STEP
                
                print(f"Keeping phase {current_phase} for {DECISION_INTERVAL}s")
                
            else:  # Switch to next phase
                throughput, classified_vehicles = apply_action_switch(current_phase)
                # Total time: 5s current + 3s yellow + 1s red + 10s next = 19s
                new_reward = calculate_reward(action, throughput, 19)

                time_in_current_phase = 10  # Already spent 10s in new phase
                step += 19 * SEC_TO_STEP

                print(f"Switching from phase {current_phase} to next phase for 19s")
            
            # Count vehicles through the junction
            for k, v in classified_vehicles.items():
                vehicles_through[k] += v
            
            new_state = get_state(time_in_current_phase)

            print(f"Current reward: {current_reward:.4f}, New reward: {new_reward:.4f}")

            # update reward 
            reward = new_reward - current_reward
            current_reward = new_reward

            
            print(f"Throughput: {throughput}, Reward: {reward:.4f}")
            
            cumulative_reward += reward
            cycle_count += 1
            
            # Check for traffic jam
            vehicles_in_junction = get_total_vehicle_in_junction("J3")
            if len(vehicles_in_junction) > 40:
                for vehID in vehicles_in_junction:
                    try:
                        traci.vehicle.remove(vehID)
                    except Exception:
                        pass
                count_jam += 1
                reward -= 1
            
            done = step >= STEP_SIMULATION
            replay_buffer.append((state, action, reward, new_state, done))
            
            # Training
            result = dqn.train()
            if result is not None:
                epoch_data[i]['loss'].append(result.history['loss'][0])
                print(f"Loss: {result.history['loss'][0]:.4f}")
            
            if cycle_count % TARGET_UPDATE_FREQ == 0:
                dqn.update_target_model()
            
            # global EPSILON
            EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)
            
            print(f"Epoch: {i + 1}, Step: {step}, Cycle: {cycle_count}, Epsilon: {EPSILON:.2f}, Cumulative: {cumulative_reward:.2f}")
            
            # Record data
            if cycle_count % RECORD_DATA_FREQ == 0:
                epoch_data[i]['cycle_count'].append(cycle_count)
                epoch_data[i]['reward'].append(reward)
                epoch_data[i]['cumulative_reward'].append(cumulative_reward)
                if result is not None:
                    epoch_data[i]['loss'].append(result.history['loss'][0])
                epoch_data[i]['waiting_time'].append(get_total_waiting_time("J3"))
                epoch_data[i]['waiting_time_per_vehicle'].append(get_avg_waiting_time())
                epoch_data[i]['epsilon'].append(EPSILON)
                epoch_data[i]['action'].append(action)
                epoch_data[i]['queue_length'].append(get_total_vehicle_in_lane())
                epoch_data[i]['throughput'].append(throughput)
                epoch_data[i]['vehicles_through']['motorcycle'] = vehicles_through['motorcycle']
                epoch_data[i]['vehicles_through']['passenger'] = vehicles_through['passenger']
                epoch_data[i]['vehicles_through']['bus'] = vehicles_through['bus']
                epoch_data[i]['vehicles_through']['truck'] = vehicles_through['truck']
        
        else:
            # Just continue simulation without making decision
            traci.simulationStep()
            time_in_current_phase += STEP_LENGTH
            time_in_current_phase = round(time_in_current_phase, 2)  # Round to avoid floating point issues
            step += 1
            
            # Reset time counter if phase changed externally
            if get_current_phase(TLS_ID) != current_phase:
                time_in_current_phase = 0

    # Save configuration and model after each epoch
    print(f"Epoch {i + 1} completed. Total steps: {step}, Cumulative reward: {cumulative_reward:.2f}")
    with open(f"epoch_{i + 1}_config_2_1.txt", "w") as f:
        f.write(f"Epoch: {i + 1}\n")
        f.write(f"Epsilon: {EPSILON:.2f}\n")
        f.write(f"Average Loss: {np.mean(epoch_data[i]['loss']) if epoch_data[i]['loss'] else 0:.4f}\n")
        f.write(f"Average Queue Length: {np.mean(epoch_data[i]['queue_length']) if epoch_data[i]['queue_length'] else 0:.2f}\n")
        f.write(f"Average Waiting Time: {np.mean(epoch_data[i]['waiting_time']) if epoch_data[i]['waiting_time'] else 0:.2f} s\n")
        f.write(f"Average Waiting Time per Vehicle: {np.mean(epoch_data[i]['waiting_time_per_vehicle']) if epoch_data[i]['waiting_time_per_vehicle'] else 0:.2f} s\n")
        f.write(f"Average Reward: {np.mean(epoch_data[i]['reward']) if epoch_data[i]['reward'] else 0:.2f}\n")
        f.write(f"Cumulative Reward: {cumulative_reward:.2f}\n")
        f.write(f"Average Throughput: {np.mean(epoch_data[i]['throughput']) if epoch_data[i]['throughput'] else 0:.2f}\n")
        f.write(f"Vehicles Through:\n")
        for vehicle_type, count in epoch_data[i]['vehicles_through'].items():
            f.write(f"  {vehicle_type}: {count}\n")

        # Total action distribution
        if epoch_data[i]['action']:
            action_counts = np.bincount(epoch_data[i]['action'], minlength=len(ACTIONS))
            f.write(f"Action 0 (Keep): {action_counts[0]}\n")
            f.write(f"Action 1 (Switch): {action_counts[1]}\n")

        f.write(f"Count Jam: {count_jam}\n")

    print("Saving model...")
    dqn.save("quad_dqn_main_2_1.keras")
    
    traci.close()

# ==========================================================
# Step 9: Visualization (same as before but adapted for new actions)
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
    avg_queue_length_per_epoch.append(np.mean(data['queue_length']) if data['queue_length'] else 0)
    avg_total_waiting_time_per_epoch.append(np.mean(data['waiting_time']) if data['waiting_time'] else 0)
    avg_waiting_time_per_vehicle_per_epoch.append(np.mean(data['waiting_time_per_vehicle']) if data['waiting_time_per_vehicle'] else 0)
    avg_reward_per_epoch.append(np.mean(data['reward']) if data['reward'] else 0)
    avg_cumulative_reward_per_epoch.append(data['cumulative_reward'][-1] if data['cumulative_reward'] else 0)
    avg_throughput_per_epoch.append(np.mean(data['throughput']) if data['throughput'] else 0)

# Plot results
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.plot(range(1, EPOCHS + 1), avg_loss_per_epoch, marker='o')
plt.title("Average Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)

plt.subplot(2, 3, 2)
plt.plot(range(1, EPOCHS + 1), avg_queue_length_per_epoch, marker='o')
plt.title("Average Queue Length per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Number of Vehicles")
plt.grid(True)

plt.subplot(2, 3, 3)
plt.plot(range(1, EPOCHS + 1), avg_total_waiting_time_per_epoch, marker='o')
plt.title("Average Waiting Time per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Waiting Time (s)")
plt.grid(True)

plt.subplot(2, 3, 4)
plt.plot(range(1, EPOCHS + 1), avg_reward_per_epoch, marker='o')
plt.title("Average Reward per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Reward")
plt.grid(True)

plt.subplot(2, 3, 5)
plt.plot(range(1, EPOCHS + 1), avg_cumulative_reward_per_epoch, marker='o')
plt.title("Cumulative Reward per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Cumulative Reward")
plt.grid(True)

plt.subplot(2, 3, 6)
plt.plot(range(1, EPOCHS + 1), avg_throughput_per_epoch, marker='o')
plt.title("Average Throughput per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Throughput")
plt.grid(True)

plt.tight_layout()
plt.savefig("training_results.png", dpi=300, bbox_inches='tight')
plt.show()

print("\n=== Training Completed ===")
print("Results saved to training_results.png")
print("Model saved as quad_dqn.keras")