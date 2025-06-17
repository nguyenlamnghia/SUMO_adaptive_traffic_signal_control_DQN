import os
import sys
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import deque
import matplotlib.pyplot as plt
import traci
import pandas as pd
from scipy.ndimage import uniform_filter1d

# ==========================================================
# Step 1: Configuration and Constants
# ==========================================================
SUMO_CONFIG = [
    "sumo",
    "-c", "./Sumo/v1/datn.sumocfg",
    "--step-length", "0.1",
    "--lateral-resolution", "0.1"
]

# DQN Parameters
ACTIONS = [0, 1, 2, 3, 4]
GREEN_TIMES = [15, 25, 35, 45, 55]
ALPHA = 0.001  # Learning rate
GAMMA = 0.9    # Discount factor
EPSILON = 1.0  # Initial exploration rate
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.999
BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 10000
TARGET_UPDATE_FREQ = 10
RECORD_DATA_FREQ = 1

# Traffic Light Parameters
TLS_ID = "clusterJ12_J2_J3_J6_#2more"
YELLOW_TIME = 3
RED_TIME = 1
STEP_LENGTH = 0.1
SEC_TO_STEP = int(1 / STEP_LENGTH)
TIME_SIMULATION = 1000
STEP_SIMULATION = TIME_SIMULATION * SEC_TO_STEP
STATE_SIZE = 17
ACTION_SIZE = len(ACTIONS)

# Detectors
DETECTORS = [
    "PVB_J6_J3_0", "PVB_J6_J3_1", "PVB_J6_J3_2",
    "PVB_J0_J3_0", "PVB_J0_J3_1", "PVB_J0_J3_2",
    "HQC_J2_J3_0", "HQC_J4_J3_0"
]

VEHICLE_WEIGHTS = {
    "motorcycle": 0.3,
    "passenger": 1,
    "bus": 2,
    "truck": 2
}

# ==========================================================
# Step 2: SUMO Environment Setup
# ==========================================================
def setup_sumo():
    """Setup SUMO environment variable and path."""
    if "SUMO_HOME" not in os.environ:
        sys.exit("Please declare environment variable 'SUMO_HOME'")
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)

# ==========================================================
# Step 3: Traffic Light Environment
# ==========================================================
class TrafficLightEnv:
    """Environment class for traffic light control using SUMO and TraCI."""
    
    def __init__(self, sumo_config, tls_id):
        self.sumo_config = sumo_config
        self.tls_id = tls_id
        self.replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)
        self.step = 0
        self.cycle_count = 0
        self.cumulative_reward = 0.0
        self.count_jam = 0
        self.vehicles_through = {v: 0 for v in VEHICLE_WEIGHTS.keys()}
        self.incoming_vehicles_count = {d: 0 for d in DETECTORS}

    def start(self):
        """Start SUMO simulation."""
        traci.start(self.sumo_config)

    def close(self):
        """Close SUMO simulation."""
        traci.close()

    def get_state(self, action=None):
        """Get current state of the environment."""
        current_phase = traci.trafficlight.getPhase(self.tls_id)
        occupancy = [traci.lanearea.getLastStepOccupancy(d) for d in DETECTORS]
        vehicle_counts = [self._convert_mean_vehicle(traci.lanearea.getLastStepVehicleIDs(d)) for d in DETECTORS]
        flow = ([self.incoming_vehicles_count[d] / (GREEN_TIMES[action] + YELLOW_TIME + RED_TIME)
                for d in DETECTORS] if action is not None else [0] * len(DETECTORS))
        
        state = np.array(vehicle_counts + flow + [current_phase], dtype=np.float32)
        return state

    def apply_action(self, action):
        """Apply action (set traffic light phase) and simulate for green, yellow, red times."""
        current_phase = traci.trafficlight.getPhase(self.tls_id)
        green_time = GREEN_TIMES[action]
        previous_vehicles = {d: set(traci.lanearea.getLastStepVehicleIDs(d)) for d in DETECTORS}
        counted_vehicles = []
        throughput = 0

        # Simulate green phase
        throughput += self._simulate_phase(current_phase, green_time, previous_vehicles, counted_vehicles)
        
        # Simulate yellow phase
        traci.trafficlight.setPhase(self.tls_id, current_phase + 1)
        traci.trafficlight.setPhaseDuration(self.tls_id, YELLOW_TIME * SEC_TO_STEP)
        throughput += self._simulate_phase(current_phase + 1, YELLOW_TIME, previous_vehicles, counted_vehicles)

        # Simulate red phase
        traci.trafficlight.setPhase(self.tls_id, current_phase + 2)
        traci.trafficlight.setPhaseDuration(self.tls_id, RED_TIME * SEC_TO_STEP)
        throughput += self._simulate_phase(current_phase + 2, RED_TIME, previous_vehicles, counted_vehicles)

        # Move to next phase
        next_phase = (current_phase + 3) % 9
        traci.trafficlight.setPhase(self.tls_id, next_phase)
        
        return throughput

    def _simulate_phase(self, phase, duration, previous_vehicles, counted_vehicles):
        """Simulate a single phase (green, yellow, or red) and calculate throughput."""
        throughput = 0
        for _ in range(int(duration * SEC_TO_STEP)):
            current_vehicles_in_junction = set(self._get_total_vehicle_in_junction())
            for detector in DETECTORS:
                current_vehicles = set(traci.lanearea.getLastStepVehicleIDs(detector))
                new_vehicles = current_vehicles - previous_vehicles[detector]
                for veh_id in new_vehicles:
                    if veh_id not in counted_vehicles:
                        counted_vehicles.append(veh_id)
                        self.incoming_vehicles_count[detector] += self._convert_mean_vehicle([veh_id])
                previous_vehicles[detector] = current_vehicles

            traci.simulationStep()
            before_vehicles = current_vehicles_in_junction
            current_vehicles_in_junction = set(self._get_total_vehicle_in_junction())
            
            if phase in [3, 6]:
                before_vehicles -= set(traci.lane.getLastStepVehicleIDs(":J3_0_0"))
                before_vehicles -= set(traci.lane.getLastStepVehicleIDs(":J3_8_0"))

            vehicles_passed = before_vehicles - current_vehicles_in_junction
            for veh_id in vehicles_passed:
                vehicle_class = traci.vehicle.getVehicleClass(veh_id)
                if vehicle_class in self.vehicles_through:
                    self.vehicles_through[vehicle_class] += 1
            throughput += self._convert_mean_vehicle(vehicles_passed)
        
        return throughput

    def get_reward(self, throughput, action, current_phase):
        """Calculate reward based on throughput and phase."""
        green_time = GREEN_TIMES[action]
        if current_phase == 0:
            return throughput / (green_time + YELLOW_TIME + RED_TIME) / 2.69
        elif current_phase == 3:
            return throughput / (green_time + YELLOW_TIME + RED_TIME) / 0.71
        else:
            return throughput / (green_time + YELLOW_TIME + RED_TIME) / 0.63

    def _convert_mean_vehicle(self, vehicles):
        """Convert vehicle list to weighted sum based on vehicle type."""
        return sum(VEHICLE_WEIGHTS.get(traci.vehicle.getVehicleClass(v), 0) for v in vehicles)

    def _get_total_vehicle_in_junction(self):
        """Get all vehicles in the junction."""
        lanes = [l for l in traci.lane.getIDList() if l.startswith(":J3_")]
        return [veh for lane in lanes for veh in traci.lane.getLastStepVehicleIDs(lane)]

    def get_metrics(self):
        """Get environment metrics: waiting time, vehicles, queue length."""
        total_waiting_time = sum(traci.lanearea.getLastStepVehicleNumber(d) for d in DETECTORS)
        total_vehicles = sum(traci.lanearea.getLastStepVehicleNumber(d) for d in DETECTORS)
        avg_waiting_time = total_waiting_time / total_vehicles if total_vehicles > 0 else 0
        queue_length = total_vehicles
        return total_waiting_time, avg_waiting_time, queue_length

    def handle_jam(self):
        """Remove vehicles if junction is jammed."""
        vehicles_in_junction = self._get_total_vehicle_in_junction()
        if len(vehicles_in_junction) > 30:
            self.count_jam += 1
            for veh_id in vehicles_in_junction:
                try:
                    traci.vehicle.remove(veh_id)
                except Exception:
                    pass
            return -1
        return 0

# ==========================================================
# Step 4: DQN Agent
# ==========================================================
class QuadDQN:
    """Deep Q-Network agent for traffic light control."""
    
    def __init__(self):
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        """Build neural network model for DQN."""
        model = keras.Sequential([
            layers.Dense(24, input_dim=STATE_SIZE, activation="relu"),
            layers.Dense(24, activation="relu"),
            layers.Dense(ACTION_SIZE, activation="linear")
        ])
        model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=keras.optimizers.Adam(learning_rate=ALPHA))
        return model

    def update_target_model(self):
        """Update target model weights with main model weights."""
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state, epsilon):
        """Choose action using epsilon-greedy policy."""
        if np.random.rand() <= epsilon:
            return random.randrange(ACTION_SIZE)
        q_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)[0]
        return np.argmax(q_values)

    def train(self, replay_buffer):
        """Train DQN using experience replay with Double DQN."""
        if len(replay_buffer) < BATCH_SIZE:
            return None
        minibatch = random.sample(replay_buffer, BATCH_SIZE)
        states, actions, rewards, next_states, dones = map(np.array, zip(*minibatch))

        targets = self.model.predict(states, verbose=0)
        next_actions = np.argmax(self.model.predict(next_states, verbose=0), axis=1)
        target_next = self.target_model.predict(next_states, verbose=0)

        for i in range(BATCH_SIZE):
            targets[i][actions[i]] = rewards[i] if dones[i] else rewards[i] + GAMMA * target_next[i][next_actions[i]]
        
        return self.model.fit(states, targets, epochs=1, verbose=0)

    def save(self, filename):
        """Save the model to file."""
        self.model.save(filename)

    @staticmethod
    def load(filename):
        """Load a model from file."""
        dqn = QuadDQN()
        dqn.model = keras.models.load_model(filename)
        dqn.target_model = keras.models.load_model(filename)
        return dqn

# ==========================================================
# Step 5: Training Loop
# ==========================================================
def train_dqn(epochs):
    """Main training loop for DQN-based traffic light control."""
    setup_sumo()
    env = TrafficLightEnv(SUMO_CONFIG, TLS_ID)
    
    try:
        dqn = QuadDQN.load("quad_dqn.keras")
        print("Loaded existing DQN model.")
    except Exception as e:
        print(f"Failed to load model: {e}, training new model.")
        dqn = QuadDQN()

    epoch_data = {i: {
        "cycle_count": [], "reward": [], "cumulative_reward": [], "waiting_time": [],
        "waiting_time_per_vehicle": [], "loss": [], "epsilon": [], "action": [],
        "queue_length": [], "throughput": [], "vehicles_through": {v: 0 for v in VEHICLE_WEIGHTS},
        "jam_count": []
    } for i in range(epochs)}

    print("\n=== Starting DQN Training ===")
    epsilon = EPSILON

    for i in range(epochs):
        print(f"\nEpoch {i + 1}/{epochs}")
        env.start()
        env.step = 0
        env.cycle_count = 0
        env.cumulative_reward = 0.0
        env.count_jam = 0
        env.vehicles_through = {v: 0 for v in VEHICLE_WEIGHTS}
        env.incoming_vehicles_count = {d: 0 for d in DETECTORS}

        while env.step < STEP_SIMULATION:
            state = env.get_state()
            action = dqn.get_action(state, epsilon)
            current_phase = traci.trafficlight.getPhase(TLS_ID)

            total_waiting_time, avg_waiting_time, queue_length = env.get_metrics()
            throughput = env.apply_action(action)
            reward = env.get_reward(throughput, action, current_phase) + env.handle_jam()

            new_state = env.get_state(action)
            env.replay_buffer.append((state, action, reward, new_state, env.step >= STEP_SIMULATION))
            result = dqn.train(env.replay_buffer)

            env.cumulative_reward += reward
            env.cycle_count += 1
            env.step += (GREEN_TIMES[action] + YELLOW_TIME + RED_TIME) * SEC_TO_STEP

            if env.cycle_count % TARGET_UPDATE_FREQ == 0:
                dqn.update_target_model()

            if env.cycle_count % RECORD_DATA_FREQ == 0:
                epoch_data[i]["cycle_count"].append(env.cycle_count)
                epoch_data[i]["reward"].append(reward)
                epoch_data[i]["cumulative_reward"].append(env.cumulative_reward)
                epoch_data[i]["loss"].append(result.history["loss"][0] if result else 0)
                epoch_data[i]["waiting_time"].append(total_waiting_time)
                epoch_data[i]["waiting_time_per_vehicle"].append(avg_waiting_time)
                epoch_data[i]["epsilon"].append(epsilon)
                epoch_data[i]["action"].append(action)
                epoch_data[i]["queue_length"].append(queue_length)
                epoch_data[i]["throughput"].append(throughput)
                epoch_data[i]["vehicles_through"] = env.vehicles_through.copy()
                epoch_data[i]["jam_count"].append(env.count_jam)

            # Hiển thị thông tin quan trọng
            if env.cycle_count % RECORD_DATA_FREQ == 0:
                print(f"Epoch: {i + 1}, Step: {env.step}, Cycle: {env.cycle_count}, "
                      f"Action: {action} ({GREEN_TIMES[action]}s), Reward: {reward:.2f}, "
                      f"Cumulative Reward: {env.cumulative_reward:.2f}, Epsilon: {epsilon:.2f}, "
                      f"Queue Length: {queue_length:.2f}, Waiting Time: {total_waiting_time:.2f}, "
                      f"Loss: {result.history['loss'][0]:.4f}" if result else "Loss: N/A")

            epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

        # Save model and metrics
        dqn.save(f"quad_dqn_epoch_{i + 1}.keras")
        save_metrics(i, epoch_data[i])
        env.close()

    save_summary_data(epoch_data, epochs)
    plot_results(epoch_data, epochs)
    return epoch_data

# ==========================================================
# Step 6: Save Metrics
# ==========================================================
def save_metrics(epoch, data):
    """Save training metrics to a file for each epoch."""
    with open(f"epoch_{epoch + 1}_config.txt", "w") as f:
        f.write(f"Epoch: {epoch + 1}\n")
        f.write(f"Epsilon: {data['epsilon'][-1]:.2f}\n")
        f.write(f"Average Loss: {np.mean(data['loss']):.4f}\n")
        f.write(f"Average Queue Length: {np.mean(data['queue_length']):.2f}\n")
        f.write(f"Average Waiting Time: {np.mean(data['waiting_time']):.2f} s\n")
        f.write(f"Average Waiting Time per Vehicle: {np.mean(data['waiting_time_per_vehicle']):.2f} s\n")
        f.write(f"Average Reward: {np.mean(data['reward']):.2f}\n")
        f.write(f"Cumulative Reward: {np.mean(data['cumulative_reward']):.2f}\n")
        f.write(f"Average Throughput: {np.mean(data['throughput']):.2f}\n")
        f.write(f"Jam Count: {data['jam_count'][-1] if data['jam_count'] else 0}\n")
        f.write("Vehicles Through:\n")
        for vehicle_type, count in data["vehicles_through"].items():
            f.write(f"  {vehicle_type}: {count}\n")
        action_counts = np.bincount(data["action"], minlength=len(ACTIONS))
        for j, count in enumerate(action_counts):
            f.write(f"Action {j} ({GREEN_TIMES[j]}s): {count}\n")

# ==========================================================
# Step 7: Save Summary Data
# ==========================================================
def save_summary_data(epoch_data, epochs):
    """Save summary data to CSV for comparison with other algorithms."""
    summary = {
        "Epoch": list(range(1, epochs + 1)),
        "Avg_Loss": [np.mean(epoch_data[i]["loss"]) for i in range(epochs)],
        "Avg_Queue_Length": [np.mean(epoch_data[i]["queue_length"]) for i in range(epochs)],
        "Avg_Waiting_Time": [np.mean(epoch_data[i]["waiting_time"]) for i in range(epochs)],
        "Avg_Waiting_Time_Per_Vehicle": [np.mean(epoch_data[i]["waiting_time_per_vehicle"]) for i in range(epochs)],
        "Avg_Reward": [np.mean(epoch_data[i]["reward"]) for i in range(epochs)],
        "Cumulative_Reward": [np.mean(epoch_data[i]["cumulative_reward"]) for i in range(epochs)],
        "Avg_Throughput": [np.mean(epoch_data[i]["throughput"]) for i in range(epochs)],
        "Jam_Count": [epoch_data[i]["jam_count"][-1] if epoch_data[i]["jam_count"] else 0 for i in range(epochs)],
        "Motorcycle_Through": [epoch_data[i]["vehicles_through"]["motorcycle"] for i in range(epochs)],
        "Passenger_Through": [epoch_data[i]["vehicles_through"]["passenger"] for i in range(epochs)],
        "Bus_Through": [epoch_data[i]["vehicles_through"]["bus"] for i in range(epochs)],
        "Truck_Through": [epoch_data[i]["vehicles_through"]["truck"] for i in range(epochs)]
    }
    df = pd.DataFrame(summary)
    df.to_csv("summary_data.csv", index=False)
    print("Summary data saved to 'summary_data.csv'")

# ==========================================================
# Step 8: Plot Results
# ==========================================================
def plot_results(epoch_data, epochs):
    """Plot training results for analysis."""
    metrics = [
        ("loss", "Average Loss per Epoch", "Loss", "avg_loss_per_epoch.png"),
        ("queue_length", "Average Queue Length per Epoch", "Number of Vehicles", "avg_queue_length_per_epoch.png"),
        ("waiting_time", "Total Waiting Time per Epoch", "Waiting Time (s)", "avg_total_waiting_time_per_epoch.png"),
        ("waiting_time_per_vehicle", "Average Waiting Time per Vehicle per Epoch", "Waiting Time (s)", "avg_waiting_time_per_vehicle_per_epoch.png"),
        ("reward", "Average Reward per Epoch", "Reward", "avg_reward_per_epoch.png"),
        ("cumulative_reward", "Cumulative Reward per Epoch", "Reward", "cumulative_reward_per_epoch.png"),
        ("throughput", "Average Throughput per Epoch", "Throughput", "avg_throughput_per_epoch.png"),
        ("jam_count", "Jam Count per Epoch", "Number of Jams", "jam_count_per_epoch.png")
    ]

    for metric, title, ylabel, filename in metrics:
        plt.figure(figsize=(8, 5))
        values = [np.mean(epoch_data[i][metric]) for i in range(epochs)]
        plt.plot(range(1, epochs + 1), values, marker="o", label=title)
        if metric == "reward":
            moving_avg = uniform_filter1d(values, size=3)
            plt.plot(range(1, epochs + 1), moving_avg, linestyle="--", label="Moving Average (window=3)")
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.legend()
        plt.savefig(filename)
        plt.close()

    # Plot vehicle type throughput
    plt.figure(figsize=(8, 5))
    for vehicle_type in VEHICLE_WEIGHTS.keys():
        values = [epoch_data[i]["vehicles_through"][vehicle_type] for i in range(epochs)]
        plt.plot(range(1, epochs + 1), values, marker="o", label=vehicle_type.capitalize())
    plt.title("Vehicle Type Throughput per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Number of Vehicles")
    plt.grid(True)
    plt.legend()
    plt.savefig("vehicle_type_throughput_per_epoch.png")
    plt.close()

    # Plot action distribution
    for i in range(epochs):
        plt.figure(figsize=(8, 5))
        plt.hist(epoch_data[i]["action"], bins=range(len(ACTIONS) + 1), align="left", rwidth=0.8, density=True)
        plt.xticks(range(len(ACTIONS)), [f"A{i} ({GREEN_TIMES[i]}s)" for i in range(len(ACTIONS))])
        plt.title(f"Action Frequency (Epoch {i + 1})")
        plt.xlabel("Action")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.savefig(f"action_distribution_epoch_{i + 1}.png")
        plt.close()

# ==========================================================
# Step 9: Main Execution
# ==========================================================
if __name__ == "__main__":
    train_dqn(2)