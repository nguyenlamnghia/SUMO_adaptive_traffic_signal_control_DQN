# ==========================================================
# Step 1: Import Modules
# ==========================================================
import os
import sys
import numpy as np
from collections import deque
import traci

# ==========================================================
# Step 2: Setup SUMO Path
# ==========================================================
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# ==========================================================
# Step 3: SUMO Configuration
# ==========================================================
Sumo_config = [
    'sumo-gui',
    '-c', './DATN/datn.sumocfg',
    '--step-length', '0.10',
    '--lateral-resolution', '0.1'
]

# ==========================================================
# Step 4: Define Constants & Parameters
# ==========================================================
TLS_ID = "clusterJ12_J2_J3_J6_#2more"
YELLOW_TIME = 3
RED_TIME = 1
STEP_LENGTH = 0.1
SEC_TO_STEP = int(1 / STEP_LENGTH)
TIME_SIMULATION = 7200
STEP_SIMULATION = TIME_SIMULATION * SEC_TO_STEP

detectors = [
    "PVB_J6_J3_0", "PVB_J6_J3_1", "PVB_J6_J3_2",
    "PVB_J0_J3_0", "PVB_J0_J3_1", "PVB_J0_J3_2",
    "HQC_J2_J3_0", "HQC_J4_J3_0"
]

# ==========================================================
# Step 5: Define Functions
# ==========================================================
def get_lane_occupancy(detector_id):
    return traci.lanearea.getLastStepOccupancy(detector_id)

def get_lane_mean_speed(detector_id):
    return traci.lanearea.getLastStepMeanSpeed(detector_id)

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

def get_total_vehicle_in_junction(junction_id):
    lanes = [l for l in traci.lane.getIDList() if l.startswith(f":{junction_id}_")]
    return [veh for lane in lanes for veh in traci.lane.getLastStepVehicleIDs(lane)]

def get_throughput(junction_id, steps):
    count = 0
    for _ in range(steps):
        current_vehicles = set(get_total_vehicle_in_junction(junction_id))
        traci.simulationStep()
        before_vehicles = set(get_total_vehicle_in_junction(junction_id))
        throughput = before_vehicles - current_vehicles
        for veh_id in throughput:
            if traci.vehicle.getVehicleClass(veh_id) == "passenger":
                count += 3
            elif traci.vehicle.getVehicleClass(veh_id) == "bus":
                count += 10
            elif traci.vehicle.getVehicleClass(veh_id) == "truck":
                count += 5
            else:
                count += 1
    return count

def get_current_phase(tls_id):
    return traci.trafficlight.getPhase(tls_id)

# ==========================================================
# Step 6: Main Simulation Loop
# ==========================================================
print("\n=== Starting Fixed Traffic Control Simulation ===")

# Data storage
data = {
    'cycle_count': [],
    'reward': [],
    'cumulative_reward': [],
    'waiting_time': [],
    'waiting_time_per_vehicle': [],
    'queue_length': []
}

traci.start(Sumo_config)

step = 0
cycle_count = 0
cumulative_reward = 0.0
GREEN_TIMES = [15, 25, 35, 45, 55]  # Assuming fixed green times for reward calculation

while step < STEP_SIMULATION:
    before_avg_waiting_time = get_avg_waiting_time()
    before_waiting_time = get_total_waiting_time("J3")
    before_total_vehicle = get_total_vehicle_in_lane()
    before_vehicle_ids = get_total_vehicle_in_junction("J3")
    
    print("_______________________________________")
    print("BEFORE:")
    print(f"Waiting Time: {before_waiting_time:.2f} s")
    
    current_phase = get_current_phase(TLS_ID)
    # Assume green time is managed by sumocfg; use a default for reward calculation
    green_time = traci.trafficlight.getPhaseDuration(TLS_ID) / SEC_TO_STEP
    if green_time <= 0:  # Fallback if green time is not set
        green_time = GREEN_TIMES[0]  # Use minimum green time as default

    # Simulate one cycle (green + yellow + red) as defined in sumocfg
    throughput = get_throughput("J3", int((green_time + YELLOW_TIME + RED_TIME) * SEC_TO_STEP))
    print("THROUGHPUT:", throughput)
    
    # Calculate reward similar to DQN for consistency
    reward = (1 + 0.05 * 0) * (throughput / green_time / 3) if current_phase == 0 else (1 + 0.05 * 0) * (throughput / green_time)
    
    after_avg_waiting_time = get_avg_waiting_time()
    after_waiting_time = get_total_waiting_time("J3")
    after_total_vehicle = get_total_vehicle_in_lane()
    print("AFTER:")
    print(f"Waiting Time: {after_waiting_time:.2f} s")
    
    step += int((green_time + YELLOW_TIME + RED_TIME) * SEC_TO_STEP)
    cycle_count += 1
    cumulative_reward += reward

    # Remove vehicles if junction is congested
    vehicles_in_junction = get_total_vehicle_in_junction("J3")
    if len(vehicles_in_junction) > 30:
        for vehID in vehicles_in_junction:
            try:
                traci.vehicle.remove(vehID)
            except Exception:
                pass
        reward -= 10

    print(f"Step: {step}, Phase: {current_phase}, Cycle: {cycle_count}, Reward: {reward:.2f}, Cumulative: {cumulative_reward:.2f}")

    # Store data
    data['cycle_count'].append(cycle_count)
    data['reward'].append(reward)
    data['cumulative_reward'].append(cumulative_reward)
    data['waiting_time'].append(after_waiting_time)
    data['waiting_time_per_vehicle'].append(after_avg_waiting_time)
    data['queue_length'].append(get_total_vehicle_in_lane())

# Save configuration
print("Simulation completed. Saving metrics...")
with open("fixed_control_config.txt", "w") as f:
    f.write("Fixed Traffic Control Simulation\n")
    f.write(f"Average Queue Length: {np.mean(data['queue_length']):.2f}\n")
    f.write(f"Average Waiting Time: {np.mean(data['waiting_time']):.2f} s\n")
    f.write(f"Average Waiting Time per Vehicle: {np.mean(data['waiting_time_per_vehicle']):.2f} s\n")
    f.write(f"Average Reward: {np.mean(data['reward']):.2f}\n")
    f.write(f"Cumulative Reward: {np.mean(data['cumulative_reward']):.2f}\n")

traci.close()

print("Metrics saved to fixed_control_config.txt")