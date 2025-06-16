# ==========================================================
# Fixed Traffic Light Control Evaluation
# ==========================================================
import os
import sys
import numpy as np
from collections import deque

# ==========================================================
# Setup SUMO Path
# ==========================================================
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import traci

# ==========================================================
# Configuration
# ==========================================================
Sumo_config = [
    'sumo',
    '-c', './Sumo/datn.sumocfg',
    '--step-length', '0.10',
    '--lateral-resolution', '0.1'
]

# Fixed timing parameters
FIXED_GREEN_TIMES = {
    0: 50,  # Phase 0: 50s
    3: 30,  # Phase 3: 30s
    6: 20   # Phase 6: 20s
}

TLS_ID = "clusterJ12_J2_J3_J6_#2more"
YELLOW_TIME = 3
RED_TIME = 1

EPOCHS = 1
STEP_LENGTH = 0.1
SEC_TO_STEP = int(1 / STEP_LENGTH)
TIME_SIMULATION = 7200
STEP_SIMULATION = TIME_SIMULATION * SEC_TO_STEP
RECORD_DATA_FREQ = 1

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
# Helper Functions
# ==========================================================
def get_state(incoming_vehicles_count, phase):
    current_phase = get_current_phase(TLS_ID)
    occupancy_values = [get_lane_occupancy(d) for d in detectors]
    mean_speech_values = [get_lane_mean_speech(d) for d in detectors]
    number_of_vehicle_values = [convert_mean_vehicle(get_vehicles_in_lane_area(d)) for d in detectors]
    stop_vehicle_values = [convert_mean_vehicle(get_vehicles_stop_in_lane_area(d)) for d in detectors]
    
    if phase in FIXED_GREEN_TIMES:
        green_time = FIXED_GREEN_TIMES[phase]
        average_flow = [incoming_vehicles_count[d]/(green_time+YELLOW_TIME+RED_TIME) for d in detectors]
        print(f"Phase time: {(green_time+YELLOW_TIME+RED_TIME)} seconds")
    else:
        average_flow = [0 for d in detectors]
    
    state = np.array(stop_vehicle_values + average_flow + [current_phase])
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
    vehicle_ids = traci.lanearea.getLastStepVehicleIDs(detector_id)
    stopped_vehicles = [veh_id for veh_id in vehicle_ids if traci.vehicle.getSpeed(veh_id) < 0.1]
    return stopped_vehicles

def apply_fixed_action(current_phase, green_time):
    count_throughput = 0
    counted_vehicles = []

    previous_vehicles = {detector: set(traci.lanearea.getLastStepVehicleIDs(detector)) for detector in detectors}
    incoming_vehicles_count = {detector: 0 for detector in detectors}
    vehicles_through = {"motorcycle": 0, "passenger": 0, "bus": 0, "truck": 0}

    def process_phase(duration):
        nonlocal count_throughput
        for _ in range(duration * SEC_TO_STEP):
            current_vehicles_in_junction = set(get_total_vehicle_in_junction("J3"))

            for detector in detectors:
                current_vehicles = set(traci.lanearea.getLastStepVehicleIDs(detector))
                new_vehicles = current_vehicles - previous_vehicles[detector]

                for veh_id in new_vehicles:
                    if veh_id not in counted_vehicles:
                        counted_vehicles.append(veh_id)
                        incoming_vehicles_count[detector] += convert_mean_vehicle(set(new_vehicles))

                previous_vehicles[detector] = current_vehicles

            traci.simulationStep()

            before_vehicles_in_junction = set(get_total_vehicle_in_junction("J3"))
            new_passed_vehicles = before_vehicles_in_junction - current_vehicles_in_junction

            # Bỏ xe rẽ phải khỏi thống kê nếu là pha 3 hoặc 6
            if current_phase in [3, 6]:
                new_passed_vehicles -= set(traci.lane.getLastStepVehicleIDs(":J3_0_0"))
                new_passed_vehicles -= set(traci.lane.getLastStepVehicleIDs(":J3_8_0"))

            classified = classify_vehicles(new_passed_vehicles)
            for vtype in classified:
                if vtype in vehicles_through:
                    vehicles_through[vtype] += 1

            count_throughput += convert_mean_vehicle(new_passed_vehicles)

    # Green
    traci.trafficlight.setPhase(TLS_ID, current_phase)
    process_phase(green_time)

    # Yellow
    traci.trafficlight.setPhase(TLS_ID, current_phase + 1)
    process_phase(YELLOW_TIME)

    # Red
    traci.trafficlight.setPhase(TLS_ID, current_phase + 2)
    process_phase(RED_TIME)

    next_phase = (current_phase + 3) % 9
    traci.trafficlight.setPhase(TLS_ID, next_phase)

    return count_throughput, incoming_vehicles_count, vehicles_through


def get_total_vehicle_in_junction(junction_id):
    lanes = [l for l in traci.lane.getIDList() if l.startswith(f":{junction_id}_")]
    return [veh for lane in lanes for veh in traci.lane.getLastStepVehicleIDs(lane)]

def classify_vehicles(vehicles):
    vehicle_classes = [traci.vehicle.getVehicleClass(veh_id) for veh_id in vehicles]
    return vehicle_classes


# ==========================================================
# Main Fixed Control Loop
# ==========================================================
# Data storage for each epoch
epoch_data = {i: {
    'cycle_count': [],
    'reward': [],
    'cumulative_reward': [],
    'waiting_time': [],
    'waiting_time_per_vehicle': [],
    'phase': [],
    'queue_length': [],
    'throughput': [],
    'vehicles_through': {
        'motorcycle': 0,
        'passenger': 0,
        'bus': 0,
        'truck': 0
}

} for i in range(EPOCHS)}

print("\n=== Starting Fixed Traffic Light Control Evaluation ===")

for i in range(EPOCHS):
    print(f"\nEpoch {i + 1}/{EPOCHS}")
    step = 0
    cycle_count = 0
    cumulative_reward = 0.0
    
    incoming_vehicles_count = {detector: 0 for detector in detectors}
    
    traci.start(Sumo_config)
    
    # Fixed phase sequence: 0 -> 3 -> 6 -> 0 -> ...
    phase_sequence = [0, 3, 6]
    phase_index = 0
    
    while step < STEP_SIMULATION:
        current_phase = phase_sequence[phase_index]
        green_time = FIXED_GREEN_TIMES[current_phase]
        
        before_avg_waiting_time = get_avg_waiting_time()
        before_waiting_time = get_total_waiting_time("J3")
        before_total_vehicle = get_total_vehicle_in_lane()
        before_vehicle_ids = get_total_vehicle_in_junction("J3")
        before_total_vehicle_in_junction = convert_mean_vehicle(before_vehicle_ids)
        
        print("_______________________________________")
        print("BEFORE:")
        state = get_state(incoming_vehicles_count, current_phase)
        
        # Set traffic light to current phase
        traci.trafficlight.setPhase(TLS_ID, current_phase)
        
        throughput, incoming_vehicles_count, classified_vehicles = apply_fixed_action(current_phase, green_time)

        # Cộng dồn
    for vtype in epoch_data[i]['vehicles_through']:
        epoch_data[i]['vehicles_through'][vtype] += classified_vehicles[vtype]

        print("INCOMING:", incoming_vehicles_count)
        
        # Calculate reward similar to DQN version
        if current_phase == 0:
            reward1 = throughput/(green_time + YELLOW_TIME + RED_TIME)/2.5
        elif current_phase == 3:
            reward1 = throughput/(green_time + YELLOW_TIME + RED_TIME) * 1.5
        else:
            reward1 = throughput/(green_time + YELLOW_TIME + RED_TIME)/0.5
        
        after_avg_waiting_time = get_avg_waiting_time()
        after_waiting_time = get_total_waiting_time("J3")
        after_total_vehicle = get_total_vehicle_in_lane()
        after_vehicle_ids = get_total_vehicle_in_junction("J3")
        after_total_vehicle_in_junction = convert_mean_vehicle(after_vehicle_ids)
        
        print("AFTER:")
        new_state = get_state(incoming_vehicles_count, current_phase)
        
        old_occupancy_mean = np.mean(state[:8])/8.1
        old_occupancy_max = np.max(state[:8])/8.1
        new_occupancy_mean = np.mean(new_state[:8])/8.1
        new_occupancy_max = np.max(new_state[:8])/8.1
        
        print("Before waiting time:", before_waiting_time)
        print("After waiting time:", after_waiting_time)
        print(f"Old Occupancy Mean: {old_occupancy_mean:.2f}, New Occupancy Mean: {new_occupancy_mean:.2f}")
        print(f"Old Occupancy Max: {old_occupancy_max:.2f}, New Occupancy Max: {new_occupancy_max:.2f}")
        
        reward2 = before_waiting_time - after_waiting_time
        reward = reward1
        
        print("THROUGHPUT:", throughput/(green_time + YELLOW_TIME + RED_TIME))
        print(f"Reward2: {reward2:.2f}, Reward1: {reward1:.2f}, Final Reward: {reward:.2f}")
        
        step += (green_time + YELLOW_TIME + RED_TIME) * SEC_TO_STEP
        cycle_count += 1
        
        # Remove vehicles if junction is overcrowded
        vehicles_in_junction = get_total_vehicle_in_junction("J3")
        if len(vehicles_in_junction) > 30:
            for vehID in vehicles_in_junction:
                try:
                    traci.vehicle.remove(vehID)
                except Exception:
                    pass
            reward -= 1
        
        cumulative_reward += reward
        
        print(f"Epoch: {i + 1}, Step: {step}, Phase: {current_phase}, Cycle: {cycle_count}, Green Time: {green_time}s, Reward: {reward:.2f}, Cumulative: {cumulative_reward:.2f}")
        
        if cycle_count % RECORD_DATA_FREQ == 0:
            epoch_data[i]['cycle_count'].append(cycle_count)
            epoch_data[i]['reward'].append(reward)
            epoch_data[i]['cumulative_reward'].append(cumulative_reward)
            epoch_data[i]['waiting_time'].append(after_waiting_time)
            epoch_data[i]['waiting_time_per_vehicle'].append(after_avg_waiting_time)
            epoch_data[i]['phase'].append(current_phase)
            epoch_data[i]['queue_length'].append(get_total_vehicle_in_lane())
            epoch_data[i]['throughput'].append(throughput)
        
        # Move to next phase in sequence
        phase_index = (phase_index + 1) % len(phase_sequence)
    
    # Save configuration after each epoch
    print(f"Epoch {i + 1} completed. Total steps: {step}, Cumulative reward: {cumulative_reward:.2f}")
    
    with open(f"fixed_control_epoch_{i + 1}_config.txt", "w") as f:
        f.write(f"=== Fixed Traffic Light Control - Epoch {i + 1} ===\n")
        f.write(f"Control Method: Fixed Timing\n")
        f.write(f"Phase 0 Green Time: {FIXED_GREEN_TIMES[0]}s\n")
        f.write(f"Phase 3 Green Time: {FIXED_GREEN_TIMES[3]}s\n")
        f.write(f"Phase 6 Green Time: {FIXED_GREEN_TIMES[6]}s\n")
        f.write(f"Yellow Time: {YELLOW_TIME}s\n")
        f.write(f"Red Time: {RED_TIME}s\n")
        f.write(f"Total Cycles: {cycle_count}\n")
        f.write(f"Average Queue Length: {np.mean(epoch_data[i]['queue_length']):.2f}\n")
        f.write(f"Average Waiting Time: {np.mean(epoch_data[i]['waiting_time']):.2f} s\n")
        f.write(f"Average Waiting Time per Vehicle: {np.mean(epoch_data[i]['waiting_time_per_vehicle']):.2f} s\n")
        f.write(f"Average Reward: {np.mean(epoch_data[i]['reward']):.2f}\n")
        f.write(f"Cumulative Reward: {cumulative_reward:.2f}\n")
        f.write(f"Average Throughput: {np.mean(epoch_data[i]['throughput']):.2f}\n")
        f.write(f"\nVehicles Through:\n")
        for vtype, count in epoch_data[i]['vehicles_through'].items():
            f.write(f"  {vtype}: {count}\n")
        
        # Phase distribution
        phase_counts = {}
        for phase in epoch_data[i]['phase']:
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
        
        f.write(f"\nPhase Distribution:\n")
        for phase, count in phase_counts.items():
            f.write(f"Phase {phase} ({FIXED_GREEN_TIMES[phase]}s): {count} times\n")
    
    traci.close()

# ==========================================================
# Final Summary
# ==========================================================
print("\n=== Fixed Traffic Light Control Summary ===")

# Compute overall averages
avg_queue_length_per_epoch = []
avg_total_waiting_time_per_epoch = []
avg_waiting_time_per_vehicle_per_epoch = []
avg_reward_per_epoch = []
avg_cumulative_reward_per_epoch = []
avg_throughput_per_epoch = []

for i in range(EPOCHS):
    data = epoch_data[i]
    avg_queue_length_per_epoch.append(np.mean(data['queue_length']))
    avg_total_waiting_time_per_epoch.append(np.mean(data['waiting_time']))
    avg_waiting_time_per_vehicle_per_epoch.append(np.mean(data['waiting_time_per_vehicle']))
    avg_reward_per_epoch.append(np.mean(data['reward']))
    avg_cumulative_reward_per_epoch.append(np.mean(data['cumulative_reward']))
    avg_throughput_per_epoch.append(np.mean(data['throughput']))

# Save overall summary
with open("fixed_control_summary.txt", "w") as f:
    f.write("=== FIXED TRAFFIC LIGHT CONTROL - OVERALL SUMMARY ===\n\n")
    f.write("Control Configuration:\n")
    f.write(f"- Phase 0 (Green): {FIXED_GREEN_TIMES[0]}s\n")
    f.write(f"- Phase 3 (Green): {FIXED_GREEN_TIMES[3]}s\n")
    f.write(f"- Phase 6 (Green): {FIXED_GREEN_TIMES[6]}s\n")
    f.write(f"- Yellow Time: {YELLOW_TIME}s\n")
    f.write(f"- Red Time: {RED_TIME}s\n")
    f.write(f"- Total Epochs: {EPOCHS}\n")
    f.write(f"- Simulation Time per Epoch: {TIME_SIMULATION}s\n\n")
    
    f.write("Performance Metrics (Averaged across all epochs):\n")
    f.write(f"- Average Queue Length: {np.mean(avg_queue_length_per_epoch):.2f} vehicles\n")
    f.write(f"- Average Total Waiting Time: {np.mean(avg_total_waiting_time_per_epoch):.2f} s\n")
    f.write(f"- Average Waiting Time per Vehicle: {np.mean(avg_waiting_time_per_vehicle_per_epoch):.2f} s\n")
    f.write(f"- Average Reward per Cycle: {np.mean(avg_reward_per_epoch):.2f}\n")
    f.write(f"- Average Cumulative Reward: {np.mean(avg_cumulative_reward_per_epoch):.2f}\n")
    f.write(f"- Average Throughput: {np.mean(avg_throughput_per_epoch):.2f} vehicles/cycle\n\n")
    
    f.write("Per-Epoch Results:\n")
    for i in range(EPOCHS):
        f.write(f"\nEpoch {i+1}:\n")
        f.write(f"  Queue Length: {avg_queue_length_per_epoch[i]:.2f}\n")
        f.write(f"  Total Waiting Time: {avg_total_waiting_time_per_epoch[i]:.2f} s\n")
        f.write(f"  Waiting Time per Vehicle: {avg_waiting_time_per_vehicle_per_epoch[i]:.2f} s\n")
        f.write(f"  Average Reward: {avg_reward_per_epoch[i]:.2f}\n")
        f.write(f"  Cumulative Reward: {avg_cumulative_reward_per_epoch[i]:.2f}\n")
        f.write(f"  Throughput: {avg_throughput_per_epoch[i]:.2f}\n")

    f.write("\nVehicles Through (per epoch):\n")
    for i in range(EPOCHS):
        f.write(f"Epoch {i+1}:\n")
        for vtype, count in epoch_data[i]['vehicles_through'].items():
            f.write(f"  {vtype}: {count}\n")


print("Fixed traffic light control evaluation completed!")
print("Results saved to 'fixed_control_summary.txt' and individual epoch files.")