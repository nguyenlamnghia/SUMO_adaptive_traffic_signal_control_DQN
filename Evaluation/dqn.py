import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import traci
import pandas as pd
from scipy.ndimage import uniform_filter1d
import json
from datetime import datetime

# ==========================================================
# Configuration and Constants
# ==========================================================
SUMO_CONFIG = [
    "sumo", "--no-warnings",
    "-c", "../Sumo/v1/datn.sumocfg",
    "--step-length", "0.1",
    "--lateral-resolution", "0.1"
]

# DQN Parameters
ACTIONS = [0, 1, 2, 3, 4]
GREEN_TIMES = [15, 25, 35, 45, 55]

# Traffic Light Parameters
TLS_ID = "clusterJ12_J2_J3_J6_#2more"
YELLOW_TIME = 3
RED_TIME = 1
STEP_LENGTH = 0.1
SEC_TO_STEP = int(1 / STEP_LENGTH)
TIME_SIMULATION = 7200  # 2 hours simulation
STEP_SIMULATION = TIME_SIMULATION * SEC_TO_STEP
STATE_SIZE = 17

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
# SUMO Environment Setup
# ==========================================================
def setup_sumo():
    """Setup SUMO environment variable and path."""
    if "SUMO_HOME" not in os.environ:
        sys.exit("Please declare environment variable 'SUMO_HOME'")
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)

# ==========================================================
# Traffic Light Evaluation Environment
# ==========================================================
class TrafficLightEvaluator:
    """Environment class for evaluating trained DQN model."""
    
    def __init__(self, sumo_config, tls_id, model_path):
        self.sumo_config = sumo_config
        self.tls_id = tls_id
        self.model = keras.models.load_model(model_path)
        self.step = 0
        self.cycle_count = 0
        self.vehicles_through = {v: 0 for v in VEHICLE_WEIGHTS.keys()}
        self.incoming_vehicles_count = {d: 0 for d in DETECTORS}
        
        # Data collection for each cycle
        self.cycle_data = {
            "cycle": [],
            "simulation_time": [],
            "action": [],
            "green_time": [],
            "phase": [],
            "queue_length": [],
            "total_waiting_time": [],
            "avg_waiting_time_per_vehicle": [],
            "occupancy": [],
            "density": [],
            "throughput": [],
            "vehicles_in_detectors": [],
            "flow_rate": [],
            "jam_detected": [],
            "vehicles_through_motorcycle": [],
            "vehicles_through_passenger": [],
            "vehicles_through_bus": [],
            "vehicles_through_truck": [],
            "total_vehicles_through": [],
            "cumulative_throughput": [],
            "avg_speed": [],
            "fuel_consumption": [],
            "co2_emission": []
        }

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
        return state, occupancy, vehicle_counts

    def get_action(self, state):
        """Get action from trained model."""
        q_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)[0]
        return np.argmax(q_values)

    def apply_action(self, action):
        """Apply action and simulate for green, yellow, red times."""
        current_phase = traci.trafficlight.getPhase(self.tls_id)
        green_time = GREEN_TIMES[action]
        previous_vehicles = {d: set(traci.lanearea.getLastStepVehicleIDs(d)) for d in DETECTORS}
        counted_vehicles = []
        throughput = 0

        # Simulate green phase
        traci.trafficlight.setPhase(self.tls_id, current_phase)
        traci.trafficlight.setPhaseDuration(self.tls_id, green_time * SEC_TO_STEP)
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
        """Simulate a single phase and calculate throughput."""
        throughput = 0
        for _ in range(int(duration * SEC_TO_STEP)):
            current_vehicles_in_junction = set(self._get_total_vehicle_in_junction())
            
            # Count incoming vehicles
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
            
            # Handle specific junction lanes
            if phase in [3, 6]:
                before_vehicles -= set(traci.lane.getLastStepVehicleIDs(":J3_0_0"))
                before_vehicles -= set(traci.lane.getLastStepVehicleIDs(":J3_8_0"))

            # Count vehicles that passed through
            vehicles_passed = before_vehicles - current_vehicles_in_junction
            for veh_id in vehicles_passed:
                try:
                    vehicle_class = traci.vehicle.getVehicleClass(veh_id)
                    if vehicle_class in self.vehicles_through:
                        self.vehicles_through[vehicle_class] += 1
                except:
                    pass
            throughput += self._convert_mean_vehicle(vehicles_passed)
        
        return throughput

    def get_detailed_metrics(self):
        """Get detailed metrics for current state."""
        # Basic metrics
        total_waiting_time = sum(traci.lanearea.getLastStepVehicleNumber(d) for d in DETECTORS)
        total_vehicles = sum(traci.lanearea.getLastStepVehicleNumber(d) for d in DETECTORS)
        avg_waiting_time = total_waiting_time / total_vehicles if total_vehicles > 0 else 0
        queue_length = total_vehicles
        
        # Occupancy and density
        occupancy_values = [traci.lanearea.getLastStepOccupancy(d) for d in DETECTORS]
        avg_occupancy = np.mean(occupancy_values)
        
        # Calculate density (vehicles per km)
        density_values = []
        for detector in DETECTORS:
            try:
                lane_length = traci.lanearea.getLength(detector) / 1000  # Convert to km
                vehicle_count = traci.lanearea.getLastStepVehicleNumber(detector)
                density = vehicle_count / lane_length if lane_length > 0 else 0
                density_values.append(density)
            except:
                density_values.append(0)
        avg_density = np.mean(density_values)
        
        # Flow rate (vehicles per hour)
        flow_rates = []
        for detector in DETECTORS:
            flow_rate = self.incoming_vehicles_count[detector] * 3600 / (self.step * STEP_LENGTH) if self.step > 0 else 0
            flow_rates.append(flow_rate)
        avg_flow_rate = np.mean(flow_rates)
        
        # Average speed
        try:
            all_vehicles = []
            for detector in DETECTORS:
                all_vehicles.extend(traci.lanearea.getLastStepVehicleIDs(detector))
            
            speeds = []
            for veh_id in all_vehicles:
                try:
                    speed = traci.vehicle.getSpeed(veh_id)
                    speeds.append(speed)
                except:
                    pass
            avg_speed = np.mean(speeds) if speeds else 0
        except:
            avg_speed = 0
        
        # Environmental metrics
        fuel_consumption = 0
        co2_emission = 0
        try:
            all_vehicles = []
            for detector in DETECTORS:
                all_vehicles.extend(traci.lanearea.getLastStepVehicleIDs(detector))
            
            for veh_id in all_vehicles:
                try:
                    fuel_consumption += traci.vehicle.getFuelConsumption(veh_id)
                    co2_emission += traci.vehicle.getCO2Emission(veh_id)
                except:
                    pass
        except:
            pass
        
        # Jam detection
        vehicles_in_junction = self._get_total_vehicle_in_junction()
        jam_detected = len(vehicles_in_junction) > 30
        
        return {
            'total_waiting_time': total_waiting_time,
            'avg_waiting_time': avg_waiting_time,
            'queue_length': queue_length,
            'occupancy': occupancy_values,
            'avg_occupancy': avg_occupancy,
            'density': density_values,
            'avg_density': avg_density,
            'flow_rates': flow_rates,
            'avg_flow_rate': avg_flow_rate,
            'avg_speed': avg_speed,
            'fuel_consumption': fuel_consumption,
            'co2_emission': co2_emission,
            'jam_detected': jam_detected,
            'vehicles_in_detectors': [traci.lanearea.getLastStepVehicleNumber(d) for d in DETECTORS]
        }

    def collect_cycle_data(self, action, throughput, metrics):
        """Collect data for current cycle."""
        current_phase = traci.trafficlight.getPhase(self.tls_id)
        simulation_time = self.step * STEP_LENGTH
        
        self.cycle_data["cycle"].append(self.cycle_count)
        self.cycle_data["simulation_time"].append(simulation_time)
        self.cycle_data["action"].append(action)
        self.cycle_data["green_time"].append(GREEN_TIMES[action])
        self.cycle_data["phase"].append(current_phase)
        self.cycle_data["queue_length"].append(metrics['queue_length'])
        self.cycle_data["total_waiting_time"].append(metrics['total_waiting_time'])
        self.cycle_data["avg_waiting_time_per_vehicle"].append(metrics['avg_waiting_time'])
        self.cycle_data["occupancy"].append(metrics['avg_occupancy'])
        self.cycle_data["density"].append(metrics['avg_density'])
        self.cycle_data["throughput"].append(throughput)
        self.cycle_data["vehicles_in_detectors"].append(sum(metrics['vehicles_in_detectors']))
        self.cycle_data["flow_rate"].append(metrics['avg_flow_rate'])
        self.cycle_data["jam_detected"].append(metrics['jam_detected'])
        
        # Vehicle type throughput
        self.cycle_data["vehicles_through_motorcycle"].append(self.vehicles_through["motorcycle"])
        self.cycle_data["vehicles_through_passenger"].append(self.vehicles_through["passenger"])
        self.cycle_data["vehicles_through_bus"].append(self.vehicles_through["bus"])
        self.cycle_data["vehicles_through_truck"].append(self.vehicles_through["truck"])
        
        total_through = sum(self.vehicles_through.values())
        self.cycle_data["total_vehicles_through"].append(total_through)
        
        # Cumulative throughput
        cumulative_throughput = sum(self.cycle_data["throughput"])
        self.cycle_data["cumulative_throughput"].append(cumulative_throughput)
        
        self.cycle_data["avg_speed"].append(metrics['avg_speed'])
        self.cycle_data["fuel_consumption"].append(metrics['fuel_consumption'])
        self.cycle_data["co2_emission"].append(metrics['co2_emission'])

    def _convert_mean_vehicle(self, vehicles):
        """Convert vehicle list to weighted sum based on vehicle type."""
        try:
            return sum(VEHICLE_WEIGHTS.get(traci.vehicle.getVehicleClass(v), 0) for v in vehicles)
        except:
            return 0

    def _get_total_vehicle_in_junction(self):
        """Get all vehicles in the junction."""
        try:
            lanes = [l for l in traci.lane.getIDList() if l.startswith(":J3_")]
            return [veh for lane in lanes for veh in traci.lane.getLastStepVehicleIDs(lane)]
        except:
            return []

# ==========================================================
# Data Analysis and Visualization
# ==========================================================
def save_evaluation_data(evaluator, model_name):
    """Save evaluation data to CSV and JSON files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save cycle data to CSV
    df = pd.DataFrame(evaluator.cycle_data)
    csv_filename = f"evaluation_{model_name}_{timestamp}.csv"
    df.to_csv(csv_filename, index=False)
    
    # Save summary statistics
    summary_stats = {
        "model_name": model_name,
        "timestamp": timestamp,
        "total_cycles": len(evaluator.cycle_data["cycle"]),
        "simulation_time": TIME_SIMULATION,
        "avg_queue_length": np.mean(evaluator.cycle_data["queue_length"]),
        "avg_waiting_time": np.mean(evaluator.cycle_data["total_waiting_time"]),
        "avg_waiting_time_per_vehicle": np.mean(evaluator.cycle_data["avg_waiting_time_per_vehicle"]),
        "avg_occupancy": np.mean(evaluator.cycle_data["occupancy"]),
        "avg_density": np.mean(evaluator.cycle_data["density"]),
        "total_throughput": sum(evaluator.cycle_data["throughput"]),
        "avg_throughput_per_cycle": np.mean(evaluator.cycle_data["throughput"]),
        "avg_flow_rate": np.mean(evaluator.cycle_data["flow_rate"]),
        "total_jams": sum(evaluator.cycle_data["jam_detected"]),
        "total_vehicles_through": evaluator.cycle_data["total_vehicles_through"][-1] if evaluator.cycle_data["total_vehicles_through"] else 0,
        "avg_speed": np.mean(evaluator.cycle_data["avg_speed"]),
        "total_fuel_consumption": sum(evaluator.cycle_data["fuel_consumption"]),
        "total_co2_emission": sum(evaluator.cycle_data["co2_emission"]),
        "action_distribution": {
            f"action_{i}_{GREEN_TIMES[i]}s": evaluator.cycle_data["action"].count(i) 
            for i in range(len(ACTIONS))
        }
    }
    
    json_filename = f"evaluation_summary_{model_name}_{timestamp}.json"
    with open(json_filename, 'w') as f:
        json.dump(summary_stats, f, indent=4)
    
    print(f"Evaluation data saved to: {csv_filename}")
    print(f"Summary statistics saved to: {json_filename}")
    
    return csv_filename, json_filename, summary_stats

def plot_evaluation_results(evaluator, model_name):
    """Plot evaluation results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle(f'DQN Traffic Light Control Evaluation - {model_name}', fontsize=16)
    
    # 1. Queue Length over Time
    axes[0, 0].plot(evaluator.cycle_data["simulation_time"], evaluator.cycle_data["queue_length"], 'b-', alpha=0.7)
    axes[0, 0].plot(evaluator.cycle_data["simulation_time"], 
                    uniform_filter1d(evaluator.cycle_data["queue_length"], size=10), 'r-', linewidth=2)
    axes[0, 0].set_title('Queue Length Over Time')
    axes[0, 0].set_xlabel('Simulation Time (s)')
    axes[0, 0].set_ylabel('Queue Length')
    axes[0, 0].grid(True)
    axes[0, 0].legend(['Raw Data', 'Moving Average'])
    
    # 2. Waiting Time over Time
    axes[0, 1].plot(evaluator.cycle_data["simulation_time"], evaluator.cycle_data["total_waiting_time"], 'g-', alpha=0.7)
    axes[0, 1].plot(evaluator.cycle_data["simulation_time"], 
                    uniform_filter1d(evaluator.cycle_data["total_waiting_time"], size=10), 'r-', linewidth=2)
    axes[0, 1].set_title('Total Waiting Time Over Time')
    axes[0, 1].set_xlabel('Simulation Time (s)')
    axes[0, 1].set_ylabel('Waiting Time (s)')
    axes[0, 1].grid(True)
    axes[0, 1].legend(['Raw Data', 'Moving Average'])
    
    # 3. Throughput over Time
    axes[0, 2].plot(evaluator.cycle_data["simulation_time"], evaluator.cycle_data["throughput"], 'm-', alpha=0.7)
    axes[0, 2].plot(evaluator.cycle_data["simulation_time"], 
                    uniform_filter1d(evaluator.cycle_data["throughput"], size=10), 'r-', linewidth=2)
    axes[0, 2].set_title('Throughput Over Time')
    axes[0, 2].set_xlabel('Simulation Time (s)')
    axes[0, 2].set_ylabel('Throughput')
    axes[0, 2].grid(True)
    axes[0, 2].legend(['Raw Data', 'Moving Average'])
    
    # 4. Occupancy and Density
    axes[1, 0].plot(evaluator.cycle_data["simulation_time"], evaluator.cycle_data["occupancy"], 'c-', label='Occupancy')
    axes[1, 0].plot(evaluator.cycle_data["simulation_time"], evaluator.cycle_data["density"], 'orange', label='Density')
    axes[1, 0].set_title('Occupancy and Density Over Time')
    axes[1, 0].set_xlabel('Simulation Time (s)')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].grid(True)
    axes[1, 0].legend()
    
    # 5. Flow Rate
    axes[1, 1].plot(evaluator.cycle_data["simulation_time"], evaluator.cycle_data["flow_rate"], 'purple', alpha=0.7)
    axes[1, 1].plot(evaluator.cycle_data["simulation_time"], 
                    uniform_filter1d(evaluator.cycle_data["flow_rate"], size=10), 'r-', linewidth=2)
    axes[1, 1].set_title('Flow Rate Over Time')
    axes[1, 1].set_xlabel('Simulation Time (s)')
    axes[1, 1].set_ylabel('Flow Rate (veh/h)')
    axes[1, 1].grid(True)
    axes[1, 1].legend(['Raw Data', 'Moving Average'])
    
    # 6. Cumulative Throughput
    axes[1, 2].plot(evaluator.cycle_data["simulation_time"], evaluator.cycle_data["cumulative_throughput"], 'brown')
    axes[1, 2].set_title('Cumulative Throughput')
    axes[1, 2].set_xlabel('Simulation Time (s)')
    axes[1, 2].set_ylabel('Cumulative Throughput')
    axes[1, 2].grid(True)
    
    # 7. Vehicle Type Throughput
    axes[2, 0].plot(evaluator.cycle_data["simulation_time"], evaluator.cycle_data["vehicles_through_motorcycle"], label='Motorcycle')
    axes[2, 0].plot(evaluator.cycle_data["simulation_time"], evaluator.cycle_data["vehicles_through_passenger"], label='Passenger')
    axes[2, 0].plot(evaluator.cycle_data["simulation_time"], evaluator.cycle_data["vehicles_through_bus"], label='Bus')
    axes[2, 0].plot(evaluator.cycle_data["simulation_time"], evaluator.cycle_data["vehicles_through_truck"], label='Truck')
    axes[2, 0].set_title('Vehicle Type Throughput')
    axes[2, 0].set_xlabel('Simulation Time (s)')
    axes[2, 0].set_ylabel('Number of Vehicles')
    axes[2, 0].grid(True)
    axes[2, 0].legend()
    
    # 8. Action Distribution
    action_counts = [evaluator.cycle_data["action"].count(i) for i in range(len(ACTIONS))]
    action_labels = [f'A{i}\n({GREEN_TIMES[i]}s)' for i in range(len(ACTIONS))]
    axes[2, 1].bar(action_labels, action_counts, color=['red', 'orange', 'yellow', 'green', 'blue'])
    axes[2, 1].set_title('Action Distribution')
    axes[2, 1].set_xlabel('Action (Green Time)')
    axes[2, 1].set_ylabel('Frequency')
    axes[2, 1].grid(True, axis='y')
    
    # 9. Average Speed and Environmental Metrics
    ax_speed = axes[2, 2]
    ax_env = ax_speed.twinx()
    
    line1 = ax_speed.plot(evaluator.cycle_data["simulation_time"], evaluator.cycle_data["avg_speed"], 'b-', label='Avg Speed')
    line2 = ax_env.plot(evaluator.cycle_data["simulation_time"], evaluator.cycle_data["fuel_consumption"], 'r-', label='Fuel Consumption')
    
    ax_speed.set_xlabel('Simulation Time (s)')
    ax_speed.set_ylabel('Average Speed (m/s)', color='b')
    ax_env.set_ylabel('Fuel Consumption (ml/s)', color='r')
    ax_speed.tick_params(axis='y', labelcolor='b')
    ax_env.tick_params(axis='y', labelcolor='r')
    ax_speed.set_title('Speed and Environmental Metrics')
    ax_speed.grid(True)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax_speed.legend(lines, labels, loc='upper left')
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = f"evaluation_plots_{model_name}_{timestamp}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Evaluation plots saved to: {plot_filename}")
    return plot_filename

# ==========================================================
# Main Evaluation Function
# ==========================================================
def evaluate_model(model_path, model_name=None):
    """Main function to evaluate trained DQN model."""
    if model_name is None:
        model_name = os.path.basename(model_path).replace('.keras', '')
    
    print(f"\n=== Starting DQN Model Evaluation ===")
    print(f"Model: {model_path}")
    print(f"Simulation Time: {TIME_SIMULATION} seconds")
    
    setup_sumo()
    
    try:
        evaluator = TrafficLightEvaluator(SUMO_CONFIG, TLS_ID, model_path)
        evaluator.start()
        
        print("Starting simulation...")
        start_time = datetime.now()
        
        while evaluator.step < STEP_SIMULATION:
            # Get current state
            state, occupancy, vehicle_counts = evaluator.get_state()
            
            # Get action from trained model
            action = evaluator.get_action(state)
            
            # Get detailed metrics before applying action
            metrics = evaluator.get_detailed_metrics()
            
            # Apply action and get throughput
            throughput = evaluator.apply_action(action)
            
            # Collect data for this cycle
            evaluator.collect_cycle_data(action, throughput, metrics)
            
            # Update counters
            evaluator.cycle_count += 1
            evaluator.step += (GREEN_TIMES[action] + YELLOW_TIME + RED_TIME) * SEC_TO_STEP
            
            # Print progress every 10 cycles
            if evaluator.cycle_count % 10 == 0:
                progress = (evaluator.step / STEP_SIMULATION) * 100
                print(f"Progress: {progress:.1f}% - Cycle: {evaluator.cycle_count}, "
                      f"Action: {action} ({GREEN_TIMES[action]}s), "
                      f"Queue: {metrics['queue_length']:.1f}, "
                      f"Throughput: {throughput:.2f}")
        
        end_time = datetime.now()
        simulation_duration = (end_time - start_time).total_seconds()
        
        print(f"\nSimulation completed in {simulation_duration:.2f} seconds")
        print(f"Total cycles: {evaluator.cycle_count}")
        
        # Save data and create plots
        csv_file, json_file, summary_stats = save_evaluation_data(evaluator, model_name)
        plot_file = plot_evaluation_results(evaluator, model_name)
        
        # Print summary statistics
        print(f"\n=== Summary Statistics ===")
        print(f"Average Queue Length: {summary_stats['avg_queue_length']:.2f}")
        print(f"Average Waiting Time: {summary_stats['avg_waiting_time']:.2f} s")
        print(f"Average Waiting Time per Vehicle: {summary_stats['avg_waiting_time_per_vehicle']:.2f} s")
        print(f"Average Occupancy: {summary_stats['avg_occupancy']:.3f}")
        print(f"Average Density: {summary_stats['avg_density']:.2f} veh/km")
        print(f"Total Throughput: {summary_stats['total_throughput']:.2f}")
        print(f"Average Flow Rate: {summary_stats['avg_flow_rate']:.2f} veh/h")
        print(f"Total Jams Detected: {summary_stats['total_jams']}")
        print(f"Total Vehicles Through: {summary_stats['total_vehicles_through']}")
        print(f"Average Speed: {summary_stats['avg_speed']:.2f} m/s")
        
        evaluator.close()
        
        return {
            'csv_file': csv_file,
            'json_file': json_file,
            'plot_file': plot_file,
            'summary_stats': summary_stats,
            'evaluator': evaluator
        }
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        try:
            evaluator.close()
        except:
            pass
        raise

# ==========================================================
# Main Execution
# ==========================================================
if __name__ == "__main__":
    # Example usage
    model_path = "quad_dqn.keras"  # Change this to your model path
    try:
        results = evaluate_model(model_path)
        print(f"\nEvaluation completed successfully!")
        print(f"Results saved to: {results['csv_file']}")
        
    except FileNotFoundError:
        print(f"Model file not found: {model_path}")
        print("Please make sure the model file exists and the path is correct.")
    except Exception as e:
        print(f"Evaluation failed: {e}")