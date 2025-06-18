import os
import sys
import numpy as np
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

# Fixed Timer Parameters
FIXED_GREEN_TIMES = {
    0: 30,  # Phase 0 green time (seconds)
    3: 30,  # Phase 3 green time (seconds)
    6: 30   # Phase 6 green time (seconds)
}

# Traffic Light Parameters
TLS_ID = "clusterJ12_J2_J3_J6_#2more"
YELLOW_TIME = 3
RED_TIME = 1
STEP_LENGTH = 0.1
SEC_TO_STEP = int(1 / STEP_LENGTH)
TIME_SIMULATION = 7200  # 2 hours simulation
STEP_SIMULATION = TIME_SIMULATION * SEC_TO_STEP

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
# Fixed Timer Traffic Light Controller
# ==========================================================
class FixedTimerController:
    """Fixed timer traffic light controller."""
    
    def __init__(self, sumo_config, tls_id, green_times):
        self.sumo_config = sumo_config
        self.tls_id = tls_id
        self.green_times = green_times
        self.step = 0
        self.cycle_count = 0
        self.vehicles_through = {v: 0 for v in VEHICLE_WEIGHTS.keys()}
        self.incoming_vehicles_count = {d: 0 for d in DETECTORS}
        
        # Phase cycle for the traffic light (0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8 -> 0)
        self.phase_cycle = [0, 3, 6]  # Main phases (green phases)
        self.current_phase_index = 0
        
        # Data collection for each cycle
        self.cycle_data = {
            "cycle": [],
            "simulation_time": [],
            "phase": [],
            "green_time": [],
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

    def get_current_phase(self):
        """Get current traffic light phase."""
        return traci.trafficlight.getPhase(self.tls_id)

    def apply_fixed_timer_cycle(self):
        """Apply one complete cycle of fixed timer control."""
        current_main_phase = self.phase_cycle[self.current_phase_index]
        green_time = self.green_times[current_main_phase]
        
        previous_vehicles = {d: set(traci.lanearea.getLastStepVehicleIDs(d)) for d in DETECTORS}
        counted_vehicles = []
        throughput = 0

        # Set to green phase
        traci.trafficlight.setPhase(self.tls_id, current_main_phase)
        traci.trafficlight.setPhaseDuration(self.tls_id, green_time * SEC_TO_STEP)
        throughput += self._simulate_phase(current_main_phase, green_time, previous_vehicles, counted_vehicles)
        
        # Yellow phase
        yellow_phase = current_main_phase + 1
        traci.trafficlight.setPhase(self.tls_id, yellow_phase)
        traci.trafficlight.setPhaseDuration(self.tls_id, YELLOW_TIME * SEC_TO_STEP)
        throughput += self._simulate_phase(yellow_phase, YELLOW_TIME, previous_vehicles, counted_vehicles)

        # Red phase
        red_phase = current_main_phase + 2
        traci.trafficlight.setPhase(self.tls_id, red_phase)
        traci.trafficlight.setPhaseDuration(self.tls_id, RED_TIME * SEC_TO_STEP)
        throughput += self._simulate_phase(red_phase, RED_TIME, previous_vehicles, counted_vehicles)

        # Move to next phase in cycle
        self.current_phase_index = (self.current_phase_index + 1) % len(self.phase_cycle)
        
        return throughput, current_main_phase, green_time

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

    def collect_cycle_data(self, phase, green_time, throughput, metrics):
        """Collect data for current cycle."""
        simulation_time = self.step * STEP_LENGTH
        
        self.cycle_data["cycle"].append(self.cycle_count)
        self.cycle_data["simulation_time"].append(simulation_time)
        self.cycle_data["phase"].append(phase)
        self.cycle_data["green_time"].append(green_time)
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
def save_evaluation_data(controller, control_name, green_times_config):
    """Save evaluation data to CSV and JSON files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save cycle data to CSV
    df = pd.DataFrame(controller.cycle_data)
    csv_filename = f"evaluation_{control_name}_{timestamp}.csv"
    df.to_csv(csv_filename, index=False)
    
    # Save summary statistics
    summary_stats = {
        "control_method": control_name,
        "green_times_config": green_times_config,
        "timestamp": timestamp,
        "total_cycles": len(controller.cycle_data["cycle"]),
        "simulation_time": TIME_SIMULATION,
        "avg_queue_length": np.mean(controller.cycle_data["queue_length"]),
        "avg_waiting_time": np.mean(controller.cycle_data["total_waiting_time"]),
        "avg_waiting_time_per_vehicle": np.mean(controller.cycle_data["avg_waiting_time_per_vehicle"]),
        "avg_occupancy": np.mean(controller.cycle_data["occupancy"]),
        "avg_density": np.mean(controller.cycle_data["density"]),
        "total_throughput": sum(controller.cycle_data["throughput"]),
        "avg_throughput_per_cycle": np.mean(controller.cycle_data["throughput"]),
        "avg_flow_rate": np.mean(controller.cycle_data["flow_rate"]),
        "total_jams": sum(controller.cycle_data["jam_detected"]),
        "total_vehicles_through": controller.cycle_data["total_vehicles_through"][-1] if controller.cycle_data["total_vehicles_through"] else 0,
        "avg_speed": np.mean(controller.cycle_data["avg_speed"]),
        "total_fuel_consumption": sum(controller.cycle_data["fuel_consumption"]),
        "total_co2_emission": sum(controller.cycle_data["co2_emission"]),
        "phase_distribution": {
            f"phase_{phase}_{green_times_config[phase]}s": controller.cycle_data["phase"].count(phase) 
            for phase in green_times_config.keys()
        }
    }
    
    json_filename = f"evaluation_summary_{control_name}_{timestamp}.json"
    with open(json_filename, 'w') as f:
        json.dump(summary_stats, f, indent=4)
    
    print(f"Evaluation data saved to: {csv_filename}")
    print(f"Summary statistics saved to: {json_filename}")
    
    return csv_filename, json_filename, summary_stats

def plot_evaluation_results(controller, control_name, green_times_config):
    """Plot evaluation results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle(f'Fixed Timer Traffic Light Control Evaluation - {control_name}', fontsize=16)
    
    # 1. Queue Length over Time
    axes[0, 0].plot(controller.cycle_data["simulation_time"], controller.cycle_data["queue_length"], 'b-', alpha=0.7)
    axes[0, 0].plot(controller.cycle_data["simulation_time"], 
                    uniform_filter1d(controller.cycle_data["queue_length"], size=10), 'r-', linewidth=2)
    axes[0, 0].set_title('Queue Length Over Time')
    axes[0, 0].set_xlabel('Simulation Time (s)')
    axes[0, 0].set_ylabel('Queue Length')
    axes[0, 0].grid(True)
    axes[0, 0].legend(['Raw Data', 'Moving Average'])
    
    # 2. Waiting Time over Time
    axes[0, 1].plot(controller.cycle_data["simulation_time"], controller.cycle_data["total_waiting_time"], 'g-', alpha=0.7)
    axes[0, 1].plot(controller.cycle_data["simulation_time"], 
                    uniform_filter1d(controller.cycle_data["total_waiting_time"], size=10), 'r-', linewidth=2)
    axes[0, 1].set_title('Total Waiting Time Over Time')
    axes[0, 1].set_xlabel('Simulation Time (s)')
    axes[0, 1].set_ylabel('Waiting Time (s)')
    axes[0, 1].grid(True)
    axes[0, 1].legend(['Raw Data', 'Moving Average'])
    
    # 3. Throughput over Time
    axes[0, 2].plot(controller.cycle_data["simulation_time"], controller.cycle_data["throughput"], 'm-', alpha=0.7)
    axes[0, 2].plot(controller.cycle_data["simulation_time"], 
                    uniform_filter1d(controller.cycle_data["throughput"], size=10), 'r-', linewidth=2)
    axes[0, 2].set_title('Throughput Over Time')
    axes[0, 2].set_xlabel('Simulation Time (s)')
    axes[0, 2].set_ylabel('Throughput')
    axes[0, 2].grid(True)
    axes[0, 2].legend(['Raw Data', 'Moving Average'])
    
    # 4. Occupancy and Density
    axes[1, 0].plot(controller.cycle_data["simulation_time"], controller.cycle_data["occupancy"], 'c-', label='Occupancy')
    axes[1, 0].plot(controller.cycle_data["simulation_time"], controller.cycle_data["density"], 'orange', label='Density')
    axes[1, 0].set_title('Occupancy and Density Over Time')
    axes[1, 0].set_xlabel('Simulation Time (s)')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].grid(True)
    axes[1, 0].legend()
    
    # 5. Flow Rate
    axes[1, 1].plot(controller.cycle_data["simulation_time"], controller.cycle_data["flow_rate"], 'purple', alpha=0.7)
    axes[1, 1].plot(controller.cycle_data["simulation_time"], 
                    uniform_filter1d(controller.cycle_data["flow_rate"], size=10), 'r-', linewidth=2)
    axes[1, 1].set_title('Flow Rate Over Time')
    axes[1, 1].set_xlabel('Simulation Time (s)')
    axes[1, 1].set_ylabel('Flow Rate (veh/h)')
    axes[1, 1].grid(True)
    axes[1, 1].legend(['Raw Data', 'Moving Average'])
    
    # 6. Cumulative Throughput
    axes[1, 2].plot(controller.cycle_data["simulation_time"], controller.cycle_data["cumulative_throughput"], 'brown')
    axes[1, 2].set_title('Cumulative Throughput')
    axes[1, 2].set_xlabel('Simulation Time (s)')
    axes[1, 2].set_ylabel('Cumulative Throughput')
    axes[1, 2].grid(True)
    
    # 7. Vehicle Type Throughput
    axes[2, 0].plot(controller.cycle_data["simulation_time"], controller.cycle_data["vehicles_through_motorcycle"], label='Motorcycle')
    axes[2, 0].plot(controller.cycle_data["simulation_time"], controller.cycle_data["vehicles_through_passenger"], label='Passenger')
    axes[2, 0].plot(controller.cycle_data["simulation_time"], controller.cycle_data["vehicles_through_bus"], label='Bus')
    axes[2, 0].plot(controller.cycle_data["simulation_time"], controller.cycle_data["vehicles_through_truck"], label='Truck')
    axes[2, 0].set_title('Vehicle Type Throughput')
    axes[2, 0].set_xlabel('Simulation Time (s)')
    axes[2, 0].set_ylabel('Number of Vehicles')
    axes[2, 0].grid(True)
    axes[2, 0].legend()
    
    # 8. Phase Distribution
    phase_counts = [controller.cycle_data["phase"].count(phase) for phase in green_times_config.keys()]
    phase_labels = [f'Phase {phase}\n({green_times_config[phase]}s)' for phase in green_times_config.keys()]
    axes[2, 1].bar(phase_labels, phase_counts, color=['red', 'green', 'blue'])
    axes[2, 1].set_title('Phase Distribution')
    axes[2, 1].set_xlabel('Phase (Green Time)')
    axes[2, 1].set_ylabel('Frequency')
    axes[2, 1].grid(True, axis='y')
    
    # 9. Average Speed and Environmental Metrics
    ax_speed = axes[2, 2]
    ax_env = ax_speed.twinx()
    
    line1 = ax_speed.plot(controller.cycle_data["simulation_time"], controller.cycle_data["avg_speed"], 'b-', label='Avg Speed')
    line2 = ax_env.plot(controller.cycle_data["simulation_time"], controller.cycle_data["fuel_consumption"], 'r-', label='Fuel Consumption')
    
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
    plot_filename = f"evaluation_plots_{control_name}_{timestamp}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Evaluation plots saved to: {plot_filename}")
    return plot_filename

# ==========================================================
# Main Evaluation Function
# ==========================================================
def evaluate_fixed_timer(green_times_config, control_name="FixedTimer"):
    """Main function to evaluate fixed timer traffic light control."""
    
    print(f"\n=== Starting Fixed Timer Traffic Light Control Evaluation ===")
    print(f"Control Method: {control_name}")
    print(f"Green Times Configuration: {green_times_config}")
    print(f"Simulation Time: {TIME_SIMULATION} seconds")
    
    setup_sumo()
    
    try:
        controller = FixedTimerController(SUMO_CONFIG, TLS_ID, green_times_config)
        controller.start()
        
        print("Starting simulation...")
        start_time = datetime.now()
        
        while controller.step < STEP_SIMULATION:
            # Get detailed metrics before applying cycle
            metrics = controller.get_detailed_metrics()
            
            # Apply fixed timer cycle
            throughput, current_phase, green_time = controller.apply_fixed_timer_cycle()
            
            # Collect data for this cycle
            controller.collect_cycle_data(current_phase, green_time, throughput, metrics)
            
            # Update counters
            controller.cycle_count += 1
            controller.step += (green_time + YELLOW_TIME + RED_TIME) * SEC_TO_STEP
            
            # Print progress every 10 cycles
            if controller.cycle_count % 10 == 0:
                progress = (controller.step / STEP_SIMULATION) * 100
                print(f"Progress: {progress:.1f}% - Cycle: {controller.cycle_count}, "
                      f"Phase: {current_phase} ({green_time}s), "
                      f"Queue: {metrics['queue_length']:.1f}, "
                      f"Throughput: {throughput:.2f}")
        
        end_time = datetime.now()
        simulation_duration = (end_time - start_time).total_seconds()
        
        print(f"\nSimulation completed in {simulation_duration:.2f} seconds")
        print(f"Total cycles: {controller.cycle_count}")
        
        # Save data and create plots
        csv_file, json_file, summary_stats = save_evaluation_data(controller, control_name, green_times_config)
        plot_file = plot_evaluation_results(controller, control_name, green_times_config)
        
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
        
        controller.close()
        
        return {
            'csv_file': csv_file,
            'json_file': json_file,
            'plot_file': plot_file,
            'summary_stats': summary_stats,
            'controller': controller
        }
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        try:
            controller.close()
        except:
            pass
        raise

# ==========================================================
# Main Execution
# ==========================================================
if __name__ == "__main__":
    # Example usage with different configurations
    
    # Configuration 1: Balanced timing
    config_1 = {0: 50, 3: 30, 6: 20}
    
    # Configuration 2: Prioritize phase 0 (main road)
    config_2 = {0: 45, 3: 20, 6: 20}
    
    # Configuration 3: Equal long timing
    config_3 = {0: 40, 3: 40, 6: 40}
    
    # Choose configuration to run
    selected_config = config_1
    control_name = "FixedTimer_Balanced"
    
    try:
        results = evaluate_fixed_timer(selected_config, control_name)
        print(f"\nEvaluation completed successfully!")
        print(f"Results saved to: {results['csv_file']}")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")