"""Single drone simulation using PyBullet and the gym_pybullet_drones library.

The drone follows a trajectory loaded from external sources (function or CSV file).
Control is managed by the PID implementation in DSLPIDControl.

To run:
    $ python single_drone.py
    $ python single_drone.py --trajectory_function figure8
    $ python single_drone.py --create_example_csv
"""
import os
import time
import argparse
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
import sys  # Add this import

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Warning: pandas not installed. CSV trajectory loading will not be available.")

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

# Default parameters
DEFAULT_DRONE = DroneModel("cf2x")  # Crazyflie 2.x drone model
DEFAULT_PHYSICS = Physics("pyb")    # PyBullet physics engine
DEFAULT_GUI = True                  # Show visualization
DEFAULT_RECORD_VIDEO = False        # Don't record video
DEFAULT_PLOT = True                 # Plot results after simulation
DEFAULT_USER_DEBUG_GUI = False      # Don't show debug GUI
DEFAULT_OBSTACLES = False           # Don't add obstacles
DEFAULT_SIMULATION_FREQ_HZ = 240    # Physics update frequency
DEFAULT_CONTROL_FREQ_HZ = 48        # Control update frequency
DEFAULT_DURATION_SEC = 12           # Simulation duration
DEFAULT_OUTPUT_FOLDER = 'results'   # Results folder
DEFAULT_TRAJECTORY_SOURCE = 'built_in'  # Default trajectory source
DEFAULT_TRAJECTORY_FUNCTION = 'circle'  # Default trajectory function
DEFAULT_TRAJECTORY_FILE = None      # Default trajectory file
DEFAULT_EXAMPLE_CSV_PATH = 'trajectories/example_square.csv'  # Default example CSV path

# Collection of trajectory generation functions
def generate_circle_trajectory(num_waypoints, height=1.0, radius=0.5):
    """Generate a circular trajectory."""
    trajectory = np.zeros((num_waypoints, 3))
    for i in range(num_waypoints):
        angle = (i / num_waypoints) * (2 * np.pi)
        trajectory[i, :] = [radius * np.cos(angle), radius * np.sin(angle), height]
    return trajectory

def generate_figure8_trajectory(num_waypoints, height=1.0, radius=0.5):
    """Generate a figure-8 trajectory."""
    trajectory = np.zeros((num_waypoints, 3))
    for i in range(num_waypoints):
        t = (i / num_waypoints) * (2 * np.pi)
        trajectory[i, :] = [radius * np.sin(t), radius * np.sin(t) * np.cos(t), height]
    return trajectory

def generate_spiral_trajectory(num_waypoints, height_range=(0.5, 1.5), radius=0.5):
    """Generate a spiral trajectory that changes height."""
    trajectory = np.zeros((num_waypoints, 3))
    min_height, max_height = height_range
    height_increment = (max_height - min_height) / num_waypoints
    
    for i in range(num_waypoints):
        angle = (i / num_waypoints) * (4 * np.pi)  # 2 full rotations
        current_height = min_height + i * height_increment
        current_radius = radius * (1 - i / (2 * num_waypoints))  # Decreasing radius
        trajectory[i, :] = [current_radius * np.cos(angle), 
                           current_radius * np.sin(angle), 
                           current_height]
    return trajectory

# Dictionary mapping function names to generator functions
TRAJECTORY_FUNCTIONS = {
    'circle': generate_circle_trajectory,
    'figure8': generate_figure8_trajectory,
    'spiral': generate_spiral_trajectory
}

def load_trajectory_from_csv(file_path):
    """Load trajectory from a CSV file."""
    if not PANDAS_AVAILABLE:
        print("Error: pandas is required to load CSV files.")
        return None
        
    try:
        df = pd.read_csv(file_path)
        # Check if the CSV has the required columns
        required_columns = ['x', 'y', 'z']
        if not all(col in df.columns for col in required_columns):
            print(f"Error: CSV must contain columns {required_columns}")
            return None
            
        # Extract the trajectory points
        trajectory = df[required_columns].values
        return trajectory
    except Exception as e:
        print(f"Error loading trajectory from CSV: {e}")
        return None

def create_example_trajectory_csv(file_path=DEFAULT_EXAMPLE_CSV_PATH):
    """Create an example trajectory CSV file."""
    if not PANDAS_AVAILABLE:
        print("Error: pandas is required to create CSV files.")
        return
        
    # Generate a simple square trajectory
    points = []
    # Square in the xy-plane at z=1.0
    height = 1.0
    size = 0.5
    
    # Bottom side (moving along +x)
    for i in range(20):
        t = i / 19
        points.append([-size + 2*size*t, -size, height])
    
    # Right side (moving along +y)
    for i in range(20):
        t = i / 19
        points.append([size, -size + 2*size*t, height])
    
    # Top side (moving along -x)
    for i in range(20):
        t = i / 19
        points.append([size - 2*size*t, size, height])
    
    # Left side (moving along -y)
    for i in range(20):
        t = i / 19
        points.append([-size, size - 2*size*t, height])
    
    # Create a DataFrame and save to CSV
    df = pd.DataFrame(points, columns=['x', 'y', 'z'])
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)
    print(f"Created example trajectory file at {file_path}")

def run(
        drone=DEFAULT_DRONE,
        physics=DEFAULT_PHYSICS,
        gui=DEFAULT_GUI,
        record_video=DEFAULT_RECORD_VIDEO,
        plot=DEFAULT_PLOT,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        obstacles=DEFAULT_OBSTACLES,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        trajectory_source=DEFAULT_TRAJECTORY_SOURCE,
        trajectory_function=DEFAULT_TRAJECTORY_FUNCTION,
        trajectory_file=DEFAULT_TRAJECTORY_FILE
        ):
    """Run the simulation with a single drone following a trajectory."""
    
    # Setup parameters
    num_drones = 1  # We only need one drone
    
    # Initialize drone position and orientation
    H = 1.0         # Height of the drone
    
    # Starting position [x, y, z] and orientation [roll, pitch, yaw]
    INIT_XYZS = np.array([[0, 0, H]])  # Start at position (0,0,H)
    INIT_RPYS = np.array([[0, 0, 0]])  # No initial rotation
    
    # Create/load trajectory based on specified source
    if trajectory_source == 'built_in':
        # Use built-in trajectory generators
        NUM_WP = control_freq_hz * duration_sec
        if trajectory_function in TRAJECTORY_FUNCTIONS:
            TARGET_POS = TRAJECTORY_FUNCTIONS[trajectory_function](NUM_WP)
            print(f"Using built-in {trajectory_function} trajectory with {NUM_WP} waypoints")
        else:
            print(f"Unknown trajectory function '{trajectory_function}'. Using circle instead.")
            TARGET_POS = generate_circle_trajectory(NUM_WP)
    
    elif trajectory_source == 'csv':
        # Load trajectory from CSV file
        if trajectory_file:
            loaded_trajectory = load_trajectory_from_csv(trajectory_file)
            if loaded_trajectory is not None:
                TARGET_POS = loaded_trajectory
                NUM_WP = len(TARGET_POS)
                print(f"Loaded trajectory from {trajectory_file} with {NUM_WP} waypoints")
            else:
                # Fallback to circle if CSV loading fails
                NUM_WP = control_freq_hz * duration_sec
                TARGET_POS = generate_circle_trajectory(NUM_WP)
                print(f"Failed to load trajectory from CSV. Using circle instead with {NUM_WP} waypoints")
        else:
            print("No trajectory file specified. Using circle instead.")
            NUM_WP = control_freq_hz * duration_sec
            TARGET_POS = generate_circle_trajectory(NUM_WP)
    
    elif trajectory_source == 'function':
        # Use a specific trajectory generation function
        NUM_WP = control_freq_hz * duration_sec
        if trajectory_function in TRAJECTORY_FUNCTIONS:
            TARGET_POS = TRAJECTORY_FUNCTIONS[trajectory_function](NUM_WP)
            print(f"Using {trajectory_function} trajectory function with {NUM_WP} waypoints")
        else:
            print(f"Unknown trajectory function '{trajectory_function}'. Using circle instead.")
            TARGET_POS = generate_circle_trajectory(NUM_WP)
    
    else:
        # Default to circle if source is not recognized
        print(f"Unknown trajectory source '{trajectory_source}'. Using circle instead.")
        NUM_WP = control_freq_hz * duration_sec
        TARGET_POS = generate_circle_trajectory(NUM_WP)
    
    # Ensure we have a valid trajectory
    if TARGET_POS is None or len(TARGET_POS) == 0:
        print("Error: No valid trajectory generated. Exiting.")
        return
        
    # Start at the first waypoint
    wp_counter = 0
    
    # Create the simulation environment
    env = CtrlAviary(
        drone_model=drone,
        num_drones=num_drones,
        initial_xyzs=INIT_XYZS,
        initial_rpys=INIT_RPYS,
        physics=physics,
        neighbourhood_radius=10,
        pyb_freq=simulation_freq_hz,
        ctrl_freq=control_freq_hz,
        gui=gui,
        record=record_video,
        obstacles=obstacles,
        user_debug_gui=user_debug_gui
    )
    
    # Get PyBullet client ID
    PYB_CLIENT = env.getPyBulletClient()
    
    # Initialize the logger
    logger = Logger(
        logging_freq_hz=control_freq_hz,
        num_drones=num_drones,
        output_folder=output_folder
    )
    
    # Initialize the controller
    ctrl = DSLPIDControl(drone_model=drone)
    
    # Initialize action array (control inputs)
    action = np.zeros((num_drones, 4))
    
    # Run the simulation
    START = time.time()
    print("Starting simulation...")
    
    # Calculate the actual number of steps based on duration and control frequency
    total_steps = int(duration_sec * env.CTRL_FREQ)
    # If we have fewer waypoints than steps, we'll loop through the trajectory
    
    for i in range(total_steps):
        # Step the simulation
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Get the current waypoint index, looping if necessary
        current_wp_idx = wp_counter % len(TARGET_POS)
        
        # Compute control for the current waypoint
        action[0, :], _, _ = ctrl.computeControlFromState(
            control_timestep=env.CTRL_TIMESTEP,
            state=obs[0],
            target_pos=TARGET_POS[current_wp_idx],
            target_rpy=INIT_RPYS[0, :]
        )
        
        # Move to the next waypoint (loop back to start when we reach the end)
        wp_counter = (wp_counter + 1) % len(TARGET_POS)
        
        # Log the simulation data
        logger.log(
            drone=0,
            timestamp=i / env.CTRL_FREQ,
            state=obs[0],
            control=np.hstack([TARGET_POS[current_wp_idx], INIT_RPYS[0, :], np.zeros(6)])
        )
        
        # Render the simulation
        env.render()
        
        # Sync simulation timing with real-time if GUI is enabled
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)
    
    # Close the environment
    env.close()
    
    # Save the simulation results
    logger.save()
    logger.save_as_csv("drone_trajectory")  # Save as CSV
    
    # Plot the simulation results
    if plot:
        logger.plot()
        
    print("Simulation complete!")

if __name__ == "__main__":
    # Check for --create_example_csv flag BEFORE parsing arguments
    if '--create_example_csv' in sys.argv:
        # Find the example path if specified
        csv_path = DEFAULT_EXAMPLE_CSV_PATH
        for i, arg in enumerate(sys.argv):
            if arg == '--example_csv_path' and i+1 < len(sys.argv):
                csv_path = sys.argv[i+1]
                break
        
        # Create the example CSV file
        create_example_trajectory_csv(csv_path)
        print("Example CSV created. Exiting.")
        exit()
    
    # Parse command line arguments for normal operation
    parser = argparse.ArgumentParser(description='Single drone flight simulation with custom trajectories')
    parser.add_argument('--drone', default=DEFAULT_DRONE, type=DroneModel, 
                        help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--physics', default=DEFAULT_PHYSICS, type=Physics, 
                        help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--gui', default=DEFAULT_GUI, type=str2bool, 
                        help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video', default=DEFAULT_RECORD_VIDEO, type=str2bool, 
                        help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot', default=DEFAULT_PLOT, type=str2bool, 
                        help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui', default=DEFAULT_USER_DEBUG_GUI, type=str2bool, 
                        help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--obstacles', default=DEFAULT_OBSTACLES, type=str2bool, 
                        help='Whether to add obstacles to the environment (default: False)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ, type=int, 
                        help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz', default=DEFAULT_CONTROL_FREQ_HZ, type=int, 
                        help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec', default=DEFAULT_DURATION_SEC, type=int, 
                        help='Duration of the simulation in seconds (default: 12)', metavar='')
    parser.add_argument('--output_folder', default=DEFAULT_OUTPUT_FOLDER, type=str, 
                        help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--trajectory_source', default=DEFAULT_TRAJECTORY_SOURCE, type=str,
                        choices=['built_in', 'csv', 'function'],
                        help='Source of the trajectory (built_in, csv, function)', metavar='')
    parser.add_argument('--trajectory_function', default=DEFAULT_TRAJECTORY_FUNCTION, type=str,
                        choices=['circle', 'figure8', 'spiral'],
                        help='Trajectory function to use (circle, figure8, spiral)', metavar='')
    parser.add_argument('--trajectory_file', default=DEFAULT_TRAJECTORY_FILE, type=str,
                        help='Path to CSV file containing the trajectory', metavar='')
    
    ARGS = parser.parse_args()
    
    # Run the simulation with parsed arguments
    run(**vars(ARGS))