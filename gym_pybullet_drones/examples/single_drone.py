"""Single drone simulation using PyBullet and the gym_pybullet_drones library.

The drone follows a circular trajectory in the X-Y plane.
Control is managed by the PID implementation in DSLPIDControl.

To run:
    $ python single_drone.py
"""
import os
import time
import argparse
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

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
        output_folder=DEFAULT_OUTPUT_FOLDER
        ):
    """Run the simulation with a single drone following a circular trajectory."""
    
    # Setup parameters
    num_drones = 1  # We only need one drone
    
    # Initialize drone position and orientation
    H = 1.0         # Height of the drone
    R = 0.5         # Radius of the circular trajectory
    
    # Starting position [x, y, z] and orientation [roll, pitch, yaw]
    INIT_XYZS = np.array([[0, 0, H]])  # Start at position (0,0,H)
    INIT_RPYS = np.array([[0, 0, 0]])  # No initial rotation
    
    # Create circular trajectory
    PERIOD = 10                             # Time to complete one circle (seconds)
    NUM_WP = control_freq_hz * PERIOD      # Number of waypoints
    TARGET_POS = np.zeros((NUM_WP, 3))     # Array to store waypoints
    
    # Generate circular waypoints
    for i in range(NUM_WP):
        angle = (i / NUM_WP) * (2 * np.pi)
        TARGET_POS[i, :] = [R * np.cos(angle), R * np.sin(angle), H]
    
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
    
    for i in range(0, int(duration_sec * env.CTRL_FREQ)):
        # Step the simulation
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Compute control for the current waypoint
        action[0, :], _, _ = ctrl.computeControlFromState(
            control_timestep=env.CTRL_TIMESTEP,
            state=obs[0],
            target_pos=TARGET_POS[wp_counter, :],
            target_rpy=INIT_RPYS[0, :]
        )
        
        # Move to the next waypoint (loop back to start when we reach the end)
        wp_counter = wp_counter + 1 if wp_counter < (NUM_WP - 1) else 0
        
        # Log the simulation data
        logger.log(
            drone=0,
            timestamp=i / env.CTRL_FREQ,
            state=obs[0],
            control=np.hstack([TARGET_POS[wp_counter, :], INIT_RPYS[0, :], np.zeros(6)])
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
    logger.save_as_csv("single_drone")  # Save as CSV
    
    # Plot the simulation results
    if plot:
        logger.plot()
        
    print("Simulation complete!")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Single drone circular flight simulation')
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
    
    ARGS = parser.parse_args()
    
    # Run the simulation with parsed arguments
    run(**vars(ARGS))