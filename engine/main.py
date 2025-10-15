import numpy as np
from . import terrain as terrain_gen
from . import physics
from . import visualization

def setup_environment(dims=(64, 64, 64)):
    """Initializes the terrain and water arrays."""
    print(f"Setting up environment with dimensions {dims}...")
    terrain = np.zeros(dims, dtype=np.int8)
    water = np.zeros(dims, dtype=np.float32)
    return terrain, water

def run_simulation(terrain, water, iterations=100):
    """Runs the main simulation loop."""
    print(f"Running simulation for {iterations} iterations...")
    for i in range(iterations):
        if (i + 1) % 10 == 0:
            print(f"  ...Iteration {i+1}/{iterations}")
        physics.simulation_step(terrain, water)
    print("Simulation complete.")
    return terrain, water