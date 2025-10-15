import numpy as np
from engine import main, terrain, visualization

def run_dam_test_scenario():
    """
    Sets up and runs the 'dam test' scenario to validate the simulation engine.
    """
    print("--- Starting Dam Test Scenario ---")

    # 1. Setup World
    WORLD_DIMS = (64, 64, 64)
    sim_terrain, sim_water = main.setup_environment(WORLD_DIMS)

    # 2. Generate Terrain
    sim_terrain = terrain.generate_terrain_mathematical(sim_terrain)

    # 3. Build a Dam
    print("Building a dam...")
    dam_x = WORLD_DIMS[0] // 2
    dam_y_start, dam_y_end = 10, WORLD_DIMS[1] - 10

    # Find the ground height along the dam's path to build on top of it
    ground_height = 0
    for y in range(dam_y_start, dam_y_end):
        rock_indices = np.where(sim_terrain[dam_x, y, :] == 1)[0]
        if len(rock_indices) > 0:
            ground_height = max(ground_height, rock_indices.max())

    dam_height = ground_height + 10 # Build dam 10 units high
    sim_terrain[dam_x-1:dam_x+1, dam_y_start:dam_y_end, :dam_height] = 1

    # 4. Add Water Source
    print("Adding water source...")
    water_x = WORLD_DIMS[0] // 4
    water_y = WORLD_DIMS[1] // 2
    water_z = WORLD_DIMS[2] - 10 # Start water high up
    # Add a large volume of water to a single cell
    sim_water[water_x, water_y, water_z] = 5000.0

    # 5. Run Simulation
    sim_terrain, sim_water = main.run_simulation(sim_terrain, sim_water, iterations=100)

    # 6. Display Final Result
    visualization.visualize_world(
        sim_terrain,
        sim_water,
        title="Final State: Dam Test Scenario"
    )

    print("--- Dam Test Scenario Complete ---")

if __name__ == "__main__":
    run_dam_test_scenario()