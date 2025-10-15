import numpy as np
import matplotlib.pyplot as plt

def setup_environment(dims=(64, 64, 64)):
    """Initializes the terrain and water arrays."""
    print(f"Setting up environment with dimensions {dims}...")
    terrain = np.zeros(dims, dtype=np.int8)
    water = np.zeros(dims, dtype=np.float32)
    return terrain, water

def generate_terrain_mathematical(terrain_array):
    """Generates a mathematical valley without using a noise library."""
    print("Generating mathematical terrain (opensimplex not found)...")
    dims = terrain_array.shape
    center_x, center_y = dims[0] / 2, dims[1] / 2
    
    for x in range(dims[0]):
        for y in range(dims[1]):
            # Calculate distance from the center
            dist_x = (x - center_x) ** 2
            dist_y = (y - center_y) ** 2
            
            # Create a parabolic valley shape
            height = int((dist_x + dist_y) / (center_x**2 + center_y**2) * (dims[2] / 2)) + 5
            
            # Ensure height is within bounds
            height = min(height, dims[2] - 1)
            
            # Fill terrain from the bottom up to the calculated height
            terrain_array[x, y, :height] = 1 # 1 = rock
    print("Terrain generation complete.")
    return terrain_array

def simulation_step(terrain, water):
    """
    Runs one iteration of the water physics simulation using a vectorized approach
    for more realistic and efficient horizontal water flow.
    """
    dims = terrain.shape
    max_water_per_cell = 1.0
    flow_speed = 0.5  # Controls how quickly water equalizes per step
    epsilon = 1e-5    # Small constant to prevent division by zero

    # --- Downward Flow (Gravity) ---
    # Water flows from a cell to the one directly below it if it's not rock.
    # This loop is necessary to simulate gravity layer by layer correctly.
    for z in range(dims[2] - 1):
        water_above = water[:, :, z + 1]
        can_flow_down = (water_above > 0) & (terrain[:, :, z] == 0)
        free_space_below = max_water_per_cell - water[:, :, z]
        water_to_move = np.minimum(water_above, free_space_below)
        flow_amount = np.where(can_flow_down, water_to_move, 0)
        water[:, :, z + 1] -= flow_amount
        water[:, :, z] += flow_amount

    # --- Horizontal Flow (Pressure) ---
    # Water flows from cells with more water to adjacent cells with less.
    # This process is vectorized for performance.
    water_old = np.copy(water)

    # Calculate potential flow to each of the 4 horizontal neighbors
    # Flow is calculated based on the difference in water levels.

    # Calculate differences with neighbors (positive means current cell is higher)
    # Using np.roll and then correcting for boundaries is a clean way to handle this.
    diff_N = water_old - np.roll(water_old, 1, axis=1)
    diff_S = water_old - np.roll(water_old, -1, axis=1)
    diff_W = water_old - np.roll(water_old, 1, axis=0)
    diff_E = water_old - np.roll(water_old, -1, axis=0)

    # Calculate flow only where the current cell is higher and the neighbor is not terrain
    flow_N = np.maximum(0, diff_N) * (terrain == 0) * (np.roll(terrain, 1, axis=1) == 0)
    flow_S = np.maximum(0, diff_S) * (terrain == 0) * (np.roll(terrain, -1, axis=1) == 0)
    flow_W = np.maximum(0, diff_W) * (terrain == 0) * (np.roll(terrain, 1, axis=0) == 0)
    flow_E = np.maximum(0, diff_E) * (terrain == 0) * (np.roll(terrain, -1, axis=0) == 0)

    # Correct for wrap-around at boundaries
    flow_N[:, 0, :] = 0
    flow_S[:, -1, :] = 0
    flow_W[0, :, :] = 0
    flow_E[-1, :, :] = 0

    # Sum of all potential outgoing flows from each cell
    total_flow_out = flow_N + flow_S + flow_W + flow_E

    # Normalize flows to prevent a cell from giving away more water than it has
    scale_factor = np.divide(water_old, total_flow_out, out=np.ones_like(water_old), where=total_flow_out > epsilon)
    scale_factor = np.minimum(1.0, scale_factor)

    # Apply the scaling factor and flow speed
    flow_N *= scale_factor * flow_speed
    flow_S *= scale_factor * flow_speed
    flow_W *= scale_factor * flow_speed
    flow_E *= scale_factor * flow_speed

    # Calculate the net change in water for each cell
    delta_water = np.zeros_like(water)

    # Subtract outflows
    delta_water -= (flow_N + flow_S + flow_W + flow_E)

    # Add inflows from neighbors (by rolling the outflow arrays)
    delta_water += np.roll(flow_S, 1, axis=1)  # Inflow from North is neighbor's flow_S
    delta_water += np.roll(flow_N, -1, axis=1) # Inflow from South is neighbor's flow_N
    delta_water += np.roll(flow_E, 1, axis=0)  # Inflow from West is neighbor's flow_E
    delta_water += np.roll(flow_W, -1, axis=0) # Inflow from East is neighbor's flow_W

    # Update the water array
    water += delta_water


def visualize_world(terrain, water):
    """Creates and displays a 2D top-down visualization of the world."""
    print("Visualizing world...")
    
    heightmap = np.zeros((terrain.shape[0], terrain.shape[1]))
    for x in range(terrain.shape[0]):
        for y in range(terrain.shape[1]):
            rock_indices = np.where(terrain[x, y, :] == 1)[0]
            if len(rock_indices) > 0:
                heightmap[x, y] = rock_indices.max()
            else:
                heightmap[x, y] = 0

    water_depth = np.sum(water, axis=2)
    
    plt.figure(figsize=(10, 8))
    plt.title("World Simulation State - Dam Test")
    
    plt.imshow(heightmap.T, cmap='gray', origin='lower', vmin=0, vmax=terrain.shape[2])
    
    masked_water = np.ma.masked_where(water_depth <= 0.01, water_depth)
    plt.imshow(masked_water.T, cmap='Blues', origin='lower', alpha=0.7, vmin=0)
    
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.colorbar(label="Terrain Height")
    plt.show()

# --- Execute the Test Scenario ---
WORLD_DIMS = (64, 64, 64)

# 1. Setup
terrain, water = setup_environment(WORLD_DIMS)

# 2. Generate Terrain
terrain = generate_terrain_mathematical(terrain)

# 3. Build a Dam
print("Building a dam...")
dam_x = WORLD_DIMS[0] // 2
dam_y_start, dam_y_end = 10, WORLD_DIMS[1] - 10
# Find the ground height along the dam's path
ground_height = 0
for y in range(dam_y_start, dam_y_end):
    rock_indices = np.where(terrain[dam_x, y, :] == 1)[0]
    if len(rock_indices) > 0:
        ground_height = max(ground_height, rock_indices.max())

dam_height = ground_height + 5 # Build dam 5 units high
terrain[dam_x-1:dam_x+1, dam_y_start:dam_y_end, :dam_height] = 1

# 4. Add Water Source
print("Adding water source...")
# Place water upstream of the dam
water[WORLD_DIMS[0] // 4, WORLD_DIMS[1] // 2, WORLD_DIMS[2] - 2] = 2000.0

# 5. Run Simulation
num_iterations = 100
print(f"Running simulation for {num_iterations} iterations...")
for i in range(num_iterations):
    if (i + 1) % 10 == 0:
        print(f"  ...Iteration {i+1}/{num_iterations}")
    simulation_step(terrain, water)
print("Simulation complete.")

# 6. Display Final Result
visualize_world(terrain, water)