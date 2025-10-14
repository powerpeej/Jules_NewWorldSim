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
    """Runs one iteration of the water physics simulation."""
    dims = terrain.shape
    max_water_per_cell = 1.0
    
    # --- Downward Flow ---
    for z in range(dims[2] - 1):
        can_flow_down = (water[:, :, z+1] > 0) & (terrain[:, :, z] == 0)
        free_space_below = max_water_per_cell - water[:, :, z]
        water_to_move = np.minimum(water[:, :, z+1], free_space_below)
        flow_amount = np.where(can_flow_down, water_to_move, 0)
        water[:, :, z+1] -= flow_amount
        water[:, :, z] += flow_amount

    # --- Horizontal Flow ---
    # A copy is needed to avoid cascading updates within a single step
    water_copy = np.copy(water)
    for x in range(1, dims[0]-1):
        for y in range(1, dims[1]-1):
            for z in range(dims[2]):
                if water_copy[x, y, z] > 0.001 and terrain[x,y,z] == 0:
                    neighbors = []
                    if terrain[x-1, y, z] == 0: neighbors.append((x-1, y, z))
                    if terrain[x+1, y, z] == 0: neighbors.append((x+1, y, z))
                    if terrain[x, y-1, z] == 0: neighbors.append((x, y-1, z))
                    if terrain[x, y+1, z] == 0: neighbors.append((x, y+1, z))
                    
                    if not neighbors:
                        continue

                    total_water = water_copy[x, y, z]
                    for nx, ny, nz in neighbors:
                        total_water += water_copy[nx, ny, nz]
                    
                    all_cells = neighbors + [(x, y, z)]
                    avg_water = total_water / len(all_cells)
                    
                    # Distribute the water in the original array
                    for nx, ny, nz in all_cells:
                        water[nx, ny, nz] = min(avg_water, max_water_per_cell)


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