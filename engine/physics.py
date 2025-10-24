import numpy as np

def simulation_step(terrain, water):
    """Runs one iteration of the water physics simulation."""
    dims = terrain.shape
    max_water_per_cell = 1.0

    # --- Downward Flow ---
    # Iterates from the top down, moving water into empty cells below.
    for z in range(dims[2] - 1, 0, -1):
        water_above = water[:, :, z]
        can_flow_down = (water_above > 0) & (terrain[:, :, z-1] == 0)
        free_space_below = max_water_per_cell - water[:, :, z-1]
        water_to_move = np.minimum(water_above, free_space_below)
        flow_amount = np.where(can_flow_down, water_to_move, 0)

        water[:, :, z] -= flow_amount
        water[:, :, z-1] += flow_amount

    # --- Horizontal Flow (Vectorized) ---
    # This implementation is significantly faster than the nested-loop version.
    # It calculates the flow between a cell and its four horizontal neighbors.

    # A copy is needed to read from a consistent state within the step.
    water_copy = np.copy(water)

    # Define masks for valid cells (air and containing water)
    is_air = terrain == 0
    has_water = water_copy > 0.001
    can_flow = is_air & has_water

    # Calculate total water in each cell and its open neighbors
    # and the number of open neighbors for averaging.
    total_water = np.copy(water_copy)
    neighbor_count = np.ones(water.shape) # Start with 1 for the cell itself

    # Right neighbor
    flow_potential = can_flow & np.roll(is_air, -1, axis=0)
    total_water += np.where(flow_potential, np.roll(water_copy, -1, axis=0), 0)
    neighbor_count += np.where(flow_potential, 1, 0)

    # Left neighbor
    flow_potential = can_flow & np.roll(is_air, 1, axis=0)
    total_water += np.where(flow_potential, np.roll(water_copy, 1, axis=0), 0)
    neighbor_count += np.where(flow_potential, 1, 0)

    # Back neighbor
    flow_potential = can_flow & np.roll(is_air, -1, axis=1)
    total_water += np.where(flow_potential, np.roll(water_copy, -1, axis=1), 0)
    neighbor_count += np.where(flow_potential, 1, 0)

    # Front neighbor
    flow_potential = can_flow & np.roll(is_air, 1, axis=1)
    total_water += np.where(flow_potential, np.roll(water_copy, 1, axis=1), 0)
    neighbor_count += np.where(flow_potential, 1, 0)

    # Avoid division by zero
    neighbor_count[neighbor_count == 0] = 1

    # Calculate the new, equalized water level
    avg_water = total_water / neighbor_count

    # Apply the averaged water level to the cells that can flow
    water[can_flow] = avg_water[can_flow]