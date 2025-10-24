import numpy as np

def simulation_step(terrain, water):
    """
    Runs one iteration of the water physics simulation with a focus on
    conserving water during horizontal flow.
    """
    dims = terrain.shape
    max_water_per_cell = 1.0
    flow_coefficient = 0.5  # Controls how quickly water equalizes

    # --- Downward Flow ---
    # This part remains the same, as it correctly handles gravity.
    for z in range(dims[2] - 1, 0, -1):
        water_above = water[:, :, z]
        can_flow_down = (water_above > 0) & (terrain[:, :, z - 1] == 0)
        free_space_below = max_water_per_cell - water[:, :, z - 1]
        water_to_move = np.minimum(water_above, free_space_below)
        flow_amount = np.where(can_flow_down, water_to_move, 0)

        water[:, :, z] -= flow_amount
        water[:, :, z - 1] += flow_amount

    # --- Horizontal Flow (Pressure-Based and Conservative) ---
    # A copy is needed to calculate flow based on a consistent state
    water_copy = np.copy(water)

    # This array will accumulate the net change in water for each cell
    flow_delta = np.zeros_like(water, dtype=np.float32)

    # Iterate over each horizontal axis to calculate flow
    for axis in [0, 1]: # 0 for x-axis, 1 for y-axis
        # Positive direction flow (e.g., right or back)
        # Calculate the difference in water levels between adjacent cells
        delta_pos = water_copy - np.roll(water_copy, 1, axis=axis)

        # Determine where flow can occur
        # 1. Not into terrain
        # 2. Not out of terrain
        # 3. Water must flow from high to low (delta > 0)
        can_flow_pos = (terrain == 0) & (np.roll(terrain, 1, axis=axis) == 0) & (delta_pos > 0)

        # Calculate the amount of water to transfer
        flow = np.where(can_flow_pos, delta_pos * flow_coefficient / 2.0, 0)

        # Update the net change for the cells
        flow_delta -= flow
        flow_delta += np.roll(flow, -1, axis=axis)

    # Apply the accumulated net changes to the water array
    # This ensures that water is perfectly conserved.
    water += flow_delta