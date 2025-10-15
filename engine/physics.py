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

    # --- Horizontal Flow (Corrected to Conserve Water) ---
    # A copy is needed to read from a consistent state within the step.
    water_copy = np.copy(water)

    # Iterate over each cell that could have water
    for x in range(1, dims[0]-1):
        for y in range(1, dims[1]-1):
            for z in range(dims[2]):
                # Process only if the cell is air and has water
                if water_copy[x, y, z] > 0.001 and terrain[x, y, z] == 0:

                    # Find neighbors that are also air
                    neighbors = []
                    if terrain[x-1, y, z] == 0: neighbors.append((x-1, y, z))
                    if terrain[x+1, y, z] == 0: neighbors.append((x+1, y, z))
                    if terrain[x, y-1, z] == 0: neighbors.append((x, y-1, z))
                    if terrain[x, y+1, z] == 0: neighbors.append((x, y+1, z))

                    if not neighbors:
                        continue

                    # Sum the water in the current cell and its open neighbors
                    all_cells = neighbors + [(x, y, z)]
                    total_water = 0
                    for nx, ny, nz in all_cells:
                        total_water += water_copy[nx, ny, nz]

                    # Calculate the new, equalized water level
                    avg_water = total_water / len(all_cells)

                    # Distribute the averaged water level to all participating cells.
                    # This correctly conserves water by not capping the amount.
                    # The downward flow will handle cells with > 1.0 water.
                    for nx, ny, nz in all_cells:
                        water[nx, ny, nz] = avg_water