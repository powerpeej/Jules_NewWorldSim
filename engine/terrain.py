import numpy as np

def generate_terrain_mathematical(terrain_array):
    """Generates a mathematical valley without using a noise library."""
    print("Generating mathematical terrain...")
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