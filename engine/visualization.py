import numpy as np
import matplotlib.pyplot as plt

def visualize_world(terrain, water, title="World Simulation State"):
    """Creates and displays a 2D top-down visualization of the world."""
    print("Visualizing world...")

    heightmap = np.zeros((terrain.shape[0], terrain.shape[1]))
    # This loop is slow, but a vectorized approach is complex.
    # We can optimize this later if needed.
    for x in range(terrain.shape[0]):
        for y in range(terrain.shape[1]):
            rock_indices = np.where(terrain[x, y, :] == 1)[0]
            if len(rock_indices) > 0:
                heightmap[x, y] = rock_indices.max()
            else:
                heightmap[x, y] = 0

    water_depth = np.sum(water, axis=2)

    plt.figure(figsize=(10, 8))
    plt.title(title)

    # Display terrain heightmap
    plt.imshow(heightmap.T, cmap='gist_earth', origin='lower', vmin=0, vmax=terrain.shape[2])

    # Overlay water depth, masking cells with very little water
    masked_water = np.ma.masked_where(water_depth <= 0.01, water_depth)
    plt.imshow(masked_water.T, cmap='Blues', origin='lower', alpha=0.7, vmin=0)

    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.colorbar(label="Terrain Height")

    # Save the figure instead of showing it directly
    # This is better for non-interactive environments
    filename = "world_view.png"
    plt.savefig(filename)
    print(f"Visualization saved to {filename}")
    plt.close() # Close the plot to free memory