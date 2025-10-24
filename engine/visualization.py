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

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title(title)

    # Display terrain heightmap
    terrain_plot = ax.imshow(heightmap.T, cmap='gist_earth', origin='lower', vmin=0, vmax=terrain.shape[2])
    fig.colorbar(terrain_plot, ax=ax, label="Terrain Height", shrink=0.8, aspect=20)

    # Overlay water depth
    masked_water = np.ma.masked_where(water_depth <= 0.1, water_depth)
    water_plot = ax.imshow(masked_water.T, cmap='Blues', origin='lower', alpha=0.7, vmin=0)
    fig.colorbar(water_plot, ax=ax, label="Water Depth", shrink=0.8, aspect=20)

    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")

    # Save the figure instead of showing it directly
    # This is better for non-interactive environments
    filename = "world_view.png"
    plt.savefig(filename)
    print(f"Visualization saved to {filename}")
    plt.close() # Close the plot to free memory