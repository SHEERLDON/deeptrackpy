import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_lidar_pos(arena_size):
    return arena_size / 2, 0


def distance_to_line_projection(a, b, c):
    """
    Calculates distance from point c to the line formed by points a and b using projection.

    Args:
        a: Tuple (x, y) representing point a.
        b: Tuple (x, y) representing point b.
        c: Tuple (x, y) representing point c.

    Returns:
        The distance between point c and the line formed by points a and b.
    """

    # Direction vector
    direction_vector = (b[0] - a[0], b[1] - a[1])

    # Vector from c to a
    vector_ca = c[0] - a[0], c[1] - a[1]

    vector_ca = np.array(vector_ca)
    direction_vector = np.array(direction_vector)
    # Projection of vector_ca onto direction_vector
    projection_vector = np.dot(vector_ca, direction_vector) / \
        np.dot(direction_vector, direction_vector) * direction_vector

    # Distance between c and its projection on the line
    return np.sqrt(np.dot(vector_ca - projection_vector, vector_ca - projection_vector))


def generate_synthetic_dataset(num_steps, arena_size=10, num_objects=2, max_speed=2, radius=0.5):
    """
    Generates a synthetic dataset for 2D object tracking in laser data.

    Args:
        num_samples: Number of samples to generate.
        arena_size: Size of the arena (square).
        num_objects: Number of objects to simulate.
        max_speed: Maximum speed of objects.

    Returns:
        A dictionary containing the following keys:
            lidar_data: List of numpy arrays, each representing a single lidar scan.
            object_data: List of dictionaries, each containing information about the objects in the scan.
    """

    lidar_data = []
    object_data = []

    # Set lidar position
    lidar_x, lidar_y = get_lidar_pos(arena_size)

    # Initialize objects
    object_info = []
    for _ in range(num_objects):
        while True:
            start_x = np.random.rand() * arena_size
            start_y = np.random.rand() * arena_size
            speed = np.random.rand() * max_speed
            angle = np.random.rand() * 2 * np.pi  # Random direction

            new_object = {
                "x": start_x,
                "y": start_y,
                "radius": radius,
                "speed": speed,
                "angle": angle
            }
            # Check for overlap with existing objects
            if all(np.sqrt((obj["x"] - new_object["x"])**2 + (obj["y"] - new_object["y"])**2) >= obj["radius"] + new_object["radius"] for obj in object_info):
                object_info.append(new_object)
                break

    for _ in range(num_steps):
        # Simulate object movement
        for obj in object_info:
            obj["x"] += np.cos(obj["angle"]) * obj["speed"]
            obj["y"] += np.sin(obj["angle"]) * obj["speed"]
            # Check if the object is still within the arena, if not, reverse direction
            if not (0 <= obj["x"] <= arena_size) or not (0 <= obj["y"] <= arena_size):
                obj["angle"] += np.pi  # Reverse direction

        # Generate lidar data (simulated as range readings to points on object boundaries)
        fov = np.pi  # Field of view
        angles = np.linspace(-fov/2, fov/2, 180)  # 360 lidar points
        lidar_points = []
        for angle in angles:
            max_dist = arena_size  # Initialize with maximum distance
            ray_x = lidar_x + np.cos(angle) * arena_size
            ray_y = lidar_y + np.sin(angle) * arena_size
            for obj in object_info:
                # Calculate distace to object center
                distance_to_center = distance_to_line_projection(
                    (lidar_x, lidar_y), (ray_x, ray_y), (obj["x"], obj["y"]))

                # Check for intersection along ray direction
                if distance_to_center <= obj["radius"] and dx * np.cos(angle) + dy * np.sin(angle) > 0:
                    # The Distance
                    dist = distance_to_center - obj["radius"]
                    if dist < max_dist:
                        max_dist = dist
            lidar_points.append(max_dist)

        lidar_data.append(np.array(lidar_points))
        object_data.append(object_info)

    result = pd.DataFrame(
        {"lidar_data": lidar_data, "object_data": object_data})

    return result


def plot_lidar_and_objects(lidar_data, object_data, arena_size=10):
    fig, ax = plt.subplots()

    # Plot lidar position
    lidar_x, lidar_y = get_lidar_pos(arena_size)
    ax.plot([lidar_x], [lidar_y], 'ko')

    # Plot lidar data
    # angles = np.linspace(0, np.pi, len(lidar_data))
    # x = lidar_data * np.cos(angles) + lidar_x
    # y = lidar_data * np.sin(angles) + lidar_y
    # ax.plot(x, y, 'r.')

    # Plot objects
    for obj in object_data:
        circle = plt.Circle((obj["x"], obj["y"]),
                            obj["radius"], color='g', fill=True)
        ax.add_artist(circle)

    ax.set_xlim(-2, arena_size + 1)
    ax.set_ylim(-2, arena_size + 1)
    ax.set_aspect('equal', adjustable='datalim')

    plt.show()


# Generate the synthetic dataset
data = generate_synthetic_dataset(10, num_objects=10)
lidar_data = data["lidar_data"].iloc[0]  # Use the first lidar scan
object_data = data["object_data"].iloc[0]  # Use the first object data

print(
    f'One raw data sample: {data.iloc[0].lidar_data.shape}, {data.iloc[0].object_data}')
print(data.head())
# Plot the lidar data and objects
plot_lidar_and_objects(lidar_data, object_data)


# data = generate_synthetic_dataset(10)
