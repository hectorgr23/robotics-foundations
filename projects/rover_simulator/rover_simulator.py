import math
import time

import cv2
import numpy as np


# Simulation window
WORLD_WIDTH = 900
WORLD_HEIGHT = 600

# Rover starting pose
rover_x = WORLD_WIDTH / 2
rover_y = WORLD_HEIGHT / 2
rover_heading = 0.0

# Rover movement settings
linear_speed = 180.0       # pixels per second
rotation_speed = 140.0     # degrees per second
rover_radius = 22


def draw_grid(world: np.ndarray, spacing: int = 50) -> None:
    """Draw a reference grid in the simulated world."""
    for x in range(0, WORLD_WIDTH, spacing):
        cv2.line(world, (x, 0), (x, WORLD_HEIGHT), (55, 55, 55), 1)

    for y in range(0, WORLD_HEIGHT, spacing):
        cv2.line(world, (0, y), (WORLD_WIDTH, y), (55, 55, 55), 1)


def draw_rover(
    world: np.ndarray,
    x: float,
    y: float,
    heading_degrees: float,
) -> None:
    """Draw the rover and its direction indicator."""
    center = (int(x), int(y))

    cv2.circle(world, center, rover_radius, (0, 200, 255), -1)
    cv2.circle(world, center, rover_radius, (255, 255, 255), 2)

    heading_radians = math.radians(heading_degrees)

    direction_x = int(x + math.cos(heading_radians) * 38)
    direction_y = int(y - math.sin(heading_radians) * 38)

    cv2.line(
        world,
        center,
        (direction_x, direction_y),
        (0, 0, 255),
        4,
    )


previous_time = time.perf_counter()

while True:
    current_time = time.perf_counter()
    delta_time = current_time - previous_time
    previous_time = current_time

    # Prevent a large jump if the program momentarily pauses
    delta_time = min(delta_time, 0.05)

    world = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 3), dtype=np.uint8)
    world[:] = (30, 30, 30)

    draw_grid(world)

    key = cv2.waitKey(1) & 0xFF

    moving_forward = key == ord("w")
    moving_backward = key == ord("s")
    turning_left = key == ord("a")
    turning_right = key == ord("d")

    if turning_left:
        rover_heading += rotation_speed * delta_time

    if turning_right:
        rover_heading -= rotation_speed * delta_time

    heading_radians = math.radians(rover_heading)

    if moving_forward:
        rover_x += math.cos(heading_radians) * linear_speed * delta_time
        rover_y -= math.sin(heading_radians) * linear_speed * delta_time

    if moving_backward:
        rover_x -= math.cos(heading_radians) * linear_speed * delta_time
        rover_y += math.sin(heading_radians) * linear_speed * delta_time

    # Keep the rover inside the simulated world
    rover_x = max(rover_radius, min(WORLD_WIDTH - rover_radius, rover_x))
    rover_y = max(rover_radius, min(WORLD_HEIGHT - rover_radius, rover_y))

    draw_rover(world, rover_x, rover_y, rover_heading)

    cv2.putText(
        world,
        "W/S: Move    A/D: Turn    Q: Quit",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        world,
        f"Position: ({rover_x:.1f}, {rover_y:.1f})",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        world,
        f"Heading: {rover_heading % 360:.1f} degrees",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    cv2.imshow("Module 2 - Rover Simulator", world)

    if key == ord("q"):
        break


cv2.destroyAllWindows()