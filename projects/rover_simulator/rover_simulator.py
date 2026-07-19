import math
import time

import cv2
import numpy as np


WINDOW = "Module 2 - Landmark Sensor Fusion"
WORLD_W, WORLD_H = 700, 600
PANEL_W = 430

START_X, START_Y, START_HEADING = 350.0, 300.0, 0.0

# True physical pose
true_x, true_y, true_heading = START_X, START_Y, START_HEADING

# Raw wheel-odometry estimate
odom_x, odom_y, odom_heading = START_X, START_Y, START_HEADING

# Corrected estimate: odometry plus landmark observations
fused_x, fused_y, fused_heading = START_X, START_Y, START_HEADING

ROVER_RADIUS = 20
MANUAL_SPEED = 180.0
AUTO_SPEED = 115.0
TURN_SPEED = 145.0

# Simulated wheel-encoder noise
LINEAR_NOISE_PERCENT = 0.04
LINEAR_NOISE_OFFSET = 2.5
ANGULAR_NOISE = 3.0

# Simulated 360-degree LiDAR
LIDAR_MAX_RANGE = 240
LIDAR_ANGLE_STEP = 3
LIDAR_DISTANCE_STEP = 3

# Known landmarks, similar to mapped AprilTags
LANDMARKS = [
    (1, 55.0, 300.0),
    (2, 350.0, 45.0),
    (3, 650.0, 300.0),
    (4, 350.0, 555.0),
]

LANDMARK_MAX_RANGE = 285.0
LANDMARK_UPDATE_INTERVAL = 0.25
LANDMARK_RANGE_NOISE = 3.0
LANDMARK_BEARING_NOISE = 1.4
LANDMARK_HEADING_NOISE = 1.8

# Strength of each landmark correction
POSITION_GAIN = 0.35
HEADING_GAIN = 0.30

# Occupancy map
MAP_CELL = 10
MAP_COLS = WORLD_W // MAP_CELL
MAP_ROWS = WORLD_H // MAP_CELL

UNKNOWN = -1
FREE = 0
OCCUPIED = 1

occupancy_map = np.full(
    (MAP_ROWS, MAP_COLS),
    UNKNOWN,
    dtype=np.int8,
)

# Obstacles: left, top, right, bottom
OBSTACLES = [
    (70, 80, 230, 150),
    (510, 70, 640, 240),
    (120, 400, 300, 500),
    (420, 340, 560, 520),
    (300, 190, 410, 250),
]

random_generator = np.random.default_rng()

landmark_corrections = True
visible_landmarks = []
localization_status = "WAITING FOR LANDMARK"


def normalize_angle(angle):
    """Normalize an angle to -180 through 180 degrees."""
    return (angle + 180) % 360 - 180


def point_outside(x, y):
    """Check whether a point is outside the world."""
    return (
        x < 0
        or x >= WORLD_W
        or y < 0
        or y >= WORLD_H
    )


def point_hits_obstacle(x, y):
    """Check whether a point is inside an obstacle."""
    return any(
        left <= x <= right
        and top <= y <= bottom
        for left, top, right, bottom in OBSTACLES
    )


def rover_collides(x, y):
    """Check the rover body against all obstacles."""
    for left, top, right, bottom in OBSTACLES:
        closest_x = max(left, min(x, right))
        closest_y = max(top, min(y, bottom))

        if (
            (x - closest_x) ** 2
            + (y - closest_y) ** 2
            <= ROVER_RADIUS**2
        ):
            return True

    return False


def rover_outside(x, y):
    """Check whether the rover would leave the world."""
    return (
        x - ROVER_RADIUS < 0
        or x + ROVER_RADIUS >= WORLD_W
        or y - ROVER_RADIUS < 0
        or y + ROVER_RADIUS >= WORLD_H
    )


def line_of_sight_clear(
    start_x,
    start_y,
    end_x,
    end_y,
):
    """Check whether an obstacle blocks a landmark."""
    distance = math.hypot(
        end_x - start_x,
        end_y - start_y,
    )

    steps = max(
        1,
        int(distance // 4),
    )

    for step in range(1, steps):
        fraction = step / steps

        sample_x = (
            start_x
            + (end_x - start_x) * fraction
        )

        sample_y = (
            start_y
            + (end_y - start_y) * fraction
        )

        if point_hits_obstacle(
            sample_x,
            sample_y,
        ):
            return False

    return True


def lidar_scan():
    """Generate LiDAR readings from the true pose."""
    results = []

    for relative_angle in range(
        0,
        360,
        LIDAR_ANGLE_STEP,
    ):
        relative_angle = normalize_angle(
            float(relative_angle)
        )

        scan_heading = (
            true_heading
            + relative_angle
        )

        radians = math.radians(
            scan_heading
        )

        measured_distance = float(
            LIDAR_MAX_RANGE
        )

        hit = False

        endpoint = (
            int(true_x),
            int(true_y),
        )

        for distance in range(
            0,
            LIDAR_MAX_RANGE + 1,
            LIDAR_DISTANCE_STEP,
        ):
            scan_x = (
                true_x
                + math.cos(radians) * distance
            )

            scan_y = (
                true_y
                - math.sin(radians) * distance
            )

            if point_outside(scan_x, scan_y):
                measured_distance = float(
                    distance
                )

                hit = True

                endpoint = (
                    int(
                        max(
                            0,
                            min(
                                WORLD_W - 1,
                                scan_x,
                            ),
                        )
                    ),
                    int(
                        max(
                            0,
                            min(
                                WORLD_H - 1,
                                scan_y,
                            ),
                        )
                    ),
                )

                break

            if point_hits_obstacle(
                scan_x,
                scan_y,
            ):
                measured_distance = float(
                    distance
                )

                hit = True

                endpoint = (
                    int(scan_x),
                    int(scan_y),
                )

                break

            endpoint = (
                int(scan_x),
                int(scan_y),
            )

        results.append(
            (
                relative_angle,
                measured_distance,
                endpoint,
                hit,
            )
        )

    return results


def sector_distance(
    scan,
    minimum_angle,
    maximum_angle,
):
    """Return the closest LiDAR reading in a sector."""
    distances = [
        distance
        for (
            relative_angle,
            distance,
            _,
            _,
        ) in scan
        if (
            minimum_angle
            <= relative_angle
            <= maximum_angle
        )
    ]

    if not distances:
        return float(LIDAR_MAX_RANGE)

    return min(distances)


def update_true_pose(
    linear_velocity,
    angular_velocity,
    delta_time,
):
    """Move the true physical rover."""
    global true_x
    global true_y
    global true_heading

    true_heading += (
        angular_velocity
        * delta_time
    )

    heading_radians = math.radians(
        true_heading
    )

    proposed_x = (
        true_x
        + math.cos(heading_radians)
        * linear_velocity
        * delta_time
    )

    proposed_y = (
        true_y
        - math.sin(heading_radians)
        * linear_velocity
        * delta_time
    )

    movement_blocked = (
        rover_collides(
            proposed_x,
            proposed_y,
        )
        or rover_outside(
            proposed_x,
            proposed_y,
        )
    )

    if movement_blocked:
        return True, 0.0

    true_x = proposed_x
    true_y = proposed_y

    return False, linear_velocity


def integrate_noisy_odometry(
    actual_linear_velocity,
    angular_velocity,
    delta_time,
):
    """Integrate noisy wheel measurements."""
    global odom_x
    global odom_y
    global odom_heading

    global fused_x
    global fused_y
    global fused_heading

    measured_linear_velocity = (
        actual_linear_velocity
        * (
            1.0
            + random_generator.normal(
                0.0,
                LINEAR_NOISE_PERCENT,
            )
        )
        + random_generator.normal(
            0.0,
            LINEAR_NOISE_OFFSET,
        )
    )

    measured_angular_velocity = (
        angular_velocity
        + random_generator.normal(
            0.0,
            ANGULAR_NOISE,
        )
    )

    # Raw odometry estimate
    odom_heading += (
        measured_angular_velocity
        * delta_time
    )

    odom_radians = math.radians(
        odom_heading
    )

    odom_x += (
        math.cos(odom_radians)
        * measured_linear_velocity
        * delta_time
    )

    odom_y -= (
        math.sin(odom_radians)
        * measured_linear_velocity
        * delta_time
    )

    # Fused estimate begins with the same odometry
    fused_heading += (
        measured_angular_velocity
        * delta_time
    )

    fused_radians = math.radians(
        fused_heading
    )

    fused_x += (
        math.cos(fused_radians)
        * measured_linear_velocity
        * delta_time
    )

    fused_y -= (
        math.sin(fused_radians)
        * measured_linear_velocity
        * delta_time
    )


def circular_mean(angles):
    """Average headings correctly around 0 and 360 degrees."""
    sine_sum = sum(
        math.sin(math.radians(angle))
        for angle in angles
    )

    cosine_sum = sum(
        math.cos(math.radians(angle))
        for angle in angles
    )

    return math.degrees(
        math.atan2(
            sine_sum,
            cosine_sum,
        )
    )


def observe_landmarks():
    """Generate noisy observations of visible landmarks."""
    observations = []

    for landmark_id, landmark_x, landmark_y in LANDMARKS:
        true_range = math.hypot(
            landmark_x - true_x,
            landmark_y - true_y,
        )

        if true_range > LANDMARK_MAX_RANGE:
            continue

        if not line_of_sight_clear(
            true_x,
            true_y,
            landmark_x,
            landmark_y,
        ):
            continue

        absolute_bearing = math.degrees(
            math.atan2(
                true_y - landmark_y,
                landmark_x - true_x,
            )
        )

        relative_bearing = normalize_angle(
            absolute_bearing
            - true_heading
        )

        measured_range = max(
            0.0,
            true_range
            + random_generator.normal(
                0.0,
                LANDMARK_RANGE_NOISE,
            ),
        )

        measured_bearing = (
            relative_bearing
            + random_generator.normal(
                0.0,
                LANDMARK_BEARING_NOISE,
            )
        )

        measured_heading = (
            true_heading
            + random_generator.normal(
                0.0,
                LANDMARK_HEADING_NOISE,
            )
        )

        observations.append(
            {
                "id": landmark_id,
                "x": landmark_x,
                "y": landmark_y,
                "range": measured_range,
                "bearing": measured_bearing,
                "heading": measured_heading,
            }
        )

    return observations


def apply_landmark_correction(
    observations,
):
    """Correct the fused pose using known landmarks."""
    global fused_x
    global fused_y
    global fused_heading

    global visible_landmarks
    global localization_status

    visible_landmarks = [
        observation["id"]
        for observation in observations
    ]

    if not landmark_corrections:
        localization_status = (
            "CORRECTIONS DISABLED"
        )
        return

    if not observations:
        localization_status = (
            "NO LANDMARK VISIBLE"
        )
        return

    observed_heading = circular_mean(
        [
            observation["heading"]
            for observation in observations
        ]
    )

    x_estimates = []
    y_estimates = []

    for observation in observations:
        absolute_bearing = (
            observed_heading
            + observation["bearing"]
        )

        bearing_radians = math.radians(
            absolute_bearing
        )

        estimated_rover_x = (
            observation["x"]
            - math.cos(bearing_radians)
            * observation["range"]
        )

        estimated_rover_y = (
            observation["y"]
            + math.sin(bearing_radians)
            * observation["range"]
        )

        x_estimates.append(
            estimated_rover_x
        )

        y_estimates.append(
            estimated_rover_y
        )

    observed_x = float(
        np.mean(x_estimates)
    )

    observed_y = float(
        np.mean(y_estimates)
    )

    fused_x += (
        observed_x - fused_x
    ) * POSITION_GAIN

    fused_y += (
        observed_y - fused_y
    ) * POSITION_GAIN

    heading_error = normalize_angle(
        observed_heading
        - fused_heading
    )

    fused_heading += (
        heading_error
        * HEADING_GAIN
    )

    localization_status = (
        f"CORRECTED WITH "
        f"{len(observations)} LANDMARK(S)"
    )


def map_cell(x, y):
    """Convert world coordinates into a map cell."""
    if point_outside(x, y):
        return None

    return (
        int(x // MAP_CELL),
        int(y // MAP_CELL),
    )


def mark_free(x, y):
    """Mark a map cell as free."""
    cell = map_cell(x, y)

    if cell is None:
        return

    column, row = cell

    if occupancy_map[row, column] != OCCUPIED:
        occupancy_map[row, column] = FREE


def mark_occupied(x, y):
    """Mark a map cell as occupied."""
    cell = map_cell(x, y)

    if cell is None:
        return

    column, row = cell
    occupancy_map[row, column] = OCCUPIED


def update_map(scan):
    """Place LiDAR measurements using the fused pose."""
    for (
        relative_angle,
        distance,
        _,
        hit,
    ) in scan:
        scan_heading = (
            fused_heading
            + relative_angle
        )

        radians = math.radians(
            scan_heading
        )

        for ray_distance in range(
            0,
            int(distance),
            LIDAR_DISTANCE_STEP,
        ):
            map_x = (
                fused_x
                + math.cos(radians)
                * ray_distance
            )

            map_y = (
                fused_y
                - math.sin(radians)
                * ray_distance
            )

            mark_free(
                map_x,
                map_y,
            )

        if hit:
            occupied_x = (
                fused_x
                + math.cos(radians)
                * distance
            )

            occupied_y = (
                fused_y
                - math.sin(radians)
                * distance
            )

            mark_occupied(
                occupied_x,
                occupied_y,
            )


def draw_grid(image):
    """Draw the world grid."""
    for x in range(0, WORLD_W, 50):
        cv2.line(
            image,
            (x, 0),
            (x, WORLD_H),
            (55, 55, 55),
            1,
        )

    for y in range(0, WORLD_H, 50):
        cv2.line(
            image,
            (0, y),
            (WORLD_W, y),
            (55, 55, 55),
            1,
        )


def draw_obstacles(image):
    """Draw physical obstacles."""
    for left, top, right, bottom in OBSTACLES:
        cv2.rectangle(
            image,
            (left, top),
            (right, bottom),
            (105, 105, 105),
            -1,
        )

        cv2.rectangle(
            image,
            (left, top),
            (right, bottom),
            (220, 220, 220),
            2,
        )


def draw_landmarks(image):
    """Draw landmarks and visibility lines."""
    true_center = (
        int(true_x),
        int(true_y),
    )

    for landmark_id, x, y in LANDMARKS:
        center = (
            int(x),
            int(y),
        )

        is_visible = (
            landmark_id
            in visible_landmarks
        )

        color = (
            (255, 255, 0)
            if is_visible
            else (255, 120, 0)
        )

        cv2.rectangle(
            image,
            (
                center[0] - 9,
                center[1] - 9,
            ),
            (
                center[0] + 9,
                center[1] + 9,
            ),
            color,
            -1,
        )

        cv2.putText(
            image,
            f"L{landmark_id}",
            (
                center[0] - 13,
                center[1] - 14,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
        )

        if is_visible:
            cv2.line(
                image,
                true_center,
                center,
                (255, 255, 0),
                1,
            )


def draw_lidar(image, scan):
    """Draw a subset of the LiDAR rays."""
    center = (
        int(true_x),
        int(true_y),
    )

    for index, (
        _,
        distance,
        endpoint,
        _,
    ) in enumerate(scan):
        if index % 3 != 0:
            continue

        if distance < 60:
            color = (0, 0, 255)

        elif distance < 130:
            color = (0, 255, 255)

        else:
            color = (0, 180, 0)

        cv2.line(
            image,
            center,
            endpoint,
            color,
            1,
        )


def draw_pose(
    image,
    x,
    y,
    heading,
    color,
    radius,
):
    """Draw an estimated rover pose."""
    center = (
        int(x),
        int(y),
    )

    cv2.circle(
        image,
        center,
        radius,
        color,
        2,
    )

    heading_radians = math.radians(
        heading
    )

    nose = (
        int(
            x
            + math.cos(heading_radians)
            * 40
        ),
        int(
            y
            - math.sin(heading_radians)
            * 40
        ),
    )

    cv2.line(
        image,
        center,
        nose,
        color,
        2,
    )


def draw_true_rover(image):
    """Draw the true physical rover."""
    center = (
        int(true_x),
        int(true_y),
    )

    cv2.circle(
        image,
        center,
        ROVER_RADIUS,
        (0, 200, 255),
        -1,
    )

    cv2.circle(
        image,
        center,
        ROVER_RADIUS,
        (255, 255, 255),
        2,
    )

    heading_radians = math.radians(
        true_heading
    )

    nose = (
        int(
            true_x
            + math.cos(heading_radians)
            * 36
        ),
        int(
            true_y
            - math.sin(heading_radians)
            * 36
        ),
    )

    cv2.line(
        image,
        center,
        nose,
        (0, 0, 255),
        4,
    )


def create_panel():
    """Render the corrected map and localization errors."""
    panel = np.full(
        (
            WORLD_H,
            PANEL_W,
            3,
        ),
        25,
        dtype=np.uint8,
    )

    map_image = np.zeros(
        (
            MAP_ROWS,
            MAP_COLS,
            3,
        ),
        dtype=np.uint8,
    )

    map_image[
        occupancy_map == UNKNOWN
    ] = (75, 75, 75)

    map_image[
        occupancy_map == FREE
    ] = (225, 225, 225)

    map_image[
        occupancy_map == OCCUPIED
    ] = (10, 10, 10)

    rendered_width = PANEL_W - 30

    rendered_height = int(
        rendered_width
        * WORLD_H
        / WORLD_W
    )

    enlarged_map = cv2.resize(
        map_image,
        (
            rendered_width,
            rendered_height,
        ),
        interpolation=cv2.INTER_NEAREST,
    )

    map_left = 15
    map_top = 205

    panel[
        map_top : map_top + rendered_height,
        map_left : map_left + rendered_width,
    ] = enlarged_map

    cv2.rectangle(
        panel,
        (map_left, map_top),
        (
            map_left + rendered_width,
            map_top + rendered_height,
        ),
        (255, 255, 255),
        2,
    )

    scale_x = (
        rendered_width
        / WORLD_W
    )

    scale_y = (
        rendered_height
        / WORLD_H
    )

    mapped_x = int(
        map_left
        + fused_x * scale_x
    )

    mapped_y = int(
        map_top
        + fused_y * scale_y
    )

    cv2.circle(
        panel,
        (
            mapped_x,
            mapped_y,
        ),
        6,
        (255, 255, 0),
        -1,
    )

    raw_position_error = math.hypot(
        true_x - odom_x,
        true_y - odom_y,
    )

    fused_position_error = math.hypot(
        true_x - fused_x,
        true_y - fused_y,
    )

    raw_heading_error = abs(
        normalize_angle(
            true_heading
            - odom_heading
        )
    )

    fused_heading_error = abs(
        normalize_angle(
            true_heading
            - fused_heading
        )
    )

    correction_text = (
        "ON"
        if landmark_corrections
        else "OFF"
    )

    cv2.putText(
        panel,
        "Landmark Sensor Fusion",
        (20, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        panel,
        f"Corrections: {correction_text}",
        (20, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.57,
        (
            (0, 255, 0)
            if landmark_corrections
            else (0, 0, 255)
        ),
        2,
    )

    cv2.putText(
        panel,
        f"Visible landmarks: {visible_landmarks}",
        (20, 99),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.51,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        panel,
        (
            f"Raw odometry error: "
            f"{raw_position_error:.1f} px"
        ),
        (20, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.51,
        (255, 0, 255),
        2,
    )

    cv2.putText(
        panel,
        (
            f"Fused position error: "
            f"{fused_position_error:.1f} px"
        ),
        (20, 161),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.51,
        (255, 255, 0),
        2,
    )

    cv2.putText(
        panel,
        (
            f"Heading error: raw "
            f"{raw_heading_error:.1f} | "
            f"fused {fused_heading_error:.1f}"
        ),
        (20, 192),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
    )

    return panel


cv2.namedWindow(WINDOW)

autonomous_mode = False
avoidance_direction = 0

previous_time = time.perf_counter()
last_landmark_update = 0.0

while True:
    current_time = time.perf_counter()

    delta_time = min(
        current_time - previous_time,
        0.05,
    )

    previous_time = current_time

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key == ord("m"):
        autonomous_mode = (
            not autonomous_mode
        )

        avoidance_direction = 0

    if key == ord("l"):
        landmark_corrections = (
            not landmark_corrections
        )

    if key == ord("c"):
        occupancy_map.fill(UNKNOWN)

    if key == ord("r"):
        true_x = START_X
        true_y = START_Y
        true_heading = START_HEADING

        odom_x = START_X
        odom_y = START_Y
        odom_heading = START_HEADING

        fused_x = START_X
        fused_y = START_Y
        fused_heading = START_HEADING

        autonomous_mode = False
        avoidance_direction = 0

        visible_landmarks = []

        localization_status = (
            "WAITING FOR LANDMARK"
        )

        occupancy_map.fill(UNKNOWN)

    scan = lidar_scan()

    front_distance = sector_distance(
        scan,
        -25,
        25,
    )

    left_distance = sector_distance(
        scan,
        25,
        100,
    )

    right_distance = sector_distance(
        scan,
        -100,
        -25,
    )

    linear_command = 0.0
    angular_command = 0.0
    controller_action = "MANUAL"

    if autonomous_mode:
        if (
            avoidance_direction == 0
            and front_distance < 70
        ):
            avoidance_direction = (
                1
                if left_distance >= right_distance
                else -1
            )

        if avoidance_direction != 0:
            angular_command = (
                avoidance_direction
                * TURN_SPEED
            )

            controller_action = (
                "TURN LEFT"
                if avoidance_direction > 0
                else "TURN RIGHT"
            )

            if front_distance > 125:
                avoidance_direction = 0

        else:
            linear_command = AUTO_SPEED

            controller_action = (
                "EXPLORE FORWARD"
            )

            if left_distance < 55:
                angular_command = (
                    -TURN_SPEED * 0.45
                )

                controller_action = (
                    "STEER RIGHT"
                )

            elif right_distance < 55:
                angular_command = (
                    TURN_SPEED * 0.45
                )

                controller_action = (
                    "STEER LEFT"
                )

    else:
        if key == ord("w"):
            linear_command = MANUAL_SPEED

        if key == ord("s"):
            linear_command = -MANUAL_SPEED

        if key == ord("a"):
            angular_command = TURN_SPEED

        if key == ord("d"):
            angular_command = -TURN_SPEED

    (
        collision_warning,
        actual_linear_velocity,
    ) = update_true_pose(
        linear_command,
        angular_command,
        delta_time,
    )

    integrate_noisy_odometry(
        actual_linear_velocity,
        angular_command,
        delta_time,
    )

    if (
        current_time - last_landmark_update
        >= LANDMARK_UPDATE_INTERVAL
    ):
        observations = observe_landmarks()

        apply_landmark_correction(
            observations
        )

        last_landmark_update = (
            current_time
        )

    updated_scan = lidar_scan()

    update_map(
        updated_scan
    )

    world = np.full(
        (
            WORLD_H,
            WORLD_W,
            3,
        ),
        30,
        dtype=np.uint8,
    )

    draw_grid(world)
    draw_obstacles(world)
    draw_landmarks(world)
    draw_lidar(world, updated_scan)

    # Raw odometry: magenta
    draw_pose(
        world,
        odom_x,
        odom_y,
        odom_heading,
        (255, 0, 255),
        ROVER_RADIUS + 7,
    )

    # Fused estimate: cyan
    draw_pose(
        world,
        fused_x,
        fused_y,
        fused_heading,
        (255, 255, 0),
        ROVER_RADIUS + 3,
    )

    draw_true_rover(world)

    mode_text = (
        "AUTONOMOUS EXPLORATION"
        if autonomous_mode
        else "MANUAL"
    )

    cv2.putText(
        world,
        (
            "W/S: Move  A/D: Turn  M: Explore  "
            "L: Corrections  C: Clear map  R: Reset  Q: Quit"
        ),
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.41,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        world,
        f"Mode: {mode_text}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.57,
        (
            (0, 255, 255)
            if autonomous_mode
            else (255, 255, 255)
        ),
        2,
    )

    cv2.putText(
        world,
        (
            f"Localization: "
            f"{localization_status}"
        ),
        (10, 92),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.51,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        world,
        (
            "Yellow = true  "
            "Magenta = raw odometry  "
            "Cyan = fused estimate"
        ),
        (10, 124),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
    )

    cv2.putText(
        world,
        f"Controller: {controller_action}",
        (10, 154),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        2,
    )

    if collision_warning:
        cv2.putText(
            world,
            "COLLISION BLOCKED",
            (10, 187),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (0, 0, 255),
            2,
        )

    display = np.hstack(
        (
            world,
            create_panel(),
        )
    )

    cv2.imshow(
        WINDOW,
        display,
    )


cv2.destroyAllWindows()