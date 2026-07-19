import math

import cv2
import numpy as np
from ultralytics import YOLO


model = YOLO("yolo11n.pt")

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not open webcam.")


world_width = 700
world_height = 500

rover_x = world_width / 2
rover_y = world_height / 2
rover_angle = -90.0

maximum_speed = 4.0
desired_person_height = 300


while True:
    success, frame = camera.read()

    if not success:
        break

    frame_height, frame_width = frame.shape[:2]
    camera_center_x = frame_width // 2

    results = model.predict(
        frame,
        classes=[0],
        conf=0.5,
        verbose=False,
    )

    display_frame = frame.copy()
    boxes = results[0].boxes

    steering = 0
    throttle = 0
    command = "SEARCHING"

    if boxes is not None and len(boxes) > 0:
        best_index = int(boxes.conf.argmax())
        box = boxes[best_index]

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        target_x = (x1 + x2) // 2
        target_y = (y1 + y2) // 2

        person_height = y2 - y1

        horizontal_error = target_x - camera_center_x

        steering = int(
            (horizontal_error / (frame_width / 2)) * 100
        )

        steering = max(-100, min(100, steering))

        if abs(steering) <= 10:
            steering = 0

        distance_error = desired_person_height - person_height

        throttle = int(
            (distance_error / desired_person_height) * 100
        )

        throttle = max(-100, min(100, throttle))

        if abs(distance_error) < 30:
            throttle = 0

        if throttle > 0:
            command = "FOLLOWING FORWARD"
        elif throttle < 0:
            command = "TOO CLOSE - REVERSING"
        else:
            command = "DISTANCE LOCKED"

        cv2.rectangle(
            display_frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.circle(
            display_frame,
            (target_x, target_y),
            6,
            (0, 0, 255),
            -1,
        )

        cv2.line(
            display_frame,
            (camera_center_x, frame_height // 2),
            (target_x, target_y),
            (0, 255, 255),
            2,
        )

        cv2.putText(
            display_frame,
            f"PERSON HEIGHT: {person_height}px",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            display_frame,
            f"STEERING: {steering:+d}",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            display_frame,
            f"THROTTLE: {throttle:+d}",
            (20, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
        )

    speed = (throttle / 100) * maximum_speed

    rover_angle += steering * 0.015

    angle_radians = math.radians(rover_angle)

    rover_x += math.cos(angle_radians) * speed
    rover_y += math.sin(angle_radians) * speed

    rover_x = max(20, min(world_width - 20, rover_x))
    rover_y = max(20, min(world_height - 20, rover_y))

    world = np.zeros(
        (world_height, world_width, 3),
        dtype=np.uint8,
    )

    rover_center = (int(rover_x), int(rover_y))

    front_x = int(rover_x + math.cos(angle_radians) * 30)
    front_y = int(rover_y + math.sin(angle_radians) * 30)

    cv2.circle(
        world,
        rover_center,
        20,
        (0, 255, 0),
        -1,
    )

    cv2.line(
        world,
        rover_center,
        (front_x, front_y),
        (0, 0, 255),
        5,
    )

    cv2.putText(
        world,
        command,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        world,
        f"STEERING: {steering:+d}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        world,
        f"THROTTLE: {throttle:+d}",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    cv2.imshow("AI Person Following Camera", display_frame)
    cv2.imshow("Virtual Autonomous Rover", world)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()