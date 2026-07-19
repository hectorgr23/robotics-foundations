"""Simple three-dimensional simulated drone."""

import time

from core.messages import (
    DroneState,
    Pose3D,
    Vector3,
    VelocityCommand,
)
from flight_interface.base import FlightInterface


class SimulatedDrone(FlightInterface):
    """Minimal simulated multirotor flight backend."""

    def __init__(self) -> None:
        """Initialize the drone on the ground."""

        self._state = DroneState(
            pose=Pose3D(
                x=0.0,
                y=0.0,
                z=0.0,
                yaw=0.0,
            ),
            velocity=Vector3(
                x=0.0,
                y=0.0,
                z=0.0,
            ),
            armed=False,
            airborne=False,
            timestamp=time.time(),
        )

    def arm(self) -> None:
        """Arm the simulated drone."""

        self._state.armed = True
        self._state.timestamp = time.time()

    def disarm(self) -> None:
        """Disarm only when the drone is on the ground."""

        if self._state.airborne:
            self._state.health_messages.append(
                "Disarm rejected: drone is airborne."
            )
            return

        self._state.armed = False
        self._state.timestamp = time.time()

    def takeoff(self, target_altitude: float) -> None:
        """Instantly place the simulated drone at an altitude."""

        if not self._state.armed:
            self._state.health_messages.append(
                "Takeoff rejected: drone is not armed."
            )
            return

        self._state.pose.z = max(
            0.0,
            target_altitude,
        )

        self._state.airborne = (
            self._state.pose.z > 0.0
        )

        self._state.timestamp = time.time()

    def land(self) -> None:
        """Land the simulated drone."""

        self._state.pose.z = 0.0

        self._state.velocity = Vector3(
            x=0.0,
            y=0.0,
            z=0.0,
        )

        self._state.airborne = False
        self._state.timestamp = time.time()

    def apply_velocity_command(
        self,
        command: VelocityCommand,
        delta_time: float,
    ) -> None:
        """Update the drone pose using a velocity command."""

        if not self._state.armed:
            self._state.health_messages.append(
                "Velocity command rejected: drone is not armed."
            )
            return

        if not self._state.airborne:
            self._state.health_messages.append(
                "Velocity command rejected: drone is not airborne."
            )
            return

        self._state.velocity = Vector3(
            x=command.velocity_x,
            y=command.velocity_y,
            z=command.velocity_z,
        )

        self._state.pose.x += (
            command.velocity_x * delta_time
        )

        self._state.pose.y += (
            command.velocity_y * delta_time
        )

        self._state.pose.z = max(
            0.0,
            self._state.pose.z
            + command.velocity_z * delta_time,
        )

        self._state.pose.yaw += (
            command.yaw_rate * delta_time
        )

        self._state.airborne = (
            self._state.pose.z > 0.0
        )

        self._state.timestamp = time.time()

    def get_state(self) -> DroneState:
        """Return a copy of the current drone state."""

        return DroneState(
            pose=Pose3D(
                x=self._state.pose.x,
                y=self._state.pose.y,
                z=self._state.pose.z,
                yaw=self._state.pose.yaw,
            ),
            velocity=Vector3(
                x=self._state.velocity.x,
                y=self._state.velocity.y,
                z=self._state.velocity.z,
            ),
            armed=self._state.armed,
            airborne=self._state.airborne,
            timestamp=self._state.timestamp,
            health_messages=list(
                self._state.health_messages
            ),
        )