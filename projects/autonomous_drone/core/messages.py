"""Shared data exchanged by the autonomous drone subsystems."""

from dataclasses import dataclass, field


@dataclass
class Vector3:
    """A three-dimensional vector."""

    x: float
    y: float
    z: float


@dataclass
class Pose3D:
    """
    Drone position and orientation in a local world frame.

    x: horizontal position
    y: horizontal position
    z: altitude
    yaw: compass-like horizontal rotation in degrees
    """

    x: float
    y: float
    z: float
    yaw: float


@dataclass
class VelocityCommand:
    """Velocity command sent toward the flight-control layer."""

    velocity_x: float
    velocity_y: float
    velocity_z: float
    yaw_rate: float


@dataclass
class MissionGoal:
    """A requested destination for the autonomous drone."""

    x: float
    y: float
    z: float
    yaw: float = 0.0


@dataclass
class DroneState:
    """Current state estimate used by the autonomy software."""

    pose: Pose3D
    velocity: Vector3
    armed: bool
    airborne: bool
    timestamp: float
    health_messages: list[str] = field(
        default_factory=list
    )