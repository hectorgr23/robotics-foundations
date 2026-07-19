"""Common flight interface for simulated and PX4-controlled drones."""

from abc import ABC, abstractmethod

from core.messages import DroneState, VelocityCommand


class FlightInterface(ABC):
    """Contract implemented by every drone flight backend."""

    @abstractmethod
    def arm(self) -> None:
        """Arm the drone."""

    @abstractmethod
    def disarm(self) -> None:
        """Disarm the drone."""

    @abstractmethod
    def takeoff(self, target_altitude: float) -> None:
        """Take off to the requested altitude."""

    @abstractmethod
    def land(self) -> None:
        """Land the drone."""

    @abstractmethod
    def apply_velocity_command(
        self,
        command: VelocityCommand,
        delta_time: float,
    ) -> None:
        """Apply one velocity command."""

    @abstractmethod
    def get_state(self) -> DroneState:
        """Return the current drone state."""