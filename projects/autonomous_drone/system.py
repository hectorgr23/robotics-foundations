"""Top-level autonomous drone system orchestrator."""

from core.messages import VelocityCommand
from simulation.simulated_drone import SimulatedDrone


class AutonomousDroneSystem:
    """Coordinate the autonomous drone subsystems."""

    def __init__(self) -> None:
        """Construct the current software stack."""

        self.flight_interface = SimulatedDrone()

    def start(self) -> None:
        """Run the first integrated 3D flight test."""

        print("AUTONOMOUS DRONE SOFTWARE")
        print("-------------------------")

        print(f"Initial state: {self.flight_interface.get_state()}")

        self.flight_interface.arm()
        print(f"After arming: {self.flight_interface.get_state()}")

        self.flight_interface.takeoff(
            target_altitude=10.0,
        )

        print(
            f"After takeoff: "
            f"{self.flight_interface.get_state()}"
        )

        command = VelocityCommand(
            velocity_x=4.0,
            velocity_y=2.0,
            velocity_z=1.0,
            yaw_rate=15.0,
        )

        self.flight_interface.apply_velocity_command(
            command=command,
            delta_time=2.0,
        )

        print(
            f"After velocity command: "
            f"{self.flight_interface.get_state()}"
        )

        self.flight_interface.land()
        print(f"After landing: {self.flight_interface.get_state()}")

        self.flight_interface.disarm()
        print(
            f"After disarming: "
            f"{self.flight_interface.get_state()}"
        )

        print()
        print(
            "Simulated 3D flight interface connected successfully."
        )