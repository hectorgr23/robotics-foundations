"""Permanent entry point for the autonomous drone software."""

from system import AutonomousDroneSystem


def main() -> None:
    """Build and start the autonomous drone system."""

    drone_system = AutonomousDroneSystem()
    drone_system.start()


if __name__ == "__main__":
    main()