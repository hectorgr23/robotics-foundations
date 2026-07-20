import asyncio

from mavsdk import System


async def main() -> None:
    """Connect to PX4 SITL and verify that telemetry is available."""

    drone = System()

    print("Listening for PX4 on UDP port 14540...")

    await drone.connect(
        system_address="udpin://0.0.0.0:14540"
    )

    async for connection_state in drone.core.connection_state():
        if connection_state.is_connected:
            print("SUCCESS: MAVSDK connected directly to PX4.")
            break

    async for armed in drone.telemetry.armed():
        print(f"Armed: {armed}")
        break

    async for flight_mode in drone.telemetry.flight_mode():
        print(f"Flight mode: {flight_mode}")
        break

    print("Telemetry received successfully. No flight command was sent.")


if __name__ == "__main__":
    asyncio.run(main())
