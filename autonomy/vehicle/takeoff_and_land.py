import asyncio

from mavsdk import System


async def wait_for_connection(drone: System) -> None:
    print("Waiting for PX4 connection...")

    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Connected to PX4.")
            return


async def wait_for_position_estimate(drone: System) -> None:
    print("Waiting for PX4 position estimate...")

    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("Position estimate ready.")
            return


async def main() -> None:
    drone = System()

    await drone.connect(
        system_address="udpin://0.0.0.0:14540"
    )

    await wait_for_connection(drone)
    await wait_for_position_estimate(drone)

    print("Setting takeoff altitude to 3 meters...")
    await drone.action.set_takeoff_altitude(3.0)

    print("Arming...")
    await drone.action.arm()

    print("Taking off...")
    await drone.action.takeoff()

    print("Holding altitude for 8 seconds...")
    await asyncio.sleep(8)

    print("Landing...")
    await drone.action.land()

    async for in_air in drone.telemetry.in_air():
        if not in_air:
            print("Landing complete.")
            break


if __name__ == "__main__":
    asyncio.run(main())
