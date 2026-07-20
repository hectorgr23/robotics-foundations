import asyncio
import math

from mavsdk import System
from mavsdk.offboard import PositionNedYaw


async def first(stream):
    async for value in stream:
        return value


async def main():
    drone = System()
    print("Connecting to PX4...")
    await drone.connect(system_address="udpin://0.0.0.0:14540")

    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Connected.")
            break

    print("Waiting for local position...")
    async for health in drone.telemetry.health():
        if health.is_local_position_ok:
            print("Local position ready.")
            break

    await drone.action.set_takeoff_altitude(3.0)

    print("Arming and taking off...")
    await drone.action.arm()
    await drone.action.takeoff()

    async for data in drone.telemetry.position_velocity_ned():
        altitude = -data.position.down_m
        if altitude >= 2.5:
            print(f"Altitude confirmed: {altitude:.2f} m")
            break

    start = await first(drone.telemetry.position_velocity_ned())
    p = start.position

    target_north = p.north_m + 1.0
    target_east = p.east_m
    target_down = p.down_m

    await drone.offboard.set_position_ned(
        PositionNedYaw(p.north_m, p.east_m, p.down_m, 0.0)
    )

    print("Starting Offboard control...")
    await drone.offboard.start()

    print("Moving exactly 1 meter north...")
    await drone.offboard.set_position_ned(
        PositionNedYaw(target_north, target_east, target_down, 0.0)
    )

    good_samples = 0

    async for data in drone.telemetry.position_velocity_ned():
        p = data.position
        error = math.hypot(
            target_north - p.north_m,
            target_east - p.east_m,
        )

        if error <= 0.20:
            good_samples += 1
        else:
            good_samples = 0

        if good_samples >= 5:
            print(
                f"Target confirmed: north={p.north_m:.2f}, "
                f"east={p.east_m:.2f}, error={error:.2f} m"
            )
            break

    await asyncio.sleep(2)

    print("Stopping Offboard and landing...")
    await drone.offboard.stop()
    await drone.action.land()

    async for in_air in drone.telemetry.in_air():
        if not in_air:
            print("Landing confirmed.")
            break


if __name__ == "__main__":
    asyncio.run(main())
