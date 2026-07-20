import asyncio
import math

from mavsdk import System
from mavsdk.offboard import OffboardError, PositionNedYaw


async def first(stream):
    async for value in stream:
        return value


async def wait_until_reached(drone, north, east, down):
    good = 0

    async for data in drone.telemetry.position_velocity_ned():
        p = data.position
        horizontal_error = math.hypot(
            north - p.north_m,
            east - p.east_m,
        )
        vertical_error = abs(down - p.down_m)

        if horizontal_error < 0.20 and vertical_error < 0.25:
            good += 1
        else:
            good = 0

        if good >= 5:
            print(
                f"Reached N={p.north_m:.2f}, "
                f"E={p.east_m:.2f}, "
                f"error={horizontal_error:.2f} m"
            )
            return


async def main():
    drone = System()
    offboard_started = False

    print("Connecting...")
    await drone.connect(system_address="udpin://0.0.0.0:14540")

    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Connected.")
            break

    async for health in drone.telemetry.health():
        if health.is_local_position_ok:
            print("Local position ready.")
            break

    try:
        await drone.action.set_takeoff_altitude(3.0)

        print("Arming and taking off...")
        await drone.action.arm()
        await drone.action.takeoff()

        async for data in drone.telemetry.position_velocity_ned():
            if -data.position.down_m >= 2.5:
                print("Takeoff confirmed.")
                break

        start_data = await first(
            drone.telemetry.position_velocity_ned()
        )
        start = start_data.position
        down = start.down_m

        waypoints = [
            (start.north_m + 1.0, start.east_m),
            (start.north_m + 1.0, start.east_m + 1.0),
            (start.north_m, start.east_m + 1.0),
            (start.north_m, start.east_m),
        ]

        await drone.offboard.set_position_ned(
            PositionNedYaw(
                start.north_m,
                start.east_m,
                down,
                0.0,
            )
        )

        print("Starting Offboard mode...")
        await drone.offboard.start()
        offboard_started = True

        for number, (north, east) in enumerate(waypoints, 1):
            print(
                f"Waypoint {number}: "
                f"N={north:.2f}, E={east:.2f}"
            )

            await drone.offboard.set_position_ned(
                PositionNedYaw(north, east, down, 0.0)
            )

            await asyncio.wait_for(
                wait_until_reached(drone, north, east, down),
                timeout=20,
            )

            await asyncio.sleep(1)

        print("Square completed.")

    except asyncio.TimeoutError:
        print("ERROR: Waypoint timed out. Landing.")

    except OffboardError as error:
        print(f"ERROR: Offboard failed: {error}")

    finally:
        if offboard_started:
            try:
                await drone.offboard.stop()
            except OffboardError as error:
                print(f"Offboard stop warning: {error}")

        print("Landing...")
        await drone.action.land()

        async for in_air in drone.telemetry.in_air():
            if not in_air:
                print("Landing confirmed.")
                break


if __name__ == "__main__":
    asyncio.run(main())
