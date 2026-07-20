import asyncio

from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityNedYaw


async def wait_for_connection(drone: System) -> None:
    print("Waiting for PX4 connection...")

    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Connected to PX4.")
            return


async def wait_for_position(drone: System) -> None:
    print("Waiting for position estimate...")

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
    await wait_for_position(drone)

    print("Arming...")
    await drone.action.arm()

    # PX4 requires an initial setpoint before entering Offboard mode.
    print("Sending initial zero-velocity setpoint...")
    await drone.offboard.set_velocity_ned(
        VelocityNedYaw(
            north_m_s=0.0,
            east_m_s=0.0,
            down_m_s=0.0,
            yaw_deg=0.0,
        )
    )

    print("Starting Offboard mode...")

    try:
        await drone.offboard.start()
    except OffboardError as error:
        print(f"Could not start Offboard mode: {error._result.result}")
        await drone.action.disarm()
        return

    try:
        print("Ascending for 3 seconds...")
        await drone.offboard.set_velocity_ned(
            VelocityNedYaw(0.0, 0.0, -1.0, 0.0)
        )
        await asyncio.sleep(3)

        print("Holding...")
        await drone.offboard.set_velocity_ned(
            VelocityNedYaw(0.0, 0.0, 0.0, 0.0)
        )
        await asyncio.sleep(2)

        print("Moving north...")
        await drone.offboard.set_velocity_ned(
            VelocityNedYaw(1.0, 0.0, 0.0, 0.0)
        )
        await asyncio.sleep(3)

        print("Moving east...")
        await drone.offboard.set_velocity_ned(
            VelocityNedYaw(0.0, 1.0, 0.0, 90.0)
        )
        await asyncio.sleep(3)

        print("Moving south...")
        await drone.offboard.set_velocity_ned(
            VelocityNedYaw(-1.0, 0.0, 0.0, 180.0)
        )
        await asyncio.sleep(3)

        print("Moving west...")
        await drone.offboard.set_velocity_ned(
            VelocityNedYaw(0.0, -1.0, 0.0, 270.0)
        )
        await asyncio.sleep(3)

        print("Stopping horizontal motion...")
        await drone.offboard.set_velocity_ned(
            VelocityNedYaw(0.0, 0.0, 0.0, 0.0)
        )
        await asyncio.sleep(2)

    finally:
        print("Stopping Offboard mode...")

        try:
            await drone.offboard.stop()
        except OffboardError as error:
            print(f"Offboard stop reported: {error._result.result}")

        print("Landing...")
        await drone.action.land()

        async for in_air in drone.telemetry.in_air():
            if not in_air:
                print("Landing complete.")
                break


if __name__ == "__main__":
    asyncio.run(main())
