import asyncio
import subprocess
import os
import argparse
from typing import List
from stochsync.utils.print_utils import print_with_box, print_info, print_error, print_warning

# Function to run a command asynchronously
async def run_command(command: str, device: str, timeout: int, lock: asyncio.Lock) -> int:
    try:
        async with lock:
            print_with_box(f"Running command: {command} \ndevice: {device}", title="Command Execution")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = device

        # Run the command with the specified environment variable
        process = await asyncio.create_subprocess_shell(
            command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        timed_out = False
        try:
            await asyncio.wait_for(process.communicate(), timeout)
        except asyncio.TimeoutError:
            timed_out = True
            async with lock:
                print_warning(f"Command timed out: {command}")
            process.kill()
            await process.communicate()

        if process.returncode == 0 or timed_out:
            async with lock:
                status = "timed out" if timed_out else "succeeded"
                print_info(f"Command {status}: {command}")
                # Output is not printed to prevent terminal corruption
        else:
            async with lock:
                print_error(f"Command failed with return code {process.returncode}: {command}")
                stdout, stderr = await process.communicate()
                print_with_box(stderr.decode(), title="Command Error Output")
            return process.returncode

    except Exception as e:
        async with lock:
            print_error(f"Exception while running command: {command}")
            # Exception details are not printed to prevent terminal corruption
        return 1

    return 0

# Wrapper function to manage device allocation
async def run_command_with_device_queue(command: str, devices_queue: asyncio.Queue, timeout: int, lock: asyncio.Lock):
    device = await devices_queue.get()
    try:
        result = await run_command(command, device, timeout, lock)
    finally:
        devices_queue.put_nowait(device)
    return result

# Function to run all commands asynchronously
async def run_tests(commands: List[str], devices: List[str], timeout: int):
    lock = asyncio.Lock()
    devices_queue = asyncio.Queue()
    for device in devices:
        devices_queue.put_nowait(device)

    tasks = []
    for command in commands:
        task = asyncio.create_task(run_command_with_device_queue(command, devices_queue, timeout, lock))
        tasks.append(task)

    # Wait for all tasks to complete
    await asyncio.gather(*tasks)

# Main entry function
def main():
    parser = argparse.ArgumentParser(description="Run asynchronous unit tests with CUDA_VISIBLE_DEVICES.")
    parser.add_argument('--devices', nargs='+', help='List of CUDA visible devices (e.g., 0 1 2)', required=True)
    parser.add_argument('--timeout', type=int, default=120, help='Maximum runtime for each command (in seconds).')

    args = parser.parse_args()

    commands = [
        "python main.py --config config/ddim_image.yaml root_dir=./unit_test_results/",
        "python main.py --config config/sds_image.yaml root_dir=./unit_test_results/ max_steps=100",
        "python main.py --config config/better_pano.yaml root_dir=./unit_test_results/ max_steps=10",
        "python main.py --config config/better_mesh.yaml root_dir=./unit_test_results/ max_steps=10",
        "python main.py --config config/better_torus.yaml root_dir=./unit_test_results/ max_steps=6",
        "python main.py --config config/synctweedies_wide.yaml root_dir=./unit_test_results/",
        "python main.py --config config/sds_3dgs_fast.yaml root_dir=./unit_test_results/ max_steps=300",
    ]

    # Run asynchronous tests
    asyncio.run(run_tests(commands, args.devices, args.timeout))

if __name__ == "__main__":
    main()
