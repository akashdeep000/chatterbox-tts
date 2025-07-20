"""
Master Process for the Multi-Process TTS Server.

This module is responsible for:
- Spawning and managing the lifecycle of worker processes.
- Setting up the ZeroMQ sockets for IPC.
- Storing global state, such as the mapping of requests to response queues.
"""
import asyncio
import subprocess
import sys
import torch
import zmq.asyncio
import pickle
from typing import List, Dict
from src.config import settings
from src.logging_config import log
from src.ipc import setup_master_sockets, TTSStreamChunk, BroadcastCommand
from src.voice_manager import VoiceManager

# Global state for the master process
worker_processes: List[subprocess.Popen] = []
# A dictionary to hold the asyncio Queues for each active request
active_requests: Dict[str, asyncio.Queue] = {}

async def result_listener(result_socket: zmq.asyncio.Socket):
    """Listens for results from workers and puts them into the correct request queue."""
    while True:
        try:
            result_payload = await result_socket.recv()
            result: TTSStreamChunk = pickle.loads(result_payload)

            if result.request_id in active_requests:
                await active_requests[result.request_id].put(result)
            else:
                log.warning(f"Received result for unknown request_id: {result.request_id}")
        except Exception as e:
            log.error(f"Error in result listener: {e}", exc_info=True)

def spawn_workers(broadcast_socket: zmq.asyncio.Socket):
    """
    Detects devices, spawns workers, and then broadcasts the voice list for cache warming.
    """
    devices = []
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        devices.extend([f"cuda:{i}" for i in range(num_gpus)])
    else:
        devices.append("cpu")

    worker_id_counter = 0
    for device in devices:
        for _ in range(settings.WORKERS_PER_DEVICE):
            log.info(f"Spawning worker {worker_id_counter} on device {device}...")
            process = subprocess.Popen(
                [sys.executable, "-m", "src.worker", str(worker_id_counter), device]
            )
            worker_processes.append(process)
            worker_id_counter += 1

    log.info(f"Spawned a total of {len(worker_processes)} worker processes.")

    # --- Broadcast Voice List for Cache Warming ---
    async def broadcast_voice_list():
        # Give workers a moment to initialize their sockets
        await asyncio.sleep(5)

        voice_manager = VoiceManager()
        voice_ids = voice_manager.list_voices()
        if voice_ids:
            log.info(f"Broadcasting voice list to workers for cache warming: {voice_ids}")
            command = BroadcastCommand(command="warm_up_voices", details={"voice_ids": voice_ids})
            await broadcast_socket.send(pickle.dumps(command))
        else:
            log.info("No voices to broadcast for cache warming.")

    asyncio.create_task(broadcast_voice_list())

def shutdown_workers():
    """Terminates all spawned worker processes."""
    log.info("Terminating all worker processes...")
    for process in worker_processes:
        process.terminate()
    for process in worker_processes:
        process.wait()
    log.info("All worker processes terminated.")