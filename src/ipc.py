"""
Inter-Process Communication (IPC) using ZeroMQ.

This module sets up the ZeroMQ sockets and defines the message structures
for communication between the master process and the worker processes.
"""
import zmq.asyncio
from dataclasses import dataclass, field
from typing import Optional, Any

# Use the high-water mark setting to prevent queues from growing indefinitely.
ZMQ_HWM = 100

# --- Socket Addresses ---
# Workers connect to these addresses to pull jobs from the master.
JOB_SOCKET_ADDR = "tcp://127.0.0.1:5555"

# The master connects to this address to pull results from the workers.
RESULT_SOCKET_ADDR = "tcp://127.0.0.1:5556"

# A publish socket for the master to send broadcast commands to all workers.
BROADCAST_SOCKET_ADDR = "tcp://127.0.0.1:5557"


@dataclass
class TTSRequest:
    """A dataclass for TTS job requests."""
    request_id: str
    text: str
    output_format: str
    voice_id: Optional[str]
    cfg_guidance_weight: float
    synthesis_temperature: float
    text_processing_chunk_size: int
    audio_tokens_per_slice: int
    remove_trailing_milliseconds: int
    remove_leading_milliseconds: int
    chunk_overlap_strategy: str
    crossfade_duration_milliseconds: int

@dataclass
class TTSStreamChunk:
    """A dataclass for a chunk of streamed audio data."""
    request_id: str
    chunk: bytes
    is_final: bool = False

@dataclass
class BroadcastCommand:
    """A dataclass for broadcast commands."""
    command: str
    details: dict = field(default_factory=dict)

@dataclass
class WorkerStatus:
    """A dataclass for worker status updates."""
    worker_id: int
    status: str  # e.g., "ready", "error"
    message: Optional[str] = None

def setup_master_sockets(context: zmq.asyncio.Context) -> tuple:
    """Sets up the sockets for the master process."""
    job_socket = context.socket(zmq.PUSH)
    job_socket.set_hwm(ZMQ_HWM)
    job_socket.bind(JOB_SOCKET_ADDR)

    result_socket = context.socket(zmq.PULL)
    result_socket.set_hwm(ZMQ_HWM)
    result_socket.bind(RESULT_SOCKET_ADDR)

    broadcast_socket = context.socket(zmq.PUB)
    broadcast_socket.bind(BROADCAST_SOCKET_ADDR)

    return job_socket, result_socket, broadcast_socket

def setup_worker_sockets(context: zmq.asyncio.Context) -> tuple:
    """Sets up the sockets for a worker process."""
    job_socket = context.socket(zmq.PULL)
    job_socket.set_hwm(ZMQ_HWM)
    job_socket.connect(JOB_SOCKET_ADDR)

    result_socket = context.socket(zmq.PUSH)
    result_socket.set_hwm(ZMQ_HWM)
    result_socket.connect(RESULT_SOCKET_ADDR)

    broadcast_socket = context.socket(zmq.SUB)
    broadcast_socket.connect(BROADCAST_SOCKET_ADDR)
    broadcast_socket.setsockopt_string(zmq.SUBSCRIBE, "") # Subscribe to all messages

    return job_socket, result_socket, broadcast_socket