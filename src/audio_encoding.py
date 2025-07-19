import asyncio
import logging
import struct
import subprocess
import io
import wave
from typing import AsyncGenerator, Optional, Dict, Any
from enum import Enum

log = logging.getLogger(__name__)

class AudioFormat(Enum):
    WAV = "wav"
    RAW_PCM = "raw_pcm"
    FMP4 = "fmp4"
    MP3 = "mp3"
    WEBM = "webm"

class AudioEncoder:
    """Multi-format audio encoder with instant chunk processing for true real-time streaming."""

    def __init__(self, output_format: str, sample_rate: int, channels: int = 1,
                 bit_depth: int = 16, log_prefix: str = "", **kwargs):
        """
        Initialize the audio encoder.
        Args:
            output_format: Target format ("wav", "raw_pcm", "fmp4", "mp3", "webm")
            sample_rate: Audio sample rate in Hz
            channels: Number of audio channels (1=mono, 2=stereo)
            bit_depth: Bits per sample (8, 16, 24, 32)
            log_prefix: Prefix for log messages.
            **kwargs: Additional format-specific options
        """
        self.output_format = AudioFormat(output_format.lower())
        self.sample_rate = sample_rate
        self.channels = channels
        self.bit_depth = bit_depth
        self.log_prefix = log_prefix
        self.kwargs = kwargs

        # Format-specific settings
        self.ffmpeg_process: Optional[subprocess.Popen] = None
        self.wav_header_sent = False
        self.bytes_per_sample = bit_depth // 8
        self.frame_size = self.bytes_per_sample * channels

        # Validate format
        self._validate_format()

    def _validate_format(self):
        """Validate format-specific requirements."""
        if self.bit_depth not in [8, 16, 24, 32]:
            raise ValueError(f"Unsupported bit depth: {self.bit_depth}")
        if self.channels not in [1, 2]:
            raise ValueError(f"Unsupported channel count: {self.channels}")

    async def encode(self, pcm_generator: AsyncGenerator[bytes, None]) -> AsyncGenerator[bytes, None]:
        """
        Encode PCM chunks to target format with instant processing.
        Each input chunk is processed immediately and yielded back.
        """
        if self.output_format == AudioFormat.RAW_PCM:
            async for chunk in self._encode_raw_pcm(pcm_generator):
                yield chunk
        elif self.output_format == AudioFormat.WAV:
            async for chunk in self._encode_wav(pcm_generator):
                yield chunk
        elif self.output_format == AudioFormat.FMP4:
            async for chunk in self._encode_fmp4(pcm_generator):
                yield chunk
        elif self.output_format == AudioFormat.MP3:
            async for chunk in self._encode_mp3(pcm_generator):
                yield chunk
        elif self.output_format == AudioFormat.WEBM:
            async for chunk in self._encode_webm(pcm_generator):
                yield chunk
        else:
            raise ValueError(f"Unsupported format: {self.output_format}")

    async def _encode_raw_pcm(self, pcm_generator: AsyncGenerator[bytes, None]) -> AsyncGenerator[bytes, None]:
        """Raw PCM - pass through instantly."""
        async for pcm_chunk in pcm_generator:
            yield pcm_chunk

    async def _encode_wav(self, pcm_generator: AsyncGenerator[bytes, None]) -> AsyncGenerator[bytes, None]:
        """WAV encoding with instant header + chunk processing."""
        # Send WAV header immediately
        if not self.wav_header_sent:
            header = self._create_wav_header()
            yield header
            self.wav_header_sent = True

        # Process each chunk instantly
        async for pcm_chunk in pcm_generator:
            yield pcm_chunk

    def _create_wav_header(self, data_size: int = 0xFFFFFFFF) -> bytes:
        """Create WAV header with unknown size for streaming."""
        fmt_chunk_size = 16
        num_channels = self.channels
        sample_rate = self.sample_rate
        bits_per_sample = self.bit_depth
        byte_rate = sample_rate * num_channels * bits_per_sample // 8
        block_align = num_channels * bits_per_sample // 8

        # Use maximum size for streaming
        file_size = data_size + 36 if data_size != 0xFFFFFFFF else 0xFFFFFFFF

        header = struct.pack('<4sL4s', b'RIFF', file_size, b'WAVE')
        header += struct.pack('<4sLHHLLHH',
                             b'fmt ', fmt_chunk_size, 1, num_channels,
                             sample_rate, byte_rate, block_align, bits_per_sample)
        header += struct.pack('<4sL', b'data', data_size)

        return header

    async def _encode_fmp4(self, pcm_generator: AsyncGenerator[bytes, None]) -> AsyncGenerator[bytes, None]:
        """fMP4 encoding with instant chunk processing using FFmpeg."""
        try:
            # Start FFmpeg process
            self.ffmpeg_process = self._create_ffmpeg_process_fmp4()

            # Start background tasks for real-time processing
            writer_task = asyncio.create_task(self._ffmpeg_writer(pcm_generator))

            # Yield chunks as they come from the reader generator
            async for chunk in self._ffmpeg_reader():
                if chunk:
                    yield chunk

            # Ensure the writer task is complete and propagate any exceptions
            await writer_task

        except Exception as e:
            log.error(f"Error in fMP4 encoding: {e}")
            raise
        finally:
            await self._cleanup_ffmpeg()

    async def _encode_mp3(self, pcm_generator: AsyncGenerator[bytes, None]) -> AsyncGenerator[bytes, None]:
        """MP3 encoding with instant chunk processing."""
        try:
            self.ffmpeg_process = self._create_ffmpeg_process_mp3()

            writer_task = asyncio.create_task(self._ffmpeg_writer(pcm_generator))

            # Yield chunks as they come from the reader generator
            async for chunk in self._ffmpeg_reader():
                if chunk:
                    yield chunk

            # Ensure the writer task is complete and propagate any exceptions
            await writer_task

        except Exception as e:
            log.error(f"Error in MP3 encoding: {e}")
            raise
        finally:
            await self._cleanup_ffmpeg()

    async def _encode_webm(self, pcm_generator: AsyncGenerator[bytes, None]) -> AsyncGenerator[bytes, None]:
        """WebM encoding with instant chunk processing."""
        try:
            self.ffmpeg_process = self._create_ffmpeg_process_webm()

            writer_task = asyncio.create_task(self._ffmpeg_writer(pcm_generator))

            # Yield chunks as they come from the reader generator
            async for chunk in self._ffmpeg_reader():
                if chunk:
                    yield chunk

            # Ensure the writer task is complete and propagate any exceptions
            await writer_task

        except Exception as e:
            log.error(f"Error in WebM encoding: {e}")
            raise
        finally:
            await self._cleanup_ffmpeg()

    def _create_ffmpeg_process_fmp4(self) -> subprocess.Popen:
        """Create FFmpeg process for fMP4 output."""
        sample_format = f's{self.bit_depth}le'

        cmd = [
            'ffmpeg',
            '-f', sample_format,
            '-ar', str(self.sample_rate),
            '-ac', str(self.channels),
            '-i', 'pipe:0',
            '-c:a', 'aac',
            '-b:a', self.kwargs.get('bitrate', '64k'),
            '-f', 'mp4',
            '-movflags', 'frag_keyframe+empty_moov+default_base_moof+dash',
            '-frag_duration', str(self.kwargs.get('fragment_duration', 20000)),  # 200ms
            '-flush_packets', '1',
            '-reset_timestamps', '1',
            '-avoid_negative_ts', 'make_zero',
            'pipe:1',
            '-loglevel', 'error'
        ]

        return subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, bufsize=0
        )

    def _create_ffmpeg_process_mp3(self) -> subprocess.Popen:
        """Create FFmpeg process for MP3 output."""
        sample_format = f's{self.bit_depth}le'

        cmd = [
            'ffmpeg',
            '-f', sample_format,
            '-ar', str(self.sample_rate),
            '-ac', str(self.channels),
            '-i', 'pipe:0',
            '-c:a', 'libmp3lame',
            '-b:a', self.kwargs.get('bitrate', '128k'),
            '-f', 'mp3',
            '-flush_packets', '1',
            'pipe:1',
            '-loglevel', 'error'
        ]

        return subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, bufsize=0
        )

    def _create_ffmpeg_process_webm(self) -> subprocess.Popen:
        """Create FFmpeg process for WebM output."""
        sample_format = f's{self.bit_depth}le'

        cmd = [
            'ffmpeg',
            '-f', sample_format,
            '-ar', str(self.sample_rate),
            '-ac', str(self.channels),
            '-i', 'pipe:0',
            '-c:a', 'libopus',
            '-b:a', self.kwargs.get('bitrate', '64k'),
            '-f', 'webm',
            '-cluster_size_limit', '2k',
            '-cluster_time_limit', '50',  # 50ms clusters
            '-flush_packets', '1',
            'pipe:1',
            '-loglevel', 'error'
        ]

        return subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, bufsize=0
        )

    async def _ffmpeg_writer(self, pcm_generator: AsyncGenerator[bytes, None]):
        """Write PCM data to FFmpeg stdin with instant processing."""
        try:
            async for pcm_chunk in pcm_generator:
                if self.ffmpeg_process and self.ffmpeg_process.stdin:
                    # Write immediately, no buffering
                    await asyncio.get_event_loop().run_in_executor(
                        None, self.ffmpeg_process.stdin.write, pcm_chunk
                    )
                    await asyncio.get_event_loop().run_in_executor(
                        None, self.ffmpeg_process.stdin.flush
                    )
        except Exception as e:
            log.error(f"{self.log_prefix}Error writing to FFmpeg: {e}")
        finally:
            if self.ffmpeg_process and self.ffmpeg_process.stdin:
                try:
                    self.ffmpeg_process.stdin.close()
                except:
                    pass

    async def _ffmpeg_reader(self) -> AsyncGenerator[bytes, None]:
        """Read encoded data from FFmpeg stdout with instant processing."""
        if not self.ffmpeg_process or not self.ffmpeg_process.stdout:
            return

        try:
            while True:
                # Read small chunks for instant processing
                chunk = await asyncio.get_event_loop().run_in_executor(
                    None, self.ffmpeg_process.stdout.read, 1024
                )

                if not chunk:
                    break

                # Yield immediately
                yield chunk

        except Exception as e:
            log.error(f"{self.log_prefix}Error reading from FFmpeg: {e}")

    async def _cleanup_ffmpeg(self):
        """Clean up FFmpeg process."""
        if self.ffmpeg_process:
            try:
                if self.ffmpeg_process.stdin:
                    self.ffmpeg_process.stdin.close()

                # Give it a moment to finish
                await asyncio.sleep(0.01)

                if self.ffmpeg_process.poll() is None:
                    self.ffmpeg_process.terminate()
                    await asyncio.sleep(0.1)

                    if self.ffmpeg_process.poll() is None:
                        self.ffmpeg_process.kill()

            except Exception as e:
                log.error(f"Error cleaning up FFmpeg: {e}")

    def get_mime_type(self) -> str:
        """Get MIME type for the output format."""
        mime_types = {
            AudioFormat.WAV: 'audio/wav',
            AudioFormat.RAW_PCM: 'audio/pcm',
            AudioFormat.FMP4: 'audio/mp4',
            AudioFormat.MP3: 'audio/mpeg',
            AudioFormat.WEBM: 'audio/webm'
        }
        return mime_types.get(self.output_format, 'application/octet-stream')

    def get_file_extension(self) -> str:
        """Get file extension for the output format."""
        extensions = {
            AudioFormat.WAV: '.wav',
            AudioFormat.RAW_PCM: '.pcm',
            AudioFormat.FMP4: '.mp4',
            AudioFormat.MP3: '.mp3',
            AudioFormat.WEBM: '.webm'
        }
        return extensions.get(self.output_format, '.bin')

# # Usage examples:
# """
# # WAV streaming
# encoder = AudioEncoder("wav", 24000, channels=1, bit_depth=16)

# # fMP4 for MSE
# encoder = AudioEncoder("fmp4", 24000, channels=1, bitrate="64k", fragment_duration=100000)

# # MP3 streaming
# encoder = AudioEncoder("mp3", 24000, channels=1, bitrate="128k")

# # WebM for real-time web streaming
# encoder = AudioEncoder("webm", 24000, channels=1, bitrate="64k")

# # Raw PCM (no processing)
# encoder = AudioEncoder("raw_pcm", 24000, channels=1, bit_depth=16)
# """