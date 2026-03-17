"""RTMP Restreamer — Simultaneously restreams to YouTube Live, Twitch, Facebook Live."""
import subprocess
from loguru import logger

class RTMPRestreamer:
    """Manages RTMP restreaming to multiple destinations."""

    def __init__(self):
        self.active_processes: dict[str, subprocess.Popen] = {}

    def start_restream(self, stream_id: str, input_url: str, targets: list[dict]):
        """Start restreaming to one or more RTMP destinations.
        
        Each target: {"name": "youtube", "url": "rtmp://a.rtmp.youtube.com/live2/KEY"}
        """
        for target in targets:
            key = f"{stream_id}:{target['name']}"
            cmd = ["ffmpeg", "-i", input_url, "-c", "copy", "-f", "flv", target["url"]]
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            self.active_processes[key] = proc
            logger.info("Restreaming {} → {} (PID: {})", stream_id, target["name"], proc.pid)

    def stop_restream(self, stream_id: str, target_name: str = None):
        keys_to_remove = []
        for key, proc in self.active_processes.items():
            if key.startswith(stream_id) and (target_name is None or key.endswith(target_name)):
                proc.terminate()
                keys_to_remove.append(key)
                logger.info("Stopped restream: {}", key)
        for key in keys_to_remove:
            del self.active_processes[key]

    def get_status(self) -> dict:
        return {key: {"pid": proc.pid, "running": proc.poll() is None} for key, proc in self.active_processes.items()}
