"""SRT Broadcaster — Sends enhanced video to TV broadcast trucks via SRT.

SRT → SDI conversion at the broadcaster end using hardware encoders (Haivision, Kiloview).
Target latency: <10 seconds for TV broadcast.
"""
from loguru import logger

class SRTBroadcaster:
    """Manages SRT output streams to TV broadcast infrastructure."""

    def __init__(self):
        self.active_outputs: dict[str, dict] = {}

    def add_output(self, stream_id: str, dest_host: str, dest_port: int, passphrase: str = ""):
        self.active_outputs[stream_id] = {"host": dest_host, "port": dest_port, "passphrase": passphrase, "status": "connected"}
        logger.info("SRT broadcast output: {} → {}:{}", stream_id, dest_host, dest_port)

    def remove_output(self, stream_id: str):
        if stream_id in self.active_outputs:
            del self.active_outputs[stream_id]

    def get_ffmpeg_output_args(self, stream_id: str) -> list[str]:
        output = self.active_outputs.get(stream_id)
        if not output:
            return []
        srt_url = f"srt://{output['host']}:{output['port']}?mode=caller"
        if output["passphrase"]:
            srt_url += f"&passphrase={output['passphrase']}"
        return ["-f", "mpegts", srt_url]
