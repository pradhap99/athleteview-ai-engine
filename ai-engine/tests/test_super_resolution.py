import numpy as np

def test_frame_buffer():
    from ai_engine.src.inference.frame_buffer import FrameBuffer
    buf = FrameBuffer(max_size=10)
    assert buf.is_empty
    buf.push(np.zeros((480, 640, 3), dtype=np.uint8), timestamp=1.0)
    assert buf.size == 1
    frame = buf.pop()
    assert frame is not None
    assert frame["timestamp"] == 1.0

def test_highlight_detector_rules():
    from ai_engine.src.models.highlight_detector import HighlightDetector
    import asyncio
    det = HighlightDetector()
    asyncio.run(det.load())
    result = det.detect(audio_level=0.9, imu_magnitude=4.0, hr_delta=25, frame_detections=[{} for _ in range(6)])
    assert result["is_highlight"] == True
    assert result["confidence"] >= 0.5
