def test_sport_classes():
    from ai_engine.src.models.object_tracker import SPORT_CLASSES
    assert "cricket" in SPORT_CLASSES
    assert "ball" in SPORT_CLASSES["cricket"]

def test_placeholder_detections():
    from ai_engine.src.models.object_tracker import ObjectTracker
    import asyncio
    tracker = ObjectTracker()
    dets = tracker.detect_placeholder()
    assert len(dets) > 0
    assert "bbox" in dets[0]
    assert "confidence" in dets[0]
