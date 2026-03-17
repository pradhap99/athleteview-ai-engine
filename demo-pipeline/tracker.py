"""
AthleteView — Player Tracking Module
Uses background subtraction + contour detection for fast CPU tracking.
For production: YOLOv11 + BoT-SORT on GPU.
"""

import cv2
import numpy as np
from collections import OrderedDict


class CentroidTracker:
    """Simple centroid-based object tracker."""

    def __init__(self, max_disappeared=15):
        self.next_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared

    def register(self, centroid, bbox):
        self.objects[self.next_id] = {"centroid": centroid, "bbox": bbox}
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, detections):
        if len(detections) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return []

        input_centroids = []
        input_bboxes = []
        input_confs = []

        for det in detections:
            x1, y1, x2, y2, conf = det
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            input_centroids.append((cx, cy))
            input_bboxes.append((x1, y1, x2, y2))
            input_confs.append(conf)

        input_centroids = np.array(input_centroids)

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], input_bboxes[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = np.array([self.objects[oid]["centroid"] for oid in object_ids])

            # Simple distance-based matching
            results_matched = {}
            used_cols = set()

            for row_idx, oid in enumerate(object_ids):
                oc = object_centroids[row_idx]
                best_dist = 150
                best_col = -1
                for col_idx in range(len(input_centroids)):
                    if col_idx in used_cols:
                        continue
                    d = np.sqrt((oc[0] - input_centroids[col_idx][0])**2 +
                               (oc[1] - input_centroids[col_idx][1])**2)
                    if d < best_dist:
                        best_dist = d
                        best_col = col_idx

                if best_col >= 0:
                    self.objects[oid]["centroid"] = input_centroids[best_col]
                    self.objects[oid]["bbox"] = input_bboxes[best_col]
                    self.disappeared[oid] = 0
                    used_cols.add(best_col)
                    results_matched[oid] = best_col
                else:
                    self.disappeared[oid] += 1
                    if self.disappeared[oid] > self.max_disappeared:
                        self.deregister(oid)

            # Register new detections
            for col_idx in range(len(input_centroids)):
                if col_idx not in used_cols:
                    self.register(input_centroids[col_idx], input_bboxes[col_idx])

        # Build output
        results = []
        for oid, data in self.objects.items():
            x1, y1, x2, y2 = data["bbox"]
            conf = 0.85
            for i, (bx1, by1, bx2, by2) in enumerate(input_bboxes):
                if abs(bx1 - x1) < 5 and abs(by1 - y1) < 5:
                    conf = input_confs[i]
                    break
            results.append((x1, y1, x2, y2, oid, conf))

        return results


class PlayerTracker:
    """Fast CPU-based player tracker using background subtraction + contours."""

    def __init__(self, use_yolo=False):
        self.centroid_tracker = CentroidTracker(max_disappeared=15)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=100, varThreshold=40, detectShadows=False
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        print("[Tracker] Using background subtraction + contour detection (CPU-fast)")

    def detect(self, frame):
        """Detect moving objects using background subtraction."""
        fg_mask = self.bg_subtractor.apply(frame)

        # Clean up mask
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel)
        fg_mask = cv2.dilate(fg_mask, self.kernel, iterations=2)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500 or area > 50000:  # Filter noise and too-large regions
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            # Aspect ratio filter (roughly person-shaped)
            aspect = h / w if w > 0 else 0
            if aspect < 0.8 or aspect > 4.0:
                continue

            conf = min(1.0, area / 5000)  # Confidence from area
            detections.append((x, y, x + w, y + h, conf))

        return detections

    def track(self, frame):
        """Detect and track objects in frame."""
        detections = self.detect(frame)
        tracked = self.centroid_tracker.update(detections)
        return tracked
