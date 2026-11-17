"""Apply the inherited offline main-camera eligibility input to prefix seeding."""
from near_prefix_initialization import NearPrefix


class CameraValidNearPrefix(NearPrefix):
    def __init__(self, homography, intervals, adaptive_size=True):
        super().__init__(homography, adaptive_size)
        self.intervals = tuple(intervals)

    def step(self, frame, rows, features):
        if not any(first <= frame <= last for first,last in self.intervals):
            # Fixed-view calibration cannot be carried across another shot.
            self.window.clear()
            self.states.clear()
            return []
        return super().step(frame, rows, features)
