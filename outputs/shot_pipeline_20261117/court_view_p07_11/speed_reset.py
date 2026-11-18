"""Source-frame rejection streak; absent observations do not break the streak."""
STATUS = 'observed_after_speed_reset'


class SpeedReset:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.start = None
        self.rally = None

    def update(self, frame, status, rally=None):
        if rally != self.rally:
            self.start = None
            self.rally = rally
        if not self.enabled or status != 'rejected_speed':
            self.start = None
            return None
        if self.start is None:
            self.start = frame
        if frame - self.start > 30:
            start = self.start
            self.start = None
            return start
        return None
