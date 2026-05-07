import time
from contextlib import contextmanager


class Timer:
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.records = {}

    @contextmanager
    def track(self, key):
        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.records.setdefault(key, []).append(duration)

    def summary(self):
        summary = {}
        for key, values in self.records.items():
            if not values:
                continue
            summary[key] = sum(values) / len(values)
        return summary

    def adjust_last(self, key, delta):
        if not self.enabled:
            return
        values = self.records.get(key)
        if not values:
            return
        values[-1] = max(0.0, values[-1] + float(delta))
