import json

import numpy as np

from frontier.profiling.utils import ProfileMethod, normalize_profile_method
from frontier.profiling.utils.singleton import Singleton


class TimerStatsStore(metaclass=Singleton):
    def __init__(self, profile_method: str, disabled: bool = False):
        self.disabled = disabled
        self.profile_method = ProfileMethod[normalize_profile_method(profile_method).upper()]
        self.clear_stats()

    def record_time(self, name: str, time):
        name = name.replace("vidur_", "")
        if name not in self.TIMING_STATS:
            self.TIMING_STATS[name] = []

        self.TIMING_STATS[name].append(time)

    def clear_stats(self):
        self.TIMING_STATS = {}
        self._warmup_counts = {}

    def mark_warmup_end(self):
        """Snapshot how many records each scope has so far; those are the warm-up runs."""
        self._warmup_counts = {name: len(times) for name, times in self.TIMING_STATS.items()}

    def get_stats(self):
        """Aggregate the recorded runs.

        Records made before ``mark_warmup_end()`` are excluded from the
        aggregates (per scope, so a scope timed twice per forward still gets
        only its warm-up records dropped) but kept, in run order, in
        ``samples`` (JSON list) with their count in ``warmup_count``.
        """
        stats = {}
        for name, times in self.TIMING_STATS.items():
            times = [
                (time if isinstance(time, float) else time[0].elapsed_time(time[1]))
                for time in times
            ]
            warmup_count = self._warmup_counts.get(name, 0)
            timed = times[warmup_count:]
            if not timed:
                raise ValueError(
                    f"{name}: {len(times)} recorded run(s), all of them warm-up"
                )

            stats[name] = {
                "min": np.min(timed),
                "max": np.max(timed),
                "mean": np.mean(timed),
                "median": np.median(timed),
                "std": np.std(timed),
                "count": len(timed),
                "warmup_count": warmup_count,
                "samples": json.dumps([round(t, 6) for t in times], separators=(",", ":")),
            }

        return stats
