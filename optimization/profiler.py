from collections import defaultdict
from contextlib import contextmanager
from time import perf_counter

import psutil

from ncm_engine.device import synchronize


class Profile:
    def __init__(self, device, detailed=False):
        self.device = device
        self.detailed = detailed
        self.ms = defaultdict(float)

    @contextmanager
    def stage(self, name, *, detail=False):
        if detail and not self.detailed:
            yield
            return
        # The outer request timer supplies the completion barrier in normal mode.
        if self.detailed or name == "total":
            synchronize(self.device)
        start = perf_counter()
        yield
        if self.detailed or name == "total":
            synchronize(self.device)
        self.ms[name] += (perf_counter() - start) * 1000

    def result(self):
        return dict(self.ms), psutil.Process().memory_info().rss
