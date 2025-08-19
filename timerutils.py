import time
import contextlib
import resource

class TimerManager:
    def __init__(self, timer, section):
        self.timer = timer
        self.section = section

    def __enter__(self):
        #self.res = resource.getrusage(resource.RUSAGE_SELF)
        return self.timer.start(self.section)

    def __exit__(self, exc_type, exc_value, traceback):
        self.timer.stop(self.section)
        #self.after_res = resource.getrusage(resource.RUSAGE_SELF)
        #print("utime", self.after_res.ru_utime - self.res.ru_utime)

class Timer:
    def __init__(self):
        self.sections = []

    def time(self, section):
        return TimerManager(self, section)

    def push(self, section):
        t = time.monotonic_ns()
        self.sections.append((section, t))

    start = push

    def pop(self, section, display=True):
        t = time.monotonic_ns()
        assert len(self.sections) > 0, "Timer::Pop when stack is empty"
        top = self.sections.pop()
        assert top[0] == section, f"Timer stack top '{top[0]}' does not match provided '{section}'"
        if display:
            print(section, t - top[1], "ns")
        return t - top[1], (top[0], top[1])

    stop = pop
