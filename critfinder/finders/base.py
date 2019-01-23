import autograd
import autograd.numpy as np


class Finder(object):
    """Abstract base class for critical-point-finding algorithms (critfinders).

    The main piece of shared capability is in logging: update_logs is called
    during the execution of the algorithm to record information about the run.
    """

    def __init__(self, f, grad_kwargs={}, log_kwargs={}):
        self.f = f
        self.grad_f = autograd.grad(f, **grad_kwargs)

        self.log = {}
        self.loggers = []
        self.log_kwargs = log_kwargs

        self.setup_logs(**log_kwargs)

    def run(self):
        raise NotImplementedError

    def update_logs(self, step_info):
        for logger in self.loggers:
            logger.write_log(step_info, self.log)

    def setup_logs(self, track_theta=False, track_f=True, track_grad_f=False, track_g=False):
        if track_theta:
            self.loggers.append(
                Logger("theta",
                       lambda step_info: step_info["theta"]))

        if track_f:
            self.loggers.append(
                Logger("f_theta",
                       lambda step_info: self.f(step_info["theta"])))

        if track_grad_f:
            self.loggers.append(
                Logger("grad_theta",
                       lambda step_info: self.grad_f(step_info["theta"])))

        if track_g:
            self.loggers.append(
                Logger("g_theta",
                       lambda step_info: 0.5 * np.sum(np.square(self.grad_f(step_info["theta"])))))


class Logger(object):
    """
    """

    def __init__(self, key, log_func):
        self.key = key
        self.log_func = log_func

    def write_log(self, step_info, log):
        if self.key not in log.keys():
            log[self.key] = []
        log[self.key].append(self.log_func(step_info))
