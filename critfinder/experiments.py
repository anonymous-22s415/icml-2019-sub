import json
import random

import autograd
import autograd.numpy as np

from . import finders
from . import networks

_NETWORK_INITS = {"feedforward": networks.FeedforwardNetwork}

_OPTIMIZERS = {"gd": networks.gradient_descent,
               "momentum": networks.momentum}

_FINDER_INITS = {"newtonMR": finders.newtons.FastNewtonMR,
                 "newtonTR": finders.newtons.FastNewtonTR,
                 "gnm": finders.gradnormmin.GradientNormMinimizer}

DEFAULT_LR = 0.1
DEFAULT_LOG_KWARGS = {"track_theta": True, "track_f": True, "track_grad_f": False}


class Experiment(object):
    """Abstract base class for OptimizationExperiments and CritFinderExperiments.

    Concrete classes should implement a .run method that executes the experiment
    and stores the results of runs in self.runs, a list. These should be save-able
    into .npz format by np.savez.

    They should further implement a construct_dictionary method that saves
    all of the relevant arguments necessary for a constructor call as a dictionary
    that can be written to a .json file. These .json files are used to reconstruct
    experiments and their components.
    """

    def __init__(self):
        self.runs = []

    def to_json(self, filename):
        dictionary = self.construct_dictionary()

        with open(filename, "w") as f:
            json.dump(dictionary, f)

    def save_results(self, filename):
        results_dict = self.runs[-1]
        np.savez(filename, **results_dict)

    def construct_dictionary(self):
        raise NotImplementedError


class OptimizationExperiment(Experiment):
    """Concrete Experiment that performs optimization on a function.
    """

    def __init__(self, f, grad_f=None, optimizer_str="gd", optimizer_kwargs=None,
                 log_kwargs=None):
        """Create an OptimizationExperiment on callable f according to kwargs.

        Parameters
        ----------

        f : callable
            Function to optimize. Should require only parameters as input.
            For stochastic functions, e.g. for stochastic gradient descent,
            function must perform batching.

        grad_f : callable or None, default is None
            A gradient oracle for f. If None, autograd.grad is called on f.

        optimizer_str : str
            String to key into _OPTIMIZERS. Default is "gd", which is
            networks.gradient_descent.

        optimizer_kwargs : dict or None, default is None
            A dictionary of keyword arguments for the optimizer selected with
            optimizer_str. See networks for call signatures. If None, the default
            is constructed from DEFAULT_LR.

        log_kwargs : dict or None, default is None
            A dictionary of keyword arguments for the log_run method, which
            determines which features of the run are saved. If None,
            DEFAULT_LOG_KWARGS is used. See log_run for details.
        """
        Experiment.__init__(self)

        if log_kwargs is None:
            self.log_kwargs = DEFAULT_LOG_KWARGS.copy()
        else:
            self.log_kwargs = log_kwargs

        self.f = f

        if grad_f is None:
            self.grad_f = autograd.grad(f)
        else:
            self.grad_f = grad_f

        self.optimizer_str = optimizer_str
        self.optimizer = _OPTIMIZERS[self.optimizer_str]

        if optimizer_kwargs is None:
            self.optimizer_kwargs = {}
        else:
            self.optimizer_kwargs = optimizer_kwargs

        if "lr" not in self.optimizer_kwargs.keys():
            self.optimizer_kwargs["lr"] = DEFAULT_LR

    def run(self, init_theta, num_iters):
        """Execute optimizer on self.f, starting with init_theta, for num_iters.
        """
        thetas = self.optimizer(self.grad_f, init_theta, num_iters,
                                **self.optimizer_kwargs)
        self.log_run(thetas, **self.log_kwargs)
        return thetas

    def log_run(self, thetas,
                track_theta=False, track_f=False, track_grad_f=False, track_g=False):
        """Append a dictionary containing data identified by kwargs to self.runs.
        """
        run = {}
        if track_theta:
            run["theta"] = thetas
        if track_f:
            run["f_theta"] = [self.f(theta) for theta in thetas]
        if track_grad_f:
            run["grad_f_theta"] = [self.grad_f(theta) for theta in thetas]
        if track_g:
            run["g_theta"] = [0.5 * np.sum(np.square(self.grad_f(theta))) for theta in thetas]
        self.runs.append(run)

    @classmethod
    def from_json(cls, f, filename, grad_f=None):
        """Given a function and possibly a gradient oracle and the path to a .json file,
        creates an OptimizationExperiment on f using kwargs in the .json file.
        """
        with open(filename) as fn:
            dictionary = json.load(fn)

        return cls(f, grad_f, **dictionary)

    def construct_dictionary(self):
        """Construct a dictionary containing necessary information for
        reconstructing OptimizationExperiment when combined with self.f.

        See OptimizationExperiment.from_json for details.
        """
        return {"optimizer_str": self.optimizer_str,
                "optimizer_kwargs": self.optimizer_kwargs,
                "log_kwargs": self.log_kwargs}


class CritFinderExperiment(Experiment):
    """Concrete Experiment that finds critical points on a function.
    """

    def __init__(self, f, finder_str, finder_kwargs=None):
        """

        Parameters
        ----------

        f : callable
            Function to search on. Should require only parameters as input.
            For stochastic functions, function must perform batching.

        finder_str : str
            String to key into _FINDER_INITS. Identifies the critical point-
            finding algorithm to use.

        finder_kwargs: dict or None, default is None
            Dictionary with keyword arguments to provide to self.finder_init.
            If None, an empty dictionary is used.
        """
        Experiment.__init__(self)
        self.f = f

        self.finder_str = finder_str

        if finder_kwargs is None:
            self.finder_kwargs = {}
        else:
            self.finder_kwargs = finder_kwargs

        if "log_kwargs" not in self.finder_kwargs.keys():
            self.finder_kwargs.update({"log_kwargs": DEFAULT_LOG_KWARGS.copy()})

        self.finder_init = _FINDER_INITS[self.finder_str]

        self.finder = self.finder_init(self.f, **self.finder_kwargs)

    def run(self, init_theta, num_iters):
        """Execute finder on self.f, starting with init_theta, for num_iters.
        """
        self.finder.log = {}
        thetas = self.finder.run(init_theta, num_iters)
        self.runs.append(self.finder.log)
        return thetas

    @classmethod
    def from_json(cls, f, filename):
        """Given a function f and the path to a .json file,
        creates a CritFinderExperiment for f using kwargs in the .json file.
        """
        with open(filename) as fn:
            dictionary = json.load(fn)

        return cls(f, **dictionary)

    def construct_dictionary(self):
        """Construct a dictionary containing necessary information for
        reconstructing CritFinderExperiment when combined with self.f.

        See CritFinderExperiment.from_json for details.
        """
        dictionary = {"finder_kwargs": self.finder_kwargs,
                      "finder_str": self.finder_str}
        return dictionary

    def uniform(self, thetas):
        """Select a theta at random from list thetas.
        """
        return random.choice(thetas)

    def uniform_f(self, thetas):
        """Select a theta from thetas uniformly across values of self.f.

        This can be slow. Overwrite this method by calling freeze_uniform_f
        if this function needs to be called multiple times.
        """
        return self.uniform_cd(*self.sort_and_calculate_cds(thetas, self.f))

    def freeze_uniform_f(self, thetas):
        """Overwrites self.uniform_f with a function that has pre-computed
        the sorted version of thetas and the cumulative densities, supporting
        much faster random selection.
        """
        sorted_thetas, cds = self.sort_and_calculate_cds(thetas, self.f)
        self.uniform_f = lambda thetas: self.uniform_cd(sorted_thetas, cds)

    @staticmethod
    def sort_and_calculate_cds(thetas, f):
        f_thetas = [f(theta) for theta in thetas]
        min_f, max_f = min(f_thetas), max(f_thetas)
        cds = [(f_theta - min_f) / (max_f - min_f) for f_theta in f_thetas]
        thetas, cds = zip(*sorted(zip(thetas, cds), key=lambda tup: tup[1]))
        return thetas, cds

    @staticmethod
    def uniform_cd(sorted_thetas, cds):
        """Select randomly from sorted_thetas with respect to the cumulative
        density implied by cds, an equal-length list of cumulative density values
        for each element in sorted_thetas.
        """
        rand_cd = random.uniform(0, 1)
        idx = next(filter(lambda tup: tup[1] >= rand_cd, enumerate(cds)))[0]
        return sorted_thetas[idx]
