import json

import autograd
import autograd.numpy as np

from .base import Finder, Logger


DEFAULT_ALPHA = 0.1
DEFAULT_BETA = 0.5
DEFAULT_RHO = 1e-4
DEFAULT_GAMMA = 0.9

DEFAULT_MINIMIZER_PARAMS = {"lr": DEFAULT_ALPHA}


class GradientNormMinimizer(Finder):
    r"""Find critical points of function f by minimizing
    auxiliary function g where
    $$
    g(theta) = \frac{1]{2}\lvert\nabla f(theta)\rvert^2
    $$

    The gradient of g is the product of the hessian with the gradient.
    This can be more efficiently computed as a hessian-vector product.
    """

    def __init__(self, f, log_kwargs={}, minimizer_str="gd", minimizer_params=None,
                 criterion_str=None, criterion_params=None):
        Finder.__init__(self, f, log_kwargs=log_kwargs)

        def g(theta):
            return 0.5 * np.sum(np.square(self.grad_f(theta)))

        self.g = g
        self.grad_g = autograd.grad(g)
        self.hvp = autograd.hessian_vector_product(self.f)
        self.fast_grad_g = lambda x: self.hvp(x, self.grad_f(x))

        self.minimizer_str = minimizer_str
        self.minimizer_params = minimizer_params or DEFAULT_MINIMIZER_PARAMS.copy()
        self.set_minimizer(minimizer_str)

        self.criterion_str = criterion_str
        if self.criterion_str is not None:
            self.criterion_params = criterion_params or {}
            self.set_criterion()

    def run(self, init_theta, num_iters, init_step=0.):
        theta = init_theta
        step = init_step
        self.update_logs({"theta": theta, "step": step})

        for ii in range(num_iters):
            step = self.minimizer(theta, step)
            theta_new = theta + step
            self.update_logs({"theta": theta_new, "step": step})

            if np.array_equal(theta, theta_new):
                return theta

            theta = theta_new

        return theta

    def setup_log(self, track_thetas=False, track_f_thetas=False, track_g_thetas=False):

        if track_thetas:
            self.loggers.append(Logger("theta", lambda step_info: step_info["theta"]))

        if track_f_thetas:
            self.loggers.append(Logger("f_theta", lambda step_info: self.f(step_info["theta"])))

        if track_g_thetas:
            self.loggers.append(Logger("g_theta", lambda step_info: self.g(step_info["theta"])))

    def set_minimizer(self, minimizer_str):
        if minimizer_str == "gd":
            self.minimizer = self.gradient_descent
            self.lr = self.minimizer_params["lr"]
        elif minimizer_str == "momentum":
            self.minimizer = self.momentum_gradient_descent
            self.lr = self.minimizer_params["lr"]
            self.momentum = self.minimizer_params["momentum"]
        elif minimizer_str == "btls":
            self.minimizer = self.btls
            self.alpha = self.minimizer_params["alpha"]
            if "beta" in self.minimizer_params.keys():
                self.beta = self.minimizer_params["beta"]
            else:
                self.beta = DEFAULT_BETA

            self.min_step_size = self.compute_min_step_size(self.alpha, self.beta)

        else:
            raise NotImplementedError

    def set_criterion(self):
        if self.criterion_str is None:
            return

        if self.criterion_str == "roosta":
            self.check_convergence = self.roosta_criterion
        elif self.criterion_str == "wolfe":
            self.check_convergence = self.wolfe_criterion
        else:
            raise NotImplementedError

        if "rho" in self.criterion_params.keys():
            self.rho = self.criterion_params["rho"]
        else:
            self.rho = DEFAULT_RHO

        if "gamma" in self.criterion_params.keys():
            self.gamma = self.criterion_params["gamma"]
        else:
            self.gamma = DEFAULT_GAMMA

    def gradient_descent(self, theta, last_step):
        return -self.lr * self.fast_grad_g(theta)

    def momentum_gradient_descent(self, theta, last_step):
        return -self.lr * self.fast_grad_g(theta) + self.momentum * last_step

    def btls(self, theta, last_step):
        update_direction = -self.fast_grad_g(theta)
        converged = self.check_convergence(theta, update_direction, self.alpha)
        while not converged:
            self.alpha *= self.beta
            if self.alpha <= self.min_step_size:
                break
            converged = self.check_convergence(theta, update_direction, self.alpha)
        step = self.alpha * update_direction
        self.alpha /= self.beta
        return step

    def roosta_criterion(self, theta, update_direction, alpha):
        proposed_update = theta + alpha * update_direction
        updated_g = self.g(proposed_update)
        current_g = self.g(theta)
        sufficient_decrease = 2 * self.rho * alpha * np.dot(self.hvp(theta, update_direction).T,
                                                            self.grad_f(theta))

        return (updated_g <=
                current_g + sufficient_decrease)

    def wolfe_criterion(self, theta, update_direction, alpha):
        proposed_update = theta + alpha * update_direction

        current_g = self.g(theta)
        new_g = self.g(proposed_update)

        current_grad_g = self.fast_grad_g(theta)
        grad_update_product = np.dot(update_direction.T, current_grad_g)

        new_grad_g = self.fast_grad_g(proposed_update)
        new_grad_update_product = np.dot(update_direction.T, new_grad_g)

        passed_armijo = new_g <= current_g + self.rho * alpha * grad_update_product

        passed_curvature = -new_grad_update_product <= -self.gamma * grad_update_product

        return passed_armijo and passed_curvature

    @staticmethod
    def compute_min_step_size(alpha, beta):
        while alpha * beta != alpha:
            alpha *= beta
        return alpha

    def to_json(self, json_path):
        dictionary = self.construct_dictionary()
        with open(json_path, "w") as fp:
            json.write(dictionary, fp)

    @classmethod
    def from_json(cls, f, json_path):
        with open(json_path) as fp:
            dictionary = json.load(fp)
        return cls(f, **dictionary)

    def construct_dictionary(self):
        dictionary = {"log_kwargs": self.log_kwargs,
                      "minimizer_str": self.minimizer_str,
                      "minimzer_params": self.minimzer_params,
                      "criterion_str": self.criterion_str,
                      "criterion_params": self.criterion_params}
        return dictionary
