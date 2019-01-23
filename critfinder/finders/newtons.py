import autograd
import autograd.numpy as np

from .minresQLP import MinresQLP as mrqlp

from .base import Finder


class NewtonMethod(Finder):

    def __init__(self, f, step_size=1.0, log_kwargs={}):
        Finder.__init__(self, f, log_kwargs=log_kwargs)
        self.H = lambda theta: np.squeeze(autograd.hessian(f)(theta))
        self.hvp = lambda theta, v: np.dot(self.H(theta), v)

        self.step_size = step_size

        self.parameters = {"step_size": step_size}

    def run(self, init_theta, iters=1):
        theta = init_theta
        self.update_logs({"theta": theta,
                          "update_direction": None,
                          "parameters": self.parameters})

        for ii in range(iters):

            update_direction = self.get_update_direction(theta)
            theta_new = self.select_update(theta, update_direction)

            self.update_logs({"theta": theta_new,
                              "update_direction": update_direction,
                              "parameters": self.parameters})

            if np.array_equal(theta, theta_new):
                return theta

            theta = theta_new

        return theta

    def get_update_direction(self, theta):
        update_direction = -np.linalg.inv(self.H(theta)).dot(self.grad_f(theta))
        return update_direction

    def select_update(self, theta, update_direction):
        return theta + self.step_size * update_direction

    def squared_grad_norm(self, theta):
        return np.sum(np.square(self.grad_f(theta)))


class NewtonPI(NewtonMethod):

    def __init__(self, f, step_size=1.0, log_kwargs={}):
        NewtonMethod.__init__(self, f, step_size=step_size, log_kwargs=log_kwargs)
        self.pinv = np.linalg.pinv

    def get_update_direction(self, theta):
        update_direction = -self.pinv(self.H(theta)).dot(self.grad_f(theta))
        return update_direction


class NewtonBTLS(NewtonMethod):

    def __init__(self, f, alpha, beta, rho, log_kwargs={}):
        NewtonMethod.__init__(self, f, log_kwargs=log_kwargs)
        self.alpha = alpha
        self.beta = beta
        self.rho = rho

        self.parameters.update({"alpha": alpha,
                                "beta": beta,
                                "rho": rho})

    def select_update(self, theta, update_direction):
        converged = self.check_convergence(theta, update_direction, self.alpha, self.rho)
        while not converged:
            self.alpha *= self.beta
            converged = self.check_convergence(theta, update_direction, self.alpha, self.rho)
        update = theta + self.alpha * update_direction
        self.alpha /= self.beta
        return update

    def check_convergence(self, theta, update_direction, alpha, rho):
        proposed_update = theta + alpha * update_direction
        updated_squared_gradient_norm = self.squared_grad_norm(self.grad_f(proposed_update))
        current_squared_gradient_norm = self.squared_grad_norm(self.grad_f(theta))
        sufficient_decrease = 2 * rho * alpha * np.dot(self.hvp(theta, update_direction).T,
                                                       self.grad_f(theta))

        return (updated_squared_gradient_norm <=
                current_squared_gradient_norm + sufficient_decrease)


class NewtonMR(NewtonBTLS):

    def __init__(self, f, alpha, beta, rho, rtol=1e-10, maxit=10,
                 log_kwargs={}):
        NewtonBTLS.__init__(self, f, alpha, beta, rho, log_kwargs=log_kwargs)
        self.rtol = rtol
        self.maxit = maxit

        self.parameters.update({"rtol": rtol,
                                "maxit": maxit})

    def get_update_direction(self, theta):
        current_hvp = lambda v: self.hvp(theta, v)
        mr_update_direction = mrqlp(current_hvp, -1 * self.grad_f(theta),
                                    rtol=self.rtol, maxit=self.maxit)[0]
        return mr_update_direction


class FastNewtonMR(NewtonMR):

    def __init__(self, f, alpha, beta, rho, rtol=1e-10, maxit=10, log_kwargs={}):
        NewtonMR.__init__(self, f, alpha, beta, rho, rtol=rtol, maxit=maxit, log_kwargs=log_kwargs)
        self.hvp = autograd.hessian_vector_product(self.f)


class NewtonTR(NewtonPI):

    def __init__(self, f, gammas, step_size=1.0, log_kwargs={}):
        NewtonPI.__init__(self, f, step_size=step_size, log_kwargs=log_kwargs)
        self.gammas = gammas
        self.Hs = [lambda theta: self.H(theta) + np.diag([gamma] * theta.shape[0])
                   for gamma in gammas]

        self.parameters.update({"gammas": gammas})

    def get_update_direction(self, theta):
        update_directions = []

        for H in self.Hs:
            update_directions.append(-self.pinv(H(theta))
                                     .dot(self.grad_f(theta)))

        return update_directions

    def select_update(self, theta, update_directions):
        best_update = theta
        best_grad_norm = self.squared_grad_norm(best_update)
        for update_direction in update_directions:
            proposed_update = theta + self.step_size * update_direction
            if self.squared_grad_norm(proposed_update) < best_grad_norm:
                best_update = proposed_update

        return best_update


class FastNewtonTR(NewtonTR):

    def __init__(self, f, gammas, step_size=1.0, log_kwargs={}, rtol=1e-10, maxit=10):
        NewtonTR.__init__(self, f, gammas, step_size=step_size, log_kwargs=log_kwargs)
        self.rtol = rtol
        self.maxit = maxit

        self.hvps = [lambda theta, v: autograd.hessian_vector_product(self.f)(theta, v) +
                     np.sum(gamma * theta) for gamma in gammas]

    def get_update_direction(self, theta):
        update_directions = []
        current_hvps = [lambda v: hvp(theta, v) for hvp in self.hvps]

        for current_hvp in current_hvps:
            mr_update_direction = mrqlp(current_hvp, -1 * self.grad_f(theta),
                                        rtol=self.rtol, maxit=self.maxit)[0]
            update_directions.append(mr_update_direction)

        return update_directions


class NewtonBTTR(NewtonPI):

    def __init__(self, f, gamma, beta,
                 rho_low=1e-4, rho_mid=0.9, rho_high=1.1,
                 step_size=1.0, log_kwargs={}):
        NewtonPI.__init__(self, f, step_size=step_size, log_kwargs=log_kwargs)
        self.gamma = gamma
        self.beta = beta
        self.set_H(gamma)

        self.parameters.update({"gamma": gamma, "beta": beta,
                                "rho_low": rho_low, "rho_mid": rho_mid,
                                "rho_high": rho_high})

    def get_update_direction(self, theta):
        update_accepted = False

        while not update_accepted:
            self.set_H(self.gamma)
            proposed_update = -self.pinv(self.H(theta)).dot(self.grad_f(theta))

            rho = self.check_criterion(theta, proposed_update)
            if (self.rho_high > rho) & (rho > self.rho_mid):
                update_accepted = True
                self.gamma *= self.beta
            elif rho < self.rho_low:
                update_accepted = False
                self.gamma *= 1 / self.beta
            else:
                update_accepted = True
                self.gamma *= 1 / self.beta

        return proposed_update

    def compute_criterion(self, theta, proposed_update):

        f_theta = self.f(theta)
        f_at_update = self.f(theta + proposed_update)

        predicted_f = self.compute_predicted_f(theta, proposed_update)

        rho = (f_theta - f_at_update) / (f_theta - predicted_f)

        return rho

    def compute_predicted_f(self, theta, proposed_update):
        constant_term = self.f(theta)
        linear_term = np.dot(self.grad_f(theta), proposed_update)
        quadratic_term = 0.5 * np.dot(proposed_update.T, self.H(theta).dot(proposed_update))
        return constant_term + linear_term + quadratic_term

    def set_H(self, gamma):
        self.H = lambda theta: self.H(theta) + np.diag([gamma] * theta.shape[0])


class FastNewtonBTTR(NewtonBTTR):
    def __init__(self, f, gamma, beta,
                 rtol=1e-10, maxit=10,
                 rho_low=1e-4, rho_mid=0.9, rho_high=1.1,
                 step_size=1.0, log_kwargs={}):
        NewtonBTTR.__init__(self, f, gamma, beta, rho_low, rho_mid, rho_high,
                            step_size=step_size, log_kwargs=log_kwargs)

        self.set_hvp(gamma)

    def get_update_direction(self, theta):
        update_accepted = False

        while not update_accepted:
            self.set_hvp(self.gamma)
            proposed_update = mrqlp(self.current_hvp, -1 * self.grad_f(theta),
                                    rtol=self.rtol, maxit=self.maxit)[0]

            rho = self.check_criterion(theta, proposed_update)
            if (self.rho_high > rho) & (rho > self.rho_mid):
                update_accepted = True
                self.gamma *= self.beta
            elif rho < self.rho_low:
                update_accepted = False
                self.gamma *= 1 / self.beta
            else:
                update_accepted = True
                self.gamma *= 1 / self.beta

        return proposed_update

    def compute_predicted_f(self, theta, proposed_update):
        constant_term = self.f(theta)
        linear_term = np.dot(self.grad_f(theta), proposed_update)
        quadratic_term = 0.5 * np.dot(proposed_update.T, self.hvp(theta, proposed_update))
        return constant_term + linear_term + quadratic_term

    def set_hvp(self, gamma):
        self.hvp = lambda theta, v: \
            autograd.hessian_vector_product(self.f)(theta, v) + np.sum(gamma * theta)
