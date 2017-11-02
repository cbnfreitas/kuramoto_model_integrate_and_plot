import numpy as np
from scipy.integrate import odeint
from ddeint import ddeint


class KuramotoModel:
    """ Class to hold the Kuramoto Model configuration and perform numerical integration.

        It instantiates a callable object f in such a way that f(theta, t) returns the value of the Kuramoto Model
        vector field, where theta is the N-dimensional state variable of the system and t is the time variable.

        Attributes
        ----------
        N : int
            Quantity of oscillators in the network.
        w : ndarray of shape (N)
            Natural frequency of each oscillator.
        A : list of ndarray of size (N)
            Coupling graph as adjacency list, i.e., A[i] contains the neighbor list of the i-th vertex.
        d : ndarray of shape (N)
            In-degree of the each vertex of the coupling graph.
        c : float
            Coupling parameter.
        tau : float
            Overall time delay among nodes.
        d_theta : ndarray of shape (N)
            Value evaluated of the vector field.
    """

    def __init__(self, w, E, c=1, tau=0):
        """

        Parameters
        ----------
        w : ndarray of shape (N)
            Natural frequency of each oscillator.
        E : ndarray of shape (Ne,2)
            Coupling graph as directed edge list of size Ne, i.e,
            e[i] = [p, q] means that the p -> q is the i-th directed edge of the coupling graph.
        c : float
            Coupling parameter.
        tau : float
            Overall time delay among nodes.
        """
        self.N = w.size
        self.w = w
        self.A = [E[E[:, 0] == i, 1].astype(int) for i in range(0, self.N)]  # Edge list E into adjacency list A.
        self.d = [len(self.A[i]) for i in range(0, self.N)]
        self.c = c
        self.d_theta = np.zeros(self.N)
        self.tau = tau

    def __call__(self, theta, t=0):
        """Evaluate the instance of the Kuramoto Model vector field with the given attributes.

        Parameters
        ----------
        theta : ndarray of shape (N)
            The phase value of the N oscillators in the network. It must be a function if tau>0.
        t: float
            The time variable. This parameter allows compatibility with odeint and ddeint from scipy.integrate.
        tau: float
            Overall time delay among nodes. This parameter is included here to allow compatiblitly with ddeint.

        Returns
        -------
        d_theta : ndarray of shape (N)
            The evaluated value, that is, d_theta = f(theta, t)

        """
        if self.tau == 0:
            theta_t, theta_tau = theta, theta
        else:
            theta_t, theta_tau = theta(t), theta(t - self.tau)

        for i in range(0, self.N):
            self.d_theta[i] = self.w[i] + \
                              (self.c / self.d[i]) * sum(np.sin(theta_tau[j] - theta_t[i]) for j in self.A[i])

        return self.d_theta

    def integrate(self, theta0, tf=100, h=0.01):
        """Numerical integation of the Kuramoto Model with odeint (tau=0) or ddeint (tau>0).

        Parameters
        ----------
        theta0 : ndarray of shape (N)
            Initial condition. If tau>0, (lambda t: theta0 - t * self.w) is used as initial condition.
        tf : float
            The end of the integration interval. It starts at t=0.
        h : float
            Numerical integration step.

        Returns
        -------
        t : ndarray
            Time discretization points, t = [0, h, ..., tf-h, tf].
        theta : ndarray of shape (len(t), N)
            Numerical solution for each time in t.
        """

        t = np.arange(0, tf + h / 2, h)
        if self.tau == 0:
            theta = odeint(self, theta0, t, hmax=h)
        else:
            theta0_hist = lambda t: theta0 - t * self.w
            theta = ddeint(self, theta0_hist, t)

        return t, theta
