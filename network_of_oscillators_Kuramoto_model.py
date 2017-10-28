import numpy as np


class KuramotoModel:
    """ Class to hold the Kuramoto Model configuration and evaluate its vector field to allow numerical integration.

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
            d_theta : ndarray of shape (N)
                Value evaluated of the vector field.
    """

    def __init__(self, w, E, c=1):
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
        """
        self.N = w.size
        self.w = w
        self.A = [E[E[:, 0] == i, 1].astype(int) for i in range(0, self.N)]
        self.d = [len(self.A[i]) for i in range(0, self.N)]
        self.c = c
        self.d_theta = np.zeros(self.N)

    def __call__(self, theta, t=0):
        """Evaluate the instance of the Kuramoto Model vector field with the given attributes.

        Parameters
        ----------
        theta : ndarray of shape (N)
            The phase value of the N oscillators in the network.
        t: float
            The time variable. This parameter allows compatibility with odeint from scipy.integrate.

        Returns
        -------
        d_theta : ndarray of shape (N)
            The evaluated value, that is, d_theta = f(theta, t)

        """
        for i in range(0, self.N):
            self.d_theta[i] = self.w[i] + (self.c / self.d[i]) * sum(np.sin(theta[j] - theta[i]) for j in self.A[i])
        return self.d_theta
