# Copied from http://codegist.net/snippet/python/ddeintpy_julianeagu_python
# Obs: This DDE integrator uses linear interpolation.
# I still need to determine the convergency order of this method...

# REQUIRES PACKAGES Numpy AND Scipy INSTALLED
import numpy as np
import scipy.integrate
import scipy.interpolate

class ddeVar:
    """ special function-like variables for the integration of DDEs """

    def __init__(self,g,tc=0):
        """ g(t) = expression of Y(t) for t<tc """

        self.g = g
        self.tc= tc
        # We must fill the interpolator with 2 points minimum
        self.itpr = scipy.interpolate.interp1d(
            np.array([tc-1,tc]), # X
            np.array([self.g(tc),self.g(tc)]).T, # Y
            kind='linear', bounds_error=False,
            fill_value = self.g(tc))

    def update(self,t,Y):
        """ Add one new (ti,yi) to the interpolator """

        self.itpr.x = np.hstack([self.itpr.x, [t]])
        Y2 = Y if (Y.size==1) else np.array([Y]).T
        self.itpr.y = np.hstack([self.itpr.y, Y2])
        self.itpr.fill_value = Y
        self.itpr._y = self.itpr._reshape_yi(self.itpr.y)

    def __call__(self,t=0):
        """ Y(t) will return the instance's value at time t """

        return (self.g(t) if (t<=self.tc) else self.itpr(t))

class dde(scipy.integrate.ode):
    """ Overwrites a few functions of scipy.integrate.ode"""

    def __init__(self,f,jac=None):

        def f2(t,y,args):
            return f(self.Y,t,*args)
        scipy.integrate.ode.__init__(self,f2,jac)
        self.set_f_params(None)

    def integrate(self, t, step=0, relax=0):

        scipy.integrate.ode.integrate(self,t,step,relax)
        self.Y.update(self.t,self.y)
        return self.y

    def set_initial_value(self,Y):

        self.Y = Y #!!! Y will be modified during integration
        scipy.integrate.ode.set_initial_value(self, Y(Y.tc), Y.tc)

def ddeint(func,g,tt,fargs=None):
    """
    Similar to scipy.integrate.odeint. Solves a Delay differential
    Equation system (DDE) defined by ``func`` with history function ``g``
    and potential additional arguments for the model, ``fargs``.
    Returns the values of the solution at the times given by the array ``tt``.

    Example:
    --------

    We will solve the delayed Lotka-Volterra system defined as

    For t < 0:
    x(t) = 1+t
    y(t) = 2-t

    For t > 0:
    dx/dt =  0.5* ( 1- y(t-d) )
    dy/dt = -0.5* ( 1- x(t-d) )

    Note that here the delay ``d`` is a tunable parameter of the model.

    ---

    import numpy as np

    def model(XY,t,d):
        x, y = XY(t)
        xd, yd = XY(t-d)
        return np.array([0.5*x*(1-yd), -0.5*y*(1-xd)])

    g = lambda t : np.array([1+t,2-t]) # 'history' at t<0
    tt = np.linspace(0,30,20000) # times for integration
    d = 0.5 # set parameter d
    yy = ddeint(model,g,tt,fargs=(d,)) # solve the DDE !

    """

    dde_ = dde(func)
    dde_.set_initial_value(ddeVar(g,tt[0]))
    dde_.set_f_params(fargs if fargs else [])
    return np.array([g(tt[0])]+[dde_.integrate(dde_.t + dt)
                                 for dt in np.diff(tt)])