import numpy as np
from scipy.integrate import odeint


def arange_(start=0, stop=None, step=1, dtype=None):
    """ Alias to np.arange including endpoint.

    Parameters
    ----------
    start : number, optional
        Start of interval. The default value is 0.
    stop : number
        End of interval. The interval includes this value.
    step : number, optional
        Spacing between values. The default vaule is 1.
    dtype : dtype
        The type of the output array.

    Returns
    -------
    arange_ : ndarray
        Array of evenly spaced steps from start to stop.

    See Also
    --------
    np.arange : Return evenly spaced values within a given interval.

    Examples
    --------
    >>> np.arange(3,7)
    array([3, 4, 5, 6, 7])
    >>> np.arange(0,1,0.2)
    array([0, 0.2, 0.4, 0.6, 0.8, 1])
    """
    return np.arange(start=start, stop=stop+step/2, step=step, dtype=dtype)


def order_parameter(theta):
    """ Magninute R and angle psi of the complex average of phase oscillators,
        where $R \exp \psi = 1/N \Sum_{i=0}^{N-1} \exp \ii \theta_i$. 
        
    Parameters
    ----------
    theta : ndarray of shape (N) or (M, N)
        Phase variables of N oscilators at a given instant of time
        or at M instants of time.

    Returns
    -------
    order_parameter : ndarray of shape (2) or (M, 2)
        array([R, psi]), at a single instant of time
        or [order_parameter(theta[0,:]), ...,  order_parameter(theta[M-1,:])] for M instants of time.

    Examples
    --------
    >>> #single point in the unit circle yield R = 1.
    >>> order_parameter(np.array([0, 2*np.pi, 6*np.pi]))
    array([  1.00000000e+00,  -3.26572480e-16])

    >>> #oscillators uniformly distributed yield R = 0.
    >>> order_parameter(np.array([0, np.pi/2, np.pi, 3*np.pi/2]))
    array([  6.20633538e-17,   2.67794504e+00])
    """
    if theta.ndim == 1:
        z = sum(np.exp(theta*1j))/len(theta)
        return np.array([np.absolute(z), np.angle(z)])
    elif theta.ndim == 2:
        return np.array([order_parameter(theta[i]) for i in range(np.shape(theta)[0])]).T
    raise Exception('error')


def chop(x, p_start=0.0, p_end=1.0):
    """ Chops time series by percentages.

    Parameters
    ----------
    x : ndarray of shape (T) or (N, T)
        Time series of size T or N time series of size T.
    p_start : number, optional
        Percentage to be chopped at the begining of the time series.
    p_end : number, optional
        Chop until this percentage.

    Returns
    -------
    chop : ndarray of shape (T1) or (N, T1)
        Time series chopped.

    Examples
    --------
    >>> chop(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 0.1, 0.8)
    array([2, 3, 4, 5, 6, 7, 8])

    """
    if isinstance(x, list):
        return [chop(x[i], p_start, p_end) for i in range(len(x))]
    elif x.ndim == 1:
        return x[max(0, int(len(x) * p_start)) : min(len(x), int(len(x) * p_end))]
    raise Exception('error')


def partial_sync(theta, psi, h=0.01, s=0.2):
    """ Partial Synchronization Index of theta against psi,
    """
    if theta.ndim == 2:
        return np.array([partial_sync(theta[:,i], psi, h=h, s=s) for i in range(np.shape(theta)[1])])
    elif theta.ndim == 1:
        z = sum(h*np.exp((theta[k] - psi[k])*1j) for k in range(int(np.shape(theta)[0]*s), np.shape(theta)[0]-1))
        return np.absolute(z / (h * int(np.shape(theta)[0] * (1 - s))) )
    raise Exception('error')


def integrate_and_measure(kuramoto, theta0, tf = 300, h=0.01, s=0.2):
    #Integrate network and compute some synchronization quantifiers
    #s is the percentage of transiente discarded to evalutate these quantifiers.
    
    t = arange_(0, tf, h)
    theta = odeint(kuramoto, theta0, t, hmax=h)

    r, psi = order_parameter(theta)

    #Evaluate synchronization quantifiers
    r_without_transient = chop(r, s)

    min_r = min(r_without_transient)
    mean_r = np.mean(r_without_transient)
    max_r = max(r_without_transient)

    dot_psi = np.diff(np.unwrap(psi))/h
    mean_dot_psi = np.mean(chop(dot_psi, s))

    s_per_node = partial_sync(theta, psi, h=h, s=s)
    mean_s = np.mean(s_per_node)

    info_out = {'t': t, 'theta': theta,
                'r': r, 'psi': psi,
                'dot_psi': dot_psi, 'mean_dot_psi': mean_dot_psi,
                'min_r': min_r, 'mean_r': mean_r, 'max_r': max_r,
                'partial_sync_per_node': s_per_node, 'partial_sync': mean_s}

    return info_out


def unpack_print(out):
    return out['t'], out['theta'], out['r'], out['psi']
