
from autoeis.utils import parse_initial_guess, generate_initial_guess, generate_circuit_fn
from collections.abc import Iterable, Mapping
from autoeis import parser
from scipy.optimize import least_squares
from scipy import optimize
import jax.numpy as jnp
from numpy.linalg import norm
import logging
import numpy as np
import autoeis as ae
import matplotlib.pyplot as plt
ec = ae.core.ec

log = logging.getLogger(__name__)

def get_parameter_bounds(circuit: str) -> tuple:
    """Returns a 2-element tuple of lower and upper bounds, to be used in
    SciPy's ``least_squares``.

    Parameters
    ----------
    circuit : str
        CDC string representation of the input circuit. See
        `here <https://autodial.github.io/AutoEIS/circuit.html>`_ for details.

    Returns
    -------
    tuple
        A 2-element tuple of lower and upper bounds for the circuit parameters.
    """
    bounds_dict = {
        "R": (0.0, 1e9),
        "C": (0.0, 10.0),
        "Pw": (0.0, 1e9),
        "Pn": (0.0, 1.0),
        "L": (0.0, 5.0),
    }
    types = parser.get_parameter_types(circuit)
    bounds = [bounds_dict[type_] for type_ in types]
    bounds = tuple(zip(*bounds))
    return bounds

def fit_circuit_parameters_NEW(
    circuit: str,
    freq: np.ndarray[float],
    Z: np.ndarray[complex],
    p0: Mapping[str, float] | Iterable[float] = None,
    max_iters: int = 50,
    min_iters: int = 25,
    bounds: Iterable[tuple] = None,
    max_nfev: int = None,
    ftol: float = 1e-15,
    xtol: float = 1e-15,
    tol_chi_squared: float = 1e-3,
    method: str = "bode",
    verbose: bool = False,
) -> dict[str, float]:
    """Fits and returns the parameters of a circuit to impedance data.

    Parameters
    ----------
    circuit : str
        CDC string representation of the input circuit. See
        `here <https://autodial.github.io/AutoEIS/circuit.html>`_ for details.
    freq : np.ndarray[float]
        Frequencies corresponding to the impedance data.
    Z : np.ndarray[complex]
        Impedance data.
    p0 : Mapping[str, float] | Iterable[float], optional
        Initial guess for the circuit parameters. Default is None.
    max_iters : int, optional
        Maximum number of iterations for the circuit fitter. Default is 10.
    min_iters : int, optional
        Minimum number of iterations for the circuit fitter. Default is 5.
        If ``min_iters`` is reached AND circuit fitter converges, the fitting
        process stops.
    bounds : Iterable[tuple], optional
        List of two tuples, each containing the lower and upper bounds,
        respectively, for the circuit parameters. Default is None. The order
        of the values should match the order of the circuit parameters as
        returned by ``parser.get_parameter_labels``.
    maxfev : int, optional
        Maximum number of function evaluations for the circuit fitter.
        Default is None. See ``scipy.optimize.leastsq`` for details.
    ftol : float, optional
        See ``scipy.optimize.leastsq`` for details. Default is 1e-8.
    xtol : float, optional
        See ``scipy.optimize.leastsq`` for details. Default is 1e-8.
    tol_chi_squared : float, optional
        Tolerance for the chi-squared error. This only gets triggered if
        ``min_iters`` is set. A good chi-squared value is 1e-3 or smaller.
        Default is 1e-3.
    method : str, optional
        Method to use for fitting. Choose from 'chi-squared', 'nyquist',
        'bode', and 'magnitude'. The objective function is different for each
        method:

          * 'chi-squared':
            ``w * ((Re(Zp) - Re(Z)) ** 2 + (Im(Zp) - Im(Z)) ** 2)`` where
            ``w = 1 / (Re(Z)**2 + Im(Z)**2)``
          * 'nyquist': ``[Re(Zp) - Re(Z), Im(Zp) - Im(Z)]``
          * 'bode': ``[log10(mag / mag_gt), phase - phase_gt]``
          * 'magnitude': ``abs(Z - Zp)``

        Default is 'bode'. ``Zp`` is the predicted impedance and ``Z`` is
        ground truth. ``mag`` and ``phase`` are the magnitude and phase of the
        predicted impedance, and finally ``_gt`` denotes ground truth values.

    verbose : bool, optional
        If True, prints the fitting results. Default is False.

    Returns
    -------
    dict[str, float]
        Fitted parameters as a dictionary of parameter names and values.

    Notes
    -----
    This function uses SciPy's ``least_squares`` to fit the circuit parameters.
    """

    def obj_UW(p):
        """Computes ECM error based on the Nyquist plot."""
        Z_pred = fn(freq, p)
        res = jnp.hstack((Z_pred.real - Z.real, Z_pred.imag - Z.imag))
        return res
    
    def obj_X2(p):
        """Computes ECM error based on residual-based χ2."""
        Z_pred = fn(freq, p)
        residual_real = (Z_pred.real - Z.real)
        residual_imag = (Z_pred.imag - Z.imag)
        weight = 1 / np.sqrt(Z.real**2 + Z.imag**2)
        res = jnp.hstack((residual_real*weight, residual_imag*weight))
        return res

    def obj_PW(p):
        """Computes ECM error based on residual-based χ2."""
        Z_pred = fn(freq, p)
        residual_real = (Z_pred.real - Z.real)
        residual_imag = (Z_pred.imag - Z.imag)
        weight_real = 1 / Z.real
        weight_imag = 1 / Z.imag
        res = jnp.hstack((residual_real*weight_real, residual_imag*weight_imag))
        return res
    
    def obj_B(p):
        """Computes ECM error based on the Bode plot."""
        Z_pred = fn(freq, p)
        mag = jnp.abs(Z_pred)
        phase = jnp.angle(Z_pred)
        res = jnp.hstack((mag - mag_gt, phase - phase_gt))
        # res = jnp.hstack((mag - mag_gt, phase - phase_gt))
        return res
    
    def obj_log_B(p):
        """Computes ECM error based on the Bode plot."""
        Z_pred = fn(freq, p)
        mag = jnp.abs(Z_pred)
        phase = jnp.angle(Z_pred)
        res = jnp.hstack((jnp.log10(mag) - jnp.log10(mag_gt), phase - phase_gt))
        # res = jnp.hstack((mag - mag_gt, phase - phase_gt))
        return res
    
    def obj_log_BW(p):
        """Computes ECM error based on the Bode plot."""
        Z_pred = fn(freq, p)
        mag = jnp.abs(Z_pred)
        phase = jnp.angle(Z_pred)
        res = jnp.hstack((jnp.log10(mag / mag_gt)/jnp.log10(mag_gt), (phase - phase_gt)/phase_gt))
        # res = jnp.hstack((mag - mag_gt, phase - phase_gt))
        return res
    
    def obj_chi_squared(p):
        """Computes ECM error based on residual-based χ2."""
        Z_pred = fn(freq, p)
        residual = (Z_pred.real - Z.real) ** 2 + (Z_pred.imag - Z.imag) ** 2
        weight = 1 / (Z.real**2 + Z.imag**2)
        return residual * weight

    def obj_phase_chi(p):
        """Computes ECM error based on the Bode plot."""
        Z_pred = fn(freq, p)

        residual = (Z_pred.real - Z.real) ** 2 + (Z_pred.imag - Z.imag) ** 2
        weight = 1 / (Z.real**2 + Z.imag**2)
    
        phase = jnp.angle(Z_pred)
        res = jnp.hstack(((phase - phase_gt)/phase_gt, residual * weight))
        return res
    
    def obj_mag(p):
        """Computes ECM error based on the magnitude of impedance deviation."""
        Z_pred = fn(freq, p)
        res = jnp.abs(Z - Z_pred)
        return res

    msg = f"Invalid method: {method}. Use 'chi-squared', 'nyquist', 'bode', or 'magnitude'."
    assert method in ["UW", "X2", "PW", "B", "log-B", "log-BW"], msg
    assert len(freq) == len(Z), "Length of frequency and impedance data must match."

    fn = generate_circuit_fn(circuit, jit=True)
    obj = {
        "UW": obj_UW,
        "X2": obj_X2,
        "PW": obj_PW,
        "B": obj_B,
        "log-B" : obj_log_B,
        "log-BW": obj_log_BW

    }[method]

    mag_gt = jnp.abs(Z)
    phase_gt = jnp.angle(Z)

    # Sanitize initial guess
    p0 = parse_initial_guess(p0) if p0 is not None else generate_initial_guess(circuit)
    num_params = parser.count_parameters(circuit)
    assert len(p0) == num_params, "Wrong number of parameters in initial guess."

    # Assemble kwargs for curve_fit
    bounds = get_parameter_bounds(circuit) if bounds is None else bounds
    kwargs = {"x0": p0, "bounds": bounds, "max_nfev": max_nfev, "ftol": ftol, "xtol": xtol}

    # Ensure p0 is not out-of-bounds
    if p0 is not None:
        for i, (lower, upper) in enumerate(zip(*bounds)):
            p0[i] = np.clip(p0[i], lower, upper)

    # Fit circuit parameters by brute force
    min_iters = max_iters if min_iters is None else min_iters
    err_min = np.inf

    converged = False
    for i in range(max_iters):
        res = least_squares(obj, verbose=verbose, **kwargs)
        if (err := norm(obj(res.x))) < err_min:
            err_min = err
            p0 = res.x
            jacobian = res.jac
            residuals = res.fun
        X2 = obj_chi_squared(res.x).mean()
        r2_mag = ae.metrics.r2_score(jnp.abs(Z), jnp.abs(fn(freq, p0)))
        r2_phase = ae.metrics.r2_score(jnp.angle(Z), jnp.angle(fn(freq, p0)))
        if X2 < tol_chi_squared and r2_mag > 0.9  and r2_phase > 0.9:
            converged = True
        if i + 1 >= min_iters and converged:
            break
        kwargs["x0"] = generate_initial_guess(circuit)

    try:
        res_var = np.sum(residuals**2) / (len(residuals) - len(p0))
        cov = np.linalg.inv(jacobian.T @ jacobian) * res_var
        perror = np.sqrt(np.diag(cov))
    except:
        perror = None

    r2_mag = ae.metrics.r2_score(jnp.abs(Z), jnp.abs(fn(freq, p0)))
    r2_phase = ae.metrics.r2_score(jnp.angle(Z), jnp.angle(fn(freq, p0)))
    X2 = obj_chi_squared(p0).mean()
    log.info(
        f"Converged in {i+1} iterations with "
        f"X^2 = {X2:.3e}, R^2 (|Z|) = {r2_mag:.4f}, R^2 (phase) = {r2_phase:.4f}"
    )

    if err_min == np.inf:
        raise DivergenceError(
            "Failed to fit the circuit parameters. Try increasing 'iters' or "
            "'maxfev', or narrow down the search by providing 'bounds'."
        )

    variables = parser.get_parameter_labels(circuit)
    return dict(zip(variables, p0)), X2, r2_mag, r2_phase, perror, converged

class CustomStep:
    def __init__(self, scales):
        self.scales = np.array(scales)
        
    def __call__(self, x):
        # Random normal step, scaled per parameter
        step = np.random.normal(0, 1, size=len(x)) * self.scales
        return x + step

def chi_obj_func(exp_Z, predicted_Z, number_parameters):
    w = np.real(exp_Z)**2 + np.imag(exp_Z)**2
    n = len(exp_Z)
    v = n - number_parameters # degree of freedom
    chi_square =  np.sum( ( (np.real(exp_Z) - np.real(predicted_Z))**2  +  (np.imag(exp_Z) - np.imag(predicted_Z))**2 ) / w ) / v
    return chi_square
        
def fit_circuit_global_min(
    circuit: str,
    freq: np.ndarray[float],
    Z: np.ndarray[complex],
    p0: Mapping[str, float] | Iterable[float] = None,
    bounds: Iterable[tuple] = None,
    method: str = "log-B",
    tol_chi_squared: float = 1e-2,
    seed: int = None,
) -> dict[str, float]:
    """Fits and returns the parameters of a circuit to impedance data.

    Parameters
    ----------
    circuit : str
        CDC string representation of the input circuit. See
        `here <https://autodial.github.io/AutoEIS/circuit.html>`_ for details.
    freq : np.ndarray[float]
        Frequencies corresponding to the impedance data.
    Z : np.ndarray[complex]
        Impedance data.
    p0 : Mapping[str, float] | Iterable[float], optional
        Initial guess for the circuit parameters. Default is None.
   
    bounds : Iterable[tuple], optional
        List of two tuples, each containing the lower and upper bounds,
        respectively, for the circuit parameters. Default is None. The order
        of the values should match the order of the circuit parameters as
        returned by ``parser.get_parameter_labels``.

    method : str, optional
        Method to use for fitting. Choose from 'chi-squared', 'nyquist',
        'bode', and 'magnitude'. The objective function is different for each
        method: "UW", "X2", "PW", "B", "log-B", "log-BW"

    Returns
    -------
    dict[str, float]
        Fitted parameters as a dictionary of parameter names and values.

    Notes
    -----
    This function uses SciPy's ``basinhopping`` to fit the circuit parameters.
    """
    def obj_UW(p):
        """Computes ECM error based on the Nyquist plot."""
        Z_pred = fn(freq, p)
        res = jnp.hstack((Z_pred.real - Z.real, Z_pred.imag - Z.imag))
        res = jnp.sum(res**2)
        return res
    
    def obj_X2(p):
        """Computes ECM error based on residual-based χ2."""
        Z_pred = fn(freq, p)
        residual_real = (Z_pred.real - Z.real)
        residual_imag = (Z_pred.imag - Z.imag)
        weight = 1 / np.sqrt(Z.real**2 + Z.imag**2)
        res = np.sum((residual_real*weight)**2 + (residual_imag*weight)**2)
        return res

    def obj_PW(p):
        """Computes ECM error based on residual-based χ2."""
        Z_pred = fn(freq, p)
        residual_real = (Z_pred.real - Z.real)
        residual_imag = (Z_pred.imag - Z.imag)
        weight_real = 1 / Z.real
        weight_imag = 1 / Z.imag

        # res = jnp.hstack((residual_real*weight_real, residual_imag*weight_imag))
        # res = jnp.sum(res**2)

        res = np.sum((residual_real*weight_real)**2 + (residual_imag*weight_imag)**2)

        return res
    
    def obj_B(p):
        """Computes ECM error based on the Bode plot."""
        Z_pred = fn(freq, p)
        mag = jnp.abs(Z_pred)
        phase = jnp.angle(Z_pred)
        res = jnp.hstack((mag - mag_gt, phase - phase_gt))

        res = jnp.sum(res**2)
        return res
    
    def obj_log_B(p):
        """Computes ECM error based on the Bode plot."""
        Z_pred = fn(freq, p)
        mag = jnp.abs(Z_pred)
        phase = jnp.angle(Z_pred)
        res = jnp.hstack((jnp.log10(mag) - jnp.log10(mag_gt), phase - phase_gt))

        res = jnp.sum(res**2)
        return res
    
    def obj_log_BW(p):
        """Computes ECM error based on the Bode plot."""
        Z_pred = fn(freq, p)
        mag = jnp.abs(Z_pred)
        phase = jnp.angle(Z_pred)
        res = jnp.hstack((jnp.log10(mag / mag_gt)/jnp.log10(mag_gt), (phase - phase_gt)/phase_gt))

        res = jnp.sum(res**2)
        return res
    
    def obj_chi_squared(p):
        """Computes ECM error based on residual-based χ2."""
        Z_pred = fn(freq, p)
        residual = (Z_pred.real - Z.real) ** 2 + (Z_pred.imag - Z.imag) ** 2
        weight = 1 / (Z.real**2 + Z.imag**2)
        return residual * weight

    def obj_phase_chi(p):
        """Computes ECM error based on the Bode plot."""
        Z_pred = fn(freq, p)

        residual = (Z_pred.real - Z.real) ** 2 + (Z_pred.imag - Z.imag) ** 2
        weight = 1 / (Z.real**2 + Z.imag**2)
    
        phase = jnp.angle(Z_pred)
        res = jnp.hstack(((phase - phase_gt)/phase_gt, residual * weight))
        return res

    msg = f"Invalid method: {method}. Use 'chi-squared', 'nyquist', 'bode', or 'magnitude'."
    assert method in ["UW", "X2", "PW", "B", "log-B", "log-BW"], msg
    assert len(freq) == len(Z), "Length of frequency and impedance data must match."

    fn = generate_circuit_fn(circuit, jit=True)
    obj = {
        "UW": obj_UW,
        "X2": obj_X2,
        "PW": obj_PW,
        "B": obj_B,
        "log-B" : obj_log_B,
        "log-BW": obj_log_BW

    }[method]

    mag_gt = jnp.abs(Z)
    phase_gt = jnp.angle(Z)

    # Sanitize initial guess
    p0 = parse_initial_guess(p0) if p0 is not None else generate_initial_guess(circuit)
    num_params = parser.count_parameters(circuit)
    assert len(p0) == num_params, "Wrong number of parameters in initial guess."

    # Assemble kwargs for curve_fit
    bounds = get_parameter_bounds(circuit) if bounds is None else bounds

    # Ensure p0 is not out-of-bounds
    if p0 is not None:
        for i, (lower, upper) in enumerate(zip(*bounds)):
            p0[i] = np.clip(p0[i], lower, upper)

    class BasinhoppingBounds(object):
        """ Adapted from the basinhopping documetation
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html
        """
        def __init__(self, xmin, xmax):
            self.xmin = np.array(xmin)
            self.xmax = np.array(xmax)
        def __call__(self, **kwargs):
            x = kwargs['x_new']
            tmax = bool(np.all(x <= self.xmax))
            tmin = bool(np.all(x >= self.xmin))
            return tmax and tmin

    kwargs = {}
    if seed is not None:
        kwargs = {'seed': seed}
    
    basinhopping_bounds = BasinhoppingBounds(xmin=bounds[0],
                                             xmax=bounds[1])
    # minimizer_kwargs = {
    # "method": "BFGS",
    # }
    results = optimize.basinhopping(obj, x0=p0,
                           accept_test=basinhopping_bounds, **kwargs)
    p0 = results.x
    # Calculate perror
    
    try:
        jac = results.lowest_optimization_result['jac'][np.newaxis]
        # jacobian -> covariance
        # https://stats.stackexchange.com/q/231868
        pcov = np.linalg.inv(np.dot(jac.T, jac)) * obj(p0) ** 2
        # covariance -> perror (one standard deviation
        # error estimates for fit parameters)
        perror = np.sqrt(np.diag(pcov))
    except (ValueError, np.linalg.LinAlgError):
        perror = None

    r2_mag = ae.metrics.r2_score(jnp.abs(Z), jnp.abs(fn(freq, p0)))
    r2_phase = ae.metrics.r2_score(jnp.angle(Z), jnp.angle(fn(freq, p0)))
    X2 = obj_chi_squared(p0).mean()

    converged = False
    if X2 < tol_chi_squared and r2_mag > 0.9  and r2_phase > 0.9:
        converged = True
    log.info(
        f"Converged in {i+1} iterations with "
        f"X^2 = {X2:.3e}, R^2 (|Z|) = {r2_mag:.4f}, R^2 (phase) = {r2_phase:.4f}"
    )


    variables = parser.get_parameter_labels(circuit)
    return dict(zip(variables, p0)), X2, r2_mag, r2_phase, perror, converged


def plot_nyquist_bode(predicted_Z, exp_Z, experiment_freq, titile_nyquist, title_bode_phase, title_bode_mag,  show = True, save = False, name= None, dir = None, convergance= None):
    fig, axes = plt.subplots(3, 1, figsize=(5, 10))  # 3 rows, 1 column
    axes[0].scatter(exp_Z.real, -exp_Z.imag, label='True')
    axes[0].scatter(predicted_Z.real, -predicted_Z.imag, label='predicted')
    axes[0].legend(loc='upper left', fontsize='medium')
    axes[0].set_xlabel("Re(Z)")
    axes[0].set_ylabel("-Im(Z)")
    axes[0].set_title(titile_nyquist)

    axes[1].scatter(experiment_freq, np.angle(exp_Z, deg=True), label="True")
    axes[1].scatter(experiment_freq, np.angle(predicted_Z, deg=True), label="Predicted")
    axes[1].legend(loc='upper left', fontsize='medium')
    axes[1].set_xlabel("frequency (Hz)")
    axes[1].set_ylabel(r"$\phi$ (deg)")
    axes[1].set_xscale("log")
    axes[1].set_title(title_bode_phase)

    axes[2].scatter(experiment_freq, np.abs(exp_Z), label="True")
    axes[2].scatter(experiment_freq, np.abs(predicted_Z), label="Predicted")
    axes[2].legend(loc='upper left', fontsize='medium')
    axes[2].set_xlabel("frequency (Hz)")
    axes[2].set_ylabel("magnitude")
    axes[2].set_xscale("log")
    axes[2].set_title(title_bode_mag)

    fig.suptitle(f"LS convergance: {convergance}", fontweight='bold')

    plt.tight_layout()

    if save:
        plt.savefig(f"{dir}/{name}_combined.png")

    if show:
        plt.show()

    plt.close()