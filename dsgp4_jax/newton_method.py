import numpy as np
import jax
import jax.numpy as jnp
from .sgp4 import sgp4
from .sgp4init import sgp4init
from . import util
from .tle import TLE

def update_TLE(old_tle, y0):
    """
    This function updates the TLE object with the new keplerian elements.

    Parameters:
    ----------------
    old_tle (``dsgp4.TLE``): The old TLE object to be updated.
    y0 (``torch.tensor``): The new keplerian elements.

    Returns:
    ----------------
    TLE: The updated TLE object.
    """
    xpdotp = 1440.0 / (2.0 * np.pi)
    mean_motion = float(y0[4]) * xpdotp * (np.pi / 43200.0)

    tle_elements = {
        'b_star': old_tle._bstar,
        'raan': float(y0[5]),
        'eccentricity': float(y0[0]),
        'argument_of_perigee': float(y0[1]),
        'inclination': float(y0[2]),
        'mean_anomaly': float(y0[3]),
        'mean_motion': mean_motion,
        'mean_motion_first_derivative': old_tle.mean_motion_first_derivative,
        'mean_motion_second_derivative': old_tle.mean_motion_second_derivative,
        'epoch_days': old_tle.epoch_days,
        'epoch_year': old_tle.epoch_year,
        'classification': old_tle.classification,
        'satellite_catalog_number': old_tle.satellite_catalog_number,
        'ephemeris_type': old_tle.ephemeris_type,
        'international_designator': old_tle.international_designator,
        'revolution_number_at_epoch': old_tle.revolution_number_at_epoch,
        'element_number': old_tle.element_number,
    }

    return TLE(tle_elements)

def initial_guess_tle(time_mjd, tle_object, gravity_constant_name="wgs-84"):
    """
    This function generates an initial guess for the TLE object at a given time.
    It uses the current TLE object to propagate the state and extract the keplerian elements.
    The function returns a new TLE object with the updated elements.
    
    Parameters:
    ----------------
    time_mjd (``float``): The time in MJD format.
    tle_object (``dsgp4.TLE``): The TLE object to be updated.
    gravity_constant_name (``str``): The name of the gravity constant to be used. Default is "wgs-84".
    
    Returns:
    ----------------
    TLE: The updated TLE object.
    """
    #then we need to propagate the state, and extract the keplerian elements:
    util.initialize_tle(tle_object,gravity_constant_name=gravity_constant_name)
    tsince=(time_mjd-util.from_datetime_to_mjd(tle_object._epoch))*1440.
    target_state=np.array(util.propagate(tle_object, tsince))*1e3
    _,mu_earth,_,_,_,_,_,_=util.get_gravity_constants(gravity_constant_name)
    mu_earth=float(mu_earth)*1e9
    kepl_el=util.from_cartesian_to_keplerian(target_state[0],target_state[1],mu_earth)

    # Check if we're at the original epoch (zero-time case)
    # If so, preserve exact epoch values to avoid precision loss
    if abs(tsince) < 1e-6:  # Less than ~0.06 seconds
        epoch_year = tle_object.epoch_year
        epoch_days = tle_object.epoch_days
    else:
        # For non-zero time, compute new epoch from datetime
        datetime_obj = util.from_mjd_to_datetime(time_mjd)
        epoch_year = datetime_obj.year
        epoch_days = util.from_datetime_to_fractional_day(datetime_obj)

    #we need to convert the keplerian elements to TLE elements:
    data = dict(
                satellite_catalog_number=tle_object.satellite_catalog_number,
                classification=tle_object.classification,
                international_designator=tle_object.international_designator,
                epoch_year=epoch_year,
                epoch_days=epoch_days,
                ephemeris_type=tle_object.ephemeris_type,
                element_number=tle_object.element_number,
                revolution_number_at_epoch=tle_object.revolution_number_at_epoch,
                mean_motion=np.sqrt(mu_earth/((kepl_el[0])**(3.0))),
                mean_motion_first_derivative=tle_object.mean_motion_first_derivative,
                mean_motion_second_derivative=tle_object.mean_motion_second_derivative,
                eccentricity=kepl_el[1],
                inclination=kepl_el[2],
                argument_of_perigee=kepl_el[4],
                raan=kepl_el[3],
                mean_anomaly=kepl_el[5],
                b_star=tle_object.b_star)
    return TLE(data)

def _propagate(x, tle_sat, tsince, gravity_constant_name="wgs-84"):
    whichconst=util.get_gravity_constants(gravity_constant_name)
    sgp4init(whichconst=whichconst,
                        opsmode='i',
                        satn=tle_sat.satellite_catalog_number,
                        epoch=(tle_sat._jdsatepoch+tle_sat._jdsatepochF)-2433281.5,
                        xbstar=tle_sat._bstar,
                        xndot=tle_sat._ndot,
                        xnddot=tle_sat._nddot,
                        xecco=x[0],
                        xargpo=x[1],
                        xinclo=x[2],
                        xmo=x[3],
                        xno_kozai=x[4],
                        xnodeo=x[5],
                        satellite=tle_sat)
    state=sgp4(tle_sat, tsince*jnp.ones((1,1)))
    return state

def newton_method(tle0, time_mjd, max_iter=20, new_tol=1e-12, verbose=False, target_state=None, gravity_constant_name="wgs-84"):
    """
    This function implements the Newton-Raphson method to find the TLE elements that match a given target state.
    It uses the SGP4 propagator to propagate the TLE elements and compare them with the target state.
    The function returns the updated TLE object and the final state vector.
    Parameters:
    ----------------
    tle0 (``dsgp4.TLE``): The initial TLE object to be updated.
    time_mjd (``float``): The time in MJD format.
    max_iter (``int``): The maximum number of iterations for the Newton-Raphson method. Default is 50.
    new_tol (``float``): The tolerance for convergence. Default is 1e-12.
    verbose (``bool``): If True, prints the convergence information. Default is False.
    target_state (``torch.tensor``): The target state vector to be matched. If None, the function will propagate the initial TLE object to find the target state.
    gravity_constant_name (``str``): The name of the gravity constant to be used. Default is "wgs-84".

    Returns:
    ----------------
    TLE: The updated TLE object.
    torch.tensor: The tensor of the updated elements.
    """
    if target_state is None:
        util.initialize_tle(tle0,gravity_constant_name=gravity_constant_name)
        target_state=util.propagate(tle0, (time_mjd-util.from_datetime_to_mjd(tle0._epoch))*1440.)

    i,tol=0,1e9
    next_tle=initial_guess_tle(time_mjd, tle0,gravity_constant_name=gravity_constant_name)
    y0 = jnp.array([
                        next_tle._ecco,
                        next_tle._argpo,
                        next_tle._inclo,
                        next_tle._mo,
                        next_tle._no_kozai,
                        next_tle._nodeo,
                    ])

    # Compute tsince relative to next_tle's epoch
    # For zero-time case: if time_mjd equals tle0's MJD and next_tle's epoch is approximately same,
    # use tsince=0.0 to avoid floating point precision issues
    tsince = (time_mjd - util.from_datetime_to_mjd(next_tle._epoch)) * 1440.
    target_tsince = (time_mjd - util.from_datetime_to_mjd(tle0._epoch)) * 1440.

    # If both tsinces are very small (zero-time case), use 0.0 to avoid precision issues
    if abs(target_tsince) < 1e-6 and abs(tsince) < 1e-3:
        tsince = 0.0

    def propagate_fn(x):
        return _propagate(x,next_tle,tsince,gravity_constant_name=gravity_constant_name).flatten()

    # Compute Jacobian function once
    jacobian_fn = jax.jacobian(propagate_fn)

    while i<max_iter and tol>new_tol:
        # Compute state and residual
        current_state = propagate_fn(y0).reshape(2, 3)
        F = (current_state - np.array(target_state)).flatten()
        tol = np.linalg.norm(F)

        if tol<new_tol:
            if verbose:
                print(f'Solution F(y) = {tol}, converged in {i} iterations')
            return next_tle, y0

        # Compute Jacobian
        DF = np.array(jacobian_fn(y0))

        # Solve for update using least squares (more robust than direct solve for ill-conditioned systems)
        dY, _, _, _ = np.linalg.lstsq(DF, -F, rcond=None)

        # Avoid negative or >1 eccentricity
        if y0[0]+dY[0]<0:
            dY[0] = 1e-10 - float(y0[0])
        if y0[0]+dY[0]>1.:
            dY[0] = 1-1e-10 - float(y0[0])

        # Update the state
        y0 = y0 + jnp.array(dY)
        next_tle = update_TLE(next_tle, y0)
        i+=1
    if verbose:
        print("Solution not found, returning best found so far")
        print(f"F(y): {tol:.3e}")
    return next_tle, y0