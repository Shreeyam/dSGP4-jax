import numpy
import jax.numpy as jnp
from .tle import TLE

#@torch.jit.script
def sgp4(satellite, tsince):
    """
    This function represents the SGP4 propagator. Having created the TLE object, and
    initialized the propagator (using `dsgp4.sgp4.sgp4init`), one can use this method
    to propagate the TLE at future times. The method returns the satellite position and velocity
    in km and km/s, respectively, after `tsince` minutes.

    Parameters:
    ----------------
    satellite (``dsgp4.tle.TLE``): a TLE object.
    tsince (``torch.tensor``): a torch.tensor of times since the TLE epoch in minutes.

    Returns:
    ----------------
    (``torch.tensor``): a tensor of len(tsince)x2x3 representing the satellite position and velocity in km and km/s, respectively.
    """
    #quick check to see if the satellite has been initialized
    if not isinstance(satellite, TLE):
        raise TypeError('The satellite object should be a dsgp4.tle.TLE object.')
    if not hasattr(satellite, '_radiusearthkm'):
        raise AttributeError('It looks like the satellite has not been initialized. Please use the `initialize_tle` method or directly `sgp4init` to initialize the satellite. Otherwise, if you are propagating, another option is to use `dsgp4.propagate` and pass `initialized=True` in the arguments.')
    #in case an int, float or list are passed, convert them to jax array
    if isinstance(tsince, (int, float, list)):
        tsince = jnp.array(tsince)
    mrt = jnp.zeros(tsince.shape)
    temp4 = jnp.ones(tsince.shape)*1.5e-12
    x2o3  = jnp.array(2.0 / 3.0)

    vkmpersec    = jnp.ones(tsince.shape)*(satellite._radiusearthkm * satellite._xke/60.0)

    # sgp4 error flag
    satellite._t    = jnp.array(tsince)
    satellite._error = jnp.array(0)

    #  secular gravity and atmospheric drag
    xmdf    = satellite._mo + satellite._mdot * satellite._t
    argpdf  = satellite._argpo + satellite._argpdot * satellite._t
    nodedf  = satellite._nodeo + satellite._nodedot * satellite._t
    argpm   = argpdf
    mm     = xmdf
    t2     = satellite._t * satellite._t
    nodem   = nodedf + satellite._nodecf * t2
    tempa   = 1.0 - satellite._cc1 * satellite._t
    tempe   = satellite._bstar * satellite._cc4 * satellite._t
    templ   = satellite._t2cof * t2

    if satellite._isimp != 1:
        delomg = satellite._omgcof * satellite._t
        delmtemp =  1.0 + satellite._eta * jnp.cos(xmdf)
        delm   = satellite._xmcof * \
                  (delmtemp * delmtemp * delmtemp -
                  satellite._delmo)
        temp   = delomg + delm
        mm     = xmdf + temp
        argpm  = argpdf - temp
        t3     = t2 * satellite._t
        t4     = t3 * satellite._t
        tempa  = tempa - satellite._d2 * t2 - satellite._d3 * t3 - \
                          satellite._d4 * t4
        tempe  = tempe + satellite._bstar * satellite._cc5 * (jnp.sin(mm) -
                          satellite._sinmao)
        templ  = templ + satellite._t3cof * t3 + t4 * (satellite._t4cof +
                          satellite._t * satellite._t5cof)

    nm    = jnp.array(satellite._no_unkozai)
    em    = jnp.array(satellite._ecco)
    inclm = jnp.array(satellite._inclo)

    satellite._error=jnp.any(nm<=0)*2

    am = jnp.power((satellite._xke / nm),x2o3) * tempa * tempa
    nm = satellite._xke / jnp.power(am, 1.5)
    em = em - tempe

    if satellite._error==0.:
        satellite._error=jnp.any((em>=1.0) | (em<-0.001))*1

    #  sgp4fix fix tolerance to avoid a divide by zero
    em=jnp.where(em<1.0e-6,1.0e-6,em)
    mm     = mm + satellite._no_unkozai * templ
    xlm    = mm + argpm + nodem
    emsq   = em * em
    temp   = 1.0 - emsq

    nodem = jnp.fmod(nodem, jnp.array(2*numpy.pi))

    argpm  = argpm % (2*numpy.pi)
    xlm    = xlm % (2*numpy.pi)
    mm     = (xlm - argpm - nodem) % (2*numpy.pi)

    satellite._am = jnp.array(am)
    satellite._em = jnp.array(em)
    satellite._im = jnp.array(inclm)
    satellite._Om = jnp.array(nodem)
    satellite._om = jnp.array(argpm)
    satellite._mm = jnp.array(mm)
    satellite._nm = jnp.array(nm)

    # compute extra mean quantities
    sinim = jnp.sin(inclm)
    cosim = jnp.cos(inclm)

    # add lunar-solar periodics
    ep     = em
    xincp  = inclm
    argpp  = argpm
    nodep  = nodem
    mp     = mm
    sinip  = sinim
    cosip  = cosim

    axnl = ep * jnp.cos(argpp)
    temp = 1.0 / (am * (1.0 - ep * ep))
    aynl = ep* jnp.sin(argpp) + temp * satellite._aycof
    xl   = mp + argpp + nodep + temp * satellite._xlcof * axnl

    # solve kepler's equation
    u    = (xl - nodep) % (2*numpy.pi)
    eo1  = u
    tem5 = jnp.ones(tsince.shape)
    # kepler iteration
    for _ in range(10):
        coseo1=jnp.cos(eo1)
        sineo1=jnp.sin(eo1)
        tem5   = 1.0 - coseo1 * axnl - sineo1 * aynl
        tem5   = (u - aynl * coseo1 + axnl * sineo1 - eo1) / tem5
        tem5=jnp.where(tem5>=0.95, 0.95, tem5)
        tem5=jnp.where(tem5<=-0.95, -0.95, tem5)
        #we need to break if abs value of tem5 is less than 1e-12:
        eo1    = eo1 + tem5
        if jnp.all(jnp.abs(tem5) < 1e-12):
            break

    #  short period preliminary quantities
    ecose = axnl*coseo1 + aynl*sineo1
    esine = axnl*sineo1 - aynl*coseo1
    el2   = axnl*axnl + aynl*aynl
    pl    = am*(1.0-el2)
    if satellite._error==0.:
        satellite._error=jnp.any(pl<0.)*4

    rl     = am * (1.0 - ecose)
    rdotl  = jnp.sqrt(am) * esine/rl
    rvdotl = jnp.sqrt(pl) / rl
    betal  = jnp.sqrt(1.0 - el2)
    temp   = esine / (1.0 + betal)
    sinu   = am / rl * (sineo1 - aynl - axnl * temp)
    cosu   = am / rl * (coseo1 - axnl + aynl * temp)
    su     = jnp.arctan2(sinu, cosu)
    sin2u  = (cosu + cosu) * sinu
    cos2u  = 1.0 - 2.0 * sinu * sinu
    temp   = 1.0 / pl
    temp1  = 0.5 * satellite._j2 * temp
    temp2  = temp1 * temp

    mrt   = rl * (1.0 - 1.5 * temp2 * betal * satellite._con41) + \
             0.5 * temp1 * satellite._x1mth2 * cos2u
    su    = su - 0.25 * temp2 * satellite._x7thm1 * sin2u
    xnode = nodep + 1.5 * temp2 * cosip * sin2u
    xinc  = xincp + 1.5 * temp2 * cosip * sinip * cos2u
    mvt   = rdotl - nm * temp1 * satellite._x1mth2 * sin2u / satellite._xke
    rvdot = rvdotl + nm * temp1 * (satellite._x1mth2 * cos2u +
             1.5 * satellite._con41) / satellite._xke

    # orientation vectors
    sinsu =  jnp.sin(su)
    cossu =  jnp.cos(su)
    snod  =  jnp.sin(xnode)
    cnod  =  jnp.cos(xnode)
    sini  =  jnp.sin(xinc)
    cosi  =  jnp.cos(xinc)
    xmx   = -snod * cosi
    xmy   =  cnod * cosi
    ux    =  xmx * sinsu + cnod * cossu
    uy    =  xmy * sinsu + snod * cossu
    uz    =  sini * sinsu
    vx    =  xmx * cossu - cnod * sinsu
    vy    =  xmy * cossu - snod * sinsu
    vz    =  sini * cossu

    # position and velocity (in km and km/sec)
    _mr = mrt * satellite._radiusearthkm

    r = jnp.stack((_mr * ux, _mr * uy, _mr * uz))
    v = jnp.stack(((mvt * ux + rvdot * vx) * vkmpersec,
          (mvt * uy + rvdot * vy) * vkmpersec,
          (mvt * uz + rvdot * vz) * vkmpersec))

    # decaying satellites
    if satellite._error==0.:
        satellite._error=jnp.any(mrt<1.0)*6
    return jnp.swapaxes(jnp.stack((r.squeeze(),v.squeeze()),axis=1),0,-1)#jnp.cat((r.swapaxes(0,2),v.swapaxes(0,2)),1)#jnp.stack(list(r)+list(v)).reshape(2,3)
