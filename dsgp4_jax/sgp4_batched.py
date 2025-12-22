import jax.numpy as jnp
import numpy
from .tle import TLE

def sgp4_batched(satellite_batch, tsince):
    """
    This function represents the batch SGP4 propagator. 
    It resembles `sgp4`, but accepts batches of TLEs.
    Having created the TLE object, and initialized the propagator (using `dsgp4.sgp4.sgp4init`), 
    one can use this method to propagate the TLE at future times. 
    The method returns the satellite position and velocity
    in km and km/s, respectively, after `tsince` minutes.

    Parameters:
    ----------------
    satellite_batch (``dsgp4.tle.TLE``): a TLE object.
    tsince (``jnp.array``): a jnp.array of times since the TLE epoch in minutes.

    Returns:
    ----------------
    batch_state (``jnp.array``): a batch of 2x3 arrays, where the first row represents the spacecraft
                                    position (in km) and the second the spacecraft velocity (in km/s)
    """
    if not isinstance(satellite_batch, TLE):
        raise ValueError("satellite_batch should be a TLE object.")
    #in case a numpy array, list, int, or float are passed, convert them to jax array
    if isinstance(tsince, (int, float, list, numpy.ndarray)):
        tsince = jnp.array(tsince)
    if tsince.ndim!=1:
        raise ValueError("tsince should be a one dimensional tensor.")
    if len(tsince)!=len(satellite_batch._argpo):
        raise ValueError(f"in batch mode, tsince and satellite_batch shall have attributes of same length. Instead {len(tsince)} for time, and {len(satellite_batch._argpo)} for satellites' attributes found")
    if not hasattr(satellite_batch, '_radiusearthkm'):
        raise AttributeError('It looks like the satellite_batch has not been initialized. Please use the `initialize_tle` method or directly `sgp4init` to initialize the satellite_batch. Otherwise, if you are propagating, another option is to use `dsgp4.propagate` and pass `initialized=True` in the arguments.')
    
    batch_size = len(tsince)
    mrt = jnp.zeros(batch_size)
    x2o3  = jnp.array(2.0 / 3.0)

    vkmpersec    = jnp.ones(batch_size)*(satellite_batch._radiusearthkm * satellite_batch._xke/60.0)
    #  sgp4 error flag
    satellite_batch._t    = tsince
    satellite_batch._error = jnp.zeros(batch_size)

    # update for secular gravity and atmospheric drag
    xmdf    = satellite_batch._mo + satellite_batch._mdot * satellite_batch._t
    argpdf  = satellite_batch._argpo + satellite_batch._argpdot * satellite_batch._t
    nodedf  = satellite_batch._nodeo + satellite_batch._nodedot * satellite_batch._t
    argpm1   = argpdf
    mm1     = xmdf
    t2     = satellite_batch._t * satellite_batch._t
    nodem   = nodedf + satellite_batch._nodecf * t2
    tempa1   = 1.0 - satellite_batch._cc1 * satellite_batch._t
    tempe1   = satellite_batch._bstar * satellite_batch._cc4 * satellite_batch._t
    templ1   = satellite_batch._t2cof * t2


    delomg = satellite_batch._omgcof * satellite_batch._t

    delmtemp =  1.0 + satellite_batch._eta * jnp.cos(xmdf)
    delm   = satellite_batch._xmcof * \
                (delmtemp * delmtemp * delmtemp -
                satellite_batch._delmo)
    temp   = delomg + delm
    mm0     = xmdf + temp
    argpm0  = argpdf - temp
    t3     = t2 * satellite_batch._t
    t4     = t3 * satellite_batch._t
    tempa0  = tempa1 - satellite_batch._d2 * t2 - satellite_batch._d3 * t3 - \
                        satellite_batch._d4 * t4
    tempe0  = tempe1 + satellite_batch._bstar * satellite_batch._cc5 * (jnp.sin(mm0) -
                        satellite_batch._sinmao)
    templ0  = templ1 + satellite_batch._t3cof * t3 + t4 * (satellite_batch._t4cof +
                        satellite_batch._t * satellite_batch._t5cof)
    
    mm    = jnp.where(satellite_batch._isimp==0,mm0,mm1)
    argpm = jnp.where(satellite_batch._isimp==0,argpm0,argpm1)
    tempa = jnp.where(satellite_batch._isimp==0,tempa0,tempa1)
    tempe = jnp.where(satellite_batch._isimp==0,tempe0,tempe1)
    templ = jnp.where(satellite_batch._isimp==0,templ0,templ1)

    nm    = jnp.array(satellite_batch._no_unkozai)
    em    = jnp.array(satellite_batch._ecco)
    inclm = jnp.array(satellite_batch._inclo)
    satellite_batch._error = jnp.full(nm.shape[0],2) * (nm <= 0.)

    am = jnp.power((satellite_batch._xke / nm),x2o3) * tempa * tempa
    nm = satellite_batch._xke / jnp.power(am, 1.5)
    em = em - tempe

    satellite_batch._error = jnp.full(em.shape[0], 1) * ((em>=1.0) | (em<-0.001))

    em=jnp.clip(em, min=1.0e-6)
    mm     = mm + satellite_batch._no_unkozai * templ
    xlm    = mm + argpm + nodem
    emsq   = em * em
    temp   = 1.0 - emsq

    nodem = jnp.fmod(nodem, jnp.array(2*numpy.pi))

    argpm  = argpm % (2*numpy.pi)
    xlm    = xlm % (2*numpy.pi)
    mm     = (xlm - argpm - nodem) % (2*numpy.pi)

    satellite_batch._am = jnp.array(am)
    satellite_batch._em = jnp.array(em)
    satellite_batch._im = jnp.array(inclm)
    satellite_batch._Om = jnp.array(nodem)
    satellite_batch._om = jnp.array(argpm)
    satellite_batch._mm = jnp.array(mm)
    satellite_batch._nm = jnp.array(nm)
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
    aynl = ep* jnp.sin(argpp) + temp * satellite_batch._aycof
    xl   = mp + argpp + nodep + temp * satellite_batch._xlcof * axnl

    #  solve kepler's equation
    u    = (xl - nodep) % (2*numpy.pi)
    eo1  = u
    tem5 = jnp.ones(tsince.shape[0])

    for _ in range(10):
        coseo1=jnp.cos(eo1)
        sineo1=jnp.sin(eo1)
        tem5   = 1.0 - coseo1 * axnl - sineo1 * aynl
        tem5   = (u - aynl * coseo1 + axnl * sineo1 - eo1) / tem5
        tem5=jnp.where(tem5>=0.95, 0.95, tem5)
        tem5=jnp.where(tem5<=-0.95, -0.95, tem5)
        eo1    = eo1 + tem5
        if jnp.all(jnp.abs(tem5) < 1e-12):
            break

    # short period preliminary quantities
    ecose = axnl*coseo1 + aynl*sineo1
    esine = axnl*sineo1 - aynl*coseo1
    el2   = axnl*axnl + aynl*aynl
    pl    = am*(1.0-el2)
    satellite_batch._error=jnp.where(satellite_batch._error==0.,jnp.any(pl<0.)*4,1)

    rl     = am * (1.0 - ecose)
    rdotl  = jnp.sqrt(am) * esine/rl
    rvdotl = jnp.sqrt(pl) / rl
    betal  = jnp.sqrt(1.0 - el2)
    temp   = esine / (1.0 + betal)
    sinu   = am / rl * (sineo1 - aynl - axnl * temp)
    cosu   = am / rl * (coseo1 - axnl + aynl * temp)
    su     = jnp.atan2(sinu, cosu)
    sin2u  = (cosu + cosu) * sinu
    cos2u  = 1.0 - 2.0 * sinu * sinu
    temp   = 1.0 / pl
    temp1  = 0.5 * satellite_batch._j2 * temp
    temp2  = temp1 * temp

    mrt   = rl * (1.0 - 1.5 * temp2 * betal * satellite_batch._con41) + \
             0.5 * temp1 * satellite_batch._x1mth2 * cos2u
    su    = su - 0.25 * temp2 * satellite_batch._x7thm1 * sin2u
    xnode = nodep + 1.5 * temp2 * cosip * sin2u
    xinc  = xincp + 1.5 * temp2 * cosip * sinip * cos2u
    mvt   = rdotl - nm * temp1 * satellite_batch._x1mth2 * sin2u / satellite_batch._xke
    rvdot = rvdotl + nm * temp1 * (satellite_batch._x1mth2 * cos2u +
             1.5 * satellite_batch._con41) / satellite_batch._xke

    #  orientation vectors
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
    _mr = mrt * satellite_batch._radiusearthkm

    r = jnp.stack((_mr * ux, _mr * uy, _mr * uz))
    v = jnp.stack(((mvt * ux + rvdot * vx) * vkmpersec,
          (mvt * uy + rvdot * vy) * vkmpersec,
          (mvt * uz + rvdot * vz) * vkmpersec))

    satellite_batch._error=jnp.where(satellite_batch._error==0.,jnp.any(mrt<1.0)*6,satellite_batch._error)
    # r and v have shape (3, batch_size), we want output shape (batch_size, 2, 3)
    return jnp.transpose(jnp.stack((r, v), axis=0), axes=(2, 0, 1))
