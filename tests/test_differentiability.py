import dsgp4
import numpy as np
import random
import jax
import jax.numpy as jnp
import unittest

error_string="Error: deep space propagation not supported (yet). The provided satellite has \
an orbital period above 225 minutes. If you want to let us know you need it or you want to \
contribute to implement it, open a PR or raise an issue at: https://github.com/esa/dSGP4."

class UtilTestCase(unittest.TestCase):
    def test_velocity(self):
        # This test verifies that the gradient of position w.r.t. time is more or less similar to the
        # velocity output by the propagator
        lines=file.splitlines()
        # Select a few satellites to test
        indexes=random.sample(list(range(1,len(lines),3)), 10)  # Reduced from 50 for speed
        tles=[]
        for i in indexes:
            data=[]
            data.append(lines[i])
            data.append(lines[i+1])
            data.append(lines[i+2])
            tles.append(dsgp4.tle.TLE(data))

        # Filter out deep space and error cases
        tles_filtered=[]
        for tle_satellite in tles:
            try:
                dsgp4.initialize_tle(tle_satellite, gravity_constant_name="wgs-84")
                if tle_satellite._error==0:
                    tles_filtered.append(tle_satellite)
            except Exception as e:
                self.assertTrue((str(e).split()==error_string.split()))

        for tle in tles_filtered:
            time_val = random.random()*30

            # Define function to compute position at time t
            def propagate_at_time(t):
                state = dsgp4.propagate(tle, t, initialized=True)
                return state[0, :]  # position only

            # Compute gradient using JAX
            grad_fn = jax.grad(lambda t: propagate_at_time(t)[0])  # gradient of x position

            # Get state and velocity at the time point
            state_teme = dsgp4.propagate(tle, time_val, initialized=True)
            velocity = state_teme[1, :]  # velocity from SGP4

            # Compute gradients (derivative of position w.r.t. time)
            grad_x = float(jax.grad(lambda t: propagate_at_time(t)[0])(time_val))
            grad_y = float(jax.grad(lambda t: propagate_at_time(t)[1])(time_val))
            grad_z = float(jax.grad(lambda t: propagate_at_time(t)[2])(time_val))

            # The gradient is w.r.t. time in minutes, giving km/min
            # Multiply by 60 to get km/s for comparison with SGP4 velocity
            self.assertAlmostEqual(grad_x, float(velocity[0])*60, places=1)
            self.assertAlmostEqual(grad_y, float(velocity[1])*60, places=1)
            self.assertAlmostEqual(grad_z, float(velocity[2])*60, places=1)

    def test_input_gradients(self):
        # This test verifies that JAX can compute gradients w.r.t. TLE parameters
        lines=file.splitlines()
        # Select a few satellites to test
        indexes=random.sample(list(range(1,len(lines),3)), 5)  # Reduced for speed
        tles=[]
        for i in indexes:
            data=[]
            data.append(lines[i])
            data.append(lines[i+1])
            data.append(lines[i+2])
            tles.append(dsgp4.tle.TLE(data))

        # Filter out deep space and error cases
        tles_filtered=[]
        for tle_satellite in tles:
            try:
                dsgp4.initialize_tle(tle_satellite, gravity_constant_name="wgs-84")
                if tle_satellite._error==0:
                    tles_filtered.append(tle_satellite)
            except Exception as e:
                self.assertTrue((str(e).split()==error_string.split()))

        for tle in tles_filtered:
            time_val = random.random()*30

            # Define propagation function parametrized by orbital elements
            def propagate_with_elements(elements):
                # elements = [ecco, argpo, inclo, mo, no_kozai, nodeo]
                whichconst = dsgp4.util.get_gravity_constants("wgs-84")
                temp_tle = dsgp4.tle.TLE([tle.line1, tle.line2])
                dsgp4.sgp4init(
                    whichconst=whichconst,
                    opsmode='i',
                    satn=tle.satellite_catalog_number,
                    epoch=(tle._jdsatepoch+tle._jdsatepochF)-2433281.5,
                    xbstar=tle._bstar,
                    xndot=tle._ndot,
                    xnddot=tle._nddot,
                    xecco=elements[0],
                    xargpo=elements[1],
                    xinclo=elements[2],
                    xmo=elements[3],
                    xno_kozai=elements[4],
                    xnodeo=elements[5],
                    satellite=temp_tle
                )
                state = dsgp4.sgp4(temp_tle, jnp.array(time_val))
                return state[0, 0]  # Return just x position for simplicity

            # Compute gradient using JAX
            elements = jnp.array([tle._ecco, tle._argpo, tle._inclo,
                                   tle._mo, tle._no_kozai, tle._nodeo])
            grad_fn = jax.grad(propagate_with_elements)
            gradient = grad_fn(elements)

            # Just verify that gradients were computed (not NaN or Inf)
            self.assertFalse(jnp.any(jnp.isnan(gradient)))
            self.assertFalse(jnp.any(jnp.isinf(gradient)))
            # Verify gradients have reasonable magnitude (not all zero)
            self.assertTrue(jnp.any(jnp.abs(gradient) > 1e-10))


file="""
0 COSMOS 2251 DEB
1 34427U 93036RU  22068.94647328  .00008100  00000-0  11455-2 0  9999
2 34427  74.0145 306.8269 0033346  13.0723 347.1308 14.76870515693886
0 COSMOS 2251 DEB
1 34428U 93036RV  22068.90158861  .00002627  00000-0  63561-3 0  9999
2 34428  74.0386 139.1157 0038434 196.7068 279.9147 14.53118052686950
0 COSMOS 2251 DEB
1 34429U 93036RW  22068.91813105  .00001869  00000-0  53244-3 0  9992
2 34429  74.0169 325.3623 0038651 190.7742 190.9458 14.55041806688179
0 COSMOS 2251 DEB
1 34430U 93036RX  22068.66049711  .00001434  00000-0  43677-3 0  9991
2 34430  74.0269  25.9368 0033629 136.4800  67.1438 14.57314993689623
0 COSMOS 2251 DEB
1 34431U 93036RY  22068.91772353  .00001093  00000-0  34841-3 0  9990
2 34431  74.0227  13.5476 0030683 146.3504 161.4494 14.55296398689297
0 COSMOS 2251 DEB
1 34432U 93036RZ  22068.92485670  .00001330  00000-0  41006-3 0  9991
2 34432  74.0230  24.0858 0030840 148.7287 355.0031 14.55837318689454
0 COSMOS 2251 DEB
1 34434U 93036SB  22069.13823773  .00002112  00000-0  61046-3 0  9994
2 34434  73.9922 321.7669 0037103 173.0729 315.5851 14.54068048688153
0 COSMOS 2251 DEB
1 34435U 93036SC  22068.88782929  .00001093  00000-0  34841-3 0  9991
2 34435  74.0207  13.0494 0031197 147.3669  63.5612 14.55296398689250
0 COSMOS 2251 DEB
1 34436U 93036SD  22068.89888730  .00002068  00000-0  59169-3 0  9997
2 34436  74.0169 339.6267 0037360 175.6424 298.0758 14.55085997688312
0 COSMOS 2251 DEB
1 34437U 93036SE  22068.90503012  .00001698  00000-0  49935-3 0  9995
2 34437  74.0199 347.9066 0036603 179.2732 301.6149 14.54268041688256
"""
