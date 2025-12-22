import jax
import jax.numpy as jnp
import numpy as np
import pickle

from .util import initialize_tle, propagate, propagate_batch

def linear(W, b, x):
    """Simple linear layer: y = xW^T + b"""
    return x @ W.T + b

class mldsgp4:
    def __init__(self,
                normalization_R=6958.137,
                normalization_V=7.947155867983262,
                hidden_size=100,
                input_correction=1e-2,
                output_correction=0.8):
        """
        This class implements the ML-dSGP4 model, where dSGP4 inputs and outputs are corrected via neural networks,
        to better match simulated or observed higher-precision data.

        Parameters:
        ----------------
        normalization_R (``float``): normalization constant for x,y,z coordinates.
        normalization_V (``float``): normalization constant for vx,vy,vz coordinates.
        hidden_size (``int``): number of neurons in the hidden layers.
        input_correction (``float``): correction factor for the input layer.
        output_correction (``float``): correction factor for the output layer.
        """
        self.normalization_R = normalization_R
        self.normalization_V = normalization_V
        self.hidden_size = hidden_size

        # Initialize parameters (will be overwritten by load_model)
        # Input correction network
        self.fc1_weight = jnp.zeros((hidden_size, 6))
        self.fc1_bias = jnp.zeros(hidden_size)
        self.fc2_weight = jnp.zeros((hidden_size, hidden_size))
        self.fc2_bias = jnp.zeros(hidden_size)
        self.fc3_weight = jnp.zeros((6, hidden_size))
        self.fc3_bias = jnp.zeros(6)

        # Output correction network
        self.fc4_weight = jnp.zeros((hidden_size, 6))
        self.fc4_bias = jnp.zeros(hidden_size)
        self.fc5_weight = jnp.zeros((hidden_size, hidden_size))
        self.fc5_bias = jnp.zeros(hidden_size)
        self.fc6_weight = jnp.zeros((6, hidden_size))
        self.fc6_bias = jnp.zeros(6)

        # Correction factors
        self.input_correction = jnp.array(input_correction * jnp.ones(6))
        self.output_correction = jnp.array(output_correction * jnp.ones(6))

    def __call__(self, tles, tsinces):
        """
        This method computes the forward pass of the ML-dSGP4 model.
        It can take either a single or a list of `dsgp4.tle.TLE` objects,
        and a jax array of times since the TLE epoch in minutes.
        It then returns the propagated state in the TEME coordinate system. The output
        is normalized, to unnormalize and obtain km and km/s, you can use self.normalization_R constant for the position
        and self.normalization_V constant for the velocity.

        Parameters:
        ----------------
        tles (``dsgp4.tle.TLE`` or ``list``): a TLE object or a list of TLE objects.
        tsinces (``jnp.array``): a jax array of times since the TLE epoch in minutes.

        Returns:
        ----------------
        (``jnp.array``): an array of len(tsince)x6 representing the corrected satellite position and velocity in normalized units (to unnormalize to km and km/s, use `self.normalization_R` for position, and `self.normalization_V` for velocity).
        """
        is_batch = hasattr(tles, '__len__')
        if is_batch:
            # Batch case: initialize the batch
            _, tles = initialize_tle(tles, with_grad=False)  # JAX doesn't need with_grad
            x0 = jnp.stack((tles._ecco, tles._argpo, tles._inclo, tles._mo, tles._no_kozai, tles._nodeo), axis=1)
        else:
            # Single TLE case
            initialize_tle(tles, with_grad=False)
            x0 = jnp.stack((tles._ecco, tles._argpo, tles._inclo, tles._mo, tles._no_kozai, tles._nodeo), axis=0).reshape(-1, 6)

        # Input correction network
        x = jax.nn.leaky_relu(linear(self.fc1_weight, self.fc1_bias, x0), negative_slope=0.01)
        x = jax.nn.leaky_relu(linear(self.fc2_weight, self.fc2_bias, x), negative_slope=0.01)
        x = x0 * (1 + self.input_correction * jnp.tanh(linear(self.fc3_weight, self.fc3_bias, x)))

        # Update TLE parameters
        if is_batch:
            tles._ecco = x[:, 0]
            tles._argpo = x[:, 1]
            tles._inclo = x[:, 2]
            tles._mo = x[:, 3]
            tles._no_kozai = x[:, 4]
            tles._nodeo = x[:, 5]
        else:
            # For single TLE, keep as scalars
            tles._ecco = x[0, 0]
            tles._argpo = x[0, 1]
            tles._inclo = x[0, 2]
            tles._mo = x[0, 3]
            tles._no_kozai = x[0, 4]
            tles._nodeo = x[0, 5]

        # Propagate with SGP4
        if is_batch:
            states_teme = propagate_batch(tles, tsinces)
        else:
            states_teme = propagate(tles, tsinces)
        states_teme = states_teme.reshape(-1, 6)

        # Normalize output
        x_out = jnp.concatenate((states_teme[:, :3] / self.normalization_R,
                                  states_teme[:, 3:] / self.normalization_V), axis=1)

        # Output correction network
        x = jax.nn.leaky_relu(linear(self.fc4_weight, self.fc4_bias, x_out), negative_slope=0.01)
        x = jax.nn.leaky_relu(linear(self.fc5_weight, self.fc5_bias, x), negative_slope=0.01)
        x = x_out * (1 + self.output_correction * jnp.tanh(linear(self.fc6_weight, self.fc6_bias, x)))

        return x

    def load_model(self, path, device='cpu'):
        """
        This method loads a model from a PyTorch checkpoint file.

        Parameters:
        ----------------
        path (``str``): path to the file where the PyTorch model is stored.
        device (``str``): device parameter (kept for compatibility, not used in JAX).
        """
        import torch

        # Load PyTorch state dict
        state_dict = torch.load(path, map_location=torch.device(device))

        # Convert PyTorch tensors to JAX arrays
        self.fc1_weight = jnp.array(state_dict['fc1.weight'].detach().numpy())
        self.fc1_bias = jnp.array(state_dict['fc1.bias'].detach().numpy())
        self.fc2_weight = jnp.array(state_dict['fc2.weight'].detach().numpy())
        self.fc2_bias = jnp.array(state_dict['fc2.bias'].detach().numpy())
        self.fc3_weight = jnp.array(state_dict['fc3.weight'].detach().numpy())
        self.fc3_bias = jnp.array(state_dict['fc3.bias'].detach().numpy())

        self.fc4_weight = jnp.array(state_dict['fc4.weight'].detach().numpy())
        self.fc4_bias = jnp.array(state_dict['fc4.bias'].detach().numpy())
        self.fc5_weight = jnp.array(state_dict['fc5.weight'].detach().numpy())
        self.fc5_bias = jnp.array(state_dict['fc5.bias'].detach().numpy())
        self.fc6_weight = jnp.array(state_dict['fc6.weight'].detach().numpy())
        self.fc6_bias = jnp.array(state_dict['fc6.bias'].detach().numpy())

        self.input_correction = jnp.array(state_dict['input_correction'].detach().numpy())
        self.output_correction = jnp.array(state_dict['output_correction'].detach().numpy())
