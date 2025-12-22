# $\partial\textrm{SGP4}$ - JAX

> **Note**: This is a JAX port of the original PyTorch implementation. For the original PyTorch version, see [esa/dSGP4](https://github.com/esa/dSGP4).

Differentiable SGP4 implemented in JAX.
<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/esa/dSGP4">
    <img src="doc/_static/logo_dsgp4.png" alt="Logo" width="280">
  </a>
  <p align="center">
    Differentiable SGP4
    <br />
    <a href="https://esa.github.io/dSGP4"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/esa/dSGP4/issues/new/choose">Report bug</a>
    ·
    <a href="https://github.com/esa/dSGP4/issues/new/choose">Request feature</a>
  </p>
</p>

## Info

![orbits](https://github.com/esa/dSGP4/assets/33602846/2f42992d-0838-4c11-ae4b-68ad76e2bf33)

This repository contains a JAX port of the code discussed in [the original paper](https://doi.org/10.1016/j.actaastro.2024.10.063).

$\partial \textrm{SGP4}$ is a differentiable version of SGP4 implemented using JAX. By making SGP4 differentiable, $\partial \textrm{SGP4}$ facilitates various space-related applications, including spacecraft orbit determination, covariance transformation, state transition matrix computation, and covariance propagation.

The JAX implementation provides several key advantages:
* **High-performance automatic differentiation** using JAX's grad API
* **JIT compilation** for optimal performance on CPUs, GPUs, and TPUs
* **Parallel orbital propagation** across batches of Two-Line Element Sets (TLEs)
* **Vectorized operations** leveraging modern hardware accelerators
* **Functional programming paradigm** for cleaner, more composable code

Furthermore, $\partial \textrm{SGP4}$'s differentiability enables integration with modern machine learning techniques.
Thus, we propose a novel orbital propagation paradigm, $\textrm{ML}-\partial \textrm{SGP4}$, where neural networks are integrated into the orbital propagator.
Through gradient-based optimization, this combined model's inputs, outputs, and parameters can be iteratively refined, surpassing SGP4's precision while maintaining computational speed. This empowers satellite operators and researchers to train the model using high-precision simulated or observed data, advancing orbital prediction capabilities compared to the standard SGP4.

## Goals

* Differentiable version of SGP4 (implemented in JAX)
* High-performance automatic differentiation using JAX's grad API
* JIT compilation for optimal performance on CPUs, GPUs, and TPUs
* Hybrid SGP4 and machine learning propagation: input/output/parameters corrections of SGP4 from accurate simulated or observed data are learned
* Parallel TLE propagation with vectorized operations
* Use of differentiable SGP4 on several spaceflight mechanics problems (state transition matrix computation, covariance transformation, and propagation, orbit determination, ML hybrid orbit propagation, etc.)

## How to cite

If you use `dsgp4`, we would be grateful if you could star the repository and/or cite our work.
$\partial \textrm{SGP4}$ and its applications for ML hybrid propagation and more, can be found in our [publication](https://doi.org/10.1016/j.actaastro.2024.10.063):

```bibtex
@article{acciarini2024closing,
title = {Closing the gap between SGP4 and high-precision propagation via differentiable programming},
journal = {Acta Astronautica},
volume = {226},
pages = {694-701},
year = {2025},
issn = {0094-5765},
doi = {https://doi.org/10.1016/j.actaastro.2024.10.063},
url = {https://www.sciencedirect.com/science/article/pii/S0094576524006374},
author = {Giacomo Acciarini and Atılım Güneş Baydin and Dario Izzo},
keywords = {SGP4, Orbital propagation, Differentiable programming, Machine learning, Spacecraft collision avoidance, Kessler, Kessler syndrome, AI for space, Applied machine learning for space},
}
```

## Installation

### Local installation from source:

```bash
git clone https://github.com/esa/dSGP4-jax.git
cd dSGP4-jax
pip install -e .
```

**Note:** The package name is `dsgp4-jax` but you import it as `dsgp4_jax`:
```python
import dsgp4_jax  # Note the underscore
```

### Requirements

This JAX port requires:
- `jax` >= 0.4.0
- `jaxlib` >= 0.4.0
- `numpy`
- `matplotlib`

For GPU support, install JAX with CUDA support:
```bash
pip install jax[cuda12]
```

See the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html) for more details on GPU/TPU setup.

### Original PyTorch version

For the original PyTorch implementation, see:
- PyPI: [dsgp4](https://pypi.org/project/dsgp4/)
- Conda: `conda install conda-forge::dsgp4`
- GitHub: [esa/dSGP4](https://github.com/esa/dSGP4)

## Documentation and examples

To get started, follow the examples in the [documentation](https://esa.github.io/dSGP4/). You will find tutorials with basic and more advanced functionalities and applications. 

## Authors:
* [Giacomo Acciarini](https://www.esa.int/gsp/ACT/team/giacomo_acciarini/)
* [Atılım Güneş Baydin](http://gbaydin.github.io/)
* [Dario Izzo](https://www.esa.int/gsp/ACT/team/dario_izzo/)

The project originated after the work of the authors at the [University of Oxford AI4Science Lab](https://oxai4science.github.io/).

## Acknowledgements:

We would like to thank Dr. T.S. Kelso for his support and useful pointers on how to correctly validate the code with respect to the [official release](https://www.space-track.org/documentation#/sgp4) by Space-Track.

## License:

$\partial\textrm{SGP4}$ is distributed under the GNU General Public License version 3. Get in touch with the authors for other licensing options.

## Contact:
* `giacomo.acciarini@gmail.com`
