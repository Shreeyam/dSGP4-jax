Installation
============

.. _installation_deps:

.. note::
   This is a JAX port of dSGP4. For the original PyTorch version, see the `official repository <https://github.com/esa/dSGP4>`_.

Requirements
------------

This JAX implementation requires:

- ``jax`` >= 0.4.0
- ``jaxlib`` >= 0.4.0
- ``numpy``
- ``matplotlib``

GPU Support
^^^^^^^^^^^

For GPU support, install JAX with CUDA support:

.. code-block:: console

   $ pip install jax[cuda12]

For TPU support or other hardware accelerators, see the `JAX installation guide <https://jax.readthedocs.io/en/latest/installation.html>`_.

Packages
--------

pip (PyTorch version)
^^^^^^^^^^^^^^^^^^^^^

The original PyTorch version is available on `Pypi <https://pypi.org/project/dsgp4/>`_:

.. code-block:: console

   $ pip install dsgp4

conda (PyTorch version)
^^^^^^^^^^^^^^^^^^^^^^^

The original PyTorch version can be installed via conda:

.. code-block:: console

   $ conda config --add channels conda-forge
   $ conda config --set channel_priority strict
   $ conda install dsgp4

or mamba:

.. code-block:: console

   $ mamba install dsgp4

Installation from source
------------------------

JAX version
^^^^^^^^^^^

Using ``git``:

.. code-block:: console

   $ git clone https://github.com/esa/dSGP4-jax.git
   $ cd dSGP4-jax
   $ pip install -e .

This will install the JAX version with all required dependencies.

PyTorch version (original)
^^^^^^^^^^^^^^^^^^^^^^^^^^

For the original PyTorch implementation:

.. code-block:: console

   $ git clone https://github.com/esa/dSGP4.git
   $ cd dSGP4
   $ pip install -e .

We follow the usual PR-based development workflow, thus dSGP4's ``master``
branch is normally kept in a working state.

Verifying the installation
--------------------------

You can verify that dSGP4 was successfully compiled and
installed by running the tests. To do so, you must first install the
optional dependencies.

.. code-block:: bash

   $ pytest

If this command executes without any error, then
your dSGP4 installation is ready for use.

Getting help
------------

If you run into troubles installing dsgp4, please do not hesitate
to contact us by opening an issue report on `github <https://github.com/esa/dSGP4/issues>`__.
