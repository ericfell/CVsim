.. cvsim documentation master file, created by
   sphinx-quickstart on Wed Nov 27 14:39:22 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

cvsim.py
===================

:code:`cvsim.py` is a Python package for simulating cyclic voltammograms (CV) on a disk macroelectrode,
via a semi-analytical method.

This package contains modules to simulate CV of different (electro)chemical mechanisms and to fit experimental CV according to these mechanisms.


If you have a feature request or find a bug, please
`file an issue <https://github.com/ericfell/cvsim/issues>`_
or contribute code improvements and
`submit a pull request <https://help.github.com/articles/creating-a-pull-request-from-a-fork/>`_!


Installation
------------

:code:`cvsim.py` can be installed from PyPI with pip:

.. code-block:: bash

   pip install cvsim


Dependencies
~~~~~~~~~~~~

cvsim.py requires:

-   Python (>=3.10)
-   SciPy


Examples and Documentation
---------------------------

See :doc:`./getting-started` for instructions
on getting started with CV simulations.



.. toctree::
   :maxdepth: 2
   :caption: Contents

   getting-started
   mechanisms
   fit_curve
   faq



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`