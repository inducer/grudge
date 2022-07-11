Welcome to grudge's Documentation!
==================================

Here's an example to solve the PDE

.. math::
    \begin{cases}
    u_t + 2\pi u_x = 0, \\
    u(0, t) = -\sin(2\pi t), \\
    u(x, 0) = \sin(x),
    \end{cases}

on the domain :math:`x \in [0, 2\pi]`. We closely follow Chapter 3 of
[Hesthaven_2008]_.


.. literalinclude:: ../examples/hello-grudge.py
   :start-after: BEGINEXAMPLE
   :end-before: ENDEXAMPLE

Plotting numerical solution ``uh`` in results in

.. plot:: ../examples/hello-grudge.py

Contents:

.. toctree::
    :maxdepth: 2

    discretization
    dof_desc
    geometry
    operators
    utils
    models
    references
    misc
    🚀 Github <https://github.com/inducer/grudge>
    💾 Download Releases <https://pypi.org/project/grudge>

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
