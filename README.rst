grudge
======

.. image:: https://gitlab.tiker.net/inducer/grudge/badges/master/pipeline.svg
    :alt: Gitlab Build Status
    :target: https://gitlab.tiker.net/inducer/grudge/commits/master
.. image:: https://github.com/inducer/grudge/workflows/CI/badge.svg?branch=master&event=push
    :alt: Github Build Status
    :target: https://github.com/inducer/grudge/actions?query=branch%3Amaster+workflow%3ACI+event%3Apush

..
    .. image:: https://badge.fury.io/py/grudge.png
        :alt: Python Package Index Release Page
        :target: https://pypi.org/project/grudge/

grudge helps you discretize discontinuous Galerkin operators, quickly
and accurately.

It relies on

* `numpy <http://pypi.org/project/numpy>`_ for arrays
* `modepy <http://pypi.org/project/modepy>`_ for modes and nodes on simplices
* `meshmode <http://pypi.org/project/meshmode>`_ for modes and nodes on simplices
* `loopy <http://pypi.org/project/loopy>`_ for fast array operations
* `leap <http://pypi.org/project/leap>`_ for time integration
* `dagrt <http://pypi.org/project/dagrt>`_ as an execution runtime
* `pytest <http://pypi.org/project/pytest>`_ for automated testing

and, indirectly,

* `PyOpenCL <http://pypi.org/project/pyopencl>`_ as computational infrastructure

PyOpenCL is likely the only package you'll have to install
by hand, all the others will be installed automatically.

.. image:: https://badge.fury.io/py/grudge.png
    :target: http://pypi..org/project/grudge

Resources:

* `documentation <http://documen.tician.de/grudge>`_
* `wiki home page <http://wiki.tiker.net/Grudge>`_
* `source code via git <http://gitlab.tiker.net/inducer/grudge>`_
