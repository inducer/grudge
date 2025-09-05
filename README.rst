grudge
======

.. image:: https://gitlab.tiker.net/inducer/grudge/badges/main/pipeline.svg
    :alt: Gitlab Build Status
    :target: https://gitlab.tiker.net/inducer/grudge/commits/main
.. image:: https://github.com/inducer/grudge/actions/workflows/ci.yml/badge.svg
    :alt: Github Build Status
    :target: https://github.com/inducer/grudge/actions/workflows/ci.yml

..
    .. image:: https://badge.fury.io/py/grudge.png
        :alt: Python Package Index Release Page
        :target: https://pypi.org/project/grudge/

grudge helps you discretize discontinuous Galerkin operators, quickly
and accurately.

It relies on

* `numpy <https://pypi.org/project/numpy>`__ for arrays
* `modepy <https://pypi.org/project/modepy>`__ for modes and nodes on simplices
* `meshmode <https://pypi.org/project/meshmode>`__ for modes and nodes on simplices
* `loopy <https://pypi.org/project/loopy>`__ for fast array operations
* `pytest <https://pypi.org/project/pytest>`__ for automated testing

and, indirectly,

* `PyOpenCL <https://pypi.org/project/pyopencl>`__ as computational infrastructure

PyOpenCL is likely the only package you'll have to install
by hand, all the others will be installed automatically.

.. image:: https://badge.fury.io/py/grudge.png
    :target: https://pypi..org/project/grudge

Resources:

* `documentation <https://documen.tician.de/grudge>`__
* `wiki home page <https://wiki.tiker.net/Grudge>`__
* `source code via git <https://gitlab.tiker.net/inducer/grudge>`__
