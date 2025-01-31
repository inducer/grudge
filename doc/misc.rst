.. _installation:

Installation
============

Installing :mod:`grudge`
------------------------

This set of instructions is intended for 64-bit Linux computers.
MacOS support is in the works.

#.  Make sure your system has the basics to build software.

    On Debian derivatives (Ubuntu and many more),
    installing ``build-essential`` should do the trick.

    Everywhere else, just making sure you have the ``g++`` package should be
    enough.

#.  Install `miniforge <https://github.com/conda-forge/miniforge>`_::

        curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh
        # then run
        bash ./Miniforge3-*.sh

#.  ``export CONDA=/WHERE/YOU/INSTALLED/miniforge3``

    If you accepted the default location, this should work:

    ``export CONDA=$HOME/miniforge3``

#.  ``$CONDA/bin/conda create -n dgfem``

#.  ``source $CONDA/bin/activate dgfem``

#.  ``conda install git pip pocl islpy pyopencl``

#.  Type the following command::

        hash -r; for i in pymbolic cgen genpy modepy pyvisfile loopy arraycontext meshmode dagrt leap grudge; do python -m pip install --editable "git+https://github.com/inducer/$i.git#egg=$i"; done

.. note::

    In each case, you may leave out the ``--editable`` flag if you would not like
    an editable copy of the source code checked out in a subfolder.

Next time you want to use `grudge`, just run the following command::

    source /WHERE/YOU/INSTALLED/miniforge3/bin/activate dgfem

You may also like to add this to a startup file (like :file:`$HOME/.bashrc`) or create an alias for it.

After this, you should be able to run the `tests <https://gitlab.tiker.net/inducer/grudge/tree/master/test>`_
or `examples <https://gitlab.tiker.net/inducer/grudge/tree/master/examples>`_.

Troubleshooting the Installation
--------------------------------

/usr/bin/ld: cannot find -lstdc++
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Try::

    sudo apt-get install libstdc++-6-dev

to install the missing C++ development package.

No CL platforms found/unknown error -1001
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you get::

    pyopencl.cffi_cl.LogicError: clGetPlatformIDs failed: <unknown error -1001>

try::

    conda update ocl-icd pocl

(This indicates that the OpenCL driver loader didn't find any drivers, or the
drivers were themselves missing dependencies.)

Assertion 'error == 0'
~~~~~~~~~~~~~~~~~~~~~~~

If you get::

    /opt/conda/conda-bld/home_1484016200338/work/pocl-0.13/lib/CL/devices/common.c:108:
    llvm_codegen: Assertion 'error == 0 ' failed. Aborted (core dumped)

then you're likely out of memory.

User-visible Changes
====================

Version 2016.1
--------------

.. note::

    This version is currently under development. You can get snapshots from
    grudge's `git repository <https://github.com/inducer/grudge>`_

Licensing
=========

:mod:`grudge` is licensed to you under the MIT/X Consortium license:

Copyright (c) 2014-21 Andreas Klöckner and Contributors.

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

Acknowledgments
===============

Work on grudge was supported in part by

* the Department of Energy, National Nuclear Security Administration,
  under Award Number DE-NA0003963,
* the US Navy ONR, under grant number N00014-14-1-0117, and
* the US National Science Foundation under grant numbers CCF-1524433,
  and OAC-1931577.

AK also gratefully acknowledges a hardware gift from Nvidia Corporation.

The views and opinions expressed herein do not necessarily reflect those of the
funding agencies.
