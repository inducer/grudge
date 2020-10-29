from __future__ import absolute_import
#!/usr/bin/env python
# -*- coding: utf-8 -*-


def main():
    from setuptools import setup, find_packages

    version_dict = {}
    init_filename = "grudge/version.py"
    exec(
        compile(open(init_filename, "r").read(), init_filename, "exec"),
        version_dict)

    setup(name="grudge",
          version=version_dict["VERSION_TEXT"],
          description=(
              "Discretize discontinuous Galerkin operators quickly, "
              "on heterogeneous hardware"),
          long_description=open("README.rst", "rt").read(),
          author="Andreas Kloeckner",
          author_email="inform@tiker.net",
          license="MIT",
          url="https://github.com/inducer/grudge",
          classifiers=[
              'Development Status :: 3 - Alpha',
              'Intended Audience :: Developers',
              'Intended Audience :: Other Audience',
              'Intended Audience :: Science/Research',
              'License :: OSI Approved :: MIT License',
              'Natural Language :: English',
              'Programming Language :: Python',

              'Programming Language :: Python :: 3',
              'Topic :: Scientific/Engineering',
              'Topic :: Scientific/Engineering :: Information Analysis',
              'Topic :: Scientific/Engineering :: Mathematics',
              'Topic :: Scientific/Engineering :: Visualization',
              'Topic :: Software Development :: Libraries',
              'Topic :: Utilities',
              ],

          packages=find_packages(),

          python_requires="~=3.6",
          install_requires=[
              "pytest>=2.3",
              "pytools>=2020.3",
              "modepy>=2013.3",
              "meshmode>=2020.2",
              "pyopencl>=2013.1",
              "pymbolic>=2013.2",
              "loopy>=2020.2",
              "cgen>=2013.1.2",
              "leap>=2019.1",
              "dagrt>=2019.1",
              ])


if __name__ == '__main__':
    main()
