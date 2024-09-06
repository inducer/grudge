import os
from importlib import metadata
from urllib.request import urlopen


_conf_url = \
        "https://raw.githubusercontent.com/inducer/sphinxconfig/main/sphinxconfig.py"
with urlopen(_conf_url) as _inf:
    exec(compile(_inf.read(), _conf_url, "exec"), globals())

extensions = globals()["extensions"] + [
    "matplotlib.sphinxext.plot_directive"]

copyright = "2015-2024, Grudge contributors"
author = "Grudge contributors"
release = metadata.version("grudge")
version = ".".join(release.split(".")[:2])

intersphinx_mapping = {
        "arraycontext": ("https://documen.tician.de/arraycontext/", None),
        "loopy": ("https://documen.tician.de/loopy/", None),
        "meshmode": ("https://documen.tician.de/meshmode/", None),
        "modepy": ("https://documen.tician.de/modepy/", None),
        "mpi4py": ("https://mpi4py.readthedocs.io/en/stable", None),
        "numpy": ("https://numpy.org/doc/stable/", None),
        "pymbolic": ("https://documen.tician.de/pymbolic/", None),
        "pyopencl": ("https://documen.tician.de/pyopencl/", None),
        "python": ("https://docs.python.org/3/", None),
}

# index-page demo uses pyopencl via plot_directive
os.environ["PYOPENCL_TEST"] = "port:cpu"
