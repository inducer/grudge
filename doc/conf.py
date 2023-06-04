from urllib.request import urlopen

_conf_url = \
        "https://raw.githubusercontent.com/inducer/sphinxconfig/main/sphinxconfig.py"
with urlopen(_conf_url) as _inf:
    exec(compile(_inf.read(), _conf_url, "exec"), globals())

extensions = globals()["extensions"] + [
    "matplotlib.sphinxext.plot_directive"]

copyright = "2015-21, grudge contributors"
author = "grudge contributors"


def get_version():
    conf = {}
    src = "../grudge/version.py"
    exec(
            compile(open(src).read(), src, "exec"),
            conf)
    return conf["VERSION_TEXT"]


version = get_version()

# The full version, including alpha/beta/rc tags.
release = version

intersphinx_mapping = {
        "python": ("https://docs.python.org/3/", None),
        "numpy": ("https://numpy.org/doc/stable/", None),
        "pyopencl": ("https://documen.tician.de/pyopencl/", None),
        "modepy": ("https://documen.tician.de/modepy/", None),
        "pymbolic": ("https://documen.tician.de/pymbolic/", None),
        "arraycontext": ("https://documen.tician.de/arraycontext/", None),
        "meshmode": ("https://documen.tician.de/meshmode/", None),
        "loopy": ("https://documen.tician.de/loopy/", None),
        "mpi4py": ("https://mpi4py.readthedocs.io/en/stable", None),
        }

# index-page demo uses pyopencl via plot_directive
import os
# switch to "port:cpu" once we're firmly migrated to pocl 4.0
os.environ["PYOPENCL_TEST"] = "port:0"
