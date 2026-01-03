import os
from importlib import metadata
from urllib.request import urlopen


_conf_url = "https://tiker.net/sphinxconfig-v0.py"
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
    "jax": ("https://docs.jax.dev/en/latest/", None),
    "meshmode": ("https://documen.tician.de/meshmode/", None),
    "modepy": ("https://documen.tician.de/modepy/", None),
    "mpi4py": ("https://mpi4py.readthedocs.io/en/stable", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pytools": ("https://documen.tician.de/pytools/", None),
    "pymbolic": ("https://documen.tician.de/pymbolic/", None),
    "pyopencl": ("https://documen.tician.de/pyopencl/", None),
    "python": ("https://docs.python.org/3/", None),
}


# index-page demo uses pyopencl via plot_directive
os.environ["PYOPENCL_TEST"] = "port:cpu"

nitpick_ignore_regex = [
    ["py:data|py:class", r"arraycontext.*ContainerTc"],
]


sphinxconfig_missing_reference_aliases = {
    # numpy
    "DTypeLike": "obj:numpy.typing.DTypeLike",
    "np.floating": "class:numpy.floating",

    # mpi4py
    "Intracomm": "mpi4py.MPI.Intracomm",
    "MPI.Intracomm": "mpi4py.MPI.Intracomm",

    # pytools
    "ObjectArray2D": "obj:pytools.obj_array.ObjectArray2D",

    # pyopencl
    "cl_array.Allocator": "class:pyopencl.array.Allocator",

    # actx
    "ArithArrayContainer": "obj:arraycontext.ArithArrayContainer",
    "ArithArrayContainerT": "obj:arraycontext.ArithArrayContainerT",
    "Array": "obj:arraycontext.Array",
    "ArrayContainer": "obj:arraycontext.ArrayContainer",
    "ArrayOrArithContainer": "obj:arraycontext.ArrayOrArithContainer",
    "ArrayOrContainer": "obj:arraycontext.ArrayOrContainer",
    "ArrayOrContainerOrScalar": "obj:arraycontext.ArrayOrContainerOrScalar",
    "ArrayOrContainerOrScalarT": "obj:arraycontext.ArrayOrContainerOrScalarT",
    "ScalarLike": "obj:arraycontext.ScalarLike",
    "arraycontext.typing.ArithArrayContainerT": "obj:arraycontext.ArithArrayContainerT",

    # meshmode
    "DOFArray": "meshmode.dof_array.DOFArray",
    "Discretization": "class:meshmode.discretization.Discretization",
    "Mesh": "meshmode.mesh.Mesh",

    # grudge
    "DiscretizationTag": "obj:grudge.dof_desc.DiscretizationTag",
    "TracePair": "class:grudge.trace_pair.TracePair",
    "VolumeTag": "obj:grudge.dof_desc.VolumeTag",
}


def setup(app):
    app.connect("missing-reference", process_autodoc_missing_reference)  # noqa: F821
