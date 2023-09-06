import numpy as np
import pyopencl as cl
from meshmode.array_context import PytatoPyOpenCLArrayContext
import meshmode.mesh.generation as mgen
from grudge import op, DiscretizationCollection
from pytools.obj_array import make_obj_array


class MyArrayContext(PytatoPyOpenCLArrayContext):
    pass


def main():
    order = 4

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    actx = MyArrayContext(queue)

    dim = 3
    n = 5

    from meshmode.mesh import TensorProductElementGroup
    from meshmode.discretization.poly_element import \
            LegendreGaussLobattoTensorProductGroupFactory as LGL

    mesh = mgen.generate_regular_rect_mesh(
            a=(-1,)*dim, b=(1,)*dim,
            nelements_per_axis=(n,)*dim,
            group_cls=TensorProductElementGroup)

    import grudge.dof_desc as dd
    dcoll = DiscretizationCollection(
            actx,
            mesh,
            discr_tag_to_group_factory={
                dd.DISCR_TAG_BASE: LGL(order)})

    def f(x):
        result = dcoll.zeros(actx) + 1
        for i in range(dim-1):
            result = result * actx.np.sin(np.pi*x[i])
        result = result * actx.np.cos(np.pi/2*x[dim-1])
        return result


    x = actx.thaw(dcoll.nodes())

    u = f(x)

    op.local_grad(dcoll, u)


if __name__ == "__main__":
    main()

