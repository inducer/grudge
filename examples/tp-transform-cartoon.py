import numpy as np
import pyopencl as cl
import pytato as pt
import loopy as lp
from meshmode.array_context import PytatoPyOpenCLArrayContext
import meshmode.mesh.generation as mgen
from grudge import op, DiscretizationCollection
from grudge.array_context import OutputIsTensorProductDOFArrayOrdered


class PytatoTensorProductArrayContext(PytatoPyOpenCLArrayContext):
    def transform_loopy_program(self, t_unit):

        knl = t_unit.default_entrypoint
        if knl.tags_of_type(OutputIsTensorProductDOFArrayOrdered):
            new_args = []
            for arg in knl.args:
                if arg.is_output:
                    arg = arg.copy(dim_tags=(
                        f"N{len(arg.shape)-1},"
                        + ",".join(f"N{i}"
                                   for i in range(len(arg.shape)-1))
                        ))

                new_args.append(arg)

            knl = knl.copy(args=new_args)
            t_unit = t_unit.with_kernel(knl)
        return super().transform_loopy_program(t_unit)


def main():
    order = 4

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    actx = PytatoTensorProductArrayContext(queue)

    dim = 3
    res = 5

    from meshmode.mesh import TensorProductElementGroup
    from meshmode.discretization.poly_element import \
            LegendreGaussLobattoTensorProductGroupFactory as LGL

    mesh = mgen.generate_regular_rect_mesh(
            a=(-1,)*dim, b=(1,)*dim,
            nelements_per_axis=(res,)*dim,
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

    grad_u = op.local_grad(dcoll, u)
    grad_u = actx.np.stack(grad_u)[0]

    prg = pt.generate_loopy(grad_u).program
    code = lp.generate_code_v2(prg).device_code()

    print(code)
    pu.db

if __name__ == "__main__":
    main()

