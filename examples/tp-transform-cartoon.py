import loopy as lp

import meshmode.mesh.generation as mgen

import numpy as np
import pyopencl as cl
import pytato as pt

from grudge import op
from grudge.array_context import OutputIsTensorProductDOFArrayOrdered
from grudge.discretization import make_discretization_collection

from meshmode.array_context import PytatoPyOpenCLArrayContext


class PytatoTensorProductArrayContext(PytatoPyOpenCLArrayContext):
    def transform_dag(self, dag):
        if "dag_dots" not in dir(self):
            self.dag_dots = []

        self.dag_dots.append(pt.get_dot_graph(dag))

        return super().transform_dag(dag)

    def transform_loopy_program(self, t_unit):
        knl = t_unit.default_entrypoint

        # {{{ adjust strides according to tensor product structure
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
        # }}}

        # {{{ prefetch
        # }}}

        # {{{ tile
        # }}}

        # FIXME: remove this (eventually)
        knl = lp.set_options(knl, insert_gbarriers=True)
        t_unit = t_unit.with_kernel(knl)
        self.dev_code = lp.generate_code_v2(t_unit).device_code()

        return super().transform_loopy_program(t_unit)


def main():
    order = 1

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    actx = PytatoTensorProductArrayContext(queue)

    dim = 2
    res = 2

    from meshmode.mesh import TensorProductElementGroup
    from meshmode.discretization.poly_element import \
            LegendreGaussLobattoTensorProductGroupFactory as LGL

    mesh = mgen.generate_regular_rect_mesh(
            a=(-1,)*dim, b=(1,)*dim,
            nelements_per_axis=(res,)*dim,
            group_cls=TensorProductElementGroup)

    import grudge.dof_desc as dd
    dcoll = make_discretization_collection(
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
    pt.show_dot_graph(grad_u)

if __name__ == "__main__":
    main()

