from typing import Tuple
import torch
from torch import fx

from LieCG import so13
from LieCG.so13.utils import CodeGenMixin



class Extract(CodeGenMixin, torch.nn.Module):
    # pylint: disable=abstract-method

    def __init__(self, irreps_in, irreps_outs, instructions, squeeze_out: bool = False):
        r"""Extract sub sets of irreps
        """
        super().__init__()
        self.irreps_in = so13.Lorentz_Irreps(irreps_in)
        self.irreps_outs = tuple(so13.Lorentz_Irreps(irreps) for irreps in irreps_outs)
        self.instructions = instructions

        assert len(self.irreps_outs) == len(self.instructions)
        for irreps_out, ins in zip(self.irreps_outs, self.instructions):
            assert len(irreps_out) == len(ins)

        # == generate code ==
        graph = fx.Graph()
        x = fx.Proxy(graph.placeholder('x', torch.Tensor))
        torch._assert(x.shape[-1] == self.irreps_in.dim, "invalid input shape")

        out = []
        for irreps in self.irreps_outs:
            out.append(
                x.new_zeros(x.shape[:-1] + (irreps.dim,))
            )

        for i, (irreps_out, ins) in enumerate(zip(self.irreps_outs, self.instructions)):
            if ins == tuple(range(len(self.irreps_in))):
                out[i].copy_(x)
            else:
                for s_out, i_in in zip(irreps_out.slices(), ins):
                    i_start = self.irreps_in[:i_in].dim
                    i_len = self.irreps_in[i_in].dim
                    out[i].narrow(
                        -1, s_out.start, s_out.stop - s_out.start
                    ).copy_(
                        x.narrow(-1, i_start, i_len)
                    )

        out = tuple(e.node for e in out)
        if squeeze_out and len(out) == 1:
            graph.output(out[0], torch.Tensor)
        else:
            graph.output(out, Tuple[(torch.Tensor,)*len(self.irreps_outs)])

        self._codegen_register({"_compiled_forward": fx.GraphModule({}, graph)})

    def forward(self, x: torch.Tensor):
        return self._compiled_forward(x)


class ExtractIr(Extract):
    # pylint: disable=abstract-method

    def __init__(self, irreps_in, ir):
        r"""Extract ``ir`` from irreps
        Parameters
        ----------
        irreps_in : `so13.Lorentz_Irreps`
            representation of the input
        ir : `so13.Lorentz_Irrep`
            representation to extract
        """
        ir = so13.Lorentz_Irrep(ir)
        irreps_in = so13.Lorentz_Irreps(irreps_in)
        self.irreps_out = so13.Lorentz_Irreps([mul_ir for mul_ir in irreps_in if mul_ir.ir == ir])
        instructions = [tuple(i for i, mul_ir in enumerate(irreps_in) if mul_ir.ir == ir)]

        super().__init__(irreps_in, [self.irreps_out], instructions, squeeze_out=True)
