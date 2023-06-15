import torch
import lie_nn as lie
from ._tensor_product import TensorProduct, tp_out_irreps_with_instructions


def tensor_product_test():
    irrep1 = [lie.MulIrrep(3, lie.irreps.SU3((2, 1, 0))), lie.MulIrrep(3, lie.irreps.SU3((1, 1, 0)))]
    irrep2 = [lie.MulIrrep(1, lie.irreps.SU3((1, 1, 0)))]
    irrep3 = [lie.MulIrrep(1, lie.irreps.SU3((1, 0, 0)))]

    irreps1 = lie.ReducedRep.from_irreps(irrep1)
    irreps2 = lie.ReducedRep.from_irreps(irrep2)
    irreps3 = lie.ReducedRep.from_irreps(irrep3)

    irrep_out, instructions = tp_out_irreps_with_instructions(irreps1.irreps, irreps2.irreps, irreps3.irreps)
    tp = TensorProduct(irreps1.irreps, irreps2.irreps, irrep_out.irreps, instructions)

    x1 = torch.randn(1, irreps1.dim)
    x2 = torch.randn(1, irreps2.dim)

    out = tp(x1, x2)

    assert out.shape == (1, irrep_out.dim)