import torch
import lie_nn as lie
from ._linear import Linear


def tensor_product_test():
    irrep1 = [lie.MulIrrep(3, lie.irreps.SU3((2, 1, 0))), lie.MulIrrep(3, lie.irreps.SU3((1, 1, 0)))]
    irrep2 = [lie.MulIrrep(5, lie.irreps.SU3((1, 1, 0)))]

    irreps1 = lie.ReducedRep.from_irreps(irrep1)
    irreps2 = lie.ReducedRep.from_irreps(irrep2)

    linear = Linear(irreps1, irreps2)

    x1 = torch.randn(1, irreps1.dim)

    out = linear(x1)

    assert out.shape == (1, irreps2.dim)
