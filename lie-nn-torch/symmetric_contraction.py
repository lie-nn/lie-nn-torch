###########################################################################################
# Implementation of the symmetric contraction algorithm presented in the MACE paper
# (Batatia et al, MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields , Eq.10 and 11)
# Authors: Ilyes Batatia
# This program is distributed under the MIT License
###########################################################################################

from typing import Dict, Optional, Union

import lie_nn as lie
import torch
import torch.fx
from e3nn.util.codegen import CodeGenMixin
from e3nn.util.jit import compile_mode
from LieACE.tools.torch_tools import get_complex_default_dtype
from opt_einsum import contract


@compile_mode("script")
class SymmetricContraction(CodeGenMixin, torch.nn.Module):
    def __init__(
        self,
        irreps_in: lie.rep,
        irreps_out: lie.rep,
        correlation: Union[int, Dict[str, int]],
        irrep_normalization: str = "component",
        path_normalization: str = "element",
        internal_weights: Optional[bool] = None,
        shared_weights: Optional[torch.Tensor] = None,
        element_dependent: Optional[bool] = None,
        num_elements: Optional[int] = None,
    ) -> None:
        super().__init__()

        if irrep_normalization is None:
            irrep_normalization = "component"

        if path_normalization is None:
            path_normalization = "element"

        assert irrep_normalization in ["component", "norm", "none"]
        assert path_normalization in ["element", "path", "none"]

        self.irreps_in = irreps_in
        self.irreps_out = irreps_out

        del irreps_in, irreps_out

        if not isinstance(correlation, tuple):
            corr = correlation
            correlation = {}
            for irrep_out in self.irreps_out:
                correlation[irrep_out] = corr

        assert shared_weights or not internal_weights

        if internal_weights is None:
            internal_weights = True

        if element_dependent is None:
            element_dependent = True

        self.internal_weights = internal_weights
        self.shared_weights = shared_weights

        del internal_weights, shared_weights

        self.contractions = torch.nn.ModuleDict()
        for irrep_out in self.irreps_out.irreps:
            self.contractions[str(irrep_out)] = Contraction(
                irreps_in=self.irreps_in,
                irrep_out=irrep_out,
                correlation=correlation[irrep_out],
                internal_weights=self.internal_weights,
                element_dependent=element_dependent,
                num_elements=num_elements,
                weights=self.shared_weights,
            )

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        outs = []
        for irrep in self.irreps_out:
            outs.append(self.contractions[str(irrep)](x, y))
        return torch.cat(outs, dim=-1)


class Contraction(torch.nn.Module):
    def __init__(
        self,
        irreps_in: lie.rep,
        irrep_out: lie.rep,
        correlation: int,
        internal_weights: bool = True,
        element_dependent: bool = True,
        num_elements: Optional[int] = None,
        weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()

        self.element_dependent = element_dependent
        self.num_features = irreps_in.count((0, 0))
        self.correlation = correlation
        dtype = torch.get_default_dtype()
        for nu in range(1, correlation + 1):
            U = lie.reduced_symmetric_tensor_product_basis(
                irreps_in, nu,
            )
            irreps = U.rep.irreps
            for (mul_ir_out), u in zip(irreps, U.list):
                ir_out = mul_ir_out.rep
                if ir_out not in map(lambda x: x.rep, irrep_out):
                    continue
                self.register_buffer(
                    f"U_matrix_{nu}",
                    torch.moveaxis(
                        torch.tensor(u, dtype=torch.get_default_dtype()), -1, 0
                    ).to(get_complex_default_dtype()),
                )
        if element_dependent:
            # Tensor contraction equations
            self.equation_main = "...ik,ekc,bci,be -> bc..."
            self.equation_weighting = "...k,ekc,be->bc..."
            self.equation_contract = "bc...i,bci->bc..."
            if internal_weights:
                # Create weight for product basis
                self.weights = torch.nn.ParameterDict({})
                for i in range(1, correlation + 1):
                    num_params = self.U_tensors(i).size()[-1]
                    w = torch.nn.Parameter(
                        torch.randn(num_elements, num_params, self.num_features)
                        / num_params
                    )
                    self.weights[str(i)] = w
            else:
                self.register_buffer("weights", weights)

        else:
            # Tensor contraction equations
            self.equation_main = "...ik,kc,bci -> bc..."
            self.equation_weighting = "...k,kc->c..."
            self.equation_contract = "bc...i,bci->bc..."
            if internal_weights:
                # Create weight for product basis
                self.weights = torch.nn.ParameterDict({})
                for i in range(1, correlation + 1):
                    num_params = self.U_tensors(i).size()[-1]
                    w = torch.nn.Parameter(
                        torch.randn(num_params, self.num_features) / num_params
                    )
                    self.weights[str(i)] = w
            else:
                self.register_buffer("weights", weights)

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor]):
        if self.element_dependent:
            out = contract(
                self.equation_main,
                self.U_tensors(self.correlation),
                self.weights[str(self.correlation)].to(get_complex_default_dtype()),
                x,
                y.to(get_complex_default_dtype()),
            )  # TODO: use optimize library and cuTENSOR  # pylint: disable=fixme
            for corr in range(self.correlation - 1, 0, -1):
                c_tensor = contract(
                    self.equation_weighting,
                    self.U_tensors(corr).real,
                    self.weights[str(corr)].real,
                    y,
                )
                c_tensor = c_tensor + out
                out = contract(self.equation_contract, c_tensor, x)

        else:
            out = contract(
                self.equation_main,
                self.U_tensors(self.correlation),
                self.weights[str(self.correlation)].type(torch.complex64),
                x,
            )  # TODO: use optimize library and cuTENSOR  # pylint: disable=fixme
            for corr in range(self.correlation - 1, 0, -1):
                c_tensor = contract(
                    self.equation_weighting,
                    self.U_tensors(corr),
                    self.weights[str(corr)].type(torch.complex64),
                )
                c_tensor = c_tensor + out
                out = contract(self.equation_contract, c_tensor, x)
        resize_shape = torch.prod(torch.tensor(out.shape[1:]))
        return out.view(out.shape[0], resize_shape)

    def U_tensors(self, nu):
        return self._buffers[f"U_matrix_{nu}"]
