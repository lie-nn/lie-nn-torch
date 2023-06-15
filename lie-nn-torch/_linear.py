from ast import Tuple
from typing import List, NamedTuple, Optional, Union

import lie_nn as lie
import torch
from opt_einsum_fx import optimize_einsums_full
from torch import fx

from .utils import CodeGenMixin, _sum_tensors, prod


class Instruction(NamedTuple):
    i_in: int
    i_out: int
    path_shape: tuple
    path_weight: float


class Linear(CodeGenMixin, torch.nn.Module):

    def __init__(
        self,
        irreps_in: lie.Rep,
        irreps_out: lie.Rep,
        *,
        internal_weights: Optional[bool] = None,
        shared_weights: Optional[bool] = None,
        instructions: Optional[List[int]] = None,
        biases: Union[bool, List[bool]] = False,
        _optimize_einsums: Optional[bool] = False,
        use_complex: Optional[bool] = False,
        path_normalization: str = 'element',
    ) -> None:
        super().__init__()

        irreps_in = irreps_in.irreps
        irreps_out = irreps_out.irreps
        
        assert path_normalization in ['element', 'path']

        if use_complex:
            self.type = torch.get_default_dtype()
        else:
            self.type = torch.get_default_dtype()

        if instructions is None:
            # By default, make all possible connections
            instructions = [(i_in, i_out)
                            for i_in, ir_in in enumerate(irreps_in)
                            for i_out, ir_out in enumerate(irreps_out)
                            if ir_in.rep == ir_out.rep]

        instructions = [
            Instruction(
                i_in=i_in,
                i_out=i_out,
                path_shape=(irreps_in[i_in].mul, irreps_out[i_out].mul),
                path_weight=1,
            ) for i_in, i_out in instructions
        ]

        def alpha(ins):
            x = sum(irreps_in[i.i_in if path_normalization ==
                              'element' else ins.i_in].mul
                    for i in instructions if i.i_out == ins.i_out)
            return 1.0 if x == 0 else x

        instructions = [
            Instruction(i_in=ins.i_in,
                        i_out=ins.i_out,
                        path_shape=ins.path_shape,
                        path_weight=alpha(ins)**(-0.5)) for ins in instructions
        ]

        for ins in instructions:
            if not ins.i_in < len(irreps_in):
                raise IndexError(
                    f"{ins.i_in} is not a valid index for irreps_in")
            if not ins.i_out < len(irreps_out):
                raise IndexError(
                    f"{ins.i_out} is not a valid index for irreps_out")
            if not (ins.i_in == -1
                    or irreps_in[ins.i_in].rep == irreps_out[ins.i_out].rep):
                raise ValueError(
                    f"{ins.i_in} and {ins.i_out} do not have the same irrep")

        if biases is None:
            biases = len(irreps_out) * (False, )
        if isinstance(biases, bool):
            biases = [biases and ir.rep.is_scalar() for ir in irreps_out]

        assert len(biases) == len(irreps_out)
        assert all(ir.rep.is_scalar() or (not b)
                   for b, ir in zip(biases, irreps_out))

        instructions += [
            Instruction(i_in=-1,
                        i_out=i_out,
                        path_shape=(mul_ir.mul * mul_ir.rep.dim, ),
                        path_weight=1.0)
            for i_out, (bias, mul_ir) in enumerate(zip(biases, irreps_out))
            if bias
        ]

        # == Process arguments ==
        if shared_weights is False and internal_weights is None:
            internal_weights = False

        if shared_weights is None:
            shared_weights = True

        if internal_weights is None:
            internal_weights = True

        assert shared_weights or not internal_weights
        self.internal_weights = internal_weights
        self.shared_weights = shared_weights

        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.instructions = instructions

        self._optimize_einsums = _optimize_einsums

        # == Generate code ==
        graphmod, self.weight_numel, self.bias_numel = _codegen_linear(
            self.irreps_in,
            self.irreps_out,
            self.instructions,
            shared_weights=shared_weights,
            optimize_einsums=self._optimize_einsums,
            use_complex=use_complex,
        )
        self._codegen_register({"_compiled_main": graphmod})

        # == Generate weights ==
        if internal_weights and self.weight_numel > 0:
            assert self.shared_weights, "Having internal weights impose shared weights"
            self.weight = torch.nn.Parameter(
                torch.randn(*(()), self.weight_numel).to(self.type))

        else:
            # For TorchScript, there always has to be some kind of defined .weight
            self.register_buffer('weight', torch.Tensor())

        # == Generate biases ==
        if internal_weights and self.bias_numel > 0:
            assert self.shared_weights, "Having internal weights impose shared weights"
            self.bias = torch.nn.Parameter(torch.zeros(*((
            )), self.bias_numel)).to(
                self.type
            )  # see appendix C.1 and Eq.5 of https://arxiv.org/pdf/2011.14522.pdf

        else:
            self.register_buffer('bias', torch.Tensor())

        # == Compute output mask ==
        irreps_out_dim = sum(mul_ir.mul * mul_ir.rep.dim for mul_ir in self.irreps_out)
        if irreps_out_dim > 0:
            output_mask = torch.cat([
                torch.ones(mul_ir.mul * mul_ir.rep.dim) if any(
                    (ins.i_out == i_out) and (0 not in ins.path_shape)
                    for ins in self.instructions) else torch.zeros(mul_ir.mul * mul_ir.rep.dim)
                for i_out, mul_ir in enumerate(self.irreps_out)
            ])
        else:
            output_mask = torch.ones(0)
        self.register_buffer('output_mask', output_mask)

    def __repr__(self):
        irreps_in = lie.ReducedRep.from_irreps(self.irreps_in)
        irreps_out = lie.ReducedRep.from_irreps(self.irreps_out)
        return f"{self.__class__.__name__}({irreps_in} -> {irreps_out} | {self.weight_numel} weights)"

    def forward(self,
                features,
                weight: Optional[torch.Tensor] = None,
                bias: Optional[torch.Tensor] = None):
        """evaluate
        Parameters
        ----------
        features : `torch.Tensor`
            tensor of shape ``(..., irreps_in.dim)``
        weight : `torch.Tensor`, optional
            required if ``internal_weights`` is `False`
        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(..., irreps_out.dim)``
        """
        if weight is None:
            if self.weight_numel > 0 and not self.internal_weights:
                raise RuntimeError(
                    "Weights must be provided when internal_weights = False")
            weight = self.weight
        if bias is None:
            if self.bias_numel > 0 and not self.internal_weights:
                raise RuntimeError(
                    "Biases must be provided when internal_weights = False")
            bias = self.bias
        return self._compiled_main(features, weight, bias)


def _codegen_linear(
    irreps_in: lie.Rep,
    irreps_out: lie.Rep,
    instructions: List[Instruction],
    shared_weights: bool = False,
    optimize_einsums: bool = True,
    use_complex: bool = True,
):
    graph_out = fx.Graph()
    tracer_out = fx.proxy.GraphAppendingTracer(graph_out)

    # = Function definitions =
    x = fx.Proxy(graph_out.placeholder('x', torch.Tensor), tracer_out)
    ws = fx.Proxy(graph_out.placeholder('w', torch.Tensor), tracer_out)
    bs = fx.Proxy(graph_out.placeholder('b', torch.Tensor), tracer_out)

    size = x.shape[:-1]
    irreps_out_dim = sum(ir.mul * ir.rep.dim for ir in irreps_out)
    outsize = size + (irreps_out_dim, )

    bias_numel = sum(irreps_out[i.i_out].dim for i in instructions
                     if i.i_in == -1)
    if bias_numel > 0:
        bs = bs.reshape(-1, bias_numel)

    # = Short-circut for nothing to do =
    # We produce no code for empty instructions
    instructions = [ins for ins in instructions if 0 not in ins.path_shape]

    if len(instructions) == 0 and bias_numel == 0:
        out = x.new_zeros(outsize)

        graph_out.output(out.node, torch.Tensor)
        # Short circut
        # 0 is weight_numel
        return fx.GraphModule({}, graph_out, "linear_forward"), 0, 0

    irreps_in_dim = sum(ir.mul * ir.rep.dim for ir in irreps_in)
    x = x.reshape(-1, irreps_in_dim)
    batch_out = x.shape[0]

    weight_numel = sum(
        prod(ins.path_shape) for ins in instructions if ins.i_in != -1)
    if weight_numel > 0:
        ws = ws.reshape(-1, weight_numel)

    # = extract individual input irreps =
    if len(irreps_in) == 1:
        x_list = [
            x.reshape(batch_out, *(()), irreps_in[0].mul, irreps_in[0].ir.dim)
        ]
    else:
        x_list = []
        d = 0
        for mul_ir in irreps_in:
            x_list.append(
                x.narrow(-1, d,
                        mul_ir.mul * mul_ir.rep.dim).reshape(batch_out, *(()), mul_ir.mul,
                                            mul_ir.rep.dim)
            )
            d += mul_ir.mul * mul_ir.rep.dim

    z = '' if shared_weights else 'z'

    flat_weight_index = 0
    flat_bias_index = 0

    out_list = []
    print(instructions)

    for ins in instructions:
        mul_ir_out = irreps_out[ins.i_out]

        if ins.i_in == -1:
            # = bias =
            b = bs.narrow(-1, flat_bias_index, prod(ins.path_shape))
            flat_bias_index += prod(ins.path_shape)
            out_list += [(ins.path_weight * b).reshape(1, *(()),
                                                       mul_ir_out.dim)]
        else:
            mul_ir_in = irreps_in[ins.i_in]

            # Short-circut for empty irreps
            if mul_ir_in.dim == 0 or mul_ir_out.dim == 0:
                continue

            # Extract the weight from the flattened weight tensor
            path_nweight = prod(ins.path_shape)
            if len(instructions) == 1:
                # Avoid unnecessary view when there is only one weight
                w = ws
            else:
                w = ws.narrow(-1, flat_weight_index, path_nweight)
            w = w.reshape((() if shared_weights else (-1, )) + (()) +
                          ins.path_shape)
            flat_weight_index += path_nweight

            ein_out = torch.einsum(f"{z}uw,zui->zwi", w, x_list[ins.i_in])

            ein_out = ins.path_weight * ein_out

            out_list += [ein_out.reshape(batch_out, *(()), mul_ir_out.dim)]

    # = Return the result =
    out = [
        _sum_tensors([
            out
            for ins, out in zip(instructions, out_list) if ins.i_out == i_out
        ],
                     shape=(batch_out, *(()), mul_ir_out.dim),
                     like=x) for i_out, mul_ir_out in enumerate(irreps_out)
        if mul_ir_out.mul > 0
    ]
    if len(out) > 1:
        out = torch.cat(out, dim=-1)
    else:
        out = out[0]

    out = out.reshape(outsize)

    graph_out.output(out.node, torch.Tensor)

    # check graphs
    graph_out.lint()

    graphmod_out = fx.GraphModule({}, graph_out, "linear_forward")

    # TODO: when eliminate_dead_code() is in PyTorch stable, use that
    if optimize_einsums:
        # See _tensor_product/_codegen.py for notes
        batchdim = 4
        if use_complex:
            type = torch.get_default_dtype()
        else:
            type = torch.get_default_dtype()
        irreps_in_dim = sum(ir.mul * ir.rep.dim for ir in irreps_in)
        example_inputs = (
            torch.zeros((batchdim, *(()), irreps_in_dim), dtype=type),
            torch.zeros(
                (1 if shared_weights else batchdim, 1, 1, weight_numel),
                dtype=type,
            ),
            torch.zeros((1 if shared_weights else batchdim, 1, bias_numel),
                        dtype=type),
        )
        graphmod_out = optimize_einsums_full(graphmod_out, example_inputs)

    return graphmod_out, weight_numel, bias_numel
