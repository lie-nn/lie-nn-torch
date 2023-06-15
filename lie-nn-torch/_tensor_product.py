"""
Adapted from e3nn tensor product by Mario Geiger
"""

from collections import OrderedDict
from math import sqrt
from typing import List, NamedTuple, Optional, Union, Dict, Tuple
import lie_nn as lie

import torch
import torch.fx
from opt_einsum_fx import optimize_einsums_full

from torch import fx
from .utils import CodeGenMixin, _sum_tensors, prod


class Instruction(NamedTuple):
    i_in1: int
    i_in2: int
    i_out: int
    connection_mode: str
    has_weight: bool
    path_weight: float
    path_shape: tuple

def tp_out_irreps_with_instructions(
    irreps1: lie.Rep, irreps2: lie.Rep, target_reps: lie.Rep
) -> Tuple[lie.Rep, List]:
    trainable = True

    # Collect possible irreps and their instructions
    irreps_out_list: List[Tuple[int, lie.Rep]] = []
    instructions = []
    target_irreps = [mulrep.rep for mulrep in target_reps]
    for i, irrep1 in enumerate(irreps1):
        mul, ir_in = irrep1.mul, irrep1.rep
        for j, irrep2 in enumerate(irreps2):
            ir_edge = irrep2.rep
            for ir_out in ir_in * ir_edge:  # | l1 - l2 | <= l <= l1 + l2
                if ir_out in target_irreps:
                    k = len(irreps_out_list)  # instruction index
                    irreps_out_list.append((mul, ir_out))
                    instructions.append((i, j, k, "uvu", trainable))

    # We sort the output irreps of the tensor product so that we can simplify them
    # when they are provided to the second o3.Linear
    irreps_out = [lie.MulIrrep(mul, irreps_out) for mul, irreps_out in irreps_out_list]
    irreps_out = lie.ReducedRep.from_irreps(irreps_out)
    # Permute the output indexes of the instructions to match the sorted irreps:
    instructions = [
        (i_in1, i_in2, i_out, mode, train)
        for i_in1, i_in2, i_out, mode, train in instructions
    ]

    return irreps_out, instructions

class TensorProduct(CodeGenMixin, torch.nn.Module):

    def __init__(
        self,
        irreps_in1: lie.Rep,
        irreps_in2: lie.Rep,
        irreps_out: lie.Rep,
        instructions: List[Tuple],
        in1_var: Optional[Union[List[float], torch.Tensor]] = None,
        in2_var: Optional[Union[List[float], torch.Tensor]] = None,
        out_var: Optional[Union[List[float], torch.Tensor]] = None,
        irrep_normalization: str = 'component',
        path_normalization: str = 'element',
        internal_weights: Optional[bool] = None,
        shared_weights: Optional[bool] = None,
        compile_left_right: Optional[bool] = True,
        _specialized_code: Optional[bool] = True,
        _optimize_einsums: Optional[bool] = True,
        use_complex: Optional[bool] = False,
    ) -> None:
        super().__init__()
        if use_complex:
            self.type = torch.get_default_dtype()
        else:
            self.type = torch.get_default_dtype()

        if irrep_normalization is None:
            irrep_normalization = 'component'

        if path_normalization is None:
            path_normalization = 'element'

        assert irrep_normalization in ['component', 'norm', 'none']
        assert path_normalization in ['element', 'path', 'none']

        self.irreps_in1 = irreps_in1
        self.irreps_in2 = irreps_in2
        self.irreps_out = irreps_out

        del irreps_in1, irreps_in2, irreps_out

        instructions = [
            x if len(x) == 6 else x + (1.0, ) for x in instructions
        ]
        instructions = [
            Instruction(
                i_in1=i_in1,
                i_in2=i_in2,
                i_out=i_out,
                connection_mode=connection_mode,
                has_weight=has_weight,
                path_weight=path_weight,
                path_shape={
                    'uvw':
                    (self.irreps_in1[i_in1].mul, self.irreps_in2[i_in2].mul,
                     self.irreps_out[i_out].mul),
                    'uvu':
                    (self.irreps_in1[i_in1].mul, self.irreps_in2[i_in2].mul),
                    'uvv':
                    (self.irreps_in1[i_in1].mul, self.irreps_in2[i_in2].mul),
                    'uuw':
                    (self.irreps_in1[i_in1].mul, self.irreps_out[i_out].mul),
                    'uuu': (self.irreps_in1[i_in1].mul, ),
                    'uvuv': (self.irreps_in1[i_in1].mul,
                             self.irreps_in2[i_in2].mul),
                    'uvu<v': (self.irreps_in1[i_in1].mul *
                              (self.irreps_in2[i_in2].mul - 1) // 2, ),
                    'u<vw': (self.irreps_in1[i_in1].mul *
                             (self.irreps_in2[i_in2].mul - 1) // 2,
                             self.irreps_out[i_out].mul),
                }[connection_mode],
            ) for i_in1, i_in2, i_out, connection_mode, has_weight, path_weight
            in instructions
        ]

        if in1_var is None:
            in1_var = [1.0 for _ in range(len(self.irreps_in1))]
        else:
            in1_var = [float(var) for var in in1_var]
            assert len(in1_var) == len(
                self.irreps_in1
            ), "Len of ir1_var must be equal to len(irreps_in1)"

        if in2_var is None:
            in2_var = [1.0 for _ in range(len(self.irreps_in2))]
        else:
            in2_var = [float(var) for var in in2_var]
            assert len(in2_var) == len(
                self.irreps_in2
            ), "Len of ir2_var must be equal to len(irreps_in2)"

        if out_var is None:
            out_var = [1.0 for _ in range(len(self.irreps_out))]
        else:
            out_var = [float(var) for var in out_var]
            assert len(out_var) == len(
                self.irreps_out
            ), "Len of out_var must be equal to len(irreps_out)"

        def num_elements(ins):
            return {
                'uvw': (self.irreps_in1[ins.i_in1].mul *
                        self.irreps_in2[ins.i_in2].mul),
                'uvu':
                self.irreps_in2[ins.i_in2].mul,
                'uvv':
                self.irreps_in1[ins.i_in1].mul,
                'uuw':
                self.irreps_in1[ins.i_in1].mul,
                'uuu':
                1,
                'uvuv':
                1,
                'uvu<v':
                1,
                'u<vw':
                self.irreps_in1[ins.i_in1].mul *
                (self.irreps_in2[ins.i_in2].mul - 1) // 2,
            }[ins.connection_mode]

        normalization_coefficients = []
        for ins in instructions:
            mul_ir_in1 = self.irreps_in1[ins.i_in1]
            mul_ir_in2 = self.irreps_in2[ins.i_in2]
            mul_ir_out = self.irreps_out[ins.i_out]
            assert ins.connection_mode in [
                'uvw', 'uvu', 'uvv', 'uuw', 'uuu', 'uvuv', 'uvu<v', 'u<vw'
            ]

            if irrep_normalization == 'component':
                alpha = mul_ir_out.rep.dim
            if irrep_normalization == 'norm':
                alpha = mul_ir_in1.rep.dim * mul_ir_in2.rep.dim
            if irrep_normalization == 'none':
                alpha = 1

            if path_normalization == 'element':
                x = sum(in1_var[i.i_in1] * in2_var[i.i_in2] * num_elements(i)
                        for i in instructions if i.i_out == ins.i_out)
            if path_normalization == 'path':
                x = in1_var[ins.i_in1] * in2_var[ins.i_in2] * num_elements(ins)
                x *= len([i for i in instructions if i.i_out == ins.i_out])
            if path_normalization == 'none':
                x = 1

            if x > 0.0:
                alpha /= x

            alpha *= out_var[ins.i_out]
            alpha *= ins.path_weight

            normalization_coefficients += [sqrt(alpha)]

        self.instructions = [
            Instruction(ins.i_in1, ins.i_in2, ins.i_out, ins.connection_mode,
                        ins.has_weight, alpha, ins.path_shape)
            for ins, alpha in zip(instructions, normalization_coefficients)
        ]

        self._in1_dim = sum(self.irreps_in1[i].mul * self.irreps_in1[i].rep.dim for i in range(len(self.irreps_in1)))
        self._in2_dim = sum(self.irreps_in2[i].mul * self.irreps_in2[i].rep.dim for i in range(len(self.irreps_in2)))

        if shared_weights is False and internal_weights is None:
            internal_weights = False

        if shared_weights is None:
            shared_weights = True

        if internal_weights is None:
            internal_weights = shared_weights and any(
                i.has_weight for i in self.instructions)

        assert shared_weights or not internal_weights
        self.internal_weights = internal_weights
        self.shared_weights = shared_weights

        # Generate the actual tensor product code
        if compile_left_right:
            graphmod_left_right = codegen_tensor_product_left_right(
                self.irreps_in1,
                self.irreps_in2,
                self.irreps_out,
                self.instructions,
                self.shared_weights,
                _specialized_code,
                _optimize_einsums,
                use_complex,
            )
        else:
            graphmod_left_right = fx.Graph()
            graphmod_left_right.placeholder('x1', torch.Tensor)
            graphmod_left_right.placeholder('x2', torch.Tensor)
            graphmod_left_right.placeholder('w', torch.Tensor)
            graphmod_left_right.call_function(
                torch._assert,
                args=
                (False,
                 "`left_right` method is not compiled, set `compile_left_right` to True when creating the TensorProduct"
                 ))
            graphmod_left_right = fx.GraphModule(torch.nn.Module(),
                                                 graphmod_left_right,
                                                 class_name="tp_forward")

        graphmod_right = fx.Graph()
        graphmod_right.placeholder('x2', torch.Tensor)
        graphmod_right.placeholder('w', torch.Tensor)
        graphmod_right.call_function(
            torch._assert,
            args=
            (False,
             "`right` method is not compiled, set `compile_right` to True when creating the TensorProduct"
             ))
        graphmod_right = fx.GraphModule(torch.nn.Module(),
                                        graphmod_right,
                                        class_name="tp_forward")

        self._codegen_register({
            "_compiled_main_left_right": graphmod_left_right,
            "_compiled_main_right": graphmod_right
        })

        # === Determine weights ===
        self.weight_numel = sum(
            prod(ins.path_shape) for ins in self.instructions
            if ins.has_weight)

        if internal_weights and self.weight_numel > 0:
            assert self.shared_weights, "Having internal weights impose shared weights"
            self.weight = torch.nn.Parameter(
                torch.randn(self.weight_numel).to(self.type))
        else:
            # For TorchScript, there always has to be some kind of defined .weight
            self.register_buffer('weight', torch.Tensor().to(self.type))

        irreps_out_dim = sum(
            ir.mul * ir.rep.dim for ir in self.irreps_out) if len(self.irreps_out) > 0 else 0
        if irreps_out_dim > 0:
            output_mask = torch.cat([
                torch.ones(ir.mul * ir.rep.dim) if any(
                    (ins.i_out == i_out) and (ins.path_weight != 0) and (
                        0 not in ins.path_shape)
                    for ins in self.instructions) else torch.zeros(ir.mul *
                                                                   ir.rep.dim)
                for i_out, ir in enumerate(self.irreps_out)
            ])
        else:
            output_mask = torch.ones(0)
        self.register_buffer('output_mask', output_mask)

        # For TorchScript, this needs to be done in advance:
        self._profiling_str = str(self)

        # === Determine weights ===
        self.weight_numel = sum(
            prod(ins.path_shape) for ins in self.instructions
            if ins.has_weight)

    def __repr__(self):
        npath = sum(prod(i.path_shape) for i in self.instructions)
        irreps_in1 = lie.ReducedRep.from_irreps(self.irreps_in1)
        irreps_in2 = lie.ReducedRep.from_irreps(self.irreps_in2)
        irreps_out = lie.ReducedRep.from_irreps(self.irreps_out)
        return (
            f"{self.__class__.__name__}"
            f"({irreps_in1} ⊗ {irreps_in2} "
            f"-> {irreps_out} | {npath} paths | {self.weight_numel} weights)"
        )

    @torch.jit.unused
    def _prep_weights_python(
        self, weight: Optional[Union[torch.Tensor, List[torch.Tensor]]]
    ) -> Optional[torch.Tensor]:
        if isinstance(weight, list):
            weight_shapes = [
                ins.path_shape for ins in self.instructions if ins.has_weight
            ]
            if not self.shared_weights:
                weight = [
                    w.reshape(-1, prod(shape))
                    for w, shape in zip(weight, weight_shapes)
                ]
            else:
                weight = [
                    w.reshape(prod(shape))
                    for w, shape in zip(weight, weight_shapes)
                ]
            return torch.cat(weight, dim=-1)
        else:
            return weight

    def _get_weights(self, weight: Optional[torch.Tensor]) -> torch.Tensor:
        if not torch.jit.is_scripting():
            # If we're not scripting, then we're in Python and `weight` could be a List[Tensor]
            # deal with that:
            weight = self._prep_weights_python(weight)
        if weight is None:
            if self.weight_numel > 0 and not self.internal_weights:
                raise RuntimeError(
                    "Weights must be provided when the TensorProduct does not have `internal_weights`"
                )
            return self.weight
        else:
            if self.shared_weights:
                assert weight.shape == (
                    self.weight_numel, ), "Invalid weight shape"
            else:
                assert weight.shape[
                    -1] == self.weight_numel, "Invalid weight shape"
                assert weight.ndim > 1, "When shared weights is false, weights must have batch dimension"
            return weight

    def forward(self, x, y, weight: Optional[torch.Tensor] = None):
        r"""Evaluate :math:`w x \otimes y`.
        Parameters
        ----------
        x : `torch.Tensor`
            tensor of shape ``(..., irreps_in1.dim)``
        y : `torch.Tensor`
            tensor of shape ``(..., irreps_in2.dim)``
        weight : `torch.Tensor` or list of `torch.Tensor`, optional
            required if ``internal_weights`` is ``False``
            tensor of shape ``(self.weight_numel,)`` if ``shared_weights`` is ``True``
            tensor of shape ``(..., self.weight_numel)`` if ``shared_weights`` is ``False``
            or list of tensors of shapes ``weight_shape`` / ``(...) + weight_shape``.
            Use ``self.instructions`` to know what are the weights used for.
        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(..., irreps_out.dim)``
        """
        assert x.shape[-1] == self._in1_dim, "Incorrect last dimension for x"
        assert y.shape[-1] == self._in2_dim, "Incorrect last dimension for y"

        # - PROFILER - with torch.autograd.profiler.record_function(self._profiling_str):
        real_weight = self._get_weights(weight)
        return self._compiled_main_left_right(x, y, real_weight)


def codegen_tensor_product_left_right(
    irreps_in1: lie.Rep,
    irreps_in2: lie.Rep,
    irreps_out: lie.Rep,
    instructions: List[Instruction],
    shared_weights: bool = False,
    specialized_code: bool = True,
    optimize_einsums: bool = True,
    use_complex: bool = False,
) -> Tuple[fx.GraphModule, fx.GraphModule]:
    graph = fx.Graph()

    # = Function definitions =
    tracer = fx.proxy.GraphAppendingTracer(graph)
    constants = OrderedDict()

    x1s = fx.Proxy(graph.placeholder('x1', torch.Tensor), tracer=tracer)
    x2s = fx.Proxy(graph.placeholder('x2', torch.Tensor), tracer=tracer)
    weights = fx.Proxy(graph.placeholder('w', torch.Tensor), tracer=tracer)

    empty = fx.Proxy(graph.call_function(torch.empty, ((), ),
                                         dict(device='cpu')),
                     tracer=tracer)
    if shared_weights:
        output_shape = torch.broadcast_tensors(empty.expand(x1s.shape[:-1]),
                                               empty.expand(
                                                   x2s.shape[:-1]))[0].shape
    else:
        output_shape = torch.broadcast_tensors(
            empty.expand(x1s.shape[:-1]), empty.expand(x2s.shape[:-1]),
            empty.expand(weights.shape[:-1]))[0].shape
    del empty

    # = Short-circut for zero dimensional =
    # We produce no code for empty instructions
    instructions = [ins for ins in instructions if 0 not in ins.path_shape]

    if len(instructions) == 0:
        outputs = x1s.new_zeros(output_shape + (irreps_out.dim, ))

        graph.output(outputs.node, torch.Tensor)
        # Short circut
        return fx.GraphModule({}, graph, "tp_forward")

    # = Broadcast inputs =
    if shared_weights:
        x1s, x2s = x1s.broadcast_to(output_shape +
                                    (-1, )), x2s.broadcast_to(output_shape +
                                                              (-1, ))
    else:
        x1s, x2s, weights = x1s.broadcast_to(
            output_shape + (-1, )), x2s.broadcast_to(output_shape + (
                -1, )), weights.broadcast_to(output_shape + (-1, ))
    irreps_out_dim = sum(irreps_out[i].mul * irreps_out[i].rep.dim for i in range(len(irreps_out)))
    output_shape = output_shape + (irreps_out_dim, )
    
    irreps_in1_dim = sum(irreps_in1[i].mul * irreps_in1[i].rep.dim for i in range(len(irreps_in1)))
    irreps_in2_dim = sum(irreps_in2[i].mul * irreps_in2[i].rep.dim for i in range(len(irreps_in2)))
    x1s = x1s.reshape(-1, irreps_in1_dim)
    x2s = x2s.reshape(-1, irreps_in2_dim)

    batch_numel = x1s.shape[0]

    # = Determine number of weights and reshape weights ==
    weight_numel = sum(
        prod(ins.path_shape) for ins in instructions if ins.has_weight)
    if weight_numel > 0:
        weights = weights.reshape(-1, weight_numel)
    del weight_numel

    # = book-keeping for cg =
    cg_keys = []
    cg_dict_out = dict()

    # = extract individual input irreps =
    # If only one input irrep, can avoid creating a view
    if len(irreps_in1) == 1:
        x1_list = [
            x1s.reshape(batch_numel, irreps_in1[0].mul, irreps_in1[0].rep.dim)
        ]
    else:
        x1_list = []
        d = 0
        for _, mul_ir in enumerate(irreps_in1):
            i = mul_ir.mul * mul_ir.rep.dim
            x1_list.append(x1s[:, d : d + i].reshape(batch_numel, mul_ir.mul, mul_ir.rep.dim))
            d += i
    x2_list = []
    # If only one input irrep, can avoid creating a view
    if len(irreps_in2) == 1:
        x2_list.append(
            x2s.reshape(batch_numel, irreps_in2[0].mul, irreps_in2[0].rep.dim))
    else:
        d=0
        for _, mul_ir in enumerate(irreps_in2):
            i = mul_ir.mul * mul_ir.rep.dim
            x2_list.append(x2s[:, d : d + i].reshape(batch_numel, mul_ir.mul,
                                             mul_ir.rep.dim))
    # The einsum string index to prepend to the weights if the weights are not shared and have a batch dimension
    z = '' if shared_weights else 'z'

    # Cache of input irrep pairs whose outer products (xx) have already been computed
    xx_dict = dict()

    # Current index in the flat weight tensor
    flat_weight_index = 0

    outputs = []

    for ins in instructions:
        mul_ir_in1 = irreps_in1[ins.i_in1]
        mul_ir_in2 = irreps_in2[ins.i_in2]
        mul_ir_out = irreps_out[ins.i_out]

        if mul_ir_in1.dim == 0 or mul_ir_in2.dim == 0 or mul_ir_out.dim == 0:
            continue

        x1 = x1_list[ins.i_in1]
        x2 = x2_list[ins.i_in2]

        assert ins.connection_mode in [
            'uvw', 'uvu', 'uvv', 'uuw', 'uuu', 'uvuv', 'uvu<v', 'u<vw'
        ]

        if ins.has_weight:
            # Extract the weight from the flattened weight tensor
            w = weights[:, flat_weight_index:flat_weight_index +
                        prod(ins.path_shape)].reshape((
                            () if shared_weights else (-1, )) +
                                                      tuple(ins.path_shape))
            flat_weight_index += prod(ins.path_shape)

        # Construct the general xx in case this instruction isn't specialized
        # If this isn't used, the dead code will get removed
        key = (ins.i_in1, ins.i_in2, ins.connection_mode[:2])
        if key not in xx_dict:
            if ins.connection_mode[:2] == 'uu':
                xx_dict[key] = torch.einsum('zui,zuj->zuij', x1, x2)
            else:
                xx_dict[key] = torch.einsum('zui,zvj->zuvij', x1, x2)
        xx = xx_dict[key]
        del key

        # Create a proxy & request for the relevant wigner cg
        # If not used (because of specialized code), will get removed later.
        key = (ins.i_in1, ins.i_in2, ins.i_out)
        if key not in cg_keys:
            cg_dict_out[key] = fx.Proxy(graph.get_attr(
                f"_cg_{key[0]}_{key[1]}_{key[2]}"
            ),
                                        tracer=tracer)
            cg_keys.append(key)
        cg = cg_dict_out[key]
        #Store the cg in constants.pt to not compute them
        #cg = fx.Proxy(graph.get_attr(cg_name), tracer=tracer)

        if ins.connection_mode == 'uvw':
            assert ins.has_weight
            result = torch.einsum(f"{z}uvw,ijk,zuvij->zwk", w, cg, xx)
        if ins.connection_mode == 'uvu':
            assert mul_ir_in1.mul == mul_ir_out.mul
            result = torch.einsum("ijk,zuvij->zuk", cg, xx)
        if ins.connection_mode == 'uvv':
            assert mul_ir_in2.mul == mul_ir_out.mul
            if ins.has_weight:
                result = torch.einsum(f"{z}uv,ijk,zuvij->zvk", w, cg, xx)
            else:
                result = torch.einsum("ijk,zuvij->zvk", cg, xx)
        if ins.connection_mode == 'uuw':
            assert mul_ir_in1.mul == mul_ir_in2.mul
            if ins.has_weight:
                result = torch.einsum(f"{z}uw,ijk,zuij->zwk", w, cg, xx)
            else:
                # equivalent to tp(x, y, 'uuu').sum('u')
                assert mul_ir_out.mul == 1
                result = torch.einsum("ijk,zuij->zk", cg, xx)
        if ins.connection_mode == 'uuu':
            assert mul_ir_in1.mul == mul_ir_in2.mul == mul_ir_out.mul
            if ins.has_weight:
                result = torch.einsum(f"{z}u,ijk,zuij->zuk", w, cg, xx)
            else:
                result = torch.einsum("ijk,zuij->zuk", cg, xx)
        if ins.connection_mode == 'uvuv':
            assert mul_ir_in1.mul * mul_ir_in2.mul == mul_ir_out.mul
            if ins.has_weight:
                # TODO implement specialized code
                result = torch.einsum(f"{z}uv,ijk,zuvij->zuvk", w, cg, xx)
            else:
                # TODO implement specialized code
                result = torch.einsum("ijk,zuvij->zuvk", cg, xx)

        result = ins.path_weight * result

        outputs += [result.reshape(batch_numel, mul_ir_out.dim)]

        # Remove unused w3js:
        if len(cg.node.users) == 0:
            del cg_keys[-1]
            # The w3j nodes are reshapes, so we have to remove them from the graph
            # Although they are dead code, they try to reshape to dimensions that don't exist
            # (since the corresponding w3js are not in w3j)
            # so they screw up the shape propagation, even though they would be removed later as dead code by TorchScript.
            graph.erase_node(cg_dict_out.pop(key).node)

    # = Return the result =
    outputs = [
        _sum_tensors([
            out
            for ins, out in zip(instructions, outputs) if ins.i_out == i_out
        ],
                     shape=(batch_numel, mul_ir_out.dim),
                     like=x1s) for i_out, mul_ir_out in enumerate(irreps_out)
        if mul_ir_out.mul > 0
    ]
    if len(outputs) > 1:
        outputs = torch.cat(outputs, dim=1)
    else:
        # Avoid an unnecessary copy in a size one torch.cat
        outputs = outputs[0]

    outputs = outputs.reshape(output_shape)

    graph.output(outputs.node, torch.Tensor)

    # check graphs
    graph.lint()

    if use_complex:
        type = torch.get_default_dtype()
    else:
        type = torch.get_default_dtype()
    # Make GraphModules
    cg_mats = {}
    for i_1, i_2, i_3 in cg_keys:
        cg_mats[f"_cg_{i_1}_{i_2}_{i_3}"] = torch.tensor(lie.clebsch_gordan(
            irreps_in1[i_1].rep, irreps_in2[i_2].rep, irreps_out[i_3].rep
        ), dtype=type).sum(0) # TODO: add weights for null space dimension
    # Make GraphModules
    constants_root = torch.nn.Module()
    for key, value in cg_mats.items():
        constants_root.register_buffer(key, value)
    graphmod = fx.GraphModule(constants_root, graph, class_name="tp_forward")
    # == Optimize ==
    # TODO: when eliminate_dead_code() is in PyTorch stable, use that
    if optimize_einsums:
        # Note that for our einsums, we can optimize _once_ for _any_ batch dimension
        # and still get the right path for _all_ batch dimensions.
        # This is because our einsums are essentially of the form:
        #    zuvw,ijk,zuvij->zwk    OR     uvw,ijk,zuvij->zwk
        # In the first case, all but one operands have the batch dimension
        #    => The first contraction gains the batch dimension
        #    => All following contractions have batch dimension
        #    => All possible contraction paths have cost that scales linearly in batch size
        #    => The optimal path is the same for all batch sizes
        # For the second case, this logic follows as long as the first contraction is not between the first two operands. Since those two operands do not share any indexes, contracting them first is a rare pathological case. See
        # https://github.com/dgasmith/opt_einsum/issues/158
        # for more details.
        #
        # TODO: consider the impact maximum intermediate result size on this logic
        #         \- this is the `memory_limit` option in opt_einsum
        # TODO: allow user to choose opt_einsum parameters?
        #
        # We use float32 and zeros to save memory and time, since opt_einsum_fx looks only at traced shapes, not values or dtypes.
        batchdim = 4
        irreps_in1_dim = sum(irreps_in1[i].mul * irreps_in1[i].rep.dim for i in range(len(irreps_in1)))
        irreps_in2_dim = sum(irreps_in2[i].mul * irreps_in2[i].rep.dim for i in range(len(irreps_in2)))
        example_inputs = (
            torch.zeros((batchdim, irreps_in1_dim), dtype=type),
            torch.zeros((batchdim, irreps_in2_dim), dtype=type),
            torch.zeros((1 if shared_weights else batchdim, flat_weight_index),
                        dtype=type),
        )
        graphmod = optimize_einsums_full(graphmod, example_inputs)
    return graphmod

