from collections import OrderedDict
from math import sqrt
from typing import List, NamedTuple, Optional, Union, Dict, Tuple
from LieCG import so13
from LieCG.CG_coefficients.CG_lorentz import CGDict, clebschmat
from LieCG.so13.utils import CodeGenMixin, _sum_tensors, prod

import torch
import torch.fx
from opt_einsum_fx import optimize_einsums_full

from torch import fx


class Instruction(NamedTuple):
    i_in1: int
    i_in2: int
    i_out: int
    connection_mode: str
    has_weight: bool
    path_weight: float
    path_shape: tuple


class TensorProduct(CodeGenMixin, torch.nn.Module):

    def __init__(
        self,
        irreps_in1: so13.Lorentz_Irreps,
        irreps_in2: so13.Lorentz_Irreps,
        irreps_out: so13.Lorentz_Irreps,
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

        self.irreps_in1 = so13.Lorentz_Irreps(irreps_in1)
        self.irreps_in2 = so13.Lorentz_Irreps(irreps_in2)
        self.irreps_out = so13.Lorentz_Irreps(irreps_out)

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
            assert abs(
                mul_ir_in1.ir.l - mul_ir_in2.ir.l
            ) <= mul_ir_out.ir.l <= mul_ir_in1.ir.l + mul_ir_in2.ir.l
            assert abs(
                mul_ir_in1.ir.k - mul_ir_in2.ir.k
            ) <= mul_ir_out.ir.k <= mul_ir_in1.ir.k + mul_ir_in2.ir.k
            assert ins.connection_mode in [
                'uvw', 'uvu', 'uvv', 'uuw', 'uuu', 'uvuv', 'uvu<v', 'u<vw'
            ]

            if irrep_normalization == 'component':
                alpha = mul_ir_out.ir.dim
            if irrep_normalization == 'norm':
                alpha = mul_ir_in1.ir.dim * mul_ir_in2.ir.dim
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

        self._in1_dim = self.irreps_in1.dim
        self._in2_dim = self.irreps_in2.dim

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

        if self.irreps_out.dim > 0:
            output_mask = torch.cat([
                torch.ones(mul * ir.dim) if any(
                    (ins.i_out == i_out) and (ins.path_weight != 0) and (
                        0 not in ins.path_shape)
                    for ins in self.instructions) else torch.zeros(mul *
                                                                   ir.dim)
                for i_out, (mul, ir) in enumerate(self.irreps_out)
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
        return (
            f"{self.__class__.__name__}"
            f"({self.irreps_in1.simplify()} x {self.irreps_in2.simplify()} "
            f"-> {self.irreps_out.simplify()} | {npath} paths | {self.weight_numel} weights)"
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
    irreps_in1: so13.Lorentz_Irreps,
    irreps_in2: so13.Lorentz_Irreps,
    irreps_out: so13.Lorentz_Irreps,
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

    output_shape = output_shape + (irreps_out.dim, )

    x1s = x1s.reshape(-1, irreps_in1.dim)
    x2s = x2s.reshape(-1, irreps_in2.dim)

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
            x1s.reshape(batch_numel, irreps_in1[0].mul, irreps_in1[0].ir.dim)
        ]
    else:
        x1_list = [
            x1s[:, i].reshape(batch_numel, mul_ir.mul, mul_ir.ir.dim)
            for i, mul_ir in zip(irreps_in1.slices(), irreps_in1)
        ]

    x2_list = []
    # If only one input irrep, can avoid creating a view
    if len(irreps_in2) == 1:
        x2_list.append(
            x2s.reshape(batch_numel, irreps_in2[0].mul, irreps_in2[0].ir.dim))
    else:
        for i, mul_ir in zip(irreps_in2.slices(), irreps_in2):
            x2_list.append(x2s[:, i].reshape(batch_numel, mul_ir.mul,
                                             mul_ir.ir.dim))

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

        assert abs(mul_ir_in1.ir.l - mul_ir_in2.ir.l
                   ) <= mul_ir_out.ir.l <= mul_ir_in1.ir.l + mul_ir_in2.ir.l
        assert abs(mul_ir_in1.ir.k - mul_ir_in2.ir.k
                   ) <= mul_ir_out.ir.k <= mul_ir_in1.ir.k + mul_ir_in2.ir.k

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
        key = ((mul_ir_in1.ir.l, mul_ir_in1.ir.k),
               (mul_ir_in2.ir.l, mul_ir_in2.ir.k), (mul_ir_out.ir.l,
                                                    mul_ir_out.ir.k))
        if key not in cg_keys:
            cg_dict_out[key] = fx.Proxy(graph.get_attr(
                f"_cg_{key[0][0],key[0][1]}_{key[1][0],key[1][1]}_{key[2][0],key[2][1]}"
            ),
                                        tracer=tracer)
            cg_keys.append(key)
        cg = cg_dict_out[key]

        #Store the cg in constants.pt to not compute them
        #cg = fx.Proxy(graph.get_attr(cg_name), tracer=tracer)

        l1k1l2k2l3k3 = ((mul_ir_in1.ir.l, mul_ir_in1.ir.k),
                        (mul_ir_in2.ir.l, mul_ir_in2.ir.k), (mul_ir_out.ir.l,
                                                             mul_ir_out.ir.k))
        if ins.connection_mode == 'uvw':
            assert ins.has_weight
            if specialized_code and l1k1l2k2l3k3 == ((0, 0), (0, 0), (0, 0)):
                result = torch.einsum(f"{z}uvw,zu,zv->zw", w,
                                      x1.reshape(batch_numel, mul_ir_in1.dim),
                                      x2.reshape(batch_numel, mul_ir_in2.dim))
            elif specialized_code and mul_ir_in1.ir.l == 0 and mul_ir_in1.ir.k == 0:
                result = torch.einsum(f"{z}uvw,zu,zvj->zwj", w,
                                      x1.reshape(batch_numel, mul_ir_in1.dim),
                                      x2) / sqrt(mul_ir_out.ir.dim)
            elif specialized_code and mul_ir_in2.ir.l == 0 and mul_ir_in2.ir.k == 0:
                result = torch.einsum(f"{z}uvw,zui,zv->zwi", w, x1,
                                      x2.reshape(batch_numel,
                                                 mul_ir_in2.dim)) / sqrt(
                                                     mul_ir_out.ir.dim)
            elif specialized_code and mul_ir_out.ir.l == 0 and mul_ir_out.ir.k == 0:
                result = torch.einsum(f"{z}uvw,zui,zvi->zw", w, x1, x2) / sqrt(
                    mul_ir_in1.ir.dim)
            else:
                result = torch.einsum(f"{z}uvw,ijk,zuvij->zwk", w, cg, xx)
        if ins.connection_mode == 'uvu':
            assert mul_ir_in1.mul == mul_ir_out.mul
            if ins.has_weight:
                if specialized_code and l1k1l2k2l3k3 == ((0, 0), (0, 0),
                                                         (0, 0)):
                    result = torch.einsum(
                        f"{z}uv,zu,zv->zu", w,
                        x1.reshape(batch_numel, mul_ir_in1.dim),
                        x2.reshape(batch_numel, mul_ir_in2.dim))
                elif specialized_code and mul_ir_in1.ir.l == 0 and mul_ir_in1.ir.k == 0:
                    result = torch.einsum(
                        f"{z}uv,zu,zvj->zuj", w,
                        x1.reshape(batch_numel, mul_ir_in1.dim), x2) / sqrt(
                            mul_ir_out.ir.dim)
                elif specialized_code and mul_ir_in2.ir.l == 0 and mul_ir_in2.ir.k == 0:
                    result = torch.einsum(
                        f"{z}uv,zui,zv->zui", w, x1,
                        x2.reshape(batch_numel, mul_ir_in2.dim)) / sqrt(
                            mul_ir_out.ir.dim)
                else:
                    result = torch.einsum(f"{z}uv,ijk,zuvij->zuk", w, cg, xx)
            else:
                # not so useful operation because v is summed
                result = torch.einsum("ijk,zuvij->zuk", cg, xx)
        if ins.connection_mode == 'uvv':
            assert mul_ir_in2.mul == mul_ir_out.mul
            if ins.has_weight:
                if specialized_code and l1k1l2k2l3k3 == ((0, 0), (0, 0),
                                                         (0, 0)):
                    result = torch.einsum(
                        f"{z}uv,zu,zv->zv", w,
                        x1.reshape(batch_numel, mul_ir_in1.dim),
                        x2.reshape(batch_numel, mul_ir_in2.dim))
                elif specialized_code and mul_ir_in1.ir.l == 0 and mul_ir_in1.ir.k == 0:
                    result = torch.einsum(
                        f"{z}uv,zu,zvj->zvj", w,
                        x1.reshape(batch_numel, mul_ir_in1.dim), x2) / sqrt(
                            mul_ir_out.ir.dim)
                elif specialized_code and mul_ir_in2.ir.l == 0 and mul_ir_in2.ir.k == 0:
                    result = torch.einsum(
                        f"{z}uv,zui,zv->zvi", w, x1,
                        x2.reshape(batch_numel, mul_ir_in2.dim)) / sqrt(
                            mul_ir_out.ir.dim)
                elif specialized_code and mul_ir_out.ir.l == 0 and mul_ir_out.ir.k == 0:
                    result = torch.einsum(f"{z}uv,zui,zvi->zv", w, x1,
                                          x2) / sqrt(mul_ir_in1.ir.dim)
                else:
                    result = torch.einsum(f"{z}uv,ijk,zuvij->zvk", w, cg, xx)
            else:
                # not so useful operation because u is summed
                # only specialize out for this path
                if specialized_code and l1k1l2k2l3k3 == ((0, 0), (0, 0),
                                                         (0, 0)):
                    result = torch.einsum(
                        "zu,zv->zv", x1.reshape(batch_numel, mul_ir_in1.dim),
                        x2.reshape(batch_numel, mul_ir_in2.dim))
                elif specialized_code and mul_ir_in1.ir.l == 0 and mul_ir_in1.ir.k == 0:
                    result = torch.einsum(
                        "zu,zvj->zvj", x1.reshape(batch_numel, mul_ir_in1.dim),
                        x2) / sqrt(mul_ir_out.ir.dim)
                elif specialized_code and mul_ir_in2.ir.l == 0 and mul_ir_in2.ir.k == 0:
                    result = torch.einsum(
                        "zui,zv->zvi", x1,
                        x2.reshape(batch_numel, mul_ir_in2.dim)) / sqrt(
                            mul_ir_out.ir.dim)
                elif specialized_code and mul_ir_out.ir.l == 0 and mul_ir_out.ir.k == 0:
                    result = torch.einsum("zui,zvi->zv", x1, x2) / sqrt(
                        mul_ir_in1.ir.dim)
                else:
                    result = torch.einsum("ijk,zuvij->zvk", cg, xx)
        if ins.connection_mode == 'uuw':
            assert mul_ir_in1.mul == mul_ir_in2.mul
            if ins.has_weight:
                if specialized_code and l1k1l2k2l3k3 == ((0, 0), (0, 0),
                                                         (0, 0)):
                    result = torch.einsum(
                        f"{z}uw,zu,zu->zw", w,
                        x1.reshape(batch_numel, mul_ir_in1.dim),
                        x2.reshape(batch_numel, mul_ir_in2.dim))
                elif specialized_code and mul_ir_in1.ir.l == 0 and mul_ir_in1.ir.k == 0:
                    result = torch.einsum(
                        f"{z}uw,zu,zuj->zwj", w,
                        x1.reshape(batch_numel, mul_ir_in1.dim), x2) / sqrt(
                            mul_ir_out.ir.dim)
                elif specialized_code and mul_ir_in2.ir.l == 0 and mul_ir_in2.ir.k == 0:
                    result = torch.einsum(
                        f"{z}uw,zui,zu->zwi", w, x1,
                        x2.reshape(batch_numel, mul_ir_in2.dim)) / sqrt(
                            mul_ir_out.ir.dim)
                elif specialized_code and mul_ir_out.ir.l == 0 and mul_ir_out.ir.k == 0:
                    result = torch.einsum(f"{z}uw,zui,zui->zw", w, x1,
                                          x2) / sqrt(mul_ir_in1.ir.dim)
                else:
                    result = torch.einsum(f"{z}uw,ijk,zuij->zwk", w, cg, xx)
            else:
                # equivalent to tp(x, y, 'uuu').sum('u')
                assert mul_ir_out.mul == 1
                result = torch.einsum("ijk,zuij->zk", cg, xx)
        if ins.connection_mode == 'uuu':
            assert mul_ir_in1.mul == mul_ir_in2.mul == mul_ir_out.mul
            if ins.has_weight:
                if specialized_code and l1k1l2k2l3k3 == ((0, 0), (0, 0),
                                                         (0, 0)):
                    result = torch.einsum(
                        f"{z}u,zu,zu->zu", w,
                        x1.reshape(batch_numel, mul_ir_in1.dim),
                        x2.reshape(batch_numel, mul_ir_in2.dim))
                elif specialized_code and mul_ir_in1.ir.l == 0 and mul_ir_in1.ir.k == 0:
                    result = torch.einsum(
                        f"{z}u,zu,zuj->zuj", w,
                        x1.reshape(batch_numel, mul_ir_in1.dim), x2) / sqrt(
                            mul_ir_out.ir.dim)
                elif specialized_code and mul_ir_in2.ir.l == 0 and mul_ir_in2.ir.k == 0:
                    result = torch.einsum(
                        f"{z}u,zui,zu->zui", w, x1,
                        x2.reshape(batch_numel, mul_ir_in2.dim)) / sqrt(
                            mul_ir_out.ir.dim)
                elif specialized_code and mul_ir_out.ir.l == 0 and mul_ir_out.ir.k == 0:
                    result = torch.einsum(f"{z}u,zui,zui->zu", w, x1,
                                          x2) / sqrt(mul_ir_in1.ir.dim)
                else:
                    result = torch.einsum(f"{z}u,ijk,zuij->zuk", w, cg, xx)
            else:
                if specialized_code and l1k1l2k2l3k3 == ((0, 0), (0, 0),
                                                         (0, 0)):
                    result = torch.einsum(
                        "zu,zu->zu", x1.reshape(batch_numel, mul_ir_in1.dim),
                        x2.reshape(batch_numel, mul_ir_in2.dim))
                elif specialized_code and mul_ir_in1.ir.l == 0 and mul_ir_in1.ir.k == 0:
                    result = torch.einsum(
                        "zu,zuj->zuj", x1.reshape(batch_numel, mul_ir_in1.dim),
                        x2) / sqrt(mul_ir_out.ir.dim)
                elif specialized_code and mul_ir_in2.ir.l == 0 and mul_ir_in2.ir.k == 0:
                    result = torch.einsum(
                        "zui,zu->zui", x1,
                        x2.reshape(batch_numel, mul_ir_in2.dim)) / sqrt(
                            mul_ir_out.ir.dim)
                elif specialized_code and mul_ir_out.ir.l == 0 and mul_ir_out.ir.k == 0:
                    result = torch.einsum("zui,zui->zu", x1, x2) / sqrt(
                        mul_ir_in1.ir.dim)
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
    for (l_1, k_1), (l_2, k_2), (l_out, k_out) in cg_keys:
        cg_mats[f"_cg_{l_1,k_1}_{l_2,k_2}_{l_out,k_out}"] = clebschmat(
            (l_1, k_1), (l_2, k_2), (l_out, k_out)).to(type)
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
        example_inputs = (
            torch.zeros((batchdim, irreps_in1.dim), dtype=type),
            torch.zeros((batchdim, irreps_in2.dim), dtype=type),
            torch.zeros((1 if shared_weights else batchdim, flat_weight_index),
                        dtype=type),
        )
        graphmod = optimize_einsums_full(graphmod, example_inputs)
    return graphmod


class FullyConnectedTensorProduct(TensorProduct):
    r"""Fully-connected weighted tensor product
    All the possible path allowed by :math:`|l_1 - l_2| \leq l_{out} \leq l_1 + l_2` 
                                     :math:`|k_1 - k_2| \leq l_{out} \leq k_1 + k_2`are made.
    The output is a sum on different paths:
    .. math::
        z_w = \sum_{u,v} w_{uvw} x_u \otimes y_v + \cdots \text{other paths}
    where :math:`u,v,w` are the indices of the multiplicities.
    Parameters
    ----------
    irreps_in1 : `so13.Lorentz_Irreps`
        representation of the first input
    irreps_in2 : `so13.Lorentz_Irreps`
        representation of the second input
    irreps_out : `so13.Lorentz_Irreps`
        representation of the output
    irrep_normalization : {'component', 'norm'}
        see `so13.Lorentz_Irreps`
    path_normalization : {'element', 'path'}
        see `so13.Lorentz_Irreps`
    internal_weights : bool
        see `so13.Lorentz_Irreps`
    shared_weights : bool
        see `so13.Lorentz_Irreps`
    """

    def __init__(self,
                 irreps_in1,
                 irreps_in2,
                 irreps_out,
                 irrep_normalization: str = None,
                 path_normalization: str = None,
                 **kwargs):
        irreps_in1 = so13.Lorentz_Irreps(irreps_in1)
        irreps_in2 = so13.Lorentz_Irreps(irreps_in2)
        irreps_out = so13.Lorentz_Irreps(irreps_out)

        instr = [(i_1, i_2, i_out, 'uvw', True, 1.0)
                 for i_1, (_, ir_1) in enumerate(irreps_in1)
                 for i_2, (_, ir_2) in enumerate(irreps_in2)
                 for i_out, (_, ir_out) in enumerate(irreps_out)
                 if ir_out in ir_1 * ir_2]
        super().__init__(irreps_in1,
                         irreps_in2,
                         irreps_out,
                         instr,
                         irrep_normalization=irrep_normalization,
                         path_normalization=path_normalization,
                         **kwargs)


class ElementwiseTensorProduct(TensorProduct):
    r"""Elementwise connected tensor product.
    .. math::
        z_u = x_u \otimes y_u
    where :math:`u` runs over the irreps. Note that there are no weights.
    The output representation is determined by the two input representations.
    Parameters
    ----------
    irreps_in1 : `so13.Lorentz_Irreps`
        representation of the first input
    irreps_in2 : `so13.Lorentz_Irreps`
        representation of the second input
    filter_ir_out : iterator of `so13.Lorentz_Irrep`, optional
        filter to select only specific `so13.Lorentz_Irrep` of the output
    normalization : {'component', 'norm'}
        see `so13.TensorProduct`
    Examples
    --------
    Elementwise scalar product
    """

    def __init__(self, irreps_in1, irreps_in2, filter_ir_out=None, **kwargs):

        irreps_in1 = so13.Lorentz_Irreps(irreps_in1).simplify()
        irreps_in2 = so13.Lorentz_Irreps(irreps_in2).simplify()
        if filter_ir_out is not None:
            filter_ir_out = [so13.Lorentz_Irrep(ir) for ir in filter_ir_out]

        assert irreps_in1.num_irreps == irreps_in2.num_irreps

        irreps_in1 = list(irreps_in1)
        irreps_in2 = list(irreps_in2)

        i = 0
        while i < len(irreps_in1):
            mul_1, ir_1 = irreps_in1[i]
            mul_2, ir_2 = irreps_in2[i]

            if mul_1 < mul_2:
                irreps_in2[i] = (mul_1, ir_2)
                irreps_in2.insert(i + 1, (mul_2 - mul_1, ir_2))

            if mul_2 < mul_1:
                irreps_in1[i] = (mul_2, ir_1)
                irreps_in1.insert(i + 1, (mul_1 - mul_2, ir_1))
            i += 1

        out = []
        instr = []
        for i, ((mul, ir_1), (mul_2,
                              ir_2)) in enumerate(zip(irreps_in1, irreps_in2)):
            assert mul == mul_2
            for ir in ir_1 * ir_2:

                if filter_ir_out is not None and ir not in filter_ir_out:
                    continue

                i_out = len(out)
                out.append((mul, ir))
                instr += [(i, i, i_out, 'uuu', False)]

        super().__init__(irreps_in1, irreps_in2, out, instr, **kwargs)