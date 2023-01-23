import io
from abc import ABC
from typing import Dict, List

import torch
from opt_einsum_fx import jitable
from torch import fx
"""MIT License

Euclidean neural networks (e3nn) Copyright (c) 2020, The Regents of the
University of California, through Lawrence Berkeley National Laboratory
(subject to receipt of any required approvals from the U.S. Dept. of Energy), 
Ecole Polytechnique Federale de Lausanne (EPFL), Free University of Berlin 
and Kostiantyn Lapchevskyi. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy 
of this software and associated documentation files (the "Software"), to deal 
in the Software without restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the 
Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:"""


def prod(x):
    """Compute the product of a sequence."""
    out = 1
    for a in x:
        out *= a
    return out


def _sum_tensors(xs: List[torch.Tensor], shape: torch.Size,
                 like: torch.Tensor):
    if len(xs) > 0:
        out = xs[0]
        for x in xs[1:]:
            out = out + x
        return out
    return like.new_zeros(shape)


class CodeGenMixin(ABC):
    """Mixin for classes that dynamically generate TorchScript code using FX.
    This class manages evaluating and compiling generated code for subclasses
    while remaining pickle/deepcopy compatible. If subclasses need to override
    ``__getstate__``/``__setstate__``, they should be sure to call CodeGenMixin's
    implimentation first and use its output.
    """

    # pylint: disable=super-with-arguments

    def _codegen_register(
        self,
        funcs: Dict[str, fx.GraphModule],
    ) -> None:
        """Register ``fx.GraphModule``s as TorchScript submodules.
        Parameters
        ----------
            funcs : Dict[str, fx.GraphModule]
                Dictionary mapping submodule names to graph modules.
        """
        if not hasattr(self, "__codegen__"):
            # list of submodule names that are managed by this object
            self.__codegen__ = []
        self.__codegen__.extend(funcs.keys())

        for fname, graphmod in funcs.items():
            assert isinstance(graphmod, fx.GraphModule)

            scriptmod = torch.jit.script(jitable(graphmod))
            assert isinstance(scriptmod, torch.jit.ScriptModule)

            # Add the ScriptModule as a submodule so it can be called
            setattr(self, fname, scriptmod)

    # In order to support copy.deepcopy and pickling, we need to not save the compiled TorchScript functions:
    # See pickle docs: https://docs.python.org/3/library/pickle.html#pickling-class-instances
    def __getstate__(self):
        # - Get a state to work with -
        # We need to check if other parent classes of self define __getstate__
        # torch.nn.Module does not currently impliment __get/setstate__ but
        # may in the future, which is why we have these hasattr checks for
        # other superclasses.
        if hasattr(super(CodeGenMixin, self), "__getstate__"):
            out = super(CodeGenMixin, self).__getstate__()
        else:
            out = self.__dict__

        out = out.copy()
        # We need a copy of the _modules OrderedDict
        # Otherwise, modifying the returned state will modify the current module itself
        out["_modules"] = out["_modules"].copy()

        # - Add saved versions of the ScriptModules to the state -
        codegen_state = {}
        if hasattr(self, "__codegen__"):
            for fname in self.__codegen__:
                # Get the module
                smod = getattr(self, fname)
                if isinstance(smod, fx.GraphModule):
                    smod = torch.jit.script(jitable(smod))
                assert isinstance(smod, torch.jit.ScriptModule)
                # Save the compiled code as TorchScript IR
                buffer = io.BytesIO()
                torch.jit.save(smod, buffer)
                # Serialize that IR (just some `bytes`) instead of
                # the ScriptModule
                codegen_state[fname] = buffer.getvalue()
                # Remove the compiled submodule from being a submodule
                # of the saved module
                del out["_modules"][fname]

            out["__codegen__"] = codegen_state
        return out

    def __setstate__(self, d):
        d = d.copy()
        # We don't want to add this to the object when we call super's __setstate__
        codegen_state = d.pop("__codegen__", None)

        # We need to initialize self first so that we can add submodules
        # We need to check if other parent classes of self define __getstate__
        if hasattr(super(CodeGenMixin, self), "__setstate__"):
            super(CodeGenMixin, self).__setstate__(d)
        else:
            self.__dict__.update(d)

        if codegen_state is not None:
            for fname, buffer in codegen_state.items():
                assert isinstance(fname, str)
                # Make sure bytes, not ScriptModules, got made
                assert isinstance(buffer, bytes)
                # Load the TorchScript IR buffer
                buffer = io.BytesIO(buffer)
                smod = torch.jit.load(buffer)
                assert isinstance(smod, torch.jit.ScriptModule)
                # Add the ScriptModule as a submodule
                setattr(self, fname, smod)
            self.__codegen__ = list(codegen_state.keys())


def moment(f, n, dtype=None, device=None):
    r"""
    compute n th moment
    <f(z)^n> for z normal
    """
    gen = torch.Generator(device="cpu").manual_seed(0)
    z = torch.randn(1_000_000, generator=gen, dtype=torch.get_default_dtype())
    return f(z).pow(n).mean()


class normalize2mom(torch.nn.Module):
    _is_id: bool
    cst: float

    def __init__(
            # pylint: disable=unused-argument
            self,
            f,
            dtype=None,
            device=None):
        super().__init__()

        # Try to infer a device:
        if device is None and isinstance(f, torch.nn.Module):
            # Avoid circular import
            device = f.buffer().device if f.buffer(
            ).device is not None else 'cpu'

        with torch.no_grad():
            cst = moment(f, 2, dtype=torch.float64,
                         device='cpu').pow(-0.5).item()

        if abs(cst - 1) < 1e-4:
            self._is_id = True
        else:
            self._is_id = False

        self.f = f
        self.cst = cst

    def forward(self, x):
        if self._is_id:
            return self.f(x)
        else:
            return self.f(x).mul(self.cst)
