from LieCG.so13.utils import normalize2mom
import torch

from LieCG import so13


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


class Activation(torch.nn.Module):
    r"""Scalar activation function.
    Odd scalar inputs require activation functions with a defined parity (odd or even).
    Parameters
    ----------
    irreps_in : `g_irrep`
        representation of the input
    acts : list of function or None
        list of activation functions, `None` if non-scalar or identity
    Examples
    --------
    >>> a = Activation("256x(0,0)", [torch.abs])
    >>> a.irreps_out
    """

    def __init__(self, irreps_in, acts, group):
        super().__init__()
        g_irrep = group.irrep
        irreps_in = g_irrep(irreps_in)
        assert len(irreps_in) == len(acts), (irreps_in, acts)

        # normalize the second moment
        acts = [normalize2mom(act) if act is not None else None for act in acts]

        irreps_out = []
        for (mul, (l_in, k_in)), act in zip(irreps_in, acts):
            if act is not None:
                if l_in != 0 or k_in != 0:
                    raise ValueError(
                        "Activation: cannot apply an activation function to a non-scalar input."
                    )

                irreps_out.append((mul, (l_in, k_in)))

        self.irreps_in = irreps_in
        self.irreps_out = g_irrep(irreps_out)
        self.acts = torch.nn.ModuleList(acts)
        assert len(self.irreps_in) == len(self.acts)

    def __repr__(self):
        acts = "".join(["x" if a is not None else " " for a in self.acts])
        return f"{self.__class__.__name__} [{acts}] ({self.irreps_in} -> {self.irreps_out})"

    def forward(self, features, dim=-1):
        """evaluate
        Parameters
        ----------
        features : `torch.Tensor`
            tensor of shape ``(...)``
        Returns
        -------
        `torch.Tensor`
            tensor of shape the same shape as the input
        """
        # - PROFILER - with torch.autograd.profiler.record_function(repr(self)):
        output = []
        index = 0
        for (mul, ir), act in zip(self.irreps_in, self.acts):
            if act is not None:
                output.append(act(features.narrow(dim, index, mul)))
            else:
                output.append(features.narrow(dim, index, mul * ir.dim))
            index += mul * ir.dim

        if len(output) > 1:
            return torch.cat(output, dim=dim)
        elif len(output) == 1:
            return output[0]
        else:
            return torch.zeros_like(features)
