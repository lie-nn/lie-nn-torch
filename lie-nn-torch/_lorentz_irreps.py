import itertools
import collections
from typing import List, Optional, Union
from LieCG.CG_coefficients.CG_lorentz import LorentzD

import torch


class Lorentz_Irrep(tuple):
    r"""Irreducible representation of :math:`SO(1,3)`
    This class does not contain any data, it is a structure that describe the representation.
    It is typically used as argument of other classes of the library to define the input and output representations of functions.
    Parameters
    ----------
    l : int
        non-negative integer, the first degree of the representation, :math:`l = 0, 1, \dots`
    k : int
        non-negative integer, the second degree of the representation, :math:`l = 0, 1, \dots`
    Examples
    --------
    Create a scalar representation (:math:`(l,k)=(0,0)`) of even parity.
    >>> Irrep(0, 0) 
    (0,0)
    Create a pseudotensor representation (:math:`l=2`) of odd parity.
    >>> Irrep("(0,0)") + Irrep("(1,0)")
    1x(0,0)+1x(1,0)
    """
    def __new__(cls, l: Union[int, 'Lorentz_Irrep', str, tuple], 
                k: Optional[Union[int, 'Lorentz_Irrep', str, tuple]]=None):

        if isinstance(l, Lorentz_Irrep):
                return l

        if isinstance(l, str):
                try:
                    name = l.strip().split(',')
                    l = int(name[0][1:])
                    assert l >= 0
                    k = int(name[1][:-1])
                    assert k >= 0
                  
                except Exception:
                    raise ValueError(f"unable to convert string \"{name}\" into an Irrep")
        elif isinstance(l, tuple):
                l, k = l

        assert isinstance(l, int) and l >= 0, l
        assert isinstance(k, int) and k >= 0, k
        return super().__new__(cls, (l, k))

    @property
    def l(self) -> int:
        r"""The first degree of the representation, :math:`l = 0, 1, \dots`."""
        return self[0]

    @property
    def k(self) -> int:
        r"""The second degree of the representation, :math:`l = 0, 1, \dots`."""
        return self[1]

    def __repr__(self):
        return f"({self.l},{self.k})"

    @classmethod
    def iterator(cls, lmax=None, kmax=None):
        r"""Iterator through all the irreps of :math:`SO(1,3)`
        Examples
        --------
        >>> it = Irrep.iterator()
        >>> next(it), next(it), next(it), next(it)
        (0e, 0o, 1o, 1e)
        """
        if kmax is None :
            kmax = lmax

        for l in itertools.count(): 
            if l == lmax:
                break
            for k in itertools.count():
                yield Lorentz_Irrep(l,k)
                if k == kmax:
                    break

    def D_from_angles(self, alpha, beta, gamma, k=None):
        r"""Matrix :math:`p^k D^l(\alpha, \beta, \gamma)`
        (matrix) Representation of :math:`SO(1,3)`. :math:`D` is the representation of :math:`SO(1,3)`, see `wigner_D`.
        Parameters
        ----------
        alpha : `torch.Tensor`
            tensor of shape :math:`(...)`
            Rotation :math:`\alpha` around Y axis, applied third.
        beta : `torch.Tensor`
            tensor of shape :math:`(...)`
            Rotation :math:`\beta` around X axis, applied second.
        gamma : `torch.Tensor`
            tensor of shape :math:`(...)`
            Rotation :math:`\gamma` around Y axis, applied first.
        k : `torch.Tensor`, optional
            tensor of shape :math:`(...)`
            How many times the parity is applied.
        Returns
        -------
        `torch.Tensor`
            tensor of shape :math:`(..., (2l+1)(2k+1), (2l+1)(2k+1))`
        See Also
        --------
        so13.wigner_D
        Irreps.D_from_angles
        """

        alpha, beta, gamma= torch.broadcast_tensors(alpha, beta, gamma)
        return LorentzD((self.l, self.k), alpha, beta, gamma)


    @property
    def dim(self) -> int:
        """The dimension of the representation, :math:`2 l + 1`."""
        return (self.l + 1)*(self.k + 1)

    def is_scalar(self) -> bool:
        """Equivalent to ``l == 0 and k == 0``"""
        return self.l == 0 and self.k == 0

    def __mul__(self, other):
        r"""Generate the irreps from the product of two irreps.
        Returns
        -------
        generator of `e3nn.o3.Irrep`
        """
        other = Lorentz_Irrep(other)
        lmin = abs(self.l - other.l)
        lmax = self.l + other.l
        kmin = abs(self.k - other.k)
        kmax = abs(self.k + other.k)
        for l in range(lmin, lmax + 1,2):
            for k in range(kmin, kmax + 1,2):
                yield Lorentz_Irrep(l, k)

    def count(self, _value):
        raise NotImplementedError

    def index(self, _value):
        raise NotImplementedError

    def __rmul__(self, other):
        r"""
        >>> 3 * Irrep('1e')
        3x1e
        """
        assert isinstance(other, int)
        return Lorentz_Irreps([(other, self)])

    def __add__(self, other):
        return Lorentz_Irreps(self) + Lorentz_Irreps(other)

    def __contains__(self, _object):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class _MulIr(tuple):
    def __new__(cls, mul, ir=None):
        if ir is None:
            mul, ir = mul

        assert isinstance(mul, int)
        assert isinstance(ir, Lorentz_Irrep)
        return super().__new__(cls, (mul, ir))

    @property
    def mul(self) -> int:
        return self[0]

    @property
    def ir(self) -> Lorentz_Irrep:
        return self[1]

    @property
    def dim(self) -> int:
        return self.mul * self.ir.dim

    def __repr__(self):
        return f"{self.mul}x{self.ir}"

    def count(self, _value):
        raise NotImplementedError

    def index(self, _value):
        raise NotImplementedError


class Lorentz_Irreps(tuple):
    r"""Direct sum of irreducible representations of :math:`SO(1,3)`
    This class does not contain any data, it is a structure that describe the representation.
    It is typically used as argument of other classes of the library to define the input and output representations of functions.
    Attributes
    ----------
    dim : int
        the total dimension of the representation
    num_irreps : int
        number of irreps. the sum of the multiplicities
    ls : list of int
        list of :math:`l` values
    lmax : int
        maximum :math:`l` value
    ks : list of int
        list of :math:`l` values
    kmax : int
        maximum :math:`k` value
    Examples
    --------
    Create a representation of 100 :math:`l=0` of even parity and 50 vectors.
    >>> x = Irreps([(100, (0, 1)), (50, (1, 1))])
    >>> x
    100x(0,0)+50x(1,1)
    >>> x.dim
    300
    Create a representation of 100 :math:`(l,k)=(0,0)` of even parity and 50 vectors.
    >>> Irreps("100x(0,0) + 50x(1,1)")
    100x(0,0)+50x(1,1)
    """
    def __new__(cls, irreps=None) -> Union[_MulIr, 'Lorentz_Irreps']:
        if isinstance(irreps, Lorentz_Irreps):
            return super().__new__(cls, irreps)

        out = []
        if isinstance(irreps, Lorentz_Irrep):
            out.append(_MulIr(1, Lorentz_Irrep(irreps)))
        elif isinstance(irreps, str):
            try:
                if irreps.strip() != "":
                    for mul_ir in irreps.split('+'):
                        if 'x' in mul_ir:
                            mul, ir = mul_ir.split('x')
                            mul = int(mul)
                            ir = Lorentz_Irrep(ir)
                        else:
                            mul = 1
                            ir = Lorentz_Irrep(mul_ir)

                        assert isinstance(mul, int) and mul >= 0
                        out.append(_MulIr(mul, ir))
            except Exception:
                raise ValueError(f"Unable to convert string \"{irreps}\" into an Irreps")
        elif irreps is None:
            pass
        else:
            for mul_ir in irreps:
                mul = None
                ir = None

                if isinstance(mul_ir, str):
                    mul = 1
                    ir = Lorentz_Irrep(mul_ir)
                elif isinstance(mul_ir, Lorentz_Irrep):
                    mul = 1
                    ir = mul_ir
                elif isinstance(mul_ir, _MulIr):
                    mul, ir = mul_ir
                elif len(mul_ir) == 2:
                    mul, ir = mul_ir
                    ir = Lorentz_Irrep(ir)

                if not (isinstance(mul, int) and mul >= 0 and ir is not None):
                    raise ValueError(f"Unable to interpret \"{mul_ir}\" as an irrep.")

                out.append(_MulIr(mul, ir))
        return super().__new__(cls, out)

    @staticmethod
    def spherical_harmonics(lmax):
        r"""representation of the spherical harmonics
        Parameters
        ----------
        lmax : int
            maximum :math:`l`
        p : {1, -1}
            the parity of the representation
        Returns
        -------
        `e3nn.o3.Irreps`
            representation of :math:`(Y^0, Y^1, \dots, Y^{\mathrm{lmax}})`
        Examples
        --------
        >>> Irreps.spherical_harmonics(3)
        1x0e+1x1o+1x2e+1x3o
        >>> Irreps.spherical_harmonics(4, p=1)
        1x0e+1x1e+1x2e+1x3e+1x4e
        """
        return Lorentz_Irreps([(1, (l, l)) for l in range(lmax + 1)])

    def slices(self):
        r"""List of slices corresponding to indices for each irrep.
        Examples
        --------
        >>> Irreps('2x0e + 1e').slices()
        [slice(0, 2, None), slice(2, 5, None)]
        """
        s = []
        i = 0
        for mul_ir in self:
            s.append(slice(i, i + mul_ir.dim))
            i += mul_ir.dim
        return s

    def randn(self, *size, normalization='component', requires_grad=False, dtype=None, device=None):
        r"""Random tensor.
        Parameters
        ----------
        *size : list of int
            size of the output tensor, needs to contains a ``-1``
        normalization : {'component', 'norm'}
        Returns
        -------
        `torch.Tensor`
            tensor of shape ``size`` where ``-1`` is replaced by ``self.dim``
        Examples
        --------
        >>> Irreps("5x0e + 10x1o").randn(5, -1, 5, normalization='norm').shape
        torch.Size([5, 35, 5])
        >>> random_tensor = Irreps("2o").randn(2, -1, 3, normalization='norm')
        >>> random_tensor.norm(dim=1).sub(1).abs().max().item() < 1e-5
        True
        """
        di = size.index(-1)
        lsize = size[:di]
        rsize = size[di + 1:]

        if normalization == 'component':
            return torch.randn(*lsize, self.dim, *rsize, requires_grad=requires_grad, dtype=dtype, device=device)
        elif normalization == 'norm':
            x = torch.zeros(*lsize, self.dim, *rsize, requires_grad=requires_grad, dtype=dtype, device=device)
            with torch.no_grad():
                for s, (mul, ir) in zip(self.slices(), self):
                    r = torch.randn(*lsize, mul, ir.dim, *rsize, dtype=dtype, device=device)
                    r.div_(r.norm(2, dim=di + 1, keepdim=True))
                    x.narrow(di, s.start, mul * ir.dim).copy_(r.reshape(*lsize, -1, *rsize))
            return x
        else:
            raise ValueError("Normalization needs to be 'norm' or 'component'")

    def __getitem__(self, i) -> Union[_MulIr, 'Lorentz_Irreps']:
        x = super().__getitem__(i)
        if isinstance(i, slice):
            return Lorentz_Irreps(x)
        return x

    def __contains__(self, ir) -> bool:
        ir = Lorentz_Irrep(ir)
        return ir in (irrep for _, irrep in self)

    def count(self, ir) -> int:
        r"""Multiplicity of ``ir``.
        Parameters
        ----------
        ir : `Lorentz_Irrep`
        Returns
        -------
        `int`
            total multiplicity of ``ir``
        """
        ir = Lorentz_Irrep(ir)
        return sum(mul for mul, irrep in self if ir == irrep)

    def index(self, _object):
        raise NotImplementedError

    def __add__(self, irreps):
        irreps = Lorentz_Irreps(irreps)
        return Lorentz_Irreps(super().__add__(irreps))

    def __mul__(self, other):
        r"""
        >>> (Irreps('2x1e') * 3).simplify()
        6x1e
        """
        if isinstance(other, Lorentz_Irreps):
            raise NotImplementedError("Use o3.TensorProduct for this, see the documentation")
        return Lorentz_Irreps(super().__mul__(other))

    def __rmul__(self, other):
        r"""
        >>> 2 * Irreps('(0,0) + (1,0)')
        1x(0,0)+1x(1,0)+1x(0,0)+1x(1,0)
        """
        return Lorentz_Irreps(super().__rmul__(other))

    def simplify(self):
        """Simplify the representations.
        Returns
        -------
        `e3nn.o3.Irreps`
        Examples
        --------
        Note that simplify does not sort the representations.
        >>> Irreps("1e + 1e + 0e").simplify()
        2x1e+1x0e
        Equivalent representations which are separated from each other are not combined.
        >>> Irreps("1e + 1e + 0e + 1e").simplify()
        2x1e+1x0e+1x1e
        """
        out = []
        for mul, ir in self:
            if out and out[-1][1] == ir:
                out[-1] = (out[-1][0] + mul, ir)
            elif mul > 0:
                out.append((mul, ir))
        return Lorentz_Irreps(out)

    def remove_zero_multiplicities(self):
        """Remove any irreps with multiplicities of zero.
        Returns
        -------
        `e3nn.o3.Irreps`
        Examples
        --------
        >>> Irreps("4x0e + 0x1o + 2x3e").remove_zero_multiplicities()
        4x0e+2x3e
        """
        out = [(mul, ir) for mul, ir in self if mul > 0]
        return Lorentz_Irreps(out)

    def sort(self):
        r"""Sort the representations.
        Returns
        -------
        irreps : `e3nn.o3.Irreps`
        p : tuple of int
        inv : tuple of int
        Examples
        --------
        >>> Irreps("1e + 0e + 1e").sort().irreps
        1x0e+1x1e+1x1e
        >>> Irreps("2o + 1e + 0e + 1e").sort().p
        (3, 1, 0, 2)
        >>> Irreps("2o + 1e + 0e + 1e").sort().inv
        (2, 1, 3, 0)
        """
        Ret = collections.namedtuple("sort", ["irreps", "p", "inv"])
        out = [(ir, i, mul) for i, (mul, ir) in enumerate(self)]
        out = sorted(out)
        inv = tuple(i for _, i, _ in out)
        p = inverse(inv)
        irreps = Lorentz_Irreps([(mul, ir) for ir, _, mul in out])
        return Ret(irreps, p, inv)

    @property
    def dim(self) -> int:
        return sum(mul * ir.dim for mul, ir in self)

    @property
    def num_irreps(self) -> int:
        return sum(mul for mul, _ in self)

    @property
    def ls(self) -> List[int]:
        return [l for mul, (l, p) in self for _ in range(mul)]

    @property
    def lmax(self) -> int:
        if len(self) == 0:
            raise ValueError("Cannot get lmax of empty Irreps")
        return max(self.ls)

    def __repr__(self):
        return "+".join(f"{mul_ir}" for mul_ir in self)

    def D_from_angles(self, alpha, beta, gamma, k=None):
        r"""Matrix of the representation
        Parameters
        ----------
        alpha : `torch.Tensor`
            tensor of shape :math:`(...)`
        beta : `torch.Tensor`
            tensor of shape :math:`(...)`
        gamma : `torch.Tensor`
            tensor of shape :math:`(...)`
        k : `torch.Tensor`, optional
            tensor of shape :math:`(...)`
        Returns
        -------
        `torch.Tensor`
            tensor of shape :math:`(..., \mathrm{dim}, \mathrm{dim})`
        """
        return direct_sum(*[ir.D_from_angles(alpha, beta, gamma, k) for mul, ir in self for _ in range(mul)])

    def D_from_matrix(self, R):
        r"""Matrix of the representation
        Parameters
        ----------
        R : `torch.Tensor`
            tensor of shape :math:`(..., 3, 3)`
        Returns
        -------
        `torch.Tensor`
            tensor of shape :math:`(..., \mathrm{dim}, \mathrm{dim})`
        """
        d = torch.det(R).sign()
        R = d[..., None, None] * R
        k = (1 - d) / 2
        return self.D_from_angles(matrix_to_angles(R), k)

def direct_sum(*matrices):
    r"""Direct sum of matrices, put them in the diagonal
    """
    front_indices = matrices[0].shape[:-2]
    m = sum(x.size(-2) for x in matrices)
    n = sum(x.size(-1) for x in matrices)
    total_shape = list(front_indices) + [m, n]
    out = matrices[0].new_zeros(total_shape)
    i, j = 0, 0
    for x in matrices:
        m, n = x.shape[-2:]
        out[..., i: i + m, j: j + n] = x
        i += m
        j += n
    return out

def inverse(p):
    r"""
    compute the inverse permutation
    """
    return tuple(p.index(i) for i in range(len(p)))

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