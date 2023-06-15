# lie-nn-torch

Create neural networks equivariant to reductive Lie groups in torch. This library implements to torch functions to use the group operations implemented in [lie-nn](https://github.com/lie-nn/lie-nn) on torch tensors data.

# Functions

## Tensor product 

Tensor products between two irreducible representations of Lie group, reduced into a sum of irreducible representations.

```python
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
```

## Linear

```python
irrep1 = [lie.MulIrrep(3, lie.irreps.SU3((2, 1, 0))), lie.MulIrrep(3, lie.irreps.SU3((1, 1, 0)))]
irrep2 = [lie.MulIrrep(5, lie.irreps.SU3((1, 1, 0)))]

irreps1 = lie.ReducedRep.from_irreps(irrep1)
irreps2 = lie.ReducedRep.from_irreps(irrep2)

linear = Linear(irreps1, irreps2)

x1 = torch.randn(1, irreps1.dim)

out = linear(x1)

```

## Symmetric Power

## Formatting the code
```
pycln .
black .
```
## References

If you use this code, please cite our papers:
```text
@misc{batatia2023general,
      title={A General Framework for Equivariant Neural Networks on Reductive Lie Groups}, 
      author={Ilyes Batatia and Mario Geiger and Jose Munoz and Tess Smidt and Lior Silberman and Christoph Ortner},
      year={2023},
      eprint={2306.00091},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}
```
