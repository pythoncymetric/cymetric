# cymetric

cymetric is a Python package for learning of moduli-dependent Calabi-Yau metrics
using neural networks implemented in TensorFlow. 

## Features

The current version is an alpha-release so not all features mentioned below
are on the main branch yet. Features with an (*) will be released soonish.

* Point Generators for Complete Intersection Calabi-Yau manifolds and hypersurfaces
 from the Kreuzer-Skarke list (requires SageMath).
* A collection of custom TensorFlow models with different metric Ansätze.
* A matehmatica API for the point generators and TensorFlow models(*).
* Custom models for the bundle metric(*).
* Documentation exists(*).

## Installation

Create a new environment with

```console
conda create -n cymetric python=3.9
```

Then install with pip directly from github 

```console
conda activate cymetric
pip install git+https://github.com/robin-schneider/cymetric.git
```

or if you want to have a more stable release with

```console
pip install cymetric
```

## Tutorials

There are some tutorials

1. In [1.PointGenerator.ipynb](notebooks/1.PointGenerator.ipynb) we explore the three
 different PointGenerators for codimension-1 CICY, general CICYs and CY in toric varieties
 on the Fermat Quintic. 
2. In [2.TensorFlow_models.ipynb](notebooks/2.TensorFlow_models.ipynb) we explore some of the
 TF custom models with the data generated in the first notebook. 
3. (*) There exists a [Mathematica integration](/notebooks/CYMetrics.nb), which allows to call the PointGenerators and the TensorFlow models. Furthermore, there are arbitrary
precision PointGenerators based on the wolfram language.

## Contributing

We welcome contributions to the project. Those can be bug reports or new features, 
that you have or want to be implemented. Please read more [here](CONTRIBUTING.md).

## Citation

There will soon be an accompanying paper.
