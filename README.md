# cymetric

cymetric is a Python package for learning of moduli-dependent Calabi-Yau metrics
using neural networks implemented in TensorFlow. 

## Features

The current version is an alpha-release so not all features mentioned below
are on the main branch yet. Features with an (*) will be released soonish.

* Point Generators for Complete Intersection Calabi-Yau manifolds and hypersurfaces
 from the Kreuzer-Skarke list (requires SageMath and Mathematica).
* A collection of custom TensorFlow models with different metric Ansätze.
* A matehmatica API for the point generators and TensorFlow models.
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
pip install git+https://github.com/pythoncymetric/cymetric.git
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

You can find our paper on the [arXiv](https://arxiv.org/abs/2111.01436). It will be presented at the [ML4PS workshop](https://ml4physicalsciences.github.io/2021/) of [NeurIPS 2021](https://neurips.cc/Conferences/2021/Schedule?showEvent=21862). If you find this package useful in your work, cite the following bib entry:

```
@article{Larfors:2021pbb,
    author = "Larfors, Magdalena and Lukas, Andre and Ruehle, Fabian and Schneider, Robin",
    title = "{Learning Size and Shape of Calabi-Yau Spaces}",
    eprint = "2111.01436",
    archivePrefix = "arXiv",
    primaryClass = "hep-th",
    reportNumber = "UUITP-53/21",
    year = "2021",
    journal = "Machine Learning and the Physical Sciences, Workshop at 35th NeurIPS",
}
```
