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

### 1. Just install it with python
If you want to use any existing python installation (note that we recommend using a virtual environment, see below), just run
```console
pip install git+https://github.com/pythoncymetric/cymetric.git
```

### 2. Install with virtual environment
#### Using standard virtual environment
Create a new virtual environment with

```console
python3 -m venv ~/cymetric
```

Then install with pip directly from github 

```console
source ~/cymetric/bin/activate
pip install git+https://github.com/pythoncymetric/cymetric.git
```

#### Using anaconda
Create a new environment with

```console
conda create -n cymetric python=3.9
```

Then install with pip directly from github 

```console
conda activate cymetric
pip install git+https://github.com/pythoncymetric/cymetric.git
```

### 3. Install within sage
Since sage comes with python, all you need to do is run 
```console
pip install git+https://github.com/pythoncymetric/cymetric.git
```
from within a sage notebook. If you'd rather keep ML and sage separate, you can just install the package (without tensorflow etc.) using 
```console
pip install --no-dependencies git+https://github.com/pythoncymetric/cymetric.git
```
Then you can use the function ```prepare_toric_cy_data(tv, "toric_data.pickle"))``` to create and store all the toric data needed, and then run the ML algorithms with this data file from a separate package installation (with tensorflow).

### 4. Install with Mathematica
The whole installation process is fully automatic in the [Mathematica notebook](/notebooks/4.Mathematica_integration_example.nb). Just download it and follow the instructions in the notebook. In a nutshell, you run
```console
Get["https://raw.githubusercontent.com/pythoncymetric/cymetric/main/cymetric/wolfram/cymetric.m"];
PathToVenv = FileNameJoin[{$HomeDirectory, "cymetric"}];
python = Setup[PathToVenv];
```
You can also use an already existing installation. To do so, you run
```console
Get["https://raw.githubusercontent.com/pythoncymetric/cymetric/main/cymetric/wolfram/cymetric.m"];
PathToVenv = FileNameJoin[{$HomeDirectory, "cymetric"}];
ChangeSetting["Python", PathToVenv]
python = Setup[PathToVenv];
```


## Tutorials
Once you have installed the package (either in python, or in sage, or in Mathematica), you are probably looking for some examples on how to use it. We provide some tutorials/examples for each case. Just download the example file somewhere on your computer and run it:

1. In [1.PointGenerator.ipynb](notebooks/1.PointGenerator.ipynb) we explore the three different PointGenerators for codimension-1 CICY, general CICYs and CY in toric varieties on the Fermat Quintic. 
2. In [2.TensorFlow_models.ipynb](notebooks/2.TensorFlow_models.ipynb) we explore some of the TF custom models with the data generated in the first notebook. 
3. In [3.Sage_integration_.ipynb](notebooks/3.Sage_integration_example.ipynb) we illustrate how to run the package from within Sage to compute the CY metric on a Kreuzer-Skarke model.
4. In [Mathematica_integration_example.nb](/notebooks/4.Mathematica_integration_example.nb), we illustrate how to call the PointGenerators and the TensorFlow models for training and evaluation. Furthermore, there are arbitrary precision PointGenerators based on the wolfram language.

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
