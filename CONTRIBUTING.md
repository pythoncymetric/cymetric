# Contributing

Thanks for using the package and being interested in contributing to the project. 
There are several ways to contribute.

## Style, Documentation and Tutorials

We are mostly following the [PEP 8](https://www.python.org/dev/peps/pep-0008/)
python style guide with line width 80 and [Google](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
style docstrings. At parts we diverge from these styles and could probably need
some help to find back to it. Contributions to a more extensive 
and better documentation are always welcome. In similar vein, the tutorials can
be improved with more usage capabilities and more interesting examples.

## Bugs and Requests

If you find any bugs or unexpected behaviour, please open an issue. Include a 
minimal working (or not working) example following above mentioned style guidelines.
If you already have a workaround for a bug/issue, open a pull request. Fork
the repository and run the testing scripts in order to check that the code doesn't 
break unexpectedly.

## Testing

Our current testing scripts are rather bare bone. A more extensive testing suit
will improve the package.

## TODOs

The different modules often contain various little TODO-tags, which indicate 
possible performance improvements, clean-up, or additional features. We will be
grateful, when someone ticks them off the list.

## Road Map

There are several more features planned for the future. Those include in no particular
order:

### TensorFlow models

1. Adding more general SU(3)-structure learning as in [2021.04656](https://arxiv.org/abs/2012.04656).
2. Adding (vanishing) divisor integrals to fix the metric to particular points in 
    Kaehler moduli space for other Ansätze than the PhiModels.

### PointGeneration

1. Adding MCMC-methods as comparison to the theorem by Shiffman and Zelditch.
2. Work out generalisation of [SZ] for the Toric manifolds.
3. Improve performance of point generation. Either by adding other backend options 
    to faster libraries or utilizing GPU, or ... ?
4. Adding toric interface to utilize triangulations and toric data from [cytools](https://github.com/LiamMcAllisterGroup/cytools).

### Miscellaneous

1. Add more potential methods, such as energy functionals, learning the hbalanced
    metric, learning function basis.
2. Add bundle metric.

We are of course open for more suggestions and welcome every feedback.