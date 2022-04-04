Point Generators
----------------

There are three Point Generators

Point Generator
~~~~~~~~~~~~~~~

is used for co dimension one CICYs.

.. automodule:: cymetric.pointgen.pointgen
    :members:

Point Generator CICY
~~~~~~~~~~~~~~~~~~~~

is used for any CICY.

.. automodule:: cymetric.pointgen.pointgen_cicy
    :members:

Point Generator Toric
~~~~~~~~~~~~~~~~~~~~~

is used for toric Calabi Yaus.

.. automodule:: cymetric.pointgen.pointgen_toric
    :members:

Point Generator Mathematica
~~~~~~~~~~~~~~~~~~~~~~~~~~~

is used for both projective CICY and toric Calabi Yaus. It calls Mathematica and its powerful solvers as a backend to generate points.

PointGeneratorMathematica
=========================
.. autoclass:: cymetric.pointgen.pointgen_mathematica.PointGeneratorMathematica
    :members:

ToricPointGeneratorMathematica
==============================
.. autoclass:: cymetric.pointgen.pointgen_mathematica.ToricPointGeneratorMathematica
    :members:

ComplexFunctionConsumer
=========================
.. autoclass:: cymetric.pointgen.pointgen_mathematica.ComplexFunctionConsumer
    :members:

Numpy Helper
~~~~~~~~~~~~

are important and can make our life easier.

.. automodule:: cymetric.pointgen.nphelper
    :members:
