TensorFlow models
-----------------

Metrics are approximated with neural networks in TensorFlow.

Models based on Neural Networks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The various TensorFlow models are:

.. automodule:: cymetric.models.tfmodels
    :members:

Custom metrics
~~~~~~~~~~~~~~

There are several custom metrics tracking the training process.

.. automodule:: cymetric.models.metrics
    :members:

Callbacks
~~~~~~~~~

Error measures, such as Ricci- and Sigma-measures are implemented as Callbacks.
There are also callbacks for controlling the alpha parameters and other
hyperparameters.

.. automodule:: cymetric.models.callbacks
    :members:

Measures
~~~~~~~~

The callbacks are tracking various error measures, such as

.. automodule:: cymetric.models.measures
    :members:

Fubini~Study
~~~~~~~~~~~~

The Fubini~Study metric is the base class from which all other 
tensorflow models inherit.

.. automodule:: cymetric.models.fubinistudy
    :members:

Hermitian Yang-Mills equation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

can be computed

.. automodule:: cymetric.bundle.hym_helper
    :members:

TensorFlow helpers
~~~~~~~~~~~~~~~~~~

are often needed, cause they make our life easier.

.. automodule:: cymetric.models.tfhelper
    :members:

Loss functions
~~~~~~~~~~~~~~

are implemented

.. automodule:: cymetric.models.losses
    :members:

