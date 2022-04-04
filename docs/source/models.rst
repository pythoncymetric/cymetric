TensorFlow models
-----------------

Metrics are approximated with neural networks in TensorFlow.

Models based on Neural Networks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The various TensorFlow models are:

.. automodule:: cymetric.models.tfmodels

AddFSModel
==========
.. autoclass:: cymetric.models.tfmodels.AddFSModel
    :members:
    
FreeModel
=========
.. autoclass:: cymetric.models.tfmodels.FreeModel
    :members:
    
MatrixFSModel
=============
.. autoclass:: cymetric.models.tfmodels.MatrixFSModel
    :members:
    
MultFSModel
===========
.. autoclass:: cymetric.models.tfmodels.MultFSModel
    :members:
    
PhiFSModel
==========
.. autoclass:: cymetric.models.tfmodels.PhiFSModel
    :members:
    
ToricModel
==========
.. autoclass:: cymetric.models.tfmodels.ToricModel
    :members:
    
MatrixFSModelToric
==================
.. autoclass:: cymetric.models.tfmodels.MatrixFSModelToric
    :members:
    
PhiFSModelToric
===============
.. autoclass:: cymetric.models.tfmodels.PhiFSModelToric
    :members:

Custom metrics
~~~~~~~~~~~~~~

There are several custom metrics tracking the training process.

.. automodule:: cymetric.models.metrics

KaehlerLoss
===========
.. autoclass:: cymetric.models.metrics.KaehlerLoss
    :members:
    
RicciLoss
=========
.. autoclass:: cymetric.models.metrics.RicciLoss
    :members:
    
TotalLoss
=========
.. autoclass:: cymetric.models.metrics.TotalLoss
    :members:
    
TransitionLoss
==============
.. autoclass:: cymetric.models.metrics.TransitionLoss
    :members:
    
SigmaLoss
=========
.. autoclass:: cymetric.models.metrics.SigmaLoss
    :members:
    
VolkLoss
========
.. autoclass:: cymetric.models.metrics.VolkLoss
    :members:


Callbacks
~~~~~~~~~

Error measures, such as Ricci- and Sigma-measures are implemented as Callbacks.
There are also callbacks for controlling the alpha parameters and other
hyperparameters.

.. automodule:: cymetric.models.callbacks

AlphaCallback
=============
.. autoclass:: cymetric.models.callbacks.AlphaCallback
    :members:
    
KaehlerCallback
===============
.. autoclass:: cymetric.models.callbacks.KaehlerCallback
    :members:
    
RicciCallback
=============
.. autoclass:: cymetric.models.callbacks.RicciCallback
    :members:
    
SigmaCallback
=============
.. autoclass:: cymetric.models.callbacks.SigmaCallback
    :members:
    
TransitionCallback
==================
.. autoclass:: cymetric.models.callbacks.TransitionCallback
    :members:
    
VolkCallback
============
.. autoclass:: cymetric.models.callbacks.VolkCallback
    :members:

Measures
~~~~~~~~

The callbacks are tracking various error measures, such as

.. automodule:: cymetric.models.measures
    :members:

Fubini Study
~~~~~~~~~~~~

The Fubini Study metric is the base class from which all other 
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

