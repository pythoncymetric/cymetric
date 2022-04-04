Installation
------------
This guide assumes that you have a working Python 3 (preferably python 3.7 or above) installation (and Sage and Mathematica, if you want to use these features as well). So running python3 should work on your system. Moreover, it assumes that you have installed git. Note that both are standard on Mac and most Linux distributions. For Windows, you will typically have to install them and make sure that for example Python works correctly with Mathematica if you are planing on using the Mathematica interface.

1. Install with Python
======================

If you want to use any existing python installation (note that we recommend using a virtual environment, see below), just run in a terminal

.. code-block:: bash

    pip install git+https://github.com/pythoncymetric/cymetric.git

To run the example notebooks, you need jupyter. You can install it with

.. code-block:: bash

    pip install jupyter notebook


2. Install with virtual environment
===================================

Using standard virtual environment

Create a new virtual environment in a terminal with

.. code-block:: bash

	python3 -m venv ~/cymetric

Then install with pip directly from github

.. code-block:: bash

	source ~/cymetric/bin/activate
	pip install --upgrade pip
	pip install git+https://github.com/pythoncymetric/cymetric.git
	pip install jupyter notebook
	python -m ipykernel install --user --name=cymetric


**Using anaconda**

Create a new environment with

.. code-block:: bash

    conda create -n cymetric python=3.9

Then install with pip directly from github

.. code-block:: bash

    conda activate cymetric
    pip install git+https://github.com/pythoncymetric/cymetric.git


3. Install within Sage
======================

Since sage comes with python, all you need to do is run

.. code-block:: bash

	pip install git+https://github.com/pythoncymetric/cymetric.git

from within a sage notebook. If you'd rather keep ML and sage separate, you can just install the package (without tensorflow etc.) using

.. code-block:: bash

	pip install --no-dependencies git+https://github.com/pythoncymetric/cymetric.git

Then you can use the function prepare_toric_cy_data(tv, "toric_data.pickle")) to create and store all the toric data needed, and then run the ML algorithms with this data file from a separate package installation (with tensorflow).


4. Install within Mathematica
=============================

The whole installation process is fully automatic in the Mathematica notebook. Just download it and follow the instructions in the notebook. In a nutshell, you run

.. code-block:: bash

	Get["https://raw.githubusercontent.com/pythoncymetric/cymetric/main/cymetric/wolfram/cymetric.m"];
	PathToVenv = FileNameJoin[{$HomeDirectory, "cymetric"}];
	python = Setup[PathToVenv];


You can also use an already existing installation. To do so, you run

.. code-block:: bash

	Get["https://raw.githubusercontent.com/pythoncymetric/cymetric/main/cymetric/wolfram/cymetric.m"];
	PathToVenv = FileNameJoin[{$HomeDirectory, "cymetric"}];
	ChangeSetting["Python", PathToVenv]
	python = Setup[PathToVenv];

Note that this will create a .m file (in the same folder and with the same name as the mathematica notebook) which stores the location of the virtual environment. If you delete this file, mathematica will install a new virtual environment the next time you call Setup[PathToVenv].

The help and documentation for Mathematica is built into the files. Download the `mathematica notebook  from the GitHub repository <https://github.com/pythoncymetric/cymetric/tree/main/notebooks>`_ for more details.
