NeuroR
======

Introduction
------------

NeuroR is a collection of tools to repair morphologies.

Citation
--------

NeuroR implements the methods discussed in the following paper:

   Anwar H., Riachi I., Schürmann F., Markram H. (2009). “An approach to capturing neuron morphological diversity,”
   in Computational Neuroscience: Realistic Modeling for Experimentalistsed. De Schutter E.,
   editor. (Cambridge: The MIT Press) 211–232

   `DOI: 10.7551/mitpress/9780262013277.003.0010 <https://doi.org/10.7551/mitpress/9780262013277.003.0010>`__

Morphology repair with NeuroR
-----------------------------

There are presently three types of repair which are outlined below.

Sanitization
~~~~~~~~~~~~

This is the process of curating a morphological file. It currently:

- removes zero-length segments
- raises if the morphology has no soma
- raises if the morphology has negative diameters

Note: more functionality may be added in the future


Cut plane repair
~~~~~~~~~~~~~~~~

The cut plane repair aims at regrowing part of a morphologies that have been cut out
when the cell has been experimentally sliced.

``neuror cut-plane repair`` contains the collection of CLIs to perform this repair.

Additionally, there are CLIs for the cut plane detection and writing detected cut planes to
JSON files:

- If the cut plane is aligned with one of the X, Y or Z axes, the cut plane detection
  can be done automatically with the CLIs:

.. code-block:: shell

   neuror cut-plane file
   neuror cut-plane folder

- If the cut plane is not one the X, Y or Z axes, the detection has to be performed
  through the helper web application that can be launched with the following CLI:

.. code-block:: shell

   neuror cut-plane hint

Unravelling
~~~~~~~~~~~

Unravelling is the action of "stretching" the cell that has been shrunk because of the dehydratation caused by the slicing.

The unravelling CLI sub-group is:

.. code-block:: shell

   neuror unravel

.. admonition:: Info
   :class: info

   Unravelling is also part of the "full" process that performs unravelling and cut plane repair.
   The corresponding CLI is:

   .. code-block:: shell

      neuror cut-plane repair full

The unravelling algorithm can be described as follows:

* Segments are unravelled iteratively.
* Each segment direction is replaced by the averaged direction in a sliding window around this segment.
* The original segment length is preserved.
* The start position of the new segment is the end of the latest unravelled segment.

Installation
------------

NeuroR is distributed as a Python package available on PyPi:

.. code-block:: console

    $ pip install neuror[plotly]

Only Python 3.6 and above are supported.

Prior to running ``pip install``, we recommend updating ``pip`` in your virtual environment unless you have a compelling reason not to do it:

.. code:: console

    $ pip install --upgrade pip setuptools

Contributing
------------

If you want to improve the project or you see any issue, every contribution is welcome.
Please check the `contribution guidelines <https://github.com/BlueBrain/NeuroR/blob/master/CONTRIBUTING.md>`__ for more information.

License
-------

NeuroR is licensed under the terms of the GNU Lesser General Public License version 3.
Refer to COPYING.LESSER and COPYING for details.
