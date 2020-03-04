Introduction
------------

morph-repair is a collection of tools to repair morphologies

It presently offers three types of repair:

- Sanitization:

  This is the process of curating a morphological file.
  It currently:

  - removes zero-length segments
  - raises if the morphology has no soma
  - raises if the morphology has negative diameters

  Note: more functionalities will be added in the future


- Cut plane repair:

  The cut plane repair aims at regrowing part of a morphologies that have been cut out
  when the cell has been experimentally sliced.

  'morph-repair cut-plane repair' contains the collection of CLIs to perform this repair.

  Additionally, there are CLIs for the cut plane detection and writing detected cut planes to
  JSON files:

  - If the cut plane is aligned with one of the X, Y or Z axes, the cut plane detection
    can be done automatically with the CLIs:

    morph-repair cut-plane file
    morph-repair cut-plane folder

  - If the cut plane is not one the X, Y or Z axes, the detection has to be performed
    through the helper web application that can be launched with the following CLI:

    morph-repair cut-plane hint


- Unravelling

  Unravelling is the action of "stretching" the cell that has been shrunk because of the dehydratation caused by the slicing.

  The unravelling CLI sub-group is:
  morph-repair unravel

  Additionally, unravelling is also part of the "full" process that performs unravelling and cut plane repair.
  The corresponding CLI is:
  morph-repair cut-plane repair full


  Description of the algorithm:

  Segment are unravelled iteratively. Each segment direction is replaced by the averaged direction in a sliding window
  around this segment. And the original segment length is preserved. The start position of the new segment is the end of the latest unravelled segment.


Installation
============

morph-repair is distributed as a Python package available on PyPi:

.. code-block:: console

    $ pip install morph-repair[plotly]

Only Python 3.6 and above are supported.

Prior to running *pip install*, we recommend updating *pip* in your virtual environment unless you have a compelling reason not to do it:

.. code:: console

    $ pip install --upgrade pip setuptools
