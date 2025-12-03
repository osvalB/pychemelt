Installation
============

Requirements
------------

PyPhotoMol requires Python 3.7 or later and the following packages:

* numpy
* pandas
* scipy
* xlrd
* openpyxl


Install from Source - Development
-----------------------------
Clone the repository and install in development mode (requires `uv`):

.. code-block:: bash

    git clone https://github.com/osvalB/pychemelt.git
    cd pychemelt
    uv sync --extra dev

Verify Installation
-------------------

By running the tests:

.. code-block:: bash

    uv run pytest

By creating the documentation:

.. code-block:: bash

    uv run build_docs.py