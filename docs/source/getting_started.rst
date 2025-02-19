Getting Started with MMCCI
============================

Installation
------------

Install MMCCI via pip:

.. code-block:: bash

    pip install mmcci

Usage
-----

Below is a simple example of loading a CCIData object:

.. code-block:: python

    from mmcci.CCIData_class import CCIData
    from mmcci.io import read_CCIData

    # Load data from a JSON file
    data = read_CCIData("path/to/your/data.json")
    print(data)

For further examples, see the "Usage Examples" section.
