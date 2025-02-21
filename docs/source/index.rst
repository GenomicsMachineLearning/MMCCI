|PyPI| |Downloads| |Docs|

MMCCI
======

Multi-platform, Multi-sample Cell-Cell Interaction Integrative Analysis of Single Cell and Spatial Data
----------------------------------------------------------------------------------------------------------------

**MMCCI** is a Python package for analyzing cell–cell interaction data from different CCI tools. 
This documentation provides an overview, installation guidelines, API reference, and usage examples.

.. image:: https://raw.githubusercontent.com/GenomicsMachineLearning/MMCCI/refs/heads/main/docs/images/analyses_pipeline.png
    :alt: MMCCI title figure
    :width: 800px
    :align: center
    :target: https://doi.org/10.1101/2024.02.28.582639

|
MMCCI's key applications
--------------------------

MMCCI allows users to integrate multiple CCI results together, both:

   1. Samples from a single platform (eg. Visium)
   2. Samples from multiple platforms (eg. Visium, Xenium and CosMX)

MMCCI provides multiple useful analyses that can be run on the integrated networks or from a single sample:

   1. Network comparison between groups with permutation testing
   2. Clustering of LR pairs with similar networks
   3. Clustering of spots/cells with similar interaction scores
   4. Sender-receiver LR querying
   5. GSEA pathway analysis


.. toctree::
    :caption: General
    :maxdepth: 2
    :hidden:

    getting_started
    api

.. toctree::
    :caption: Examples
    :maxdepth: 2
    :hidden:

    notebooks/brain_aging_example
    notebooks/melanoma_example
    notebooks/loading_CCI_results

.. |PyPI| image:: https://img.shields.io/pypi/v/mmcci.svg
    :target: https://pypi.org/project/mmcci/
    :alt: PyPI

.. |Docs| image:: https://img.shields.io/readthedocs/mmcci
    :target: https://mmcci.readthedocs.io/en/stable/
    :alt: Documentation

.. |Downloads| image:: https://pepy.tech/badge/mmcci
    :target: https://pepy.tech/project/mmcci
    :alt: Downloads


.. _github: https://github.com/GenomicsMachineLearning/MMCCI
